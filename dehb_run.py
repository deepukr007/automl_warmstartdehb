"""
===========================
Optimization using DEHB
===========================
"""
import argparse
from functools import partial
import json
import logging
import time 
import os
import numpy as np
import torch
import pickle
import sys


from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from torchvision import datasets
from torchvision import transforms
import aug_lib


from cnn import torchModel
from searchspace import get_configspace
import zerocost

from dehb import DEHB
from ConfigSpace import Configuration

seed = 42
device = "cuda"

def get_optimizer_and_crit(cfg):
    if cfg['optimizer'] == 'AdamW':
        model_optimizer = torch.optim.AdamW
    else:
        model_optimizer = torch.optim.Adam

    if cfg['train_criterion'] == 'mse':
        train_criterion = torch.nn.MSELoss
    else:
        train_criterion = torch.nn.CrossEntropyLoss
    return model_optimizer, train_criterion


def objective_function(cfg: Configuration, budget: float):

    print(cfg)
  
    lr = cfg['learning_rate_init'] if cfg['learning_rate_init'] else 0.001
    batch_size = 64
    #TODO : Pass data_dir and device as kwargs
    data_dir = 'FashionMNIST'

    # Device configuration
    np.random.seed(seed)
    torch.manual_seed(seed)
    model_device = torch.device(device)

    img_width = 28
    img_height = 28
    input_shape = (1, img_width, img_height)
    
    train_val = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
    )
    
    FASHION_MINST_MEAN = (train_val.data/255).mean()
    FASHION_MINST_STD = (train_val.data/255).std()

    pre_processing_train_val = transforms.Compose(
        ([
            transforms.TrivialAugmentWide(),
        ] if cfg["augment"] else []) + [
            transforms.ToTensor(),
        ] + ([
            transforms.Normalize(FASHION_MINST_MEAN, FASHION_MINST_STD),
        ] if cfg["standardize"] else [])
    )
    
    pre_processing_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ] + ([
            transforms.Normalize(FASHION_MINST_MEAN, FASHION_MINST_STD),
        ] if cfg["standardize"] else [])
    )

    train_val = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=pre_processing_train_val
    )

    test_set = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=pre_processing_test
    )
    
    train_set_size = int(len(train_val) * 0.8)
    valid_set_size = len(train_val) - train_set_size
    train_set, valid_set = torch.utils.data.random_split(train_val, [train_set_size, valid_set_size])

    num_epochs = int(np.ceil(budget))

    start = time.time()

    train_loader = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(dataset=valid_set,batch_size=batch_size,shuffle=False)
    model = torchModel(cfg,input_shape=input_shape,num_classes=len(train_val.classes)).to(model_device)

    #summary(model, input_shape, device=device)

    model_optimizer, train_criterion = get_optimizer_and_crit(cfg)
    optimizer = model_optimizer(model.parameters(), lr=lr)
    train_criterion = train_criterion().to(device)

    for epoch in range(num_epochs):
        logging.info('#' * 50)
        logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        train_score, train_loss = model.train_fn(optimizer, train_criterion, train_loader, model_device)
        logging.info('Train accuracy %f', train_score)

    val_score = model.eval_fn(val_loader, device)

    cost = time.time() -start 

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    test_loss =  1- (model.eval_fn(test_loader, device))
    val_loss =  1 - np.mean(val_score)  

    res = {
        "fitness": val_loss,
        "cost": cost,
        "info": {"test_loss": test_loss, "budget": budget, "config": cfg.get_dictionary() }
    }

    print (res)
    return res



if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(description='JAHS')
    parser.add_argument('--data_dir', type=str, default='./FashionMNIST')
    parser.add_argument('--working_dir', default='./tmp', type=str,
                        help="directory where intermediate results are stored")
    
    parser.add_argument('--runtime', default=7200, type=int, help='Running time allocated to run the algorithm')
    parser.add_argument('--max_epochs', type=int, default=100, help='maximal number of epochs to train the network')
    parser.add_argument('--min_epochs', type=int, default=20, help='maximal number of epochs to train the network')
    parser.add_argument('--seed', type=int, default=42 ,  help='random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device to run the models')
    parser.add_argument('--eta', type=int, default='3', help='eta')
    parser.add_argument('--expname', type=str, default='test', help='Experiment_name')
    parser.add_argument('--n_proxy', type=int, default='100', help='Number of proxies to evaluate')


    args = parser.parse_args()

    logger = logging.getLogger("DEHB")
    logging.basicConfig(level=logging.INFO)

    data_dir = args.data_dir
    runtime = args.runtime
    device = args.device
    max_epochs = args.max_epochs
    min_epochs = args.min_epochs
    eta = args.eta
    working_dir = args.working_dir
    expname = args.expname
    n_proxy = args.n_proxy
    seed = args.seed

    np.random.seed(seed)
    torch.manual_seed(seed)

    working_dir = os.path.join(working_dir,expname)

    if(n_proxy>0):
        cs , dimensions = get_configspace(seed=seed)
        population, time_spent_for_proxies = zerocost.main(time_limit=3600 , size=n_proxy, cs=cs, seed=seed, device=device)
        print("Length of intial population: {} " .format(len(population)))
        runtime = runtime - time_spent_for_proxies

    init_time = time.time()

    dehb = DEHB(f=objective_function, cs=cs, dimensions=dimensions, min_budget=min_epochs,
                max_budget=max_epochs, eta= eta, output_path=working_dir,n_workers=1 , custom_initial_population = population
                )

    dehb.logger.info("Arguments: {} " .format(sys.argv[1:]))

    if(n_proxy>0):    
        for i in range(len(population)):
            population[i] = population[i].get_dictionary()
        
        with open(os.path.join(working_dir, "proxies.pkl"), "wb") as f:
            pickle.dump(population, f)        

        dehb.logger.info("Length of intial population: {} " .format(len(population)))
        dehb.logger.info("Time spent for proxies: {}  ".format(time_spent_for_proxies))

    traj, runtime, history = dehb.run(total_cost=runtime,verbose=False, save_intermediate=True)
    print(dehb.get_incumbents())

    total_time = time.time() - init_time 

    dehb.logger.info("Total time taken: {} ".format(total_time))
    
    # Saving optimisation trace history
    name = time.strftime("%x %X %Z", time.localtime(dehb.start))
    name = name.replace("/", '-').replace(":", '-').replace(" ", '_')
    dehb.logger.info("Saving optimisation trace history...")
    
    with open(os.path.join(working_dir, "history_{}.pkl".format(name)), "wb") as f:
        pickle.dump(history, f)

    with open(os.path.join(working_dir, "traj_{}.pkl".format(name)), "wb") as f:
        pickle.dump(traj, f)
    
    incumbent = dehb.vector_to_configspace(dehb.inc_config)
    print(incumbent)

    



