from foresight.pruners import predictive
from tqdm import tqdm

import json
import logging
import time
import math
import pickle

import os
import numpy as np
import torch

import ConfigSpace as CS
from ConfigSpace import Configuration
from collections import OrderedDict


from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from torchvision import datasets
from torchvision import transforms
from searchspace import get_configspace
from ConfigSpace import Configuration

from cnn import torchModel


proxies_used = ["synflow", "snip"]


def proxy_score_from_cfg(cfg: Configuration, seed: int, device="cpu"):
    #TODO : Pass data_dir and device as kwargs
 
    data_dir = 'FashionMNIST'
    batch_size = 100

    # Device configuration
    np.random.seed(seed)
    torch.manual_seed(seed)
    model_device = torch.device(device)

    img_width = 28
    img_height = 28
    input_shape = (1, img_width, img_height)

    pre_processing = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_val = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=pre_processing
    )


    train_loader = DataLoader(dataset=train_val,
                                  batch_size=batch_size,
                                  shuffle=True)

    num_classes = len(train_val.classes)

    model = torchModel(cfg,input_shape=input_shape,num_classes=num_classes).to(model_device)

    #summary(model, input_shape, device=device)
    train_criterion = torch.nn.CrossEntropyLoss()

    # To change 
    measures = predictive.find_measures ( net_orig = model , dataloader = train_loader , dataload_info =('random' , 1 , num_classes ) , device=device  , loss_fn = train_criterion ,  measure_names=proxies_used)

    return measures


def main(size , time_limit, cs, seed, device):
    logger = logging.getLogger("Zero Cost proxies")
    logging.basicConfig(level=logging.INFO)


    cs = cs
    #TODO : make the number of sampled configuration dynamic
    
    config = cs.sample_configuration(size)
    if not isinstance(config, list):
        config = [config]
    measures_dict = {}
    start_time = time.time()

    for i in tqdm(range(len(config))):
        measures = proxy_score_from_cfg(config[i], seed, device)
        measures["synflow"] =  math.log(measures["synflow"]) if measures["synflow"] > 0 else -math.log(-measures["synflow"])
        measures_dict[config[i]]= measures 
        time_spent = time.time() - start_time
        if (time_spent > time_limit):
            print("Reached time limit")
            break
     
    architectures = measures_dict.keys()
    rank_sum = np.zeros(len(architectures),)
    for proxy in proxies_used:
        proxy_scores = [measures_dict[architecture][proxy] for architecture in architectures]
        proxy_ranking = np.argsort(np.argsort(proxy_scores))
        rank_sum += proxy_ranking
    
    architecture_to_rank_sum = dict(zip(architectures, rank_sum))
    
    measures_dict_sorted = sorted (measures_dict.items() , key = lambda x:-architecture_to_rank_sum[x[0]] )
    list_config_sorted = [e[0] for e in measures_dict_sorted]

    
    return list_config_sorted , time_spent

if __name__ == '__main__':
    seed=42
    cs, _ = get_configspace(seed)
    a,b=main(size=100, time_limit=50, cs=cs, seed=seed, device="cpu")
    print(a)

   

   
