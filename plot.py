import matplotlib.pyplot as plt 
import pickle
import os
import numpy as np
import ConfigSpace.util
from fanova import fANOVA
import numpy as np
import matplotlib.ticker

from searchspace import get_configspace

tmp ="tmp"
p_time = 2061.668258666992 

history_file_path_vanilla = os.path.join(tmp ,"ex_16_100_25_2_vanilla" , "history_02-26-23_18-21-31_CET.pkl")

traj_values_vanilla = []
time_vanilla = []

with open(history_file_path_vanilla , 'rb') as file:
    loaded = pickle.load(file)
    x = [x[:][2] for x in loaded]
    x = np.cumsum(x)
    time_vanilla =x

traj_path_vanilla = os.path.join(tmp , "ex_16_100_25_2_vanilla" , "traj_02-26-23_18-21-31_CET.pkl")

with open(traj_path_vanilla , 'rb') as file:
    loaded = pickle.load(file)
    traj_values_vanilla = loaded


history_file_path_pop = os.path.join(tmp , "ex_15_100_25_2_init_pop" , "history_02-26-23_11-06-05_CET.pkl")

traj_values_pop = []
time_pop = []
val_acc = []
configs = []

def config_to_vec(config, param_names):
    result_vec = []
    for key in param_names:
        result_vec += [float(config[key])]
    return result_vec
    

with open(history_file_path_pop , 'rb') as file:
    loaded = pickle.load(file)
    x = [x[:][2] for x in loaded]
    x = np.cumsum(x)
    time_pop = x

    #for fanova
    val_acc = [val_acc[:][1] for val_acc in loaded]
    configs = [config[:][4]['config'] for config in loaded]
    cs, _ = get_configspace()
    param_names = [x.name for x in cs.get_hyperparameters()]
    configs = [config_to_vec(x, param_names) for x in configs]
    print(param_names, configs)
    f = fANOVA(X = np.array(configs), Y = np.array(val_acc), config_space=cs)
    importance_list = []
    name_to_total_importance=dict()
    name_to_total_std=dict()
    name_to_individual_importance=dict()
    name_to_individual_std=dict()
    for i in range(len(param_names)):
        name=param_names[i]
        res = f.quantify_importance((i, ))
        name_to_total_importance[name] = res[(i,)]["total importance"]
        name_to_total_std[name] = res[(i,)]["total std"]
        name_to_individual_importance[name] = res[(i,)]["individual importance"]
        name_to_individual_std[name] = res[(i,)]["individual std"]
    
    importance_list = sorted(param_names, key=lambda x:-name_to_total_importance[x])
    mean = []
    std = []
    for param in importance_list:
        mean+=[name_to_total_importance[param]]
        std+=[name_to_total_std[param]]
        print(param, name_to_total_importance[param], name_to_total_std[param], name_to_individual_importance[param], name_to_individual_std[param])
    num_params = 14
    mean=np.array(mean)[:num_params]
    std=np.array(std)[:num_params]
    importance_list = importance_list[:num_params]
    
    
    
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(importance_list)), mean, yerr=mean*std, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Importance')
    ax.set_xticks(np.arange(len(importance_list)))
    ax.set_xticklabels(importance_list)
    plt.xticks(rotation=80)
    ax.set_title('Hyperparameter')
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig('hyperparamimportance.png')

    

time_pop = [t + p_time for t in time_pop]


traj_path_pop = os.path.join(tmp , "ex_15_100_25_2_init_pop" , "traj_02-26-23_11-06-05_CET.pkl")

with open(traj_path_pop , 'rb') as file:
    loaded = pickle.load(file)
    traj_values_pop = loaded



fig, ax = plt.subplots(figsize=(8, 6))  # set the figure size
ax.plot(time_vanilla, traj_values_vanilla, 'o-',label='DEHB')
ax.plot(time_pop, traj_values_pop,'o-', label='DEHB+ZC')

ax.set_xlabel('Time')
ax.set_xscale('log')
ax.set_xticks([100, 1000, 10000])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_ylabel('Validation Error')
ax.set_title('DEHB vs. DEHB+ZC')

ax.legend()

ax.grid(True) 
ax.spines['right'].set_visible(False) 
ax.spines['top'].set_visible(False)
ax.tick_params(direction='in')  


plt.savefig('dehb_vs_dehb_zc.png', dpi=300) 


