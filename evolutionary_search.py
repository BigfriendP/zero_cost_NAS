import random
import numpy as np
import pandas as pd
import torch
import os
import copy
from tqdm import trange
from statistics import mean
#import time

"""## Parameters"""

args_GPU = '0'
args_seed = 0

args_dataset = 'ImageNet16-120'
args_data_loc = './data/' + args_dataset
args_batch_size = 128
args_save_loc = './results'
args_nruns = 30

args_score = 'snip'

os.environ['CUDA_VISIBLE_DEVICES'] = args_GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""## Reproducibility"""

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args_seed)
np.random.seed(args_seed)
torch.manual_seed(args_seed)

"""## NATS-Bench initialization"""

#installing libraries and download of the benchmark file
os.system("pip install nats_bench")
if not(os.path.exists("NATS-tss-v1_0-3ffb9-simple")):
  os.system("wget 'https://www.dropbox.com/s/pasubh1oghex3g9/?dl=1' -O 'NATS-tss-v1_0-3ffb9-simple.tar'")
  os.system("tar xf 'NATS-tss-v1_0-3ffb9-simple.tar'")

#importing nats_bench library
from nats_bench import create

#API initialization
searchspace = create('NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=False)

"""## Importing scores of the network"""

results = pd.read_csv(f'{args_save_loc}/{args_dataset}/{args_score}/{args_dataset}-{args_score}.csv')
proxy = results.iloc[:,1]
t = results['time']

"""# Evolutionary Search Definition"""

#dictionary operation_name-operation_id

_opname_to_index = {
    'none': 0,
    'skip_connect': 1,
    'nor_conv_1x1': 2,
    'nor_conv_3x3': 3,
    'avg_pool_3x3': 4
}

#function to get the spec of the architecture 

def get_spec_from_arch_str(arch_str):

  nodes = arch_str.split('+')
  nodes = [node[1:-1].split('|') for node in nodes]
  nodes = [[op_and_input.split('~')[0]  for op_and_input in node] for node in nodes]

  spec = [_opname_to_index[op] for node in nodes for op in node]
  return spec

#dictionary mapping architecture_uid-spec

idx_to_spec = {}
for i, arch_str in enumerate(searchspace):
    idx_to_spec[i] = get_spec_from_arch_str(arch_str)
spec_to_idx = {}
for idx,spec in idx_to_spec.items():
    spec_to_idx[str(spec)] = idx

def random_combination(iterable, sample_size):
  pool = tuple(iterable)
  n = len(pool)
  indices = sorted(random.sample(range(n), sample_size))
  return tuple(pool[i] for i in indices)

"""## Metric based Mutation"""

def mutate_spec_zero_cost(old_spec):
    possible_specs = []
    metric_time = 0.0
    #mut_time = 0.0
    #start_time = time.time()
    for idx_to_change in range(len(old_spec)): 
        entry_to_change = old_spec[idx_to_change]
        possible_entries = [x for x in range(5) if x != entry_to_change]
        for new_entry in possible_entries:
            new_spec = copy.copy(old_spec)
            new_spec[idx_to_change] = new_entry
            metric_time += t[spec_to_idx[str(new_spec)]]
            possible_specs.append((proxy[spec_to_idx[str(new_spec)]], new_spec))
    best_new_spec = sorted(possible_specs, key=lambda i:i[0])[-1][1]
    #mut_time = (time.time()-start_time)+metric_time
    if random.random() > 0.75:
        best_new_spec = random.choice(possible_specs)[1]
    return best_new_spec, metric_time,  #mut_time

def run_evolution_search(max_visited_models=1000, 
                         pool_size=64, 
                         tournament_size=10, 
                         zero_cost_warmup=0):
  
  best_proxy = 0.0
  best_proxy_id = 0
  pool = []   # (proxy, spec) tuples
  num_visited_models = 0
  search_time = 0.0
  
  # fill the initial pool
  pool_init_time = 0.0
  if zero_cost_warmup > 0:
    zero_cost_pool = []
    for _ in range(zero_cost_warmup):
      spec = random.choice(list(idx_to_spec.values()))
      spec_idx = spec_to_idx[str(spec)]
      pool_init_time += t[spec_idx]
      zero_cost_pool.append((proxy[spec_idx], spec))
      zero_cost_pool = sorted(zero_cost_pool, key=lambda i:i[0], reverse=True)

  for i in range(pool_size):
    if zero_cost_warmup > 0:
      spec = zero_cost_pool[i][1]
      
    else:
      spec = random.choice(list(idx_to_spec.values()))
      pool_init_time += t[spec_to_idx[str(spec)]]
    
    spec_idx = spec_to_idx[str(spec)]
    num_visited_models += 1
    pool.append((proxy[spec_idx], spec))

    if proxy[spec_idx] > best_proxy:
      best_proxy = proxy[spec_idx]
      best_proxy_id = spec_idx

  search_time += pool_init_time
  #print(f'inizial_pool: \nscore: {best_proxy}, id: {best_proxy_id}\n')

  # After the pool is seeded, proceed with evolving the population.
  while(True):
    sample = random_combination(pool, tournament_size)   
    best_spec = sorted(sample, key=lambda i:i[0])[-1][1]
    
    #mut_time
    new_spec, mut_time = mutate_spec_zero_cost(best_spec)
    num_visited_models += 1 

    search_time += mut_time

    new_spec_idx = spec_to_idx[str(new_spec)]

    # add the new generation and kill the oldest individual in the population.
    pool.append((proxy[new_spec_idx], new_spec))
    pool.pop(0)

    if proxy[new_spec_idx] > best_proxy:
      best_proxy = proxy[new_spec_idx]
      best_proxy_id = new_spec_idx
      #print(f'\n{best_proxy_id}, {best_proxy}')
    
    #print(f'visited_mod: {i} \nid: {best_proxy_id} ; best proxy: {best_proxy}\n')

    if num_visited_models >= max_visited_models:
      break

  #accuracy = searchspace.get_more_info(int(best_proxy_id), args_dataset, hp = '200')['test-accuracy']
  
  return best_proxy_id, best_proxy, search_time


runs = trange(args_nruns, desc='acc: ')
top_acc = []
search_times = []

for N in runs:

  id_best_net, best_score, search_time = run_evolution_search(max_visited_models=80, 
                                                   pool_size=64,
                                                   tournament_size=10,
                                                   zero_cost_warmup=0)
  
  acc_best_net = searchspace.get_more_info(int(id_best_net), args_dataset, hp = '200')['test-accuracy']
  
  #print(f'\niteration: {N} \nid: {id_best_net} ; best_score: {best_score} ; acc_best_net: {acc_best_net}\n')

  top_acc.append(acc_best_net)
  search_times.append(search_time)

mean_acc = np.mean(top_acc)
std_acc = np.std(top_acc)

mean_time = np.mean(search_times)
std_time = np.std(search_times)

print(f'\nmetric: {args_score} ; {args_dataset}\n')
print("accuracy: {} + {}\n".format(mean_acc,std_acc))
print("search time: {} + {}\n".format(mean_time,std_time))

