import random
import numpy as np
import pandas as pd
import torch
import os
import copy
from tqdm import trange
import time
import argparse

"""## Parameters"""

args_GPU = '0'
args_seed = 0

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', required=True, type=str, help='dataset on which to perform the search')
parser.add_argument('--score', '-s', required=True, type=str, help='metrics used as proxy')
parser.add_argument('--save_loc', default='./results', type=str, help='path to the directory where the csv are saved and where the progress files of the search will be saved')
args = parser.parse_args()

args_dataset = args.dataset
args_score = args.score
args_save_loc = args.save_loc

#check if the inputs are valid
valid_datasets = ['cifar10', 'cifar100', 'ImageNet16-120']
valid_metrics = ['val12', 'hook_logdet', 'synflow', 'snip', 'combined']

if args_dataset not in valid_datasets:
  print('ERROR -> INVALID DATASET: dataset must be cifar10 or cifar100 or ImageNet16-120')
  quit()
if args_score not in valid_metrics:
  print('ERROR -> INVALID metric: metric must be hook_logdet or synflow or snip')
  quit()

os.environ['CUDA_VISIBLE_DEVICES'] = args_GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""## Aging Evolution Parameters"""
args_nruns = 30
max_n_models = 500 
pool_size = 64
subpop_size = 10
warmup = 0

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

"""## Importing scores and accuracies of the network"""

results = pd.read_csv(f'{args_save_loc}/{args_dataset}/{args_score}/{args_dataset}-{args_score}.csv')
proxy = results.iloc[:,1]
t = results['time']
accuracies = pd.read_csv(f'accuracy_full_trained_models/accuracy_{args_dataset}.csv')

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
    start_time = time.time()
    for idx_to_change in range(len(old_spec)): 
        entry_to_change = old_spec[idx_to_change]
        possible_entries = [x for x in range(5) if x != entry_to_change]
        for new_entry in possible_entries:
            new_spec = copy.copy(old_spec)
            new_spec[idx_to_change] = new_entry
            metric_time += t[spec_to_idx[str(new_spec)]]
            possible_specs.append((proxy[spec_to_idx[str(new_spec)]], new_spec))
    best_new_spec = sorted(possible_specs, key=lambda i:i[0])[-1][1]
    
    if random.random() > 0.75:
        best_new_spec = random.choice(possible_specs)[1]
        metric_time = 0.0
    end_time = time.time()
    mutation_time = metric_time+ (end_time-start_time)
    return best_new_spec, mutation_time  

def run_evolution_search(max_visited_models=1000, 
                         pool_size=64, 
                         tournament_size=10, 
                         zero_cost_warmup=0):
  
  best_proxies, best_tests = [0.0],  [0.0]
  best_proxy_id = 0
  pool = []   # (proxy, spec) tuples
  num_visited_models = 0
  search_time = 0.0
  
  start_time = time.time()
  # fill the initial pool
  pool_init_time = 0.0
  if zero_cost_warmup > pool_size:
    zero_cost_pool = []
    for _ in range(zero_cost_warmup):
      spec = random.choice(list(idx_to_spec.values()))
      spec_idx = spec_to_idx[str(spec)]
      pool_init_time += t[spec_idx]
      zero_cost_pool.append((proxy[spec_idx], spec))
      zero_cost_pool = sorted(zero_cost_pool, key=lambda i:i[0], reverse=True)

  for i in range(pool_size):
    if zero_cost_warmup > pool_size:
      spec = zero_cost_pool[i][1]
      
    else:
      spec = random.choice(list(idx_to_spec.values()))
      pool_init_time += t[spec_to_idx[str(spec)]]
    
    spec_idx = spec_to_idx[str(spec)]
    num_visited_models += 1
    pool.append((proxy[spec_idx], spec))

    test_accuracy = accuracies.loc[spec_idx,'accuracy']
    
    if proxy[spec_idx] > best_proxies[-1]:
      best_proxies.append(proxy[spec_idx])
      best_proxy_id = spec_idx
      best_tests.append(test_accuracy)
    else:
      best_proxies.append(best_proxies[-1])
      best_tests.append(best_tests[-1])

  search_time += pool_init_time
  
  # After the pool is seeded, proceed with evolving the population.
  while(True):

    # take a subsample of the population and choose the highest score network as parent of the new generation
    sample = random_combination(pool, tournament_size)   
    best_spec = sorted(sample, key=lambda i:i[0])[-1][1]
    
    # metric based mutation
    new_spec, mut_time = mutate_spec_zero_cost(best_spec)
    num_visited_models += 1 

    search_time += mut_time

    new_spec_idx = spec_to_idx[str(new_spec)]

    # add the new generation and kill the oldest individual in the population.
    pool.append((proxy[new_spec_idx], new_spec))
    pool.pop(0)

    test_accuracy = accuracies.loc[new_spec_idx,'accuracy']

    if proxy[new_spec_idx] > best_proxies[-1]:
      best_proxies.append(proxy[new_spec_idx])
      best_proxy_id = new_spec_idx
      best_tests.append(test_accuracy)
    else:
      best_proxies.append(best_proxies[-1])
      best_tests.append(best_tests[-1])

    if num_visited_models >= max_visited_models:
      break
  
  end_time = time.time()   
  search_time += (end_time-start_time)
  best_proxies.pop(0)    
  best_tests.pop(0) 
  best_accuracy = accuracies.loc[best_proxy_id,'accuracy']

  
  return best_proxy_id, best_accuracy, best_proxies, best_tests, search_time


runs = trange(args_nruns, desc='acc: ')
top_acc = []
search_times = []
best_tests_progress = []

for N in runs:

  id_best_net, acc_best_net, best_proxies, best_tests, search_time = run_evolution_search(max_visited_models=max_n_models, 
                                                   pool_size=pool_size,
                                                   tournament_size=subpop_size,
                                                   zero_cost_warmup=warmup)
  
  
  top_acc.append(acc_best_net)
  search_times.append(search_time)
  best_tests_progress.append(best_tests)

if warmup == 0:
  exp_name = 'AEVsearch'
else:
  exp_name = 'AEVsearchWU'

# save the file containing the progress of the test accuracy of all the experiments
np.save(f'{args_save_loc}/{args_dataset}/{args_score}/{exp_name}_{args_dataset}-{args_score}', best_tests_progress)

print(f'aging evolutionary search test accuracy progress file saved at {args_save_loc}/{args_dataset}/{args_score}/')

# calculate the mean and std of the test accuracy and search time over all the experiments
mean_acc = np.mean(top_acc)
std_acc = np.std(top_acc)

mean_time = np.mean(search_times)
std_time = np.std(search_times)

print(f'\nmetric: {args_score} ; {args_dataset}\n')
print("accuracy: {} + {}\n".format(mean_acc,std_acc))
print("search time: {} + {}\n".format(mean_time,std_time))


