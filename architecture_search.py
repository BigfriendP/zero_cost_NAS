import random
import numpy as np
import pandas as pd
import torch
import os
from tqdm import trange
from statistics import mean
import time

"""## Parameters"""

args_GPU = '0'
args_seed = 0

args_dataset = 'ImageNet16-120'
args_data_loc = './data/' + args_dataset
args_batch_size = 128
args_save_loc = './results'
args_nruns = 30
args_nsamples = 100
args_criterion = 'valid'

args_score = 'synflow'
args_spcorr = True

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

if args_criterion == 'score':
  results = pd.read_csv(f'{args_save_loc}/{args_dataset}/{args_score}/{args_dataset}-{args_score}.csv')
  results

"""## Execute the search """

runs = trange(args_nruns, desc='acc: ')
top_acc = []
search_times = []

for N in runs:
  indices = np.random.randint(0,len(searchspace),args_nsamples)

  if args_criterion == 'score':

    net_sample = results.loc[indices]

    computing_time = net_sample['time'].sum()
    start_time = time.time()
    uid_max = net_sample.iloc[:,1].idxmax()
    t = time.time() - start_time
    tot_time = computing_time + t
    accuracy = searchspace.get_more_info(int(uid_max), args_dataset, hp = '200')['test-accuracy']
    
  elif args_criterion == 'valid':

    best_valid = 0
    best_net = 0
    tot_time = 0
    for uid in indices:
      # Simulate the training of the uid-th candidate:
      validation_accuracy, latency, time_cost, current_total_time_cost = searchspace.simulate_train_eval(int(uid), dataset=args_dataset, hp='12')
      if validation_accuracy>best_valid:
        best_net = uid
        best_valid = validation_accuracy
      tot_time += time_cost
    accuracy = searchspace.get_more_info(int(best_net), args_dataset, hp = '200')['test-accuracy']

  top_acc.append(accuracy)
  search_times.append(tot_time)

mean_acc = np.mean(top_acc)
std_acc = np.std(top_acc)

mean_time = np.mean(search_times)
std_time = np.std(search_times)

print(f'{args_criterion} method ; {args_dataset}')

if args_criterion == 'score':
  print(f'metric: {args_score}\n')

print("accuracy: {} + {}\n".format(mean_acc,std_acc))
print("search time: {} + {}\n".format(mean_time,std_time))