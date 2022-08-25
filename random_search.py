import random
import numpy as np
import pandas as pd
import torch
import os
from tqdm import trange
import time

"""## Parameters"""

args_GPU = '0'
args_seed = 0

args_dataset = 'cifar10'
args_save_loc = './results'


args_score = 'hook_logdet'


os.environ['CUDA_VISIBLE_DEVICES'] = args_GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""## Random Search Parameters"""
args_nruns = 30
args_nsamples = 1000

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


"""## Execute the search """

runs = trange(args_nruns, desc='acc: ')
top_acc = []
search_times = []
best_tests_progress = []

for N in runs:

  # choose a random subsample of the searchspace to be examined
  indices = np.random.randint(0,len(searchspace),args_nsamples)
  net_sample = results.loc[indices]
  net_sample.reset_index(inplace=True, drop=True)
  start_time = time.time()
  best_proxies, best_tests = [0.0],  [0.0]
  best_proxy_id = 0
  search_time = 0.0
  # iterate over the networks to choose the one with the highest proxy
  for idx in range(args_nsamples):
    uid = net_sample.iloc[idx,0]
    proxy = net_sample.iloc[idx,1]
    t = net_sample.iloc[idx,2]
    test_accuracy = searchspace.get_more_info(int(uid), args_dataset, hp = '200')['test-accuracy']
    if proxy > best_proxies[-1]:
      best_proxies.append(proxy)
      best_proxy_id = uid
      best_tests.append(test_accuracy)
    else:
      best_proxies.append(best_proxies[-1])
      best_tests.append(best_tests[-1])
    search_time += t
  end_time = time.time()
  search_time += (start_time-end_time)
  best_proxies.pop(0)
  best_tests.pop(0)
  top_acc.append(best_tests[-1])
  search_times.append(search_time)
  best_tests_progress.append(best_tests)

# save the file containing the progress of the test accuracy of all the experiments
np.save(f'{args_save_loc}/{args_dataset}/{args_score}/RandSearch_{args_dataset}-{args_score}', best_tests_progress)

# calculate the mean and std of the test accuracy and search time over all the experiments
mean_acc = np.mean(top_acc)
std_acc = np.std(top_acc)

mean_time = np.mean(search_times)
std_time = np.std(search_times)

print(f'\nmetric: {args_score} ; {args_dataset}\n')

print("accuracy: {} + {}\n".format(mean_acc,std_acc))
print("search time: {} + {}\n".format(mean_time,std_time))