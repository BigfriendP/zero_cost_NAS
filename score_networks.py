# install xautodl package if it is not installed
!pip install xautodl

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import xautodl
from xautodl.models import get_cell_based_tiny_net
import time
import gc
from dataset import get_data
from metrics import compute_score_metric, compute_snip_score, compute_synflow_score
import argparse

"""## Parameters"""

args_GPU = '0'
args_seed = 0
args_batch_size = 128

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', required=True, type=str, help='dataset on which to calculate the metric')
parser.add_argument('--score', '-s', required=True, type=str, help='metrics used as proxy of the test accuracy')
parser.add_argument('--data_loc', default='./data/', type=str, help='path to the data directory')
parser.add_argument('--save_loc', default='./results', type=str, help='path to the directory where to save the results')
args = parser.parse_args()

args_dataset = args.dataset
args_score = args.score
args_data_loc = args.data_loc + args_dataset
args_save_loc = args.save_loc

if not(os.path.isdir('data')):
  os.system("mkdir data")

if args_dataset.startswith('ImageNet16') and not(os.path.exists("data/ImageNet16-120")):
  os.system("wget 'https://www.dropbox.com/s/o2fg17ipz57nru1/?dl=1' -O ImageNet16.tar.gz")
  os.system("tar xvf 'ImageNet16.tar.gz'")
  os.system("mv ImageNet16 data/ImageNet16-120")

#set GPU

os.environ['CUDA_VISIBLE_DEVICES'] = args_GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""## Reproducibility"""

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args_seed)
np.random.seed(args_seed)
torch.manual_seed(args_seed)

"""## Dataset initialization"""

train_loader = get_data(args_dataset, args_data_loc, args_batch_size)

os.makedirs(args_save_loc, exist_ok=True)

"""## NATS-Bench initialization"""

#installing libraries and download the benchmark file
os.system("pip install nats_bench")
if not(os.path.exists("NATS-tss-v1_0-3ffb9-simple")):
  os.system("wget 'https://www.dropbox.com/s/pasubh1oghex3g9/?dl=1' -O 'NATS-tss-v1_0-3ffb9-simple.tar'")
  os.system("tar xf 'NATS-tss-v1_0-3ffb9-simple.tar'")

#importing nats_bench library
from nats_bench import create

#API initialization
searchspace = create('NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=False)

"""## Scoring Network cycle"""

results = []

for uid in range(len(searchspace)):
  config = searchspace.get_net_config(uid, args_dataset)
  network = get_cell_based_tiny_net(config)
  
  start_time = time.time()
  
  network = network.to(device)
  random.seed(args_seed)
  np.random.seed(args_seed)
  torch.manual_seed(args_seed)
    

  data_iterator = iter(train_loader)
  x, target = next(data_iterator)

  x, target = x.to(device), target.to(device)


  if args_score == 'hook_logdet':
    score = compute_score_metric(network, x, args_batch_size)
  elif args_score == 'snip':
    score = compute_snip_score(network, x, target, F.cross_entropy)
  elif args_score == 'synflow':
    score = compute_synflow_score(network, x)

  execution_time = time.time()-start_time

  del network
  torch.cuda.empty_cache()
  gc.collect()

  print(uid, score, execution_time)
  results.append([uid, score, execution_time])

import os
if not(os.path.isdir(args_save_loc+'/'+args_dataset)):
  os.system(f"mkdir {args_save_loc}/{args_dataset}")

import csv

fields = ['uid', 'score', 'time']
with open(f'{args_save_loc}/{args_dataset}/{args_dataset}-{args_score}.csv', 'w') as f:

  write = csv.writer(f)
      
  write.writerow(fields)
  write.writerows(results)
