
"""# Datasets"""
import os
os.system("pip install xautodl")
if not(os.path.exists("data/ImageNet16-120")):
  os.system("wget 'https://www.dropbox.com/s/o2fg17ipz57nru1/?dl=1' -O ImageNet16.tar.gz")
  os.system("tar xvf 'ImageNet16.tar.gz'")
  if not(os.path.isdir('data')):
    os.system("mkdir data")
  os.system("mv ImageNet16 data/ImageNet16-120")

from dataset import get_data

"""# Score Networks

## Importing Libraries
"""

import psutil
import random
import numpy as np
import torch
import os
from scipy import stats
import xautodl
from xautodl.models import get_cell_based_tiny_net
import time
import gc
import csv

"""## Parameters"""

args_GPU = '0'
args_seed = 0

args_dataset = 'cifar10'
args_data_loc = './data/cifar10'
args_batch_size = 128
args_save_loc = './results/'

args_score = 'hook_logdet'

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

#installing libraries and download of the benchmark file
os.system("pip install nats_bench")
if not(os.path.exists("NATS-tss-v1_0-3ffb9-simple")):
  os.system("wget 'https://www.dropbox.com/s/pasubh1oghex3g9/?dl=1' -O 'NATS-tss-v1_0-3ffb9-simple.tar'")
  os.system("tar xf 'NATS-tss-v1_0-3ffb9-simple.tar'")


#importing nats_bench library
from nats_bench import create

#API initialization
searchspace = create('NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=False)

"""## Hook functions definition"""

def counting_forward_hook(module, inp, out):
  try:
    #if not module.visited_backwards:
      #return
    if isinstance(inp, tuple):
      inp = inp[0]
    inp = inp.view(inp.size(0), -1)
    x = (inp > 0).float()
    K = x @ x.t()
    K2 = (1.-x) @ (1.-x.t())
    network.K = network.K + K.cpu().numpy() + K2.cpu().numpy()
  except:
    pass

"""## Scores calculation"""


results = []

for (uid, _) in enumerate(searchspace):
  
  config = searchspace.get_net_config(uid, args_dataset)
  network = get_cell_based_tiny_net(config)
  
  start_time = time.time()
  network.K = np.zeros((args_batch_size, args_batch_size))

  for name, module in network.named_modules():
    if 'ReLU' in str(type(module)):
      module.register_forward_hook(counting_forward_hook)
      

  network = network.to(device)
  random.seed(args_seed)
  np.random.seed(args_seed)
  torch.manual_seed(args_seed)
  
  data_iterator = iter(train_loader)
  x, target = next(data_iterator)
  #x2 = torch.clone(x)
  #x2 = x2.to(device)
  x, target = x.to(device), target.to(device)
  
  with torch.no_grad():
    y, out = network(x)

  #network(x2.to(device)
  
  y = y.detach()
  out = out.detach()
  x.detach()
  target.detach()
  



  sign, score = np.linalg.slogdet(network.K)
  execution_time = time.time()-start_time
  
  del config
  del network.K
  del network
  del x
  del target 
  del y
  del out
  torch.cuda.empty_cache()
  gc.collect()

  print(uid, score, execution_time, searchspace.get_more_info(uid, args_dataset)['test-accuracy'])
  results.append([uid, score, execution_time, searchspace.get_more_info(uid, args_dataset)['test-accuracy']])

fields = ['uid', 'score', 'time', 'accuracy']

with open(args_save_loc+args_dataset+'/scores.csv', 'w') as f:

  write = csv.writer(f)
      
  write.writerow(fields)
  write.writerows(results)
