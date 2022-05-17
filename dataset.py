"""## Get dataset with transform"""

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from xautodl.datasets.DownsampledImageNet import ImageNet16

Dataset2Class = {'cifar10' : 10,
                 'cifar100' : 100,
                 'ImageNet16-120': 120}

def get_datasets(name, root):

  if name == 'cifar10':
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std  = [x / 255 for x in [63.0, 62.1, 66.7]]
  elif name == 'cifar100':
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std  = [x / 255 for x in [68.2, 65.4, 70.4]]
  elif name.startswith('ImageNet16'):
    mean = [x / 255 for x in [122.68, 116.66, 104.01]]
    std  = [x / 255 for x in [63.22,  61.26 , 65.09]]
  else:
    raise TypeError("Unknow dataset : {:}".format(name))

  
  # Data Argumentation
  if name == 'cifar10' or name == 'cifar100':
    lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)]
    train_transform = transforms.Compose(lists)
    test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    xshape = (1, 3, 32, 32)
  elif name.startswith('ImageNet16'):
    lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(16, padding=2), transforms.ToTensor(), transforms.Normalize(mean, std)]
    train_transform = transforms.Compose(lists)
    test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    xshape = (1, 3, 16, 16)
  else:
    raise TypeError("Unknow dataset : {:}".format(name))


  if name == 'cifar10':
    train_data = dset.CIFAR10 (root, train=True , transform=train_transform, download=True)
    test_data  = dset.CIFAR10 (root, train=False, transform=test_transform , download=True)
    assert len(train_data) == 50000 and len(test_data) == 10000
  elif name == 'cifar100':
    train_data = dset.CIFAR100(root, train=True , transform=train_transform, download=True)
    test_data  = dset.CIFAR100(root, train=False, transform=test_transform , download=True)
    assert len(train_data) == 50000 and len(test_data) == 10000
  elif name == 'ImageNet16-120':
    train_data = ImageNet16(root, True , train_transform, 120)
    test_data  = ImageNet16(root, False, test_transform , 120)
    assert len(train_data) == 151700 and len(test_data) == 6000
  else: raise TypeError("Unknow dataset : {:}".format(name))


  class_num = Dataset2Class[name]
  return train_data, test_data, xshape, class_num

"""## Get data"""

def get_data(dataset, data_loc, batch_size, pin_memory=True):
  
  train_data, valid_data, xshape, class_num = get_datasets(dataset, data_loc)
  train_data.transform.transforms = train_data.transform.transforms[2:]
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                             num_workers=0, pin_memory=pin_memory)
  return train_loader
