# NASWOT
Neural Architecture Search without training.
The networks of the NATS bench can be evaluated for three different datasets: cifar 10, cifar100, ImageNet16-120.
The dataset to use can be decided changing the parameters in naswot.py args_dataset, args_data_loc respectively at row 49,50.

## cifar10
args_dataset = 'cifar10' ;
args_data_loc = 'data/cifar10'

## cifar100
args_dataset = 'cifar100' ;
args_data_loc = 'data/cifar100'

## ImageNet16-120
args_dataset = 'ImageNet16-120' ;
args_data_loc = 'data/ImageNet16-120'
