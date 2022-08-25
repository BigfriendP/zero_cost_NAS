import pandas as pd
from scipy import stats
import math

"""## Reading relu_score and synflow CSV files for each dataset"""

# importing cifar10 scores
cifar10_relu_csv = pd.read_csv('results/cifar10/hook_logdet/cifar10-hook_logdet.csv')
cifar10_relu_csv.replace(to_replace=-math.inf, value=0.0, inplace=True)
cifar10_synflow_csv = pd.read_csv('results/cifar10/synflow/cifar10-synflow.csv')

# importing cifar100 scores
cifar100_relu_csv = pd.read_csv('results/cifar100/hook_logdet/cifar100-hook_logdet.csv')
cifar100_relu_csv.replace(to_replace=-math.inf, value=0.0, inplace=True)
cifar100_synflow_csv = pd.read_csv('results/cifar100/synflow/cifar100-synflow.csv')

# importing imagenet scores
imagenet_relu_csv = pd.read_csv('results/ImageNet16-120/hook_logdet/ImageNet16-120-hook_logdet.csv')
imagenet_relu_csv.replace(to_replace=-math.inf, value=0.0, inplace=True)
imagenet_synflow_csv = pd.read_csv('results/ImageNet16-120/synflow/ImageNet16-120-synflow.csv')

# importing accuracies full trained models
cifar10_test_acc = pd.read_csv('accuracy_full_trained_models/accuracy_cifar10.csv')
cifar100_test_acc = pd.read_csv('accuracy_full_trained_models/accuracy_cifar100.csv')
imagenet_test_acc = pd.read_csv('accuracy_full_trained_models/accuracy_ImageNet16-120.csv')

"""## Calculating the top 10

### Performing normalization or standardization (don't execute both) to make the two metrics more comparable
"""

# normalizing scores
cifar10_relu_csv['score'] = (cifar10_relu_csv['score']-cifar10_relu_csv['score'].min())/(cifar10_relu_csv['score'].max()-cifar10_relu_csv['score'].min())
cifar10_synflow_csv['synflow'] = (cifar10_synflow_csv['synflow']-cifar10_synflow_csv['synflow'].min())/(cifar10_synflow_csv['synflow'].max()-cifar10_synflow_csv['synflow'].min())

cifar100_relu_csv['score'] = (cifar100_relu_csv['score']-cifar100_relu_csv['score'].min())/(cifar100_relu_csv['score'].max()-cifar100_relu_csv['score'].min())
cifar100_synflow_csv['synflow'] = (cifar100_synflow_csv['synflow']-cifar100_synflow_csv['synflow'].min())/(cifar100_synflow_csv['synflow'].max()-cifar100_synflow_csv['synflow'].min())

imagenet_relu_csv['score'] = (imagenet_relu_csv['score']-imagenet_relu_csv['score'].min())/(imagenet_relu_csv['score'].max()-imagenet_relu_csv['score'].min())
imagenet_synflow_csv['synflow'] = (imagenet_synflow_csv['synflow']-imagenet_synflow_csv['synflow'].min())/(imagenet_synflow_csv['synflow'].max()-imagenet_synflow_csv['synflow'].min())

'''# standardizing scores
cifar10_relu_csv['score'] = (cifar10_relu_csv['score']-cifar10_relu_csv['score'].mean())/cifar10_relu_csv['score'].std()
cifar10_synflow_csv['synflow'] = (cifar10_synflow_csv['synflow']-cifar10_synflow_csv['synflow'].mean())/cifar10_synflow_csv['synflow'].std()

cifar100_relu_csv['score'] = (cifar100_relu_csv['score']-cifar100_relu_csv['score'].mean())/cifar100_relu_csv['score'].std()
cifar100_synflow_csv['synflow'] = (cifar100_synflow_csv['synflow']-cifar100_synflow_csv['synflow'].mean())/cifar100_synflow_csv['synflow'].std()

imagenet_relu_csv['score'] = (imagenet_relu_csv['score']-imagenet_relu_csv['score'].mean())/imagenet_relu_csv['score'].std()
imagenet_synflow_csv['synflow'] = (imagenet_synflow_csv['synflow']-imagenet_synflow_csv['synflow'].mean())/imagenet_synflow_csv['synflow'].std()'''

"""## Creating the CSV files containing the combined metrics"""

# trying different coefficients for the convex combinations
coeff = {"Comb1": 0.2, "Comb2": 0.4, "Comb3": 0.5, "Comb4": 0.6, "Comb5": 0.8}

datasets = {"cifar10": (cifar10_relu_csv, cifar10_synflow_csv), "cifar100": (cifar100_relu_csv, cifar100_synflow_csv), "ImageNet16-120": (imagenet_relu_csv, imagenet_synflow_csv)}

convex_comb_csv = {"cifar10": [], "cifar100":[], "ImageNet16-120":[]}

for dataset_name, metrics_tuple in datasets.items():
  df = pd.DataFrame(columns=['uid', 'Comb1', 'Comb2', 'Comb3', 'Comb4', 'Comb5', 'time'])
  df['time'] = metrics_tuple[0]['time']+metrics_tuple[1]['time']
  df['uid'] = metrics_tuple[0]['uid']
   
  for column_name, c in coeff.items():
    new_metric = c*metrics_tuple[0]['score']+(1-c)*metrics_tuple[1]['synflow']
    df[column_name] = new_metric
  convex_comb_csv[dataset_name] = df

cifar10_convex_comb = convex_comb_csv["cifar10"]
cifar100_convex_comb = convex_comb_csv["cifar100"]
imagenet_convex_comb = convex_comb_csv["ImageNet16-120"]

"""## Calculating the correlations of the convex combinations with the test accuracy"""

# finding the best convex combinations according to the spearman rank
cifar10_corr = []
cifar100_corr = []
imagenet_corr = []
for comb in ['Comb1', 'Comb2', 'Comb3', 'Comb4', 'Comb5']:
  sp_cifar10 = stats.spearmanr(cifar10_convex_comb[comb],cifar10_test_acc['accuracy']).correlation
  cifar10_corr.append((comb, sp_cifar10))
  sp_cifar100 = stats.spearmanr(cifar100_convex_comb[comb],cifar100_test_acc['accuracy']).correlation
  cifar100_corr.append((comb, sp_cifar100))
  sp_imagenet = stats.spearmanr(imagenet_convex_comb[comb],imagenet_test_acc['accuracy']).correlation
  imagenet_corr.append((comb,sp_imagenet))

cifar10_corr.sort(key=lambda y: y[1], reverse = True)
cifar100_corr.sort(key=lambda y: y[1], reverse = True)
imagenet_corr.sort(key=lambda y: y[1], reverse = True)

print(cifar10_corr,'\n')
print(cifar100_corr,'\n')
print(imagenet_corr,'\n')

"""## Exporting the CSV files with the best convex combination according to the spearman correlation"""

# creating the csv files to export
cifar10_comb = pd.DataFrame(columns=['uid', 'combined_score', 'time'])
cifar100_comb = pd.DataFrame(columns=['uid', 'combined_score', 'time'])
imagenet_comb = pd.DataFrame(columns=['uid', 'combined_score', 'time'])

cifar10_comb['uid'] = cifar10_convex_comb['uid']
cifar10_comb['combined_score'] = cifar10_convex_comb['Comb4']
cifar10_comb['time'] = cifar10_convex_comb['time']

cifar100_comb['uid'] = cifar100_convex_comb['uid']
cifar100_comb['combined_score'] = cifar100_convex_comb['Comb4']
cifar100_comb['time'] = cifar100_convex_comb['time']

imagenet_comb['uid'] = imagenet_convex_comb['uid']
imagenet_comb['combined_score'] = imagenet_convex_comb['Comb4']
imagenet_comb['time'] = imagenet_convex_comb['time']

# saving csv files
cifar10_comb.to_csv("results/cifar10/combined/cifar10-combined.csv", index = False) 
cifar100_comb.to_csv("results/cifar100/combined/cifar100-combined.csv", index = False)
imagenet_comb.to_csv("results/ImageNet16-120/combined/ImageNet16-120-combined.csv", index = False)