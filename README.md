# NASWOT
Neural Architecture Search is the task of automati-
cally design a neural network that will yield good performance on
a given dataset. This task has seen growing interest since to man-
ually craft an architecture requires substantial effort of human
experts, because of the immense variety of possible alternatives
faced during the hand process. However the traditional NAS
approaches are heavy in terms of time and resources required,
due to the need to train and evaluate multiple networks. Thus
we perform an empirical study to investigate the capacity of
four different training-free metrics to score the performance a
network will have on a dataset after it is trained. Furthermore we
analyze two search strategies which take advantage of one of these
metric to explore the architectures search space, with the aim to
find the best possible network. Eventually we perform multiple
experiments using all the possible (metric, search algorithm)
combination. We show in results section that the NAS framework
we implement is often able to find networks whose performance
are near to the best architecture in the search space we explore,
while requiring computation time orders of magnitude lower than
traditional NAS.

# Script details
- score_networks.py score the networks on a particular dataset with a specific free-training metric.
- random_search.py execute the random search on a particular dataset with a specific free-training metric.
- evolutionary_search.py execute the evolutionary search on a particular dataset with a specific free-training metric. By default it does not take advantage of the warmup initializion, to perform it just change the parameter "warmup" inside the script to a number greater than "pool_size".
- metrics_convex_combination.py compute the convex combination of hook_logdet score and synflow for all the datasets, and save the new metrics combined inside a CSV.
- Result_analysis.ipynb is a colab notebook that show some details about the free-training metrics and some plots about the search experiment performed.

# Guide for running scripts  
score_networks.py, random_search.py and evolutionary_search.py take command line inputs, in particular all of them require two mandatory inputs: --dataset (-d) and --score (-s). For all the three script the accepted datasets are "cifar10", "cifar100" and "ImageNet16-120", while the accepted scores change between score_networks.py and random_search.py, evolutionary_search.py.
The valid scores taken by score_networks.py are ["hook_logdet",  ]
