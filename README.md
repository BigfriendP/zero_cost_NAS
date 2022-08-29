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
