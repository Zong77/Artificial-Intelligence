# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:31:47 2023

@author: Qi
"""

%matplotlib inline
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib import pyplot as plt

X = [[0, 0], [1, 1], [2, 1], [1, 3], [2, 4], [3, 3]]
Z = linkage(X, "centroid")
fig = plt.figure(figsize = (5, 3))
dendrogram(Z)

plt.show()