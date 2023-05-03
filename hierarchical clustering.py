# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:31:47 2023
@author: Qi
"""

%matplotlib inline
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib import pyplot as plt

X = [[0, 0], [1, 1], [2, 1], [1, 3], [2, 4], [3, 3]]
Z = linkage(X, "centroid") #將X矩陣傳入"linkage"函數中，使用"centroid"方法計算矩陣中所有點的距離
fig = plt.figure(figsize = (5, 3)) #創建一個大小為(5, 3)的新圖形，存放樹狀圖
dendrogram(Z)

plt.show()
