# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 16:50:34 2023

@author: Qi
"""

%matplotlib inline
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#函式生成了300個樣本，其中有3個聚類中心，標準差為1
#"make_blobs"函式返回一個包含樣本特徵和對應標籤的元組，但在這只對特徵矩陣"X"感興趣，所以使用"_"將標籤部分丟棄，因此，程式碼將生成的樣本特徵矩陣賦值給變數"X"
X, _ = make_blobs(n_samples = 300, centers = 3, cluster_std = 1)

plt.figure(figsize = (6, 6))
plt.scatter(X[:, 0], X[:, 1], s = 30, marker = "x")
plt.show()

kmeans = KMeans(n_clusters = 3) #創建一個"KMeans"聚類器對象"kmeans"，"n_clusters"指定聚類器的數量
kmeans.fit(X) #"fit()"更新聚類質心的位置，直到達到指定的停止準則，或直到質心位置不再發生變化為止
Y = kmeans.predict(X) #使用訓練好的"KMeans"模型對輸入數據"X"進行預測。將X中的每個數據點進行聚類，將聚類結果存在Y數據中。Y中的每個元素都是一個整數，表示對應數據點所屬的族

plt.figure(figsize = (6, 6))
plt.scatter(X[:, 0], X[:, 1], c = Y, s = 30, marker = "x", cmap = "plasma") #"s"參數表示每個點的大小。"cmap"參數指定使用的顏色映射
centers = kmeans.cluster_centers_ #獲取聚類算法得到的每個族的中心點座標
plt.scatter(centers[:, 0], centers[:, 1], c = "black", s = 200, alpha = 0.8) #將聚類算法得到的每個族的中心點可視化。"centers[:, 0]"和"centers[:, 1]"分别表示中心点的x和y坐标。"c"參數表示中心點的顏色。"s"參數表示中心點大小。"alpha"參數表示中心點的透明度
plt.show()
