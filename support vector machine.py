# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 14:24:58 2023

@author: Qi
"""

%matplotlib inline
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

#讀取iris分類資料集
iris = datasets.load_iris() #"load_iris()"函數加載鳶尾花數據集，將其儲存在"iris"變量中
X = iris.data #從加載的鳶尾花數據集"iris"中獲取特徵數據，儲存在"X"變量中。數據集包含了150個樣本，每個樣本都有4個特徵，X大小為150 × 4，其中每行對應一個樣本，每列對應一個特徵
X = X[:, ::2] #對X進行切片操作，將每個樣本特徵向量中的奇數列全部刪除，只保留偶數列的數據。X的大小變為150 × 2
Y = iris.target

np.random.seed(29) #設置隨機數種子，以確保隨機生成器產生的隨機數序是可重現的

#資料集中，2/3當作訓練集，1/3當作測試集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33)
svm_clf = svm.SVC(kernel = "linear", C = 1, gamma = "auto") #"svm.SVC"函數創建一個支持向量機，其中"kernel = "linear"表示使用線性函數進行分類，"C = 1"表示正則化參數的值為1，"gamma = "auto"表示使用默認的gamma值
svm_clf = svm_clf.fit(X_train, Y_train) #使用訓練數據"X_train"和"Y_train"對支持向量分類機"svm_clf"進行訓練
Y_predict = svm_clf.predict(X_test) #使用訓練好的支持向量分類機"svm_clf"對測試數據"X_test"進行預測，從而得到預測結果"Y_predict"
score = accuracy_score(Y_test, Y_predict) #使用scikit-learn中的"accuracy_score"函數來計算支持向量機在測試數據集上的準確率
print("鳶尾花分類的預測準確率: ", score)

plt.figure(figsize = (6, 6)) #創建圖形窗口
colmap = np.array(["blue", "green", "red"]) #定義"colmap"的numpy數組，其中"blue", "green", "red"分別表示三種顏色，用於散點圖中區分不同的數據族
plt.scatter(X_test[:, 0], X_test[:, 1], c = colmap[Y_test], s = 150, marker = "o", alpha = 0.5) #X_test[:, 0]"和"X_test[:, 1]"表示樣本在第一個、第二個特徵或屬性的值，第一個和第二個特徵值為x和y座標，"c"指定散點的顏色，"s"指定散點大小，"marker"指定散點的形狀，"alpha"指定散點的透明度
plt.scatter(X_test[:, 0], X_test[:, 1], c = colmap[Y_predict], s = 50, marker = "o", alpha = 0.5)
plt.xlabel("Sepal length", fontsize = 12) #設置x軸的標籤文字
plt.ylabel("Petal length", fontsize = 12) #設置y軸的標籤文字
plt.show()
