# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 15:01:47 2023

@author: Qi
"""

import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(41)

#讀取iris分類資料集
iris = datasets.load_iris()
X = iris.data
Y = iris.target

#資料集中，2/3當作訓練集，1/3當作測試集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33)
naive_bayes_clf = GaussianNB()
naive_bayes_clf = naive_bayes_clf.fit(X_train, Y_train)
Y_predict = naive_bayes_clf.predict(X_test)
score = accuracy_score(Y_test, Y_predict)
print("鳶尾花分類的預測準確率: ", score)