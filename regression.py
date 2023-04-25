# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 00:12:07 2023

@author: Qi
"""

%matplotlib inline
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

np.random.seed(29)

boston = datasets.load_boston()
print("資料集的特徵欄位名稱為: ", boston.feature_names)
X = boston.data
Y = boston.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
regression_clf = LinearRegression()
regression_clf.fit(X_train, Y_train)
Y_predict = regression_clf.predict(X_test)
score = r2_score(Y_test, Y_predict)
print("房價的預測準確率: ", score)

plt.plot(Y_test, label = "Real Price")
plt.plot(Y_predict, label = "Predict Price")
plt.legend()

![](https://github.com/Zong77/Artificial-Intelligence/blob/main/regression.png)
