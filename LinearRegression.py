# -*- coding:utf-8 -*-
#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():

    def __init__(self, iteration=100, learning_rate=0.001):
        self.iteration = iteration
        self.learning_rate = learning_rate
    def fit(self, X, y):
        n, m = X.shape
        self.weight = np.zeros((n, 1))
        self.bias = np.zeros(1)
        for i in range(self.iteration):
            loss = np.sum((np.dot(self.weight.T, X) - y) ** 2)
            print(loss)
            dw = 2 * np.dot(X, np.dot(X.T, self.weight) - y.T)
            self.weight = self.weight - self.learning_rate * dw
        print(self.weight)
    def predict(self, X):
        return np.dot(self.weight, X)

X = np.random.uniform(-10, 10, (2, 10))
w = np.asarray([[1], [2]])
y = np.dot(w.T, X)

lr = LinearRegression()
lr.fit(X, y)