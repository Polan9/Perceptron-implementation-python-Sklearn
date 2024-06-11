import cv2
import uuid
import os
import time
import numpy as np
import sklearn as sk
from sklearn import datasets
from sklearn import model_selection
from sklearn.metrics import accuracy_score


class Preceptron_binarny():
    def __init__(self, learning_rate=0.01, n_iters=5000,):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.activation_function = self.activation
        self.weights = [0.5,-0.5]
        self.bias = 0

    def activation(self, x):
        return 1 if x > 0 else 0

    def suma_wazona(self, X):
        return np.dot(X, self.weights) + self.bias
    def predict(self, X):
        X = np.asarray(X)
        net_input_value = self.suma_wazona(X)
        return np.where(net_input_value >= 0, 1, 0)

    def update_w(self, X, y, d):
        error = d - y
        self.weights += self.learning_rate * error * X
        self.bias += self.learning_rate * error


    def suma(self, wagi, x , bias):
        for i in range(len(wagi)):

            return wagi[i] *  x[i] + bias

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, xi in enumerate(X):
                net_input_value = self.suma_wazona(xi)
                y_pred = self.activation(net_input_value)
                self.update_w(xi, y_pred, y[idx])



Prc = Preceptron_binarny()

X = np.array([[2, 3], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 0, 0, 1])

x1 = np.array([[3,2]])

Prc.fit(X,y)
y_pred = Prc.predict(x1)
print(y_pred)


data = datasets.load_iris()

x = data['data']
y = data['target']

x = x[(y == 0) | (y == 1)]
y = y[(y == 0) | (y == 1)]


y[y == 0] = -1
y[y == 1] = 1

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, train_size=0.8)
Prc.fit(X_train,Y_train)

y_pred = Prc.predict(X_test)
print(y_pred)

accuracy = accuracy_score(Y_test, y_pred)
print(accuracy)


