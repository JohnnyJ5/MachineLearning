# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 20:46:24 2016

@author: JohnnyJ
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self .w_ = np.zeros(1+X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
        
    def net_input(self,X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

#columns are Sepal Length, Sepal Width, Petal Length, Petal Width, Class Label
#First 50 is Iris-setosa
#Second 50 is Iris-versicolor
#Third 50 is Iris-virginica
df = pd.read_csv('https://raw.githubusercontent.com/rasbt/python-machine-learning-book/master/code/datasets/iris/iris.data', header=None)
print (len(df))

#first 100 lines get column 4 values.
y = df.iloc[0:100, 4].values

#keep an array where Iris-setosa is -1 and everything else is 1.
y = np.where(y == 'Iris-setosa', -1, 1)

#first 100 lines get column 0 and 2
X = df.iloc[0:100, [0,2]].values

#scatter plot for sepal length vs petal length.  Mark red o and label it setosa
plt.scatter(X[:50, 0], X[:50,1], color='red', marker='o', label='setosa')

#scatter plot 50-100 of sepal length vs petal length.  Mark blue 
plt.scatter(X[50:100, 0], X[50:100,1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()


ppn = Perceptron(eta=0.1, n_iter = 10)
ppn.fit(X,y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of mis classifications')
plt.show()

def plot_decision_regions(X, y, classifier, resolution = 0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0], y=X[y==c1, 1],
                    alpha = 0.8, c=cmap(idx),
                    marker=markers[idx], label=c1)
        

plot_decision_regions(X, y, classifier = ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc = 'upper left')
plt.show()

