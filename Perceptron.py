# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 20:50:38 2016

@author: JohnnyJ
"""

import numpy as np

class Perceptron(object):
    #eta - learning rate
    #n_iter - the number of epochs or passes over the training data. 
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):  #Perceptron Model 
        self .w_ = np.zeros(1+X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))  #compute the predicted output value.
                self.w_[1:] += update * xi  #Update the weights.  
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
        
    #For 2-D arrays it is equivalent to matrix multiplication, and for 1-D arrays to inner product of vectors
    # so in this case it is doing N  1X2 * 1X2 as an inner product to get so x1*w1 + x2*w2 as the result.
    # so we get a 1 row by 300 column matrix returned....ugh
    def net_input(self,X):  # Calculating Z which is X dot transpose(W)
        #print (np.dot(X, self.w_[1:]))
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):  #activation function or the unit step function
        #print (self.w_[1:])
        return np.where(self.net_input(X) >= 0.0, 1, -1)
        