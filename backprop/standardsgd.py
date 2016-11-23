# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 21:31:22 2016

@author: user

A class for building the theano equations
 specifically for standard stochatistic gradient descent.
Designed for use with the cnnclassifier.
"""


class StandardSGD:
    
    def __init__(self, network, **kwargs):
        self.cw = network.cw
        self.cb = network.cb
        self.rw = network.rw
        self.rb = network.rb
        self.lr = kwargs.get('learning_rate', 0.001)
        self.updates = []
    
    def addConvolutionGradients(self, cw_grads, cb_grads):
        self.updates += self._calculate_updates(self.cw, cw_grads,
                                               self.cb, cb_grads)
        
    def addRegressionGradients(self, rw_grads, rb_grads):
        self.updates += self._calculate_updates(self.rw, rw_grads,
                                               self.rb, rb_grads)
        
    def getUpdates(self):
        return self.updates

    def save(self, folder):
        return True

    def _calculate_updates(self, w, w_grads, b, b_grads):
        w_updates = [
            (param, param - self.lr*grad)
            for param, grad in zip(w, w_grads)                   
        ]
        b_updates = [
            (param, param - self.lr*grad)
            for param, grad in zip(b, b_grads)                   
        ]
        return w_updates + b_updates
        