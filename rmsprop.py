# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 21:31:22 2016

@author: user

A class for building the theano equations
 specifically for standard stochatistic gradient descent.
Designed for use with the cnnclassifier.
"""

import theano
from theano import tensor as T
import numpy as np

import os

class RMSProp:
    
    def __init__(self, network, **kwargs):
        self.loadFolder = kwargs.get('trainer_folder', None)
        self.cw = network.cw
        self.cb = network.cb
        self.rw = network.rw
        self.rb = network.rb
        self.convnet = network.convnet_shape
        self.regnet = network.regnet_shape
        self.lr = kwargs.get('learning_rate', 0.001)
        self.b2 = kwargs.get('beta_decay', 0.9)
        self.damp = kwargs.get('damping', 1.0e-08)
        self.updates = []
    
    def addConvolutionGradients(self, cw_grads, cb_grads):
        if self.convnet is not None:
            self._prepare_convolution_velocity()
            self.updates += self._calculate_updates(self.cw, self.vcw, cw_grads,
                                                    self.cb, self.vcb, cb_grads)
        
    def addRegressionGradients(self, rw_grads, rb_grads):
        if self.regnet is not None:
            self._prepare_regression_velocity()
            self.updates += self._calculate_updates(self.rw, self.vrw, rw_grads,
                                                    self.rb, self.vrb, rb_grads)
        
    def getUpdates(self):
        return self.updates
        
    def store(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        for i in range(0, self.convnet.shape[0]):
            self.vcw[i].get_value().tofile(folder + 'convlayer_'+str(i)+'_weight_var.csv', sep=',')
            self.vcb[i].get_value().tofile(folder + 'convlayer_'+str(i)+'_bias_var.csv', sep=',')
        for i in range(0, self.regnet.shape[0]):
            self.vrw[i].get_value().tofile(folder + 'reglayer_'+str(i)+'_weight_var.csv', sep=',')
            self.vrb[i].get_value().tofile(folder + 'reglayer_'+str(i)+'_bias_var.csv', sep=',')

    def _calculate_updates(self, w, vw, w_grads, b, vb, b_grads):
        vw_updates = [
            (r, (self.b2*r) + (1-self.b2)*grad**2)
            for r, grad in zip(vw, w_grads)         
        ]
        vb_updates = [
            (r, (self.b2*r) + (1-self.b2)*grad**2)
            for r, grad in zip(vb, b_grads)
        ]
        w_updates = [
            (param, param - self.lr*(grad/(T.sqrt(r) + self.damp)))
            for param, r, grad in zip(w, vw, w_grads)                   
        ]
        b_updates = [
            (param, param - self.lr*(grad/(T.sqrt(r) + self.damp)))
            for param, r, grad in zip(b, vb, b_grads)
        ]  
        return vw_updates + vb_updates + w_updates + b_updates

    def _prepare_convolution_velocity(self):
        self.vcw = ()
        self.vcb = ()
        if self.load_folder:            
            trainer_folder = self.load_folder + 'trainer/rmsprop/'
            if os.path.exists(trainer_folder):
                for layer in range(0, len(self.cw)):
                    self.vcw = self.vcw + (theano.shared(np.genfromtxt(
                        self.load_folder+'convlayer_'+str(layer)+'_weight_var.csv', 
                        delimiter=',').reshape(self.convnet[layer,:]).astype( 
                            theano.config.floatX), name='vcw'+str(layer)) ,)
                    self.vcb = self.vcb + (theano.shared(np.genfromtxt(
                        self.load_folder+'convlayer_'+str(layer)+'_bias_var.csv', 
                        delimiter=',').reshape(self.convnet[layer,0]).astype(
                            theano.config.floatX), name='vcb'+str(layer)) ,)
        
        if(len(self.vcw) == 0):
            for layer in range(0, len(self.cw)):
                    self.vcw = self.vcw + (theano.shared(np.ones(
                        self.convnet[layer,:], dtype=theano.config.floatX), 
                            name='vcw'+str(layer)) ,)
                    self.vcb = self.vcb + (theano.shared(np.ones(
                        self.convnet[layer,0], dtype=theano.config.floatX), 
                            name='vcb'+str(layer)) ,)


    def _prepare_regression_velocity(self):
        self.vrw = ()
        self.vrb = ()
        if self.load_folder:            
            trainer_folder = self.load_folder + 'trainer/rmsprop/'
            if os.path.exists(trainer_folder):
                for layer in range(0, len(self.rw)):
                    self.vrw = self.vrw + (theano.shared(np.genfromtxt(
                        self.load_folder+'reglayer_'+str(layer)+'_weight_var.csv', 
                        delimiter=',').reshape(self.regnet[layer,:]).astype( 
                            theano.config.floatX), name='vrw'+str(layer)) ,)
                    self.vrb = self.vrb + (theano.shared(np.genfromtxt(
                        self.load_folder+'reglayer_'+str(layer)+'_bias_var.csv', 
                        delimiter=',').reshape(self.regnet[layer,0]).astype(
                            theano.config.floatX), name='vrb'+str(layer)) ,)
        
        if(len(self.vrw) == 0):
            for layer in range(0, len(self.rw)):
                    self.vrw = self.vrw + (theano.shared(np.ones(
                        self.regnet[layer,:], dtype=theano.config.floatX),
                            name='vrw'+str(layer)) ,)
                    self.vrb = self.vrb + (theano.shared(np.ones(
                            self.regnet[layer,0], dtype=theano.config.floatX), 
                                name='vrb'+str(layer)) ,)
                                
        