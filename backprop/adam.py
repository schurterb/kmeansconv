# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 21:31:22 2016

@author: user

A class for building the theano equations
 specifically for standard stochatistic gradient descent.
Designed for use with the cnnclassifier.
"""

import os

import theano
from theano import tensor as T
import numpy as np

class ADAM:
    
    def __init__(self, network, **kwargs):
        self.load_folder = kwargs.get('trainer_folder', None)
        self.cw = network.cw
        self.cb = network.cb
        self.rw = network.rw
        self.rb = network.rb
        self.convnet = network.convnet_shape
        self.regnet = network.regnet_shape
        self.lr = kwargs.get('learning_rate', 0.001)
        self.b1 = kwargs.get('beta1_decay', 0.9)
        self.b2 = kwargs.get('beta2_decay', 0.9)
        self.damp = kwargs.get('damping', 1.0e-08)
        self.updates = []
        self.t = theano.shared(np.asarray(1, dtype=theano.config.floatX))
    
    def addConvolutionGradients(self, cw_grads, cb_grads):
        if self.convnet is not None:
            self._prepare_convolution_velocity()
            self._prepare_convolution_momentum()
            self.updates += self._calculate_updates(self.cw, self.vcw, self.mcw, cw_grads,
                                                    self.cb, self.vcb, self.mcb, cb_grads)
        
    def addRegressionGradients(self, rw_grads, rb_grads):
        if self.regnet is not None:
            self._prepare_regression_velocity()
            self._prepare_regression_momentum()
            self.updates += self._calculate_updates(self.rw, self.vrw, self.mrw, rw_grads,
                                                    self.rb, self.vrb, self.mrb, rb_grads)
        
    def getUpdates(self):
        return self.updates

    def save(self, folder):
        if not folder.endswith("/"):
            folder += "/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        for i in range(0, self.convnet.shape[0]):
            self.vcw[i].get_value().tofile(folder + 'convlayer_'+str(i)+'_weight_var.csv', sep=',')
            self.vcb[i].get_value().tofile(folder + 'convlayer_'+str(i)+'_bias_var.csv', sep=',')
            self.mcw[i].get_value().tofile(folder + 'convlayer_'+str(i)+'_weight_mnt.csv', sep=',')
            self.mcb[i].get_value().tofile(folder + 'convlayer_'+str(i)+'_bias_mnt.csv', sep=',')
        for i in range(0, self.regnet.shape[0]):
            self.vrw[i].get_value().tofile(folder + 'reglayer_'+str(i)+'_weight_var.csv', sep=',')
            self.vrb[i].get_value().tofile(folder + 'reglayer_'+str(i)+'_bias_var.csv', sep=',')
            self.mrw[i].get_value().tofile(folder + 'reglayer_'+str(i)+'_weight_mnt.csv', sep=',')
            self.mrb[i].get_value().tofile(folder + 'reglayer_'+str(i)+'_bias_mnt.csv', sep=',')
            
            
    def _calculate_updates(self, w, vw, mw, w_grads, b, vb, mb, b_grads):
        mw_updates = [
            (m, (self.b1*m) + ((1- self.b1)*grad))
            for m, grad in zip(mw, w_grads)                   
        ]
        mb_updates = [
            (m, (self.b1*m) + ((1- self.b1)*grad))
            for m, grad in zip(mb, b_grads)                   
        ]
        vw_updates = [
            (v, ((self.b2*v) + (1-self.b2)*(grad**2)) )
            for v, grad in zip(vw, w_grads)                   
        ]
        vb_updates = [
            (v, ((self.b2*v) + (1-self.b2)*(grad**2)) )
            for v, grad in zip(vb, b_grads)                   
        ]
        w_updates = [
            (param, param - self.lr * (m/(1- (self.b1**self.t) ))/(T.sqrt(v/(1- (self.b2**self.t) ))+self.damp))
            for param, m, v in zip(w, mw, vw)                   
        ]
        b_updates = [
            (param, param - self.lr * ( m/(1- (self.b1**self.t) ))/(T.sqrt( v/(1- (self.b2**self.t) ))+self.damp))
            for param, m, v in zip(b, mb, vb)                   
        ]
        t_update = [
            (self.t, self.t + 1)
        ]
        return mw_updates + mb_updates + vw_updates + vb_updates + w_updates + b_updates# + t_update
        

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


    def _prepare_convolution_momentum(self):
        self.mcw = ()
        self.mcb = ()
        if self.load_folder:            
            trainer_folder = self.load_folder + 'trainer/adam/'
            if os.path.exists(trainer_folder):
                for layer in range(0, len(self.cw)):
                    self.mcw = self.mcw + (theano.shared(np.genfromtxt(
                        self.load_folder+'convlayer_'+str(layer)+'_weight_mnt.csv', 
                        delimiter=',').reshape(self.convnet[layer,:]).astype( 
                            theano.config.floatX), name='mcw'+str(layer)) ,)
                    self.mcb = self.mcb + (theano.shared(np.genfromtxt(
                        self.load_folder+'convlayer_'+str(layer)+'_bias_mnt.csv', 
                        delimiter=',').reshape(self.convnet[layer,0]).astype(
                            theano.config.floatX), name='mcb'+str(layer)) ,)
        
        if(len(self.mcw) == 0):
            for layer in range(0, len(self.cw)):
                    self.mcw = self.mcw + (theano.shared(np.ones(
                        self.convnet[layer,:], dtype=theano.config.floatX), 
                            name='mcw'+str(layer)) ,)
                    self.mcb = self.mcb + (theano.shared(np.ones(
                        self.convnet[layer,0], dtype=theano.config.floatX), 
                            name='mcb'+str(layer)) ,)


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
                        delimiter=',').reshape(self.regnet[layer,1]).astype(
                            theano.config.floatX), name='vrb'+str(layer)) ,)
        
        if(len(self.vrw) == 0):
            for layer in range(0, len(self.rw)):
                    self.vrw = self.vrw + (theano.shared(np.ones(
                        self.regnet[layer,:], dtype=theano.config.floatX),
                            name='vrw'+str(layer)) ,)
                    self.vrb = self.vrb + (theano.shared(np.ones(
                            self.regnet[layer,1], dtype=theano.config.floatX), 
                                name='vrb'+str(layer)) ,)
                                
    
    def _prepare_regression_momentum(self):
        self.mrw = ()
        self.mrb = ()
        if self.load_folder:            
            trainer_folder = self.load_folder + 'trainer/adam/'
            if os.path.exists(trainer_folder):
                for layer in range(0, len(self.rw)):
                    self.mrw = self.mrw + (theano.shared(np.genfromtxt(
                        self.load_folder+'reglayer_'+str(layer)+'_weight_mnt.csv', 
                        delimiter=',').reshape(self.regnet[layer,:]).astype( 
                            theano.config.floatX), name='mrw'+str(layer)) ,)
                    self.mrb = self.mrb + (theano.shared(np.genfromtxt(
                        self.load_folder+'reglayer_'+str(layer)+'_bias_mnt.csv', 
                        delimiter=',').reshape(self.regnet[layer,1]).astype(
                            theano.config.floatX), name='mrb'+str(layer)) ,)
        
        if(len(self.mrw) == 0):
            for layer in range(0, len(self.rw)):
                    self.mrw = self.mrw + (theano.shared(np.ones(
                        self.regnet[layer,:], dtype=theano.config.floatX),
                            name='mrw'+str(layer)) ,)
                    self.mrb = self.mrb + (theano.shared(np.ones(
                            self.regnet[layer,1], dtype=theano.config.floatX), 
                                name='mrb'+str(layer)) ,)
                            