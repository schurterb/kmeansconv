# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 17:34:12 2016

@author: user

2D convolutional classifier for mnist data set
unsupervised learning
"""


import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d

import numpy as np

theano.config.floatX = 'float32'

class CNN2D(object):
    
    def __init__(self, **kwargs):
        
        self.rng = kwargs.get('rng', np.random.RandomState(42))
        load_folder = kwargs.get('weights_folder', None)
        
        #Prepare the network weights
        if load_folder is not None:
            self.__load_config(load_folder)
            self.__load_weights(load_folder)
        else:
            self.nConvLayers = kwargs.get('convolutional_layers', 1)
            self.nConvFilters = kwargs.get('convolutional_filters', 20)
            self.convFilterSize = kwargs.get('filter_size', 7)
            
            self.regLayers = kwargs.get('regression_layers', (50 ,))
            self.nClasses = kwargs.get('number_classes',10)
            self.regLayers += (self.nClasses ,)
            
            self.activation = kwargs.get('activation', 'tanh')
            
            self.imageShape = kwargs.get('image_size', (28, 28))        
            
            self.sample_size = self.nConvLayers*(self.convFilterSize -1) +1
        
            self.__define_network()
            self.__init_weights()
            
        #symbolic variable for import to network
        self.X = T.tensor3('X')
        
        #create symbolic representation for network
        self.__create_model()

        #Currently, cpu only            
        self.forward = theano.function(inputs=[self.X], outputs=self.out, allow_input_downcast=True)
        
    
    """ Make predictions on some input x """
    def run(self, x, **kwargs):
        pass
    
    """ Store weights and meta-data to folder """
    def save(self, folder):
        pass
    
    """ Generate Definition of Network Structure """
    def __define_network(self):
        ## Define convolutional network component
        self.postConvImageShape = np.ndarray(2)
        self.postConvImageShape[0] = self.imageShape[0] - self.convFilterSize +1
        self.postConvImageShape[1] = self.imageShape[1] - self.convFilterSize +1
        self.convnet_shape = np.ndarray([self.nConvLayers, 4])
        self.convnet_shape[0,:] = [self.nConvFilters, 1, self.convFilterSize, self.convFilterSize]
        #self.convnet_shape[0,:] = [1, self.nConvFilters, self.convFilterSize, self.convFilterSize]
        for i in range(1, self.nConvLayers):
            self.convnet_shape[i,:] = [self.nConvFilters, self.nConvFilters, self.convFilterSize, self.convFilterSize]
        ## Define regression network component
        if self.regLayers is not None:
            self.regnet_shape = np.ndarray([len(self.regLayers), 2]) 
            self.regnet_shape[0, :] = \
                [self.nConvFilters*self.postConvImageShape[0]*self.postConvImageShape[1], self.regLayers[0]]
            for i in range(1, len(self.regLayers)):
                self.regnet_shape[i,:] = [self.regLayers[i-1], self.regLayers[i]]
        else:
            self.regnet_shape = None
        
    """ Randomly Initialize Network Weights """
    def __init_weights(self):
        self.cw = () #Convolution weights
        self.cb = () #Convolution biases
        for layer in range(0, self.nConvLayers):
            #Initialize within optimum range for tanh activation]
            #Initialize convoluional layer weights
            fan_in = self.convnet_shape[layer, 1] * (self.sample_size**3)
            fan_out = self.convnet_shape[layer, 0] * (self.convFilterSize**3)
            bound = np.sqrt(6.0/(fan_in+fan_out))
            self.cw += (theano.shared(np.asarray(self.rng.uniform(low= -bound,
                                                                  high= bound,
                                                                  size = self.convnet_shape[layer, :]),
                                                dtype=theano.config.floatX), name='cw-'+str(layer)) ,)
            self.cb += (theano.shared(np.asarray(np.ones(self.convnet_shape[layer, 0]),
                                                dtype=theano.config.floatX), name='cb-'+str(layer)) ,)
        self.rw = () #Regression weights
        self.rb = () #Regression biases
        for layer in range(0, len(self.regLayers)):
            #Initialize regression layer weights
            bound = 0.75
            self.rw += (theano.shared(np.asarray(self.rng.uniform(low= -bound,
                                                                  high= bound,
                                                                  size = self.regnet_shape[layer, :]),
                                                dtype=theano.config.floatX), name='rw-'+str(layer)) ,)
            self.rb += (theano.shared(np.asarray(np.ones(self.regnet_shape[layer, 1]),
                                                dtype=theano.config.floatX), name='rb-'+str(layer)) ,)

        
    
    """ Load network meta-data from folder """
    def __load_config(self, folder):
        pass
    
    """ Load weights from folder """
    def __load_weights(self, folder):
        pass
    
    """ Create Symbolic Theano definition for network """
    def __create_model(self):
        
        #Prepare input tensor
        Xin = self.X.dimshuffle(0, 'x', 1, 2)
        
        #Set up convolutional layers
        out = T.tanh(conv2d(Xin, self.cw[0], border_mode='valid') + self.cb[0].dimshuffle('x',0,'x','x'))
        
        for layer in range(1, self.nConvLayers):
            out = T.tanh(conv2d(out, self.cw[layer], border_mode='valid') + self.cb[layer].dimshuffle('x',0,'x','x'))
            
        #Set up regression layers
        out = T.tanh(T.dot(out.flatten(2), self.rw[0]) + self.rb[0])
        
        for layer in range(1, len(self.regLayers)-1):
            out = T.tanh(T.dot(out, self.rw[layer]) + self.rb[layer])
            
        #Set up output layer            
        out = T.nnet.sigmoid(T.dot(out, self.rw[-1]) + self.rb[-1])
            
        self.out = out
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    