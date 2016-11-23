# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 17:34:12 2016

@author: user

2D convolutional classifier for mnist data set
unsupervised learning
"""

import os
from sh import mkdir
import configparser

import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d

import numpy as np

theano.config.floatX = 'float32'

class CNN2D(object):
    
    def __init__(self, **kwargs):
        
        self.name = kwargs.get('name', 'CNN2D')
        self.rng = kwargs.get('rng', np.random.RandomState(42))
        load_folder = kwargs.get('network_folder', None)
        
        #Prepare the network weights
        self.configData = "network.cfg"
        self.network_folder = None
        if load_folder is not None:
            try:
                self.__load_config(load_folder)
                if not self.__load_weights(load_folder):
                    return None
                self.network_folder = load_folder
            except:
                load_folder=None 
        
        if self.network_folder is None:
            self.network_folder = self.name
            mkdir('-p', self.name)
            
            self.nConvLayers = kwargs.get('convolutional_layers', 1)
            self.nConvFilters = kwargs.get('convolutional_filters', 20)
            self.convFilterSize = kwargs.get('filter_size', 7)
            
            self.nRegLayers = kwargs.get('regression_layers', 1)
            self.regLayerSize = kwargs.get('layer_size', 50)
            self.regLayers = ()
            for i in range(self.nRegLayers):
                self.regLayers += (self.regLayerSize ,)
            
            self.nClasses = kwargs.get('number_classes',10)
            self.regLayers += (self.nClasses ,)
            
            self.activation = kwargs.get('activation', 'tanh')
            
            self.imageShape = kwargs.get('image_size', (28, 28))
            self.device = kwargs.get('device', 'cpu')
            
            self.sample_size = self.nConvLayers*(self.convFilterSize -1) +1

            self.__save_config(self.network_folder)      
        
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
        return self.forward(x)
    
    
    """ Store weights and meta-data to folder """
    def save(self, folder):
        if not folder.endswith("/"):
            folder += "/"
        if not os.path.exists(folder+"weights"):
            mkdir('-p', folder+"weights/conv")
            mkdir('-p', folder+"weights/reg")
            
        for i in range(0, self.nConvLayers):
            self.cw[i].get_value().tofile(folder+"weights/conv/layer_"+str(i)+"_weights.csv", sep=',')
            self.cb[i].get_value().tofile(folder+"weights/conv/layer_"+str(i)+"_bias.csv", sep=',')
        for i in range(0, self.nRegLayers):
            self.rw[i].get_value().tofile(folder+"weights/reg/layer_"+str(i)+"_weights.csv", sep=',')
            self.rb[i].get_value().tofile(folder+"weights/reg/layer_"+str(i)+"_bias.csv", sep=',')

    
    """ Generate Definition of Network Structure """
    def __define_network(self):
        ## Define convolutional network component
        self.postConvImageShape = np.ndarray(2)
        self.postConvImageShape[0] = self.imageShape[0] - self.nConvLayers*(self.convFilterSize -1)
        self.postConvImageShape[1] = self.imageShape[1] - self.nConvLayers*(self.convFilterSize -1)
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
    def __save_config(self, folder):
        try:
            if not folder.endswith("/"):
                folder += "/"
            file = folder+self.configData
            config = configparser.ConfigParser()
            if os.path.isfile(file):
                config.read(file)
            if not config.has_section('General'):
                config.add_section('General')
                config.set('General', 'device', str(self.device))
            if not config.has_section('Network'):
                config.add_section('Network')
            config.set('Network', 'network_folder', folder)
            config.set('Network', 'name', str(self.name))
            config.set('Network', 'image_width', str(self.imageShape[0]))
            config.set('Network', 'image_height', str(self.imageShape[1]))
            
            config.set('Network', 'convolution_layers', str(self.nConvLayers))
            config.set('Network', 'convolution_filters', str(self.nConvFilters))
            config.set('Network', 'filter_size', str(self.convFilterSize))
            
            config.set('Network', 'regression_layers', str(self.nRegLayers))
            config.set('Network', 'layer_size', str(self.regLayerSize))
            config.set('Network', 'number_classes', str(self.nClasses))
            
            config.set('Network', 'activation', str(self.activation))
            
            with open(file, 'w') as f:
                config.write(f)
            
            return True
        except Exception as e:
            print("Unable to save network configuration data: ",e)
            return False
        
    
    """ Load network meta-data from folder """
    def __load_config(self, folder):
        if not folder.endswith("/"):
            folder += "/"
        file = folder+self.configData
        if os.path.isfile(file):
            config = configparser.ConfigParser(file)
            self.name=config.get('Network', 'name')
            self.imageShape=( config.getint('Network', 'image_width'), config.getint('Network', 'image_height') )
            
            self.nConvLayers=config.getint('Network', 'convolutional_layers')
            self.nConvFilters=config.getint('Network', 'convolutional_filters')
            self.convFilterSize=config.getint('Network', 'filter_size')
            
            self.nRegLayers=config.getint('Network', 'regression_layers')
            self.regLayerSize=config.getint('Network', 'layer_size')
            self.nClasses=config.getint('Network', 'number_classes')            
            self.regLayers = ()
            for i in range(self.nRegLayers):
                self.regLayers += (self.regLayerSize ,)
            self.regLayers += (self.nClasses ,)
            
            self.activation=config.get('Network', 'activation')
            self.sample_size = self.nConvLayers*(self.convFilterSize -1) +1
            
            return True
        else:
            print("Unable to load network configuration data.")
            return False
        
    
    """ Load weights from folder """
    def __load_weights(self, folder):
        if not folder.endswith("/"):
            folder += "/"
        if not os.path.exists(folder+"weights"):
            return False
        try:
            self.cw = () #Convolution weights
            self.cb = () #Convolution biases
            for layer in range(0, self.nConvLayers):
                self.cw += (theano.shared(
                            np.genfromtxt(folder+"weights/conv/layer_"+str(layer)+"_weights.csv", 
                                          delimiter=',')) ,)  
                self.cb += (theano.shared(
                            np.genfromtxt(folder+"weights/conv/layer_"+str(layer)+"_bias.csv", 
                                          delimiter=',')) ,)                 
            self.rw = () #Regression weights
            self.rb = () #Regression biases
            for layer in range(0, len(self.regLayers)):
                self.rw += (theano.shared(
                            np.genfromtxt(folder+"weights/reg/layer_"+str(layer)+"_weights.csv", 
                                          delimiter=',')) ,)  
                self.rb += (theano.shared(
                            np.genfromtxt(folder+"weights/reg/layer_"+str(layer)+"_bias.csv", 
                                          delimiter=',')) ,)                 
        except:
            return False
            
            
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
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    