#!/home/user/anaconda3/bin/python
"""
Created on Wed Oct  5 21:44:08 2016

@author: user

Train script for building (or loading)
 and training a kmeansconv network.
"""

import os
import time
import configparser
import argparse

import theano

from cnnclassifier import CNN2D
from trainer import Trainer


def trainNetwork(configuration):
    
    #Open config file
    config = configparser.ConfigParser()
    config.read(configuration)

    #Set the device on which to train the network
    device = config.get('General', 'device')
    theano.sandbox.cuda.use(device)
    if(device != 'cpu'):
        theano.config.nvcc.flags='-use=fast=math'
        theano.config.allow_gc=False

    #Create the network        
    starttime=time.clock()
    network = None
    trainer = None
    networkLoaded = False
    if os.path.exists(config.get('Network', 'weights_folder')):
        try:
            print("Loading Network")
            network = CNN2D(network_folder = config.get('Network', 'network_folder'))
            networkLoaded = True
        except Exception as e:
            print("ERROR: Unable to load network: ",e.with_traceback)
            network = None
    
    if network is None:
        try:
            print("Designing Network")
            network = CNN2D(name=config.get('Network', 'name'),
                            image_size=( config.getint('Network', 'image_width'), config.getint('Network', 'image_height') ),
                            convolutional_layers=config.getint('Network', 'convolutional_layers'),
                            convolutional_filters=config.getint('Network', 'convolutional_filters'),
                            filter_size=config.getint('Network', 'filter_size'),
                            regression_layers=config.getint('Network', 'regression_layers'),
                            layer_size=config.getint('Network', 'layer_size'),
                            number_classes=config.getint('Network', 'number_classes'),
                            activation=config.get('Network', 'activation'),
                            device=config.get('General', 'device'))
        except Exception as e:
            print("FATAL: Unable to create network: ",e.with_traceback)
            network = None
            return
    
    if networkLoaded is True and os.path.exists(config.get('Trainer', 'log_folder')):
        try:
            print("Loading Trainer")
            trainer = Trainer(network, trainer_folder=config.get('Trainer', 'trainer_folder'))        
        except Exception as e:
            print("FATAL: Unable to load trainer: ",e.with_traceback)
            return
    else:
#        try:
        print("Creating Trainer")
        trainer = Trainer(network,
                          training_type=config.get('Trainer', 'training_type'),
                          cost_function=config.get('Trainer', 'cost_function'),
                          learning_method=config.get('Trainer', 'learning_method'),
                          batch_size=config.getint('Trainer', 'batch_size'),
                          learning_rate=config.getfloat('Trainer', 'learning_rate'),
                          beta1=config.getfloat('Trainer', 'beta1'),
                          beta2=config.getfloat('Trainer', 'beta2'),
                          damping=config.getfloat('Trainer', 'damping'),
                          epoch_length=config.getint('Trainer', 'epoch_length'),
                          log_interval=config.getint('Trainer', 'log_interval'),                              
                          data_folder=config.get('General', 'data_folder'))
#        except Exception as e:
#            print("FATAL: Unable to create trainer: ",e.with_traceback)
#            print(e.__traceback__)
#            return


    #Timing event
    init_time = time.clock() - starttime
    print("Initialization = ",init_time," seconds")
    
    starttime = time.clock()    
    #Train the network
    print("Training Network: ",network.name)
    train_error = trainer.train(config.getint('Trainer', 'number_epochs'),
                                config.getboolean('Trainer', 'early_stop'),
                                config.getboolean('Trainer', 'verbosity'))
                                
    #Timing event
    train_time = time.clock() - starttime
    print("Training = ",train_time," seconds")
    
    return train_error
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="path to network configuration file")
    parser.add_argument("-f", help="path to existing network folder")
    
    args = parser.parse_args()
    config_file = "network.cfg"
    if args.c:
        config_file = args.c
    elif args.f:
        if not args.f.endswith("/"):
            args.f += "/"
        config_file = args.f + config_file
        
    trainNetwork(config_file)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
             
        