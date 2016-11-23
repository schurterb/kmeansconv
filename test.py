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
import numpy as np
from sh import mkdir
import datetime

from cnnclassifier import CNN2D
from dataloader import MNISTDataLoader
from analysis import Analyzer

def testNetwork(configuration):
    
    print("Testing network at ",configuration)
    #Current date
    now = datetime.datetime.now()
    date = str(now.year)+"."+str(now.month)+"."+str(now.day)    
    
    #Open config file
    config = configparser.ConfigParser()
    config.read(configuration)

    #Set the device on which to train the network
    device = config.get('General', 'device')
    #theano.sandbox.cuda.use(device)
    if(device != 'cpu'):
        theano.config.nvcc.flags='-use=fast=math'
        theano.config.allow_gc=False

    #Load data for testing
    if os.path.exists(config.get('General', 'data_folder')):
        mnist = MNISTDataLoader(data_folder=config.get('General', 'data_folder'))
        data = mnist.getTestingData()
    else:
        print("FATAL: Unable to load data.")
        return

    #Create the network        
    starttime=time.clock()
    network = None
    print("Loading Network")
    if os.path.exists(config.get('Network', 'weights_folder')):
        try:
            network = CNN2D(network_folder = config.get('Network', 'network_folder'))
        except Exception as e:
            print("FATAL: Unable to load network: ",e.with_traceback)
            return
    else:
        print("FATAL: Network folder does not exist!")
        return

    #Timing event
    init_time = time.clock() - starttime
    print("Initialization = ",init_time," seconds")
    
    starttime = time.clock()    
    #Train the network
    print("Testing Network:  ",network.name)
    prediction = network.run(data[0])
    
    #Timing event
    test_time = time.clock() - starttime
    print("Testing = ",test_time," seconds")
        
    #Save results
    prediction = np.asarray(prediction)
    folder = config.get('Network', 'network_folder')
    if folder.endswith("/"):
        folder += "/"
    folder += "results/"
    mkdir('-p', folder)
    prediction.tofile(folder+"prediction_"+date+".csv", sep=',')
    
    starttime = time.clock()    
    #Analyse results
    analyzer = Analyzer(prediction=prediction,
                        target=data[1],
                        threshold_min=0,
                        threshold_max=1,
                        threshold_step=0.01)
    analyzer.calculateConfusionMatrices()
    analyzer.saveCalculations(folder, date)
    
    #Timing event
    analysis_time = time.clock() - starttime
    print("Analysis = ",analysis_time," seconds")
    
    return prediction
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="path to existing network folder")
    
    args = parser.parse_args()
    config_file = "network.cfg"
    if args.f:
        if not args.f.endswith("/"):
            args.f += "/"
        config_file = args.f + config_file
        
        testNetwork(config_file)
    else:
        print("Trained network required for testing.")
        print("-f","   ","path to existing network folder")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
             
        