# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 22:11:53 2017

@author: user

Script for creating multiple config files for 
sweeping a hyper-parameter space.
"""


import configparser
from sh import mkdir

#Set network hyperparameter ranges
toggle_network_sweep = False
nclass_range = [10]
conv_nlayer_range = [3, 4, 5]
conv_filter_range = [20, 50, 100]
filter_size_range = [3, 5]
reg_nlayer_range = [1, 2, 3]
reg_size_range = [50, 100, 200]
activation = ["tanh"]

#Set training hyperparameter ranges
toggle_trainer_sweep = True
training_type = ["supervised"]
cost_function = ["class", "MSE"]
learning_method = ["StandardSGD", "RMSProp", "ADAM"]
batch_size = [100]
learning_rate = [0.0001, 0.00001, 0.000001]
beta1 = [0.9, 0.99, 0.999]
beta2 = [0.9, 0.99, 0.999]
damping = [1.0e-8]

#Set filenaming scheme
baseFolder = "networks/sweep-"
baseFilename = "network"
baseFileExtension = ".cfg"
baseConfigFile = baseFilename+baseFileExtension

#Open the default config file
baseConfig = configparser.ConfigParser()
baseConfig.read(baseConfigFile)

networks = []
counter = 0
if(toggle_trainer_sweep):
    for t_type in training_type:
        baseConfig.set('Trainer', 'training_type', str(t_type))
        for cost_f in cost_function:
            baseConfig.set('Trainer', 'cost_function', str(cost_f))
            for method in learning_method:
                baseConfig.set('Trainer', 'learning_method', str(method))
                for b_size in batch_size:
                    baseConfig.set('Trainer', 'batch_size', str(b_size))
                    for lr in learning_rate:
                        baseConfig.set('Trainer', 'learning_rate', str(lr))
                        for b1 in beta1:
                            baseConfig.set('Trainer', 'beta1', str(b1))
                            for b2 in beta2:
                                baseConfig.set('Trainer', 'beta2', str(b2))
                                for dp in damping:
                                    baseConfig.set('Trainer', 'damping', str(dp))
                                    #Set filepaths
                                    folder = baseFolder+str(counter)+"/"
                                    baseConfig.set('Network', 'network_folder', folder)
                                    baseConfig.set('Network', 'weights_folder', folder+"weights")
                                    baseConfig.set('Trainer', 'trainer_folder', folder)
                                    baseConfig.set('Trainer', 'log_folder', folder+"log")
                                    baseConfig.set('Testing', 'prediction_folder', folder)
                                    baseConfig.set('Testing', 'results_folder', folder+"results")
                                    mkdir('-p', folder)
                                    #Write config file
                                    networks += [folder+baseFilename+baseFileExtension]
                                    with open(networks[counter], 'w') as f:
                                        baseConfig.write(f)
                                    #Update counter
                                    counter += 1





