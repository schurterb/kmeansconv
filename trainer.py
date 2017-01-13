# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 06:10:40 2015

@author: schurterb

Trainer class for a cnnclassifier
Capable of using standard SGD, RMSprop, or ADAM for back propogation
 and binary-crossentropy or MSE as the cost function.
"""

import os
import time
import logging
import csv
from sh import mkdir
import configparser

import theano
import theano.sandbox.cuda
from theano import tensor as T
import numpy as np

from backprop import StandardSGD, RMSProp, ADAM
from dataloader import MNISTDataLoader

theano.config.floatX = 'float32'


class Trainer(object):        
        
    """Define the cost function used to evaluate this network"""
    def __set_cost(self):
        if(self.supervised_training):
            self.targets = self.Y
        else:
            #TODO: Add target definition
            maxOut = T.max(self.out, 1)
            self.targets = T.switch(self.out < maxOut, 0, 1)
        if(self.cost_func == 'class'):
            self.cost = T.mean(T.nnet.binary_crossentropy(self.out, self.targets), dtype=theano.config.floatX)
        else:
            self.cost = T.mean(1/2.0*((self.out - self.targets)**2), dtype=theano.config.floatX)
            

    
    """Define the updates to be performed each training round"""
    def __set_updates(self):  

        if(self.learning_method == 'RMSProp'):
            self.update_method = RMSProp(self.network, 
                                         learning_rate = self.lr,
                                         beta_decay = self.b2,
                                         damping = self.damp)
        elif(self.learning_method == 'ADAM'):
            self.update_method = ADAM(self.network, 
                                      learning_rate = self.lr,
                                      beta1_decay = self.b1,
                                      beta2_decay = self.b2,
                                      damping = self.damp)
            
        else: #The default is standard SGD   
            self.update_method = StandardSGD(self.network, learning_rate = self.lr)
            
        #Calculate gradients for convolution layers
        cw_grad = T.grad(self.cost, self.network.cw)
        cb_grad = T.grad(self.cost, self.network.cb)    
        self.update_method.addConvolutionGradients(cw_grad, cb_grad)
        
        #Calculate gradients for regression layers
        rw_grad = T.grad(self.cost, self.network.rw)
        rb_grad = T.grad(self.cost, self.network.rb)    
        self.update_method.addRegressionGradients(rw_grad, rb_grad)
        
        self.updates = self.update_method.getUpdates()


    """Define targets for unsupervised learning"""
    def __set_targets(self):
        if not self.supervised_training:
            pass
            
    
    """
    Define the function(s) for GPU training
    """
    def __generate_training_model(self):
        
        self.updates = []
        self.__set_cost()    
        self.__set_updates()
        if(self.supervised_training):
            self.train_network = theano.function(inputs=[self.X, self.Y], outputs=self.cost,
                                                 updates = self.updates,
                                                 allow_input_downcast=True)
        else:
            self.train_network = theano.function(inputs=[self.X], outputs=self.cost,
                                                 updates = self.updates,
                                                 allow_input_downcast=True)
            
                                                                                         
        
    """
    Network must be a list constaining the key components of the network
     to be trained, namely its symbolic theano reprentation (first parameter),
     its cost function (second parameter), its shared weight and bias 
     variables (second and third parameters, rspectively)
    """
    def __init__(self, network, **kwargs):
        self.rng = kwargs.get('rng', np.random.RandomState(42))
        #######################################################################
        ## Network parameters ##
        self.X = network.X
        self.out = network.out
        self.network = network
        
        ####################################################################### 
        ####################################################################### 
        trainer_folder = kwargs.get('trainer_folder', None)
        if trainer_folder is None:
            #######################################################################
            ## Logger Header ##        
            trainer_status = "\n### Convolutional Network Trainer Log ###\n\n"
            trainer_status += "Network Parameters\n"
            trainer_status += "num convolution layers = "+ str(network.nConvLayers) +"\n"
            trainer_status += "num convolution filters = "+ str(network.nConvFilters) +"\n"
            trainer_status += "convolution filter size = "+ str(network.convFilterSize) +"\n" 
            trainer_status += "regression layers = "+ str(network.regLayers) +"\n"       
            trainer_status += "num classes = "+ str(network.nClasses) +"\n"       
            trainer_status += "activation = "+ str(network.activation) +"\n\n"
                            
            #######################################################################
            ## Training parameters ##
            trainer_status += "Trainer Parameters\n"
            self.cost_func = kwargs.get('cost_function', 'MSE')
            trainer_status += "cost function = "+ str(self.cost_func) +"\n"
            
            if(kwargs.get('training_type', 'unsupervised') == 'unsupervised'):
                self.supervised_training = False
            else:
                self.supervised_training = True
            trainer_status += "training type = "+ \
                            str(kwargs.get('training_type', 'unsupervised')) +"\n"
            
            self.learning_method = kwargs.get('learning_method', 'standardSGD')
            trainer_status += "learning method = "+self.learning_method+"\n"
            self.batch_size = kwargs.get('batch_size', 100)
            trainer_status += "batch size = "+str(self.batch_size)+"\n"
            
            self.lr = kwargs.get('learning_rate', 0.0001)
            trainer_status += "learning rate = "+str(self.lr)+"\n"
            
            self.b1 = kwargs.get('beta1', 0.9)
            if(self.learning_method=='ADAM'): trainer_status += "beta 1 = "+str(self.b1)+"\n"
            
            self.b2 = kwargs.get('beta2', 0.9)
            self.damp = kwargs.get('damping', 1.0e-08)
            if(self.learning_method=='RMSProp') or (self.learning_method=='ADAM'): 
                trainer_status += "beta 2 = "+str(self.b2)+"\n"
                trainer_status += "damping term = "+str(self.damp)+"\n"
            self.epoch_length = kwargs.get('epoch_length', 1000)
            
            self.data_folder = kwargs.get('data_folder', 'data')        
            #######################################################################
            ## Logger Setup ##
            self.log_interval = kwargs.get('log_interval', self.batch_size)
            self.folder = kwargs.get('trainer_folder', network.network_folder)
            if not self.folder.endswith('/'):
                self.folder += '/'        
            self.log_folder = self.folder + 'log/'
            if not os.path.exists(self.log_folder):
                mkdir("-p", self.log_folder)
        else:
            if self.__load_config(trainer_folder):
                self.folder=trainer_folder
                self.log_folder = self.folder + 'log/'
            else:
                print("Unable to initialize or load trainer.")
                return None     
        #######################################################################
        #######################################################################          
                 
        self.log_file = self.log_folder + 'trainer.log'
        self.__clear_log()
        self.__init_lc()
        logging.basicConfig(filename=self.log_file, level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(trainer_status+"\n")    
            
        #######################################################################
        self.configData = "network.cfg"
        self.__save_config(self.folder)            
            
        #######################################################################
        ## Initialize Trainer ## 
        self.Y = T.matrix('Y')
        
        self.dataLoader = self.__load_training_data(self.data_folder)
        
        self.__generate_training_model()
                                            

    """
    Load all the training samples needed for training this network. Ensure that
    there are about equal numbers of positive and negative examples.
    """
    def __load_training_data(self, data_folder):
        return MNISTDataLoader(data_folder=data_folder)


    """
    Retrieve and format the training batch
    """
    def __get_batch(self):
        data, labels = self.dataLoader.getTrainingBatch()
        formatted_labels = np.zeros((labels.shape[0], self.network.nClasses), dtype=np.uint32)
        for x in range(labels.shape[0]):
            formatted_labels[x, labels[x]] = 1
        return (data, formatted_labels)


    """
    Store the trainining and weight values at regular intervals
    """
    def __store_status(self, error):
          
        error_file = 'learning.csv'
        trainer_folder = self.folder + 'trainer/'+self.cost_func+'/'
            
        with open(self.log_folder + error_file, 'a') as ef:
            fw = csv.writer(ef, delimiter=',')
            fw.writerow([error])
                
        self.update_method.save(trainer_folder)
        self.network.save(self.network.network_folder)
    

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
                config.set('General', 'device', 'cpu')                
            if not config.has_section('TrainerI'):
                config.add_section('Trainer')
            config.set('Trainer','cost_function',str(self.cost_func))
            config.set('Trainer', 'learning_method',str(self.learning_method))
            config.set('Trainer', 'batch_size',str(self.batch_size))
            config.set('Trainer', 'learning_rate',str(self.lr))
            config.set('Trainer', 'beta1',str(self.b1))
            config.set('Trainer', 'beta2',str(self.b2))
            config.set('Trainer', 'damping',str(self.damp))
            config.set('Trainer', 'epoch_length',str(self.epoch_length))
            config.set('Trainer', 'log_interval',str(self.log_interval))
            config.set('General', 'data_folder',str(self.data_folder))
            
            with open(file, 'w') as f:
                config.write(f)
                
            return True
        except Exception as e:
            print("Unable to save trainer configuration data: ",e)
            return False


    """ Load network meta-data from folder """
    def __load_config(self, folder):
        if not folder.endswith("/"):
            folder += "/"
        file = folder+self.configData
        if not os.path.isfile(file):
            config = configparser.ConfigParser(file)
            
            self.cost_func=config.get('Trainer', 'cost_function')
            self.learning_method=config.get('Trainer', 'learning_method')
            self.batch_size=config.getint('Trainer', 'batch_size')
            self.lr=config.getfloat('Trainer', 'learning_rate')
            self.b1=config.getfloat('Trainer', 'beta1')
            self.b2=config.getfloat('Trainer', 'beta2')
            self.damp=config.getfloat('Trainer', 'damping')
            self.epoch_length=config.getint('Trainer', 'epoch_length')
            self.log_interval=config.getint('Trainer', 'log_interval')                           
            self.data_folder=config.get('General', 'data_folder')
            
            return True
        else:
            print("Unable to load trainer configuration data.")
            return False    
    
    """Clear the logging file"""
    def __clear_log(self):
        with open(self.log_file, 'w'):
            pass
    
    
    """Make sure there is a file for the learning curve"""
    def __init_lc(self):
        if not os.path.isfile(self.log_folder + 'learning.csv'):
            open(self.log_folder + 'learning.csv', 'w').close()

    
    """
    Log the current status of the trainer and the network
    """
    def __log_trainer(self, epoch, error, train_time):
        start_cost = error[0]
        end_cost = error[-1]
        trainer_status  = "\n-- Status at epoch: "+str(epoch)+" --\n"
        trainer_status += "Change in average cost: "+str(start_cost)+" -> "+str(end_cost)+"\n"
        diff = start_cost - end_cost
        pcent = (diff/start_cost)*100
        trainer_status += "     Improvement of "+str(diff)+" or "+str(pcent)+"%\n"
        trainer_status += "Number of examples seen: "+str(self.batch_size*self.log_interval*self.epoch_length)+"\n"
        trainer_status += "Training time: "+str(train_time/60)+" minutes\n\n"
        
        self.logger.info(trainer_status)
    
    
    """   
    Train the network on a specified training set with a specified target
     (supervised training). This training uses stochastic gradient descent
     mini-batch sizes set at initialization. Training samples are selected 
     such that there is an equal number of positive and negative samples 
     each batch.        
    Returns: network cost at each update
    """
    def train(self, duration, early_stop = False, print_updates = True):
        
        train_error = np.zeros(duration * self.epoch_length)     
        epoch = 0
                
        while(epoch < duration):
            if(print_updates): print('Epoch:',epoch)
            epochstart = time.clock()
            for i in range(self.epoch_length//self.log_interval):
                                
                for j in range(self.log_interval):
                    batch = self.__get_batch()
                    if(self.supervised_training):
                        error = self.train_network(batch[0], batch[1])
                    else:
                        error = self.train_network(batch[0])
                        
                
                train_error[i] = np.mean(error)
                
                if print_updates:
                    print(self.cost_func,'error for updates',i*self.log_interval,' - ',(i+1)*self.log_interval,'(',self.batch_size*self.log_interval,' examples):',train_error[i])
                self.__store_status(train_error[i])
            epoch_time = time.clock() - epochstart
            
            self.__log_trainer(epoch, train_error, epoch_time)
            
            if early_stop and (train_error[train_error > 0][-1] < 0.002):
                  epoch = duration
            
            epoch += 1
            
        return train_error
            
