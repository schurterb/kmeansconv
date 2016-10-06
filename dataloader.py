# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 16:03:06 2016

@author: user

An on-demand data-loader for MNIST data.
"""

import numpy as np
from mnist import MNIST
import random

class MNISTDataLoader:
    
    def __init__(self, **kwargs):
        self.batchSize = kwargs.get('batch_size', 10)
        self._load_data(kwargs.get('data_folder', './'))
    
    def _load_data(self, folder):
        data = MNIST(folder)
        self.trainingData, self.trainingLabels = data.load_training()
        self.trainingData = np.asarray(self.trainingData, dtype = np.uint32)
        self.trainingLabels = np.asarray(self.trainingLabels, dtype = np.uint32)
        self.testingData, self.testingLabels = data.load_training()
        self.testingData = np.asarray(self.testingData, dtype = np.uint32)
        self.testingLabels = np.asarray(self.testingLabels, dtype = np.uint32)
        
    def _get_random_indeces(self, max_index, count):
        return np.random.choice(max_index, count, replace=False)
        
    def getTrainingBatch(self):
        subset = self._get_random_indeces(len(self.trainingLabels), self.batchSize)
        data = self.trainingData[subset]
        labels = self.trainingLabels[subset]
        data = data.reshape(data.shape[0], 28, 28)
        return (data, labels)
        
    def getTestingData(self):
        subset = self._get_random_indeces(len(self.testingLabels), len(self.testingLabels))    
        subset = self._get_random_indeces(len(self.trainingLabels), self.batchSize)
        data = self.testingData[subset]
        labels = self.testingLabels[subset]    
        data = data.reshape(data.shape[0], 28, 28)
        return (data, labels)
        
        
        
        
        
        