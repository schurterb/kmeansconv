# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 23:28:09 2016

@author: schurterb

Revised analyzer for taking predictions and analyzing them.

For supervised learning, this calculates the confusion matrix,
 which consists of the following values:
 
True Positive - prediction: X = A, target: X = A
False Positive - prediction: X = A, target: X != A
True Negative - prediction: X != A, target: X != A
False Negative - prediction: X != A, target: X = A

"""

import os
import numpy as np


class Analyzer:
    
    def __init__(self, **kwargs):
        self.prediction = kwargs.get("prediction", None)
        self.target = kwargs.get("target", None)
        #A dictionary containing the confusion matrix
        self.confusion = {}
        #threshold steps
        self.thresh_min = float(kwargs.get("threshold_min", 0))
        self.thresh_max = float(kwargs.get("threshold_max", 1))
        self.thresh_step = float(kwargs.get("threhold_step", 0.1))

    
    def saveCalculations(self, folder, basefilename=""):
        if self.confusion is not None or os.path.exists(folder):
            if not folder.endswith("/"):
                folder += "/"
            for key in self.confusion.keys():
                self.confusion[key].tofile(folder+basefilename+"_"+key+".csv", sep=',')
    
        
    def calculateConfusionMatrices(self):
        #try:
        steps = np.arange(self.thresh_min, self.thresh_max+self.thresh_step, self.thresh_step)
        true_positives = np.zeros(len(steps))
        false_positives = np.zeros(len(steps))
        true_negatives = np.zeros(len(steps))
        false_negatives = np.zeros(len(steps))
        
        for i in range(len(steps)):
            tp, fp, tn, fn = self.__calculate_confusion_matrix(steps[i])
            true_positives[i] = tp
            false_positives[i] = fp
            true_negatives[i] = tn
            false_negatives[i] = fn
        
        self.confusion["thresholds"] = steps
        self.confusion["true_positives"] = true_positives
        self.confusion["false_positives"] = false_positives
        self.confusion["true_negatives"] = true_negatives
        self.confusion["false_negatives"] = false_negatives
        return self.confusion
        #except:
        #    return None
        
    
    def __calculate_confusion_matrix(self, threshold):
        X = np.copy(self.prediction)
        X[self.prediction >= threshold] = 1
        X[self.prediction < threshold] = 0
        tp = self.__true_positives(X, self.target)
        fp = self.__false_positives(X, self.target)
        tn = self.__true_negatives(X, self.target)
        fn = self.__false_negatives(X, self.target)
        return tp, fp, tn, fn


    def __true_positives(self, x, y):
        return ((x == 1) & (y == 1)).sum()   
        
    def __false_positives(self, x, y):
        return ((x == 1) & (y == 0)).sum()
        
    def __true_negatives(self, x, y):
        return ((x == 0) & (y == 0)).sum()
        
    def __false_negatives(self, x, y):
        return ((x == 0) & (y == 1)).sum()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    