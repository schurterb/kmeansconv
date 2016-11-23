# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 00:42:23 2016

@author: schurterb

Tool for presenting results from training, testing, and analysis.
"""

import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.cm as cm
import numpy as np

class Presenter:
    
    def __init__(self, network_folder, **kwargs):
        if not network_folder.endswith("/"):
            network_folder += "/"
        self.network_folder = network_folder
        self.results_folder = self.network_folder+"results/"
        self.lc_file = self.network_folder+"log/learning.csv"
        self.cm_base = kwargs.get('data_timestamp','')
        self.name = self.network_folder.split("/")[-1]+" "+self.cm_base
    
    def __del__(self):
        pass
    
###############################################################################
 
    def plotLearning(self, avg_seg = 1, f=None):
        learning_curve = self.__load_learning_curve()
        if f is None:
            f = plt.figure()
        
        #learning_curve = learning_curve.reshape(learning_curve.size())
        if len(learning_curve) > 0:
            idx = learning_curve.size/avg_seg
            lc = np.mean(learning_curve[0:idx*avg_seg].reshape(idx, avg_seg), 1)
            plt.plot(np.arange(len(lc)), lc, linewidth=1.8, label=self.name)
            
        plt.legend(bbox_to_anchor=(0.8, 0.89 , 1., .102), loc=2, borderaxespad=0., prop={'size':20})
        plt.xlabel("averaged update interval", fontsize=12)
        plt.ylabel("cost", fontsize=12)
        plt.grid()
                

    def __load_learning_curve(self):
        if os.path.isfile(self.lc_file):
            try:
                with open(self.lc_file) as f:
                    data = f.read()
                
                data = data.split('\r\n')
                tmp = data[0].split('\n')
                data = np.append(tmp, data[1::])
                lc = np.zeros(data.size)
                for i in range(data.size):
                    if(data[i] != ''):
                        lc[i] = float(data[i])
                
                return np.asarray(lc)
            except:
                print('Warning: Unable to load learning curve from ',self.lc_file)
        return np.zeros(0)

###############################################################################    
    
    def plotPerformance(self):
        #Get Data
        confusion_matrix = self.__load_confusion_matrix()
        thresh = confusion_matrix["thresholds"]
        accuracy = self.__accuracy(confusion_matrix)
        precision = self.__precision(confusion_matrix)
        recall = self.__recall(confusion_matrix) #also sensitivity and TPR
        fallout = self.__fallout(confusion_matrix) #also FPR
        fscore = self.__fscore(precision, recall)

        #Initialize Figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2) 
        max_threshold = thresh.max()
        min_threshold = thresh.min()
        
        #Plot Fscore
        ax1.plot(thresh, fscore, linewidth=1.8, label=self.name+" (range: "+str(min_threshold)+" - "+str(max_threshold)+")")
        ax1.scatter(thresh, fscore)
        ax1.set_title('F-score', fontsize=12)
        ax1.set_ylabel('f-score', fontsize=12)
        ax1.set_xlabel('threshold', fontsize=12)
        
        #Plot Accuracy
        ax2.plot(thresh, accuracy, linewidth=1.8, label=self.name+" (range: "+str(min_threshold)+" - "+str(max_threshold)+")")
        ax2.scatter(thresh, accuracy)
        ax2.set_title('Accuracy', fontsize=12)
        ax2.set_ylabel('accuracy', fontsize=12)
        ax2.set_xlabel('threshold', fontsize=12)
                
        #Plot ROC
        ax3.plot(fallout, recall, linewidth=1.8, label=self.name+" (range: "+str(min_threshold)+" - "+str(max_threshold)+")")
        ax3.scatter(fallout, recall)
        ax3.set_title('ROC', fontsize=12)
        ax3.set_ylabel('true-positive rate', fontsize=12)
        ax3.set_xlabel('false-positive rate', fontsize=12)
                
        #Plot Precision vs Recall
        ax4.plot(recall, precision, linewidth=1.8, label=self.name+" (range: "+str(min_threshold)+" - "+str(max_threshold)+")")
        ax4.scatter(recall, precision)
        ax4.set_title('Precision vs. Recall', fontsize=12)
        ax4.set_ylabel('precision', fontsize=12)
        ax4.set_xlabel('recall', fontsize=12)
        
        ax1.grid(); ax2.grid(); ax3.grid(); ax4.grid();
        
    def __load_confusion_matrix(self):
        data = {}
        cm_blocks = ["thresholds","true_positives","false_positives","true_negatives","false_negatives"]
        for block in cm_blocks:
            file = self.results_folder+self.cm_base+"_"+block+".csv"
            if os.path.isfile(file):
                data[block] = np.genfromtxt(file, delimiter=',')
        return data

    # (TP + TN) / (TP + TN + FP + FN)
    def __accuracy(self, data):
        tp = data["true_positives"]
        fp = data["false_positives"]
        tn = data["true_negatives"]
        fn = data["false_negatives"]
        num = tp + tn
        dem = tp + tn + fp + fn
        with np.errstate(divide='ignore'):
            res = num/dem
        res[np.isnan(res)] = 0
        return res
    
    # TP / (TP + FP)
    def __precision(self, data):
        tp = data["true_positives"]
        fp = data["false_positives"]
        with np.errstate(divide='ignore'):
            res = tp/(tp+fp)
        res[np.isnan(res)] = 0
        return res
    
    # TP / (TP + FN)
    def __recall(self, data):
        tp = data["true_positives"]
        fn = data["false_negatives"]
        with np.errstate(divide='ignore'):
            res = tp/(tp+fn)
        res[np.isnan(res)] = 0
        return res
    
    # FP / (TP + TN)
    def __fallout(self, data):
        tp = data["true_positives"]
        fp = data["false_positives"]
        tn = data["true_negatives"]
        with np.errstate(divide='ignore'):
            res = fp/(tp+tn)
        res[np.isnan(res)] = 0
        return res
    
    # (2 * PRECISION * RECALL) / (PRECISION + RECALL)
    def __fscore(self, precision, recall):
        with np.errstate(divide='ignore'):
            res = (2*precision*recall) / (precision+recall)
        res[np.isnan(res)] = 0
        return res

###############################################################################
    
    def showPlots(self):
        plt.show()
    
    def plotAll(self):
        self.plotLearning()
        self.plotPerformance()
        self.showPlots()
    
    








