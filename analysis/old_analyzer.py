# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:39:19 2015

@author: schurterb

Analyzer class to perform a basic analysis of the results from training and 
 testing a convolutional network
"""

from scipy.io import loadmat
import ConfigParser
import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.cm as cm
import h5py

class Analyzer(object):

    """Load the specified learning curve"""
    def __load_learning(self, results_folder, lc_type = 'MSE', lc_name=None):
        try:
            if lc_name:
                f = open(results_folder + lc_name + '.csv', 'r')
            else:
                f = open(results_folder + lc_type + '_learning.csv', 'r')
            lc = f.read()
            f.close()
            
            lc = lc.split('\r\n')
            tmp = lc[0].split(',')
            lc = np.append(tmp, lc[1::])
            lr_curve = np.zeros(lc.size)
            for i in range(lc.size):
                if(lc[i] != ''):
                    lr_curve[i] = float(lc[i])
            
            return np.asarray(lr_curve)
        except:
            print('Warning: Unable to load learning curve for',self.name[self.results_folder.index(results_folder)])
            return None


    """Load the affinity predictions from a folder"""
    def __load_prediction(self, results_folder, pred_name='prediction'):
        try:
            self.pred_file += (h5py.File(results_folder + 'results/' + pred_name + '.h5', 'r') ,)
            pred = self.pred_file[-1]['main'][...].astype('d', order='F')
            self.prediction += (np.transpose(pred).astype(dtype='d', order='F') ,)
        except:
            self.prediction += (None ,)
            print('Warning: Unable to load test prediction.')
            
    
    def __init__(self, **kwargs):
        
        self.nsteps = kwargs.get('threshold_steps', 10) 
        self.crop = kwargs.get('image_crop', 0)
        
        self.tp = []
        self.fp = []
        self.npos = []
        self.nneg = []
        self.fscore = []
        self.randIndex = []
        self.theta = []
        
        self.tpr = []
        self.fpr = []
        
        folder = kwargs.get('results_folder', None)
        self.results_folder = ()
        self.pred_file = ()
        self.prediction = ()
                
        self.target = kwargs.get('target', None)
        targ = self.target[...].astype(dtype='d', order='F')
        self.target = np.transpose(targ).astype(dtype='d', order='F')
        self.raw = kwargs.get('raw', None)
        if self.raw: self.raw = np.transpose(self.raw)
        
        if (folder != None) and (self.target != None):
            self.results_folder += (folder ,)
            self.name = (kwargs.get('name', '') ,)
            self.__load_prediction(self.results_folder[-1])
            self.__threshold_scan(self.results_Folder[-1])
        else:
            self.name = ()            
        
        
    """Add more than one set of results to analyze at a time""" 
    def add_results(self, **kwargs):
                
        self.results_folder += (kwargs.get('results_folder', None) ,)
        prediction_name = kwargs.get('prediction_file', None)
        if not prediction_name:
            self.__load_prediction(self.results_folder[-1])
        else:
            self.__load_prediction(self.results_folder[-1], prediction_name)
        
        self.name += (kwargs.get('name', '') ,)
        
    
    """Analyzes predictions in results folder, if not already done"""
    def analyze(self, result_name, **kwargs):
        
        if (type(result_name) == str):
            res = self.name.index(result_name)
        elif (type(result_name) == int) and (result_name < len(self.prediction)):
            res = result_name
        print(self.results_folder[res])
        if not os.path.isfile(self.results_folder[res]+'errors_new.mat'):
            config_file = self.results_folder[res]+'/network.cfg'
            config = ConfigParser.ConfigParser()
            config.read(config_file)
            
            label_file = kwargs.get('label_file', None)
            pred_file = kwargs.get('pred_file', None)
            out_path = kwargs.get('out_path', None)            
            
            if not label_file:
                test_data_folder = config.get('Testing Data', 'folders').split(',')
                label_file = test_data_folder[0] + config.get('Testing Data', 'label_file')
        
            if not pred_file and not out_path:
                out_path = config.get('General', 'directory')
                pred_file = out_path + config.get('Testing', 'prediction_folder') + config.get('Testing', 'prediction_file')+'_0.h5'
            elif not pred_file: 
                pred_file = config.get('General', 'directory') + config.get('Testing', 'prediction_folder') + config.get('Testing', 'prediction_file')+'_0.h5'
            print("Files:\n",pred_file,'\n',label_file,'\n',self.results_folder[res])
            name = "evaluating prediction by "+self.results_folder[res].split('/')[-2]
            subprocess.call(["matlab -nosplash -nodisplay -r \"evaluate_predictions(\'"+label_file+"\',\'"+pred_file+"\',\'"+self.results_folder[res]+"\',\'"+name+"\'); exit\""], shell=True);
        else:
            print('Analysis of prediction already exists. No analysis performed.\n')
            

    """Plots the learning curves"""
    def learning(self, lc_type = 'MSE', averaging_segment=1, networks=None):
        
        if networks:
            learning_curve = ()
            curve_names = ()
            for net, c in networks:
                i = self.name.index(net)
                lc = self.__load_learning(self.results_folder[i], lc_type)
                if not (lc == None): 
                    learning_curve += (lc ,)
                    curve_names += ((net, c) ,)
                
            num_curves = len(learning_curve)
            n_figs = num_curves/7   #There can be seven unique auto-generated line colors
            rem_curves = num_curves%7
            
            for f in range(0, n_figs):
                plt.figure()
                for i in range(0, 7):
                    if learning_curve[f*7 + i].any():
                        idx = learning_curve[f*7 + i].size/averaging_segment
                        c = curve_names[f*7 + i][1]
                        lc = np.mean(learning_curve[f*7 + i][0:idx*averaging_segment].reshape(idx, averaging_segment), 1)
                        plt.plot(np.arange(len(lc)), lc, c, linewidth=1.8, label=curve_names[f*7 + i][0])
                plt.legend(bbox_to_anchor=(0.8, 0.89 , 1., .102), loc=2, borderaxespad=0., prop={'size':20})
                plt.grid()
            if(rem_curves > 0):
                plt.figure()
                for i in range(0, len(learning_curve[-rem_curves::])):
                    if learning_curve[n_figs*7 + i].any():
                        idx = learning_curve[n_figs*7 + i].size/averaging_segment
                        c = curve_names[f*7 + i][1]
                        lc = np.mean(learning_curve[n_figs*7 + i][0:idx*averaging_segment].reshape(idx, averaging_segment), 1)
                        plt.plot(np.arange(len(lc)), lc, c, linewidth=1.8, label=curve_names[n_figs*7 + i][0])
                plt.legend(bbox_to_anchor=(0.8, 0.89 , 1., .102), loc=2, borderaxespad=0., prop={'size':20})
                plt.grid()
        else:
            learning_curve = ()
            curve_names = ()
            for i in range(len(self.results_folder)):
                lc = self.__load_learning(self.results_folder[i], lc_type)
                if not (lc == None): 
                    learning_curve += (lc ,)
                    curve_names += (self.name[i] ,)
                
            num_curves = len(learning_curve)
            n_figs = num_curves/7   #There can be seven unique auto-generated line colors
            rem_curves = num_curves%7
            
            for f in range(0, n_figs):
                plt.figure()
                for i in range(0, 7):
                    if learning_curve[f*7 + i].any():
                        idx = learning_curve[f*7 + i].size/averaging_segment
                        lc = np.mean(learning_curve[f*7 + i][0:idx*averaging_segment].reshape(idx, averaging_segment), 1)
                        plt.plot(lc, linewidth=1.8, label=curve_names[f*7 + i])
                plt.legend(bbox_to_anchor=(0.8, 0.89 , 1., .102), loc=2, borderaxespad=0., prop={'size':20})
                plt.grid()
            if(rem_curves > 0):
                plt.figure()
                for i in range(0, len(learning_curve[-rem_curves::])):
                    if learning_curve[n_figs*7 + i].any():
                        idx = learning_curve[n_figs*7 + i].size/averaging_segment
                        lc = np.mean(learning_curve[n_figs*7 + i][0:idx*averaging_segment].reshape(idx, averaging_segment), 1)
                        plt.plot(lc, linewidth=1.8, label=curve_names[n_figs*7 + i])
                plt.legend(bbox_to_anchor=(0.8, 0.89 , 1., .102), loc=2, borderaxespad=0., prop={'size':20})
                plt.grid()
        plt.xlabel("averaged update interval")
        plt.ylabel(lc_type+" cost")
        plt.show()

    
    
    """Plots the data from threshold_scan"""
    def performance(self, networks=None):
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)       
        #fig.set_facecolor('white')
        
        if networks:
            for net, c in networks:
                i = self.name.index(net)
                if os.path.isfile(self.results_folder[i] + 'errors_new.mat'):
                    mData = loadmat(self.results_folder[i] + 'errors_new.mat')
                    min_affinity = self.prediction[i].min()
                    max_affinity = self.prediction[i].max()
                
                    #Show rand Index results
                    rTheta = mData.get('r_thresholds')[0]
                    rErr = mData.get('r_fscore')[0]
                    rFPR = mData.get('r_fp')[0]/mData.get('r_neg')[0]
                    rTPR = mData.get('r_tp')[0]/mData.get('r_pos')[0]
                
                    ax1.plot(rTheta,rErr, c, linewidth=2, label=self.name[i]+" (range: "+"{0:.5f}".format(min_affinity)+" - "+"{0:.5f}".format(max_affinity)+")")
                    ax1.scatter(rTheta,rErr)
                    ax1.set_title('Rand F-Score', fontsize=20)
                    ax1.set_ylabel('f-score', fontsize=20)
                    ax1.set_xlabel('threshold', fontsize=20)
                                    
                    ax2.plot(rFPR, rTPR, c, linewidth=2, label=self.name[i]+" (range: "+"{0:.5f}".format(min_affinity)+" - "+"{0:.5f}".format(max_affinity)+")")
                    ax2.scatter(rFPR, rTPR)
                    ax2.set_title('Rand ROC', fontsize=20)
                    ax2.set_ylabel('true-positive rate', fontsize=20)
                    ax2.set_xlabel('false-positive rate', fontsize=20)
                    
                    #Show pixel error results
                    pTheta = mData.get('p_thresholds')[0]
                    pErr = mData.get('p_err')[0]
                    pFPR = mData.get('p_fp')[0]/mData.get('p_neg')[0]
                    pTPR = mData.get('p_tp')[0]/mData.get('p_pos')[0]
                
                    ax3.plot(pTheta,pErr, c, linewidth=2, label=self.name[i]+" (range: "+"{0:.5f}".format(min_affinity)+" - "+"{0:.5f}".format(max_affinity)+")")
                    ax3.scatter(pTheta,pErr)
                    ax3.set_title('Pixel Error', fontsize=20)
                    ax3.set_ylabel('pixel error', fontsize=20)
                    ax3.set_xlabel('threshold', fontsize=20)
                
                    ax4.plot(pFPR, pTPR, c, linewidth=2, label=self.name[i]+" (range: "+"{0:.5f}".format(min_affinity)+" - "+"{0:.5f}".format(max_affinity)+")")
                    ax4.scatter(pFPR, pTPR)
                    ax4.set_title('Pixel ROC', fontsize=20)
                    ax4.set_ylabel('true-positive rate', fontsize=20)
                    ax4.set_xlabel('false-positive rate', fontsize=20)
                else:
                    print("Test results not avialable for",net)
        else:
            for i in range(len(self.results_folder)):
                if os.path.isfile(self.results_folder[i] + 'errors_new.mat'):
                    mData = loadmat(self.results_folder[i] + 'errors_new.mat')
                    min_affinity = self.prediction[i].min()
                    max_affinity = self.prediction[i].max()
                
                    #Show rand Index results
                    rTheta = mData.get('r_thresholds')[0]
                    rErr = mData.get('r_fscore')[0]
                    rFPR = mData.get('r_fp')[0]/mData.get('r_neg')[0]
                    rTPR = mData.get('r_tp')[0]/mData.get('r_pos')[0]
                
                    ax1.plot(rTheta,rErr, linewidth=2, label=self.name[i]+" (range: "+"{0:.5f}".format(min_affinity)+" - "+"{0:.5f}".format(max_affinity)+")")
                    ax1.scatter(rTheta,rErr)
                    ax1.set_title('Rand F-Score', fontsize=20)
                    ax1.set_ylabel('f-score', fontsize=20)
                    ax1.set_xlabel('threshold', fontsize=20)
                                    
                    ax2.plot(rFPR, rTPR, linewidth=2, label=self.name[i]+" (range: "+"{0:.5f}".format(min_affinity)+" - "+"{0:.5f}".format(max_affinity)+")")
                    ax2.scatter(rFPR, rTPR)
                    ax2.set_title('Rand ROC', fontsize=20)
                    ax2.set_ylabel('true-positive rate', fontsize=20)
                    ax2.set_xlabel('false-positive rate', fontsize=20)
                    
                    #Show pixel error results
                    pTheta = mData.get('p_thresholds')[0]
                    pErr = mData.get('p_err')[0]
                    pFPR = mData.get('p_fp')[0]/mData.get('p_neg')[0]
                    pTPR = mData.get('p_tp')[0]/mData.get('p_pos')[0]
                
                    ax3.plot(pTheta,pErr, linewidth=2, label=self.name[i]+" (range: "+"{0:.5f}".format(min_affinity)+" - "+"{0:.5f}".format(max_affinity)+")")
                    ax3.scatter(pTheta,pErr)
                    ax3.set_title('Pixel Error', fontsize=20)
                    ax3.set_ylabel('pixel error', fontsize=20)
                    ax3.set_xlabel('threshold', fontsize=20)
                
                    ax4.plot(pFPR, pTPR, linewidth=2, label=self.name[i]+" (range: "+"{0:.5f}".format(min_affinity)+" - "+"{0:.5f}".format(max_affinity)+")")
                    ax4.scatter(pFPR, pTPR)
                    ax4.set_title('Pixel ROC', fontsize=20)
                    ax4.set_ylabel('true-positive rate', fontsize=20)
                    ax4.set_xlabel('false-positive rate', fontsize=20)
                else:
                    print("Test results not avialable for",self.name[i])
        
        
        #ax1.set_ylim([0,1])
        ax1.set_xlim([0,1])   
        ax2.set_ylim([0,1])
        ax2.set_xlim([0,1]) 
        ax3.set_ylim([0,1])
        ax3.set_xlim([0,1]) 
        ax4.set_ylim([0,1])
        ax4.set_xlim([0,1])      
        
        x = np.arange(0.0, 1.1, 0.1)
        ax2.plot(x, x, '--k', linewidth=1.5)
        ax4.plot(x, x, '--k', linewidth=1.5)
        ax2.legend(bbox_to_anchor=(1., 1.02, 1., .102), loc=2, borderaxespad=0., prop={'size':20})
        ax1.grid(); ax2.grid(); ax3.grid(); ax4.grid();
        plt.show()
  
    
    """Displays three images: the raw data, the corresponding labels, and the predictions"""
    def display(self, result_name, equal_indexing=False):
        
        max_crop = 0
        if equal_indexing:
            for res in range(len(self.name)):
                crop = (self.target.shape[-1] - self.prediction[res].shape[-1])/2
                if(crop > max_crop):
                    max_crop = crop +1

            if (type(result_name) == str):
                res = self.name.index(result_name)
            elif (type(result_name) == int) and (result_name < len(self.prediction)):
                res = result_name
            else:
                print('Invalid Entry')
                return;                
            
            depthInit = 0 
            fig = plt.figure()
            fig.set_facecolor('white')
            ax1,ax2,ax3 = fig.add_subplot(1,3,1),fig.add_subplot(1,3,2),fig.add_subplot(1,3,3)
            fig.subplots_adjust(left=0.2, bottom=0.25)
            
            crop = (self.target.shape[-1] - self.prediction[res].shape[-1])/2
            crop = max_crop - crop
            im_size = self.target.shape[-1] - (2*max_crop)
            
            #Image is grayscale
            im1 = ax1.imshow(self.raw[max_crop+depthInit,max_crop:-max_crop,max_crop:-max_crop],cmap=cm.Greys_r)
            ax1.set_title('Raw Image')
        
            im2 = ax2.imshow(self.target[max_crop+depthInit,max_crop:-max_crop,max_crop:-max_crop,:])
            ax2.set_title('Groundtruth')
        
            im3 = ax3.imshow(self.prediction[res][crop+depthInit,crop:-crop,crop:-crop,crop:-crop])
            ax3.set_title('Predictions')
            
            axdepth = fig.add_axes([0.25, 0.3, 0.65, 0.03], axisbg='white')
            #axzoom  = fig.add_axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
        
            depth = Slider(axdepth, 'Min', 0, im_size, valinit=depthInit,valfmt='%0.0f')
            #zoom = Slider(axmax, 'Max', 0, 250, valinit=max0)
            
            def update(val):
                z = int(depth.val)
                im1.set_data(self.raw[max_crop+z,max_crop:-max_crop,max_crop:-max_crop])
                #im[:,:,:]=label[z,:,:,0]
                im2.set_data(self.target[max_crop+z,max_crop:-max_crop,max_crop:-max_crop,:])
                #im_[:,:,:]=pred[z,:,:,0]
                im3.set_data(self.prediction[res][crop+z,crop:-crop,crop:-crop,crop:-crop])
                fig.canvas.draw()
            fig.suptitle(self.name[res], fontsize = 20)
            depth.on_changed(update)
            plt.show()
        else:
            if (type(result_name) == str):
                res = self.name.index(result_name)
            elif (type(result_name) == int) and (result_name < len(self.prediction)):
                res = result_name
            else:
                print('Invalid Entry')
                return;
             
            depthInit = 0 
            im_size = self.prediction[res].shape[0]
            crop = (self.raw.shape[0]-im_size)/2
            #fig = plt.figure(figsize=(20,10))
            fig = plt.figure()
            fig.set_facecolor('white')
            ax1,ax2,ax3 = fig.add_subplot(1,3,1),fig.add_subplot(1,3,2),fig.add_subplot(1,3,3)
            
            fig.subplots_adjust(left=0.2, bottom=0.25)
        
            #Image is grayscale
            im1 = ax1.imshow(self.raw[crop+depthInit,crop:-crop,crop:-crop],cmap=cm.Greys_r)
            ax1.set_title('Raw Image')
        
            im2 = ax2.imshow(self.target[crop+depthInit,crop:-crop,crop:-crop,:])
            ax2.set_title('Groundtruth')
        
            im3 = ax3.imshow(self.prediction[res][depthInit,:,:,:])
            ax3.set_title('Predictions')
            
            axdepth = fig.add_axes([0.25, 0.3, 0.65, 0.03], axisbg='white')
            #axzoom  = fig.add_axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
        
            depth = Slider(axdepth, 'Min', 0, im_size, valinit=depthInit,valfmt='%0.0f')
            #zoom = Slider(axmax, 'Max', 0, 250, valinit=max0)
            
            def update(val):
                z = int(depth.val)
                im1.set_data(self.raw[crop+z,crop:-crop,crop:-crop])
                #im[:,:,:]=label[z,:,:,0]
                im2.set_data(self.target[crop+z,crop:-crop,crop:-crop,:])
                #im_[:,:,:]=pred[z,:,:,0]
                im3.set_data(self.prediction[res][z,:,:,:])
                fig.canvas.draw()
            fig.suptitle(self.name[res], fontsize = 20)
            depth.on_changed(update)
            plt.show()

            
