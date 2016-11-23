#!/home/user/anaconda3/bin/python
"""
Created on Wed Oct  5 21:44:08 2016

@author: user

Train script for building (or loading)
 and training a kmeansconv network.
"""

import time
import argparse
from sh import ls

from presentation import Presenter

def showResults(network_list):
    print("Presenting Results")    
    
    #Create a presentor for each set of results
    presenters = []
    for network in network_list:
        if not network.endswith("/"):
            network += "/"
        prediction_set = ls(network+"results").split()
        timestamps = []
        for prediction in prediction_set:
            if prediction.startswith("prediction"):
                timestamps.append(prediction.strip(".csv").split("_")[1])
        for timestamp in timestamps:
            presenters.append(Presenter(network, data_timestamp=timestamp))

    #Display results from all datasets
    for presenter in presenters:    
        starttime = time.clock()
        presenter.plotAll()
        print("Results for ",presenter.name,": ",(time.clock()-starttime)," seconds")
        
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="comma-separated list of network folders")
    
    args = parser.parse_args()
    if args.f:
        networks = args.f.split(",")
        
        showResults(networks)
    else:
        print("Trained network required for testing.")
        print("-f","   ","comma-separated list of network folders")
    
    
    
    
    
    
    
             
        