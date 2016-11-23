import matplotlib.pyplot as plt
from .makeErrorCurves import makeErrorCurves
from scipy.io import loadmat

def showStats(dataFile):
    
    fig_axes = plt.subplots(nrows=2, ncols=2)
    dFile = dataFile[0]
    mData = loadmat(dFile + 'errors_new.mat')
    
    data0 = [mData.get('p_thresholds'), mData.get('p_err'), mData.get('p_tp'),
             mData.get('p_fp'), mData.get('p_pos'), mData.get('p_neg'), mData.get('p_sqerr')] 
    data1 = [mData.get('p_thresholds').min(), mData.get('p_thresholds').max()]
    data2 = [mData.get('r_thresholds'), mData.get('r_err'), mData.get('r_tp'),
             mData.get('r_fp'), mData.get('r_pos'), mData.get('r_neg')]
    data3 = [mData.get('r_thresholds').min(), mData.get('r_thresholds').max()]
    
    makeErrorCurves(data0, data1, data2, data3, fig_axes)
        
    return data0, data2