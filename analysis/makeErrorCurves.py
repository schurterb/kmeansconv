import matplotlib.pyplot as plt
import numpy as np

def makeErrorCurves(data0, data1, data2, data3, fig_axes):

    #Data formatting:
    #data0 = (pThresholds, pErr, pTp, pFp, pPos, pNeg, pSqErr)
    #data1 = (minThresholdIdxPixel, maxThresholdIdxPixel)
    #data2 = (rThresholds, rErr, rTp, rFp, rPos, rNeg)
    #data3 = (minThreshIdxRand, maxThreshIdxRand)

    pThresholds, pErr, pTp, pFp, pPos, pNeg, pSqErr = data0
    minThresholdIdxPixel, maxThresholdIdxPixel = data1
    rThresholds, rErr, rTp, rFp, rPos, rNeg = data2
    minThreshIdxRand, maxThreshIdxRand = data3
    
    fig, ((ax1, ax2), (ax3, ax4)) = fig_axes

    fig.set_facecolor('white')

    labels(ax1,"Threshold","Rand F-Score","Rand F-Score")
    ax1.plot(rThresholds[0],rErr[0])
    # ax1.set_xlim([rThresholds[minThreshIdxRand], rThresholds[maxThreshIdxRand]])

    labels(ax2,"False Positive","True Positive","Rand Error ROC") #todo: do this beforehand?
#    rFpRate = rFp/rNeg
#    rFpRate, idxs = np.sort(rFpRate),np.argsort(rFpRate)
#    rTpRate = rTp[idxs]/rPos
    # check to make sure it is monotonically increasing
    ax2.plot(rFp[0]/rNeg[0],rTp[0]/rPos[0])
    ax2.set_ylim([0,1])

    labels(ax3,"Threshold","Pixel Error","Pixel Error")
    ax3.plot(pThresholds[0],pErr[0])
    # ax3.set_xlim([pThresholds[minThresholdIdxPixel], pThresholds[maxThresholdIdxPixel]])

    labels(ax4,"False Positive","True Positive","Pixel Error ROC")
    ax4.plot(pFp[0]/pNeg[0],pTp[0]/pPos[0])
    ax4.set_ylim([0,1])


    plt.tight_layout()
    plt.show()

def labels(ax, x='x-label',y='y-label',title='title',fontsize=15):
    ax.locator_params(axis = 'x', nbins = 5)
    ax.locator_params(axis = 'y', nbins = 5)
    ax.set_xlabel(x, fontsize=fontsize)
    ax.set_ylabel(y, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize+10)


