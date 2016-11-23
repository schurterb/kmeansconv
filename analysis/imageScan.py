# These are a set of functions to aid viewing of 3D EM images and their
# associated affinity graphs

import matplotlib
#matplotlib.use('Agg')
import matplotlib.cm as cm
import numpy as np
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

#Displays three images: the raw data, the corresponding labels, and the predictions
def display(raw, label, pred, depthInit=1):
    im_size = pred.shape[0]
    crop = (raw.shape[0]-im_size)/2
    #fig = plt.figure(figsize=(20,10))
    fig = plt.figure()
    fig.set_facecolor('white')
    ax1,ax2,ax3 = fig.add_subplot(1,3,1),fig.add_subplot(1,3,2),fig.add_subplot(1,3,3)

    fig.subplots_adjust(left=0.2, bottom=0.25)
    depth0 = 0

    raw = np.transpose(raw)
    label = label    
    pred = np.transpose(pred, (2,1,0,3))
    

    #Image is grayscale
    im1 = ax1.imshow(raw[crop+depthInit,crop:-crop,crop:-crop],cmap=cm.Greys_r)
    ax1.set_title('Raw Image')

    im2 = ax2.imshow(label[crop+depthInit,crop:-crop,crop:-crop,:])
    ax2.set_title('Groundtruth')

    im3 = ax3.imshow(pred[depthInit,:,:,:])
    ax3.set_title('Predictions')
    
    axdepth = fig.add_axes([0.25, 0.3, 0.65, 0.03], axisbg='white')
    #axzoom  = fig.add_axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

    depth = Slider(axdepth, 'Min', 0, im_size, valinit=depth0,valfmt='%0.0f')
    #zoom = Slider(axmax, 'Max', 0, 250, valinit=max0)
    
    def update(val):
        z = int(depth.val)
        im1.set_data(raw[crop+z,crop:-crop,crop:-crop])
        #im[:,:,:]=label[z,:,:,0]
        im2.set_data(label[crop+z,crop:-crop,crop:-crop,:])
        #im_[:,:,:]=pred[z,:,:,0]
        im3.set_data(pred[z,:,:,:])
        fig.canvas.draw()
    depth.on_changed(update)
    plt.show()
