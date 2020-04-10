import numpy as np

def scale_image(image, percentile = 5):
    
    #if image.min() > 0:
        return np.interp(image, (np.percentile(image,percentile), np.percentile(image,100 - percentile)), (0, +65535))
     
    #else:
        #return np.interp(image, (np.unique(np.sort(np.ravel(image)))[1], np.unique(np.sort(np.ravel(image)))[-1]), (0, +65535))

import skimage
from skimage.filters import threshold_otsu
    
def subtract_membrane(dapi, membrane):
    dapi = scale_image(dapi)
    membrane = scale_image(membrane)
    tmp = dapi - membrane
    tmp[tmp < 0] = 0
    thresh = threshold_otsu(tmp)
    tmp[tmp < thresh] = 0

    '''dapi = scale_image(dapi)
    membrane = scale_image(membrane)
    tmp = dapi - membrane
    thresh = threshold_otsu(tmp)
    tmp[tmp < thresh] = 0'''
    return scale_image(tmp)


from skimage.filters import threshold_niblack, threshold_local
import copy
from scipy import ndimage as ndi

def threshold(image, filter = threshold_niblack):
    niblack = filter(image)
    niblack_copy = copy.deepcopy(image)
    niblack_copy[niblack_copy < niblack] = 0
    tmp = threshold_otsu(niblack_copy)
    niblack_copy[niblack_copy < tmp] = 0
    return niblack_copy

from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import filters
import os
import Paths
from skimage import io
import imageio

def initial_segment(dapi, membrane):
    
    #if os.path.isfile(Paths.aligned_images_path() + "/MAX_Time00000_Point0000_Point00{ii}_ChannelSCF_SD/dapi_mask.tif"):
        
        #return io.imread(Paths.aligned_images_path() + "/MAX_Time00000_Point0000_Point00{ii}_ChannelSCF_SD" + "/dapi_mask.tif")
    
    #else:
        
        subtracted = subtract_membrane(dapi, membrane)
    
        threshed = threshold(subtracted)
    
        filtered = filters.gaussian(threshed, sigma=0.4, preserve_range=True, truncate = 3)
        local_maxi = peak_local_max(filtered, indices=False, footprint=np.ones((2, 2)),
                            labels=filtered)
        markers = ndi.label(local_maxi)[0]
        labels = watershed(-filtered, markers, mask=filtered)
        hist = np.ravel(labels)[np.ravel(labels) > 0]
        summary_hist = np.histogram(hist, bins = np.append(np.unique(hist),[np.unique(hist).size + 1]))
        large_clusters = summary_hist[1][np.nonzero(summary_hist[0] > 500)]
        mask2 = np.isin(labels, large_clusters)
        labels[mask2 == False] = 0
        #imageio.imwrite(os.path.join(Paths.aligned_images_path(), 'MAX_Time00000_Point0000_Point00{ii}_ChannelSCF_SD', 'dapi_mask_new.tif') , large_clusters)
        return labels
    
        

