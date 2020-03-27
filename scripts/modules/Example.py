import numpy as np

def scale_image(image):
    
    if image.min() > 0:
        return np.interp(image, (np.percentile(image,5), np.percentile(image,95)), (0, +65535))
     
    else:
        return np.interp(image, (np.unique(np.sort(np.ravel(image)))[1], np.unique(np.sort(np.ravel(image)))[-1]), (0, +65535))