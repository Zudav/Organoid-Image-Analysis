def scale_image(image):
    
    return np.interp(image, (np.percentile(image,5), np.percentile(image,95)), (0, +65535))