import numpy as np
from skimage.filters import (gaussian, threshold_isodata,threshold_li,
                             threshold_mean, threshold_minimum, threshold_otsu,
                             threshold_triangle, threshold_yen)
from skimage.feature import (corner_harris, corner_subpix, corner_peaks)
from skimage.exposure import rescale_intensity
from skimage.transform import warp, EuclideanTransform
# AffineTransform: independent scaling in x and y
# SimilarityTransform: one scaling parameter
# EuclideanTransform: no scaling
from skimage.measure import ransac
from skimage.util import img_as_uint
from skimage.morphology import remove_small_objects


# Alignment part based on: https://scikit-image.org/docs/dev/auto_examples/transform/plot_matching.html#sphx-glr-auto-examples-transform-plot-matching-py

def find_corners(img, corner_dist=50, border=101):
    print("Finding corners...")
    # extract dapi channel
    img_dapi = img[..., 2]
    # rescale
    img_dapi = rescale_intensity(img_dapi)
    # find corners
    corners_img = corner_peaks(corner_harris(img_dapi), threshold_rel=0.0001,
                              min_distance=corner_dist, exclude_border=border)
    # get subpixel position
    corners_img_subpix = corner_subpix(img_dapi, corners_img, window_size=9)
    # use only pixel-precision if no subpixel position was found
    nan_idx = np.argwhere(np.isnan(corners_img_subpix[:,0]))
    corners_img_subpix[nan_idx,:] = corners_img[nan_idx]
    print("Found", len(corners_img), "corners.")
    return corners_img_subpix


# While any weight function would do here, I think the gaussian_weight makes
# sense because by adjusting the sigma value one can quite easily adjust the
# weight distribution.
def gaussian_weights(window_ext, sigma=1):
    y, x = np.mgrid[-window_ext:window_ext+1, -window_ext:window_ext+1]
    g = np.zeros(y.shape, dtype=np.double)
    g[:] = np.exp(-0.5 * (x**2 / sigma**2 + y**2 / sigma**2))
    g /= 2 * np.pi * sigma * sigma
    return g


def match_corner(corner, corners_off, img_ref, img_off, window_ext=25):
    img_ref_dapi = img_ref[...,2]
    img_off_dapi = img_off[...,2]
    # Define a "window" around the corner that is taken into account when
    # comparing corners from the two images. A cell diameter at 40x is about
    # 50 pixels.
    r, c = np.round(corner).astype(np.intp)
    window_1 = img_ref_dapi[r-window_ext:r+window_ext+1,
                           c-window_ext:c+window_ext+1]
    # weight pixels depending on distance to center pixel
    # Increasing sigma and window_ext essentially increases the patch size.
    weights = gaussian_weights(window_ext, sigma=window_ext*0.75)

    # compute sum of squared differences to all corners in second image
    SSDs = []
    for cr, cc in np.round(corners_off).astype(np.intp):
        window_2 = img_off_dapi[cr-window_ext:cr+window_ext+1,
                                   cc-window_ext:cc+window_ext+1]
        SSD = np.sum(weights * (window_1 - window_2)**2)
        SSDs.append(SSD)

    # use corner with minimum SSD as correspondence
    min_idx = np.argmin(SSDs)
    return corners_off[min_idx]


def find_correspondences(img_ref, img_off, corners_ref):
    corners_off = find_corners(img_off)
    print("Finding correspondences...")
    src = []
    dst = []
    for corner in corners_ref:
        src.append(corner)
        dst.append(match_corner(corner, corners_off, img_ref, img_off))
    src = np.array(src)
    dst = np.array(dst)
    return src, dst


def estimate_model(img_ref, img_off, corners_ref, initial_thr=0.2, min_inliers=4, max_inliers=8):
    # Find matching corners
    src, dst = find_correspondences(img_ref, img_off, corners_ref)
    # ransac expects (column, row) instead of (row, column)
    src = np.flip(src, axis=1)
    dst = np.flip(dst, axis=1)
    # initialise inliers and scale to values outside desired range
    inliers = np.full((1,), False)
    model_robust = EuclideanTransform()
    res_thr = initial_thr
    print("Finding appropriate residual threshold...")
    print("Initial threshold: ", initial_thr)
    
    while sum(inliers)<min_inliers or sum(inliers)>max_inliers:
        model_robust, inliers = ransac((src, dst), EuclideanTransform, min_samples=3,
                                       residual_threshold=res_thr, max_trials=75000)
        # All of this increasing and decreasing could be heavily improved
        if sum(inliers)>max_inliers:
            if res_thr < 0.06:
                break
            elif res_thr < 0.11:
                res_thr -= 0.01
            elif sum(inliers) > 2*max_inliers:
                res_thr = np.round(res_thr/2, 1)
            else:
                res_thr -= 0.1
            print("Decreasing threshold to:", np.round(res_thr, 2))
            
        if sum(inliers)<min_inliers:
            if sum(inliers) < np.ceil(min_inliers/2):
                res_thr += np.round(res_thr*2, 1)
            else:
                res_thr += 0.2
            print("Increasing threshold to:", np.round(res_thr, 1))
    
    # Estimate final model
    print("Estimating final model...")
    model_robust, inliers = ransac((src, dst), EuclideanTransform, min_samples=3,
                                   residual_threshold=res_thr, max_trials=300000)
    #print("Number of inliers found:", sum(inliers))
    print("Translation:", np.round(model_robust.translation, 5))
    print("Rotation:", np.round(model_robust.rotation, 5))
    return model_robust
    

def align_offset_image(img_ref, corners_ref, img_off):
    model_robust = estimate_model(img_ref, img_off, corners_ref)
    print("Applying model to offset image...")
    img_off_warped = np.zeros(np.shape(img_off))
    for channel in range(0, img_off.shape[2]):
        img_off_warped[..., channel] = warp(img_off[..., channel], model_robust, cval=0)
    # Turn warped img into uint16
    img_off_warped = img_as_uint(img_off_warped)
    return img_off_warped

def crop(img, borders):
    img_cropped = img[borders[0]:borders[1]+1, borders[2]:borders[3]+1]
    return img_cropped


def find_organoid_region(img, excess=200, remove_size=5000):
    print("Determining organoid region...")
    # extract DAPI channel
    img_dapi = img[..., 2]
    # Heavily blur the image 
    img_thr = gaussian(img_dapi, sigma = 50)
    # Determine the threshold value and create a binary image
    thr = threshold_mean(img_thr)
    img_thr = img_thr > thr
    # Remove any potential artefacts
    img_thr = remove_small_objects(img_thr, min_size=remove_size)
    # Find non-empty rows and columns
    non_empty_columns = np.where(img_thr.max(axis=0)>0)[0]
    non_empty_rows = np.where(img_thr.max(axis=1)>0)[0]
    borders = (min(non_empty_rows)-excess,
               max(non_empty_rows)+excess,
               min(non_empty_columns)-excess, 
               max(non_empty_columns)+excess)
    # crop reference image
    img_cropped = crop(img, borders)
    return borders, img_cropped

def correlation(images):
    # Get DAPI values for all pixesl
    img_vals = []
    for image in images:
        img_vals.append(image[...,2].flatten())
    img_vals = np.asarray(img_vals)
    
    # Since the reference images might be cut off and since those (and only those) pixels
    # will have a value of 0, find those pixels and remove them
    zero_indices = np.where(img_vals == 0)[1]
    img_vals = np.delete(img_vals, zero_indices, 1)
    corr_mat = np.corrcoef(img_vals)
    
    return corr_mat


def binary_similarity(images, thr_function=threshold_otsu):
    # Binarize images
    binary_images = []
    for img in images:
        # extract DAPI channel
        img_dapi = img[..., 2]
        # Determine the threshold value and create a binary image
        thr = thr_function(img_dapi)
        binary_images.append(img_dapi > thr)
    # intialise matrix
    bin_mat = np.ones([len(images), len(images)])
    offset = 0
    total_size = binary_images[0].size
    # loop over matrix and fill 
    for i in range(len(images)):
        # increase offset such that only values above diagonal are calculated
        offset += 1
        for j in range(offset, len(images)):
            img_1 = binary_images[i]
            img_2 = binary_images[j]
            xnor_mat = np.invert(np.bitwise_xor(img_1, img_2))
            ratio = np.sum(xnor_mat)/total_size
            bin_mat[i, j] = bin_mat[j, i] = ratio
    
    return bin_mat, binary_images

if __name__ == "__main__":
    print("Only used as module")