import pandas as pd  
from skimage.measure import regionprops_table

def get_rprops_table(labels, intensity_image):
        rprops_table = regionprops_table(labels, intensity_image=intensity_image, properties=('label', 'area', 'bbox_area', 'convex_area', 'eccentricity', 'equivalent_diameter', 'euler_number', 'filled_area', 'extent', 'major_axis_length', 'minor_axis_length', 'moments_central', 'moments_hu', 'perimeter','orientation', 'solidity', 'weighted_moments_central', 'weighted_moments_hu', 'inertia_tensor', 'inertia_tensor_eigvals', 'moments', 'max_intensity', 'mean_intensity', 'min_intensity', 'bbox'))

        data = pd.DataFrame(rprops_table)

        return data.set_index("label")
    
    
import copy 
import numpy as np

def get_nuclei_from_label_mask(label_mask, rprops_table):
    
    single_nuclei = copy.deepcopy(label_mask)
    nuclei_clusters = copy.deepcopy(label_mask)

    cluster_labels = rprops_table.index.values[(rprops_table['minor_axis_length'] <= 50) & (rprops_table['major_axis_length'] <= 60)]

    single_nuclei[np.invert(np.isin(single_nuclei, cluster_labels))] = 0
    
    nuclei_clusters[np.isin(nuclei_clusters, cluster_labels)] = 0
    
    return(single_nuclei, nuclei_clusters)