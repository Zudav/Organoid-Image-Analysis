# Organoid Image Analysis

## Current pipeline:

Alignment: 
	Alignment/elastix_runfile.ipynb
	Alignment/alignment_check_1/2.ipynb (Optional)

Processing:
	various_notebooks/padding_and_renaming_aligned.ipynb
	Masking/mask_creation.ipynb

Clustering:
	Clustering/pixel_matrix_creation.ipynb
	Clustering/flowsom_clustering.R
	Clustering/pixel_matrix_clustering.ipynb

Segmentation:
	Segmentation/cellpose_runfile.ipynb