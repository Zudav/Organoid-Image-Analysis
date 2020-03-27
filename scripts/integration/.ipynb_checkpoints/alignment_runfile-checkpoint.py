import os
from skimage import io

from modules import file_handling as fh
from modules import alignment as al

# Define which files/organoids should be aligned
masterfolder, cycles, organoids = fh.define_organoids()
organoid_counter = 1
# Loop over all the organoids and do the alignment for each one
for organoid in organoids:
    print("Organoid", organoid_counter, "of", len(organoids))
    print("Loading images...")
    # create a list and load images into it
    offset_images = []
    for cycle in cycles:
        img_path = os.path.join(masterfolder, cycle, "stitched", organoid)
        img = io.imread(img_path)
        offset_images.append(img)
        
    # Remove the first image and save it as the reference image
    img_ref = offset_images.pop(0)

    # Find corner coordinates for the reference image
    print("Find corners in reference image...")
    corners_ref = al.find_corners(img_ref)
    
    # Align offset images to reference image
    aligned_images = []
    for image in offset_images:
        print("Aligning offset image", len(aligned_images)+1, "of", len(offset_images))
        aligned_images.append(al.align_offset_image(img_ref, corners_ref, image))
        
    # Find organoid region and crop reference image
    region_borders, img_ref_cropped = al.find_organoid_region(img_ref)
    
    # Crop offset images
    # Add already cropped reference image to the list and append the others
    cropped_images = [img_ref_cropped]
    for image in aligned_images:
        cropped_images.append(al.crop(image, region_borders))
        
    # Calculate correlation coefficients
    corr_matrix = al.correlation(cropped_images)
    #print("Correlation matrix: \n", corr_matrix)
    
    # Calculate similarity of binary images
    bin_matrix, binary_images = al.binary_similarity(cropped_images)
    #print("Ratio of identical pixels in binary images: \n", bin_matrix)
    
    # Save images and information
    print("Saving images...")
    fh.save_images(masterfolder, organoid, cycles, cropped_images,
                   corr_matrix, bin_matrix,
                   overwrite=False)
    
#    # Plot binary images used for calculating the bin_matrix
#    fig, axes = plt.subplots(nrows=1, ncols=len(binary_images))
#    for i in range(len(binary_images)):
#        axes[i].imshow(binary_images[i])
#
#    # Plot cropped images
#    for image in cropped_images:
#        plt.figure()
#        plt.imshow(image[...,2])
#     
#    # Compare two images directly
#    from skimage.util import compare_images
#    plt.imshow(compare_images(cropped_images[0][...,2], cropped_images[1][...,2], method="checkerboard"))
    
    # Needed for keeping track of which organoid is being aligned
    organoid_counter += 1
        