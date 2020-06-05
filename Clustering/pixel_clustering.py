import numpy as np
import matplotlib as mpl
import scipy as sp
from matplotlib import pyplot as plt
import os
import fnmatch
from skimage import io
from skimage.util import view_as_blocks
from skimage.exposure import rescale_intensity
from minisom import MiniSom
import phenograph
from scipy.stats import zscore

class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return sp.ma.masked_array(sp.interp(value, x, y))

def get_grid(img, grid_size, multichannel=True):
    x_surplus = img.shape[0]%grid_size[0]
    y_surplus = img.shape[1]%grid_size[1]
#    print(x_surplus, y_surplus)
    img_cropped = img[0:img.shape[0]-x_surplus, 0:img.shape[1]-y_surplus]
    if multichannel:
            # Along the z-axis we don't want a grid so add a "1" along the third dimension
            grid_size = grid_size + (1,)
    grid = view_as_blocks(img_cropped, block_shape=grid_size)
    return grid

print("Load images...")
# For now doing this manually
image_path = r"/links/groups/treutlein/DATA/imaging/PW/4i/plate6/AlignedOrganoids/MAX_Time00000_Point0000_Point00{ii}_ChannelSCF_SD/cycles"
organoid_files = fnmatch.filter(os.listdir(image_path), "*.tif") # just load all the .tif files
# Move reference cycle to first position. In this case I just know it's cycle3 but for the future
# I'll have to figure out how to automate this
organoid_files.insert(0, organoid_files.pop(organoid_files.index("cycle3_aligned.tif")))
images = []
for organoid in organoid_files:
    path = os.path.join(image_path, organoid)
    img = io.imread(path)
    images.append(img)

# Has to be in the same orders as the cycles, and within one cycle r,g,b,white
marker_names = ["Bassoon", "r/g opsin", "Hoechst", "Tau",   #cycle3 (r,g,b,w)
                "TUJI1", "Nestin", "EGFR",                  #cycle1 (r,g,w)
                "Rho", "Na+/K+ ATPase", "NRL",              #cycle2 (r,g,w)
                "MAP2", "KI-67", "Cxh10"]                   #cycle4 (r,g,w)

# img_ref is the only one whose DAPI channel we use for the clustering
img_ref = images[0]

# since we need the DAPI information only once, we can remove that channel from all the images but the first
antibody_images = [img_ref]
for i in range(1, len(images)):
    antibody_images.append(np.delete(images[i], 2, 2))

# Stack all images into a single array with their colour channels along the third axis
antibody_array = np.dstack((antibody_images))

# Rescale the intensities between 0 and 1
print("Rescale intensities...")
bg_vals = np.repeat(100, antibody_array.shape[2]) # values below these will be clipped
max_quantile = 0.98 # quantile at which intensities will be clipped when resacling
antibody_array = antibody_array * 1.0 # turn to float before rescaling
rescaled_array = np.zeros(antibody_array.shape) # initialise new array
for channel in range(rescaled_array.shape[2]): # loop over all channels
    img = antibody_array[...,channel]
    max_val = np.quantile(img, max_quantile)
    rescaled_array[...,channel] = rescale_intensity(img, 
                  in_range=(bg_vals[channel], max_val))

# Create a grid
print("Get grid...")
grid = get_grid(rescaled_array, grid_size=(10,10))

# Create a new array with the mean values of each block of the grid
print("Create 'blocked' array...")
blocked_array = np.zeros(grid.shape[:3])
for channel in range(grid.shape[2]):
    for column in range(grid.shape[1]):
        for row in range(grid.shape[0]):
            block = grid[row, column, channel][...,0]
            block_average = np.mean(block)
            blocked_array[row,column,channel] = block_average

# For each pixel, find the maximum intensity across all channels
print("Filter out pixels with low intensities only...")
nrows = blocked_array.shape[0]
ncols = blocked_array.shape[1]
nchan = blocked_array.shape[2]
max_pixel_int = np.zeros((nrows, ncols))
for row in range(nrows):
    for column in range(ncols):
        channel_vals = []
        for channel in range(nchan):
            max_val = blocked_array[row, column, channel]
            channel_vals.append(max_val)
        max_pixel_int[row, column] = max(channel_vals)

# Find pixels with intensities exclusively below a certain level
minimum_intensity = 1/3
excluded_pixels = tuple(zip(*np.where(max_pixel_int < minimum_intensity)))
excluded_pixels_flat = np.where(max_pixel_int.ravel() < minimum_intensity)[0]
included_pixels = tuple(zip(*np.where(max_pixel_int >= minimum_intensity)))
included_pixels_flat = np.where(max_pixel_int.ravel() >= minimum_intensity)[0]

## Self-organizing map
# Change data format such that each pixel is a row and the columns represent the intensity values
# the SOM requires such a datastructure
pixels_all = np.reshape(blocked_array, (nrows*ncols, nchan))
# Exclude pixels defined above
pixels = np.delete(pixels_all, excluded_pixels_flat, axis=0)

# Rule of thumb: Grid should contain 5*sqrt(N) neurons
n_neurons = np.sqrt(pixels.shape[0])*5
som_size = int(np.ceil(np.sqrt(n_neurons)))
# Initialise SOM
som = MiniSom(som_size, som_size, nchan, sigma=1, learning_rate=0.5)
# Initialise weights to random samples from the data
print("Initialise SOM...")
som.pca_weights_init(pixels)
starting_weights = som.get_weights().copy()  # saving the starting weights
# Train SOM by picking random samples
print("Train SOM:")
som.train_random(pixels, num_iteration=100000, verbose=True)

## Quantize each pixel, i.e. assign to each pixel the weight that's closest to it
#print("Quantization")
#qnt = som.quantization(pixels)
## Add the excluded pixels back in and assign them a weight vector of only zeros
#qnt_complete = np.zeros(pixels_all.shape)
#qnt_complete[included_pixels_flat] = qnt
#excluded_pixels_weights = np.zeros(blocked_array.shape[2])
#qnt_complete[excluded_pixels_flat,:] = excluded_pixels_weights
## Create new empy image and fill it with the quantized pixels
#print("Create new image")
#som_clustered_img = np.zeros(blocked_array.shape)
#for i, q in enumerate(qnt_complete):
#    som_clustered_img[np.unravel_index(i, shape=(nrows, ncols))] = q

# Get weights and put into datastrucutre where each row is one SOM node and the columns are the weights
som_nodes = som.get_weights()
som_nodes_flat = np.reshape(som_nodes, (som_size*som_size, nchan))

## Phenograph
## Find best neighbourhood value
#neigh_to_check = np.arange(2, 103, 10)
#mcu_detected = np.zeros(len(neigh_to_check))
#for i in range(len(neigh_to_check)):
#    cluster_labels_som_flat = phenograph.cluster(som_nodes_flat, k=neigh_to_check[i])[0]
#    mcu_detected[i] = len(np.unique(cluster_labels_som_flat))

#plt.plot(neigh_to_check, mcu_detected)

# Uncomment the above chunk to find the best neighbrouhood value. I just chose 70 below because this results in about 10 clusters
neigh_selected = 70
cluster_labels_som_flat = phenograph.cluster(som_nodes_flat, k=neigh_selected)[0]
# Get a 2D representation of all the cluster labels where each node is assigned its cluster
cluster_labels_som = np.zeros(som_nodes.shape[:2])
for i, q in enumerate(cluster_labels_som_flat):
    cluster_labels_som[np.unravel_index(i, shape=cluster_labels_som.shape)] = q

# Create the multiplexed cell unit image where each pixel block is assigned its cluster by checking
# its node in the SOM
print("Assigning pixels to MCUs...")
mcu_img_flat = np.zeros((nrows*ncols, 1))
for index in range(len(mcu_img_flat)):
    if index in excluded_pixels_flat:
        mcu_img_flat[index] = -1
    else:
        som_coords = som.winner(pixels_all[index])
        mcu_img_flat[index] = cluster_labels_som[som_coords]

# Increase MCU numbers by one (because the currently start at 0 rather than 1)
mcu_img_flat[mcu_img_flat >= 0] += 1
cluster_labels = np.unique(mcu_img_flat).astype(int)

# Turn the flat array into a 2D image
mcu_img = np.zeros((nrows, ncols))
for i, q in enumerate(mcu_img_flat):
    mcu_img[np.unravel_index(i, shape=mcu_img.shape)] = q


## For each mcu, print only those pixels that belong to that cluster 
#for value in cluster_labels:
#    plt.figure()
#    plt.title("MCU {}".format(int(value)))
#    plt.imshow(mcu_img == value)

# For every mcu get the mean of each antibody intensity
mcu_avg_list = []
for mcu in cluster_labels[1:]: # skip "-1" because those are the pixels belonging to no cluster
    idx = np.where(mcu_img_flat == mcu)[0] # [1] is just zeros (first and only column)
    mean_vals = np.mean(pixels_all[idx, :], axis=0).reshape(nchan,1) # reshape to a column vector
    mcu_avg_list.append(mean_vals)

# Each row is an antibody and each column an mcu
mcu_avg = np.hstack(mcu_avg_list)
# Compute z-score for each antibody individually
mcu_zscored = zscore(mcu_avg, axis=1)
# PLot z-scored intensities
# Normalize the z-scores around the midpoint, making sure that 0 is white in the plot
mcu_min = mcu_zscored.min()
mcu_max = mcu_zscored.max()
norm = MidpointNormalize(vmin=mcu_min, vmax=mcu_max, midpoint=0)
fig = plt.figure(figsize=(16, 9))   
ax = fig.add_subplot(111)
plt.imshow(mcu_zscored, cmap="RdBu", norm=norm)
plt.yticks(np.arange(len(marker_names)), marker_names, fontsize=15)
ax.tick_params(top=True, bottom=False,
               labeltop=True, labelbottom=False)
plt.xticks(np.arange(len(cluster_labels[1:])),
           cluster_labels[1:].astype(int), fontsize=15)
ax.set_xlabel("MCU", size=18)
ax.xaxis.set_label_position('top') 
cb = plt.colorbar()
cb.set_label(label="Marker intensity (z-score)", size=20)
cb.ax.tick_params(labelsize=15)
#ax.set_ylim(len(marker_names)-0.5, -0.5) # ONLY NECESSARY BECAUSE OF A BUG IN MATPLOTLIB 3.1.1


# Plot showing diversity within each MCU
n_clusters = len(cluster_labels[1:])
# Define ratio of x to y for each subplot
aspect_ratio = 50
fig, axes = plt.subplots(nrows=n_clusters, ncols=1, sharex=True, figsize=(16,9))
for i in range(n_clusters):
    # get current cluster
    cluster = cluster_labels[1:][i]
    # get current axis
    ax = axes[i]
    # Get all the pixels that belong to the current cluster
    idx = np.where(mcu_img_flat == cluster)[0]
    # plot image
    ax.imshow(pixels_all[idx], vmin=0, vmax=1)
    # remove yticks
    ax.set_yticks([],[])
    # add MCU number as y label of the subplot
    ax.set_ylabel(cluster, rotation=0, va="center", ha="right", fontsize=10)
    # Set the aspect ratio
    ratio_default=(abs(ax.get_xlim()[1]-ax.get_xlim()[0]))/abs((ax.get_ylim()[1]-ax.get_ylim()[0]))
    ax.set_aspect(ratio_default/aspect_ratio)
# Add marker names
plt.xticks(np.arange(len(marker_names)), marker_names)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
# Create "outer" image in order to add a common y-label
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.ylabel("MCU", fontsize=15)
plt.show()


# Create custom palette
palette = mpl.cm.get_cmap("tab10")
palette.set_bad("white", alpha=0.9) # colour for masked out pixels
palette.set_under("black") # colour for pixels with a value < vmin in imshow norm
# plot all clusters in one image
fig, ax = plt.subplots()
# plot image with all clusters in different colours
ax.imshow(mcu_img, cmap=palette, 
           norm=mpl.colors.Normalize(vmin=1, vmax=np.unique(mcu_img).max()))
ax.axis("off")
plt.tight_layout()
plt.show() 
# plot each cluster individually
for cluster in cluster_labels[1:]: #[1:] to skip "cluster" -1
    # Create mask
    mcu_mask = np.ma.masked_where(mcu_img != cluster, mcu_img)
    fig, ax = plt.subplots()
    # plot image with all clusters in different colours
    ax.imshow(mcu_img, cmap=palette, 
               norm=mpl.colors.Normalize(vmin=1, vmax=np.unique(mcu_img).max())) 
    # plot masked image over original one
    ax.imshow(mcu_mask, cmap=palette,
               norm=mpl.colors.Normalize(vmin=1, vmax=np.unique(mcu_img).max())) 
    plt.title("MCU {}".format(int(cluster)))
    ax.axis("off")
    plt.tight_layout()
    plt.show()

# Plot all markers
for i in range(len(marker_names)):
    current_marker = marker_names[i]
    fig, ax = plt.subplots()
    ax.imshow(rescaled_array[..., i], cmap="gray")
    plt.title(current_marker)      
    ax.axis("off")
    plt.tight_layout()
    plt.show()
            
