library(FlowSOM)
experiment_name = "orig_transl_affine"
point_ID = "Point0061"
input_file_name = paste0(point_ID, "_matrix.csv")
path_training = file.path("/links/groups/treutlein/USERS/pascal_noser/plate14_results/clustering",
                          experiment_name, point_ID, input_file_name)
name_output = "30x30"
path_output = file.path(dirname(path_training), name_output, "SOM")  # output path
dir.create(path_output, recursive = TRUE)  # create output directory

# Specify some variables for the SOM clustering
ydim_SOM = 30
xdim_SOM = 30
dist_metric = 2 # distance metric 2 = euclidean
num_runs = 20 # number of runs
#n_clusters = 12  # expected number of clusters

# Load processed pixel Matrix
data_train = read.csv(path_training,header=FALSE)
# Convert to numeric matrix
data_train = data.matrix(data_train)
# Select protein marker columns to use for clustering
marker_cols = 1:ncol(data_train)

##### RUN FLOWSOM
# create flowFrame object
data_FlowSOM <- flowCore::flowFrame(data_train)

# set seed for reproducibility
set.seed(1234)

# run FlowSOM
fSOM = FlowSOM::ReadInput(data_FlowSOM, transform = FALSE, scale = FALSE)
fSOM = FlowSOM::BuildSOM(fSOM, colsToUse=marker_cols, ydim=ydim_SOM, xdim=xdim_SOM, distf=dist_metric, rlen=num_runs)
# Build minimal spanning tree
fSOM = FlowSOM::BuildMST(fSOM)
PlotStars(fSOM, view = "grid")
## Meta clustering
##cluster_labels = metaClustering_consensus(fSOM$map$codes,k=n_clusters)
#cluster_labels = MetaClustering(fSOM_2$map$codes, "metaClustering_consensus", max=50)
#PlotStars(fSOM, view = "grid", backgroundValues = as.factor(cluster_labels))

## Saving results
# Save the codes/representative vectors of all nodes
file_name = paste0(name_output, "_", "SOMCodes.csv")
file_out_path = file.path(path_output, file_name)
write.csv(fSOM$map$codes, file_out_path)

# Save entire FlowSOM object
file_name = paste0(name_output, "_", "CompleteSOMOutput.rDS")
file_out_path = file.path(path_output, file_name)
saveRDS(fSOM,file_out_path)

## Save the result of the meta clustering, i.e. the cluster label for each node
#file_name = paste0(name_output, "_", "NodeClusterLabels.csv")
#file_out_path = file.path(path_output, file_name)
#write.csv(cluster_labels, file_out_path)



