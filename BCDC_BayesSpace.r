.libPaths( c( "/home/nuwaisir/R/x86_64-pc-linux-gnu-library/4.2", .libPaths()) )

library(BayesSpace)
library(ggplot2)
library(patchwork)

# bcdc <- readRDS('/home/nuwaisir/Corridor/Thesis_ug/ScribbleSeg_Revision_working/Data/others/bcdc_ffpe/BCDC_SCE_Files/bcdc_sce_unprocessed.rds')
bcdc <- readRDS('/home/nuwaisir/Corridor/Thesis_ug/ScribbleSeg_Revision_working/Data/others/bcdc_ffpe/BCDC_SCE_Files/bcdc_sce_nPCA_50.rds')

# melanoma <- getRDS("2018_thrane_melanoma", "ST_mel1_rep2")

# count_matrix <- assay(bcdc, "counts")
# 
# set.seed(2020)
# dec <- scran::modelGeneVar(bcdc)
# top <- scran::getTopHVGs(dec, n = 2000)
# 
# set.seed(2021)
# melanoma <- scater::runPCA(melanoma, subset_row = top)
df_pcs <- data.frame(bcdc@int_colData$reducedDims$PCA)
write.csv(df_pcs, '/home/nuwaisir/Corridor/Thesis_ug/ScribbleSeg_Revision_working/Data/others/bcdc_ffpe/Principal_Components/CSV/pcs_from_BayesSpace.csv')
# Calculate the variance of each principal component
pc_var <- apply(df_pcs, 2, var)

var_explained = pc_var / sum(pc_var)

plot(var_explained, type="b", xlab="Principal Component", ylab="Proportion of Variance Explained", main="Scree Plot")


## Add BayesSpace metadata
bcdc <- spatialPreprocess(bcdc, platform="ST", skip.PCA=TRUE)

q <- 2  # Number of clusters
d <- 15  # Number of PCs

## Run BayesSpace clustering
set.seed(100)
bcdc <- spatialCluster(bcdc, q=q, d=d, platform='ST',
                           nrep=50000, gamma=2)

## View results
palette <- c("purple", "red", "blue", "yellow", "darkblue")
clusterPlot(bcdc, palette=palette, color="black", size=0.1) +
    labs(title="BayesSpace")
write.csv(bcdc@colData, '/home/nuwaisir/Corridor/Thesis_ug/ScribbleSeg_Revision_working/Data/others/bcdc_ffpe/BayesSpace_output.csv')
