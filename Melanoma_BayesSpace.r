.libPaths( c( "/home/nuwaisir/R/x86_64-pc-linux-gnu-library/4.2", .libPaths()) )

library(BayesSpace)
library(ggplot2)
library(patchwork)

melanoma <- getRDS("2018_thrane_melanoma", "ST_mel1_rep2")

count_matrix <- assay(melanoma, "counts")

set.seed(2020)
dec <- scran::modelGeneVar(melanoma)
top <- scran::getTopHVGs(dec, n = 2000)

set.seed(2021)
melanoma <- scater::runPCA(melanoma, subset_row = top)
df_pcs <- data.frame(melanoma@int_colData$reducedDims$PCA)
write.csv(df_pcs, '/home/nuwaisir/Corridor/Thesis_ug/ScribbleSeg_Revision/Data/others/Melanoma/PCS/CSV/pcs_from_BayesSpace.csv')
# Calculate the variance of each principal component
pc_var <- apply(df_pcs, 2, var)

var_explained = pc_var / sum(pc_var)

plot(var_explained, type="b", xlab="Principal Component", ylab="Proportion of Variance Explained", main="Scree Plot")


## Add BayesSpace metadata
melanoma <- spatialPreprocess(melanoma, platform="ST", skip.PCA=TRUE)

q <- 4  # Number of clusters
d <- 7  # Number of PCs

## Run BayesSpace clustering
set.seed(100)
melanoma <- spatialCluster(melanoma, q=q, d=d, platform='ST',
                           nrep=50000, gamma=2)

## View results
palette <- c("purple", "red", "blue", "yellow", "darkblue")
clusterPlot(melanoma, palette=palette, color="black", size=0.1) + labs(title="BayesSpace")

write.csv(melanoma@colData, '/home/nuwaisir/Corridor/Thesis_ug/ScribbleSeg_Revision_working/Data/others/Melanoma/BayesSpace_output.csv')
