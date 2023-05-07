.libPaths( c( "/home/nuwaisir/R/x86_64-pc-linux-gnu-library/4.2", .libPaths()) )

library(BayesSpace)
library(ggplot2)

dlpfc <- getRDS("2020_maynard_prefrontal-cortex", "151673")

set.seed(101)
dec <- scran::modelGeneVar(dlpfc)
top <- scran::getTopHVGs(dec, n = 2000)

set.seed(102)
dlpfc <- scater::runPCA(dlpfc, subset_row=top)

## Add BayesSpace metadata
dlpfc <- spatialPreprocess(dlpfc, platform="Visium", skip.PCA=TRUE)

q <- 7  # Number of clusters
d <- 15  # Number of PCs

## Run BayesSpace clustering
set.seed(104)
dlpfc <- spatialCluster(dlpfc, q=q, d=d, platform='Visium',
                        nrep=50000, gamma=3, save.chain=TRUE)

## We recoded the cluster labels to match the expected brain layers
labels <- dplyr::recode(dlpfc$spatial.cluster, 3, 4, 5, 6, 2, 7, 1)

## View results
clusterPlot(dlpfc, label=labels, palette=NULL, size=0.05) +
  scale_fill_viridis_d(option = "A", labels = 1:7) +
  labs(title="BayesSpace")


labels <- dplyr::recode(dlpfc@colData$Cluster, 1, 4, 5, 6, 2, 7, 3)

## View results
clusterPlot(dlpfc, label=labels, palette=NULL, size=0.05) +
  scale_fill_viridis_d(option = "A", labels = 1:7) +
  labs(title="BayesSpace")

write.csv(data.frame(dlpfc@colData$cluster.init), '/home/nuwaisir/Corridor/Thesis_ug/ScribbleSeg_Revision_working/Data/Human_DLPFC/151673/mclust_output_from_BayesSpace.csv')

temp <- dlpfc$spatial.cluster
