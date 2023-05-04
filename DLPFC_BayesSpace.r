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