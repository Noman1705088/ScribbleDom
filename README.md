# ScribbleSeg
A method to segment Spatial Transcriptomics data using scribble annotated histology image, or using 
output of other possibly non-spatial segmentation method (e.g. mclust).

# Prerequisites
Results are generated using Google Colab Standard GPU. To ensure reproducibility, the following is done:
```
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
Note that, you do not need to set these if you run the program in cpu instead of gpu.

Recommended Python version: 3.10.6

# Installation
First set and activate your environment by using the following command:
```
conda env create -f environment.yml
conda activate scribble_seg
```
or, install the required packages by using the following command:
```
pip install -r requirements.txt
```

# Input
To run ScribbleSeg, you need 3 input files for a sample:
 - The .h5 file containing the Anndata object
    - It must contain the ```array_row```, and ```array_col``` representing the position of each spot in a grid
 - A CSV file containing the principal component values
 - A CSV file containing the scribble information for each spot

## Location of the .h5 file:
Place your .h5 file in ```./Data/[dataset name]/[sample name]/reading_h5/```. The current implementation reads the Anndata object
by using ```scanpy.read_visium``` function. Hence, it expects a folder named ```spatial```, containing the scalefactors, tissue high 
resolution image, low resolution image, tissue possitions list and so on, inside the ```reading_h5``` folder. Depending on the 
structure of the .h5 file, you can use ```scanpy.read_h5ad``` fucntion to read the .h5 file, which doesn't require this ```spatial``` folder.

## Location of the CSV containing principal components:
Place the csv containing the principal compoents in ```./Data/[dataset_name]/[sample_name]/Principal_Components/CSV/```.
Make sure to change the name of the ```pc_csv_path``` variable according to the name of your CSV file in ```preprocessor.py```.

## Location of the CSV containing scribbles:
Place the csv containing the principal compoents in ```./Data/[dataset_name]/[sample_name]/```.
Make sure to change the name of the ```scr_csv_path``` variable according to the name of your CSV file in ```preprocessor.py```.

## Other input parameters
Set the input parameters (e.g. hyperparameters, output directory name, and so on...) in ```./Inputs/[file_name].json``` by following the example json files shown inside ```./Inputs```.

# How to generate scribbles?
Scribbles can be generated using [Loupe browser](https://support.10xgenomics.com/single-cell-gene-expression/software/visualization/latest/what-is-loupe-cell-browser)

# How to run?
After setting up the input parameters on ```./Inputs/[file_name].json```, the following steps are required to run ScribbleSeg:
1. At first you have to run preprocessor.py. For expert scribble scheme, run the following:
```
python preprocessor.py --scheme expert --params "Preprocessor_input/bcdc_preprocessor_scheme.json"
```
Or, for mclust scribble scheme, run the following:
```
python preprocessor.py --scheme mclust --params Preprocessor_input/bcdc_preprocessor_scheme.json
```
2. Then, to generate the segmentations for expert scribble scheme, run:
```
python expert_scribble_pipeline.py --params ./Inputs/expert/bcdc_expert_scribble_scheme_input.json
```
Or, to generate the segmentations for mclust scribble scheme, run:
```
python mclust_scribble_pipeline.py --params ./Inputs/mclust/bcdc_mclust_scribble_scheme_input.json
```
3. Results will be put in Outputs directory.
4. To calculate adjusted rand index (ARI), you will need the ground truth labels. Put the ground truth labels at ```./Data/[dataset name]/[sample name]/manual_annotations.csv```.
5. 

# Other Informations
The folder ```Supplementary_figures``` has the high quality figures of the supplementary information of our research paper.

For the local data goto link : https://drive.google.com/drive/folders/1PIsWdA65sJVAgxi0gfrNFzrx_HGpvCyX?usp=share_link
