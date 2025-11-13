# **AugGCL: Multimodal Graph Learning for Spatial Transcriptomics Analysis with Enhanced Gene and Morphological Data**

## Project Overview

**AugGCL** is a multimodal graph-convolutional learning framework designed to enhance spatial structure decoding and gene expression reconstruction by augmenting both gene and image data. The method utilizes the **Neighborhood Information Aggregation (NIA)** module, which integrates gene expression similarity and spatial proximity to construct a weighted graph and an enhanced expression matrix. This effectively addresses sparsity issues while maintaining boundary clarity. Additionally, AugGCL employs a two-stream weighted graph convolutional network (GCN) that jointly models gene features and image-derived morphological information, with image-aware auxiliary reconstructions enhancing weak spatial signals and sharpening boundaries.

On datasets from the human dorsolateral prefrontal cortex, breast cancer, and mouse embryo, AugGCL outperforms traditional baseline methods across multiple evaluation metrics, demonstrating its robustness and generalization across various datasets.

![Image text](https://github.com/Jtfboom/AugGCL/blob/master/AugGCL.png)

AugGCL is built based on PyTorch  
Tested on: **Ubuntu 20.04**, **NVIDIA RTX A6000 GPU**, **Intel i9-14900K** (3.30GHz, 20 cores), **125 GB RAM**, **CUDA environment (CUDA 12.8)**

## Requirements
Required modules can be installed via requirements.txt under the project root
```
pip install -r requirements.txt
```

```
torchvision==0.23.0
matplotlib==3.10.6
torch==2.8.0
seaborn==0.13.2
tqdm==4.67.1
numpy==2.2.6
anndata==0.11.4
pandas==2.3.2
rpy2==3.6.2
scanpy==1.11.4
scipy==1.15.3
scikit_learn==1.7.2
torch_geometric==2.0.4
```
## Installation

Download AugGCL:
```
git clone https://github.com/Jtfboom/AugGCL.git
```

## Dataset Setting
### 10x Visium 
The dataset can be download [here](https://github.com/LieberInstitute/HumanPilot/)
### Human Breast Cancer Data 
The dataset can be download [here](https://www.10xgenomics.com/resources/datasets/humanbreast-cancerblock-a-section-1-1-standard-1-1-0)
### Mouse Embryos at E9.5 Data
The dataset can be download [here](https://db.cngb.org/stomics/mosta/)

## Data folder structure

```
├── requirement.txt
├── dataset
│   └── DLPFC
│        └── 151507
│              ├── filtered_feature_bc_matrix.h5
│              ├── metadata.tsv 
│              ├── sampledata.h5ad
│              └── spatial
│                     ├── tissue_positions_list.csv  
│                     ├── full_image.tif  
│                     ├── tissue_hires_image.png  
│                     ├── tissue_lowres_image.png
```
## Tutorial for AugGCL

Follow the complete AugGCL process:

- **Data processing, training, and visualization**: [AugGCL full tutorial](https://github.com/Jtfboom/AugGCL/blob/master/Tutorial.ipynb)

Training Script
```
# go to /path/to/Sigra
# for 10x Visium dataset
python3 run_10x.py 

# for Human Breast Cancer dataset
python3 run_hbrc.py 

# for  Mouse Embryos at E9.5 dataset
python3 run_mouse.py 

```


