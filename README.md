# **AugGCL: Multimodal Graph Learning for Spatial Transcriptomics Analysis with Enhanced Gene and Morphological Data**

## Project Overview

**AugGCL** is a multimodal graph-convolutional learning framework designed to enhance spatial structure decoding and gene expression reconstruction by augmenting both gene and image data. The method utilizes the **Neighborhood Information Aggregation (NIA)** module, which integrates gene expression similarity and spatial proximity to construct a weighted graph and an enhanced expression matrix. This effectively addresses sparsity issues while maintaining boundary clarity. Additionally, AugGCL employs a two-stream weighted graph convolutional network (GCN) that jointly models gene features and image-derived morphological information, with image-aware auxiliary reconstructions enhancing weak spatial signals and sharpening boundaries.

On datasets from the human dorsolateral prefrontal cortex, breast cancer, and mouse embryo, AugGCL outperforms traditional baseline methods across multiple evaluation metrics, demonstrating its robustness and generalization across various datasets.

![Image text](https://github.com/Jtfboom/AugGCL/blob/master/AugGCL.png)


