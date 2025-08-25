# STPath-COAD Workflow

## Overview

This folder contains the complete computational workflow for STPath-COAD (from data preparation to model training and validation). The workflow is organized into sequential steps that process raw data into trained models for cell type proportion prediction.

## Workflow Overview

![Figure 1: STPath-COAD Schematic](../Figures/Figure1_Schematic.png)

**Figure 1.** Overall schematic of the STPath-COAD workflow.

## Supplementary Methods Overview

### Data Processing Pipeline

![Figure S1: Data Processing 1](../Figures/Figure_S1.png)

**Figure S1.** 

![Figure S2: Data Processing 2](../Figures/Figure_S2.png)

**Figure S2.** 

![Figure S3: Data Processing 3](../Figures/Figure_S3.png)


## Dependencies

### Python Dependencies
- `scanpy`: Single-cell analysis
- `pandas`, `numpy`: Data manipulation
- `torch`, `torchvision`: Deep learning
- `xgboost`: Machine learning
- `PIL`, `cv2`: Image processing
- `matplotlib`, `seaborn`: Visualization

### R Dependencies
- `CARD`: Spatial deconvolution
- `Seurat`: Single-cell analysis
- `dplyr`, `tidyr`: Data manipulation

### Foundation Models
- **Conch**: Requires HuggingFace access
- **UNI2h**: Requires HuggingFace access  
- **ProvGigapath**: Requires HuggingFace access
- **Virchow/Virchow2**: Requires HuggingFace access


