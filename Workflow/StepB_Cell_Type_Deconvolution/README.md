# Step B: Cell Type Deconvolution

This directory contains scripts for performing cell type deconvolution using CARD (Conditional AutoRegressive-based Deconvolution) and validating the results.

## Overview

Cell type deconvolution is performed using CARD, which estimates cell type proportions in spatial transcriptomics spots by leveraging single-cell RNA-seq reference data. This step is crucial for providing ground truth labels for training the STPath prediction models.

## Scripts Overview

### 1. CARD Deconvolution (R Scripts)

#### `CARD_Deconvolution_Cody.R`
- **Purpose**: Performs CARD deconvolution on the main Cody's dataset
- **Features**:
  - Single-cell reference integration
  - Spatial constraint enforcement
  - Cell type proportion estimation
  - Quality control and filtering
- **Input**: Expression matrices, spatial coordinates, scRNA-seq reference
- **Output**: Cell type proportion estimates for each spot

#### `CARD_Deconvolution_FH.R`
- **Purpose**: CARD deconvolution for Fred Hutchinson dataset
- **Features**: Dataset-specific parameter optimization
- **Input**: FH spatial transcriptomics data and reference
- **Output**: FH-specific cell type proportions

#### `CARD_Deconvolution_HEST.R`
- **Purpose**: CARD deconvolution for HEST-1K dataset
- **Features**: Large-scale processing for HEST data
- **Input**: HEST spatial data and colorectal cancer reference
- **Output**: Deconvolution results for HEST samples

### 2. Results Validation (Python Scripts)

#### `CARD_Results_Validation_Cody.py`
- **Purpose**: Validates CARD deconvolution results for Cody's dataset
- **Features**:
  - Statistical validation metrics
  - Correlation analysis
  - Spatial consistency checks
  - Quality assessment plots
- **Input**: CARD output files
- **Output**: Validation metrics and visualization

#### `CARD_Results_Validation_FH.py`
- **Purpose**: Validates Fred Hutchinson CARD results
- **Features**: FH-specific validation and quality control
- **Input**: FH CARD outputs
- **Output**: FH validation reports

#### `CARD_Results_Validation_HEST.py`
- **Purpose**: Validates HEST CARD deconvolution results
- **Features**: Large-scale validation for HEST dataset
- **Input**: HEST CARD outputs
- **Output**: HEST validation metrics

#### `CARD_Results_Validation_Combined.py`
- **Purpose**: Combined validation across all datasets
- **Features**:
  - Cross-dataset comparison
  - Meta-analysis of results
  - Comprehensive quality assessment
- **Input**: All CARD results
- **Output**: Combined validation report

### 3. Visualization Preparation

#### `CARD_Results_Vis_Prepare_Cody.py`
- **Purpose**: Prepares Cody's CARD results for visualization
- **Features**:
  - Data formatting for plotting
  - Color scheme optimization
  - Spatial layout preparation
- **Input**: Validated CARD results
- **Output**: Visualization-ready data

#### `CARD_Results_Vis_Prepare_FH.py`
- **Purpose**: Prepares Fred Hutchinson's results for visualization
- **Features**: FH-specific visualization formatting
- **Input**: FH validated results
- **Output**: FH visualization data

#### `CARD_Results_Vis_Prepare_HEST.py`
- **Purpose**: Prepares HEST-1K's results for visualization
- **Features**: Large-scale visualization preparation
- **Input**: HEST validated results
- **Output**: HEST visualization data

## CARD Deconvolution Pipeline

### 1. Data Preparation
```
Expression Matrix + Spatial Coordinates + scRNA Reference → CARD Input
```

### 2. Deconvolution Process
```
CARD Input → Spatial Modeling → Cell Type Estimation → Proportion Output
```

### 3. Validation and QC
```
CARD Output → Statistical Validation → Quality Control → Validated Results
```

### 4. Visualization Preparation
```
Validated Results → Format Processing → Visualization Data → Ready for Plotting
```

## Key Features

### CARD Deconvolution
- **Spatial Awareness**: Incorporates spatial relationships between spots
- **Reference Integration**: Uses scRNA-seq data for cell type signatures
- **Regularization**: Spatial smoothing and biological constraints
- **Scalability**: Handles large spatial transcriptomics datasets

### Validation Metrics
- **Correlation Analysis**: Between predicted and expected proportions
- **Spatial Consistency**: Neighboring spot similarity
- **Biological Plausibility**: Cell type distribution patterns
- **Statistical Significance**: P-values and confidence intervals

### Quality Control
- **Spot Filtering**: Removes low-quality spatial spots
- **Proportion Constraints**: Ensures proportions sum to 1
- **Outlier Detection**: Identifies and handles anomalous results
- **Convergence Checking**: Validates CARD model convergence


## Output Structure

### CARD Results
```
CARD_Results/
├── Cody/
│   ├── sample1_region0_proportions.csv
│   ├── sample1_region1_proportions.csv
│   └── ...
├── FH/
└── HEST/
```

### Validation Reports
```
Validation_Results/
├── correlation_analysis.pdf
├── spatial_consistency.pdf
├── quality_metrics.csv
└── validation_summary.txt
```

## Dependencies

### R Dependencies
- CARD
- Seurat
- Matrix
- dplyr
- ggplot2

### Python Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn

## Cell Types

The deconvolution identifies the following cell types:
- **Cancer Cells**: Carcinoma-specific cells, Adenoma-specific cells, Serrated-specific cells
- **Normal Epithelial Cells**: Tuft, Goblet, Enteroendocrine, Absorptive colonocytes, Crypt-top colonocytes
- **Stromal Cells**: Fibroblasts, Endothelial cells
- **T Cells**: CD4+ and CD8+ T cells
- **Other Immune Cells**: B cells, Plasma cells, Myeloid cells, Mast cells



