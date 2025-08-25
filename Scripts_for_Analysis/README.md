# Scripts for Analysis

This directory contains scripts for generating all figures and performing analysis for the STPath-COAD manuscript.

## Main Figures

### Figure 2: Single-cell RNA-seq Data Analysis
- **Script**: `Figure2_scRNAseq_data_analysis.py`
- **Purpose**: Analyzes single-cell RNA-seq data, performs cell type annotation, and creates UMAP visualizations
- **Output**: Cell type annotations, marker genes, and visualization plots

### Figure 3: UMAP Analysis with Contribution Regression
- **Script**: `Figure3_UMAP_Contri_RegressOut.py`
- **Purpose**: Performs UMAP dimensionality reduction with regression of confounding factors
- **Output**: UMAP plots with contribution analysis and corrected embeddings
- **Note**: Also generates Figure S5-S8

### Figure 4: XGBoost Model Comparison
- **Script**: `Figure4_Xgboost_comparison.py`
- **Purpose**: Compares different XGBoost models and foundation model combinations
- **Output**: Performance comparison plots and statistical analysis

### Figure 5: Model Consistency Analysis
- **Script**: `Figure5_Consistency.py`
- **Purpose**: Analyzes consistency of STPath model predictions across different conditions
- **Output**: Consistency analysis plots and reliability metrics

### Figure 6: Soft Segmentation Analysis
- **Script**: `Figure6_Soft_Segmentation.py`
- **Purpose**: Performs soft tissue segmentation analysis for spatial transcriptomics data
- **Output**: Segmentation visualizations and analysis

### Figure 7: TCGA Dataset Analysis
- **Script**: `Figure7_TCGA_Analysis.py`
- **Purpose**: Analyzes TCGA colorectal cancer data using trained STPath models
- **Output**: External validation results and feature analysis

- **Script**: `Figure7_TCGA_Survival.py`
- **Purpose**: Performs survival analysis using TCGA data and STPath predictions
- **Output**: Survival curves and statistical analysis

## Supplementary Figures

### Figure S10: Gene Expression Prediction
- **Script**: `FigureS10_expression_prediction.py`
- **Purpose**: Predicts gene expression levels from histopathology features
- **Output**: Expression prediction accuracy and correlation analysis

## Tables

### Table 1: TCGA Summary Statistics
- **Script**: `Table1_TCGA.py`
- **Purpose**: Generates summary statistics and demographics for TCGA cohort
- **Output**: Patient characteristics table and molecular features

## Supplementary Files

### Supplementary File S3: Important Features
- **Script**: `Sup_file_S3_Make_Important_Features.py`
- **Purpose**: Extracts and processes important features from trained XGBoost models
- **Output**: Feature importance tables for all cell types and models

## Usage

Each script can be run independently after ensuring the required data and models are available. Make sure to:

1. Set up the correct data paths
2. Install required dependencies
3. Provide HuggingFace tokens where needed
4. Ensure trained models are available

## Dependencies

- Python 3.8+
- PyTorch
- scikit-learn
- matplotlib
- seaborn
- pandas
- numpy
- xgboost
- scanpy (for single-cell analysis)
- And other dependencies as specified in each script

## Data Requirements

- Trained XGBoost models (from StepC)
- CARD deconvolution results (from StepB)
- Processed spatial transcriptomics data
- TCGA histopathology images and clinical data
- Single-cell RNA-seq reference data
