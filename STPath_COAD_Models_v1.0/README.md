# STPath-COAD Models v1.0

This directory contains the trained models and important features for the STPath-COAD (Spatial Transcriptomics Pathology - Colorectal Adenocarcinoma) framework.

## Overview

These pre-trained models enable prediction of cell type proportions in colorectal cancer H&E histopathology images. The models were trained using features extracted from multiple foundation models combined with CARD deconvolution ground truth labels.

## Directory Structure

```
STPath_COAD_Models_v1.0/
├── Trained_Models/           # XGBoost regression models
└── Important_Features/       # Feature importance rankings
```

## Trained Models

### XGBoost Regression Models

All models are trained using XGBoost regression with the selected important features from multiple foundation models (Conch, UNI2h, ProvGigapath, Virchow, Virchow2).

#### `xgboost_model_Cancer Cells_Combined_external_prediction.model`
- **Purpose**: Predicts Cancer cells (Carcinoma-specific cells, Adenoma-specific cells, Serrated-specific cells) proportions in tissue regions


#### `xgboost_model_Normal Epithelial Cells_Combined_external_prediction.model`
- **Purpose**: Predicts Normal epithelial cells (Tuft, Goblet, Enteroendocrine, Absorptive colonocytes, Crypt-top colonocytes) proportions in tissue regions


#### `xgboost_model_Stromal Cells_Combined_external_prediction.model`
- **Purpose**: Predicts Stromal cells (Fibroblasts, Endothelial) proportions in tissue regions

#### `xgboost_model_T Cells_Combined_external_prediction.model`
- **Purpose**: Predicts T cells (CD4+ and CD8+) proportions in tissue regions

#### `xgboost_model_Other Immune Cells_Combined_external_prediction.model`
- **Purpose**: Predicts Other immune cells (B, Myeloid, Mast) proportions in tissue regions


## Important Features

### Feature Importance Files

Each cell type has an associated feature importance file that ranks the most predictive features from the foundation models.

#### `important_features_Cancer Cells.pkl`
- **Number of features**: Conch: Top 154 features
                          ProvGigapath: Top 461 features
                          Virchow: Top 768 features
                          Virchow2: Top 768 features
                          UNI2h: Top 461 features
                          Total: 2552 features

#### `important_features_Normal Epithelial Cells.pkl`
- **Number of features**: Conch: Top 154 features
                          ProvGigapath: Top 461 features
                          Virchow: Top 768 features
                          Virchow2: Top 768 features
                          UNI2h: Top 461 features
                          Total: 2552 features

#### `important_features_Stromal Cells.pkl`
- **Number of features**: Conch: Top 154 features
                          ProvGigapath: Top 461 features
                          Virchow: Top 768 features
                          Virchow2: Top 768 features
                          UNI2h: Top 461 features
                          Total: 2552 features

#### `important_features_T Cells.pkl`
- **Number of features**: Conch: Top 154 features
                          ProvGigapath: Top 461 features
                          Virchow: Top 768 features
                          Virchow2: Top 768 features
                          UNI2h: Top 461 features
                          Total: 2552 features

#### `important_features_Other Immune Cells.pkl`
- **Number of features**: Conch: Top 154 features
                          ProvGigapath: Top 461 features
                          Virchow: Top 768 features
                          Virchow2: Top 768 features
                          UNI2h: Top 461 features
                          Total: 2552 features



## Support and Updates

For questions, bug reports, or feature requests:
- Create an issue on the GitHub repository
- Check documentation for troubleshooting
- Contact the development team for collaboration

Models are regularly updated with improved performance and new features. Check the repository for the latest versions and updates.
