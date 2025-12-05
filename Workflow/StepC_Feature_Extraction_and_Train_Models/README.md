# Step C: Feature Extraction and Model Training

This directory contains scripts for extracting features from histopathology images using foundation models and training XGBoost regression models for cell type proportion prediction.

## Overview

This step forms the core of the STPath-COAD framework, where multiple pre-trained foundation models are used to extract rich feature representations from H&E histopathology patches, followed by XGBoost model training for predicting cell type proportions.

## Scripts Overview

### 1. Feature Extraction

#### `Precompute_Features_Using_Foundation_Models_COAD.py`
- **Purpose**: Extracts features from COAD H&E patches using multiple foundation models
- **Foundation Models Used**:
  - **Conch**: Lu, Ming Y., et al. "A visual-language foundation model for computational pathology." Nature medicine 30.3 (2024): 863-874.
  - **UNI2h**: Chen, Richard J., et al. "Towards a general-purpose foundation model for computational pathology." Nature medicine 30.3 (2024): 850-862.
  - **ProvGigapath**: Xu, Hanwen, et al. "A whole-slide foundation model for digital pathology from real-world data." Nature 630.8015 (2024): 181-188.
  - **Virchow**: Vorontsov, Eugene, et al. "A foundation model for clinical-grade computational pathology and rare cancers detection." Nature medicine 30.10 (2024): 2924-2935.
  - **Virchow2**: Zimmermann, Eric, et al. "Virchow2: Scaling self-supervised mixed magnification models in pathology." arXiv preprint arXiv:2408.00738 (2024)
  - **ResNet50**: He, Kaiming, et al. "Deep Residual Learning for Image Recognition." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 770–778.
- **Features**:
  - Batch processing for efficiency
  - GPU acceleration support
  - Feature normalization and standardization
  - Progress tracking and logging
- **Input**: H&E image patches (240x240 pixels)
- **Output**: Feature matrices for each foundation model

### 2. Model Training and Prediction

#### `COAD_XGBoost_Prediction.py`
- **Purpose**: Trains XGBoost regression models for each cell type and performs predictions
- **Features**:
  - Multi-target regression (5 cell types)
  - Cross-validation and hyperparameter tuning
  - Feature importance analysis
  - Model performance evaluation
  - External validation on independent datasets
- **Models Trained**:
  - Cancer Cells prediction model
  - Normal Epithelial Cells prediction model
  - Stromal Cells prediction model
  - T Cells prediction model
  - Other Immune Cells prediction model
- **Input**: Extracted features + CARD deconvolution results (ground truth)
- **Output**: Trained XGBoost models and prediction results

#### `COAD_XGBoost_WithinSample.py`
- **Purpose**: Performs within-sample XGBoost analysis for internal validation
- **Features**:
  - Sample-level cross-validation
  - Internal consistency evaluation
  - Per-sample performance metrics
- **Input**: Extracted features + CARD deconvolution results
- **Output**: Within-sample prediction results and metrics


## Output Structure

### Feature Files
```
Features/
├── Tumor Cells_training_precomputed_features_Virchow.pt
├── Tumor Cells_training_precomputed_features_Virchow2.pt
├── Tumor Cells_training_precomputed_features_UNI2h.pt
├── ...
├── ...
├── Stromal Cells_training_precomputed_features_Virchow.pt
├── ...
├── ...
├── ...
└── ...
```

### Trained Models
```
STPath_COAD_Models_v1.0/
├── Trained_Models/
│   ├── xgboost_model_Cancer Cells_Combined_external_prediction.model
│   ├── xgboost_model_Normal Epithelial Cells_Combined_external_prediction.model
│   ├── xgboost_model_Stromal Cells_Combined_external_prediction.model
│   ├── xgboost_model_T Cells_Combined_external_prediction.model
│   └── xgboost_model_Other Immune Cells_Combined_external_prediction.model
└── Important_Features/
    ├── important_features_Cancer Cells.pkl
    ├── important_features_Normal Epithelial Cells.pkl
    ├── important_features_Stromal Cells.pkl
    ├── important_features_T Cells.pkl
    └── important_features_Other Immune Cells.pkl
```

## Dependencies

### Core Dependencies
- Python 3.8+
- PyTorch
- torchvision
- xgboost
- scikit-learn
- numpy
- pandas

### Foundation Model Dependencies
- transformers (HuggingFace)
- timm (PyTorch Image Models)
- conch (for Conch model)
- huggingface_hub

### Utility Dependencies
- tqdm (progress bars)
- matplotlib
- seaborn
- pickle

## Configuration

### Model Parameters
- **Batch Size**: 32 (adjustable based on GPU memory)
- **Image Size**: 240x240 pixels
- **Feature Dimensions**: Model-specific (512-2048)
- **Normalization**: ImageNet statistics

### XGBoost Parameters
- **Objective**: reg:squarederror
- **Learning Rate**: 0.1
- **Max Depth**: 6
- **Subsample**: 0.8
- **Colsample_bytree**: 0.8

## Performance Metrics

### Feature Quality
- **Feature Correlation**: Inter-model feature relationships
- **Discriminative Power**: Cell type separability
- **Stability**: Consistency across patches
- **Completeness**: Feature extraction success rate

### Model Performance
- **R² Score**: Coefficient of determination
- **RMSE**: Root mean squared error
- **MAE**: Mean absolute error
- **Correlation**: Pearson correlation with ground truth

## Troubleshooting

### Common Issues
1. **GPU Memory Errors**: Reduce batch size or use CPU fallback
2. **Model Loading Failures**: Check HuggingFace token permissions
3. **Feature Extraction Errors**: Verify image format and size
4. **Training Convergence**: Adjust XGBoost hyperparameters

### Performance Optimization
- Use GPU acceleration when available
- Implement batch processing for large datasets
- Cache extracted features to avoid recomputation
- Use parallel processing for model training

### Memory Management
- Monitor GPU memory usage during extraction
- Use gradient checkpointing for memory efficiency
- Clear model cache between different foundation models
- Implement data streaming for very large datasets
