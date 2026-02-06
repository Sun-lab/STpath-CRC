# Step C: Feature Extraction and Model Training

This directory contains scripts for extracting features from histopathology images using foundation models and training machine learning models (primarily XGBoost) for cell type proportion prediction and gene expression prediction.

## Overview

This step forms the core of the STPath framework, where pre-trained foundation models extract rich feature representations from H&E histopathology patches, followed by supervised learning for predicting:
1. **Cell type proportions** - Spatial distribution of 5 major cell types
2. **Gene expression levels** - Marker genes and highly variable genes
3. **pan-APC subtypes** - B cells and Myeloid cells prediction

The trained models enable inference on new H&E images without requiring spatial transcriptomics data.

## Foundation Models Used

This framework evaluates six state-of-the-art foundation models for histopathology:

1. **Conch**: Lu, Ming Y., et al. "A visual-language foundation model for computational pathology." *Nature medicine* 30.3 (2024): 863-874.

2. **UNI2h**: Chen, Richard J., et al. "Towards a general-purpose foundation model for computational pathology." *Nature medicine* 30.3 (2024): 850-862.

3. **ProvGigapath**: Xu, Hanwen, et al. "A whole-slide foundation model for digital pathology from real-world data." *Nature* 630.8015 (2024): 181-188.

4. **Virchow**: Vorontsov, Eugene, et al. "A foundation model for clinical-grade computational pathology and rare cancers detection." *Nature medicine* 30.10 (2024): 2924-2935.

5. **Virchow2**: Zimmermann, Eric, et al. "Virchow2: Scaling self-supervised mixed magnification models in pathology." *arXiv preprint arXiv:2408.00738* (2024)

6. **ResNet50**: He, Kaiming, et al. "Deep Residual Learning for Image Recognition." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2016, pp. 770–778.

## Scripts Overview

### 1. Feature Extraction

#### `Precompute_Features_Using_Foundation_Models_COAD.py`
Extracts features from COAD H&E patches using multiple foundation models.

**Key Features**:
- Batch processing for computational efficiency
- GPU acceleration with automatic memory management
- Feature normalization and standardization
- Progress tracking and error handling
- Support for all 6 foundation models

**Input**: 
- H&E image patches (240×240 pixels)
- Spatial coordinates and metadata

**Output**: 
- Feature tensors (`.pt` files) for each foundation model
- Feature dimensions: 512-2048 depending on model
- Organized by cell type for supervised learning

---

### 2. Cell Type Proportion Prediction

#### `COAD_XGBoost_Prediction.py`
Trains XGBoost regression models for predicting 5 cell type proportions.

**Key Features**:
- Leave-one-individual-out cross-validation
- Tile-level and individual-level performance evaluation
- Hyperparameter optimization
- Feature importance analysis
- Model calibration for improved accuracy
- External validation on independent datasets

**Cell Types Predicted**:
1. Cancer Cells (malignant epithelial cells)
2. Stromal Cells (fibroblasts, CAFs)
3. Normal Epithelial Cells (non-malignant epithelial)
4. T Cells (T lymphocytes)
5. pan-APC Cells (B cells + Myeloid cells combined)

**Input**: 
- Precomputed foundation model features (from Step 1)
- CARD deconvolution results as ground truth (from StepB)
- Spatial coordinates and sample metadata

**Output**:
- Trained XGBoost models (`.model` files)
- Prediction results (`.pt` files with predictions and metrics)
- Feature importance scores (`.pkl` files)
- Performance plots (R², MAE, correlation)
- Individual-level metrics (`.csv` files)

**Performance Metrics**:
- Pearson correlation coefficient (R)
- R² (coefficient of determination)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

---

#### `COAD_XGBoost_WithinSample.py`
Performs within-sample cross-validation for internal validation.

**Key Features**:
- 80/20 train-test split within each sample
- Sample-level performance evaluation
- Internal consistency assessment
- Naive baseline comparison

**Purpose**: 
Evaluates model performance in simpler within-sample setting (easier task) compared to leave-one-individual-out cross-validation (harder, more realistic task).

**Input**: Same as `COAD_XGBoost_Prediction.py`

**Output**: 
- Within-sample prediction results
- Baseline performance metrics
- Comparison with cross-individual predictions

---

### 3. Immune Cell Subtype Analysis

#### `COAD_B_MYE_XGBoost_Prediction.py`
Trains separate XGBoost models for B cells and Myeloid cells prediction.

**Key Features**:
- Splits "pan-APC Cells" features into B cells and Myeloid cells
- Separate model training for each immune subtype
- Validates pan-APC grouping strategy
- Feature importance comparison

**Purpose**: 
Demonstrates that combining B cells and Myeloid cells into pan-APC improves prediction accuracy compared to predicting them separately (see Figure S1).

**Input**: 
- pan-APC features (from Step 1)
- Separate B cell and Myeloid cell proportions from CARD

**Output**:
- B cell prediction models and results
- Myeloid cell prediction models and results
- Comparison metrics with pan-APC models

---

### 4. Gene Expression Prediction

#### `COAD_GeneExpression_Prediction.py`
Trains XGBoost models to predict gene expression from histopathology features.

**Key Features**:
- Predicts 180 genes (marker genes + highly variable genes)
- Cell type marker genes:
  - Tumor markers (e.g., CEACAM5, KRT20)
  - Normal epithelial markers (e.g., CA1, CLCA1)
  - T cell markers (e.g., CD3D, CD8A)
  - pan-APC markers (e.g., CD79A, LYZ)
  - Stromal markers (e.g., COL1A1, DCN)
- Highly variable genes for tissue characterization
- Leave-one-individual-out cross-validation
- Partial correlation analysis (controlling for cell type proportions)

**Purpose**: 
Demonstrates that foundation model features capture not only cell type proportions but also gene expression patterns, validating the biological relevance of learned representations.

**Input**:
- Foundation model features (from Step 1)
- Spatial transcriptomics gene expression data
- Cell type proportions from CARD (for partial correlation)

**Output**:
- Gene expression prediction results (`.pt` files)
- Regular correlation coefficients (`.csv` files)
- Partial correlation coefficients (`.csv` files)
- Trained models for all 180 genes

---

#### `COAD_GeneExpression_Prediction_Figures.py`
Generates visualizations for gene expression prediction results.

**Key Features**:
- Correlation scatter plots for all genes
- Foundation model comparison boxplots
- Virchow2 vs ResNet50 comparison
- Gene-specific deep dive analysis
- Partial correlation vs regular correlation comparison

**Purpose**: 
Creates publication-ready figures showing gene expression prediction performance across different foundation models and gene categories.

**Input**: 
- Results from `COAD_GeneExpression_Prediction.py`

**Output**:
- High-resolution figures (`.png`, `.pdf`)
- Summary statistics tables



---

### 5. Alternative Machine Learning Models

#### `COAD_Other_ML_Models_Prediction.py`
Trains alternative ML models for comparison with XGBoost.

**Models Implemented**:
1. **Lasso Regression** - L1-regularized linear regression
2. **Multi-Layer Perceptron (MLP)** - Neural network with 2 hidden layers
3. **Random Forest** - Ensemble of decision trees

**Key Features**:
- Grid search for hyperparameter optimization
- Individual-level cross-validation
- Comparison with XGBoost baseline
- Training for Cancer Cells and Stromal Cells with Virchow2

**Purpose**: 
Benchmarks XGBoost against other popular ML methods to justify model choice.

**Input**: 
- Precomputed foundation model features
- CARD deconvolution results

**Output**:
- Trained models for Lasso, MLP, Random Forest
- Performance metrics comparing all methods
- Individual-level predictions



---


## Configuration and Parameters

### Feature Extraction Parameters
- **Image Size**: 240×240 pixels
- **Batch Size**: 32 (adjust based on GPU memory)
- **Feature Dimensions**:
  - Conch: 512
  - UNI2h: 1024
  - ProvGigapath: 1536
  - Virchow: 2048
  - Virchow2: 2048
  - ResNet50: 2048
- **Normalization**: ImageNet statistics

### XGBoost Hyperparameters
```python
params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'n_estimators': 800
}
```

### Cross-Validation Strategy
- **Individual-level CV**: Leave-one-individual-out (more challenging, more realistic)
- **Within-sample CV**: 80/20 split within each sample (baseline)
- **Evaluation**: Both tile-level and individual-level metrics

---

## Dependencies

### Core Libraries
```bash
# Python environment
Python 3.8+

# Deep learning
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0

# Machine learning
xgboost>=1.7.0
scikit-learn>=1.2.0
scipy>=1.10.0

# Data processing
numpy>=1.23.0
pandas>=1.5.0

# Visualization
matplotlib>=3.6.0
seaborn>=0.12.0

# Utilities
tqdm>=4.65.0
pillow>=9.4.0
pickle
```

### Foundation Model Dependencies
```bash
# HuggingFace
transformers>=4.30.0
huggingface_hub>=0.15.0

# CONCH model
pip install git+https://github.com/mahmoodlab/CONCH.git

# UNI model
# (Follow installation from official repository)

# ProvGigapath
# (Follow installation from official repository)

# Virchow/Virchow2
# (Follow installation from official repository)
```

### Installation
```bash
# Create conda environment
conda create -n stpath python=3.8
conda activate stpath

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# Install foundation models (follow each model's installation guide)
```

---

## Troubleshooting

### Common Issues

**1. GPU Memory Errors**
- **Symptom**: CUDA out of memory during feature extraction
- **Solution**: 
  - Reduce batch size: `batch_size = 16` or `batch_size = 8`
  - Use CPU fallback: `device = 'cpu'`
  - Clear cache: `torch.cuda.empty_cache()`

**2. HuggingFace Authentication**
- **Symptom**: Cannot download foundation models
- **Solution**: 
  - Login with token: `huggingface-cli login`
  - Check model access permissions
  - Verify token is valid

**3. Feature Dimension Mismatch**
- **Symptom**: Error during model training about feature dimensions
- **Solution**: 
  - Re-extract features with correct model
  - Check feature file integrity
  - Verify model names match between extraction and training

**4. Cross-Validation Errors**
- **Symptom**: Not enough samples for CV
- **Solution**: 
  - Ensure at least 3 individuals in dataset
  - Check sample metadata is correct
  - Verify individual IDs are properly formatted

**5. Convergence Issues**
- **Symptom**: Model does not converge or poor performance
- **Solution**: 
  - Adjust learning rate: `learning_rate = 0.05`
  - Increase number of trees: `n_estimators = 1000`
  - Check for NaN values in features or labels
  - Normalize features if not already done

### Performance Optimization

**GPU Acceleration**
- Always use GPU for feature extraction (10-20× faster)
- Use mixed precision if available: `torch.cuda.amp`
- Monitor GPU memory usage: `nvidia-smi`

**Batch Processing**
- Process multiple samples in parallel
- Use multiprocessing for CPU-bound tasks
- Cache features to avoid recomputation

**Memory Management**
- Use gradient checkpointing for large models
- Clear model cache between different foundation models
- Stream data for very large datasets
- Save intermediate results frequently

### Memory Requirements

| Task | CPU RAM | GPU VRAM | Storage |
|------|---------|----------|---------|
| Feature Extraction | 16GB | 16GB+ | 50-100GB |
| XGBoost Training | 32GB | N/A | 10-20GB |
| Gene Expression | 32GB | N/A | 20-30GB |
| Full Pipeline | 64GB | 24GB+ | 100-200GB |

---

## Expected Runtime

| Task | Runtime (GPU) | Runtime (CPU) |
|------|---------------|---------------|
| Feature extraction (1 sample) | 5-10 min | 30-60 min |
| Feature extraction (all samples) | 2-4 hours | 10-20 hours |
| XGBoost training (1 cell type) | 10-30 min | 20-60 min |
| XGBoost training (all cell types) | 1-2 hours | 3-5 hours |
| Gene expression prediction | 4-8 hours | 12-24 hours |
| Complete pipeline | 6-12 hours | 24-48 hours |

*Timings based on ~10-15 samples with ~2000-3000 spots each*

---


## Citation

If you use these scripts or trained models, please cite:

```bibtex
@article{stpath2026,
  title={STPath: Cell Type Deconvolution from Histopathology using Foundation Models},
  author={Cui, Saishi and others},
  journal={[Journal Name]},
  year={2026}
}
```

## Contact

For questions or issues:
- Saishi Cui
- Open an issue on GitHub
- Check documentation in other workflow steps
