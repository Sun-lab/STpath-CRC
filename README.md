# STPath-COAD: Spatial Transcriptomics Pathology for Colorectal Cancer

A comprehensive computational framework for predicting cell type proportions in colorectal cancer H&E images using multiple foundation models and machine learning approaches.

## Overview

This repository contains the complete analysis pipeline and scripts for the STPath-COAD project. The framework integrates multiple histopathology foundation models with XGBoost classifiers to predict spatial distributions of cell types in colorectal cancer tissues, validated against spatial transcriptomics data.

## Key Features

- **Multi-modal Foundation Models**: Integration of 6 state-of-the-art histopathology foundation models (ResNet50, Conch, UNI2-h, ProvGigaPath, Virchow, Virchow2)
- **Cell Type Deconvolution**: Spatial transcriptomics-guided cell type prediction for 5 major cell populations
- **TCGA Analysis**: Comprehensive survival analysis and clinical correlation studies
- **Cross-validation Framework**: Leave-one-individual-out (LOIO) validation strategy
- **Visualization Tools**: Hexagonal heatmaps, UMAP embeddings, and soft segmentation

## Foundation Models Used

This framework evaluates six state-of-the-art foundation models for histopathology:

1. **Conch**: Lu, Ming Y., et al. "A visual-language foundation model for computational pathology." *Nature medicine* 30.3 (2024): 863-874.

2. **UNI2h**: Chen, Richard J., et al. "Towards a general-purpose foundation model for computational pathology." *Nature medicine* 30.3 (2024): 850-862.

3. **ProvGigapath**: Xu, Hanwen, et al. "A whole-slide foundation model for digital pathology from real-world data." *Nature* 630.8015 (2024): 181-188.

4. **Virchow**: Vorontsov, Eugene, et al. "A foundation model for clinical-grade computational pathology and rare cancers detection." *Nature medicine* 30.10 (2024): 2924-2935.

5. **Virchow2**: Zimmermann, Eric, et al. "Virchow2: Scaling self-supervised mixed magnification models in pathology." *arXiv preprint arXiv:2408.00738* (2024)

6. **ResNet50**: He, Kaiming, et al. "Deep Residual Learning for Image Recognition." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2016, pp. 770–778.

## Cell Type Classification

The framework predicts proportions for 5 major cell types:

**Colorectal Cancer (COAD)**:
1. **Cancer Cells** - Malignant epithelial cells (colorectal carcinoma, adenoma, serrated-specific)
2. **Stromal Cells** - Fibroblasts and cancer-associated fibroblasts (CAFs)
3. **Normal Epithelial Cells** - Non-malignant epithelial cells (goblet cells, absorptive colonocytes, enteroendocrine cells, tuft cells)
4. **T Cells** - T lymphocytes (CD4+ and CD8+ T cells)
5. **pan-APC Cells** - B cells and Myeloid cells combined (antigen-presenting cells)

**Breast Cancer (BRCA)**:
1. **Tumor Cells** - Malignant epithelial cells
2. **Stromal Cells** - Fibroblasts and cancer-associated fibroblasts (CAFs)
3. **Normal Epithelial Cells** - Non-malignant epithelial cells
4. **T Cells** - T lymphocytes (CD4+ and CD8+ T cells)
5. **pan-APC Cells** - B cells and Myeloid cells combined (antigen-presenting cells)

## Repository Structure

```
STPath_COAD/
├── Scripts_for_Analysis/          # Scripts for generating manuscript figures
│   ├── Figure2_scRNAseq_data_analysis.py
│   ├── Figure3_UMAP_Contri_RegressOut.py
│   ├── Figure4_Xgboost_comparison.py
│   ├── Figure5_Consistency.py
│   ├── Figure6_Soft_Segmentation.py
│   ├── Figure7_TCGA_COAD_Survival.py
│   ├── FigureS7_TCGA_COAD_Analysis.py
│   ├── FigureS10_expression_prediction.py
│   ├── FigureS11_Cell_Type_Distribution_Analysis.py
│   ├── FigureS13_Compare_BRCA_COAD_Feature_Importance.py
│   ├── Table1_TCGA.py
│   └── Sup_file_S3_Make_Important_Features.py
│
├── Workflow/                      # Complete analysis workflow
│   ├── StepA_Data_Preparation/
│   │   ├── CARD_STData_Preparation_*.py
│   │   └── Create_patches_images_*.py
│   │
│   ├── StepB_Cell_Type_Deconvolution/
│   │   ├── CARD_Deconvolution_*.R
│   │   ├── CARD_Results_Validation_*.py
│   │   └── CARD_Results_Vis_Prepare_*.py
│   │
│   ├── StepC_Feature_Extraction_and_Train_Models/
│   │   ├── Precompute_Features_Using_Foundation_Models_COAD.py
│   │   ├── COAD_XGBoost_Prediction.py
│   │   ├── COAD_XGBoost_WithinSample.py
│   │   └── Compare_TIFF_JPG_Features.py
│   │
│   └── StepD_TCGA_Data_Preparation/
│       └── TCGA_COAD_DCM_to_TIFF.py
│
├── Figures/                       # Manuscript figures and supplementary figures
├── config_template.yaml           # Configuration template
├── README.md                      # This file
└── LICENSE                        # License information
```

## Workflow Overview

### Step A: Data Preparation
- Prepare spatial transcriptomics data for CARD deconvolution
- Extract H&E image patches at matched spatial locations
- Supports Cody, FredHutch, and HEST-1K datasets

### Step B: Cell Type Deconvolution
- Run CARD deconvolution using single-cell reference data
- Validate deconvolution results against known tissue regions
- Generate visualization of cell type spatial distributions

### Step C: Feature Extraction & Model Training
- Extract features using 6 foundation models
- Train XGBoost models with LOIO cross-validation
- Evaluate model performance and feature importance
- Generate calibrated predictions

### Step D: TCGA Data Processing
- Convert TCGA whole slide images from DCM to TIFF format
- Process TCGA-COAD cohort for validation studies
- Calculate distance metrics between cell types
- Perform survival analysis

## Requirements

### Software Dependencies
```
Python 3.8+
R 4.0+ (for CARD deconvolution)

Python packages:
- torch
- timm
- conch
- huggingface_hub
- xgboost
- numpy
- pandas
- matplotlib
- seaborn
- scanpy
- lifelines
- scipy
- scikit-learn
- umap-learn
- tqdm
- pillow
- opencv-python

R packages:
- CARD
- Seurat
```

### Hardware Requirements
- **Recommended**: GPU with 16GB+ VRAM (for foundation model feature extraction)
- **Minimum**: CPU with 32GB RAM
- **Storage**: 100GB+ for data, models, and intermediate results

### Data Requirements
1. **Spatial Transcriptomics Data**: Spot-level gene expression and spatial coordinates
2. **Single-cell Reference**: Annotated scRNA-seq data for CARD deconvolution
3. **H&E Images**: Whole slide images or tissue regions
4. **TCGA Data** (optional): For validation and survival analysis

## Getting Started

### 1. Environment Setup

```bash
# Create conda environment
conda create -n stpath python=3.9
conda activate stpath

# Install Python dependencies
pip install torch torchvision torchaudio
pip install timm transformers huggingface_hub
pip install xgboost scikit-learn
pip install numpy pandas matplotlib seaborn
pip install scanpy lifelines
pip install umap-learn tqdm pillow opencv-python
```

### 2. Configuration

Copy and edit the configuration template:
```bash
cp config_template.yaml config.yaml
```

Edit paths and parameters according to your data location and computing resources.

### 3. Running the Analysis

#### Complete Workflow
```bash
# Step A: Data preparation
python Workflow/StepA_Data_Preparation/CARD_STData_Preparation_Cody.py

# Step B: Cell type deconvolution (in R)
Rscript Workflow/StepB_Cell_Type_Deconvolution/CARD_Deconvolution_Cody.R

# Step C: Feature extraction and model training
python Workflow/StepC_Feature_Extraction_and_Train_Models/Precompute_Features_Using_Foundation_Models_COAD.py
python Workflow/StepC_Feature_Extraction_and_Train_Models/COAD_XGBoost_Prediction.py

# Step D: TCGA analysis
python Workflow/StepD_TCGA_Data_Preparation/TCGA_COAD_DCM_to_TIFF.py
python Scripts_for_Analysis/FigureS7_TCGA_COAD_Analysis.py
python Scripts_for_Analysis/Figure7_TCGA_COAD_Survival.py
```

#### Generate Manuscript Figures
```bash
# Main figures
python Scripts_for_Analysis/Figure2_scRNAseq_data_analysis.py
python Scripts_for_Analysis/Figure3_UMAP_Contri_RegressOut.py
python Scripts_for_Analysis/Figure4_Xgboost_comparison.py
python Scripts_for_Analysis/Figure5_Consistency.py
python Scripts_for_Analysis/Figure6_Soft_Segmentation.py
python Scripts_for_Analysis/Figure7_TCGA_COAD_Survival.py

# Supplementary figures and tables
python Scripts_for_Analysis/FigureS7_TCGA_COAD_Analysis.py
python Scripts_for_Analysis/FigureS10_expression_prediction.py
python Scripts_for_Analysis/FigureS11_Cell_Type_Distribution_Analysis.py
python Scripts_for_Analysis/Table1_TCGA.py
```

## Key Analyses

### Foundation Model Comparison
- Systematic comparison of 6 foundation models (Figure 4)
- Feature importance analysis across models
- Model complementarity assessment

### Spatial Analysis
- Cell type distance calculations using weighted minimum distance
- Hard classification based on quantile thresholds
- Hexagonal heatmap visualization

### Survival Analysis
- Cox proportional hazards models
- Kaplan-Meier survival curves
- Integration of cell type proportions, spatial metrics, and clinical variables

### Gene Expression Prediction
- Correlation between histopathology features and gene expression
- Cell type-specific marker gene analysis
- Partial correlation controlling for cell type composition

## Model Training Strategy

### Leave-One-Individual-Out (LOIO) Cross-Validation
- Each individual (patient) is held out once as test set
- Models trained on all other individuals
- Ensures generalization across patients
- Prevents overfitting to individual-specific patterns

### Calibration
- Quantile-based calibration for improved prediction accuracy
- Cell type-specific outlier handling
- Preservation of rank ordering while adjusting scale

### Feature Selection
- Top 30% most important features selected per model
- Combined feature set from multiple foundation models
- Reduces dimensionality while maintaining predictive power

## Output Files

### Model Predictions
- CSV files with patch-level predictions
- JSON files with metadata and overall proportions
- Feature importance scores

### Visualizations
- UMAP embeddings colored by various factors
- Hexagonal heatmaps for spatial distributions
- Violin plots for cell type proportions
- Scatter plots for model comparisons
- Kaplan-Meier survival curves

### Statistical Results
- Cox regression tables
- Model performance metrics (Spearman correlation, MAE, RMSE)
- Feature importance rankings

## Performance Metrics

### Model Evaluation
- **Spearman Correlation**: Primary metric for proportion prediction
- **Mean Absolute Error (MAE)**: Absolute prediction error
- **Root Mean Squared Error (RMSE)**: Squared error magnitude

### Typical Performance (LOIO validation)
- Cancer Cells: ρ = 0.75-0.85
- Stromal Cells: ρ = 0.70-0.80
- T Cells: ρ = 0.60-0.70
- Other cell types: ρ = 0.55-0.70

## Computational Resources

### Processing Time
- **Feature Extraction**: ~2-5 minutes per WSI (GPU)
- **Model Training**: ~30-60 minutes per cell type (LOIO)
- **Prediction**: ~1-3 minutes per WSI
- **TCGA Analysis**: ~10-20 minutes per sample

### Memory Requirements
- **Feature Extraction**: 16GB GPU VRAM recommended
- **Model Training**: 32GB RAM minimum
- **Large WSI Processing**: 64GB RAM recommended

## Citation

If you use this code in your research, please cite:

```
[Citation information to be added upon publication]
```

## Related Repositories

- **STPath-Software**: Standalone prediction tool for clinical use (separate repository)
- **BRCA Analysis**: Breast cancer application scripts (see `/Users/scui2/Desktop/BRCA_github/`)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, issues, or collaborations, please open an issue on GitHub or contact the authors.

## Acknowledgments

- Foundation models: Conch, UNI, ProvGigaPath, Virchow, Virchow2
- CARD deconvolution method
- TCGA consortium for data access
- All data contributors and collaborators

---

**Last Updated**: December 2025
