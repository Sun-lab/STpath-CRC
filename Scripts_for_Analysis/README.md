# Scripts for Analysis

This directory contains scripts for generating all figures, supplementary figures, tables, and supplementary files for the STPath manuscript. These scripts analyze the results from trained models and generate publication-ready visualizations.

## Main Figures

### Figure 2: Single-cell RNA-seq Data Analysis
- **Script**: `Figure2_scRNAseq_data_analysis.py`
- **Purpose**: Analyzes single-cell RNA-seq data for colorectal cancer, performs cell type annotation and deconvolution validation
- **Key Features**:
  - Single-cell data preprocessing and quality control
  - Cell type clustering and annotation
  - UMAP visualization of cell populations
  - Marker gene identification and validation
  - Cell type deconvolution benchmarking
- **Output**: Figure 2 panels A-E (UMAP plots, marker gene heatmaps, deconvolution validation)
- **Data Required**: VUMC single-cell RNA-seq data (h5ad format)

### Figure 3: UMAP Analysis with Contribution Regression
- **Script**: `Figure3_UMAP_Contri_RegressOut.py`
- **Purpose**: Performs UMAP dimensionality reduction of foundation model features with regression of confounding factors
- **Key Features**:
  - Feature extraction from trained foundation models
  - Batch effect and technical variation regression
  - UMAP visualization of tissue morphology
  - Contribution analysis of different factors
- **Output**: Figure 3 (UMAP plots with and without regression, contribution analysis)
- **Models Used**: All 5 foundation models (Conch, ProvGigapath, Virchow, Virchow2, UNI2h)

### Figure 4: XGBoost Model Comparison for Colorectal Cancer
- **Script**: `Figure4_Xgboost_comparison_Colorectal.py`
- **Purpose**: Compares cell type proportion prediction performance across different foundation models for COAD
- **Key Features**:
  - Model performance comparison (R², MAE)
  - Cell type-specific prediction accuracy
  - Foundation model benchmark analysis
  - Spatial visualization of predictions
  - Gene expression prediction validation
- **Output**: Figure 4 panels A-C (performance metrics, spatial predictions, correlation plots)
- **Cell Types**: Cancer Cells, Stromal Cells, Normal Epithelial Cells, T Cells, pan-APC Cells

### Figure 5: Model Consistency Analysis
- **Script**: `Figure5_Consistency_Analysis.py`
- **Purpose**: Analyzes consistency and robustness of STPath model predictions across different conditions
- **Key Features**:
  - Prediction reproducibility across different regions
  - Model sensitivity to image perturbations
  - Cross-sample consistency metrics
  - Reliability analysis for clinical applications
- **Output**: Figure 5 (consistency plots, reliability metrics, robustness analysis)

### Figure 6: TCGA COAD Analysis
- **Script**: `Figure6_TCGA_COAD_Analysis.py`
- **Purpose**: Applies STPath to TCGA COAD whole slide images for large-scale cell type quantification
- **Key Features**:
  - Whole slide image processing and tile extraction
  - Cell type proportion prediction for TCGA cohort
  - Hexagonal spatial heatmap generation
  - Distance-to-tumor analysis
  - Clinical correlation analysis
- **Output**: Figure 6 (hexagonal heatmaps, cell type distributions, clinical associations)
- **Data**: TCGA COAD whole slide images (n=276 patients)

### Figure 7: XGBoost Model Comparison for Breast Cancer
- **Script**: `Figure7_Xgboost_comparison_Breast.py`
- **Purpose**: Compares cell type proportion prediction performance across different foundation models for BRCA
- **Key Features**:
  - Cross-cancer validation of STPath framework
  - Model performance comparison for breast cancer
  - Cell type-specific prediction accuracy for BRCA
  - Foundation model generalization analysis
- **Output**: Figure 7 (performance metrics for BRCA, spatial predictions)
- **Cell Types**: Tumor Cells, Stromal Cells, Normal Epithelial Cells, T Cells, pan-APC Cells

## Supplementary Figures

### Figure S1: pan-APC Analysis and Gene Expression Prediction
- **Script**: `FigureS1.py`
- **Purpose**: Validates pan-APC (B cells + Myeloid cells) grouping strategy and demonstrates gene expression prediction capability
- **Key Features**:
  - Comparison of separate vs. combined immune cell prediction
  - Gene expression prediction from histopathology features
  - Partial correlation analysis between cell types and gene expression
  - Model contribution analysis
- **Output**: Figure S1 panels A-D (pan-APC comparison, expression prediction, correlation analysis)

### Figure S2-S4: Cross-reference Figures
- **Script**: `FigureS2-S4.py`
- **Purpose**: Documentation script referencing other scripts for supplementary figures S2-S4
- **Content**:
  - Figure S2: See `Figure3_UMAP_Contri_RegressOut.py`
  - Figure S3: See `Figure5_Consistency_Analysis.py`
  - Figure S4: See `FigureS6_Quant_Soft_Segmentation.py`

### Figure S6: Quantitative Soft Segmentation Analysis
- **Script**: `FigureS6_Quant_Soft_Segmentation.py`
- **Purpose**: Performs probabilistic soft tissue segmentation analysis for spatial transcriptomics data
- **Key Features**:
  - Soft segmentation of tissue regions
  - Probabilistic cell type assignment
  - Spatial uncertainty quantification
  - Comparison with hard segmentation approaches
- **Output**: Figure S6 (soft segmentation visualizations, uncertainty maps)

### Figure S8: Naive Cross-Validation Performance
- **Script**: `FigureS8_Naive_CV_performance.py`
- **Purpose**: Evaluates naive cross-validation performance using 80/20 train-test split
- **Key Features**:
  - Within-sample cross-validation (80% train, 20% test)
  - Performance metrics for all 5 cell types
  - Comparison across all foundation models
  - Baseline performance benchmarking
- **Output**: Figure S8 (naive CV performance metrics, comparison plots)
- **Models**: All 6 foundation models (Conch, ProvGigapath, Virchow, Virchow2, UNI2h, ResNet50)

## Supplementary Files

### Supplementary File 3: Important Features for Colorectal Cancer
- **Script**: `Sup_file3_Make_Important_Features_Colorectal.py`
- **Purpose**: Extracts and processes important features from trained XGBoost models for COAD
- **Key Features**:
  - Feature importance calculation from XGBoost models
  - Top 30% most important features extraction
  - Multi-model feature comparison
  - Excel file generation with formatted feature tables
- **Output**: 
  - `Important_Features_All_CellTypes_COAD.xlsx` (multi-sheet Excel file)
  - Individual pickle files for each cell type's important features
- **Models**: Conch, ProvGigapath, Virchow, Virchow2, UNI2h
- **Cell Types**: Cancer Cells, Stromal Cells, Normal Epithelial Cells, T Cells, pan-APC Cells

### Supplementary File 4: Important Features for Breast Cancer
- **Script**: `Sup_file4_Make_Important_Features_Breast.py`
- **Purpose**: Extracts and processes important features from trained XGBoost models for BRCA
- **Key Features**:
  - Feature importance calculation from XGBoost models
  - Top 30% most important features extraction
  - Cross-cancer feature comparison capability
  - Excel file generation with formatted feature tables
- **Output**: 
  - `/BRCA_XGBoost_Results/Important_Features_All_CellTypes_BRCA.xlsx` (multi-sheet Excel file)
  - Individual pickle files for each cell type's important features
- **Models**: Conch, ProvGigapath, Virchow, Virchow2, UNI2h
- **Cell Types**: Tumor Cells, Stromal Cells, Normal Epithelial Cells, T Cells, pan-APC Cells

## Directory Structure

```
Scripts_for_Analysis/
├── README.md                                          # This file
├── Figure2_scRNAseq_data_analysis.py                 # Main Figure 2
├── Figure3_UMAP_Contri_RegressOut.py                 # Main Figure 3
├── Figure4_Xgboost_comparison_Colorectal.py          # Main Figure 4 (COAD)
├── Figure5_Consistency_Analysis.py                    # Main Figure 5
├── Figure6_TCGA_COAD_Analysis.py                      # Main Figure 6
├── Figure7_Xgboost_comparison_Breast.py              # Main Figure 7 (BRCA)
├── FigureS1.py                                        # Supplementary Figure S1
├── FigureS2-S4.py                                     # Cross-reference for S2-S4
├── FigureS6_Quant_Soft_Segmentation.py               # Supplementary Figure S6
├── FigureS8_Naive_CV_performance.py                  # Supplementary Figure S8
├── Sup_file3_Make_Important_Features_Colorectal.py   # Supplementary File 3 (COAD)
└── Sup_file4_Make_Important_Features_Breast.py       # Supplementary File 4 (BRCA)
```

## Usage

### Prerequisites

1. **Completed Workflow Steps**: Ensure you have completed the workflow steps in `../Workflow/`:
   - StepA: Spatial transcriptomics data preparation and CARD deconvolution
   - StepB: Feature extraction with foundation models
   - StepC: XGBoost model training for all cell types

2. **Data Paths**: Update data paths in each script to match your local directory structure:
   - Training features from foundation models
   - CARD deconvolution results
   - Trained XGBoost models
   - Spatial coordinates and metadata

3. **HuggingFace Tokens**: Some scripts require HuggingFace authentication for accessing foundation models:
   ```python
   login(token="your_huggingface_token")
   ```

### Running Scripts

Each script can be run independently:

```bash
# Example: Generate Figure 2
python Figure2_scRNAseq_data_analysis.py

# Example: Generate Figure 4 (COAD analysis)
python Figure4_Xgboost_comparison_Colorectal.py

# Example: Generate Supplementary File 3
python Sup_file3_Make_Important_Features_Colorectal.py
```

### Expected Outputs

- **Figures**: Publication-ready PNG/PDF files with high-resolution plots
- **Excel Files**: Formatted multi-sheet Excel files with feature importance tables
- **Pickle Files**: Intermediate results for downstream analysis
- **Metrics**: Performance metrics (R², MAE, correlation coefficients)

## Dependencies

### Core Libraries
- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- Matplotlib
- Seaborn

### Machine Learning
- scikit-learn
- XGBoost
- SciPy

### Bioinformatics
- Scanpy (for single-cell analysis)
- AnnData

### Foundation Models
- timm
- CONCH (open_clip_custom)
- HuggingFace Hub

### Utilities
- tqdm
- Pillow (PIL)
- openpyxl (for Excel file generation)
- pickle

Install all dependencies:
```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn xgboost scipy scanpy anndata timm pillow openpyxl tqdm
```

For foundation models, follow specific installation instructions in `../Workflow/StepB_Feature_Extraction/`.

## Data Requirements

### Input Data
1. **Trained XGBoost Models**: From `../Workflow/StepC_Feature_Extraction_and_Train_Models/`
2. **CARD Deconvolution Results**: Cell type proportions from CARD
3. **Foundation Model Features**: Extracted features from histopathology images
4. **Spatial Coordinates**: Spot/tile locations for visualization
5. **Single-cell Reference**: scRNA-seq data for validation (Figure 2)
6. **TCGA Data**: Whole slide images and clinical data (Figure 6)

### Cell Type Nomenclature

**Colorectal Cancer (COAD)**:
- Cancer Cells (malignant epithelial)
- Stromal Cells (fibroblasts, CAFs)
- Normal Epithelial Cells (non-malignant epithelial)
- T Cells (T lymphocytes)
- pan-APC Cells (B cells + Myeloid cells)

**Breast Cancer (BRCA)**:
- Tumor Cells (malignant epithelial)
- Stromal Cells (fibroblasts, CAFs)
- Normal Epithelial Cells (non-malignant epithelial)
- T Cells (T lymphocytes)
- pan-APC Cells (B cells + Myeloid cells)

### Foundation Models
1. **Conch**: Contrastive learning-based foundation model
2. **ProvGigapath**: Providence Gigapath foundation model
3. **Virchow**: Virchow foundation model (v1)
4. **Virchow2**: Virchow foundation model (v2)
5. **UNI2h**: Universal pathology foundation model (2-headed)
6. **ResNet50**: Baseline CNN model (ImageNet pretrained)

## Output Files

### Figures
All figures are saved in `../Figures/` directory:
- `Figure2_*.png` - Single-cell analysis results
- `Figure3_*.png` - UMAP analysis
- `Figure4_*.png` - COAD model comparison
- `Figure5_*.png` - Consistency analysis
- `Figure6_*.png` - TCGA COAD analysis
- `Figure7_*.png` - BRCA model comparison
- `FigureS1_*.png` - pan-APC and expression prediction
- `FigureS6_*.png` - Soft segmentation
- `FigureS8_*.png` - Naive CV performance

### Supplementary Files
- `Important_Features_All_CellTypes_COAD.xlsx` - Feature importance for COAD
- `/BRCA_XGBoost_Results/Important_Features_All_CellTypes_BRCA.xlsx` - Feature importance for BRCA

### Intermediate Files
- `*.pkl` - Pickle files with processed results
- `*.pt` - PyTorch tensors with predictions and features

## Troubleshooting

### Common Issues

1. **Missing Data Paths**: Update all file paths to match your directory structure
2. **HuggingFace Authentication**: Ensure valid token for foundation model access
3. **Memory Issues**: Some scripts require significant RAM (16GB+ recommended)
4. **GPU Requirements**: Foundation model inference benefits from GPU acceleration

### Performance Tips

- Use GPU for faster feature extraction and model inference
- Process samples in batches to manage memory usage
- Cache extracted features to avoid redundant computation
- Use multiple CPU cores for parallel processing where available

## Citation

If you use these scripts, please cite:

```bibtex
@article{stpath2026,
  title={STPath: Cell Type Deconvolution from Histopathology using Foundation Models},
  author={Cui, Saishi and others},
  journal={[Journal Name]},
  year={2026}
}
```

## Contact

For questions or issues, please contact:
- Saishi Cui (scui2@example.edu)
- Open an issue on GitHub

## License

[License information to be added]
