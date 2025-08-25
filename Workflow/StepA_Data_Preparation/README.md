# Step A: Data Preparation

This directory contains scripts for preparing spatial transcriptomics data and creating histopathology image patches for the STPath-COAD pipeline.

## Scripts Overview

### 1. Image Patch Creation

#### `Create_patches_images_Cody.py`
- **Purpose**: Creates tissue patches from H&E whole slide images and coordinates them with spatial transcriptomics spots (Cody's dataset)
- **Features**:
  - Tissue segmentation using Otsu thresholding
  - Patch filtering and quality control
  - Spatial coordinate alignment
  - Region-based patch organization
- **Input**: H&E whole slide images, spatial transcriptomics coordinates
- **Output**: Organized patch directories and coordinate files

#### `Create_patche_images_FH.py`
- **Purpose**: Similar patch creation specifically for Fred Hutchinson dataset (Fred Hutchinson dataset)
- **Features**: Dataset-specific processing and coordinate alignment
- **Input**: FH H&E images and spatial coordinates
- **Output**: FH-specific patch organization

### 2. Spatial Transcriptomics Data Preparation

#### `CARD_STData_Preparation_Cody.py`
- **Purpose**: Prepares spatial transcriptomics data for CARD deconvolution analysis (Cody's dataset)
- **Features**:
  - Expression matrix formatting
  - Spatial coordinate processing
  - Data quality filtering
  - CARD-compatible file generation
- **Input**: Raw spatial transcriptomics data
- **Output**: CARD-ready expression and coordinate files

#### `CARD_STData_Preparation_HEST-1K.py`
- **Purpose**: Prepares HEST-1K spatial transcriptomics data for CARD analysis (HEST-1K dataset)
- **Features**:
  - HEST dataset-specific processing
  - Colorectal cancer sample filtering
  - Visium technology compatibility
- **Input**: HEST-1K dataset
- **Output**: Processed HEST data for CARD

## Data Processing Pipeline

### 1. Image Processing
```
H&E WSI → Tissue Segmentation → Patch Extraction → Quality Filtering → Organized Patches
```

### 2. Spatial Data Processing
```
Raw ST Data → Expression Filtering → Coordinate Alignment → CARD Formatting → Ready for Deconvolution
```

## Key Features

### Tissue Segmentation
- **Otsu Thresholding**: Automatic tissue region identification
- **Morphological Operations**: Noise removal and region refinement
- **Connected Components**: Region labeling and size filtering
- **Quality Control**: Minimum spot requirements per region

### Patch Creation
- **Adaptive Sizing**: Based on spot density and image resolution
- **White Patch Filtering**: Removes background and artifact regions
- **Coordinate Alignment**: Ensures spatial consistency
- **Regional Organization**: Groups patches by tissue regions

### Data Quality Control
- **Expression Filtering**: Removes low-quality spots and genes
- **Spatial Validation**: Ensures coordinate consistency
- **Size Thresholds**: Maintains minimum region sizes
- **Format Standardization**: CARD-compatible outputs


## Output Structure

### Patch Organization
```
Colorectal_Cancer_HE_patches/
├── sample1_region0/
│   ├── barcode1_sample1_region0.tif
│   ├── barcode2_sample1_region0.tif
│   └── ...
├── sample1_region1/
└── sample2_region0/
```

### CARD Input Files
```
CARD_Need_Files/
├── sample1_region0_expression.csv
├── sample1_region0_spatial.csv
├── sample1_region1_expression.csv
├── sample1_region1_spatial.csv
└── ...
```

## Dependencies

- Python 3.8+
- PIL (Pillow)
- numpy
- pandas
- matplotlib
- seaborn
- scikit-image
- opencv-python
- tqdm
- glob
- re

## Configuration

### Image Processing Parameters
- **Patch Size**: Automatically calculated based on spot density
- **White Threshold**: 220 (adjustable)
- **White Ratio Cutoff**: 0.4 (adjustable)
- **Minimum Region Size**: 30 spots

### Quality Control
- **Expression Threshold**: Gene and spot filtering criteria
- **Spatial Validation**: Coordinate consistency checks
- **File Format**: CARD-compatible CSV outputs

## Troubleshooting

### Common Issues
1. **Memory Errors**: Process images in batches for large datasets
2. **Coordinate Misalignment**: Check image scaling and rotation
3. **Empty Regions**: Adjust minimum spot thresholds
4. **File Format Errors**: Ensure proper CSV formatting for CARD

### Performance Tips
- Use SSD storage for faster I/O
- Adjust batch sizes based on available memory
- Parallel processing for multiple samples
- Regular cleanup of intermediate files
