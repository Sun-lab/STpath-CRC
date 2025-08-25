# STPath_COAD_Predictor

A comprehensive tool for predicting cell type proportions in colorectal cancer H&E images using multiple foundation models and XGBoost classifiers.

## Overview

This tool combines the power of multiple foundation models (Conch, UNI2h, ProvGigapath, Virchow, Virchow2) with XGBoost classifiers to predict the proportions of 5 different cell types in colorectal cancer H&E images:

- **Cancer Cells**: Colorectal Carcinoma-specific cells/Adenoma-specific cells/Serrated-specific cells
- **Stromal Cells**: Fibroblasts/Endothelial cells 
- **Normal Epithelial Cells**: Tuft cels/Goblet cells/Enteroendocrine cells/Absorptive colonocytes/Crypt-top colonocytes
- **T cells**: CD4+ T cells/CD8+ T cells
- **Other Immune Cells**: B cells/Plasma cells/Myeloid cells/Mast cells

## Features

- **Flexible patch size**: Configurable patch size (default: 240x240 pixels)
- **Automatic device detection**: Automatically uses the best available device (CPU, CUDA, or MPS)
- **White patch filtering**: Automatically excludes patches that are too white (likely background)
- **Multiple output formats**: Results saved in both JSON and CSV formats
- **Batch processing**: Process multiple images efficiently
- **Comprehensive logging**: Detailed progress tracking and error reporting

## Requirements

### Python Dependencies
```
torch
timm
conch
huggingface_hub
xgboost
PIL (Pillow)
numpy
pandas
tqdm
```

### Hardware Requirements
- **Minimum**: CPU with 8GB RAM
- **Recommended**: GPU with CUDA support or Apple Silicon with MPS support
- **Storage**: At least 10GB free space for models and temporary files

### Required Files
1. **HuggingFace Token**: For accessing foundation models
2. **XGBoost Models**: 5 trained XGBoost models (one for each cell type)
3. **Important Features Files**: 5 pickle files containing important feature indices for each cell type
4. **H&E Images**: Whole slide images in common formats (.tif, .tiff, .jpg, .jpeg, .png)

## Installation

1. Clone or download the repository
2. Install required dependencies:
```bash
pip install torch timm conch huggingface_hub xgboost pillow numpy pandas tqdm
```

## Usage

### Quick Start with YAML Configuration

1. **Copy and edit the configuration template:**
```bash
cp config_template.yaml config.yaml
```

2. **Edit the configuration file:**
```yaml
# config.yaml
huggingface:
  token: "your_huggingface_token_here"

models:
  xgboost_dir: "/path/to/xgboost_models"
  features_dir: "/path/to/important_features"

processing:
  patch_size: 240
  device: "auto"
  white_threshold: 220
  white_ratio_cutoff: 0.4

paths:
  output_dir: "/path/to/output/results"
```

3. **Run the predictor:**
```bash
python STPath_COAD_Predictor.py --config config.yaml --image_path /path/to/your/image.tif
```

### Command Line Override

You can also override any configuration value via command line:

```bash
python STPath_COAD_Predictor.py \
    --config config.yaml \
    --image_path /path/to/your/image.tif \
    --patch_size 320 \
    --device cuda
```

#### Parameters:
- `--config`: Path to YAML configuration file
- `--image_path`: Path to the H&E image file (overrides config)
- `--hf_token`: HuggingFace token (overrides config)
- `--model_dir`: XGBoost models directory (overrides config)
- `--features_dir`: Features directory (overrides config)
- `--output_path`: Output path (overrides config)
- `--patch_size`: Patch size in pixels (overrides config)
- `--device`: Device to use (overrides config)
- `--white_threshold`: White pixel threshold (overrides config)
- `--white_ratio_cutoff`: White ratio cutoff (overrides config)

### Python API

For more advanced usage, you can use the Python API:

```python
from STPath_COAD_Predictor import STPath_COAD_Predictor
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize predictor
predictor = STPath_COAD_Predictor(
    hf_token=config['huggingface']['token'],
    patch_size=config['processing']['patch_size'],
    device=config['processing']['device'],
    white_threshold=config['processing']['white_threshold'],
    white_ratio_cutoff=config['processing']['white_ratio_cutoff']
)

# Load models
predictor.load_foundation_models()
predictor.load_xgboost_models(model_paths, important_features_paths)

# Run prediction
results = predictor.predict_image("path/to/image.tif", "path/to/output")

# Check the overall cell type proportions
overall_proportions = predictor.get_overall_proportions(results)
print(overall_proportions)
```









## File Structure

### Required Model Files

Your model directory should contain:
```
model_dir/
├── xgboost_model_Cancer Cells_Combined_external_prediction.model
├── xgboost_model_Normal Epithelial Cells_Combined_external_prediction.model
├── xgboost_model_T Cells_Combined_external_prediction.model
├── xgboost_model_Stromal Cells_Combined_external_prediction.model
└── xgboost_model_Other Immune Cells_Combined_external_prediction.model
```

### Required Feature Files

Your features directory should contain:
```
features_dir/
├── important_features_Cancer Cells.pkl
├── important_features_Stromal Cells.pkl
├── important_features_Normal Epithelial Cells.pkl
├── important_features_T cells.pkl
└── important_features_Other Immune Cells.pkl
```

### Example Output Structure
```json
{
  "image_path": "/path/to/image.tif",
  "patch_size": 240,
  "device": "cuda:0",
  "total_patches": 1000,
  "valid_patches": 850,
  "white_patches": 150,
  "cell_types": ["Cancer Cells", "Stromal Cells", ...],
  "predictions": [
    {
      "patch_id": "row000_col000",
      "row": 0,
      "col": 0,
      "x_start": 0,
      "y_start": 0,
      "x_end": 240,
      "y_end": 240,
      "cancer_proportion": 0.25,
      "stromal_proportion": 0.35,
      "normal_epithelia_proportion": 0.20,
      "tcells_proportion": 0.15,
      "other_immune_proportion": 0.05,
      "normalized_proportions": [0.25, 0.35, 0.20, 0.15, 0.05],
      "is_white_patch": false
    }
  ],
  "timestamp": "2024-01-01T12:00:00"
}
```

## Performance Considerations

### Memory Usage
- **Foundation Models**: ~8GB GPU memory for all models
- **Patch Processing**: Memory scales with patch size and batch size
- **Large Images**: Consider processing in smaller sections for very large images

### Processing Speed
- **GPU (CUDA/MPS)**: ~100-500 patches per minute
- **CPU**: ~10-50 patches per minute
- **Batch Processing**: More efficient for multiple images

### Optimization Tips
1. Use GPU acceleration when available
2. Adjust patch size based on your needs (larger patches = fewer total patches)
3. Process multiple images in batch to reuse loaded models
4. Use SSD storage for faster I/O

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce patch size
   - Use CPU instead of GPU
   - Process smaller image sections

2. **Model Loading Errors**
   - Check HuggingFace token validity
   - Ensure internet connection for model downloads
   - Verify model file paths

3. **White Patch Detection**
   - Adjust `white_threshold` (default: 220) and `white_ratio_cutoff` (default: 0.4) parameters
   - Check image quality and staining

4. **Device Issues**
   - Use `--device cpu` for compatibility
   - Check CUDA/MPS availability
   - Update GPU drivers if needed

### Error Messages

- **"Model file not found"**: Check model directory path and file names
- **"Features file not found"**: Check features directory path and file names
- **"Image file not found"**: Verify image path and file format
- **"CUDA out of memory"**: Reduce patch size or use CPU

## Examples

### Single Image Processing
```bash
# Using YAML config
python CRC_Cell_Proportion_Predictor.py --config config.yaml --image_path image.tif

# Override specific parameters
python CRC_Cell_Proportion_Predictor.py --config config.yaml --image_path image.tif --patch_size 320 --device cuda
```

### Batch Processing
```bash
# Process multiple images
for image in images/*.tif; do
    python CRC_Cell_Proportion_Predictor.py --config config.yaml --image_path "$image"
done
```

## Citation

If you use this tool in your research, please cite the relevant foundation models and your own work.

## License

This tool is provided for research and clinical use. Please ensure compliance with relevant regulations and ethical guidelines.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review example usage
3. Verify all dependencies are installed
4. Check file paths and permissions

## Version History

- **v1.0**: Initial release with basic functionality
- Support for 5 foundation models
- XGBoost integration
- Flexible patch sizing
- Multiple output formats
