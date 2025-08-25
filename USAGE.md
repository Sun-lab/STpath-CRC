# Quick Usage Guide

## Step 1: Setup Configuration

Copy and edit the configuration file:
```bash
cp config_template.yaml config.yaml
```

Edit `config.yaml` with your settings:
```yaml
huggingface:
  token: "your_actual_token_here"

models:
  xgboost_dir: "/your/actual/model/path"
  features_dir: "/your/actual/features/path"

processing:
  patch_size: 240
  device: "auto"
  white_threshold: 220
  white_ratio_cutoff: 0.4

paths:
  output_dir: "/your/output/path"
```

## Step 2: Run Prediction

```bash
python CRC_Cell_Proportion_Predictor.py --config config.yaml --image_path your_image.tif
```

## Step 3: Get Results

Results will be saved as:
- `output_dir/results.json` - Complete results
- `output_dir/results.csv` - Patch-level data

## Optional: Override Settings

```bash
# Override patch size and device
python CRC_Cell_Proportion_Predictor.py --config config.yaml --image_path image.tif --patch_size 320 --device cuda
```

That's it! 🎉
