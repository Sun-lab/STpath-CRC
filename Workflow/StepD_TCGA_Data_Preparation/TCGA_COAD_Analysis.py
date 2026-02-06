"""
TCGA COAD Analysis Pipeline
============================
Process TCGA COAD whole slide images to predict cell type proportions and classify tiles.
This script is part of the STPath-COAD workflow for analyzing TCGA colorectal cancer data.

Author: Saishi Cui
Date: January 2026

Pipeline:
1. WSI Processing: Cut images into tiles, extract features, predict proportions
2. Classification: Merge cell types and classify tiles using quantile-based thresholds

Input: .tif whole slide image path
Output: 
1. CSV file with tile-level predictions (8 cell types)
2. Hexagonal heatmap for Cancer Cells visualization
3. Classification CSV with merged cell types (5 cell types)
4. Hexagonal visualization of tile classifications

Note: For survival analysis, see TCGA_COAD_Survival.R

Cell Type Merging (8 → 5):
- Cancer Cells (unchanged)
- Normal Epithelial Cells (unchanged)
- T Cells (unchanged)
- Stromal Cells = CAFs + Endothelial
- pan-APC = B Cells + Myeloid + Plasma

Classification Strategy:
A tile is classified as cell type X if:
  - X proportion >= 75th percentile of X in this sample
  - All other cell types < 75th percentile of themselves

Author: Saishi Cui
Date: December 2025
"""

# MUST set this BEFORE importing cv2 to handle large TIFF files (4GB+)
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))  # ~1 trillion pixels

import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
from timm.layers import SwiGLUPacked
from conch.open_clip_custom import create_model_from_pretrained
from huggingface_hub import login
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import shutil
import pickle
import glob


def login_to_huggingface(hf_token):
    """
    Login to HuggingFace once before processing multiple WSIs
    
    Args:
        hf_token (str): HuggingFace authentication token
    
    Example:
        >>> login_to_huggingface("hf_xxxxx")
        >>> # Now process multiple WSIs without logging in again
    """
    print(f"\n{'='*80}")
    print("Logging into HuggingFace...")
    print(f"{'='*80}\n")
    login(token=hf_token)
    print("✅ HuggingFace login successful!\n")


def load_foundation_models(device=None):
    """
    Load all 5 foundation models once before processing multiple WSIs
    
    Args:
        device (str or torch.device, optional): Device to load models on.
                                                If None, auto-detects (CUDA > MPS > CPU)
    
    Returns:
        dict: Dictionary of loaded models with their transforms
    
    Example:
        >>> models_dict = load_foundation_models()
        >>> # Now use models_dict for multiple WSIs
    """
    print(f"\n{'='*80}")
    print("Loading Foundation Models")
    print(f"{'='*80}\n")
    
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    
    models = {}
    
    # Standard transform for all models (except Conch which has its own)
    standard_transform = transforms.Compose([
        transforms.Resize(224), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    ### 1. UNI2h
    print("  Loading UNI2h...")
    timm_kwargs = {
        'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
        'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2,
        'num_classes': 0, 'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU,
        'reg_tokens': 8, 'dynamic_img_size': True
    }
    model_UNI2h = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    model_UNI2h.eval()
    model_UNI2h = model_UNI2h.to(device)
    models['UNI2h'] = {'model': model_UNI2h, 'transform': standard_transform}
    
    ### 2. Virchow
    print("  Loading Virchow...")
    model_Virchow = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, 
                                     mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model_Virchow.eval()
    model_Virchow = model_Virchow.to(device)
    models['Virchow'] = {'model': model_Virchow, 'transform': standard_transform}
    
    ### 3. Virchow2
    print("  Loading Virchow2...")
    model_Virchow2 = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True,
                                      mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model_Virchow2.eval()
    model_Virchow2 = model_Virchow2.to(device)
    models['Virchow2'] = {'model': model_Virchow2, 'transform': standard_transform}
    
    ### 4. ProvGigapath
    print("  Loading ProvGigapath...")
    model_ProvGigapath = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    model_ProvGigapath.eval()
    model_ProvGigapath = model_ProvGigapath.to(device)
    models['ProvGigapath'] = {'model': model_ProvGigapath, 'transform': standard_transform}
    
    ### 5. Conch
    print("  Loading Conch...")
    model_Conch, transform_Conch = create_model_from_pretrained(
        'conch_ViT-B-16', "hf_hub:MahmoodLab/conch"
    )
    model_Conch.eval()
    model_Conch = model_Conch.to(device)
    models['Conch'] = {'model': model_Conch, 'transform': transform_Conch}
    
    print(f"\n✅ All models loaded and ready on {device}!\n")
    
    return models


def process_wsi_for_cell_type_prediction(
    wsi_path,
    models_dict,
    models_dir,
    important_features_excel,
    calibration_dir,
    output_predictions_dir,
    output_heatmaps_dir,
    temp_tiles_folder="/tmp/wsi_tiles",
    white_threshold=220,
    cut_size=70
):
    """
    Complete pipeline: Cut WSI -> Extract features -> Predict (with calibration) -> Visualize Cancer Cells
    
    Args:
        wsi_path (str): Path to whole slide image (.tif)
        models_dict (dict): Loaded foundation models from load_foundation_models()
        models_dir (str): Directory with trained XGBoost models
        important_features_excel (str): Excel file with important features
        calibration_dir (str): Directory containing calibration parameters (.pkl files)
        output_predictions_dir (str): Directory to save prediction CSVs (calibrated, not normalized)
        output_heatmaps_dir (str): Directory to save Cancer Cells heatmaps
        temp_tiles_folder (str): Temporary folder for tiles (will be cleaned up)
        white_threshold (float): Threshold for white background detection
        cut_size (int): Cut size for shorter edge (e.g., 70)
    
    Returns:
        dict: Summary statistics
        
    Note:
        Before processing WSIs:
        1. Call login_to_huggingface(hf_token) ONCE
        2. Call load_foundation_models() ONCE to get models_dict
        3. Then process multiple WSIs using the same models_dict
        
        Predictions will be calibrated but NOT normalized (raw calibrated proportions)
    """
    
    print(f"\n{'='*80}")
    print(f"Processing WSI: {os.path.basename(wsi_path)}")
    print(f"{'='*80}\n")
    
    ### Extract WSI name for output files
    wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
    
    ### Create output directories
    os.makedirs(output_predictions_dir, exist_ok=True)
    os.makedirs(output_heatmaps_dir, exist_ok=True)
    os.makedirs(temp_tiles_folder, exist_ok=True)
    
    
    ### Step 1: Cut image into tiles with automatic grid calculation
    print(f"{'='*80}")
    print(f"Step 1: Cutting WSI into tiles")
    print(f"{'='*80}\n")
    
    # Use PIL to load large TIFF files (avoids OpenCV 4GB+ limit)
    Image.MAX_IMAGE_PIXELS = None  # Remove PIL's pixel limit
    img_pil = Image.open(wsi_path)
    img_width, img_height = img_pil.size
    print(f"Image dimensions: {img_width} x {img_height}")
    
    # Calculate grid based on shorter edge
    if img_width <= img_height:
        # Width is shorter
        grid_cols = cut_size
        aspect_ratio = img_height / img_width
        grid_rows = int(np.round(cut_size * aspect_ratio))
    else:
        # Height is shorter
        grid_rows = cut_size
        aspect_ratio = img_width / img_height
        grid_cols = int(np.round(cut_size * aspect_ratio))
    
    total_tiles = grid_rows * grid_cols
    print(f"Automatic grid calculation:")
    print(f"  Shorter edge cut size: {cut_size}")
    print(f"  Aspect ratio: {aspect_ratio:.2f}")
    print(f"  Grid: {grid_rows} rows x {grid_cols} cols = {total_tiles} tiles")
    
    # Calculate tile dimensions
    tile_height = img_height // grid_rows
    tile_width = img_width // grid_cols
    print(f"  Tile size: {tile_width} x {tile_height} pixels\n")
    
    # Create tiles
    valid_count = 0
    white_count = 0
    
    print("Creating tiles...")
    for row in tqdm(range(grid_rows), desc="Processing rows"):
        for col in range(grid_cols):
            y_start = row * tile_height
            y_end = (row + 1) * tile_height if row < grid_rows - 1 else img_height
            x_start = col * tile_width
            x_end = (col + 1) * tile_width if col < grid_cols - 1 else img_width
            
            # Crop tile using PIL
            tile_pil = img_pil.crop((x_start, y_start, x_end, y_end))
            
            # Convert to numpy for white detection
            tile_np = np.array(tile_pil)
            if len(tile_np.shape) == 3:  # RGB
                gray = np.mean(tile_np, axis=2)
            else:  # Grayscale
                gray = tile_np
            mean_intensity = np.mean(gray)
            
            tile_name = f"{row+1}-{col+1}"
            if mean_intensity > white_threshold:
                tile_name += "_White"
                white_count += 1
            else:
                valid_count += 1
            
            output_path = os.path.join(temp_tiles_folder, f"{tile_name}.tif")
            tile_pil.save(output_path)
    
    # Close the PIL image
    img_pil.close()
    
    print(f"\n✅ Tile creation completed!")
    print(f"   Total tiles: {total_tiles}")
    print(f"   Valid tiles: {valid_count}")
    print(f"   White tiles: {white_count}\n")
    
    
    ### Step 2: Extract features from tiles
    print(f"{'='*80}")
    print(f"Step 2: Extracting features from tiles")
    print(f"{'='*80}\n")
    
    # Get device from models
    first_model = list(models_dict.values())[0]['model']
    device = next(first_model.parameters()).device
    print(f"Using device: {device}\n")
    
    # Extract features for each model
    model_order = ['UNI2h', 'Virchow', 'Virchow2', 'ProvGigapath', 'Conch']
    all_features = {}
    
    for model_name in model_order:
        print(f"\nExtracting features using {model_name}...")
        model = models_dict[model_name]['model']
        transform = models_dict[model_name]['transform']
        features = _extract_for_one_model(model_name, model, transform, device, temp_tiles_folder)
        all_features[model_name] = features
    
    # Organize features by tile position
    print("\nOrganizing features by tile position...")
    final_features = {}
    
    for row in range(1, grid_rows + 1):
        for col in range(1, grid_cols + 1):
            tile_name = f"{row}-{col}"
            tile_name_white = f"{row}-{col}_White"
            
            white_tile_path = os.path.join(temp_tiles_folder, f"{tile_name_white}.tif")
            if os.path.exists(white_tile_path):
                final_features[tile_name] = None
            else:
                features_list = []
                for model_name in model_order:
                    if tile_name in all_features[model_name]:
                        features_list.append(all_features[model_name][tile_name])
                    else:
                        features_list.append(None)
                
                final_features[tile_name] = features_list
    
    valid_feature_count = sum(1 for v in final_features.values() if v is not None)
    white_feature_count = sum(1 for v in final_features.values() if v is None)
    
    print(f"\n✅ Feature extraction completed!")
    print(f"   Total tiles: {len(final_features)}")
    print(f"   Valid tiles with features: {valid_feature_count}")
    print(f"   White tiles (None): {white_feature_count}\n")
    
    
    ### Step 3: Predict cell type proportions (with calibration, no normalization)
    print(f"{'='*80}")
    print(f"Step 3: Predicting cell type proportions (calibrated, not normalized)")
    print(f"{'='*80}\n")
    
    predictions_df = _predict_cell_type_proportions(
        final_features, models_dir, important_features_excel, calibration_dir, 
        if_calibration=True, normalize=False
    )
    
    # Debug: Check predictions_df before saving
    print(f"\n{'='*80}")
    print(f"About to save predictions DataFrame:")
    print(f"  Shape: {predictions_df.shape}")
    print(f"  Columns: {predictions_df.columns.tolist()}")
    print(f"  Memory usage: {predictions_df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print(f"{'='*80}\n")
    
    # Save predictions CSV
    predictions_csv_path = os.path.join(output_predictions_dir, f"{wsi_name}_predictions.csv")
    predictions_df.to_csv(predictions_csv_path, index=False)
    print(f"\n✅ Predictions saved to: {predictions_csv_path}")
    
    # Verify file was written
    if os.path.exists(predictions_csv_path):
        file_size = os.path.getsize(predictions_csv_path)
        print(f"   File size: {file_size} bytes")
        if file_size == 0:
            print("   ⚠️ WARNING: File is empty (0 bytes)!")
    else:
        print(f"   ❌ ERROR: File was not created!")
    print()
    
    
    ### Step 4: Visualize Cancer Cells hexagonal heatmap
    print(f"{'='*80}")
    print(f"Step 4: Creating Cancer Cells hexagonal heatmap")
    print(f"{'='*80}\n")
    
    heatmap_path = os.path.join(output_heatmaps_dir, f"{wsi_name}_CancerCells_heatmap.png")
    _visualize_hexagonal_heatmap(predictions_df, "Cancer", heatmap_path)
    
    
    ### Clean up temporary tiles folder
    print(f"\n{'='*80}")
    print(f"Cleaning up temporary tiles folder...")
    print(f"{'='*80}\n")
    
    if os.path.exists(temp_tiles_folder):
        try:
            shutil.rmtree(temp_tiles_folder)
            print(f"✅ Temporary tiles deleted: {temp_tiles_folder}\n")
        except Exception as e:
            print(f"⚠️ Warning: Could not delete temporary tiles folder: {e}\n")
    else:
        print(f"⚠️ Temporary tiles folder not found: {temp_tiles_folder}\n")
    
    
    ### Summary
    summary = {
        'wsi_name': wsi_name,
        'total_tiles': total_tiles,
        'valid_tiles': valid_count,
        'white_tiles': white_count,
        'grid_rows': grid_rows,
        'grid_cols': grid_cols,
        'predictions_csv': predictions_csv_path,
        'heatmap_png': heatmap_path
    }
    
    print(f"\n{'='*80}")
    print(f"✅ WSI Processing Completed: {wsi_name}")
    print(f"{'='*80}")
    print(f"Grid: {grid_rows} x {grid_cols} = {total_tiles} tiles")
    print(f"Valid tiles: {valid_count}")
    print(f"Predictions CSV: {predictions_csv_path}")
    print(f"Heatmap PNG: {heatmap_path}")
    print(f"{'='*80}\n")
    
    return summary


### Helper Functions

class TileDataset(Dataset):
    """Dataset class for loading tiles"""
    def __init__(self, tile_paths, transform=None):
        self.tile_paths = tile_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.tile_paths)
    
    def __getitem__(self, idx):
        tile_path = self.tile_paths[idx]
        tile_name = os.path.basename(tile_path).replace('.tif', '')
        image = Image.open(tile_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, tile_name


def _extract_for_one_model(model_name, model, transform, device, tiles_folder, batch_size=32):
    """Extract features for a single foundation model"""
    
    all_tiles = [f for f in os.listdir(tiles_folder) if f.endswith('.tif')]
    valid_tiles = [f for f in all_tiles if '_White' not in f]
    valid_tile_paths = [os.path.join(tiles_folder, f) for f in valid_tiles]
    
    if len(valid_tile_paths) == 0:
        print(f"  No valid tiles found for {model_name}")
        return {}
    
    dataset = TileDataset(valid_tile_paths, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    embeddings_list = []
    tile_names_list = []
    
    with torch.no_grad():
        for images, tile_names in tqdm(loader, desc=f"  Extracting {model_name} features"):
            images = images.to(device)
            
            if model_name == 'Conch':
                embeddings = model.encode_image(images, proj_contrast=False, normalize=False)
            elif model_name in ['Virchow', 'Virchow2']:
                embeddings = model(images)
                embeddings = torch.cat([embeddings[:,0], embeddings[:,5:].mean(1)], dim=-1)
            else:
                embeddings = model(images)
            
            embeddings_list.append(embeddings.cpu())
            tile_names_list.extend(tile_names)
    
    all_embeddings = torch.cat(embeddings_list, 0)
    
    features_dict = {}
    for i, tile_name in enumerate(tile_names_list):
        features_dict[tile_name] = all_embeddings[i].tolist()
    
    print(f"  {model_name}: Extracted features for {len(features_dict)} tiles")
    return features_dict


def _predict_cell_type_proportions(final_features, models_dir, important_features_excel, calibration_dir, 
                                   if_calibration=True, normalize=False):
    """
    Predict cell type proportions using trained XGBoost models with optional calibration
    
    Args:
        final_features (dict): Feature dictionary from extraction step
        models_dir (str): Directory with XGBoost models
        important_features_excel (str): Excel file with important features
        calibration_dir (str): Directory with calibration parameters (.pkl files)
        if_calibration (bool): Whether to apply calibration (default: True)
        normalize (bool): Whether to normalize predictions (default: False)
    
    Returns:
        pd.DataFrame: Predictions DataFrame
    """
    
    ### Define cell types (COAD has 5 cell types, different from BRCA's 8)
    # Must match the model file names exactly
    cell_types = [
        "Cancer Cells",
        "Stromal Cells",
        "pan-APC Cells",
        "T Cells",
        "Normal Epithelial Cells"
    ]
    
    ### Load important features from Excel
    print("Loading important features from Excel...")
    important_features_dict = {}
    
    for cell_type in cell_types:
        sheet_name = cell_type
        print(f"  Loading sheet: {sheet_name}")
        
        try:
            df = pd.read_excel(important_features_excel, sheet_name=sheet_name)
        except Exception as e:
            print(f"    Warning: Could not load sheet '{sheet_name}': {e}")
            continue
        
        important_features_dict[cell_type] = {}
        model_order = ['UNI2h', 'Virchow', 'Virchow2', 'ProvGigapath', 'Conch']
        
        for model_name in model_order:
            feature_col = f'{model_name}_Features'
            if feature_col in df.columns:
                features_list = df[feature_col].dropna().tolist()
                indices = []
                for f in features_list:
                    f_str = str(f)
                    if f_str.startswith('f'):
                        try:
                            indices.append(int(f_str.replace('f', '')))
                        except ValueError:
                            continue
                important_features_dict[cell_type][model_name] = indices
            else:
                important_features_dict[cell_type][model_name] = []
    
    ### Load XGBoost models
    print("\nLoading XGBoost models...")
    models = {}
    for cell_type in cell_types:
        model_filename = f"xgboost_model_{cell_type}_Combined_external_prediction.model"
        model_path = os.path.join(models_dir, model_filename)
        
        if os.path.exists(model_path):
            models[cell_type] = xgb.Booster()
            models[cell_type].load_model(model_path)
            print(f"  Loaded: {cell_type}")
        else:
            print(f"  Warning: Model not found for {cell_type}: {model_path}")
    
    ### Load calibration parameters if calibration is enabled
    calibration_params = {}
    if if_calibration:
        print("\nLoading calibration parameters...")
        # Mapping from model names to calibration file names (which have inconsistent naming)
        calibration_file_mapping = {
            "Cancer Cells": "Cancer_cells_calibration_parameters.pkl",
            "Stromal Cells": "Stromal_Cells_calibration_parameters.pkl",
            "pan-APC Cells": "pan-APC_Cells_calibration_parameters.pkl",
            "T Cells": "T_Cells_calibration_parameters.pkl",
            "Normal Epithelial Cells": "Normal_Epithelial_calibration_parameters.pkl"
        }
        
        for cell_type in cell_types:
            pkl_filename = calibration_file_mapping.get(cell_type)
            if pkl_filename:
                pkl_path = os.path.join(calibration_dir, pkl_filename)
                
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as f:
                        calibration_params[cell_type] = pickle.load(f)
                    print(f"  ✅ Loaded calibration: {cell_type}")
                else:
                    print(f"  ⚠️  Warning: Calibration file not found for {cell_type}")
                    print(f"      Expected: {pkl_filename}")
                    print(f"      Will use raw predictions for this cell type.")
            else:
                print(f"  ⚠️  Warning: No calibration mapping for {cell_type}")
    else:
        print("\nCalibration disabled - using raw predictions")
    
    ### Prepare results
    results = []
    
    tile_names = sorted(final_features.keys(), key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))
    
    print(f"\nMaking predictions for {len(tile_names)} tiles...")
    
    for tile_name in tqdm(tile_names, desc="Predicting"):
        tile_features = final_features[tile_name]
        
        is_white = tile_features is None
        
        # Mark white tiles with _White suffix
        if is_white:
            result_row = {'tile_name': f"{tile_name}_White"}
            for cell_type in cell_types:
                result_row[cell_type] = np.nan
        else:
            # Check if any features are None (shouldn't happen but just in case)
            if any(f is None for f in tile_features):
                result_row = {'tile_name': f"{tile_name}_White"}
                for cell_type in cell_types:
                    result_row[cell_type] = np.nan
            else:
                # Valid tile with features - make predictions
                result_row = {'tile_name': tile_name}
                predictions = {}
                
                for cell_type in cell_types:
                    if cell_type not in models or cell_type not in important_features_dict:
                        predictions[cell_type] = np.nan
                        continue
                    
                    model_order = ['UNI2h', 'Virchow', 'Virchow2', 'ProvGigapath', 'Conch']
                    selected_features = []
                    
                    for i, model_name in enumerate(model_order):
                        embedding = np.array(tile_features[i])
                        indices = important_features_dict[cell_type][model_name]
                        
                        if len(indices) > 0:
                            selected = embedding[indices]
                            selected_features.append(selected)
                
                    if len(selected_features) > 0:
                        combined_features = np.concatenate(selected_features)
                        
                        # Make raw prediction
                        dmatrix = xgb.DMatrix(combined_features.reshape(1, -1))
                        raw_prediction = models[cell_type].predict(dmatrix)[0]
                        
                        # Apply calibration if enabled and available
                        if if_calibration and cell_type in calibration_params and 'lookup_table' in calibration_params[cell_type]:
                            lookup = calibration_params[cell_type]['lookup_table']
                            y_pred_quantiles = lookup['y_pred_quantiles']
                            y_true_quantiles = lookup['y_true_quantiles']
                            
                            # Apply quantile mapping with linear interpolation (constant extrapolation)
                            calibrated_prediction = np.interp(raw_prediction, y_pred_quantiles, y_true_quantiles)
                            calibrated_prediction = np.clip(calibrated_prediction, 0, 1)
                            predictions[cell_type] = calibrated_prediction
                        else:
                            # No calibration or calibration disabled, use raw prediction
                            predictions[cell_type] = raw_prediction
                    else:
                        predictions[cell_type] = np.nan
                
                for cell_type in cell_types:
                    result_row[cell_type] = predictions.get(cell_type, np.nan)
        
        results.append(result_row)
    
    ### Create DataFrame
    df_results = pd.DataFrame(results)
    
    print(f"\nDataFrame shape: {df_results.shape}")
    print(f"DataFrame columns: {df_results.columns.tolist()}")
    
    ### Rename columns to simplified names for easier downstream processing
    print(f"\nRenaming columns to simplified names...")
    column_mapping = {
        "Cancer Cells": "Cancer",
        "Stromal Cells": "Stromal",
        "pan-APC Cells": "pan-APC",
        "T Cells": "T_Cells",
        "Normal Epithelial Cells": "Normal_Epithelial"
    }
    df_results = df_results.rename(columns=column_mapping)
    
    # Update cell_types to use simplified names
    cell_types_simplified = ["Cancer", "Stromal", "pan-APC", "T_Cells", "Normal_Epithelial"]
    
    print(f"Renamed DataFrame shape: {df_results.shape}")
    print(f"Renamed DataFrame columns: {df_results.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(df_results.head())
    
    ### Normalize if requested (calibration has already been applied if enabled)
    if normalize:
        pred_type = "calibrated" if (if_calibration and len(calibration_params) > 0) else "raw"
        print(f"\nNormalizing {pred_type} predictions...")
        for idx, row in df_results.iterrows():
            if not pd.isna(row[cell_types_simplified[0]]):
                total = sum(row[cell_type] for cell_type in cell_types_simplified)
                if total > 0:
                    for cell_type in cell_types_simplified:
                        df_results.at[idx, cell_type] = row[cell_type] / total
    
    ### Print summary
    valid_tiles = df_results[cell_types_simplified[0]].notna().sum()
    white_tiles = df_results[cell_types_simplified[0]].isna().sum()
    
    calibrated_count = len(calibration_params)
    
    print(f"\n✅ Prediction completed!")
    print(f"   Total tiles: {len(df_results)}")
    print(f"   Valid predictions: {valid_tiles}")
    print(f"   White tiles (NaN): {white_tiles}")
    
    if if_calibration:
        if calibrated_count > 0:
            print(f"   Calibrated cell types: {calibrated_count}/{len(cell_types)}")
            print(f"   Cell types using calibration: {list(calibration_params.keys())}")
        else:
            print(f"   ⚠️  No calibration applied - using raw predictions for all cell types")
    else:
        print(f"   Using raw predictions (calibration disabled)")
    
    if normalize:
        print(f"   Predictions normalized (sum to 1)")
    else:
        print(f"   Predictions NOT normalized (raw calibrated proportions)")
    
    return df_results


def _visualize_hexagonal_heatmap(predictions_df, cell_type, output_path):
    """Visualize cell type proportion predictions as a hexagonal heatmap"""
    
    print(f"\nCreating hexagonal heatmap for: {cell_type}")
    
    ### Parse tile positions
    data_list = []
    for idx, row in predictions_df.iterrows():
        tile_name = row['tile_name']
        
        try:
            parts = tile_name.split('-')
            tile_row = int(parts[0])
            tile_col = int(parts[1])
        except (ValueError, IndexError):
            continue
        
        proportion = row[cell_type]
        
        x_center = tile_col
        y_center = tile_row
        
        data_list.append({
            'tile_name': tile_name,
            'row': tile_row,
            'col': tile_col,
            'x': x_center,
            'y': y_center,
            'proportion': proportion
        })
    
    df = pd.DataFrame(data_list)
    
    ### Filter out white tiles
    valid_df = df[df['proportion'].notna()].copy()
    
    if len(valid_df) == 0:
        print("❌ No valid tiles to visualize")
        return
    
    ### Convert to percentage
    valid_df['proportion_pct'] = valid_df['proportion'] * 100
    
    max_proportion_pct = valid_df['proportion_pct'].max()
    scale_ceiling = np.ceil(max_proportion_pct / 10) * 10
    
    print(f"  Total tiles: {len(df)}")
    print(f"  Valid tiles: {len(valid_df)}")
    print(f"  Proportion range: {valid_df['proportion_pct'].min():.2f}% - {valid_df['proportion_pct'].max():.2f}%")
    print(f"  Color scale: 0% - {scale_ceiling:.0f}%")
    
    ### Calculate marker size
    max_row = df['row'].max()
    max_col = df['col'].max()
    estimated_grid_size = max(max_row, max_col)
    
    base_size = 150
    size_factor = (40 / estimated_grid_size) ** 1.5
    marker_size = max(20, int(base_size * size_factor))
    
    print(f"  Grid dimensions: {max_row} rows x {max_col} columns")
    print(f"  Marker size: {marker_size}")
    
    ### Create hexagonal scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        x=valid_df['x'],
        y=valid_df['y'],
        c=valid_df['proportion_pct'],
        cmap='coolwarm',
        marker='h',
        s=marker_size,
        edgecolors='black',
        linewidth=0.5,
        vmin=0,
        vmax=scale_ceiling
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.ax.tick_params(labelsize=20, colors='black', width=3)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_color('black')
    
    ax.invert_yaxis()
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(5)
        spine.set_color('black')
    
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.05)
    print(f"\n✅ Heatmap saved to: {output_path}")
    
    plt.close()

### Example usage
if __name__ == "__main__":
    
    ### Configuration for COAD
    HF_TOKEN = "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"
    MODELS_DIR = "/Users/scui2/Desktop/STPath_COAD/STPath_Software/COAD/XGBoost_Models"
    IMPORTANT_FEATURES_EXCEL = "/Users/scui2/Desktop/STPath_COAD/STPath_Software/COAD/XGBoost_Models/Important_Features_COAD.xlsx"
    CALIBRATION_DIR = "/Users/scui2/Desktop/STPath_COAD/STPath_Software/COAD/XGBoost_Models"  # Same dir contains .pkl files
    OUTPUT_PREDICTIONS_DIR = "/Users/scui2/Desktop/TCGA/TCGA_COAD/TCGA_COAD_Predictions_CSV"
    OUTPUT_HEATMAPS_DIR = "/Users/scui2/Desktop/TCGA/TCGA_COAD/TCGA_CancerCells_Heatmaps"
    TEMP_TILES_FOLDER = "/Users/scui2/Desktop/TCGA/TCGA_COAD/TCGA_COAD_Tiles"
    
    ### Step 1: Login to HuggingFace ONCE
    login_to_huggingface(HF_TOKEN)
    
    ### Step 2: Load foundation models ONCE
    models_dict = load_foundation_models()
    
    ### Step 3: Process all WSIs in the folder
    
    wsi_folder = "/Users/scui2/Desktop/TCGA/TCGA_COAD/TCGA_COAD_HE_filtered_tiff"
    all_wsi_files = sorted(glob.glob(os.path.join(wsi_folder, "*.tiff")))
    
    # Check which files have already been processed
    processed_csv_files = glob.glob(os.path.join(OUTPUT_PREDICTIONS_DIR, "*_predictions.csv"))
    processed_names = set([os.path.basename(f).replace('_predictions.csv', '') for f in processed_csv_files])
    
    # Find unprocessed files
    unprocessed_wsi_files = []
    for wsi_path in all_wsi_files:
        wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
        if wsi_name not in processed_names:
            unprocessed_wsi_files.append(wsi_path)
    
    print(f"\n{'='*80}")
    print(f"Total WSI files: {len(all_wsi_files)}")
    print(f"Already processed: {len(processed_names)}")
    print(f"Remaining to process: {len(unprocessed_wsi_files)}")
    print(f"{'='*80}\n")
    
    if len(unprocessed_wsi_files) == 0:
        print("✅ All WSI files have been processed!")
    else:
        for i, wsi_path in enumerate(unprocessed_wsi_files, 1):
            print(f"\n\n{'#'*80}")
            print(f"Processing WSI {i}/{len(unprocessed_wsi_files)}: {os.path.basename(wsi_path)}")
            print(f"(Overall progress: {len(processed_names) + i}/{len(all_wsi_files)})")
            print(f"{'#'*80}\n")
            
            try:
                summary = process_wsi_for_cell_type_prediction(
                    wsi_path=wsi_path,
                    models_dict=models_dict,
                    models_dir=MODELS_DIR,
                    important_features_excel=IMPORTANT_FEATURES_EXCEL,
                    calibration_dir=CALIBRATION_DIR,
                    output_predictions_dir=OUTPUT_PREDICTIONS_DIR,
                    output_heatmaps_dir=OUTPUT_HEATMAPS_DIR,
                    temp_tiles_folder=TEMP_TILES_FOLDER,
                    white_threshold=220,
                    cut_size=70
                )
                print(f"\n✅ Successfully processed {os.path.basename(wsi_path)}")
            except Exception as e:
                print(f"\n❌ Error processing {os.path.basename(wsi_path)}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n\n{'='*80}")
        print(f"✅ Finished processing {len(unprocessed_wsi_files)} remaining WSI files!")
        print(f"Total processed: {len(processed_names) + len(unprocessed_wsi_files)}/{len(all_wsi_files)}")
        print(f"{'='*80}\n")
    
    
    ### ========================================================================
    ### Step 4: Summary Boxplots - Cell Type Proportions Across All Samples
    ### ========================================================================
    
    print(f"\n{'#'*80}")
    print(f"Step 4: Creating Summary Boxplots of Cell Type Proportions")
    print(f"{'#'*80}\n")
    
    # Get all prediction CSV files
    csv_files = sorted(glob.glob(os.path.join(OUTPUT_PREDICTIONS_DIR, "*_predictions.csv")))
    print(f"Found {len(csv_files)} prediction CSV files")
    
    if len(csv_files) > 0:
        # Load and concatenate all prediction CSVs
        all_predictions = []
        
        for csv_file in csv_files:
            sample_name = os.path.basename(csv_file).replace('_predictions.csv', '')
            df_sample = pd.read_csv(csv_file)
            df_sample['Sample'] = sample_name
            all_predictions.append(df_sample)
        
        df_all = pd.concat(all_predictions, ignore_index=True)
        print(f"Total tiles across all samples: {len(df_all):,}")
        
        # Define 5 cell types for COAD (predictions already in these 5 types, no merging needed)
        # Order: Cancer, Stromal, pan-APC, T_Cells, Normal_Epithelial
        cell_types_5 = ['Cancer', 'Stromal', 'pan-APC', 'T_Cells', 'Normal_Epithelial']
        
        # Filter out white tiles (tiles with NaN values)
        valid_mask = df_all[cell_types_5].notna().all(axis=1)
        df_valid = df_all[valid_mask].copy()
        print(f"Valid tiles (non-white): {len(df_valid):,}")
        print(f"White tiles: {len(df_all) - len(df_valid):,}")
        
        # =====================================================================
        # Boxplot 1: Calibrated, Not Normalized (5 cell types)
        # =====================================================================
        
        print(f"\n{'='*80}")
        print("Creating calibrated (not normalized) boxplot (5 cell types)")
        print(f"{'='*80}\n")
        
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        
        # Prepare data for boxplot
        plot_data_1 = []
        positions_1 = []
        
        for i, cell_type in enumerate(cell_types_5):
            values = df_valid[cell_type].values
            plot_data_1.append(values)
            positions_1.append(i + 1)
        
        # Create boxplot
        bp1 = ax1.boxplot(plot_data_1, positions=positions_1, widths=0.6,
                          patch_artist=True,
                          showmeans=True,
                          meanprops=dict(marker='D', markerfacecolor='red', markersize=10),
                          medianprops=dict(color='black', linewidth=2.5),
                          boxprops=dict(linewidth=2),
                          whiskerprops=dict(linewidth=2),
                          capprops=dict(linewidth=2),
                          showfliers=False)
        
        # Color boxes with distinct colors (in order: Cancer, Stromal, pan-APC, T_Cells, Normal_Epithelial)
        colors_5 = [
            '#E41A1C',   # Cancer - Red
            '#BCBD22',   # Stromal - Olive/Yellow-green
            '#9467BD',   # pan-APC - Purple
            '#4DAF4A',   # T Cells - Green
            '#A65628'    # Normal Epithelial - Brown
        ]
        
        for patch, color in zip(bp1['boxes'], colors_5):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Set labels
        ax1.set_xticks(positions_1)
        display_names = ['Cancer', 'Stromal', 'pan-APC', 'T Cells', 'Normal\nEpithelial']
        ax1.set_xticklabels(display_names, fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cell Type Proportion (Calibrated, Non-Normalized)', fontsize=14, fontweight='bold')
        ax1.set_title(f'COAD Cell Type Proportion Distribution (n={len(csv_files)} samples, {len(df_valid):,} tiles)', 
                     fontsize=15, fontweight='bold', pad=20)
        
        # Add grid
        ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
        ax1.set_ylim(-0.05, max(df_valid[cell_types_5].max().max() + 0.15, 1.05))
        
        # Style
        ax1.tick_params(axis='both', which='major', labelsize=12, width=2)
        for spine in ax1.spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
        
        # Add statistics text
        print(f"\nCalibrated (Not Normalized) Summary Statistics:")
        for cell_type in cell_types_5:
            values = df_valid[cell_type]
            print(f"  {cell_type:20s}: Mean={values.mean():.4f}, Median={values.median():.4f}, Std={values.std():.4f}")
        
        plt.tight_layout()
        
        # Save figure
        boxplot_1_output = os.path.join(OUTPUT_PREDICTIONS_DIR, "All_Samples_CellType_Calibrated_NotNormalized_Boxplot.png")
        plt.savefig(boxplot_1_output, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Boxplot 1 saved: {boxplot_1_output}")
        
        # =====================================================================
        # Boxplot 2: Calibrated and Normalized (5 cell types, sum to 1)
        # =====================================================================
        
        print(f"\n{'='*80}")
        print("Creating calibrated and normalized boxplot (5 cell types, sum to 1)")
        print(f"{'='*80}\n")
        
        # Normalize 5 cell types so each tile sums to 1
        df_valid_normalized = df_valid.copy()
        
        # Calculate sum of 5 cell types for each tile
        cell_types_sum = sum(df_valid_normalized[ct] for ct in cell_types_5)
        
        # Normalize each cell type by the sum
        for cell_type in cell_types_5:
            df_valid_normalized[f'{cell_type}_normalized'] = df_valid_normalized[cell_type] / cell_types_sum
        
        # Verify normalization (should sum to 1 for each tile)
        normalized_sum = sum(df_valid_normalized[f'{ct}_normalized'].values for ct in cell_types_5)
        print(f"Normalization check - Min sum: {normalized_sum.min():.6f}, Max sum: {normalized_sum.max():.6f}")
        print(f"All tiles sum to 1: {np.allclose(normalized_sum, 1.0)}")
        
        # Create boxplot
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        
        # Prepare data for boxplot
        plot_data_2 = []
        positions_2 = []
        
        for i, cell_type in enumerate(cell_types_5):
            values = df_valid_normalized[f'{cell_type}_normalized'].values
            plot_data_2.append(values)
            positions_2.append(i + 1)
        
        # Create boxplot
        bp2 = ax2.boxplot(plot_data_2, positions=positions_2, widths=0.6,
                          patch_artist=True,
                          showmeans=True,
                          meanprops=dict(marker='D', markerfacecolor='red', markersize=10),
                          medianprops=dict(color='black', linewidth=2.5),
                          boxprops=dict(linewidth=2),
                          whiskerprops=dict(linewidth=2),
                          capprops=dict(linewidth=2),
                          showfliers=False)
        
        # Color boxes with same colors
        for patch, color in zip(bp2['boxes'], colors_5):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Set labels
        ax2.set_xticks(positions_2)
        ax2.set_xticklabels(display_names, fontsize=14, fontweight='bold')
        ax2.set_ylabel('Cell Type Proportion (Normalized, Sum = 1)', fontsize=14, fontweight='bold')
        ax2.set_title(f'COAD Normalized Cell Type Proportion Distribution (n={len(csv_files)} samples, {len(df_valid):,} tiles)', 
                     fontsize=15, fontweight='bold', pad=20)
        
        # Add grid
        ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
        ax2.set_ylim(-0.05, 1.05)
        
        # Style
        ax2.tick_params(axis='both', which='major', labelsize=12, width=2)
        for spine in ax2.spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
        
        # Add statistics text
        print(f"\nCalibrated and Normalized Summary Statistics:")
        for cell_type in cell_types_5:
            values = df_valid_normalized[f'{cell_type}_normalized']
            print(f"  {cell_type:20s}: Mean={values.mean():.4f}, Median={values.median():.4f}, Std={values.std():.4f}")
        
        plt.tight_layout()
        
        # Save figure
        boxplot_2_output = os.path.join(OUTPUT_PREDICTIONS_DIR, "All_Samples_CellType_Calibrated_Normalized_Boxplot.png")
        plt.savefig(boxplot_2_output, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Boxplot 2 saved: {boxplot_2_output}")
    else:
        print("⚠️  No prediction CSV files found. Skipping boxplots.")
    
    
    ### ========================================================================
    ### Step 5: Group & Normalize Predictions + Generate Hexagon Heatmaps
    ### ========================================================================
    
    print(f"\n{'#'*80}")
    print(f"Step 5: Group & Normalize Cell Type Proportions + Generate Hexagon Heatmaps")
    print(f"{'#'*80}\n")
    
    # Define output directories
    OUTPUT_GROUPED_PREDICTIONS_DIR = "/Users/scui2/Desktop/TCGA/TCGA_COAD/TCGA_COAD_Grouped_Normalized_Predictions"
    OUTPUT_HEXAGON_BASE_DIR = "/Users/scui2/Desktop/TCGA/TCGA_COAD/TCGA_COAD_Hexagon_Heatmaps"
    
    os.makedirs(OUTPUT_GROUPED_PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_HEXAGON_BASE_DIR, exist_ok=True)
    
    # Create 5 subdirectories for hexagon heatmaps (one per cell type)
    # Order: Cancer, Stromal, pan-APC, T_Cells, Normal_Epithelial
    hexagon_subdirs = {
        'Cancer': os.path.join(OUTPUT_HEXAGON_BASE_DIR, '01_Cancer'),
        'Stromal': os.path.join(OUTPUT_HEXAGON_BASE_DIR, '02_Stromal'),
        'pan-APC': os.path.join(OUTPUT_HEXAGON_BASE_DIR, '03_pan-APC'),
        'T_Cells': os.path.join(OUTPUT_HEXAGON_BASE_DIR, '04_T_Cells'),
        'Normal_Epithelial': os.path.join(OUTPUT_HEXAGON_BASE_DIR, '05_Normal_Epithelial')
    }
    
    for subdir in hexagon_subdirs.values():
        os.makedirs(subdir, exist_ok=True)
    
    print(f"Output directories created:")
    print(f"  - Grouped & Normalized CSVs: {OUTPUT_GROUPED_PREDICTIONS_DIR}")
    print(f"  - Hexagon Heatmaps: {OUTPUT_HEXAGON_BASE_DIR}")
    for cell_type, subdir in hexagon_subdirs.items():
        print(f"    * {cell_type}: {subdir}")
    print()
    
    # Get all prediction CSV files
    csv_files = sorted(glob.glob(os.path.join(OUTPUT_PREDICTIONS_DIR, "*_predictions.csv")))
    print(f"Found {len(csv_files)} prediction CSV files to process\n")
    
    if len(csv_files) > 0:
        for i, csv_file in enumerate(csv_files, 1):
            sample_name = os.path.basename(csv_file).replace('_predictions.csv', '')
            
            print(f"\n{'='*80}")
            print(f"Processing {i}/{len(csv_files)}: {sample_name}")
            print(f"{'='*80}\n")
            
            try:
                # Read prediction CSV (already has 5 cell types)
                df = pd.read_csv(csv_file)
                print(f"  Loaded {len(df)} tiles")
                
                # Define 5 cell types for COAD (no merging needed)
                # Order: Cancer, Stromal, pan-APC, T_Cells, Normal_Epithelial
                cell_types_5 = ['Cancer', 'Stromal', 'pan-APC', 'T_Cells', 'Normal_Epithelial']
                
                # Identify white tiles
                white_mask = df[cell_types_5].isna().any(axis=1)
                valid_mask = ~white_mask
                
                print(f"  Valid tiles: {valid_mask.sum()}")
                print(f"  White tiles: {white_mask.sum()}")
                
                # Use cell_types_5 directly (no merging needed for COAD)
                merged_groups = cell_types_5
                
                # Normalize to sum = 1 for valid tiles only
                for idx in df.index:
                    if valid_mask[idx]:
                        # Calculate sum of 5 merged groups
                        group_sum = sum(df.at[idx, group] for group in merged_groups)
                        
                        if group_sum > 0:
                            # Normalize each group
                            for group in merged_groups:
                                df.at[idx, f'{group}_normalized'] = df.at[idx, group] / group_sum
                        else:
                            # If sum is 0, set all to NaN
                            for group in merged_groups:
                                df.at[idx, f'{group}_normalized'] = np.nan
                    else:
                        # White tiles - set normalized values to NaN
                        for group in merged_groups:
                            df.at[idx, f'{group}_normalized'] = np.nan
                
                # Verify normalization for valid tiles
                valid_df = df[valid_mask].copy()
                if len(valid_df) > 0:
                    normalized_sum = sum(valid_df[f'{group}_normalized'] for group in merged_groups)
                    print(f"  Normalization check - Min: {normalized_sum.min():.6f}, Max: {normalized_sum.max():.6f}")
                    print(f"  All valid tiles sum to 1: {np.allclose(normalized_sum, 1.0)}")
                
                # Parse tile coordinates from tile_name
                rows = []
                cols = []
                for tile_name in df['tile_name']:
                    clean_name = tile_name.replace('_White', '').replace('.tif', '')
                    parts = clean_name.split('-')
                    if len(parts) == 2:
                        rows.append(int(parts[0]))
                        cols.append(int(parts[1]))
                    else:
                        rows.append(np.nan)
                        cols.append(np.nan)
                df['row'] = rows
                df['col'] = cols
                
                # Select columns to save: tile_name, row, col, 5 normalized proportions
                columns_to_save = ['tile_name', 'row', 'col'] + [f'{group}_normalized' for group in merged_groups]
                df_output = df[columns_to_save].copy()
                
                # Save grouped & normalized CSV
                output_csv_path = os.path.join(OUTPUT_GROUPED_PREDICTIONS_DIR, f"{sample_name}_grouped_normalized.csv")
                df_output.to_csv(output_csv_path, index=False)
                print(f"  ✅ Saved: {output_csv_path}")
                
                # Generate 5 hexagon heatmaps (one per cell type)
                print(f"\n  Generating hexagon heatmaps...")
                
                # Get grid dimensions
                max_row = int(df['row'].max())
                max_col = int(df['col'].max())
                estimated_grid_size = max(max_row, max_col)
                
                # Calculate marker size
                base_size = 150
                size_factor = (40 / estimated_grid_size) ** 1.5
                marker_size = max(20, int(base_size * size_factor))
                
                # Color scheme for each cell type (Order: Cancer, Stromal, pan-APC, T_Cells, Normal_Epithelial)
                cell_type_colors = {
                    'Cancer': '#E41A1C',                # Red
                    'Stromal': '#BCBD22',              # Olive/Yellow-green
                    'pan-APC': '#9467BD',              # Purple
                    'T_Cells': '#4DAF4A',              # Green
                    'Normal_Epithelial': '#A65628'     # Brown
                }
                
                # Generate one heatmap per cell type
                for cell_type in merged_groups:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Prepare data for scatter plot (exclude White tiles)
                    plot_data = []
                    for idx, row_data in df.iterrows():
                        if pd.isna(row_data['row']) or pd.isna(row_data['col']):
                            continue
                        
                        # Skip White tiles
                        if white_mask[idx]:
                            continue
                        
                        proportion = row_data[f'{cell_type}_normalized']
                        if pd.notna(proportion):
                            plot_data.append({
                                'x': row_data['col'],
                                'y': row_data['row'],
                                'proportion': proportion
                            })
                    
                    if len(plot_data) == 0:
                        plt.close()
                        print(f"    ⚠️  No valid data for {cell_type}, skipping...")
                        continue
                    
                    df_plot = pd.DataFrame(plot_data)
                    
                    # Get min and max values for this cell type in this sample
                    vmin_val = df_plot['proportion'].min()
                    vmax_val = df_plot['proportion'].max()
                    
                    # Create scatter plot with color intensity based on proportion
                    scatter = ax.scatter(
                        x=df_plot['x'],
                        y=df_plot['y'],
                        c=df_plot['proportion'],
                        cmap='RdBu_r',  # Blue-to-Red colormap (reversed RdBu)
                        marker='h',
                        s=marker_size,
                        edgecolors='black',
                        linewidth=0.5,
                        vmin=vmin_val,
                        vmax=vmax_val
                    )
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('Normalized Proportion', fontsize=28, fontweight='black', color='black')
                    cbar.ax.tick_params(labelsize=26, width=2, length=6, color='black', labelcolor='black')
                    # Make colorbar tick labels bold and black - using multiple methods to ensure it works
                    for label in cbar.ax.get_yticklabels():
                        label.set_fontsize(26)
                        label.set_fontweight('black')
                        label.set_color('black')
                    plt.setp(cbar.ax.get_yticklabels(), fontweight='black', fontsize=26, color='black')
                    # Also set the colorbar outline to be thicker and black
                    cbar.outline.set_linewidth(2)
                    cbar.outline.set_edgecolor('black')
                    
                    # Invert y-axis for correct orientation
                    ax.invert_yaxis()
                    
                    # Set axis style
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(3)
                        spine.set_color('black')
                    
                    # Remove axis labels, ticks, and title
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    plt.tight_layout()
                    
                    # Save figure
                    output_path = os.path.join(hexagon_subdirs[cell_type], f"{sample_name}_{cell_type}_heatmap.png")
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    display_name = cell_type.replace('_', ' ')
                    print(f"    ✅ {display_name}: {output_path}")
                
            except Exception as e:
                print(f"  ❌ Error processing {sample_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n{'='*80}")
        print(f"✅ Step 5 Complete!")
        print(f"  - Processed {len(csv_files)} samples")
        print(f"  - Grouped & Normalized CSVs saved to: {OUTPUT_GROUPED_PREDICTIONS_DIR}")
        print(f"  - Hexagon Heatmaps saved to: {OUTPUT_HEXAGON_BASE_DIR}")
        print(f"{'='*80}\n")
    else:
        print("⚠️  No prediction CSV files found. Skipping Step 5.")
    
    
    ### ========================================================================
    ### Step 6: Hard Classification + Weighted Minimum Distance Analysis
    ### ========================================================================
    
    print(f"\n{'#'*80}")
    print(f"Step 6: Hard Classification + Weighted Minimum Distance Analysis")
    print(f"{'#'*80}\n")
    
    # Define output directories
    OUTPUT_DISTANCE_BASE_DIR = "/Users/scui2/Desktop/TCGA/TCGA_COAD/TCGA_COAD_Weighted_Distances"
    OUTPUT_CLASSIFICATION_HEXMAP_BASE_DIR = "/Users/scui2/Desktop/TCGA/TCGA_COAD/TCGA_COAD_Classification_Hexagon_Maps"
    OUTPUT_DISTANCE_HEATMAP_BASE_DIR = "/Users/scui2/Desktop/TCGA/TCGA_COAD/TCGA_COAD_Distance_Hexagon_Heatmaps"
    OUTPUT_DISTANCE_SUMMARY_CSV = os.path.join(OUTPUT_DISTANCE_BASE_DIR, "Distance_Summary_All_Samples.csv")
    
    os.makedirs(OUTPUT_DISTANCE_BASE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_CLASSIFICATION_HEXMAP_BASE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DISTANCE_HEATMAP_BASE_DIR, exist_ok=True)
    
    # Define 4 cell types for distance analysis (excluding Normal_Epithelial)
    # Order: Cancer, Stromal, pan-APC, T_Cells
    cell_types_distance = ['Cancer', 'Stromal', 'pan-APC', 'T_Cells']
    
    # All 5 cell types for overall proportions
    cell_types_all = ['Cancer', 'Stromal', 'pan-APC', 'T_Cells', 'Normal_Epithelial']
    
    # Hard classification thresholds (for calibrated and normalized proportions)
    classification_thresholds = {
        'Cancer': 0.30,                # >30% cancer
        'Stromal': 0.40,               # >40% stromal
        'pan-APC': 0.30,               # >30% pan-APC
        'T_Cells': 0.10                # >10% T cells
    }
    
    # Define colors for each cell type in classification maps
    cell_type_colors = {
        'Cancer': '#E41A1C',                # Red
        'Stromal': '#FFFF33',              # Yellow
        'pan-APC': '#FF7F00',              # Orange
        'T_Cells': '#4DAF4A'               # Green
    }
    
    # Create 12 subdirectories for directional distance distributions (4 cell types → 4×3=12 pairs)
    # Format: From_CellType_To_CellType (excluding Normal_Epithelial)
    directional_pairs = []
    for source in cell_types_distance:
        for target in cell_types_distance:
            if source != target:
                directional_pairs.append((source, target))
    
    print(f"Total directional pairs: {len(directional_pairs)}")
    print(f"\nCreating subdirectories for distance distributions:")
    for source, target in directional_pairs:
        subdir = os.path.join(OUTPUT_DISTANCE_BASE_DIR, f"From_{source}_To_{target}")
        os.makedirs(subdir, exist_ok=True)
        print(f"  Distributions: From_{source}_To_{target}")
    
    print(f"\nCreating subdirectories for distance hexagon heatmaps:")
    for source, target in directional_pairs:
        subdir = os.path.join(OUTPUT_DISTANCE_HEATMAP_BASE_DIR, f"From_{source}_To_{target}")
        os.makedirs(subdir, exist_ok=True)
        print(f"  Heatmaps: From_{source}_To_{target}")
    
    print(f"\nCreating subdirectories for classification hexagon maps (one per cell type):")
    for cell_type in cell_types_distance:
        subdir = os.path.join(OUTPUT_CLASSIFICATION_HEXMAP_BASE_DIR, cell_type)
        os.makedirs(subdir, exist_ok=True)
        print(f"  Classification maps: {cell_type}")
    
    # Get all grouped & normalized prediction CSV files
    grouped_csv_files = sorted(glob.glob(os.path.join(OUTPUT_GROUPED_PREDICTIONS_DIR, "*_grouped_normalized.csv")))
    print(f"\nFound {len(grouped_csv_files)} grouped & normalized prediction CSV files\n")
    
    # Check if summary CSV already exists (for resume capability)
    if os.path.exists(OUTPUT_DISTANCE_SUMMARY_CSV):
        df_existing = pd.read_csv(OUTPUT_DISTANCE_SUMMARY_CSV)
        processed_samples = set(df_existing['Sample_ID'].values)
        print(f"Found existing summary CSV with {len(processed_samples)} processed samples")
    else:
        processed_samples = set()
        print(f"No existing summary CSV found, starting fresh")
    
    # Filter out already processed samples
    remaining_csv_files = []
    for csv_file in grouped_csv_files:
        sample_name = os.path.basename(csv_file).replace('_grouped_normalized.csv', '')
        if sample_name not in processed_samples:
            remaining_csv_files.append(csv_file)
    
    print(f"Already processed: {len(processed_samples)} samples")
    print(f"Remaining to process: {len(remaining_csv_files)} samples\n")
    
    if len(remaining_csv_files) > 0:
        for i, csv_file in enumerate(remaining_csv_files, 1):
            sample_name = os.path.basename(csv_file).replace('_grouped_normalized.csv', '')
            
            print(f"\n{'='*80}")
            print(f"Processing {i}/{len(remaining_csv_files)}: {sample_name}")
            print(f"(Overall progress: {len(processed_samples) + i}/{len(grouped_csv_files)})")
            print(f"{'='*80}\n")
            
            try:
                # Read grouped & normalized predictions
                df = pd.read_csv(csv_file)
                print(f"  Loaded {len(df)} tiles")
                
                # Filter out tiles with NaN values (white tiles) - need all 5 cell types
                valid_mask = df[[f'{ct}_normalized' for ct in cell_types_all]].notna().all(axis=1)
                df_valid = df[valid_mask].copy()
                print(f"  Valid tiles: {len(df_valid)}")
                print(f"  White tiles: {len(df) - len(df_valid)}")
                
                if len(df_valid) == 0:
                    print(f"  ⚠️  No valid tiles, skipping...")
                    continue
                
                # Dictionary to store this sample's metrics
                sample_metrics = {'Sample_ID': sample_name}
                
                # Calculate overall proportions (mean across all valid tiles) for all 5 cell types
                print(f"\n  Calculating overall proportions...")
                for cell_type in cell_types_all:
                    overall_prop = df_valid[f'{cell_type}_normalized'].mean()
                    sample_metrics[f'{cell_type}_overall_proportion'] = overall_prop
                    print(f"    {cell_type}: {overall_prop:.4f}")
                
                # Hard classification: Assign each tile to a cell type category (can belong to multiple)
                print(f"\n  Performing hard classification...")
                classified_tiles = {ct: [] for ct in cell_types_distance}
                
                for idx, row_data in df_valid.iterrows():
                    for cell_type in cell_types_distance:
                        if row_data[f'{cell_type}_normalized'] > classification_thresholds[cell_type]:
                            classified_tiles[cell_type].append(idx)
                
                # Print classification summary
                for cell_type in cell_types_distance:
                    print(f"    {cell_type} tiles (>{classification_thresholds[cell_type]*100:.0f}%): {len(classified_tiles[cell_type])}")
                
                # Generate classification hexagon map for each cell type
                print(f"\n  Generating classification hexagon maps...")
                
                for cell_type in cell_types_distance:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Get grid dimensions
                    max_row = int(df_valid['row'].max())
                    max_col = int(df_valid['col'].max())
                    estimated_grid_size = max(max_row, max_col)
                    
                    # Calculate marker size
                    base_size = 150
                    size_factor = (40 / estimated_grid_size) ** 1.5
                    marker_size = max(20, int(base_size * size_factor))
                    
                    # Plot all valid tiles in light gray first
                    ax.scatter(
                        x=df_valid['col'],
                        y=df_valid['row'],
                        c='#D3D3D3',  # Light gray
                        marker='h',
                        s=marker_size,
                        edgecolors='black',
                        linewidth=0.3,
                        alpha=0.5,
                        zorder=1
                    )
                    
                    # Overlay classified tiles in the cell type's color
                    if len(classified_tiles[cell_type]) > 0:
                        classified_df = df_valid.loc[classified_tiles[cell_type]]
                        ax.scatter(
                            x=classified_df['col'],
                            y=classified_df['row'],
                            c=cell_type_colors[cell_type],
                            marker='h',
                            s=marker_size,
                            edgecolors='black',
                            linewidth=0.5,
                            alpha=0.8,
                            zorder=2,
                            label=f'{cell_type} (n={len(classified_tiles[cell_type])})'
                        )
                    
                    # Invert y-axis
                    ax.invert_yaxis()
                    
                    # Set axis style
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(3)
                        spine.set_color('black')
                    
                    # Remove axis labels and ticks
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    # Add legend
                    ax.legend(fontsize=12, loc='upper right')
                    
                    plt.tight_layout()
                    
                    # Save classification map
                    output_classif_subdir = os.path.join(OUTPUT_CLASSIFICATION_HEXMAP_BASE_DIR, cell_type)
                    output_classif_path = os.path.join(output_classif_subdir, f"{sample_name}_{cell_type}_classification.png")
                    plt.savefig(output_classif_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"    ✅ {cell_type} classification map saved")
                
                # Calculate weighted minimum distances for all 12 pairs
                print(f"\n  Calculating weighted minimum distances...")
                for source, target in directional_pairs:
                    print(f"\n    Computing: {source} → {target}")
                    
                    # Get source and target tiles from hard classification
                    source_tile_indices = classified_tiles[source]
                    target_tile_indices = classified_tiles[target]
                    
                    print(f"      Source tiles ({source}): {len(source_tile_indices)}")
                    print(f"      Target tiles ({target}): {len(target_tile_indices)}")
                    
                    # Check if we have both source and target tiles
                    if len(source_tile_indices) == 0 or len(target_tile_indices) == 0:
                        print(f"      ⚠️  Insufficient tiles, skipping this pair")
                        sample_metrics[f'{source}_to_{target}_mean'] = np.nan
                        sample_metrics[f'{source}_to_{target}_std'] = np.nan
                        sample_metrics[f'{source}_to_{target}_median'] = np.nan
                        sample_metrics[f'{source}_to_{target}_iqr'] = np.nan
                        continue
                    
                    # Get source and target dataframes
                    source_tiles_df = df_valid.loc[source_tile_indices]
                    target_tiles_df = df_valid.loc[target_tile_indices]
                    
                    # Calculate weighted minimum distances for each source tile
                    weighted_min_distances = []
                    
                    for idx_source, source_tile in source_tiles_df.iterrows():
                        source_row = source_tile['row']
                        source_col = source_tile['col']
                        source_proportion = source_tile[f'{source}_normalized']
                        
                        # Find minimum distance to any target tile
                        min_distance = np.inf
                        
                        for idx_target, target_tile in target_tiles_df.iterrows():
                            target_row = target_tile['row']
                            target_col = target_tile['col']
                            
                            # Euclidean distance
                            distance = np.sqrt((target_row - source_row)**2 + (target_col - source_col)**2)
                            
                            if distance < min_distance:
                                min_distance = distance
                        
                        # Weight by source tile's source cell type proportion
                        weighted_min_dist = min_distance * source_proportion
                        weighted_min_distances.append(weighted_min_dist)
                    
                    # Convert to numpy array
                    weighted_min_distances = np.array(weighted_min_distances)
                    
                    print(f"      Weighted min distances computed: {len(weighted_min_distances)} values")
                    
                    # Calculate summary statistics
                    mean_dist = np.mean(weighted_min_distances)
                    std_dist = np.std(weighted_min_distances)
                    median_dist = np.median(weighted_min_distances)
                    q25 = np.percentile(weighted_min_distances, 25)
                    q75 = np.percentile(weighted_min_distances, 75)
                    iqr_dist = q75 - q25
                    
                    print(f"      Mean: {mean_dist:.2f}, Std: {std_dist:.2f}, Median: {median_dist:.2f}, IQR: {iqr_dist:.2f}")
                    
                    # Store metrics
                    sample_metrics[f'{source}_to_{target}_mean'] = mean_dist
                    sample_metrics[f'{source}_to_{target}_std'] = std_dist
                    sample_metrics[f'{source}_to_{target}_median'] = median_dist
                    sample_metrics[f'{source}_to_{target}_iqr'] = iqr_dist
                    
                    # Create distribution plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    ax.hist(weighted_min_distances, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
                    
                    # Add vertical lines for statistics
                    ax.axvline(mean_dist, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dist:.2f}')
                    ax.axvline(median_dist, color='green', linestyle='--', linewidth=2, label=f'Median: {median_dist:.2f}')
                    ax.axvline(q25, color='orange', linestyle=':', linewidth=1.5, label=f'Q25: {q25:.2f}')
                    ax.axvline(q75, color='orange', linestyle=':', linewidth=1.5, label=f'Q75: {q75:.2f}')
                    
                    ax.set_xlabel('Weighted Minimum Distance (tiles)', fontsize=13, fontweight='bold')
                    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
                    ax.set_title(f'{sample_name}\n{source} → {target} (n={len(weighted_min_distances)})', 
                                fontsize=14, fontweight='bold')
                    ax.legend(fontsize=10, loc='upper right')
                    ax.grid(axis='y', alpha=0.3)
                    
                    # Style
                    ax.tick_params(axis='both', which='major', labelsize=11)
                    for spine in ax.spines.values():
                        spine.set_linewidth(1.5)
                    
                    plt.tight_layout()
                    
                    # Save figure
                    output_subdir = os.path.join(OUTPUT_DISTANCE_BASE_DIR, f"From_{source}_To_{target}")
                    output_path = os.path.join(output_subdir, f"{sample_name}_{source}_to_{target}_dist.png")
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"      ✅ Distribution plot saved")
                    
                    # Generate hexagon heatmap for this directional distance
                    print(f"      Generating distance hexagon heatmap...")
                    
                    # Create a dictionary to store weighted minimum distance for each source tile
                    tile_min_distances = {}
                    
                    for idx_source, source_tile in source_tiles_df.iterrows():
                        source_row = source_tile['row']
                        source_col = source_tile['col']
                        source_proportion = source_tile[f'{source}_normalized']
                        
                        # Find minimum distance to any target tile
                        min_distance = np.inf
                        
                        for idx_target, target_tile in target_tiles_df.iterrows():
                            target_row = target_tile['row']
                            target_col = target_tile['col']
                            
                            # Euclidean distance
                            distance = np.sqrt((target_row - source_row)**2 + (target_col - source_col)**2)
                            
                            if distance < min_distance:
                                min_distance = distance
                        
                        # Weight by source tile's source cell type proportion
                        weighted_min_dist = min_distance * source_proportion
                        tile_min_distances[(source_row, source_col)] = weighted_min_dist
                    
                    # Prepare data for hexagon plot
                    plot_data_hex = []
                    for (row, col), dist in tile_min_distances.items():
                        plot_data_hex.append({
                            'x': col,
                            'y': row,
                            'distance': dist
                        })
                    
                    if len(plot_data_hex) > 0:
                        df_plot_hex = pd.DataFrame(plot_data_hex)
                        
                        # Get grid dimensions from the full valid dataframe
                        max_row = int(df_valid['row'].max())
                        max_col = int(df_valid['col'].max())
                        estimated_grid_size = max(max_row, max_col)
                        
                        # Calculate marker size
                        base_size = 150
                        size_factor = (40 / estimated_grid_size) ** 1.5
                        marker_size = max(20, int(base_size * size_factor))
                        
                        # Get min and max distance for colormap scaling
                        vmin_dist = df_plot_hex['distance'].min()
                        vmax_dist = df_plot_hex['distance'].max()
                        
                        # Create hexagon heatmap
                        fig_hex, ax_hex = plt.subplots(figsize=(10, 8))
                        
                        # First, plot all other valid tiles (not selected as source) in gray
                        source_tile_coords = set(tile_min_distances.keys())
                        other_tiles = []
                        for idx, row_data in df_valid.iterrows():
                            coord = (row_data['row'], row_data['col'])
                            if coord not in source_tile_coords:
                                other_tiles.append({
                                    'x': row_data['col'],
                                    'y': row_data['row']
                                })
                        
                        if len(other_tiles) > 0:
                            df_other = pd.DataFrame(other_tiles)
                            ax_hex.scatter(
                                x=df_other['x'],
                                y=df_other['y'],
                                c='#D3D3D3',  # Light gray
                                marker='h',
                                s=marker_size,
                                edgecolors='black',
                                linewidth=0.3,
                                alpha=0.5,
                                zorder=1
                            )
                        
                        # Then, plot source tiles with distance-based coloring on top
                        scatter_hex = ax_hex.scatter(
                            x=df_plot_hex['x'],
                            y=df_plot_hex['y'],
                            c=df_plot_hex['distance'],
                            cmap='RdBu',  # Red (low/short) to Blue (high/long)
                            marker='h',
                            s=marker_size,
                            edgecolors='black',
                            linewidth=0.5,
                            vmin=vmin_dist,
                            vmax=vmax_dist,
                            zorder=2
                        )
                        
                        # Add colorbar
                        cbar_hex = plt.colorbar(scatter_hex, ax=ax_hex, fraction=0.046, pad=0.04)
                        cbar_hex.set_label('Minimum Distance', fontsize=28, fontweight='black', color='black')
                        cbar_hex.ax.tick_params(labelsize=26, width=2, length=6, color='black', labelcolor='black')
                        # Make colorbar tick labels bold and black - using multiple methods to ensure it works
                        for label in cbar_hex.ax.get_yticklabels():
                            label.set_fontsize(26)
                            label.set_fontweight('black')  # 'black' is the heaviest weight (900)
                            label.set_color('black')
                        plt.setp(cbar_hex.ax.get_yticklabels(), fontweight='black', fontsize=26, color='black')
                        # Also set the colorbar outline to be thicker and black
                        cbar_hex.outline.set_linewidth(2)
                        cbar_hex.outline.set_edgecolor('black')
                        
                        # Invert y-axis for correct orientation
                        ax_hex.invert_yaxis()
                        
                        # Set axis style
                        for spine in ax_hex.spines.values():
                            spine.set_visible(True)
                            spine.set_linewidth(3)
                            spine.set_color('black')
                        
                        # Remove axis labels and ticks
                        ax_hex.set_xlabel('')
                        ax_hex.set_ylabel('')
                        ax_hex.set_xticks([])
                        ax_hex.set_yticks([])
                        
                        plt.tight_layout()
                        
                        # Save hexagon heatmap
                        output_heatmap_subdir = os.path.join(OUTPUT_DISTANCE_HEATMAP_BASE_DIR, f"From_{source}_To_{target}")
                        output_heatmap_path = os.path.join(output_heatmap_subdir, f"{sample_name}_{source}_to_{target}_heatmap.png")
                        plt.savefig(output_heatmap_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"      ✅ Heatmap saved")
                    else:
                        print(f"      ⚠️  No data for heatmap, skipping...")
                
                # Save this sample's metrics to CSV immediately (incremental save)
                cols = ['Sample_ID']
                
                # Add overall proportions (5 cell types)
                for cell_type in cell_types_all:
                    cols.append(f'{cell_type}_overall_proportion')
                
                # Add distance metrics (48 columns: 12 pairs × 4 stats)
                for source, target in directional_pairs:
                    cols.append(f'{source}_to_{target}_mean')
                    cols.append(f'{source}_to_{target}_std')
                    cols.append(f'{source}_to_{target}_median')
                    cols.append(f'{source}_to_{target}_iqr')
                
                # Convert sample_metrics to DataFrame row
                df_new_row = pd.DataFrame([sample_metrics])
                df_new_row = df_new_row[cols]
                
                # Append to existing CSV or create new one
                if os.path.exists(OUTPUT_DISTANCE_SUMMARY_CSV):
                    # Read existing CSV
                    df_existing = pd.read_csv(OUTPUT_DISTANCE_SUMMARY_CSV)
                    # Append new row
                    df_updated = pd.concat([df_existing, df_new_row], ignore_index=True)
                    # Save
                    df_updated.to_csv(OUTPUT_DISTANCE_SUMMARY_CSV, index=False)
                else:
                    # Create new CSV
                    df_new_row.to_csv(OUTPUT_DISTANCE_SUMMARY_CSV, index=False)
                
                print(f"\n  💾 Sample metrics saved to CSV (total samples in CSV: {len(pd.read_csv(OUTPUT_DISTANCE_SUMMARY_CSV))})")
                
            except Exception as e:
                print(f"  ❌ Error processing {sample_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Final summary
        if os.path.exists(OUTPUT_DISTANCE_SUMMARY_CSV):
            df_final = pd.read_csv(OUTPUT_DISTANCE_SUMMARY_CSV)
            print(f"\n{'='*80}")
            print(f"✅ Step 6 Complete!")
            print(f"  - Total samples in summary CSV: {len(df_final)}")
            print(f"  - Classification hexagon maps: 4 cell types × {len(df_final)} samples")
            print(f"  - Distance distributions: {len(directional_pairs)} directional pairs × {len(df_final)} samples")
            print(f"  - Distance hexagon heatmaps: {len(directional_pairs)} directional pairs × {len(df_final)} samples")
            print(f"  - Summary CSV columns: {df_final.shape[1]} (5 overall proportions + 48 distance metrics)")
            print(f"  - Summary CSV: {OUTPUT_DISTANCE_SUMMARY_CSV}")
            print(f"{'='*80}\n")
        else:
            print("⚠️  No samples successfully processed.")
    else:
        print("⚠️  All samples already processed or no new samples to process.")
    
    
    ### ========================================================================
    ### Step 7: Distance Metrics vs sqrt(Valid Tiles) Scatter Plots
    ### ========================================================================
    
    print(f"\n{'#'*80}")
    print(f"Step 7: Distance Metrics vs sqrt(Valid Tiles) Analysis")
    print(f"{'#'*80}\n")
    
    # Define output directory for scatter plots
    OUTPUT_SCATTER_DIR = "/Users/scui2/Desktop/TCGA/TCGA_COAD/TCGA_COAD_Distance_vs_ValidTiles_Scatterplots"
    os.makedirs(OUTPUT_SCATTER_DIR, exist_ok=True)
    
    # Read Distance Summary CSV
    distance_summary_path = "/Users/scui2/Desktop/TCGA/TCGA_COAD/TCGA_COAD_Weighted_Distances/Distance_Summary_All_Samples.csv"
    
    if not os.path.exists(distance_summary_path):
        print(f"❌ Distance summary CSV not found: {distance_summary_path}")
        print("Skipping Step 7.\n")
    else:
        df_distance = pd.read_csv(distance_summary_path)
        df_distance = df_distance.dropna()  # Remove rows with any missing values
        print(f"Loaded Distance Summary CSV: {df_distance.shape[0]} samples × {df_distance.shape[1]} columns")
        
        # Extract valid tiles count for each sample
        print(f"\nExtracting valid tiles count from prediction CSVs...")
        
        valid_tiles_dict = {}
        missing_count = 0
        
        for idx, row in tqdm(df_distance.iterrows(), total=len(df_distance), desc="Processing samples"):
            sample_id = row['Sample_ID']
            
            # Look for corresponding grouped_normalized CSV
            prediction_csv = os.path.join(OUTPUT_GROUPED_PREDICTIONS_DIR, f"{sample_id}_grouped_normalized.csv")
            
            if os.path.exists(prediction_csv):
                df_pred = pd.read_csv(prediction_csv)
                
                # Count valid tiles (rows with non-NaN normalized values)
                # Check the first normalized column to identify valid tiles
                valid_mask = df_pred['Cancer_normalized'].notna()
                valid_tiles_count = valid_mask.sum()
                
                valid_tiles_dict[sample_id] = valid_tiles_count
            else:
                missing_count += 1
                valid_tiles_dict[sample_id] = np.nan
        
        print(f"\n✅ Valid tiles extraction completed:")
        print(f"  Successfully matched: {len(valid_tiles_dict) - missing_count}/{len(df_distance)}")
        print(f"  Missing prediction CSVs: {missing_count}")
        
        # Add valid_tiles and sqrt_valid_tiles to dataframe
        df_distance['valid_tiles'] = df_distance['Sample_ID'].map(valid_tiles_dict)
        df_distance['sqrt_valid_tiles'] = np.sqrt(df_distance['valid_tiles'])
        
        # Remove samples with missing valid_tiles
        df_analysis = df_distance[df_distance['valid_tiles'].notna()].copy()
        print(f"\nSamples available for analysis: {len(df_analysis)}")
        
        if len(df_analysis) == 0:
            print("❌ No samples with valid tiles information. Skipping scatter plots.")
        else:
            # Print summary statistics of valid tiles
            print(f"\nValid Tiles Summary Statistics:")
            print(f"  Mean: {df_analysis['valid_tiles'].mean():.0f}")
            print(f"  Median: {df_analysis['valid_tiles'].median():.0f}")
            print(f"  Min: {df_analysis['valid_tiles'].min():.0f}")
            print(f"  Max: {df_analysis['valid_tiles'].max():.0f}")
            print(f"  Std: {df_analysis['valid_tiles'].std():.0f}")
            
            print(f"\nsqrt(Valid Tiles) Summary Statistics:")
            print(f"  Mean: {df_analysis['sqrt_valid_tiles'].mean():.2f}")
            print(f"  Median: {df_analysis['sqrt_valid_tiles'].median():.2f}")
            print(f"  Min: {df_analysis['sqrt_valid_tiles'].min():.2f}")
            print(f"  Max: {df_analysis['sqrt_valid_tiles'].max():.2f}")
            print(f"  Std: {df_analysis['sqrt_valid_tiles'].std():.2f}")
            
            # Define 12 directional pairs for COAD (4 cell types, excluding Normal_Epithelial)
            # Order follows: Cancer, Stromal, pan-APC, T_Cells
            directional_pairs_step7 = [
                ('Cancer', 'Stromal'),
                ('Cancer', 'pan-APC'),
                ('Cancer', 'T_Cells'),
                ('Stromal', 'Cancer'),
                ('Stromal', 'pan-APC'),
                ('Stromal', 'T_Cells'),
                ('pan-APC', 'Cancer'),
                ('pan-APC', 'Stromal'),
                ('pan-APC', 'T_Cells'),
                ('T_Cells', 'Cancer'),
                ('T_Cells', 'Stromal'),
                ('T_Cells', 'pan-APC')
            ]
            
            # Define 4 statistics
            statistics = ['mean', 'std', 'median', 'iqr']
            
            # Generate 48 scatter plots (12 pairs × 4 statistics)
            print(f"\nGenerating {len(directional_pairs_step7) * len(statistics)} scatter plots...")
            plot_count = 0
            
            for source, target in tqdm(directional_pairs_step7, desc="Directional pairs"):
                for stat in statistics:
                    plot_count += 1
                    
                    # Column name in Distance Summary CSV
                    col_name = f'{source}_to_{target}_{stat}'
                    
                    # Check if column exists
                    if col_name not in df_analysis.columns:
                        print(f"  ⚠️  Column not found: {col_name}, skipping...")
                        continue
                    
                    # Extract data (remove NaN values)
                    mask = df_analysis[col_name].notna() & df_analysis['sqrt_valid_tiles'].notna()
                    x_data = df_analysis.loc[mask, 'sqrt_valid_tiles']
                    y_data = df_analysis.loc[mask, col_name]
                    
                    if len(x_data) == 0:
                        print(f"  ⚠️  No valid data for {col_name}, skipping...")
                        continue
                    
                    # Calculate correlation
                    from scipy.stats import pearsonr, spearmanr
                    pearson_r, pearson_p = pearsonr(x_data, y_data)
                    spearman_r, spearman_p = spearmanr(x_data, y_data)
                    
                    # Create scatter plot
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Scatter plot
                    ax.scatter(x_data, y_data, alpha=0.6, s=80, color='steelblue', edgecolors='black', linewidth=0.5)
                    
                    # Add trend line (linear regression)
                    z = np.polyfit(x_data, y_data, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(x_data.min(), x_data.max(), 100)
                    y_trend = p(x_trend)
                    ax.plot(x_trend, y_trend, color='red', linewidth=2.5, linestyle='--', alpha=0.8, label='Linear Fit')
                    
                    # Labels and title
                    ax.set_xlabel('sqrt(Valid Tiles)', fontsize=14, fontweight='bold')
                    ax.set_ylabel(f'Distance ({stat.capitalize()})', fontsize=14, fontweight='bold')
                    ax.set_title(f'{source} → {target} ({stat.upper()})\n'
                                f'Pearson r={pearson_r:.3f} (p={pearson_p:.2e}), '
                                f'Spearman ρ={spearman_r:.3f} (p={spearman_p:.2e})\n'
                                f'n={len(x_data)} samples',
                                fontsize=13, fontweight='bold', pad=15)
                    
                    # Grid
                    ax.grid(alpha=0.3, linestyle='--', linewidth=1)
                    
                    # Legend
                    ax.legend(fontsize=11, loc='best')
                    
                    # Style
                    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5)
                    for spine in ax.spines.values():
                        spine.set_linewidth(1.5)
                        spine.set_color('black')
                    
                    plt.tight_layout()
                    
                    # Save figure
                    output_filename = f"{plot_count:02d}_{source}_to_{target}_{stat}_vs_sqrt_valid_tiles.png"
                    output_path = os.path.join(OUTPUT_SCATTER_DIR, output_filename)
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
            
            print(f"\n✅ Step 7 Complete!")
            print(f"  - Generated {plot_count} scatter plots")
            print(f"  - Output directory: {OUTPUT_SCATTER_DIR}")
            print(f"  - Samples analyzed: {len(df_analysis)}")
            
            # Save the extended dataframe with valid_tiles information
            extended_csv_path = os.path.join(OUTPUT_DISTANCE_BASE_DIR, "Distance_Summary_With_ValidTiles.csv")
            df_distance.to_csv(extended_csv_path, index=False)
            print(f"  - Extended summary saved: {extended_csv_path}")
            print(f"{'='*80}\n")
            
            # ================================================================
            # Step 8: Scale distance metrics by sqrt(valid_tiles)
            # ================================================================
            
            print(f"\n{'='*80}")
            print(f"Scaling distance metrics by sqrt(valid_tiles)...")
            print(f"{'='*80}\n")
            
            # Create a copy for scaled version
            df_scaled = df_distance.copy()
            
            # Define all 48 distance metric columns (12 pairs × 4 stats)
            for source, target in directional_pairs_step7:
                for stat in statistics:
                    col_name = f'{source}_to_{target}_{stat}'
                    if col_name in df_scaled.columns:
                        # Scale by sqrt(valid_tiles)
                        df_scaled[col_name] = df_scaled[col_name] / df_scaled['sqrt_valid_tiles']
            
            # Save scaled CSV
            scaled_csv_path = os.path.join(OUTPUT_DISTANCE_BASE_DIR, "Distance_Summary_All_Samples_Scaled.csv")
            df_scaled.to_csv(scaled_csv_path, index=False)
            
            print(f"✅ Scaled distance metrics saved: {scaled_csv_path}")
            print(f"   All 48 distance metrics scaled by sqrt(valid_tiles)")
            print(f"   Shape: {df_scaled.shape[0]} samples × {df_scaled.shape[1]} columns\n")

