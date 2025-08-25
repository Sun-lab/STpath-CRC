"""
STPath-COAD: TCGA Dataset Analysis for Figure 7
==============================================

This script analyzes TCGA colorectal cancer data using trained STPath models
and generates visualizations for Figure 7. Validates model performance on external data.

Author: Saishi Cui
Date: Sept 2025

Purpose: Apply STPath-COAD models to TCGA colorectal cancer histopathology data
for external validation, perform comprehensive analysis, and generate 
visualizations for Figure 7 of the paper.
"""

import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import timm
from conch.open_clip_custom import create_model_from_pretrained
from huggingface_hub import login
from datetime import datetime
from timm.layers import SwiGLUPacked
import xgboost as xgb
from torchvision import transforms
import pickle
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd

# Global variables for models
global_models = None
global_transforms = None
global_important_features = None
global_xgb_models = None

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def load_Conch_model():
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token)
    model, transform = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token=hf_token)
    return model, transform

def load_UNI2h_model():
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token) 
    timm_kwargs = {
        'img_size': 224, 
        'patch_size': 14, 
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5, 
        'embed_dim': 1536,
        'mlp_ratio': 2.66667*2,
        'num_classes': 0, 
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 
        'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 
        'dynamic_img_size': True
    }
    model_UNI2h = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    return model_UNI2h

def load_ProvGigapath_model():
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token)
    tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    return tile_encoder

def load_Virchow_model():
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token)
    model_Virchow = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model_Virchow = model_Virchow.eval()
    return model_Virchow

def load_Virchow2_model():
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token)
    model_Virchow2 = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model_Virchow2 = model_Virchow2.eval()
    return model_Virchow2

def create_patches_240x240(image_path):
    """Create 240x240 pixel patches from WSI"""
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(image_path)
    image_np = np.array(image)
    
    height, width = image_np.shape[:2]
    log(f"Original image size: {width} x {height}")
    
    # Fixed patch size: 240x240 pixels
    patch_size = 240
    
    # Calculate how many patches we can fit
    num_patches_width = width // patch_size
    num_patches_height = height // patch_size
    
    log(f"Creating {num_patches_height} x {num_patches_width} = {num_patches_height * num_patches_width} patches")
    log(f"Each patch size: {patch_size} x {patch_size} pixels")
    
    # Calculate discarded border sizes
    discarded_width = width % patch_size
    discarded_height = height % patch_size
    log(f"Discarded border: {discarded_width} pixels (width) x {discarded_height} pixels (height)")
    
    patch_list = []
    
    for row in range(num_patches_height):
        for col in range(num_patches_width):
            y_start = row * patch_size
            y_end = y_start + patch_size
            x_start = col * patch_size
            x_end = x_start + patch_size
            
            patch_np = image_np[y_start:y_end, x_start:x_end]
            patch_image = Image.fromarray(patch_np)
            patch_id = f"row{row:03d}_col{col:03d}"
            
            patch_info = {
                'patch_id': patch_id,
                'row': row,
                'col': col,
                'x_start': x_start,
                'y_start': y_start,
                'x_end': x_end,
                'y_end': y_end,
                'patch_width': patch_size,
                'patch_height': patch_size,
                'patch_image': patch_image,
                'patch_array': patch_np
            }
            
            patch_list.append(patch_info)
    
    log(f"Successfully created {len(patch_list)} patches")
    return patch_list

def load_all_models_once(xgboost_model_paths):
    """Load all models once at the beginning"""
    global global_models, global_transforms, global_important_features, global_xgb_models
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    log(f"Loading foundation models on {device}")
    
    # Load all foundation models
    model_conch, transform_conch = load_Conch_model()
    model_conch.eval().to(device)
    
    model_uni2h = load_UNI2h_model()
    model_uni2h.eval().to(device)
    
    model_provgigapath = load_ProvGigapath_model()
    model_provgigapath.eval().to(device)
    
    model_virchow = load_Virchow_model()
    model_virchow.eval().to(device)
    
    model_virchow2 = load_Virchow2_model()
    model_virchow2.eval().to(device)
    
    # Define transforms for different models
    transform_standard = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    transform_provgigapath = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    # Store models and transforms globally
    global_models = {
        'conch': model_conch,
        'uni2h': model_uni2h,
        'provgigapath': model_provgigapath,
        'virchow': model_virchow,
        'virchow2': model_virchow2
    }
    
    global_transforms = {
        'conch': transform_conch,
        'standard': transform_standard,
        'provgigapath': transform_provgigapath
    }
    
    # Load important features and XGBoost models for all cell types
    cell_types = ["Cancer Cells", "Stromal Cells", "Normal Epithelial Cells", "T cells", "Other Immune Cells"]
    important_features = {}
    xgb_models = {}
    
    for cell_type in cell_types:
        # Load important features
        important_features_path = f"Colorectal_Cancer_HE_patches/xgboost_prediction/important_features_{cell_type}.pkl"
        log(f"Loading important features for {cell_type} from {important_features_path}")
        important_features[cell_type] = pickle.load(open(important_features_path, "rb"))
        
        # Load XGBoost model
        model_path = xgboost_model_paths[cell_type]
        log(f"Loading XGBoost model for {cell_type} from {model_path}")
        xgb_model = xgb.Booster()
        xgb_model.load_model(model_path)
        xgb_models[cell_type] = xgb_model
    
    global_important_features = important_features
    global_xgb_models = xgb_models
    
    log("All models loaded successfully!")
    return device

def predict_all_cell_types_optimized(image_info_list, device):
    """
    Predict all 5 cell types for each patch using pre-loaded models
    
    Args:
        image_info_list: List of patch information dictionaries
        device: PyTorch device
    
    Returns:
        List of dictionaries with patch info and normalized cell type proportions
    """
    
    # Use global models
    model_conch = global_models['conch']
    model_uni2h = global_models['uni2h']
    model_provgigapath = global_models['provgigapath']
    model_virchow = global_models['virchow']
    model_virchow2 = global_models['virchow2']
    
    transform_conch = global_transforms['conch']
    transform_standard = global_transforms['standard']
    transform_provgigapath = global_transforms['provgigapath']
    
    important_features = global_important_features
    xgb_models = global_xgb_models
    
    cell_types = ["Cancer Cells", "Stromal Cells", "Normal Epithelial Cells", "T cells", "Other Immune Cells"]
    
    # White patch detection parameters
    white_threshold = 220
    white_ratio_cutoff = 0.4
    
    def is_white_patch(patch_array):
        # Convert to grayscale if RGB
        if len(patch_array.shape) == 3:
            gray = np.mean(patch_array, axis=2)
        else:
            gray = patch_array
        white_pixels = np.sum(gray > white_threshold)
        total_pixels = gray.size
        white_ratio = white_pixels / total_pixels
        return white_ratio > white_ratio_cutoff
    
    results = []
    
    log(f"Processing {len(image_info_list)} patches")
    
    for i, image_info in enumerate(tqdm(image_info_list, desc="Predicting cell type proportions")):
        
        # Check if patch is too white
        if is_white_patch(image_info['patch_array']):
            # For white patches, assign None values
            results.append({
                'patch_id': image_info['patch_id'],
                'row': image_info['row'],
                'col': image_info['col'],
                'x_start': image_info['x_start'],
                'y_start': image_info['y_start'],
                'x_end': image_info['x_end'],
                'y_end': image_info['y_end'],
                'cancer_proportion': None,
                'stromal_proportion': None,
                'normal_epithelia_proportion': None,
                'tcells_proportion': None,
                'other_immune_proportion': None,
                'normalized_proportions': None
            })
            continue
        
        # Extract features from all 5 foundation models (only once per patch)
        patch_image = image_info["patch_image"]
        
        with torch.no_grad():
            # UNI2h features  
            img_uni2h = transform_standard(patch_image).unsqueeze(0).to(device)
            features_uni2h_full = model_uni2h(img_uni2h)
            features_uni2h_full = features_uni2h_full.cpu().numpy().flatten()
            
            # Virchow features
            img_virchow = transform_standard(patch_image).unsqueeze(0).to(device)
            embeddings_virchow = model_virchow(img_virchow)
            features_virchow_full = torch.cat([embeddings_virchow[:,0], embeddings_virchow[:,5:].mean(1)], dim=-1)
            features_virchow_full = features_virchow_full.cpu().numpy().flatten()
            
            # Virchow2 features
            img_virchow2 = transform_standard(patch_image).unsqueeze(0).to(device)
            embeddings_virchow2 = model_virchow2(img_virchow2)
            features_virchow2_full = torch.cat([embeddings_virchow2[:,0], embeddings_virchow2[:,5:].mean(1)], dim=-1)
            features_virchow2_full = features_virchow2_full.cpu().numpy().flatten()
            
            # ProvGigapath features
            img_provgigapath = transform_provgigapath(patch_image).unsqueeze(0).to(device)
            features_provgigapath_full = model_provgigapath(img_provgigapath)
            features_provgigapath_full = features_provgigapath_full.cpu().numpy().flatten()
            
            # Conch features
            img_conch = transform_conch(patch_image).unsqueeze(0).to(device)
            features_conch_full = model_conch.encode_image(img_conch, proj_contrast=False, normalize=False)
            features_conch_full = features_conch_full.cpu().numpy().flatten()
        
        # Predict each cell type
        predictions = {}
        
        for cell_type in cell_types:
            # Get important feature indices for this cell type
            UNI2h_important_indices = [int(f.replace('f', '')) for f in important_features[cell_type]["UNI2h"]]
            Virchow_important_indices = [int(f.replace('f', '')) for f in important_features[cell_type]["Virchow"]]
            Virchow2_important_indices = [int(f.replace('f', '')) for f in important_features[cell_type]["Virchow2"]]
            ProvGigapath_important_indices = [int(f.replace('f', '')) for f in important_features[cell_type]["ProvGigapath"]]
            Conch_important_indices = [int(f.replace('f', '')) for f in important_features[cell_type]["Conch"]]
            
            # Select important features
            UNI2h_selected = features_uni2h_full[UNI2h_important_indices]
            Virchow_selected = features_virchow_full[Virchow_important_indices]
            Virchow2_selected = features_virchow2_full[Virchow2_important_indices]
            ProvGigapath_selected = features_provgigapath_full[ProvGigapath_important_indices]
            Conch_selected = features_conch_full[Conch_important_indices]
            
            # Stack features in the specified order: UNI2h, Virchow, Virchow2, ProvGigapath, Conch
            combined_features = np.concatenate([
                UNI2h_selected,
                Virchow_selected,
                Virchow2_selected,
                ProvGigapath_selected,
                Conch_selected
            ])
            
            # Prepare data for XGBoost prediction
            dtest = xgb.DMatrix(combined_features.reshape(1, -1))
            
            # Make prediction
            prediction = xgb_models[cell_type].predict(dtest)[0]
            predictions[cell_type] = float(prediction)
        
        # Normalize predictions to sum to 1
        total_prediction = sum(predictions.values())
        if total_prediction > 0:
            normalized_predictions = {cell_type: pred / total_prediction for cell_type, pred in predictions.items()}
        else:
            # If all predictions are 0, assign equal proportions
            normalized_predictions = {cell_type: 0.2 for cell_type in cell_types}
        
        # Store results
        results.append({
            'patch_id': image_info['patch_id'],
            'row': image_info['row'],
            'col': image_info['col'],
            'x_start': image_info['x_start'],
            'y_start': image_info['y_start'],
            'x_end': image_info['x_end'],
            'y_end': image_info['y_end'],
            'cancer_proportion': normalized_predictions["Cancer Cells"],
            'stromal_proportion': normalized_predictions["Stromal Cells"],
            'normal_epithelia_proportion': normalized_predictions["Normal Epithelial Cells"],
            'tcells_proportion': normalized_predictions["T cells"],
            'other_immune_proportion': normalized_predictions["Other Immune Cells"],
            'normalized_proportions': [
                normalized_predictions["Cancer Cells"],
                normalized_predictions["Stromal Cells"],
                normalized_predictions["Normal Epithelial Cells"],
                normalized_predictions["T cells"],
                normalized_predictions["Other Immune Cells"]
            ]
        })
    
    log(f"Completed prediction for {len(results)} patches")
    return results

def process_single_wsi(wsi_path, output_dir, device):
    """Process a single WSI file and save results"""
    
    # Get WSI filename without extension
    wsi_filename = os.path.basename(wsi_path)
    wsi_name = os.path.splitext(wsi_filename)[0]
    
    log(f"Processing WSI: {wsi_filename}")
    
    # Step 1: Create 240x240 patches
    patch_list = create_patches_240x240(wsi_path)
    
    # Step 2: Predict all cell types
    prediction_results = predict_all_cell_types_optimized(patch_list, device)
    
    # Step 3: Save results as .pt file
    output_path = os.path.join(output_dir, f"{wsi_name}.pt")
    torch.save(prediction_results, output_path)
    
    log(f"Results saved to: {output_path}")
    
    # Print summary statistics
    valid_patches = sum(1 for r in prediction_results if r['normalized_proportions'] is not None)
    white_patches = len(prediction_results) - valid_patches
    
    log(f"=== Processing Summary for {wsi_filename} ===")
    log(f"Total patches: {len(prediction_results)}")
    log(f"Valid patches (non-white): {valid_patches}")
    log(f"White patches (skipped): {white_patches}")
    
    if valid_patches > 0:
        # Calculate average proportions for valid patches
        avg_cancer = np.mean([r['cancer_proportion'] for r in prediction_results if r['cancer_proportion'] is not None])
        avg_stromal = np.mean([r['stromal_proportion'] for r in prediction_results if r['stromal_proportion'] is not None])
        avg_normal = np.mean([r['normal_epithelia_proportion'] for r in prediction_results if r['normal_epithelia_proportion'] is not None])
        avg_tcells = np.mean([r['tcells_proportion'] for r in prediction_results if r['tcells_proportion'] is not None])
        avg_other = np.mean([r['other_immune_proportion'] for r in prediction_results if r['other_immune_proportion'] is not None])
        
        log(f"Average proportions:")
        log(f"  Cancer: {avg_cancer:.3f}")
        log(f"  Stromal: {avg_stromal:.3f}")
        log(f"  Normal Epithelial: {avg_normal:.3f}")
        log(f"  T cells: {avg_tcells:.3f}")
        log(f"  Other Immune: {avg_other:.3f}")
    
    return prediction_results


def visualize_tcga_hexagon_patches(pt_file_path, save_path=None):
    """
    Visualize TCGA prediction results using hexagon patches with color coding based on cell type thresholds
    
    Args:
        pt_file_path: Path to the .pt file containing prediction results
        save_path: Optional path to save the visualization
    
    Color coding:
    - Tumor > 40%: #E41A1C (red)
    - Stromal > 50%: #FFFF33 (yellow) 
    - Normal epithelial > 15%: #377EB8 (blue)
    - T cells > 15%: #4DAF4A (green)
    - Other immune > 15%: #FF7F00 (orange)
    - Any mixed conditions: Black
    - No classification: White
    """
    
    # Load prediction results
    log(f"Loading prediction results from: {pt_file_path}")
    prediction_results = torch.load(pt_file_path)
    
    # Define color mapping and thresholds
    color_thresholds = {
        'tumor': {'threshold': 0.40, 'color': '#E41A1C'},      # Red - increased to 40%
        'stromal': {'threshold': 0.50, 'color': '#FFFF33'},    # Yellow - increased to 50%
        'normal': {'threshold': 0.15, 'color': '#377EB8'},     # Blue - reduced to 15%
        'tcells': {'threshold': 0.15, 'color': '#4DAF4A'},     # Green - reduced to 15%
        'other_immune': {'threshold': 0.15, 'color': '#FF7F00'} # Orange - reduced to 15%
    }
    
    def hex_to_rgb(hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def mix_colors(colors):
        """Mixed colors are black, single colors remain original, no colors = white"""
        if not colors:
            return (1.0, 1.0, 1.0)  # White for no conditions met
        elif len(colors) > 1:
            return (0.0, 0.0, 0.0)  # Black for any mixed conditions
        else:
            # Single color, return normalized RGB
            c = colors[0]
            return (c[0]/255.0, c[1]/255.0, c[2]/255.0)
    
    # Sanity check: Remove patches at the very top and bottom (likely artifacts)
    if len(prediction_results) > 0:
        # Get all y coordinates to determine boundaries
        all_y_coords = []
        for result in prediction_results:
            if result['normalized_proportions'] is not None:
                y_center = (result['y_start'] + result['y_end']) / 2
                all_y_coords.append(y_center)
        
        if all_y_coords:
            min_y = min(all_y_coords)
            max_y = max(all_y_coords)
            y_range = max_y - min_y
            
            # Remove top and bottom 5% as potential artifacts
            cutoff_margin = y_range * 0.05
            y_min_cutoff = min_y + cutoff_margin
            y_max_cutoff = max_y - cutoff_margin
            
            log(f"Applying sanity check: removing patches with y < {y_min_cutoff:.1f} or y > {y_max_cutoff:.1f}")
            
            # Filter prediction results
            filtered_results = []
            removed_count = 0
            for result in prediction_results:
                if result['normalized_proportions'] is None:
                    continue  # Skip white patches
                
                y_center = (result['y_start'] + result['y_end']) / 2
                if y_min_cutoff <= y_center <= y_max_cutoff:
                    filtered_results.append(result)
                else:
                    removed_count += 1
            
            log(f"Removed {removed_count} patches due to sanity check")
            prediction_results = filtered_results
    
    # Process each patch to determine colors
    data_list = []
    for result in prediction_results:
        # Calculate patch center coordinates
        x_center = (result['x_start'] + result['x_end']) / 2
        y_center = (result['y_start'] + result['y_end']) / 2
        
        # Check which conditions are met
        active_colors = []
        conditions_met = []
        
        # Check each cell type against its threshold
        if result['cancer_proportion'] > color_thresholds['tumor']['threshold']:
            active_colors.append(hex_to_rgb(color_thresholds['tumor']['color']))
            conditions_met.append('tumor')
            
        if result['stromal_proportion'] > color_thresholds['stromal']['threshold']:
            active_colors.append(hex_to_rgb(color_thresholds['stromal']['color']))
            conditions_met.append('stromal')
            
        if result['normal_epithelia_proportion'] > color_thresholds['normal']['threshold']:
            active_colors.append(hex_to_rgb(color_thresholds['normal']['color']))
            conditions_met.append('normal')
            
        if result['tcells_proportion'] > color_thresholds['tcells']['threshold']:
            active_colors.append(hex_to_rgb(color_thresholds['tcells']['color']))
            conditions_met.append('tcells')
            
        if result['other_immune_proportion'] > color_thresholds['other_immune']['threshold']:
            active_colors.append(hex_to_rgb(color_thresholds['other_immune']['color']))
            conditions_met.append('other_immune')
        
        # Calculate final color
        final_color = mix_colors(active_colors)
        
        data_list.append({
            'patch_id': result['patch_id'],
            'x': x_center,
            'y': y_center,
            'row': result['row'],
            'col': result['col'],
            'color': final_color,
            'conditions_met': conditions_met,
            'cancer_prop': result['cancer_proportion'],
            'stromal_prop': result['stromal_proportion'],
            'normal_prop': result['normal_epithelia_proportion'],
            'tcells_prop': result['tcells_proportion'],
            'other_immune_prop': result['other_immune_proportion']
        })
    
    if len(data_list) == 0:
        log("No valid patches to visualize")
        return
    
    df = pd.DataFrame(data_list)
    log(f"Visualizing {len(df)} patches with valid predictions")
    
    # Adaptive marker size calculation based on patch density
    total_patches = len(df)
    
    # Calculate actual plot area based on patch coordinates
    x_range = df['x'].max() - df['x'].min()
    y_range = df['y'].max() - df['y'].min()
    plot_area = x_range * y_range if x_range > 0 and y_range > 0 else 1
    
    # Calculate patch density (patches per unit area)
    patch_density = total_patches / plot_area
    
    # Adaptive sizing: more patches = smaller hexagons, fewer patches = larger hexagons
    base_size = 150  # Reduced base size
    if total_patches < 100:
        marker_size = base_size * 1.5  # Reduced from 2.0
    elif total_patches < 500:
        marker_size = base_size * 1.2  # Reduced from 1.5
    elif total_patches < 1000:
        marker_size = base_size * 0.8  # Reduced from 1.0
    elif total_patches < 2000:
        marker_size = base_size * 0.5  # Reduced from 0.7
    elif total_patches < 5000:
        marker_size = base_size * 0.3  # Reduced from 0.5
    else:
        marker_size = base_size * 0.2  # Reduced from 0.3
    
    marker_size = max(15, int(marker_size))  # Minimum size
    
    log(f"Total patches: {total_patches}")
    log(f"Plot area: {plot_area:.1f}")
    log(f"Patch density: {patch_density:.4f}")
    log(f"Using adaptive marker size: {marker_size}")
    
    # Create the hexagon scatter plot
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    # Create scatter plot with individual colors for each hexagon
    colors_array = np.array([row['color'] for _, row in df.iterrows()])
    
    scatter = plt.scatter(
        x=df['x'],
        y=df['y'],
        c=colors_array,
        marker='h',  # hexagon marker
        s=marker_size,
        edgecolors='black',
        linewidth=0.3
    )
    
    # Invert y axis to match image coordinate system
    ax.invert_yaxis()
    
    # Set thick black borders
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    # Remove title and axis labels
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Create custom legend with improved labels
    from matplotlib.patches import Patch
    
    # Helper function to convert hex to normalized RGB
    def hex_to_rgb_normalized(hex_color):
        rgb = hex_to_rgb(hex_color)
        return tuple(c/255.0 for c in rgb)
    
    # Count mixed conditions for legend
    mixed_conditions = set()
    single_conditions = set()
    for _, row in df.iterrows():
        conditions = row['conditions_met']
        if len(conditions) > 1:
            # Sort conditions for consistent naming
            sorted_conditions = sorted(conditions)
            mixed_conditions.add('/'.join(sorted_conditions))
        elif len(conditions) == 1:
            single_conditions.add(conditions[0])
    
    legend_elements = [
        Patch(facecolor=hex_to_rgb_normalized(color_thresholds['tumor']['color']), 
              edgecolor='black', label="Tumor"),
        Patch(facecolor=hex_to_rgb_normalized(color_thresholds['stromal']['color']), 
              edgecolor='black', label="Stromal"),
        Patch(facecolor=hex_to_rgb_normalized(color_thresholds['normal']['color']), 
              edgecolor='black', label="Normal Epithelial"),
        Patch(facecolor=hex_to_rgb_normalized(color_thresholds['tcells']['color']), 
              edgecolor='black', label="T cells"),
        Patch(facecolor=hex_to_rgb_normalized(color_thresholds['other_immune']['color']), 
              edgecolor='black', label="Other Immune"),
    ]
    
    # Add mixed condition entry to legend (all mixed conditions are black)
    if mixed_conditions:  # Only add Mixed entry if there are any mixed conditions
        legend_elements.append(
            Patch(facecolor=(0.0, 0.0, 0.0), 
                  edgecolor='black', label="Mixed")
        )
    
    # Add "No Classification" entry
    legend_elements.append(
        Patch(facecolor=(1.0, 1.0, 1.0), 
              edgecolor='black', label="No Classification")
    )
    
    # Legend disabled
    
    # Print statistics
    condition_counts = {}
    for _, row in df.iterrows():
        if len(row['conditions_met']) == 0:
            condition_counts['No Classification'] = condition_counts.get('No Classification', 0) + 1
        elif len(row['conditions_met']) == 1:
            condition = row['conditions_met'][0]
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
        else:
            # Mixed condition
            sorted_conditions = sorted(row['conditions_met'])
            mixed_key = '/'.join(sorted_conditions) + ' Mixed'
            condition_counts[mixed_key] = condition_counts.get(mixed_key, 0) + 1
    
    log("=== Condition Statistics ===")
    for condition, count in condition_counts.items():
        percentage = (count / len(df)) * 100
        log(f"{condition}: {count} patches ({percentage:.1f}%)")
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
        log(f"Visualization saved to: {save_path}")
    
    plt.tight_layout()
    plt.close()
    
    return None


def create_tcga_distance_folders():
    """Create folder structure for TCGA distance analysis"""
    base_dir = "Colorectal_Cancer_HE_patches/Visual/TCGA_Distance"
    
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Define all possible combinations of 5 patch types (5 choose 2 = 10 combinations)
    patch_types = ["Tumor", "Stromal", "Normal_Epithelial", "T_cells", "Other_Immune"]
    
    distance_folders = []
    for i in range(len(patch_types)):
        for j in range(i+1, len(patch_types)):
            folder_name = f"{patch_types[i]}_{patch_types[j]}"
            folder_path = os.path.join(base_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            distance_folders.append(folder_name)
    
    log(f"Created {len(distance_folders)} distance folders: {distance_folders}")
    return base_dir, distance_folders


def classify_patches(prediction_results):
    """
    Classify patches based on thresholds
    
    Returns:
        dict: {patch_type: list of patches with coordinates}
    """
    # Same thresholds as in visualization
    color_thresholds = {
        'tumor': 0.40,
        'stromal': 0.50, 
        'normal': 0.15,
        'tcells': 0.15,
        'other_immune': 0.15
    }
    
    classified_patches = {
        'tumor': [],
        'stromal': [],
        'normal_epithelial': [], 
        't_cells': [],
        'other_immune': []
    }
    
    for result in prediction_results:
        if result['normalized_proportions'] is None:
            continue  # Skip white patches
        
        # Calculate patch center coordinates
        x_center = (result['x_start'] + result['x_end']) / 2
        y_center = (result['y_start'] + result['y_end']) / 2
        
        patch_info = {
            'patch_id': result['patch_id'],
            'x': x_center,
            'y': y_center,
            'row': result['row'],
            'col': result['col'],
            'cancer_prop': result['cancer_proportion'],
            'stromal_prop': result['stromal_proportion'],
            'normal_prop': result['normal_epithelia_proportion'],
            'tcells_prop': result['tcells_proportion'],
            'other_immune_prop': result['other_immune_proportion']
        }
        
        # Classify patch based on single highest threshold (no mixed classifications for distance)
        classifications = []
        
        if result['cancer_proportion'] > color_thresholds['tumor']:
            classifications.append('tumor')
        if result['stromal_proportion'] > color_thresholds['stromal']:
            classifications.append('stromal') 
        if result['normal_epithelia_proportion'] > color_thresholds['normal']:
            classifications.append('normal_epithelial')
        if result['tcells_proportion'] > color_thresholds['tcells']:
            classifications.append('t_cells')
        if result['other_immune_proportion'] > color_thresholds['other_immune']:
            classifications.append('other_immune')
        
        # For distance calculation, only use patches with single classification
        if len(classifications) == 1:
            classified_patches[classifications[0]].append(patch_info)
    
    return classified_patches


def calculate_distance_distribution(patches_type1, patches_type2, type1_name, type2_name):
    """
    Calculate distance distribution between two patch types using standardized grid coordinates
    
    Args:
        patches_type1: List of patches of first type
        patches_type2: List of patches of second type
        type1_name, type2_name: Names for logging
    
    Returns:
        list: Distances from each type1 patch to all type2 patches (averaged)
        Distance unit: 1 = distance between adjacent patches
    """
    if len(patches_type1) == 0 or len(patches_type2) == 0:
        log(f"No patches available for {type1_name}-{type2_name} distance calculation")
        return []
    
    distances = []
    
    for patch1 in patches_type1:
        # Calculate distances from this patch1 to all patches of type2
        patch_distances = []
        for patch2 in patches_type2:
            # Grid-based Euclidean distance using row/col coordinates
            # Each grid unit represents the distance between adjacent patches
            row_diff = patch1['row'] - patch2['row']
            col_diff = patch1['col'] - patch2['col']
            dist = np.sqrt(row_diff**2 + col_diff**2)
            patch_distances.append(dist)
        
        # Minimum distance from this patch1 to all type2 patches
        min_distance = np.min(patch_distances)
        distances.append(min_distance)
    
    log(f"Calculated {len(distances)} {type1_name}-{type2_name} distances (grid units)")
    return distances


def create_distance_density_plot(distances, type1_name, type2_name, sample_name, save_path):
    """
    Create density plot for distance distribution
    """
    if len(distances) == 0:
        log(f"No distances to plot for {type1_name}-{type2_name} in {sample_name}")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Create density plot
    plt.hist(distances, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black', linewidth=2)
    
    # Add kernel density estimation for smooth curve (without legend)
    from scipy import stats
    try:
        density = stats.gaussian_kde(distances)
        xs = np.linspace(min(distances), max(distances), 200)
        plt.plot(xs, density(xs), 'r-', linewidth=3)  # No label, so no legend
    except:
        pass  # Skip if scipy not available or too few points
    
    plt.xlabel(f'Minimum Distance (grid units)', fontsize=32, fontweight='bold')
    plt.ylabel('Density', fontsize=32, fontweight='bold')
    
    # Add statistics text
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    median_dist = np.median(distances)
    
    stats_text = f'Mean: {mean_dist:.1f}\nStd: {std_dist:.1f}\nMedian: {median_dist:.1f}\nN: {len(distances)}'
    plt.text(0.6, 0.6, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=28, fontweight='bold')
    
    # Style improvements - thick black borders
    plt.gca().spines['top'].set_linewidth(5)
    plt.gca().spines['right'].set_linewidth(5)
    plt.gca().spines['bottom'].set_linewidth(5)
    plt.gca().spines['left'].set_linewidth(5)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().tick_params(labelsize=30, width=3)
    
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontweight('bold')
        label.set_color('black')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log(f"Distance density plot saved: {save_path}")


def process_sample_distances(pt_file_path, base_distance_dir):
    """
    Process a single sample to calculate all distance distributions
    """
    # Load prediction results
    sample_name = os.path.splitext(os.path.basename(pt_file_path))[0]
    log(f"Processing distances for sample: {sample_name}")
    
    prediction_results = torch.load(pt_file_path)
    
    # Classify patches
    classified_patches = classify_patches(prediction_results)
    
    # Log patch counts
    for patch_type, patches in classified_patches.items():
        log(f"  {patch_type}: {len(patches)} patches")
    
    # Define all distance combinations
    combinations = [
        ('tumor', 'stromal', 'Tumor', 'Stromal'),
        ('tumor', 'normal_epithelial', 'Tumor', 'Normal_Epithelial'),
        ('tumor', 't_cells', 'Tumor', 'T_cells'),
        ('tumor', 'other_immune', 'Tumor', 'Other_Immune'),
        ('stromal', 'normal_epithelial', 'Stromal', 'Normal_Epithelial'),
        ('stromal', 't_cells', 'Stromal', 'T_cells'),
        ('stromal', 'other_immune', 'Stromal', 'Other_Immune'),
        ('normal_epithelial', 't_cells', 'Normal_Epithelial', 'T_cells'),
        ('normal_epithelial', 'other_immune', 'Normal_Epithelial', 'Other_Immune'),
        ('t_cells', 'other_immune', 'T_cells', 'Other_Immune')
    ]
    
    # Calculate distances for each combination
    for type1_key, type2_key, type1_folder, type2_folder in combinations:
        patches1 = classified_patches[type1_key]
        patches2 = classified_patches[type2_key]
        
        if len(patches1) == 0 or len(patches2) == 0:
            log(f"  Skipping {type1_folder}_{type2_folder}: insufficient patches")
            continue
        
        # Calculate distances in both directions and combine
        distances1 = calculate_distance_distribution(patches1, patches2, type1_folder, type2_folder)
        distances2 = calculate_distance_distribution(patches2, patches1, type2_folder, type1_folder)
        
        all_distances = distances1 + distances2
        
        if len(all_distances) == 0:
            continue
        
        # Create density plot
        folder_name = f"{type1_folder}_{type2_folder}"
        output_dir = os.path.join(base_distance_dir, folder_name)
        output_path = os.path.join(output_dir, f"{sample_name}_distance_density.png")
        
        create_distance_density_plot(all_distances, type1_folder, type2_folder, sample_name, output_path)


def batch_process_tcga_distances():
    """
    Batch process all TCGA samples for distance analysis
    """
    # Create folder structure
    base_distance_dir, distance_folders = create_tcga_distance_folders()
    
    # Find all .pt files
    input_dir = "TCGA_COAD_HE_predicted"
    pt_files = glob.glob(os.path.join(input_dir, "*.pt"))
    
    if not pt_files:
        log(f"No .pt files found in {input_dir}")
        return
    
    log(f"Found {len(pt_files)} .pt files for distance analysis")
    
    # Process each file
    for i, pt_file in enumerate(pt_files):
        try:
            log(f"\n=== Processing distance analysis {i+1}/{len(pt_files)} ===")
            process_sample_distances(pt_file, base_distance_dir)
            
        except Exception as e:
            sample_name = os.path.splitext(os.path.basename(pt_file))[0]
            log(f"ERROR processing distances for {sample_name}: {str(e)}")
            continue
    
    log(f"\n=== Distance analysis completed ===")
    log(f"Results saved in: {base_distance_dir}")


def extract_tcga_features_dataframe():
    """
    Extract comprehensive features from all TCGA samples and create a DataFrame
    
    Returns:
        pandas.DataFrame: 25-column DataFrame with sample-level features
        Columns:
        1. Sample_ID (e.g., TCGA-G4-6298-01Z-00-DX1)
        2. Patient_ID (e.g., TCGA-G4-6298)
        3-7. Overall proportions of 5 cell types
        8-25. Distance statistics (mean, median, std) for 6 combinations
        Note: Normal epithelial-related distances excluded, missing values imputed using KNN
    """
    
    # Find all .pt files
    input_dir = "TCGA_COAD_HE_predicted"
    pt_files = glob.glob(os.path.join(input_dir, "*.pt"))
    
    if not pt_files:
        log(f"No .pt files found in {input_dir}")
        return None
    
    log(f"Found {len(pt_files)} .pt files for feature extraction")
    
    # Initialize results list
    all_features = []
    
    # Define directional distance combinations (excluding normal epithelial)
    # Each combination is calculated in both directions (A->B and B->A)
    distance_combinations = [
        # Tumor to others
        ('tumor', 'stromal', 'Tumor_to_Stromal'),
        ('tumor', 't_cells', 'Tumor_to_T_cells'),
        ('tumor', 'other_immune', 'Tumor_to_Other_Immune'),
        # Stromal to others
        ('stromal', 'tumor', 'Stromal_to_Tumor'),
        ('stromal', 't_cells', 'Stromal_to_T_cells'),
        ('stromal', 'other_immune', 'Stromal_to_Other_Immune'),
        # T cells to others
        ('t_cells', 'tumor', 'T_cells_to_Tumor'),
        ('t_cells', 'stromal', 'T_cells_to_Stromal'),
        ('t_cells', 'other_immune', 'T_cells_to_Other_Immune'),
        # Other immune to others
        ('other_immune', 'tumor', 'Other_Immune_to_Tumor'),
        ('other_immune', 'stromal', 'Other_Immune_to_Stromal'),
        ('other_immune', 't_cells', 'Other_Immune_to_T_cells')
    ]
    
    # Process each file
    for i, pt_file in enumerate(pt_files):
        try:
            # Extract sample and patient IDs
            sample_name = os.path.splitext(os.path.basename(pt_file))[0]
            # Extract patient ID (first 12 characters: TCGA-G4-6298)
            patient_id = sample_name[:12] if len(sample_name) >= 12 else sample_name
            
            log(f"Processing {i+1}/{len(pt_files)}: {sample_name}")
            
            # Load prediction results
            prediction_results = torch.load(pt_file)
            
            # Initialize feature dictionary
            features = {
                'Sample_ID': sample_name,
                'Patient_ID': patient_id
            }
            
            # Calculate overall proportions for 5 cell types
            valid_results = []
            for result in prediction_results:
                if result['normalized_proportions'] is not None:
                    valid_results.append(result)
            
            if len(valid_results) == 0:
                log(f"  No valid patches for {sample_name}, skipping...")
                continue
            
            # Calculate mean proportions across all valid patches
            cancer_props = [r['cancer_proportion'] for r in valid_results]
            stromal_props = [r['stromal_proportion'] for r in valid_results]
            normal_props = [r['normal_epithelia_proportion'] for r in valid_results]
            tcells_props = [r['tcells_proportion'] for r in valid_results]
            other_immune_props = [r['other_immune_proportion'] for r in valid_results]
            
            features['Cancer_Overall_Proportion'] = np.mean(cancer_props)
            features['Stromal_Overall_Proportion'] = np.mean(stromal_props)
            features['Normal_Epithelial_Overall_Proportion'] = np.mean(normal_props)
            features['T_cells_Overall_Proportion'] = np.mean(tcells_props)
            features['Other_Immune_Overall_Proportion'] = np.mean(other_immune_props)
            
            log(f"  Calculated overall proportions from {len(valid_results)} valid patches")
            
            # Classify patches for distance calculations
            classified_patches = classify_patches(prediction_results)
            
            # Calculate directional distance statistics for each combination
            for type1_key, type2_key, combo_name in distance_combinations:
                patches1 = classified_patches[type1_key]
                patches2 = classified_patches[type2_key]
                
                if len(patches1) == 0 or len(patches2) == 0:
                    # Set NaN for missing combinations
                    features[f'{combo_name}_Distance_Avg'] = np.nan
                    features[f'{combo_name}_Distance_Std'] = np.nan
                    continue
                
                # Calculate distances in one direction only (from type1 to type2)
                distances = calculate_distance_distribution(patches1, patches2, type1_key, type2_key)
                
                if len(distances) == 0:
                    features[f'{combo_name}_Distance_Avg'] = np.nan
                    features[f'{combo_name}_Distance_Std'] = np.nan
                else:
                    # Calculate minimum distance
                    features[f'{combo_name}_Distance_Avg'] = np.mean(distances)
                    features[f'{combo_name}_Distance_Std'] = np.std(distances)
            
            all_features.append(features)
            log(f"  Successfully extracted features for {sample_name}")
            
        except Exception as e:
            sample_name = os.path.splitext(os.path.basename(pt_file))[0]
            log(f"ERROR processing {sample_name}: {str(e)}")
            continue
    
    # Create DataFrame
    if len(all_features) == 0:
        log("No features extracted!")
        return None
    
    df = pd.DataFrame(all_features)
    
    # Reorder columns for clarity
    column_order = ['Sample_ID', 'Patient_ID', 
                   'Cancer_Overall_Proportion', 'Stromal_Overall_Proportion', 
                   'Normal_Epithelial_Overall_Proportion', 'T_cells_Overall_Proportion', 
                   'Other_Immune_Overall_Proportion']
    
    # Add distance columns (directional, with average and standard deviation)
    for type1_key, type2_key, combo_name in distance_combinations:
        column_order.extend([f'{combo_name}_Distance_Avg', f'{combo_name}_Distance_Std'])
    
    df = df[column_order]
    
    log(f"\n=== Feature extraction completed ===")
    log(f"DataFrame shape: {df.shape}")
    log(f"Columns: {len(df.columns)}")
    
    # Apply KNN imputation for missing distance values
    df = knn_impute_distance_features(df)
    
    # Save DataFrame
    output_path = "Colorectal_Cancer_HE_patches/TCGA_Features_Complete.csv"
    df.to_csv(output_path, index=False)
    log(f"Features saved to: {output_path}")
    
    return df


def knn_impute_distance_features(df):
    """
    Use KNN imputation for missing distance values based on overall proportions
    
    Args:
        df: DataFrame with features
    
    Returns:
        DataFrame with imputed distance values
    """
    from sklearn.neighbors import NearestNeighbors
    
    log("\n=== Starting KNN imputation for distance features ===")
    
    # Overall proportion columns (columns 2-6, 0-indexed)
    proportion_cols = [
        'Cancer_Overall_Proportion', 'Stromal_Overall_Proportion', 
        'Normal_Epithelial_Overall_Proportion', 'T_cells_Overall_Proportion', 
        'Other_Immune_Overall_Proportion'
    ]
    
    # Get all distance columns
    distance_cols = [col for col in df.columns if 'Distance' in col]
    
    log(f"Found {len(distance_cols)} distance columns for imputation")
    
    # Create a copy for imputation
    df_imputed = df.copy()
    
    for dist_col in distance_cols:
        # Check if there are missing values in this column
        missing_mask = df[dist_col].isna()
        n_missing = missing_mask.sum()
        
        if n_missing == 0:
            log(f"  {dist_col}: No missing values")
            continue
            
        if n_missing == len(df):
            log(f"  {dist_col}: All values missing, skipping")
            continue
            
        log(f"  {dist_col}: Imputing {n_missing} missing values")
        
        # Get non-missing data for this distance variable
        non_missing_mask = ~missing_mask
        non_missing_data = df[non_missing_mask]
        
        if len(non_missing_data) == 0:
            log(f"    No non-missing samples available, skipping")
            continue
        
        # Prepare features (overall proportions) for KNN - no standardization
        X_complete = non_missing_data[proportion_cols].values
        y_complete = non_missing_data[dist_col].values
        
        # For each missing sample, find K nearest neighbors
        missing_data = df[missing_mask]
        X_missing = missing_data[proportion_cols].values
        
        # Fit KNN model - K will be min(5, available samples)
        k = min(5, len(X_complete))
        knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        knn.fit(X_complete)
        
        # Find neighbors and impute using KNN mean
        distances, indices = knn.kneighbors(X_missing)
        
        imputed_values = []
        for i, neighbor_indices in enumerate(indices):
            neighbor_values = y_complete[neighbor_indices]
            imputed_value = np.mean(neighbor_values)
            imputed_values.append(imputed_value)
        
        # Fill in the imputed values
        df_imputed.loc[missing_mask, dist_col] = imputed_values
        
        log(f"    Successfully imputed using K={k} neighbors (KNN mean)")
    
    # Verify no missing values remain in distance columns
    remaining_missing = df_imputed[distance_cols].isna().sum().sum()
    log(f"\n=== KNN imputation completed ===")
    log(f"Remaining missing values in distance columns: {remaining_missing}")
    
    return df_imputed


if __name__ == "__main__":
    # Define paths
    input_dir = "TCGA_COAD_HE_converted"
    output_dir = "TCGA_COAD_HE_predicted"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define XGBoost model paths for external prediction
    xgboost_model_paths = {
        "Cancer Cells": "Colorectal_Cancer_HE_patches/xgboost_prediction/Cancer Cells_Combined_external_prediction/xgboost_model_Cancer Cells_Combined_external_prediction.model",
        "Stromal Cells": "Colorectal_Cancer_HE_patches/xgboost_prediction/Stromal Cells_Combined_external_prediction/xgboost_model_Stromal Cells_Combined_external_prediction.model",
        "Normal Epithelial Cells": "Colorectal_Cancer_HE_patches/xgboost_prediction/Normal Epithelial Cells_Combined_external_prediction/xgboost_model_Normal Epithelial Cells_Combined_external_prediction.model",
        "T cells": "Colorectal_Cancer_HE_patches/xgboost_prediction/T cells_Combined_external_prediction/xgboost_model_T cells_Combined_external_prediction.model",
        "Other Immune Cells": "Colorectal_Cancer_HE_patches/xgboost_prediction/Other Immune Cells_Combined_external_prediction/xgboost_model_Other Immune Cells_Combined_external_prediction.model"
    }
    
    # Find all TIFF files in the input directory
    tiff_files = glob.glob(os.path.join(input_dir, "*.tif")) + glob.glob(os.path.join(input_dir, "*.tiff"))
    
    log(f"Found {len(tiff_files)} TIFF files to process")
    
    # Load all models and transforms once
    device = load_all_models_once(xgboost_model_paths)

    # Process each WSI file
    for i, wsi_path in enumerate(tiff_files):
        log(f"\n=== Processing file {i+1}/{len(tiff_files)} ===")
        
        # Check if output already exists
        wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
        output_path = os.path.join(output_dir, f"{wsi_name}.pt")
        
        if os.path.exists(output_path):
            log(f"Output already exists for {wsi_name}, skipping...")
            continue
        
        try:
            # Process the WSI
            process_single_wsi(wsi_path, output_dir, device)
            
        except Exception as e:
            log(f"ERROR processing {wsi_name}: {str(e)}")
            continue
    
    log("\n=== All processing completed ===")



## Visualization for all TCGA results
if __name__ == "__main__":


    # Batch visualization for all TCGA results
    tcga_predicted_dir = "TCGA_COAD_HE_predicted"
    tcga_visual_dir = "Colorectal_Cancer_HE_patches/Visual/TCGA"
    

    # Create visualization directory if it doesn't exist
    os.makedirs(tcga_visual_dir, exist_ok=True)
    
    # Find all .pt files in the predicted directory
    pt_files = glob.glob(os.path.join(tcga_predicted_dir, "*.pt"))
    log(f"Found {len(pt_files)} .pt files for visualization")
    
    # Process each file
    for i, pt_file_path in enumerate(pt_files):
        pt_filename = os.path.basename(pt_file_path)
        sample_name = os.path.splitext(pt_filename)[0]  # Remove .pt extension
        
        log(f"\n=== Visualizing {i+1}/{len(pt_files)}: {sample_name} ===")
        
        # Define output path
        output_path = os.path.join(tcga_visual_dir, f"{sample_name}_visualization.png")
        
        # Check if visualization already exists
        if os.path.exists(output_path):
            log(f"Visualization already exists for {sample_name}, skipping...")
            continue
        
        try:
            # Create visualization
            visualize_tcga_hexagon_patches(pt_file_path, output_path)
            log(f"Successfully created visualization for {sample_name}")
            
        except Exception as e:
            log(f"ERROR visualizing {sample_name}: {str(e)}")
            continue
    
    log(f"\n=== All TCGA visualizations completed ===")
    log(f"Visualizations saved in: {tcga_visual_dir}")
    
    # Example usage of individual visualization:
    # visualize_tcga_hexagon_patches("TCGA_COAD_HE_predicted/TCGA-AA-3854-01Z-00-DX1.pt", 
    #                                "Colorectal_Cancer_HE_patches/Visual/TCGA/TCGA-AA-3854-01Z-00-DX1_visualization.png")



### Distance density plotting for all samples
if __name__ == "__main__":
    """
    Generate distance density plots for all TCGA samples using existing folder structure
    Process all .pt files and create distance density plots for all 20 directional combinations
    """
    
    # Find all .pt files
    input_dir = "TCGA_COAD_HE_predicted"
    base_dir = "Colorectal_Cancer_HE_patches/Visual/TCGA_Distance"
    
    pt_files = glob.glob(os.path.join(input_dir, "*.pt"))

    # Define all 20 directional combinations (5 cell types * 4 directions each)
    cell_types = [
        ('tumor', 'Tumor'),
        ('stromal', 'Stromal'), 
        ('normal_epithelial', 'Normal_Epithelial'),
        ('t_cells', 'T_cells'),
        ('other_immune', 'Other_Immune')
    ]
    
    # Create all directional combinations
    all_combinations = []
    for i, (from_key, from_name) in enumerate(cell_types):
        for j, (to_key, to_name) in enumerate(cell_types):
            if i != j:  # Skip self-to-self combinations
                all_combinations.append((from_key, to_key, from_name, to_name))
    
    log(f"Will process {len(all_combinations)} directional combinations")
    
    for i, pt_file_path in enumerate(pt_files):
        sample_name = os.path.splitext(os.path.basename(pt_file_path))[0]
        log(f"Processing {i+1}/{len(pt_files)}: {sample_name}")
        
        # Load prediction results
        prediction_results = torch.load(pt_file_path)
        
        # Classify patches
        classified_patches = classify_patches(prediction_results)
        
        # Process each directional combination
        for from_key, to_key, from_name, to_name in all_combinations:
            patches_from = classified_patches[from_key]
            patches_to = classified_patches[to_key]
            
            if len(patches_from) == 0 or len(patches_to) == 0:
                log(f"  Skipping {from_name}_to_{to_name}: insufficient patches")
                continue
            
            # Calculate minimum distances (from -> to direction)
            distances = calculate_distance_distribution(patches_from, patches_to, from_name, to_name)
            
            if len(distances) == 0:
                continue
            
            # Create folder path for this combination
            folder_name = f"{from_name}_to_{to_name}"
            folder_path = os.path.join(base_dir, folder_name)
            
            # Create folder if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)
            
            # Create output path with just sample name
            output_filename = f"{sample_name}.png"
            output_path = os.path.join(folder_path, output_filename)
            
            # Create distance density plot
            create_distance_density_plot(distances, from_name, to_name, sample_name, output_path)
            

        
        log(f"=== Distance density plotting completed ===")
        log(f"Results saved in: {base_dir}")


## Extract comprehensive features from all TCGA samples
if __name__ == "__main__":
    # Extract comprehensive features from all TCGA samples
    features_df = extract_tcga_features_dataframe()

    clinical_df = pd.read_csv("clinical.project-tcga-coad.2025-06-24/clinical.tsv", 
                            sep='\t',  # Explicitly specify tab separator
                            on_bad_lines='skip')  # Skip problematic lines


    # Add new columns for age and gender
    features_df["Patient_Age"] = np.nan
    features_df["Patient_Gender"] = np.nan
    
    log(f"Adding clinical variables to {len(features_df)} samples...")
    
    for i in range(features_df.shape[0]):
        patient_id = features_df.iloc[i]["Patient_ID"]
        
        # Find matching clinical records
        clinical_match = clinical_df[clinical_df["cases.submitter_id"] == patient_id]
        
        if len(clinical_match) == 0:
            log(f"  No clinical data found for {patient_id}")
            continue
            
        # Extract age - handle potential missing/multiple values
        age_series = clinical_match["demographic.age_at_index"]
        if len(age_series) > 0 and not pd.isna(age_series.iloc[0]):
            try:
                patient_age = int(age_series.iloc[0])
                features_df.loc[i, "Patient_Age"] = patient_age
            except (ValueError, TypeError):
                log(f"  Invalid age data for {patient_id}: {age_series.iloc[0]}")
                
        # Extract gender
        gender_series = clinical_match["demographic.gender"]
        if len(gender_series) > 0 and not pd.isna(gender_series.iloc[0]):
            patient_gender_str = gender_series.iloc[0].lower()
            if patient_gender_str == "male":
                patient_gender = 1
            elif patient_gender_str == "female":
                patient_gender = 0
            else:
                patient_gender = np.nan
                log(f"  Unknown gender for {patient_id}: {gender_series.iloc[0]}")
            features_df.loc[i, "Patient_Gender"] = patient_gender
            

    # Log summary statistics
    age_available = features_df["Patient_Age"].notna().sum()
    gender_available = features_df["Patient_Gender"].notna().sum()
    log(f"Successfully added age data for {age_available}/{len(features_df)} samples")
    log(f"Successfully added gender data for {gender_available}/{len(features_df)} samples")
    
    features_df.to_csv("Colorectal_Cancer_HE_patches/TCGA_Features_Complete_WithClinical.csv", index=False)
    
    features_df['Patient_Gender'].value_counts()

