"""
STPath-COAD: Soft Segmentation Analysis for Figure 6
===================================================

This script performs soft tissue segmentation analysis for spatial transcriptomics data
and generates visualizations for Figure 6. Implements probabilistic segmentation approaches.

Author: Saishi Cui
Date: December 2025

Purpose: Implement and evaluate soft segmentation approaches for tissue region 
identification in spatial transcriptomics data, generating comprehensive 
segmentation analysis for Figure 6 of the paper.
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
import matplotlib.pyplot as plt
import os



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


def create_patches_100x100(image_path):
    """Create 100x100 grid patches from WSI"""
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(image_path)
    image_np = np.array(image)
    
    height, width = image_np.shape[:2]
    print(f"Original image size: {width} x {height}")
    
    # Fixed grid size: 100x100 = 10,000 patches
    cut_grid_size = 100
    patch_height = height // cut_grid_size
    patch_width = width // cut_grid_size
    
    print(f"Will create {cut_grid_size} x {cut_grid_size} = {cut_grid_size * cut_grid_size} patches")
    print(f"Each patch size: {patch_width} x {patch_height} pixels")
    
    patch_list = []
    
    for row in tqdm(range(cut_grid_size), desc="Creating patches"):
        for col in range(cut_grid_size):
            y_start = row * patch_height
            y_end = y_start + patch_height
            x_start = col * patch_width
            x_end = x_start + patch_width
            
            # Handle edge cases for the last row/column
            if row == cut_grid_size - 1:
                y_end = height
            if col == cut_grid_size - 1:
                x_end = width
            
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
                'patch_width': x_end - x_start,
                'patch_height': y_end - y_start,
                'patch_image': patch_image,
                'patch_array': patch_np
            }
            
            patch_list.append(patch_info)
    
    print(f"Successfully created {len(patch_list)} patches")
    return patch_list, image_np


def predict_cell_type_proportions(image_info_list, cell_type, xgboost_model_path):
    """Predict cell type proportions for each patch"""
    
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
    
    # Load important features
    important_features_path = f"/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/xgboost_prediction/important_features_{cell_type}.pkl"
    log(f"Loading important features from {important_features_path}")
    important_features = pickle.load(open(important_features_path, "rb"))
    
    # Get important feature indices for each model
    UNI2h_important_indices = [int(f.replace('f', '')) for f in important_features["UNI2h"]]
    Virchow_important_indices = [int(f.replace('f', '')) for f in important_features["Virchow"]]
    Virchow2_important_indices = [int(f.replace('f', '')) for f in important_features["Virchow2"]]
    ProvGigapath_important_indices = [int(f.replace('f', '')) for f in important_features["ProvGigapath"]]
    Conch_important_indices = [int(f.replace('f', '')) for f in important_features["Conch"]]
    
    log(f"Selected features - UNI2h: {len(UNI2h_important_indices)}, Virchow: {len(Virchow_important_indices)}, Virchow2: {len(Virchow2_important_indices)}, ProvGigapath: {len(ProvGigapath_important_indices)}, Conch: {len(Conch_important_indices)}")
    
    # Load XGBoost model
    log(f"Loading XGBoost model from {xgboost_model_path}")
    xgb_model = xgb.Booster()
    xgb_model.load_model(xgboost_model_path)
    
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
            results.append({
                'patch_id': image_info['patch_id'],
                'row': image_info['row'],
                'col': image_info['col'],
                'x_start': image_info['x_start'],
                'y_start': image_info['y_start'],
                'x_end': image_info['x_end'],
                'y_end': image_info['y_end'],
                'predicted_proportion': None  # NA value
            })
            continue
        
        # Extract features from all 5 foundation models
        patch_image = image_info["patch_image"]
        
        with torch.no_grad():
            # UNI2h features  
            img_uni2h = transform_standard(patch_image).unsqueeze(0).to(device)
            features_uni2h_full = model_uni2h(img_uni2h)
            features_uni2h_full = features_uni2h_full.cpu().numpy().flatten()
            UNI2h_selected = features_uni2h_full[UNI2h_important_indices]
            
            # Virchow features
            img_virchow = transform_standard(patch_image).unsqueeze(0).to(device)
            embeddings_virchow = model_virchow(img_virchow)
            features_virchow_full = torch.cat([embeddings_virchow[:,0], embeddings_virchow[:,5:].mean(1)], dim=-1)
            features_virchow_full = features_virchow_full.cpu().numpy().flatten()
            Virchow_selected = features_virchow_full[Virchow_important_indices]
            
            # Virchow2 features
            img_virchow2 = transform_standard(patch_image).unsqueeze(0).to(device)
            embeddings_virchow2 = model_virchow2(img_virchow2)
            features_virchow2_full = torch.cat([embeddings_virchow2[:,0], embeddings_virchow2[:,5:].mean(1)], dim=-1)
            features_virchow2_full = features_virchow2_full.cpu().numpy().flatten()
            Virchow2_selected = features_virchow2_full[Virchow2_important_indices]
            
            # ProvGigapath features
            img_provgigapath = transform_provgigapath(patch_image).unsqueeze(0).to(device)
            features_provgigapath_full = model_provgigapath(img_provgigapath)
            features_provgigapath_full = features_provgigapath_full.cpu().numpy().flatten()
            ProvGigapath_selected = features_provgigapath_full[ProvGigapath_important_indices]
            
            # Conch features
            img_conch = transform_conch(patch_image).unsqueeze(0).to(device)
            features_conch_full = model_conch.encode_image(img_conch, proj_contrast=False, normalize=False)
            features_conch_full = features_conch_full.cpu().numpy().flatten()
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
        prediction = xgb_model.predict(dtest)[0]
        
        results.append({
            'patch_id': image_info['patch_id'],
            'row': image_info['row'],
            'col': image_info['col'],
            'x_start': image_info['x_start'],
            'y_start': image_info['y_start'],
            'x_end': image_info['x_end'],
            'y_end': image_info['y_end'],
            'predicted_proportion': float(prediction)
        })
    
    log(f"Completed prediction for {len(results)} patches")
    return results


def create_soft_segmentation(original_image, patch_results, proportion_threshold, cell_type):
    """Create soft segmentation by filling patches above threshold with cell-type specific colors
    
    Colors:
    - Cancer/Tumor: #E41A1C (red)
    - Stromal: #FFFF33 (yellow) 
    - T cells: #4DAF4A (green)
    - 70% opacity for all colors
    """
    
    # Create a copy of the original image for visualization
    segmented_image = original_image.copy()
    
    # Create overlay with cell-type specific colors
    overlay = np.zeros_like(original_image)
    
    # Define colors for different cell types
    if "cancer" in cell_type.lower() or "tumor" in cell_type.lower():
        # Cancer/Tumor: #E41A1C (red)
        overlay[:, :, 0] = 228  # Red
        overlay[:, :, 1] = 26   # Green
        overlay[:, :, 2] = 28   # Blue
    elif "stromal" in cell_type.lower():
        # Stromal: #FFFF33 (yellow)
        overlay[:, :, 0] = 255  # Red
        overlay[:, :, 1] = 255  # Green
        overlay[:, :, 2] = 51   # Blue
    elif "t cell" in cell_type.lower():
        # T cells: #4DAF4A (green)
        overlay[:, :, 0] = 77   # Red
        overlay[:, :, 1] = 175  # Green
        overlay[:, :, 2] = 74   # Blue
    else:
        # Default: dark green for unknown cell types
        overlay[:, :, 1] = 100
    
    patches_above_threshold = 0
    valid_patches = 0
    
    log(f"Applying soft segmentation with threshold: {proportion_threshold}")
    
    for result in tqdm(patch_results, desc="Creating segmentation"):
        if result['predicted_proportion'] is None:
            continue  # Skip white patches
            
        valid_patches += 1
        
        if result['predicted_proportion'] >= proportion_threshold:
            patches_above_threshold += 1
            
            # Get patch coordinates
            x_start = result['x_start']
            y_start = result['y_start']
            x_end = result['x_end']
            y_end = result['y_end']
            
            # Fill with 70% opaque cell-type specific color
            alpha = 0.3  # 70% opacity color, 30% original image
            segmented_image[y_start:y_end, x_start:x_end] = (
                (1 - alpha) * overlay[y_start:y_end, x_start:x_end] + 
                alpha * segmented_image[y_start:y_end, x_start:x_end]
            ).astype(np.uint8)
    
    log(f"Segmentation complete: {patches_above_threshold}/{valid_patches} patches above threshold")
    
    return segmented_image, patches_above_threshold, valid_patches


def soft_segmentation_wsi_multiple_thresholds(wsi_path, cell_type, thresholds, xgboost_model_path, save_dir):
    """
    Efficient function for soft segmentation with multiple thresholds
    Predicts once, then applies multiple thresholds
    
    Args:
        wsi_path: Path to the WSI image
        cell_type: Cell type to segment (e.g., "Cancer Cells", "Stromal Cells", "T cells")
        thresholds: List of thresholds (e.g., [0.3, 0.5, 0.8])
        xgboost_model_path: Path to the trained XGBoost model
        save_dir: Directory to save results
    """
    
    log(f"Starting efficient soft segmentation for {cell_type}")
    log(f"WSI path: {wsi_path}")
    log(f"Thresholds: {thresholds}")
    
    # Step 1: Create 100x100 patches (only once)
    patch_list, original_image = create_patches_100x100(wsi_path)
    
    # Step 2: Predict cell type proportions (only once)
    log("Performing prediction (this is the slow part, only done once)...")
    prediction_results = predict_cell_type_proportions(patch_list, cell_type, xgboost_model_path)
    
    # Step 3: Apply multiple thresholds and save results
    for threshold in thresholds:
        log(f"Creating segmentation for threshold {threshold}")
        
        # Create segmentation for this threshold
        segmented_image, patches_above_threshold, valid_patches = create_soft_segmentation(
            original_image, prediction_results, threshold, cell_type
        )
        
        # Visualize results
        plt.figure(figsize=(20, 16))
        
        # Create subplot layout
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title(f'Original WSI', fontsize=20, fontweight='bold', color='black')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(segmented_image)
        plt.title(f'Soft Segmentation: {cell_type}\nThreshold: {threshold:.1%}\n'
                  f'Highlighted: {patches_above_threshold}/{valid_patches} patches', 
                  fontsize=20, fontweight='bold', color='black')
        plt.axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save as TIFF with DPI=600 - emphasize cell type in filename
        wsi_basename = os.path.basename(wsi_path).replace('.tif', '').replace('_Cropped', '')
        save_path = f"{save_dir}/{cell_type.replace(' ', '_')}_Segmentation_{wsi_basename}_threshold{threshold}.tif"
        plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1, format='tiff')
        log(f"Soft segmentation saved to: {save_path} (DPI: 600)")
        
        # Also save raw segmented image array as TIFF 
        raw_save_path = f"{save_dir}/{cell_type.replace(' ', '_')}_Segmentation_{wsi_basename}_threshold{threshold}_raw.tif"
        segmented_pil = Image.fromarray(segmented_image)
        segmented_pil.save(raw_save_path, format='TIFF', compression='lzw')
        log(f"Raw segmented image saved to: {raw_save_path}")
        
        # Close the plot
        plt.close()
        
        # Print summary statistics
        log(f"=== Threshold {threshold} Summary ===")
        log(f"Patches above threshold: {patches_above_threshold}/{valid_patches}")
        if valid_patches > 0:
            log(f"Percentage above threshold: {patches_above_threshold/valid_patches:.1%}")


def soft_segmentation_wsi(wsi_path, cell_type, proportion_threshold, xgboost_model_path, save_path=None):
    """
    Main function for soft segmentation of WSI (single threshold)
    
    Args:
        wsi_path: Path to the WSI image
        cell_type: Cell type to segment (e.g., "Cancer Cells", "Stromal Cells", "T cells")
        proportion_threshold: Threshold for highlighting (e.g., 0.5 for 50%)
        xgboost_model_path: Path to the trained XGBoost model
        save_path: Optional path to save the result as .tif (DPI=600)
    """
    
    log(f"Starting soft segmentation for {cell_type} with threshold {proportion_threshold}")
    log(f"WSI path: {wsi_path}")
    
    # Step 1: Create 100x100 patches
    patch_list, original_image = create_patches_100x100(wsi_path)
    
    # Step 2: Predict cell type proportions
    prediction_results = predict_cell_type_proportions(patch_list, cell_type, xgboost_model_path)
    
    # Step 3: Create soft segmentation
    segmented_image, patches_above_threshold, valid_patches = create_soft_segmentation(
        original_image, prediction_results, proportion_threshold, cell_type
    )
    
    # Step 4: Visualize results
    plt.figure(figsize=(20, 16))
    
    # Create subplot layout
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title(f'Original WSI', fontsize=20, fontweight='bold', color='black')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title(f'Soft Segmentation: {cell_type}\nThreshold: {proportion_threshold:.1%}\n'
              f'Highlighted: {patches_above_threshold}/{valid_patches} patches', 
              fontsize=20, fontweight='bold', color='black')
    plt.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save as TIFF with DPI=600
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1, format='tiff')
    log(f"Soft segmentation saved to: {save_path} (DPI: 600)")
    
    # Also save raw segmented image array as TIFF 
    raw_save_path = save_path.replace('.tif', '_raw_segmentation.tif').replace('.tiff', '_raw_segmentation.tiff')
    segmented_pil = Image.fromarray(segmented_image)
    segmented_pil.save(raw_save_path, format='TIFF', compression='lzw')
    log(f"Raw segmented image saved to: {raw_save_path}")
    
    # Show the result
    plt.close()
    
    # Print summary statistics
    log("=== Segmentation Summary ===")
    log(f"Cell Type: {cell_type}")
    log(f"Proportion Threshold: {proportion_threshold:.1%}")
    log(f"Total patches: {len(patch_list)}")
    log(f"Valid patches (non-white): {valid_patches}")
    log(f"Patches above threshold: {patches_above_threshold}")
    if valid_patches > 0:
        log(f"Percentage of valid patches above threshold: {patches_above_threshold/valid_patches:.1%}")


# Example usage for cancer
if __name__ == "__main__":
    
    # Efficient processing with multiple thresholds
    thresholds = [0.3, 0.4, 0.5]
    save_dir = "Visual"
    
    ### For Cancer TENX152
    log("=== Processing TENX152 ===")
    wsi_path = "H&E/TENX152_Cropped.tif"
    cell_type = "Cancer Cells"
    xgboost_model_path = "xgboost_prediction/Cancer Cells_Combined_individual_level_ratio100/models/xgboost_model_leave_TENX152_out.model"
    
    soft_segmentation_wsi_multiple_thresholds(
        wsi_path=wsi_path,
        cell_type=cell_type,
        thresholds=thresholds,
        xgboost_model_path=xgboost_model_path,
        save_dir=save_dir
    )

    ### For Cancer 6723_KL_1
    log("=== Processing 6723_KL_1 ===")
    wsi_path = "H&E/6723_KL_1_cropped_scaled.tif"
    cell_type = "Cancer Cells"
    xgboost_model_path = "xgboost_prediction/Cancer Cells_Combined_individual_level_ratio100/models/xgboost_model_leave_6723_out.model"
    
    soft_segmentation_wsi_multiple_thresholds(
        wsi_path=wsi_path,
        cell_type=cell_type,
        thresholds=thresholds,
        xgboost_model_path=xgboost_model_path,
        save_dir=save_dir
    )
    
    ### For Cancer SH-16-07447
    log("=== Processing SH-16-07447 ===")
    wsi_path = "H&E/SH-16-07447_Cropped.tif"
    cell_type = "Cancer Cells"
    xgboost_model_path = "xgboost_prediction/Cancer Cells_Combined_individual_level_ratio100/models/xgboost_model_leave_SH-16-07447_out.model"
    
    soft_segmentation_wsi_multiple_thresholds(
        wsi_path=wsi_path,
        cell_type=cell_type,
        thresholds=thresholds,
        xgboost_model_path=xgboost_model_path,
        save_dir=save_dir
    )



# Example usage for stromal


    thresholds = [0.4, 0.5, 0.6]
    save_dir = "Visual"
    
    ### For Stromal 7003_AS_4
    log("=== Processing 7003_AS_4 ===")
    wsi_path = "H&E/7003_AS_4_cropped_scaled.tif"
    cell_type = "Stromal Cells"
    xgboost_model_path = "xgboost_prediction/Stromal Cells_Combined_individual_level_ratio100/models/xgboost_model_leave_7003_out.model"
    
    soft_segmentation_wsi_multiple_thresholds(
        wsi_path=wsi_path,
        cell_type=cell_type,
        thresholds=thresholds,
        xgboost_model_path=xgboost_model_path,
        save_dir=save_dir
    )

    ## For Stromal TENX152
    log("=== Processing TENX152 ===")
    wsi_path = "H&E/TENX152_Cropped.tif"
    cell_type = "Stromal Cells"
    xgboost_model_path = "xgboost_prediction/Stromal Cells_Combined_individual_level_ratio100/models/xgboost_model_leave_TENX152_out.model"
    
    soft_segmentation_wsi_multiple_thresholds(
        wsi_path=wsi_path,
        cell_type=cell_type,
        thresholds=thresholds,  
        xgboost_model_path=xgboost_model_path,
        save_dir=save_dir
    )


    ## For Stromal TENX49
    log("=== Processing TENX49 ===")
    wsi_path = "/Users/scui2/Desktop/hest_data/wsis/TENX49_Cropped.tif"
    cell_type = "Stromal Cells"
    xgboost_model_path = "xgboost_prediction/Stromal Cells_Combined_individual_level_ratio100/models/xgboost_model_leave_TENX49_out.model"
    
    soft_segmentation_wsi_multiple_thresholds(
        wsi_path=wsi_path,
        cell_type=cell_type,
        thresholds=thresholds,
        xgboost_model_path=xgboost_model_path,
        save_dir=save_dir
    )



if __name__ == "__main__":
    
   # Example usage for T cells
    thresholds = [0.15, 0.2, 0.25]
    save_dir = "Visual"
    
    ### For T cells SH-17-06138-A1
    log("=== Processing SH-17-06138-A1 ===")
    wsi_path = "H&E/SH-17-06138-A1_Cropped.tif"
    cell_type = "T cells"
    xgboost_model_path = "xgboost_prediction/T cells_Combined_individual_level_ratio100/models/xgboost_model_leave_SH-17-06138-A1_out.model"
    
    soft_segmentation_wsi_multiple_thresholds(
        wsi_path=wsi_path,
        cell_type=cell_type,
        thresholds=thresholds,
        xgboost_model_path=xgboost_model_path,
        save_dir=save_dir
    )

    # For T cells SU-15-18753-A1
    log("=== Processing SU-15-18753-A1 ===")
    wsi_path = "H&E/SU-15-18753-A1_Cropped.tif"
    cell_type = "T cells"
    xgboost_model_path = "xgboost_prediction/T cells_Combined_individual_level_ratio100/models/xgboost_model_leave_SU-15-18753-A1_out.model"
    
    soft_segmentation_wsi_multiple_thresholds(
        wsi_path=wsi_path,
        cell_type=cell_type,
        thresholds=thresholds,
        xgboost_model_path=xgboost_model_path,
        save_dir=save_dir
    )

    # For T cells TENX152
    log("=== Processing TENX152 ===")
    wsi_path = "/Users/scui2/Desktop/hest_data/wsis/TENX152_Cropped.tif"
    cell_type = "T cells"
    xgboost_model_path = "xgboost_prediction/T cells_Combined_individual_level_ratio100/models/xgboost_model_leave_TENX152_out.model"
    
    soft_segmentation_wsi_multiple_thresholds(
        wsi_path=wsi_path,
        cell_type=cell_type,
        thresholds=thresholds,
        xgboost_model_path=xgboost_model_path,
        save_dir=save_dir
    )

