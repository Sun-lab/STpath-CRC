"""
STPath-COAD: Model Consistency Analysis for Figure 5
===================================================

This script analyzes consistency of STPath model predictions across different conditions
and generates visualizations for Figure 5. Evaluates model robustness and reliability.

Author: Saishi Cui
Date: Sept 2025

Purpose: Evaluate model consistency across different image patches, assess prediction
reliability, and generate comprehensive consistency analysis visualizations for 
Figure 5 of the paper.
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
import pandas as pd

    


def create_patches(image_path, cut_grid_size=70):
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(image_path)
    image_np = np.array(image)
    
    height, width = image_np.shape[:2]
    print(f"Original image size: {width} x {height}")
    
    # Calculate patch dimensions - divide image into cut_grid_size x cut_grid_size patches
    patch_height = height // cut_grid_size
    patch_width = width // cut_grid_size
    
    print(f"Will create {cut_grid_size} x {cut_grid_size} = {cut_grid_size * cut_grid_size} patches")
    print(f"Each patch size: {patch_width} x {patch_height} pixels")
    
    patch_list = []
    
    for row in tqdm(range(cut_grid_size), desc="Processing rows"):
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
    
    return patch_list




# define the log function
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def load_Conch_model():
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token)
    model, transform = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token=hf_token)
    return model, transform



# load the pre-trained UNI2h model
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


# load the pre-trained ProvGigapath model
def load_ProvGigapath_model():
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token)
    tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    return tile_encoder


# load the pre-trained Virchow model
def load_Virchow_model():
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token)
    model_Virchow = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model_Virchow = model_Virchow.eval()
    return model_Virchow


# load the pre-trained Virchow2 model
def load_Virchow2_model():
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token)
    model_Virchow2 = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model_Virchow2 = model_Virchow2.eval()
    return model_Virchow2




def cell_type_proportion_prediction(image_info_list, cell_type, xgboost_model_path):

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

    important_features_path = f"/Users/scui2/ST/Colorectal_Cancer_HE_patches/xgboost_prediction/important_features_{cell_type}.pkl"
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
    
    for i, image_info in enumerate(image_info_list):
        log(f"Processing patch {i+1}/{len(image_info_list)}: {image_info['patch_id']}")
        
        # Check if patch is too white
        if is_white_patch(image_info['patch_array']):
            log(f"Patch {image_info['patch_id']} is too white, assigning NA")
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
        
        log(f"Patch {image_info['patch_id']} predicted proportion: {prediction:.4f} (features: {len(combined_features)})")
        
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


def visualize_patches(prediction_results, cell_type, save_path=None):

    
    # Create DataFrame from prediction results
    data_list = []
    for result in prediction_results:
        # Calculate patch center coordinates
        x_center = (result['x_start'] + result['x_end']) / 2
        y_center = (result['y_start'] + result['y_end']) / 2
        
        data_list.append({
            'patch_id': result['patch_id'],
            'x': x_center,
            'y': y_center,
            'row': result['row'],
            'col': result['col'],
            'predicted_proportion': result['predicted_proportion']
        })
    
    df = pd.DataFrame(data_list)
    
    # Filter out NA values for visualization
    valid_df = df[df['predicted_proportion'].notna()].copy()
    
    if len(valid_df) == 0:
        print("No valid predictions to visualize")
        return
    
    # Convert proportion to percentage
    valid_df['predicted_proportion_pct'] = valid_df['predicted_proportion'] * 100
    
    print(f"Visualizing {len(valid_df)} patches with valid predictions")
    print(f"Proportion range: {valid_df['predicted_proportion_pct'].min():.2f}% - {valid_df['predicted_proportion_pct'].max():.2f}%")
    
    # Calculate grid size from the number of patches to determine marker size
    total_patches = len(prediction_results)
    estimated_grid_size = int(np.sqrt(total_patches))
    
    # Adjust marker size based on grid size - larger grid = smaller markers
    # Base size 150 for 40x40, scale down as grid size increases
    base_size = 150
    size_factor = (40 / estimated_grid_size) ** 1.5  # Power to make the scaling more pronounced
    marker_size = max(20, int(base_size * size_factor))  # Minimum size of 20
    
    print(f"Estimated grid size: {estimated_grid_size}x{estimated_grid_size}")
    print(f"Using marker size: {marker_size}")
    
    # Create the hexagon scatter plot
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # Create scatter plot with hexagon markers
    scatter = plt.scatter(
        x=valid_df['x'],
        y=valid_df['y'],
        c=valid_df['predicted_proportion_pct'],
        cmap='coolwarm',  # color from light to dark
        marker='h',  # hexagon marker
        s=marker_size,  # Dynamic marker size based on grid
        edgecolors='black',
        linewidth=0.5
        # Color range automatically adjusted based on data
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    # Remove colorbar label
    # cbar.set_label(f'Predicted {cell_type} proportion (%)', fontsize=16, fontweight='bold', color='black')
    cbar.ax.tick_params(labelsize=20, colors='black', width=3)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_color('black')
    
    # Invert y axis to match image coordinate system
    ax.invert_yaxis()
    
    # Set thick black borders like other plots
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)
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
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.05)
        print(f"Visualization saved to: {save_path}")
    
    # Show the figure
    plt.show()
    
    return df



if __name__ == "__main__":

    ### Cancer 

    # 6723_KL_1

    cell_type = "Cancer Cells"
    image_path = "H&E/6723_KL_1_cropped_scaled.tif"
    xgboost_model_path = "xgboost_prediction/Cancer Cells_Combined_individual_level_ratio100/models/xgboost_model_leave_6723_out.model"

    cancer_6723_KL_1_result_dict = {}
    for cut_grid_size in [40, 50, 60, 70, 80, 90, 100]:
        image_info_list = create_patches(
            image_path=image_path,
            cut_grid_size=cut_grid_size,
        )

        print(f"\nCompleted! Created {len(image_info_list)} patches")

        # Predict cell type proportions
        results = cell_type_proportion_prediction(
            image_info_list=image_info_list,
            cell_type=cell_type,
            xgboost_model_path=xgboost_model_path
        )

        cancer_6723_KL_1_result_dict[cut_grid_size] = results


    # TENX152
    image_path = "H&E/TENX152_Cropped.tif"
    xgboost_model_path = "xgboost_prediction/Cancer Cells_Combined_individual_level_ratio100/models/xgboost_model_leave_TENX152_out.model"

    cancer_TENX152_result_dict = {}
    for cut_grid_size in [40, 50, 60, 70, 80, 90, 100]:
        image_info_list = create_patches(
            image_path=image_path,
            cut_grid_size=cut_grid_size,
        )
        results = cell_type_proportion_prediction(
            image_info_list=image_info_list,
            cell_type=cell_type,
            xgboost_model_path=xgboost_model_path
        )
        cancer_TENX152_result_dict[cut_grid_size] = results



    # SH-16-07447
    image_path = "H&E/SH-16-07447_Cropped.tif"
    xgboost_model_path = "/xgboost_prediction/Cancer Cells_Combined_individual_level_ratio100/models/xgboost_model_leave_SH-16-07447_out.model"

    cancer_SH_16_07447_result_dict = {}
    for cut_grid_size in [40, 50, 60, 70, 80, 90, 100]:
        image_info_list = create_patches(
            image_path=image_path,
            cut_grid_size=cut_grid_size,
        )
        results = cell_type_proportion_prediction(
            image_info_list=image_info_list,
            cell_type=cell_type,
            xgboost_model_path=xgboost_model_path
        )
        cancer_SH_16_07447_result_dict[cut_grid_size] = results






    ### Stromal Cells
    cell_type = "Stromal Cells"
    # 7003_AS_4
    image_path = "H&E/7003_AS_4_cropped_scaled.tif"
    xgboost_model_path = "xgboost_prediction/Stromal Cells_Combined_individual_level_ratio100/models/xgboost_model_leave_7003_out.model"

    stromal_7003_AS_4_result_dict = {}
    for cut_grid_size in [40, 50, 60, 70, 80, 90, 100]:
        image_info_list = create_patches(
            image_path=image_path,
            cut_grid_size=cut_grid_size,
        )

        print(f"\nCompleted! Created {len(image_info_list)} patches")

        # Predict cell type proportions
        results = cell_type_proportion_prediction(
            image_info_list=image_info_list,
            cell_type=cell_type,
            xgboost_model_path=xgboost_model_path
        )

        stromal_7003_AS_4_result_dict[cut_grid_size] = results




    # TENX152
    image_path = "H&E/TENX152_Cropped.tif"
    xgboost_model_path = "xgboost_prediction/Stromal Cells_Combined_individual_level_ratio100/models/xgboost_model_leave_TENX152_out.model"

    stromal_TENX152_result_dict = {}
    for cut_grid_size in [40, 50, 60, 70, 80, 90, 100]:
        image_info_list = create_patches(
            image_path=image_path,
            cut_grid_size=cut_grid_size,
        )

        print(f"\nCompleted! Created {len(image_info_list)} patches")

        # Predict cell type proportions
        results = cell_type_proportion_prediction(
            image_info_list=image_info_list,
            cell_type=cell_type,
            xgboost_model_path=xgboost_model_path
        )

        stromal_TENX152_result_dict[cut_grid_size] = results



    # TENX49
    image_path = "H&E/TENX49_Cropped.tif"
    xgboost_model_path = "xgboost_prediction/Stromal Cells_Combined_individual_level_ratio100/models/xgboost_model_leave_TENX49_out.model"

    stromal_TENX49_result_dict = {}
    for cut_grid_size in [40, 50, 60, 70, 80, 90, 100]:
        image_info_list = create_patches(
            image_path=image_path,
            cut_grid_size=cut_grid_size,
        )

        print(f"\nCompleted! Created {len(image_info_list)} patches")

        # Predict cell type proportions
        results = cell_type_proportion_prediction(
            image_info_list=image_info_list,
            cell_type=cell_type,
            xgboost_model_path=xgboost_model_path
        )

        stromal_TENX49_result_dict[cut_grid_size] = results



    ###### T cells
    cell_type = "T cells"
    ## SH-17-06138-A1

    image_path = "H&E/SH-17-06138-A1_Cropped.tif"
    xgboost_model_path = "xgboost_prediction/T cells_Combined_individual_level_ratio100/models/xgboost_model_leave_SH-17-06138-A1_out.model"
    t_cell_SH_17_06138_A1_result_dict = {}
    for cut_grid_size in [40, 50, 60, 70, 80, 90, 100]:
        image_info_list = create_patches(
            image_path=image_path,
            cut_grid_size=cut_grid_size,
        )

        print(f"\nCompleted! Created {len(image_info_list)} patches")

        # Predict cell type proportions
        results = cell_type_proportion_prediction(
            image_info_list=image_info_list,
            cell_type=cell_type,
            xgboost_model_path=xgboost_model_path
        )

        t_cell_SH_17_06138_A1_result_dict[cut_grid_size] = results


    ## SU-15-18753-A1

    image_path = "H&E/SU-15-18753-A1_Cropped.tif"
    xgboost_model_path = "xgboost_prediction/T cells_Combined_individual_level_ratio100/models/xgboost_model_leave_SU-15-18753-A1_out.model"
    t_cell_SU_15_18753_A1_result_dict = {}
    for cut_grid_size in [40, 50, 60, 70, 80, 90, 100]:
        image_info_list = create_patches(
            image_path=image_path,
            cut_grid_size=cut_grid_size,
        )
        results = cell_type_proportion_prediction(
            image_info_list=image_info_list,
            cell_type=cell_type,
            xgboost_model_path=xgboost_model_path
        )
        t_cell_SU_15_18753_A1_result_dict[cut_grid_size] = results


    ## TENX152
    image_path = "H&E/TENX152_Cropped.tif"
    xgboost_model_path = "xgboost_prediction/T cells_Combined_individual_level_ratio100/models/xgboost_model_leave_TENX152_out.model"
    t_cell_TENX152_result_dict = {}
    for cut_grid_size in [40, 50, 60, 70, 80, 90, 100]:
        image_info_list = create_patches(
            image_path=image_path,
            cut_grid_size=cut_grid_size,
        )
        results = cell_type_proportion_prediction(
            image_info_list=image_info_list,
            cell_type=cell_type,
            xgboost_model_path=xgboost_model_path
        )
        t_cell_TENX152_result_dict[cut_grid_size] = results




# Create summary dataframe
import pandas as pd

summary_data = []

# Define all result dictionaries with sample names and cell types
all_results = [
    (cancer_6723_KL_1_result_dict, "6723_KL_1", "Cancer Cells"),
    (cancer_TENX152_result_dict, "TENX152", "Cancer Cells"),
    (cancer_SH_16_07447_result_dict, "SH-16-07447", "Cancer Cells"),
    (stromal_7003_AS_4_result_dict, "7003_AS_4", "Stromal Cells"),
    (stromal_TENX152_result_dict, "TENX152", "Stromal Cells"),
    (stromal_TENX49_result_dict, "TENX49", "Stromal Cells"),
    (t_cell_SH_17_06138_A1_result_dict, "SH-17-06138-A1", "T cells"),
    (t_cell_SU_15_18753_A1_result_dict, "SU-15-18753-A1", "T cells"),
    (t_cell_TENX152_result_dict, "TENX152", "T cells")
]

# Process each result dictionary
for result_dict, sample_name, cell_type in all_results:
    for grid_size, results in result_dict.items():
        # Extract all valid proportions
        proportions = []
        for item in results:
            if item['predicted_proportion'] is not None:
                proportions.append(item['predicted_proportion'])
        
        # Calculate average proportion
        avg_proportion = np.mean(proportions) if proportions else None
        
        # Add to summary data
        summary_data.append({
            'Sample_Name': sample_name,
            'Cell_Type': cell_type,
            'Grid_Size': f"{grid_size}x{grid_size}",
            'Average_Cell_Type_Proportion': avg_proportion,
            'Valid_Patches': len(proportions),
            'Total_Patches': len(results)
        })

# Create DataFrame
summary_df = pd.DataFrame(summary_data)
print("Summary DataFrame:")
print(summary_df.head(20))
print(f"\nDataFrame shape: {summary_df.shape}")

# Save to CSV
summary_df.to_csv("Consistency_Results/consistency_analysis_summary.csv", index=False)
summary_df = pd.read_csv("Consistency_Results/consistency_analysis_summary.csv")

# Create a new column for the grid size in pixels
summary_df['Grid_Size_Pixels'] = summary_df['Grid_Size'].str.replace('x.*', '', regex=True).astype(int) * 224


# Create line chart with flatter aspect ratio
plt.figure(figsize=(12, 6))
ax = plt.gca()

# Define colors for cell types
cell_type_colors = {
    'Cancer Cells': '#E41A1C',  # Red
    'Stromal Cells': '#FFFF33',  # Yellow  
    'T cells': '#4DAF4A'  # Green
}

# Define markers for different samples
unique_samples = summary_df['Sample_Name'].unique()
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '+', 'x']
sample_markers = {sample: markers[i % len(markers)] for i, sample in enumerate(unique_samples)}

# Convert grid size to numeric for plotting
summary_df['Grid_Size_Numeric'] = summary_df['Grid_Size'].str.replace('x.*', '', regex=True).astype(int)

# Import Rectangle for range visualization
from matplotlib.patches import Rectangle

# First, draw range rectangles for each cell type (using first sample only)
for cell_type in summary_df['Cell_Type'].unique():
    # Get first sample for this cell type to avoid rectangle overlap
    samples_for_cell_type = summary_df[summary_df['Cell_Type'] == cell_type]['Sample_Name'].unique()
    first_sample = samples_for_cell_type[0]
    
    # Filter data for this cell type and first sample only
    subset = summary_df[(summary_df['Cell_Type'] == cell_type) & 
                       (summary_df['Sample_Name'] == first_sample)]
    
    if len(subset) > 0:
        # Sort by grid size
        subset = subset.sort_values('Grid_Size_Numeric')
        
        # Calculate range for rectangle
        min_grid = subset['Grid_Size_Numeric'].min()
        max_grid = subset['Grid_Size_Numeric'].max()
        min_proportion = subset['Average_Cell_Type_Proportion'].min()
        max_proportion = subset['Average_Cell_Type_Proportion'].max()
        
        # Create rectangle representing the range (stability)
        width = max_grid - min_grid
        height = max_proportion - min_proportion
        
        # Add rectangle with cell type color but very transparent
        rect = Rectangle((min_grid, min_proportion), width, height,
                       linewidth=0, 
                       facecolor=cell_type_colors[cell_type],
                       alpha=0.15,  # Very transparent
                       zorder=1)  # Behind everything else
        ax.add_patch(rect)

# Then, plot lines and points for all cell type and sample combinations
for cell_type in summary_df['Cell_Type'].unique():
    for sample in summary_df['Sample_Name'].unique():
        # Filter data for this cell type and sample
        subset = summary_df[(summary_df['Cell_Type'] == cell_type) & 
                           (summary_df['Sample_Name'] == sample)]
        
        if len(subset) > 0:
            # Sort by grid size for proper line plotting
            subset = subset.sort_values('Grid_Size_Numeric')
            
            # Plot line
            plt.plot(subset['Grid_Size_Numeric'], 
                    subset['Average_Cell_Type_Proportion'],
                    color=cell_type_colors[cell_type],
                    linewidth=3,
                    alpha=0.8,
                    zorder=3)
            
            # Plot points
            plt.scatter(subset['Grid_Size_Numeric'], 
                       subset['Average_Cell_Type_Proportion'],
                       marker=sample_markers[sample],
                       color='black',
                       s=100,  # Point size
                       zorder=5,  # Make sure points are on top
                       edgecolors='black',
                       linewidth=1)

# Set labels and title
plt.xlabel('Patch Grid Resolution', fontsize=18, fontweight='bold', color='black')
plt.ylabel('Overall Cell Type Proportion', fontsize=18, fontweight='bold', color='black')

# Set thick black borders
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['top'].set_linewidth(5)
ax.spines['right'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
ax.spines['bottom'].set_linewidth(5)
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

# Set tick parameters
ax.tick_params(axis='both', which='major', labelsize=14, colors='black', width=3, length=8)
ax.tick_params(axis='both', which='minor', colors='black', width=2, length=4)

# Make tick labels bold
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_color('black')

# Set x-axis ticks to show all grid sizes with proper format
grid_sizes = [40, 50, 60, 70, 80, 90, 100]
grid_labels = [f'{size}×{size}' for size in grid_sizes]
plt.xticks(grid_sizes, grid_labels, rotation=20)

# Add gray dotted grid lines
ax.grid(True, linestyle=':', linewidth=2, color='gray', alpha=0.7)
ax.set_axisbelow(True)  # Put grid behind the data

# Create custom legends
from matplotlib.lines import Line2D

# Legend for cell types (colors)
cell_type_legend = [Line2D([0], [0], color=color, linewidth=3, label=cell_type) 
                   for cell_type, color in cell_type_colors.items()]

# Legend for samples (markers)
sample_legend = [Line2D([0], [0], marker=marker, color='black', linewidth=0, 
                       markersize=8, label=sample, markerfacecolor='black') 
                for sample, marker in sample_markers.items()]

# Add legends outside the plot area on the right side
legend1 = plt.legend(handles=cell_type_legend, title='Cell Type', 
                    loc='upper left', bbox_to_anchor=(1.02, 1.0),
                    title_fontsize=14, fontsize=12, frameon=True, 
                    fancybox=True, shadow=True)
legend1.get_title().set_fontweight('bold')
legend1.get_title().set_color('black')

# Make legend text bold
for text in legend1.get_texts():
    text.set_fontweight('bold')
    text.set_color('black')

legend2 = plt.legend(handles=sample_legend, title='Sample', 
                    loc='upper left', bbox_to_anchor=(1.02, 0.6),
                    title_fontsize=14, fontsize=10, frameon=True,
                    fancybox=True, shadow=True, ncol=1)
legend2.get_title().set_fontweight('bold')
legend2.get_title().set_color('black')

# Make legend text bold
for text in legend2.get_texts():
    text.set_fontweight('bold')
    text.set_color('black')

# Add the first legend back (matplotlib removes it when adding second)
plt.gca().add_artist(legend1)

# Adjust layout to accommodate external legends
plt.tight_layout()

# Save the plot with extra space for legends
save_path = "Consistency_Results/consistency_line_chart.png"
plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.2)
print(f"Line chart saved to: {save_path}")

plt.show()



cancer_6723_KL_1_40x40 = visualize_patches(cancer_6723_KL_1_result_dict[40], "Cancer Cells", save_path="Consistency_Results/cancer_6723_KL_1_40x40.png")
cancer_6723_KL_1_70x70 = visualize_patches(cancer_6723_KL_1_result_dict[70], "Cancer Cells", save_path="Consistency_Results/cancer_6723_KL_1_70x70.png")
cancer_6723_KL_1_100x100 = visualize_patches(cancer_6723_KL_1_result_dict[100], "Cancer Cells", save_path="Consistency_Results/cancer_6723_KL_1_100x100.png")

cancer_TENX152_40x40 = visualize_patches(cancer_TENX152_result_dict[40], "Cancer Cells", save_path="Consistency_Results/cancer_TENX152_40x40.png")
cancer_TENX152_70x70 = visualize_patches(cancer_TENX152_result_dict[70], "Cancer Cells", save_path="Consistency_Results/cancer_TENX152_70x70.png")
cancer_TENX152_100x100 = visualize_patches(cancer_TENX152_result_dict[100], "Cancer Cells", save_path="Consistency_Results/cancer_TENX152_100x100.png")

cancer_SH_16_07447_40x40 = visualize_patches(cancer_SH_16_07447_result_dict[40], "Cancer Cells", save_path="Consistency_Results/cancer_SH_16_07447_40x40.png")
cancer_SH_16_07447_70x70 = visualize_patches(cancer_SH_16_07447_result_dict[70], "Cancer Cells", save_path="Consistency_Results/cancer_SH_16_07447_70x70.png")
cancer_SH_16_07447_100x100 = visualize_patches(cancer_SH_16_07447_result_dict[100], "Cancer Cells", save_path="Consistency_Results/cancer_SH_16_07447_100x100.png")    


stromal_7003_AS_4_40x40 = visualize_patches(stromal_7003_AS_4_result_dict[40], "Stromal Cells", save_path="Consistency_Results/stromal_7003_AS_4_40x40.png")
stromal_7003_AS_4_70x70 = visualize_patches(stromal_7003_AS_4_result_dict[70], "Stromal Cells", save_path="Consistency_Results/stromal_7003_AS_4_70x70.png")
stromal_7003_AS_4_100x100 = visualize_patches(stromal_7003_AS_4_result_dict[100], "Stromal Cells", save_path="Consistency_Results/stromal_7003_AS_4_100x100.png")

stromal_TENX152_40x40 = visualize_patches(stromal_TENX152_result_dict[40], "Stromal Cells", save_path="Consistency_Results/stromal_TENX152_40x40.png")
stromal_TENX152_70x70 = visualize_patches(stromal_TENX152_result_dict[70], "Stromal Cells", save_path="Consistency_Results/stromal_TENX152_70x70.png")
stromal_TENX152_100x100 = visualize_patches(stromal_TENX152_result_dict[100], "Stromal Cells", save_path="Consistency_Results/stromal_TENX152_100x100.png")

stromal_TENX49_40x40 = visualize_patches(stromal_TENX49_result_dict[40], "Stromal Cells", save_path="Consistency_Results/stromal_TENX49_40x40.png")
stromal_TENX49_70x70 = visualize_patches(stromal_TENX49_result_dict[70], "Stromal Cells", save_path="Consistency_Results/stromal_TENX49_70x70.png")
stromal_TENX49_100x100 = visualize_patches(stromal_TENX49_result_dict[100], "Stromal Cells", save_path="Consistency_Results/stromal_TENX49_100x100.png")


t_cell_SH_17_06138_A1_40x40 = visualize_patches(t_cell_SH_17_06138_A1_result_dict[40], "T cells", save_path="Consistency_Results/t_cell_SH_17_06138_A1_40x40.png")
t_cell_SH_17_06138_A1_70x70 = visualize_patches(t_cell_SH_17_06138_A1_result_dict[70], "T cells", save_path="Consistency_Results/t_cell_SH_17_06138_A1_70x70.png")
t_cell_SH_17_06138_A1_100x100 = visualize_patches(t_cell_SH_17_06138_A1_result_dict[100], "T cells", save_path="Consistency_Results/t_cell_SH_17_06138_A1_100x100.png")


t_cell_SU_15_18753_A1_40x40 = visualize_patches(t_cell_SU_15_18753_A1_result_dict[40], "T cells", save_path="Consistency_Results/t_cell_SU_15_18753_A1_40x40.png")
t_cell_SU_15_18753_A1_70x70 = visualize_patches(t_cell_SU_15_18753_A1_result_dict[70], "T cells", save_path="Consistency_Results/t_cell_SU_15_18753_A1_70x70.png")
t_cell_SU_15_18753_A1_100x100 = visualize_patches(t_cell_SU_15_18753_A1_result_dict[100], "T cells", save_path="Consistency_Results/t_cell_SU_15_18753_A1_100x100.png")


t_cell_TENX152_40x40 = visualize_patches(t_cell_TENX152_result_dict[40], "T cells", save_path="Consistency_Results/t_cell_TENX152_40x40.png")
t_cell_TENX152_70x70 = visualize_patches(t_cell_TENX152_result_dict[70], "T cells", save_path="Consistency_Results/t_cell_TENX152_70x70.png")
t_cell_TENX152_100x100 = visualize_patches(t_cell_TENX152_result_dict[100], "T cells", save_path="Consistency_Results/t_cell_TENX152_100x100.png")


cancer_info = torch.load("Training_features/Cancer Cells_training_precomputed_features_Virchow2.pt")


# Find true overall cell type proportion
sample_ids_array = np.array(cancer_info["sample_ids"])
mask_6723_KL_1 = sample_ids_array == "6723_KL_1"
sample_6723_KL_1_cancer_proportion = cancer_info["celltype_proportions"][mask_6723_KL_1]
sample_6723_KL_1_cancer_proportion.mean()


