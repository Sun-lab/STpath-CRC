"""
STPath-COAD: Image Patch Creation for Colorectal Cancer Analysis
==============================================================

This script creates tissue patches from H&E whole slide images and coordinates them 
with spatial transcriptomics spots for subsequent analysis. Includes tissue 
segmentation using Otsu thresholding and patch filtering.

Author: Saishi Cui
Date: December 2025

Purpose: Process H&E whole slide images to create patches aligned with spatial
transcriptomics coordinates, perform tissue segmentation, and prepare data for
CARD deconvolution and cell type proportion prediction.
"""

import numpy as np
import tifffile as tiff
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import shutil
from tqdm import tqdm
import cv2
from skimage import measure
import glob


### find the HE files ###########

HE_files = []
for root, dirs, files in os.walk("data/H&E"):
    for file in files:
        if file.endswith(".tif"):  
            HE_files.append(os.path.join(root, file))

HE_files_sorted = sorted(HE_files)

 ## Some files need to be rotated 
HE_files_rotated = []
for root, dirs, files in os.walk("data/H&E_rotated"):
    for file in files:
        if file.endswith(".tif"):  #
            HE_files_rotated.append(os.path.join(root, file))

HE_files_rotated_sorted = sorted(HE_files_rotated)



extracted_parts = []

for file_path in HE_files_sorted:
 
    file_name = os.path.basename(file_path)
    
  
    match = re.match(r'(\d{4}_\w{2}_\d+)', file_name)
    if match:
        extracted_parts.append(match.group(1))

extracted_parts_unique = sorted(list(set(extracted_parts)))




extracted_parts_rotated = []

for file_path in HE_files_rotated_sorted:
 
    file_name = os.path.basename(file_path)
    
  
    match = re.match(r'(\d{4}_\w{2}_\d+)', file_name)
    if match:
        extracted_parts_rotated.append(match.group(1))

extracted_parts_unique_rotated = sorted(list(set(extracted_parts_rotated)))


image_path_list = []
pts_df_path_list = []
image_path_list_rotated = []    
pts_df_path_list_rotated = []
for part in extracted_parts_unique:
    image_path_list.append(f"data/H&E/{part}_cropped_scaled.tif")
    pts_df_path_list.append(f"data/ST_level3_data/{part}_tissue_positions.csv")

for part in extracted_parts_unique_rotated:
    image_path_list_rotated.append(f"data/H&E_rotated/{part}_cropped_scaled.tif")
    pts_df_path_list_rotated.append(f"data/ST_level3_data/{part}_tissue_positions.csv")


### Matching the spots with the HE images ###########

def matching_spots_with_WSI(pts_df_path, image_path, sample_id):

    pts_df = pd.read_csv(pts_df_path, header= 0)
    if pts_df.columns[0] == "barcode":
        None
    else:
        pts_df = pd.read_csv(pts_df_path, header=None)
        pts_df.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]

    pts_width = pts_df["pxl_col_in_fullres"].max() - pts_df["pxl_col_in_fullres"].min()
    pts_height = pts_df["pxl_row_in_fullres"].max() - pts_df["pxl_row_in_fullres"].min()


    Image.MAX_IMAGE_PIXELS = None 
    image = Image.open(image_path)
    image_np = np.array(image)

    WSI_width, WSI_height, channel = image_np.shape
    scale_factor_col = WSI_width / pts_width
    scale_factor_row = WSI_height / pts_height

    pts_moved_df = pts_df.copy()
    pts_moved_df["pxl_col_in_fullres"] = pts_df["pxl_col_in_fullres"] - pts_df["pxl_col_in_fullres"].min()
    pts_moved_df["pxl_row_in_fullres"] = pts_df["pxl_row_in_fullres"] - pts_df["pxl_row_in_fullres"].min()


    pts_scaled_df = pts_moved_df.copy()
    pts_scaled_df["pxl_col_in_fullres"] = pts_moved_df["pxl_col_in_fullres"] * scale_factor_col
    pts_scaled_df["pxl_row_in_fullres"] = pts_moved_df["pxl_row_in_fullres"] * scale_factor_row


    # Use Otsu segmentation method for tissue region segmentation
    from skimage.color import rgb2gray
    from skimage.filters import threshold_otsu
    from skimage.morphology import closing, remove_small_objects, dilation, erosion, disk
    
    print("Starting tissue segmentation using Otsu method...")
    
    # 1. Convert to grayscale
    gray = rgb2gray(image_np)
    
    # 2. Otsu automatic threshold segmentation
    thresh = threshold_otsu(gray)
    binary = gray < thresh  # Note: H&E tissue areas are dark, invert
    print(f"Otsu threshold: {thresh:.3f}")
    
    # 3. Morphological processing (remove small fragments, fill holes, etc., increase parameters to merge nearby regions)
    # Use larger structural elements to connect nearby regions
    cleaned = closing(binary, disk(15))  # Increase disk radius from 7 to 15
    cleaned = remove_small_objects(cleaned, min_size=2000)  # Increase minimum area from 500 to 2000
    
    # Additional morphological operations: dilate then erode to further connect nearby regions
    cleaned = dilation(cleaned, disk(10))  # Dilation operation to connect nearby regions
    cleaned = erosion(cleaned, disk(8))    # Erosion operation to restore general shape
    
    # 4. Connected component labeling
    labels = measure.label(cleaned)
    props = measure.regionprops(labels)
    print(f"Detected {len(props)} connected components")
    
    # 5. Assign region labels to each spot
    spot_labels = []
    for _, spot in pts_scaled_df.iterrows():
        x = int(spot['pxl_col_in_fullres'])
        y = int(spot['pxl_row_in_fullres'])
        if x < labels.shape[1] and y < labels.shape[0]:
            spot_labels.append(labels[y, x])
        else:
            spot_labels.append(0)
    
    pts_scaled_df['tissue_region'] = spot_labels
    
    # Step 1: Set all in_tissue=0 spots as background (-1)
    pts_scaled_df.loc[pts_scaled_df['in_tissue'] == 0, 'tissue_region'] = -1
    
    # Step 2: Count spots in remaining regions (only count in_tissue=1 spots)
    in_tissue_spots = pts_scaled_df[pts_scaled_df['in_tissue'] == 1]
    region_counts_initial = in_tissue_spots['tissue_region'].value_counts()
    print(f"Initial region distribution (in_tissue=1 only):")
    for region_id in sorted(region_counts_initial.index):
        print(f"  Region {region_id}: {region_counts_initial[region_id]} spots")
    
    # Step 3: Find regions with <30 spots (including background region 0)
    small_regions = []
    for region_id, count in region_counts_initial.items():
        if region_id == 0 or count < 30:  # Include background region 0 and small regions
            small_regions.append(region_id)
    
    # Step 4: Reassign small regions as background (-1)
    for region_id in small_regions:
        pts_scaled_df.loc[pts_scaled_df['tissue_region'] == region_id, 'tissue_region'] = -1
    
    # Step 5: Renumber main regions as 0,1,2,3...
    remaining_regions = sorted([r for r in region_counts_initial.index if r not in small_regions])
    region_mapping = {}
    for i, old_region in enumerate(remaining_regions):
        region_mapping[old_region] = i
    
    # Apply new numbering
    for old_region, new_region in region_mapping.items():
        pts_scaled_df.loc[pts_scaled_df['tissue_region'] == old_region, 'tissue_region'] = new_region
    
    print(f"\nProcessing results:")
    print(f"1. All in_tissue=0 spots → region=-1")
    print(f"2. {len(small_regions)} small regions (<30 spots) → region=-1")
    print(f"   Small region list: {small_regions}")
    print(f"3. Main regions renumbered: {region_mapping}")
    
    # Create labeled image for visualization
    labeled_img = labels
    
    # Create region_masks dictionary for compatibility
    region_masks = {}
    unique_regions = sorted([r for r in pts_scaled_df['tissue_region'].unique() if r >= 0])
    for region_id in unique_regions:
        region_spots = pts_scaled_df[pts_scaled_df['tissue_region'] == region_id]
        if len(region_spots) > 0:
            mask = np.zeros(labels.shape, dtype=bool)
            for _, spot in region_spots.iterrows():
                x = int(spot['pxl_col_in_fullres'])
                y = int(spot['pxl_row_in_fullres'])
                if 0 <= x < labels.shape[1] and 0 <= y < labels.shape[0]:
                    mask[y, x] = True
            region_masks[region_id] = mask
    
    # Visualization: using Otsu method results
    plt.figure(figsize=(15, 12))
    
    # Get all unique regions
    unique_regions = sorted(pts_scaled_df['tissue_region'].unique())
    
    # Define color scheme: -1 as blue, 0,1,2,3 use contrasting colors
    region_color_map = {
        -1: 'blue',       # Background
        0: 'red',         # Tissue region 0
        1: 'green',       # Tissue region 1  
        2: 'orange',      # Tissue region 2
        3: 'purple',      # Tissue region 3
        4: 'cyan',        # If more regions exist
        5: 'magenta',     # If more regions exist
        6: 'yellow',      # If more regions exist
        7: 'brown'        # If more regions exist
    }
    
    # Left plot: show segmented connected components (colored)
    plt.subplot(1, 2, 1)
    plt.imshow(labeled_img, cmap='nipy_spectral')
    plt.title(f"{sample_id} - Otsu Connected Components")
    plt.axis('off')
    
    # Right plot: original image + spots colored by region (show all region=-1 as blue)
    plt.subplot(1, 2, 2)
    plt.imshow(image_np)
    
    # Plot all spots
    legend_elements = []
    from matplotlib.patches import Patch
    
    for region_id in unique_regions:
        region_spots = pts_scaled_df[pts_scaled_df['tissue_region'] == region_id]
        if len(region_spots) > 0:
            color = region_color_map.get(region_id, 'gray')  # Default gray
            
            if region_id == -1:
                label = f'Background ({len(region_spots)} spots)'
                # Show all background spots
                plt.scatter(
                    region_spots['pxl_col_in_fullres'], 
                    region_spots['pxl_row_in_fullres'],
                    s=8,
                    c=color,
                    marker='x',
                    alpha=0.7,
                    label=label
                )
            else:
                # Only show spot count for in_tissue=1 tissue regions
                in_tissue_count = len(region_spots[region_spots['in_tissue'] == 1])
                label = f'Region {region_id} ({in_tissue_count} spots)'
                # Only plot in_tissue=1 spots
                tissue_spots = region_spots[region_spots['in_tissue'] == 1]
                
                if len(tissue_spots) > 0:  # Ensure there are spots to plot
                    plt.scatter(
                        tissue_spots['pxl_col_in_fullres'], 
                        tissue_spots['pxl_row_in_fullres'],
                        s=8,
                        c=color,
                        marker='x',
                        alpha=0.7,
                        label=label
                    )
            
            # Add legend elements
            legend_elements.append(Patch(facecolor=color, edgecolor='k', label=label))
    
    plt.title(f"{sample_id} - Spots by Otsu Region")
    plt.axis('off')
    if legend_elements:
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Statistical information
    region_counts = pts_scaled_df['tissue_region'].value_counts().sort_index()
    print(f"{sample_id} - Spot count per region:")
    for region_id in region_counts.index:
        if region_id == -1:
            print(f"Background: {region_counts[region_id]} spots")
        else:
            tissue_count = len(pts_scaled_df[(pts_scaled_df['tissue_region'] == region_id) & (pts_scaled_df['in_tissue'] == 1)])
            print(f"Region {region_id}: {tissue_count} spots (in tissue)")
    
    # Additional statistics
    total_background = len(pts_scaled_df[pts_scaled_df['tissue_region'] == -1])
    total_tissue = len(pts_scaled_df[pts_scaled_df['tissue_region'] >= 0])
    in_tissue_0_count = len(pts_scaled_df[pts_scaled_df['in_tissue'] == 0])
    
    print(f"\nComplete statistics:")
    print(f"Total spots: {len(pts_scaled_df)}")
    print(f"Background region (region=-1): {total_background} spots")
    print(f"  ├─ in_tissue=0: {in_tissue_0_count} spots")
    print(f"  └─ in_tissue=1 but belongs to small regions: {total_background - in_tissue_0_count} spots")
    print(f"Tissue regions (region>=0): {total_tissue} spots (all are in_tissue=1)")

    # save the dataframe with region labels
    pts_scaled_df.to_csv(f"data/Scaled_Spatial_coordinates/{sample_id}_spots_coordinates_scaled.csv")
    print(f"{sample_id} - the dataframe with region labels has been saved")


for i in range(len(extracted_parts_unique)):
    matching_spots_with_WSI(pts_df_path_list[i], image_path_list[i], extracted_parts_unique[i])








def matching_spots_with_WSI_rotated(pts_df_path, image_path, sample_id):

    pts_df = pd.read_csv(pts_df_path, header= 0)
    if pts_df.columns[0] == "barcode":
        None
    else:
        pts_df = pd.read_csv(pts_df_path, header=None)
        pts_df.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]

    pts_width = pts_df["pxl_col_in_fullres"].max() - pts_df["pxl_col_in_fullres"].min()
    pts_height = pts_df["pxl_row_in_fullres"].max() - pts_df["pxl_row_in_fullres"].min()


    Image.MAX_IMAGE_PIXELS = None 
    image = Image.open(image_path)
    image_np = np.array(image)

    WSI_width, WSI_height, channel = image_np.shape
    scale_factor_col = WSI_width / pts_width
    scale_factor_row = WSI_height / pts_height

    pts_moved_df = pts_df.copy()
    pts_moved_df["pxl_col_in_fullres"] = pts_df["pxl_col_in_fullres"] - pts_df["pxl_col_in_fullres"].min()
    pts_moved_df["pxl_row_in_fullres"] = pts_df["pxl_row_in_fullres"] - pts_df["pxl_row_in_fullres"].min()


    pts_scaled_df = pts_moved_df.copy()
    pts_scaled_df["pxl_col_in_fullres"] = pts_moved_df["pxl_col_in_fullres"] * scale_factor_col
    pts_scaled_df["pxl_row_in_fullres"] = pts_moved_df["pxl_row_in_fullres"] * scale_factor_row


    pts_rotated_df = pts_scaled_df.copy()
    center_x = pts_scaled_df["pxl_col_in_fullres"].max() / 2
    center_y = pts_scaled_df["pxl_row_in_fullres"].max() / 2

    pts_rotated_df["pxl_col_in_fullres"] = 2*center_x - pts_scaled_df["pxl_col_in_fullres"]
    pts_rotated_df["pxl_row_in_fullres"] = 2*center_y - pts_scaled_df["pxl_row_in_fullres"]

   
       # Use Otsu segmentation method for tissue region segmentation
    from skimage.color import rgb2gray
    from skimage.filters import threshold_otsu
    from skimage.morphology import closing, remove_small_objects, dilation, erosion, disk
    
    print("Starting tissue segmentation using Otsu method...")
    
    # 1. Convert to grayscale
    gray = rgb2gray(image_np)
    
    # 2. Otsu automatic threshold segmentation
    thresh = threshold_otsu(gray)
    binary = gray < thresh  # Note: H&E tissue areas are dark, invert
    print(f"Otsu threshold: {thresh:.3f}")
    
    # 3. Morphological processing (remove small fragments, fill holes, etc., increase parameters to merge nearby regions)
    # Use larger structural elements to connect nearby regions
    cleaned = closing(binary, disk(15))  # Increase disk radius from 7 to 15
    cleaned = remove_small_objects(cleaned, min_size=2000)  # Increase minimum area from 500 to 2000
    
    # Additional morphological operations: dilate then erode to further connect nearby regions
    cleaned = dilation(cleaned, disk(10))  # Dilation operation to connect nearby regions
    cleaned = erosion(cleaned, disk(8))    # Erosion operation to restore general shape
    
    # 4. Connected component labeling
    labels = measure.label(cleaned)
    props = measure.regionprops(labels)
    print(f"Detected {len(props)} connected components")
    
    # 5. Assign region labels to each spot
    spot_labels = []
    for _, spot in pts_rotated_df.iterrows():
        x = int(spot['pxl_col_in_fullres'])
        y = int(spot['pxl_row_in_fullres'])
        if x < labels.shape[1] and y < labels.shape[0]:
            spot_labels.append(labels[y, x])
        else:
            spot_labels.append(0)
    
    pts_rotated_df['tissue_region'] = spot_labels
    
    # Step 1: Set all in_tissue=0 spots as background (-1)
    pts_rotated_df.loc[pts_rotated_df['in_tissue'] == 0, 'tissue_region'] = -1
    
    # Step 2: Count spots in remaining regions (only count in_tissue=1 spots)
    in_tissue_spots = pts_rotated_df[pts_rotated_df['in_tissue'] == 1]
    region_counts_initial = in_tissue_spots['tissue_region'].value_counts()
    print(f"Initial region distribution (in_tissue=1 only):")
    for region_id in sorted(region_counts_initial.index):
        print(f"  Region {region_id}: {region_counts_initial[region_id]} spots")
    
    # Step 3: Find regions with <30 spots (including background region 0)
    small_regions = []
    for region_id, count in region_counts_initial.items():
        if region_id == 0 or count < 30:  # Include background region 0 and small regions
            small_regions.append(region_id)
    
    # Step 4: Reassign small regions as background (-1)
    for region_id in small_regions:
        pts_rotated_df.loc[pts_rotated_df['tissue_region'] == region_id, 'tissue_region'] = -1
    
    # Step 5: Renumber main regions as 0,1,2,3...
    remaining_regions = sorted([r for r in region_counts_initial.index if r not in small_regions])
    region_mapping = {}
    for i, old_region in enumerate(remaining_regions):
        region_mapping[old_region] = i
    
    # Apply new numbering
    for old_region, new_region in region_mapping.items():
        pts_rotated_df.loc[pts_rotated_df['tissue_region'] == old_region, 'tissue_region'] = new_region
    
    print(f"\nProcessing results:")
    print(f"1. All in_tissue=0 spots → region=-1")
    print(f"2. {len(small_regions)} small regions (<30 spots) → region=-1")
    print(f"   Small region list: {small_regions}")
    print(f"3. Main regions renumbered: {region_mapping}")
    
    # Create labeled image for visualization
    labeled_img = labels
    
    # Create region_masks dictionary for compatibility
    region_masks = {}
    unique_regions = sorted([r for r in pts_rotated_df['tissue_region'].unique() if r >= 0])
    for region_id in unique_regions:
        region_spots = pts_rotated_df[pts_rotated_df['tissue_region'] == region_id]
        if len(region_spots) > 0:
            mask = np.zeros(labels.shape, dtype=bool)
            for _, spot in region_spots.iterrows():
                x = int(spot['pxl_col_in_fullres'])
                y = int(spot['pxl_row_in_fullres'])
                if 0 <= x < labels.shape[1] and 0 <= y < labels.shape[0]:
                    mask[y, x] = True
            region_masks[region_id] = mask
    
    # Visualization: using Otsu method results
    plt.figure(figsize=(15, 12))
    
    # Get all unique regions
    unique_regions = sorted(pts_rotated_df['tissue_region'].unique())
    
    # Define color scheme: -1 as blue, 0,1,2,3 use contrasting colors
    region_color_map = {
        -1: 'blue',       # Background
        0: 'red',         # Tissue region 0
        1: 'green',       # Tissue region 1  
        2: 'orange',      # Tissue region 2
        3: 'purple',      # Tissue region 3
        4: 'cyan',        # If more regions exist
        5: 'magenta',     # If more regions exist
        6: 'yellow',      # If more regions exist
        7: 'brown'        # If more regions exist
    }
    
    # Left plot: show segmented connected components (colored)
    plt.subplot(1, 2, 1)
    plt.imshow(labeled_img, cmap='nipy_spectral')
    plt.title(f"{sample_id} - Otsu Connected Components")
    plt.axis('off')
    
    # Right plot: original image + spots colored by region (show all region=-1 as blue)
    plt.subplot(1, 2, 2)
    plt.imshow(image_np)
    
    # Plot all spots
    legend_elements = []
    from matplotlib.patches import Patch
    
    for region_id in unique_regions:
        region_spots = pts_rotated_df[pts_rotated_df['tissue_region'] == region_id]
        if len(region_spots) > 0:
            color = region_color_map.get(region_id, 'gray')  # Default gray
            
            if region_id == -1:
                label = f'Background ({len(region_spots)} spots)'
                # Show all background spots
                plt.scatter(
                    region_spots['pxl_col_in_fullres'], 
                    region_spots['pxl_row_in_fullres'],
                    s=8,
                    c=color,
                    marker='x',
                    alpha=0.7,
                    label=label
                )
            else:
                # Only show spot count for in_tissue=1 tissue regions
                in_tissue_count = len(region_spots[region_spots['in_tissue'] == 1])
                label = f'Region {region_id} ({in_tissue_count} spots)'
                # Only plot in_tissue=1 spots
                tissue_spots = region_spots[region_spots['in_tissue'] == 1]
                
                if len(tissue_spots) > 0:  # Ensure there are spots to plot
                    plt.scatter(
                        tissue_spots['pxl_col_in_fullres'], 
                        tissue_spots['pxl_row_in_fullres'],
                        s=8,
                        c=color,
                        marker='x',
                        alpha=0.7,
                        label=label
                    )
            
            # Add legend elements
            legend_elements.append(Patch(facecolor=color, edgecolor='k', label=label))
    
    plt.title(f"{sample_id} - Spots by Otsu Region")
    plt.axis('off')
    if legend_elements:
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Statistical information
    region_counts = pts_rotated_df['tissue_region'].value_counts().sort_index()
    print(f"{sample_id} - Spot count per region:")
    for region_id in region_counts.index:
        if region_id == -1:
            print(f"Background: {region_counts[region_id]} spots")
        else:
            tissue_count = len(pts_rotated_df[(pts_rotated_df['tissue_region'] == region_id) & (pts_rotated_df['in_tissue'] == 1)])
            print(f"Region {region_id}: {tissue_count} spots (in tissue)")
    
    # Additional statistics
    total_background = len(pts_rotated_df[pts_rotated_df['tissue_region'] == -1])
    total_tissue = len(pts_rotated_df[pts_rotated_df['tissue_region'] >= 0])
    in_tissue_0_count = len(pts_rotated_df[pts_rotated_df['in_tissue'] == 0])
    
    print(f"\nComplete statistics:")
    print(f"Total spots: {len(pts_rotated_df)}")
    print(f"Background region (region=-1): {total_background} spots")
    print(f"  ├─ in_tissue=0: {in_tissue_0_count} spots")
    print(f"  └─ in_tissue=1 but belongs to small regions: {total_background - in_tissue_0_count} spots")
    print(f"Tissue regions (region>=0): {total_tissue} spots (all are in_tissue=1)")


    # save the dataframe with region labels
    pts_rotated_df.to_csv(f"data/Scaled_Spatial_coordinates/{sample_id}_spots_coordinates_scaled.csv")
    print(f"{sample_id} - the dataframe with region labels has been saved")

    return pts_rotated_df


for i in range(len(extracted_parts_unique_rotated)):
    matching_spots_with_WSI_rotated(pts_df_path_list_rotated[i], image_path_list_rotated[i], extracted_parts_unique_rotated[i])




##### Calculate the median distance between adjacent spots ###########


def calculate_median_distance(sample_id):

    pts_scaled_df = pd.read_csv(f"data/Scaled_Spatial_coordinates/{sample_id}_spots_coordinates_scaled.csv", index_col = 0)
    
    dy_list = []
    for i in range(pts_scaled_df["array_row"].max()):
        dy = pts_scaled_df[pts_scaled_df["array_row"] == i]["pxl_row_in_fullres"].reset_index(drop = True) - pts_scaled_df[pts_scaled_df["array_row"] == i+1]["pxl_row_in_fullres"].reset_index(drop = True) 
        dy_list += dy.tolist()

    dx_list = []
    for i in range(pts_scaled_df["array_col"].max()-1):
        dx = pts_scaled_df[pts_scaled_df["array_col"] == i]["pxl_col_in_fullres"].reset_index(drop = True) - pts_scaled_df[pts_scaled_df["array_col"] == i+2]["pxl_col_in_fullres"].reset_index(drop = True)
        dx_list += dx.tolist()
    
    median_diff_x = np.median(dx_list)
    median_diff_y = np.median(dy_list)

    return median_diff_x, median_diff_y






def create_patches(sample_id):
    image_path = f"data/H&E/{sample_id}_cropped_scaled.tif"
    Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
    image = Image.open(image_path)
    image_np = np.array(image)
    aligned_coords = pd.read_csv(f"data/Scaled_Spatial_coordinates/{sample_id}_spots_coordinates_scaled.csv", index_col = 0)

    median_diff_x, median_diff_y = calculate_median_distance(sample_id)
    
    radius_x = np.abs(median_diff_x/2).round(2)
    radius_y = np.abs(median_diff_y/2).round(2)
    radius = min(radius_x, radius_y)

    # Filter spots where in_tissue == 1 AND tissue_region != -1
    aligned_coords = aligned_coords[(aligned_coords["in_tissue"] == 1) & (aligned_coords["tissue_region"] != -1)]
    
    # Count spots per region and keep only regions with at least 30 spots
    region_counts = aligned_coords["tissue_region"].value_counts()
    valid_regions = region_counts[region_counts >= 30].index.tolist()
    
    # Skip if no valid regions
    if not valid_regions:
        print(f"No valid regions with ≥30 spots found for {sample_id}")
        return
    
    print(f"Processing {len(valid_regions)} valid regions for {sample_id}: {valid_regions}")
    
    # Process each valid region separately
    for region_id in valid_regions:
        # Filter for current region
        region_coords = aligned_coords[aligned_coords["tissue_region"] == region_id].reset_index(drop=True)
        
        # Create region-specific directory
        region_dir = f"data/Colorectal_Cancer_HE_patches/{sample_id}_region{region_id}"
        os.makedirs(region_dir, exist_ok=True)
        print(f"Created folder '{sample_id}_region{region_id}'")
        
        # Save region-specific coordinates file
        region_coords.to_csv(f"data/Scaled_Spatial_coordinates/{sample_id}_region{region_id}_spots_coordinates_scaled.csv")
        
        # Create patches for this region
        for index, row in region_coords.iterrows():
            x_center = row['pxl_col_in_fullres']
            y_center = row['pxl_row_in_fullres']

            # Calculate patch boundaries
            x_start = x_center - radius
            x_end = x_center + radius
            y_start = y_center - radius
            y_end = y_center + radius

            # Check if boundaries are within image range
            if x_start >= 0 and y_start >= 0 and x_end <= image_np.shape[1] and y_end <= image_np.shape[0]:
                cropped_image = image_np[int(y_start):int(y_end), int(x_start):int(x_end)]
                cropped_image_pil = Image.fromarray(cropped_image)
                output_path = os.path.join(region_dir, f"{row['barcode']}_{sample_id}_region{region_id}.tif")
                cropped_image_pil.save(output_path)
        
        print(f"Created {len(region_coords)} patches for {sample_id}_region{region_id}")


def create_patches_rotated(sample_id):
    image_path = f"data/H&E_rotated/{sample_id}_cropped_scaled.tif"
    Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
    image = Image.open(image_path)
    image_np = np.array(image)
    aligned_coords = pd.read_csv(f"data/Scaled_Spatial_coordinates/{sample_id}_spots_coordinates_scaled.csv", index_col = 0)

    median_diff_x, median_diff_y = calculate_median_distance(sample_id)
    
    radius_x = np.abs(median_diff_x/2).round(2)
    radius_y = np.abs(median_diff_y/2).round(2)
    radius = min(radius_x, radius_y)

    # Filter spots where in_tissue == 1 AND tissue_region != -1
    aligned_coords = aligned_coords[(aligned_coords["in_tissue"] == 1) & (aligned_coords["tissue_region"] != -1)]
    
    # Count spots per region and keep only regions with at least 30 spots
    region_counts = aligned_coords["tissue_region"].value_counts()
    valid_regions = region_counts[region_counts >= 30].index.tolist()
    
    # Skip if no valid regions
    if not valid_regions:
        print(f"No valid regions with ≥30 spots found for {sample_id}")
        return
    
    print(f"Processing {len(valid_regions)} valid regions for {sample_id}: {valid_regions}")
    
    # Process each valid region separately
    for region_id in valid_regions:
        # Filter for current region
        region_coords = aligned_coords[aligned_coords["tissue_region"] == region_id].reset_index(drop=True)
        
        # Create region-specific directory
        region_dir = f"data/Colorectal_Cancer_HE_patches/{sample_id}_region{region_id}"
        os.makedirs(region_dir, exist_ok=True)
        print(f"Created folder '{sample_id}_region{region_id}'")
        
        # Save region-specific coordinates file
        region_coords.to_csv(f"data/Scaled_Spatial_coordinates/{sample_id}_region{region_id}_spots_coordinates_scaled.csv")
        
        # Create patches for this region
        for index, row in region_coords.iterrows():
            x_center = row['pxl_col_in_fullres']
            y_center = row['pxl_row_in_fullres']

            # Calculate patch boundaries
            x_start = x_center - radius
            x_end = x_center + radius
            y_start = y_center - radius
            y_end = y_center + radius

            # Check if boundaries are within image range
            if x_start >= 0 and y_start >= 0 and x_end <= image_np.shape[1] and y_end <= image_np.shape[0]:
                cropped_image = image_np[int(y_start):int(y_end), int(x_start):int(x_end)]
                cropped_image_pil = Image.fromarray(cropped_image)
                output_path = os.path.join(region_dir, f"{row['barcode']}_{sample_id}_region{region_id}.tif")
                cropped_image_pil.save(output_path)
        
        print(f"Created {len(region_coords)} patches for {sample_id}_region{region_id}")


for sample_id in extracted_parts_unique:
    create_patches(sample_id)
    print(f"patches for {sample_id} created.")



for sample_id in extracted_parts_unique_rotated:
    create_patches_rotated(sample_id)
    print(f"patches for {sample_id} created.")



###### Create data for CARD deconvolution #########



All_samples_list = extracted_parts_unique + extracted_parts_unique_rotated

for sample_id in All_samples_list:
    
    spatial_files_path = f"data/Scaled_Spatial_coordinates/{sample_id}_region*_spots_coordinates_scaled.csv"
    spatial_files_list = glob.glob(spatial_files_path)

    for spatial_file in spatial_files_list:
        region_id = "_".join(os.path.basename(spatial_file).split("_")[0:4])
        image_files_path = f"data/Colorectal_Cancer_HE_patches/{region_id}"
        
        if not os.path.exists(image_files_path):
            print(f"Skipping {region_id} - folder does not exist: {image_files_path}")
            continue
            
        image_files_list = os.listdir(image_files_path)
        image_files_stripped_list = [file.split(".")[0] for file in image_files_list]

        import cv2
        import numpy as np
        import os
        from tqdm import tqdm

        white_threshold = 220  
        white_ratio_cutoff = 0.4  

        for fname in tqdm(image_files_list):
            fpath = os.path.join(image_files_path, fname)
            
            img = cv2.imread(fpath)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            white_mask = gray > white_threshold
            white_ratio = np.sum(white_mask) / white_mask.size

            if white_ratio > white_ratio_cutoff:
                os.remove(fpath)
        
        remaining_files = [f for f in os.listdir(image_files_path) if f.endswith('.tif')]
        remaining_count = len(remaining_files)
        print(f"\nRemaining files in folder: {remaining_count}")

        image_files_stripped_list = [file.split(".")[0] for file in remaining_files]
        image_files_stripped_list = [file.split("_")[0] for file in image_files_stripped_list]
        spatial_df = pd.read_csv(spatial_file, index_col = 0)
        spatial_df = spatial_df[["barcode", "pxl_col_in_fullres", "pxl_row_in_fullres"]]
        spatial_df.columns = ["barcode", "x", "y"]
        spatial_df.index = spatial_df["barcode"]
        spatial_df = spatial_df[["x", "y"]]
        expression_df = pd.read_csv(f"data/ST_data_for_deconv/{sample_id}_expression.csv", index_col = 0)
        intersection_index = spatial_df.index.intersection(expression_df.index).intersection(image_files_stripped_list)

        if len(intersection_index) < 30:
            shutil.rmtree(image_files_path)
            print(f"Folder {image_files_path} has been deleted because it contains only {len(intersection_index)} valid files (less than 30)")
        else:
            spatial_df = spatial_df.loc[intersection_index]
            expression_df = expression_df.loc[intersection_index]
            spatial_df.to_csv(f"data/CARD_Need_Files/{region_id}_spatial.csv")
            expression_df.to_csv(f"data/CARD_Need_Files/{region_id}_expression.csv")
            print(f"Created the files for {region_id}")
    
    
    
    


### Create training patches or tiles for cancer cell proportion prediction model #########

Cancer_good_sample_list = ["6723_KL_1", "6723_KL_2",
                           "7003_AS_3", "7003_AS_4", "7003_AS_5", "7003_AS_6", "7003_AS_7", "7003_AS_8",
                           "7319_AS_2",
                           "7794_AS_1", "7794_AS_2",
                           "8270_AS_6", "8270_AS_7", "8270_AS_8", "8270_AS_9", "8270_AS_10",
                           "8578_AS_1",
                           "8899_AS_1", "8899_AS_3", "8899_AS_4", "8899_AS_6", "8899_AS_7"]

Cancer_training_sample_list = Cancer_good_sample_list

def process_cancer_training_tiles(Cancer_training_sample_list):
 
    target_folder = "data/Colorectal_Cancer_HE_patches/Training_patches/Cancer"
    os.makedirs(target_folder, exist_ok=True)
    print(f"Created the target folder: {target_folder}")

    Entire_cancer_cell_proportion_df = pd.DataFrame()
    for sample_name in Cancer_training_sample_list:
        cancer_cell_proportion_df = pd.read_csv(f"data/CARD_results_celltype_proportion/{sample_name}_celltype_proportion_modified.csv", index_col = 0)
        cancer_cell_proportion_df.index = cancer_cell_proportion_df.index.str.replace(".", "-")
        cancer_cell_proportion_df.index = cancer_cell_proportion_df.index + "_" + sample_name
        cancer_cell_proportion = cancer_cell_proportion_df.iloc[:,0:9].sum(axis = 1)

        scaled_spatial_coordinates_df = pd.read_csv(f"data/Scaled_Spatial_coordinates/{sample_name}_spots_coordinates_scaled.csv", index_col = 0)
        scaled_spatial_coordinates_df["barcode"] = scaled_spatial_coordinates_df["barcode"] + "_" + sample_name
        scaled_spatial_coordinates_df = scaled_spatial_coordinates_df[scaled_spatial_coordinates_df["in_tissue"] == 1]
        scaled_spatial_coordinates_df = scaled_spatial_coordinates_df.reset_index(drop = True)
        scaled_spatial_coordinates_df = scaled_spatial_coordinates_df[["barcode", "pxl_col_in_fullres", "pxl_row_in_fullres"]]
        scaled_spatial_coordinates_df.columns = ["barcode", "x", "y"]
        scaled_spatial_coordinates_df.index = scaled_spatial_coordinates_df["barcode"]
        scaled_spatial_coordinates_df = scaled_spatial_coordinates_df[["x", "y"]]
        
        intersection_index = cancer_cell_proportion.index.intersection(scaled_spatial_coordinates_df.index)
        cancer_cell_proportion = cancer_cell_proportion.loc[intersection_index]
        scaled_spatial_coordinates_df = scaled_spatial_coordinates_df.loc[intersection_index]

        combined_df = pd.concat([cancer_cell_proportion, scaled_spatial_coordinates_df], axis = 1)
        combined_df.columns = ["cancer_cell_proportion", "x", "y"]
        Entire_cancer_cell_proportion_df = pd.concat([Entire_cancer_cell_proportion_df, combined_df], axis = 0)
        






    for i in range(len(Cancer_training_sample_list)):
        sample_name = Cancer_training_sample_list[i]

        source_folder = "data/Colorectal_Cancer_HE_patches/"+sample_name
        # get all .tif files
        all_files = [f for f in os.listdir(source_folder) if f.endswith('.tif')] 
        files_name = [f.replace('.tif', '') for f in all_files]
        matched_files_name = list(set(Entire_cancer_cell_proportion_df.index).intersection(set(files_name)))

        # Copy the files
        for file in matched_files_name:
            original_path = os.path.join(source_folder, file+".tif")
            target_path = os.path.join(target_folder, file+".tif")
            shutil.copy2(original_path, target_path)

    white_threshold = 220                  # Consider the pixel as "white" if its gray value is larger than this threshold
    white_ratio_cutoff = 0.5               # Remove the patch if the white ratio is larger than this cutoff

    # Iterate through all .tif files
    all_files_in_target_folder = [f for f in os.listdir(target_folder) if f.endswith('.tif')]
    for fname in tqdm(all_files_in_target_folder):
        fpath = os.path.join(target_folder, fname)

        # === 1. Read the image (BGR)
        img = cv2.imread(fpath)

        # === 2. Calculate the white ratio
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        white_mask = gray > white_threshold
        white_ratio = np.sum(white_mask) / white_mask.size

        if white_ratio > white_ratio_cutoff:
            os.remove(fpath)
            continue

        # No further grayscale conversion or normalization processing, directly keep the original color image

    all_valid_files_in_target_folder = [f for f in os.listdir(target_folder) if f.endswith('.tif')]
    valid_files_name = [f.replace('.tif', '') for f in all_valid_files_in_target_folder]
    matched_valid_files_name = list(set(Entire_cancer_cell_proportion_df.index).intersection(set(valid_files_name)))
    Entire_cancer_cell_proportion_df = Entire_cancer_cell_proportion_df.loc[matched_valid_files_name]

    Entire_cancer_cell_proportion_df.to_csv(f"data/Colorectal_Cancer_HE_patches/Training_cell_type_proportion/Cancer_training_proportion.csv", index=True)



######################### Create training patches for fibroblast cell proportion prediction model #########

Fibroblast_good_sample_list = ["6723_KL_1", "6723_KL_2", "6723_KL_4",
                           "7003_AS_3", "7003_AS_4", "7003_AS_5", "7003_AS_6", "7003_AS_7", "7003_AS_8",
                           "7794_AS_1", "7794_AS_2", "7794_AS_3",
                           "8270_AS_6", "8270_AS_7", "8270_AS_8", "8270_AS_9", "8270_AS_10", "8270_AS_11", "8270_AS_12",
                           "8578_AS_1", "8578_AS_2", "8578_AS_3",
                           "8899_AS_1", "8899_AS_3", "8899_AS_4", "8899_AS_5", "8899_AS_6", "8899_AS_7"]

Fibroblast_training_sample_list = Fibroblast_good_sample_list

def process_other_cell_type_training_tiles(good_sample_list = Fibroblast_good_sample_list, cell_type_name_abbreviation = "FIB", cell_type_name = "Fibroblast"):
 
    target_folder = f"data/Colorectal_Cancer_HE_patches/Training_patches/{cell_type_name}"
    os.makedirs(target_folder, exist_ok=True)
    print(f"Created the target folder: {target_folder}")

    Entire_cell_proportion_df = pd.DataFrame()

    for sample_name in good_sample_list:
        cell_proportion_df = pd.read_csv(f"data/CARD_results_celltype_proportion/{sample_name}_celltype_proportion_modified.csv", index_col = 0)
        cell_proportion_df.index = cell_proportion_df.index.str.replace(".", "-")
        cell_proportion_df.index = cell_proportion_df.index + "_" + sample_name
        cell_proportion = cell_proportion_df[cell_type_name_abbreviation]

        scaled_spatial_coordinates_df = pd.read_csv(f"data/Scaled_Spatial_coordinates/{sample_name}_spots_coordinates_scaled.csv", index_col = 0)
        scaled_spatial_coordinates_df["barcode"] = scaled_spatial_coordinates_df["barcode"] + "_" + sample_name
        scaled_spatial_coordinates_df = scaled_spatial_coordinates_df[scaled_spatial_coordinates_df["in_tissue"] == 1]
        scaled_spatial_coordinates_df = scaled_spatial_coordinates_df.reset_index(drop = True)
        scaled_spatial_coordinates_df = scaled_spatial_coordinates_df[["barcode", "pxl_col_in_fullres", "pxl_row_in_fullres"]]
        scaled_spatial_coordinates_df.columns = ["barcode", "x", "y"]
        scaled_spatial_coordinates_df.index = scaled_spatial_coordinates_df["barcode"]
        scaled_spatial_coordinates_df = scaled_spatial_coordinates_df[["x", "y"]]
        
        intersection_index = cell_proportion.index.intersection(scaled_spatial_coordinates_df.index)
        cell_proportion = cell_proportion.loc[intersection_index]
        scaled_spatial_coordinates_df = scaled_spatial_coordinates_df.loc[intersection_index]

        combined_df = pd.concat([cell_proportion, scaled_spatial_coordinates_df], axis = 1)
        combined_df.columns = ["cell_proportion", "x", "y"]
        Entire_cell_proportion_df = pd.concat([Entire_cell_proportion_df, combined_df], axis = 0)
        


    for i in range(len(good_sample_list)):
        sample_name = good_sample_list[i]

        source_folder = f"data/Colorectal_Cancer_HE_patches/{sample_name}"
        # get all .tif files
        all_files = [f for f in os.listdir(source_folder) if f.endswith('.tif')] 
        files_name = [f.replace('.tif', '') for f in all_files]
        matched_files_name = list(set(Entire_cell_proportion_df.index).intersection(set(files_name)))

        # Copy the files
        for file in matched_files_name:
            original_path = os.path.join(source_folder, file+".tif")
            target_path = os.path.join(target_folder, file+".tif")
            shutil.copy2(original_path, target_path)

    white_threshold = 220                  # Consider the pixel as "white" if its gray value is larger than this threshold
    white_ratio_cutoff = 0.5               # Remove the patch if the white ratio is larger than this cutoff

    # Iterate through all .tif files
    all_files_in_target_folder = [f for f in os.listdir(target_folder) if f.endswith('.tif')]
    for fname in tqdm(all_files_in_target_folder):
        fpath = os.path.join(target_folder, fname)

        # === 1. Read the image (BGR)
        img = cv2.imread(fpath)

        # === 2. Calculate the white ratio
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        white_mask = gray > white_threshold
        white_ratio = np.sum(white_mask) / white_mask.size

        if white_ratio > white_ratio_cutoff:
            os.remove(fpath)
            continue

        # No further grayscale conversion or normalization processing, directly keep the original color image

    all_valid_files_in_target_folder = [f for f in os.listdir(target_folder) if f.endswith('.tif')]
    valid_files_name = [f.replace('.tif', '') for f in all_valid_files_in_target_folder]
    matched_valid_files_name = list(set(Entire_cell_proportion_df.index).intersection(set(valid_files_name)))
    Entire_cell_proportion_df = Entire_cell_proportion_df.loc[matched_valid_files_name]

    Entire_cell_proportion_df.to_csv(f"data/Colorectal_Cancer_HE_patches/Training_cell_type_proportion/{cell_type_name}_training_proportion.csv", index=True)


process_other_cell_type_training_tiles(good_sample_list = Fibroblast_good_sample_list, cell_type_name_abbreviation = "FIB", cell_type_name = "Fibroblast")

