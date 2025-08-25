import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from PIL import Image
import cv2
from skimage import measure
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.morphology import remove_small_objects, closing, disk, dilation, erosion
import scanpy as sc
import shutil
import glob






### find the HE files ###########

HE_files = []
for root, dirs, files in os.walk("/Users/scui2/Desktop/FredHutch_Colorectal/H&E_Cropped"):
    for file in files:
        if file.endswith(".tif"):  
            HE_files.append(os.path.join(root, file))

HE_files_sorted = sorted(HE_files)


Image_ID_list = [file.split("/")[-1].split(".")[0] for file in HE_files_sorted]

image_path_list = []
pts_df_path_list = []
for image_id in Image_ID_list:
    image_path_list.append(f"/Users/scui2/Desktop/FredHutch_Colorectal/H&E_Cropped/{image_id}.tif")
    first_part = image_id.split("_")[0]
    pts_df_path_list.append(f"/Users/scui2/Desktop/FredHutch_Colorectal/ST_data/{first_part}/spatial/tissue_positions.csv")



### Matching the spots with the HE images ###########





def matching_spots_with_WSI(pts_df_path, image_path, image_id):
    y_flip_sample_list = ["SH-16-04266-A1", "SH-16-07447", "SH-17-06079-B1", "SH-17-06138-A1", "SU-15-19531-A1", "SU-15-27301-B1", "SU-17-09232-A1", "SU-17-25332-D1"]
    x_flip_sample_list = ["SU-15-18753-A1", "SU-16-02468-B1", "SU-17-05145-A1", "SU-17-05594-A1", "SU-17-07002-A1", "SU-17-18554-B1", "SU-17-23223-A1", "SU-17-25385-B1", "SU-17-30257-A1"]
    no_flip_sample_list = ["SU-17-14212-A1"]
    both_flip_sample_list = ["SU-17-15020-C1"]
    right_move_3_spots_list = ["SH-17-06079-B1"]
    right_move_1_spot_list = ["SU-17-18554-B1"]
    down_move_1_spot_list = ["SU-17-05594-A1"]
    left_move_1_spots_list = ["SU-17-07002-A1"]
    left_move_3_spots_list = ["SU-17-15020-C1"]
    sample_id = image_id.split("_")[0]
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

    # Correction: correctly get image dimensions
    WSI_height, WSI_width = image_np.shape[:2]
    print(f"Image dimensions: width={WSI_width}, height={WSI_height}")
    print(f"ST coordinate range: width={pts_width}, height={pts_height}")
    
    # Calculate scaling factor
    scale_factor_col = WSI_width / pts_width
    scale_factor_row = WSI_height / pts_height
    print(f"Scaling factor: X direction={scale_factor_col:.3f}, Y direction={scale_factor_row:.3f}")


    pts_moved_df = pts_df.copy()
    pts_moved_df["pxl_col_in_fullres"] = pts_df["pxl_col_in_fullres"] - pts_df["pxl_col_in_fullres"].min()
    pts_moved_df["pxl_row_in_fullres"] = pts_df["pxl_row_in_fullres"] - pts_df["pxl_row_in_fullres"].min()

    pts_scaled_df = pts_moved_df.copy()
    pts_scaled_df["pxl_col_in_fullres"] = pts_moved_df["pxl_col_in_fullres"] * scale_factor_col
    pts_scaled_df["pxl_row_in_fullres"] = pts_moved_df["pxl_row_in_fullres"] * scale_factor_row
    
    if sample_id in y_flip_sample_list:
    # Flip Y coordinates
        pts_scaled_df["pxl_row_in_fullres"] = WSI_height - pts_scaled_df["pxl_row_in_fullres"]

    if sample_id in x_flip_sample_list:
        pts_scaled_df["pxl_col_in_fullres"] = WSI_width - pts_scaled_df["pxl_col_in_fullres"]

    if sample_id in both_flip_sample_list:
        pts_scaled_df["pxl_row_in_fullres"] = WSI_height - pts_scaled_df["pxl_row_in_fullres"]
        pts_scaled_df["pxl_col_in_fullres"] = WSI_width - pts_scaled_df["pxl_col_in_fullres"]

    if sample_id in no_flip_sample_list:
        None

    # Handle samples that need to move right by 3 spots distance
    if sample_id in right_move_3_spots_list:
        # Calculate horizontal distance between spots
        # Select consecutive spots in the same row to calculate distance
        same_row_spots = pts_scaled_df[pts_scaled_df["array_row"] == pts_scaled_df["array_row"].iloc[0]].sort_values("array_col")
        if len(same_row_spots) >= 2:
            # Calculate average distance between adjacent spots
            col_diffs = same_row_spots["pxl_col_in_fullres"].diff().dropna()
            avg_spot_distance = col_diffs.mean()
            
            # Move right by 3 spots distance
            move_distance = 3 * avg_spot_distance
            pts_scaled_df["pxl_col_in_fullres"] = pts_scaled_df["pxl_col_in_fullres"] + move_distance
            print(f"Sample {sample_id} moved right by {move_distance:.2f} pixels (3 spots distance)")

    # Handle samples that need to move down by 1 spot distance
    if sample_id in down_move_1_spot_list:
        # Calculate vertical distance between spots
        # Select consecutive spots in the same column to calculate distance
        same_col_spots = pts_scaled_df[pts_scaled_df["array_col"] == pts_scaled_df["array_col"].iloc[0]].sort_values("array_row")
        if len(same_col_spots) >= 2:
            # Calculate average vertical distance between adjacent spots
            row_diffs = same_col_spots["pxl_row_in_fullres"].diff().dropna()
            avg_spot_distance_y = row_diffs.mean()
            
            # Move down by 1 spot distance
            move_distance_y = 1 * avg_spot_distance_y
            pts_scaled_df["pxl_row_in_fullres"] = pts_scaled_df["pxl_row_in_fullres"] + move_distance_y
            print(f"Sample {sample_id} moved down by {move_distance_y:.2f} pixels (1 spot distance)")

    # Handle samples that need to move left by 1 spot distance
    if sample_id in left_move_1_spots_list:
        # Calculate horizontal distance between spots
        # Select consecutive spots in the same row to calculate distance
        same_row_spots = pts_scaled_df[pts_scaled_df["array_row"] == pts_scaled_df["array_row"].iloc[0]].sort_values("array_col")
        if len(same_row_spots) >= 2:
            # Calculate average distance between adjacent spots
            col_diffs = same_row_spots["pxl_col_in_fullres"].diff().dropna()
            avg_spot_distance = col_diffs.mean()
            
            # Move left by 1 spot distance (subtract distance)
            move_distance = 1 * avg_spot_distance
            pts_scaled_df["pxl_col_in_fullres"] = pts_scaled_df["pxl_col_in_fullres"] - move_distance
            print(f"Sample {sample_id} moved left by {move_distance:.2f} pixels (1 spot distance)")

    # Handle samples that need to move left by 3 spots distance
    if sample_id in left_move_3_spots_list:
        # Calculate horizontal distance between spots
        # Select consecutive spots in the same row to calculate distance
        same_row_spots = pts_scaled_df[pts_scaled_df["array_row"] == pts_scaled_df["array_row"].iloc[0]].sort_values("array_col")
        if len(same_row_spots) >= 2:
            # Calculate average distance between adjacent spots
            col_diffs = same_row_spots["pxl_col_in_fullres"].diff().dropna()
            avg_spot_distance = col_diffs.mean()
            
            # Move left by 3 spots distance (subtract distance)
            move_distance = 3 * avg_spot_distance
            pts_scaled_df["pxl_col_in_fullres"] = pts_scaled_df["pxl_col_in_fullres"] - move_distance
            print(f"Sample {sample_id} moved left by {move_distance:.2f} pixels (3 spots distance)")


    # Handle samples that need to move right by 1 spot distance
    if sample_id in right_move_1_spot_list:
        # Calculate horizontal distance between spots
        # Select consecutive spots in the same row to calculate distance
        same_row_spots = pts_scaled_df[pts_scaled_df["array_row"] == pts_scaled_df["array_row"].iloc[0]].sort_values("array_col")
        if len(same_row_spots) >= 2:
            # Calculate average distance between adjacent spots
            col_diffs = same_row_spots["pxl_col_in_fullres"].diff().dropna()
            avg_spot_distance = col_diffs.mean()
            
            # Move right by 1 spot distance
            move_distance = 1 * avg_spot_distance
            pts_scaled_df["pxl_col_in_fullres"] = pts_scaled_df["pxl_col_in_fullres"] + move_distance
            print(f"Sample {sample_id} moved right by {move_distance:.2f} pixels (1 spot distance)")

    plt.figure(figsize=(15, 12))
    plt.imshow(image_np)
    plt.scatter(pts_scaled_df[pts_scaled_df["in_tissue"] == 1]["pxl_col_in_fullres"], pts_scaled_df[pts_scaled_df["in_tissue"] == 1]["pxl_row_in_fullres"], s=8, c='red', marker='x', alpha=0.7)
    plt.scatter(pts_scaled_df[pts_scaled_df["in_tissue"] == 0]["pxl_col_in_fullres"], pts_scaled_df[pts_scaled_df["in_tissue"] == 0]["pxl_row_in_fullres"], s=8, c='blue', marker='x', alpha=0.7)
    plt.title(f"{sample_id} - Spots on WSI")
    plt.axis('off')
    plt.savefig(f"/Users/scui2/Desktop/FredHutch_Colorectal/Visual/{sample_id}_spots_on_WSI.png", dpi=600, bbox_inches='tight')
    plt.close()

        # Use Otsu segmentation method for tissue region segmentation
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
    label_img = measure.label(cleaned)
    props = measure.regionprops(label_img)
    print(f"Detected {len(props)} connected components")
    
    # 5. Assign region labels to each spot
    spot_labels = []
    for _, spot in pts_scaled_df.iterrows():
        x = int(spot['pxl_col_in_fullres'])
        y = int(spot['pxl_row_in_fullres'])
        if x < label_img.shape[1] and y < label_img.shape[0]:
            spot_labels.append(label_img[y, x])
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

    # Count spots in each region
    region_counts = pts_scaled_df['tissue_region'].value_counts().sort_index()
    print(f"Spot count per region:")
    for region_id in region_counts.index:
        print(f"Region {region_id}: {region_counts[region_id]} spots")

    # Visualization: only show in_tissue=1 spots, in_tissue=0 counted as background
    
    # Create visualization
    plt.figure(figsize=(15, 12))
    
    # Left plot: show segmented connected components (colored)
    plt.subplot(1, 2, 1)
    plt.imshow(label_img, cmap='nipy_spectral')
    plt.title(f"{sample_id} - Otsu Connected Components")
    plt.axis('off')
    
    # Right plot: original image + spots colored by region (show all region=-1 as blue)
    plt.subplot(1, 2, 2)
    plt.imshow(image_np)
    
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
    
    # Plot all spots
    for region_id in unique_regions:
        region_spots = pts_scaled_df[pts_scaled_df['tissue_region'] == region_id]
        if len(region_spots) > 0:
            color = region_color_map.get(region_id, 'gray')  # Default gray
            
            if region_id == -1:
                label = f'Background ({len(region_spots)} spots)'
            else:
                # Only show spot count for in_tissue=1 tissue regions
                in_tissue_count = len(region_spots[region_spots['in_tissue'] == 1])
                label = f'Region {region_id} ({in_tissue_count} spots)'
                # Only plot in_tissue=1 spots
                region_spots = region_spots[region_spots['in_tissue'] == 1]
                
            if len(region_spots) > 0:  # Ensure there are spots to plot
                plt.scatter(
                    region_spots['pxl_col_in_fullres'], 
                    region_spots['pxl_row_in_fullres'],
                    s=8,
                    c=color,
                    marker='x',
                    alpha=0.7,
                    label=label
                )
    
    plt.title(f"{sample_id} - Spots by Otsu Region")
    plt.axis('off')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"/Users/scui2/Desktop/FredHutch_Colorectal/Visual/{sample_id}_otsu_region.png", dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization statistics:")
    background_count = len(pts_scaled_df[pts_scaled_df['tissue_region'] == -1])
    tissue_regions = [r for r in unique_regions if r >= 0]
    tissue_spots_shown = sum([len(pts_scaled_df[(pts_scaled_df['tissue_region'] == r) & (pts_scaled_df['in_tissue'] == 1)]) for r in tissue_regions])
    
    print(f"Background spots (blue): {background_count} (includes all in_tissue=0 and small regions)")
    print(f"Tissue region spots: {tissue_spots_shown} (only shows in_tissue=1)")
    
    for region_id in unique_regions:
        if region_id == -1:
            print(f"  Background: {background_count} spots")
        else:
            count = len(pts_scaled_df[(pts_scaled_df['tissue_region'] == region_id) & (pts_scaled_df['in_tissue'] == 1)])
            print(f"  Region {region_id}: {count} spots")
    
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
    pts_scaled_df.to_csv(f"/Users/scui2/Desktop/FredHutch_Colorectal/Scaled_Spatial_Coordinates/{sample_id}_spots_coordinates_scaled.csv")
    print(f"{sample_id} - the dataframe with region labels has been saved")


    



    return pts_scaled_df



for i in range(len(pts_df_path_list)):
    matching_spots_with_WSI(pts_df_path_list[i], image_path_list[i], Image_ID_list[i])
    print(f"{Image_ID_list[i]} - the spots have been matched with the WSI")




###### Create patches ###########

def calculate_median_distance(image_id):
    sample_id = image_id.split("_")[0]
    pts_scaled_df = pd.read_csv(f"/Users/scui2/Desktop/FredHutch_Colorectal/Scaled_Spatial_Coordinates/{sample_id}_spots_coordinates_scaled.csv", index_col = 0)
    
    dy_list = []
    for i in range(pts_scaled_df["array_row"].max()):
        dy = pts_scaled_df[pts_scaled_df["array_row"] == i]["pxl_row_in_fullres"].reset_index(drop = True) - pts_scaled_df[pts_scaled_df["array_row"] == i+1]["pxl_row_in_fullres"].reset_index(drop = True) 
        dy_list += dy.tolist()

    dx_list = []
    for i in range(pts_scaled_df["array_col"].max()-1):
        dx = pts_scaled_df[pts_scaled_df["array_col"] == i+2]["pxl_col_in_fullres"].reset_index(drop = True) - pts_scaled_df[pts_scaled_df["array_col"] == i]["pxl_col_in_fullres"].reset_index(drop = True)
        dx_list += dx.tolist()
    
    median_diff_x = np.median(dx_list)
    median_diff_y = np.median(dy_list)

    return median_diff_x, median_diff_y






def create_patches(image_id):
    sample_id = image_id.split("_")[0]
    image_path = f"/Users/scui2/Desktop/FredHutch_Colorectal/H&E_Cropped/{image_id}.tif"
    Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
    image = Image.open(image_path)
    image_np = np.array(image)
    aligned_coords = pd.read_csv(f"/Users/scui2/Desktop/FredHutch_Colorectal/Scaled_Spatial_Coordinates/{sample_id}_spots_coordinates_scaled.csv", index_col = 0)

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
        region_dir = f"/Users/scui2/Desktop/FredHutch_Colorectal/HE_patches/{sample_id}_region{region_id}"
        os.makedirs(region_dir, exist_ok=True)
        print(f"Created folder '{sample_id}_region{region_id}'")
        
        # Save region-specific coordinates file
        region_coords.to_csv(f"/Users/scui2/Desktop/FredHutch_Colorectal/Scaled_Spatial_Coordinates/{sample_id}_region{region_id}_spots_coordinates_scaled.csv")
        
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



for image_id in Image_ID_list:
    create_patches(image_id)
    print(f"patches for {image_id} created.")





###### Create data for CARD deconvolution #########







for image_id in Image_ID_list:
    sample_id = image_id.split("_")[0]
    spatial_files_path = f"/Users/scui2/Desktop/FredHutch_Colorectal/Scaled_Spatial_Coordinates/{sample_id}_region*_spots_coordinates_scaled.csv"
    spatial_files_list = glob.glob(spatial_files_path)
    

    for spatial_file in spatial_files_list:
        region_id = "_".join(os.path.basename(spatial_file).split("_")[0:2])
        image_files_path = f"/Users/scui2/Desktop/FredHutch_Colorectal/HE_patches/{region_id}"
        
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
        expression_df = pd.read_csv(f"/Users/scui2/Desktop/FredHutch_Colorectal/ST_data/{sample_id}/{sample_id}_expression.csv", index_col = 0)
        intersection_index = spatial_df.index.intersection(expression_df.index).intersection(image_files_stripped_list)

        if len(intersection_index) < 30:
            shutil.rmtree(image_files_path)
            print(f"Folder {image_files_path} has been deleted because it contains only {len(intersection_index)} valid files (less than 30)")
        else:
            spatial_df = spatial_df.loc[intersection_index]
            expression_df = expression_df.loc[intersection_index]
            spatial_df.to_csv(f"/Users/scui2/Desktop/FredHutch_Colorectal/CARD_Need_Files/{region_id}_spatial.csv")
            expression_df.to_csv(f"/Users/scui2/Desktop/FredHutch_Colorectal/CARD_Need_Files/{region_id}_expression.csv")
            print(f"Created the files for {region_id}")
    
    