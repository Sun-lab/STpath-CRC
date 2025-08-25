"""
STPath-COAD: Gene Expression Prediction Analysis for Figure S10
==============================================================

This script predicts gene expression levels from histopathology features and
generates analysis for supplementary Figure S10.

Author: Saishi Cui
Date: Sept 2025

Purpose: Predict gene expression levels using STPath-COAD histopathology features,
validate predictions against actual expression data, and generate comprehensive
analysis for Figure S10 of the paper including correlation analysis and performance metrics.
"""

import pandas as pd
import glob
import os
from tqdm import tqdm
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
import xgboost as xgb


### STEP 1: Process each raw expression file with count normalization


# ===== PROCESSING CODY'S DATA =====
print("\nProcessing Cody's data...")
cody_data_dir = "data/CARD_Need_Files"
cody_expression_files = glob.glob(os.path.join(cody_data_dir, "*_expression.csv"))
print(f"Found {len(cody_expression_files)} Cody expression files")

cody_normalized_dfs = []
cody_common_genes = None

for i, file_path in enumerate(tqdm(cody_expression_files, desc="Processing Cody files")):
    # Extract sample identifier
    filename = os.path.basename(file_path)
    sample_id = filename.replace('_expression.csv', '')
    
    # Read raw expression data
    df = pd.read_csv(file_path, index_col=0)
    
    # Update row names: barcode -> barcode_sampleID
    df.index = df.index.astype(str) + f"_{sample_id}"
    
    # Count normalization: divide by library size (total counts per spot)
    library_sizes = df.sum(axis=1)
    # Remove spots with zero library size
    zero_lib_mask = library_sizes > 0
    df_filtered = df[zero_lib_mask]
    library_sizes_filtered = library_sizes[zero_lib_mask]
    
    # Normalize: count / library_size
    df_normalized = df_filtered.div(library_sizes_filtered, axis=0)
    
    cody_normalized_dfs.append(df_normalized)
    
    # Find intersection of genes across all Cody files
    if cody_common_genes is None:
        cody_common_genes = set(df_normalized.columns)
    else:
        cody_common_genes = cody_common_genes.intersection(set(df_normalized.columns))
    
    if i == 0:
        print(f"  Example file: {filename}")
        print(f"  Original shape: {df.shape}")
        print(f"  After filtering: {df_normalized.shape}")
        print(f"  Removed {(~zero_lib_mask).sum()} spots with zero counts")

print(f"Common genes across all Cody files: {len(cody_common_genes)}")


# ===== PROCESSING FREDHUTCH (FH) DATA =====
print("\nProcessing FredHutch data...")
fh_data_dir = "FredHutch_Colorectal/CARD_Need_Files"
fh_expression_files = glob.glob(os.path.join(fh_data_dir, "*_expression.csv"))
print(f"Found {len(fh_expression_files)} FH expression files")

fh_normalized_dfs = []
fh_common_genes = None

for i, file_path in enumerate(tqdm(fh_expression_files, desc="Processing FH files")):
    # Extract sample identifier
    filename = os.path.basename(file_path)
    sample_id = filename.replace('_expression.csv', '')
    
    # Read raw expression data
    df = pd.read_csv(file_path, index_col=0)
    
    # Update row names: barcode -> barcode_sampleID
    df.index = df.index.astype(str) + f"_{sample_id}"
    
    # Count normalization: divide by library size
    library_sizes = df.sum(axis=1)
    # Remove spots with zero library size
    zero_lib_mask = library_sizes > 0
    df_filtered = df[zero_lib_mask]
    library_sizes_filtered = library_sizes[zero_lib_mask]
    
    # Normalize: count / library_size
    df_normalized = df_filtered.div(library_sizes_filtered, axis=0)
    
    fh_normalized_dfs.append(df_normalized)
    
    # Find intersection of genes across all FH files
    if fh_common_genes is None:
        fh_common_genes = set(df_normalized.columns)
    else:
        fh_common_genes = fh_common_genes.intersection(set(df_normalized.columns))
    
    if i == 0:
        print(f"  Example file: {filename}")
        print(f"  Original shape: {df.shape}")
        print(f"  After filtering: {df_normalized.shape}")
        print(f"  Removed {(~zero_lib_mask).sum()} spots with zero counts")

print(f"Common genes across all FH files: {len(fh_common_genes)}")


### STEP 2: Find intersected genes and merge data
# Find intersection of genes across ALL files (Cody + FH)
intersected_genes = cody_common_genes.intersection(fh_common_genes)
print(f"Final intersected genes across ALL files: {len(intersected_genes)}")
print(f"Common genes in Cody files: {len(cody_common_genes)}")
print(f"Common genes in FH files: {len(fh_common_genes)}")
print(f"Intersection of Cody and FH common genes: {len(intersected_genes)}")

# Convert to sorted list for consistent ordering
intersected_genes_list = sorted(list(intersected_genes))

# Subset all dataframes to intersected genes and combine
print("Subsetting to intersected genes and combining...")

# Process Cody data
cody_final_dfs = []
for df in tqdm(cody_normalized_dfs, desc="Processing Cody normalized data"):
    # Keep only intersected genes
    df_subset = df[intersected_genes_list]
    cody_final_dfs.append(df_subset)

# Process FH data
fh_final_dfs = []
for df in tqdm(fh_normalized_dfs, desc="Processing FH normalized data"):
    # Keep only intersected genes
    df_subset = df[intersected_genes_list]
    fh_final_dfs.append(df_subset)

# Combine all data
print("Concatenating all normalized data...")
all_final_dfs = cody_final_dfs + fh_final_dfs
Final_combined_expression_normalized = pd.concat(all_final_dfs, axis=0)

print(f"\nFinal combined normalized expression matrix:")
print(f"Shape: {Final_combined_expression_normalized.shape}")
print(f"Rows (spots): {Final_combined_expression_normalized.shape[0]}")
print(f"Columns (genes): {Final_combined_expression_normalized.shape[1]}")
print(f"Data range: {Final_combined_expression_normalized.values.min():.6f} to {Final_combined_expression_normalized.values.max():.6f}")

# Save the normalized data
output_dir = "Colorectal_Cancer_HE_patches/Gene_Expression_Prediction"
os.makedirs(output_dir, exist_ok=True)
Final_combined_expression_normalized.to_csv(os.path.join(output_dir, "Final_combined_expression_normalized.csv"))
print(f"Saved normalized expression data to: {output_dir}")


### STEP 3: Process B_matrix files - UNION of all genes
print("\n" + "="*80)
print("PROCESSING B_MATRIX FILES - UNION OF ALL GENES")
print("="*80)

# Extract sample IDs from the combined data
sample_ids = ["_".join(item[1:]) for item in Final_combined_expression_normalized.index.str.split("_")]
unique_sample_ids = sorted(list(set(sample_ids)))
print(f"Found {len(unique_sample_ids)} unique sample IDs")

# Find B_matrix files from both locations
b_matrix_folder1 = "/Users/scui2/ST/CARD_Results_Regions"
b_matrix_folder2 = "FredHutch_Colorectal/CARD_Results_Regions"

print(f"Searching for B_matrix files in:")
print(f"  Location 1: {b_matrix_folder1}")
print(f"  Location 2: {b_matrix_folder2}")

# Find all B_matrix files
b_matrix_files1 = glob.glob(os.path.join(b_matrix_folder1, "*_B_Matrix_modified.csv"))
b_matrix_files2 = glob.glob(os.path.join(b_matrix_folder2, "*_B_Matrix_modified.csv"))

print(f"Found {len(b_matrix_files1)} B_matrix files in location 1")
print(f"Found {len(b_matrix_files2)} B_matrix files in location 2")

all_b_matrix_files = b_matrix_files1 + b_matrix_files2
print(f"Total B_matrix files: {len(all_b_matrix_files)}")

# Match sample IDs with existing B_matrix files
matched_b_matrix_files = []
for sample_id in unique_sample_ids:
    # Check both locations
    file1 = os.path.join(b_matrix_folder1, f"{sample_id}_B_Matrix_modified.csv")
    file2 = os.path.join(b_matrix_folder2, f"{sample_id}_B_Matrix_modified.csv")
    
    if os.path.exists(file1):
        matched_b_matrix_files.append(file1)
    elif os.path.exists(file2):
        matched_b_matrix_files.append(file2)

print(f"Matched {len(matched_b_matrix_files)} B_matrix files for our sample IDs")


# Read all B_matrix files and create UNION of genes
print("Reading B_matrix files and creating UNION of genes...")
union_genes = set()
first_b_matrix = None

for i, file_path in enumerate(tqdm(matched_b_matrix_files, desc="Processing B_matrix files")):
    try:
        df = pd.read_csv(file_path, index_col=0)
        union_genes.update(df.index)
        
        if first_b_matrix is None:
            first_b_matrix = df.copy()
            print(f"  Example B_matrix file: {os.path.basename(file_path)}")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
        
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")

print(f"UNION of genes across all B_matrix files: {len(union_genes)}")

# Create combined B_matrix with all unique genes
print("Creating combined B_matrix with union genes...")
union_genes_list = sorted(list(union_genes))

# Initialize the combined B_matrix with union genes
Combined_B_Matrix = pd.DataFrame(index=union_genes_list, columns=first_b_matrix.columns)

# Fill the combined B_matrix by reading each file and updating missing genes
genes_filled = set()

for file_path in tqdm(matched_b_matrix_files, desc="Combining B_matrix data"):
    try:
        df = pd.read_csv(file_path, index_col=0)
        
        # For each gene in this file, add it to combined matrix if not already present
        for gene in df.index:
            if gene not in genes_filled:
                Combined_B_Matrix.loc[gene] = df.loc[gene]
                genes_filled.add(gene)
                
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")

# Fill any remaining NaN values with 0
Combined_B_Matrix = Combined_B_Matrix.fillna(0)

cell_type_order = ["ASC I", "ASC II", "ASC III", "CSC I", "CSC II", 
"CSC III", "CSC IV", "SSC I", 
"ABS", "CT", "EE", "TUF",
"T", "PLA", "MAS", "MYE", "B", "FIB", "END"]


Combined_B_Matrix.columns = ['ASC I', 'ASC II', 'ASC III', 'CSC III', 'CSC I', 'CSC IV', "CSC II", 'SSC I', 'ABS', 'CT', 'EE', 'TUF', 'T', 'PLA', 'MAS', 'MYE', 'FIB', 'B', 'END']
Combined_B_Matrix = Combined_B_Matrix[cell_type_order]



# Save combined B_matrix
Combined_B_Matrix.to_csv(os.path.join(output_dir, "Combined_B_Matrix_Union.csv"))
print(f"Saved combined B_matrix to: {output_dir}")



# Calculate relative expression of marker genes - Efficient version
print("Calculating double normalized expression for all cell types...")

# Define cell type groups for normalization
cell_type_groups = {
    "Cancer Cells": ["ASC I", "ASC II", "ASC III", "CSC I", "CSC II", "CSC III", "CSC IV", "SSC I"],
    "T Cells": ["T"],
    "Other Immune Cells": ["PLA", "MAS", "MYE", "B"],
    "Stromal Cells": ["FIB", "END"]
}


Tumor_important_marker_genes = ["TMEM54", "MARCKSL1", "FXYD3", "S100A6", "S100P"]
T_important_marker_genes = ["TC2N", "CD4", "CELF2", "PLCB1", "AAK1"]
Stromal_important_marker_genes = ["SPARC", "FN1", "COL6A2", "FSTL1", "TNC"]
Other_Immune_important_marker_genes = ["IGKC", "CD74", "VIM", "POU2AF1", "CTSL"]

marker_genes_clean_dict = {}
marker_genes_clean_dict["Cancer Cells"] = Tumor_important_marker_genes
marker_genes_clean_dict["T Cells"] = T_important_marker_genes
marker_genes_clean_dict["Stromal Cells"] = Stromal_important_marker_genes
marker_genes_clean_dict["Other Immune Cells"] = Other_Immune_important_marker_genes

# Process each cell type and collect normalized data
normalized_dfs = []

for cell_type, cell_subtypes in cell_type_groups.items():
    print(f"Processing {cell_type}...")
    marker_genes = marker_genes_clean_dict[cell_type]
    
    # Create a dictionary to store normalized genes for this cell type
    normalized_genes = {}
    
    for gene in marker_genes:
        # Get the mean B_matrix value for this gene across relevant cell subtypes
        b_matrix_mean = Combined_B_Matrix.T.loc[cell_subtypes, gene].mean(axis=0)
        
        # Normalize: expression / B_matrix_mean
        normalized_gene_expr = Final_combined_expression_normalized[gene].div(b_matrix_mean)
        normalized_genes[gene] = normalized_gene_expr
    
    # Convert to DataFrame
    cell_type_df = pd.DataFrame(normalized_genes, index=Final_combined_expression_normalized.index)
    normalized_dfs.append(cell_type_df)
    print(f"  Processed {len(marker_genes)} genes for {cell_type}")

# Concatenate all normalized DataFrames at once (efficient)
print("Concatenating all normalized gene expressions...")
Final_combined_expression_double_normalized = pd.concat(normalized_dfs, axis=1)


Final_combined_expression_double_normalized_log = np.log2(Final_combined_expression_double_normalized * 10000 + 1)
Final_combined_expression_double_normalized_log.to_csv(os.path.join(output_dir, "Final_combined_expression_double_normalized_log.csv"))


### Extract Cell Type Proportions for each barcode

# Define the two directories to search for CARD results
card_dirs = [
    "FredHutch_Colorectal/CARD_Results_Regions",
    "/Users/scui2/ST/CARD_Results_Regions"
]

# Cell type grouping mapping (same as in original script)
cell_type_groups = {
    'Cancer_Cells': ['ASC I', 'ASC II', 'ASC III', 'CSC I', 'CSC II', 'CSC III', 'CSC IV', 'SSC I'],
    'Normal_Epithelial': ['TUF', 'EE', 'ABS', 'CT'],
    'Stromal': ['FIB', 'END'],
    'T_Cells': ['T'],
    'Other_Immune': ['PLA', 'MAS', 'MYE', 'B']
}

# Initialize cell proportions DataFrame with same index as expression data
cell_proportions_df = pd.DataFrame(
    index=Final_combined_expression_double_normalized_log.index,
    columns=['Cancer_Cells_Proportion', 'Normal_Epithelial_Proportion', 'Stromal_Proportion', 'T_Cells_Proportion', 'Other_Immune_Proportion']
)
cell_proportions_df = cell_proportions_df.fillna(0.0)

print(f"Initialized cell proportions DataFrame: {cell_proportions_df.shape}")

# Extract unique sample IDs from barcode index
sample_ids = ["_".join(item.split("_")[1:]) for item in Final_combined_expression_double_normalized_log.index.str.split("_")]
unique_samples = sorted(list(set(sample_ids)))
print(f"Processing {len(unique_samples)} unique samples...")

# Process each sample
matched_barcodes = 0
total_barcodes = len(Final_combined_expression_double_normalized_log.index)

for sample_id in unique_samples:
    print(f"\nProcessing sample: {sample_id}")
    
    # Find CSV files for this sample in both directories
    sample_csv_files = []
    for card_dir in card_dirs:
        if os.path.exists(card_dir):
            pattern = os.path.join(card_dir, f"*{sample_id}*_celltype_proportion_modified.csv")
            sample_csv_files.extend(glob.glob(pattern))
    
    if not sample_csv_files:
        print(f"  Warning: No CARD proportion files found for sample {sample_id}")
        continue
    
    print(f"  Found {len(sample_csv_files)} CSV files")
    
    # Collect proportions for this sample
    sample_proportions = {}  # clean_barcode -> [5 cell type proportions]
    
    for csv_file in sample_csv_files:
        print(f"    Reading: {os.path.basename(csv_file)}")
        try:
            # Read the CSV file
            prop_df = pd.read_csv(csv_file, index_col=0)
            print(f"      Shape: {prop_df.shape}")
            
            # Process each barcode in CSV
            for csv_barcode in prop_df.index:
                # Calculate 5 cell type group proportions
                cell_props = {}
                for group_name, cell_types in cell_type_groups.items():
                    # Sum proportions for cell types in this group
                    group_proportion = 0.0
                    for cell_type in cell_types:
                        if cell_type in prop_df.columns:
                            group_proportion += prop_df.loc[csv_barcode, cell_type]
                    cell_props[group_name] = group_proportion
                
                # Store proportions using CSV barcode as key
                sample_proportions[str(csv_barcode)] = [
                    cell_props['Cancer_Cells'],
                    cell_props['Normal_Epithelial'], 
                    cell_props['Stromal'],
                    cell_props['T_Cells'],
                    cell_props['Other_Immune']
                ]
        except Exception as e:
            print(f"      Error reading {csv_file}: {e}")
            continue
    
    print(f"  Collected {len(sample_proportions)} barcodes from CSV files")
    
    # Match expression data barcodes with CSV barcodes for this sample
    sample_matched = 0
    for expr_barcode in Final_combined_expression_double_normalized_log.index:
        # Check if this barcode belongs to current sample
        expr_sample = "_".join(expr_barcode.split("_")[1:])
        if expr_sample != sample_id:
            continue
            
        # Extract clean barcode (remove sample and region suffix)
        # Example: TTGTTTGTGTAAATTC-1_8899_AS_7_region1 -> TTGTTTGTGTAAATTC-1
        clean_barcode = expr_barcode.split('_')[0]
        
        # Try to find this clean barcode in sample_proportions
        if clean_barcode in sample_proportions:
            proportions = sample_proportions[clean_barcode]
            cell_proportions_df.loc[expr_barcode, 'Cancer_Cells_Proportion'] = proportions[0]
            cell_proportions_df.loc[expr_barcode, 'Normal_Epithelial_Proportion'] = proportions[1]
            cell_proportions_df.loc[expr_barcode, 'Stromal_Proportion'] = proportions[2]
            cell_proportions_df.loc[expr_barcode, 'T_Cells_Proportion'] = proportions[3]
            cell_proportions_df.loc[expr_barcode, 'Other_Immune_Proportion'] = proportions[4]
            sample_matched += 1
            matched_barcodes += 1
    
    print(f"  Matched {sample_matched} barcodes for sample {sample_id}")

# Final summary
print(f"\nFinal cell proportion matching results:")
print(f"  Total matched barcodes: {matched_barcodes}/{total_barcodes}")
print(f"  Total unmatched barcodes: {total_barcodes - matched_barcodes}/{total_barcodes}")
print(f"  Cell proportions DataFrame shape: {cell_proportions_df.shape}")
print(f"  Cell type order: {list(cell_proportions_df.columns)}")


### Filter to keep only matched barcodes (intersected)

# Find barcodes that have non-zero cell proportions (matched barcodes)
cell_prop_sums = cell_proportions_df.sum(axis=1)
matched_barcode_mask = cell_prop_sums > 0.0
matched_barcodes_list = cell_proportions_df.index[matched_barcode_mask]

print(f"Barcodes with cell proportions: {len(matched_barcodes_list)}")
print(f"Barcodes without cell proportions: {len(cell_proportions_df) - len(matched_barcodes_list)}")

# Filter expression data to keep only matched barcodes
Final_combined_expression_double_normalized_log_filtered = Final_combined_expression_double_normalized_log.loc[matched_barcodes_list]
cell_proportions_df_filtered = cell_proportions_df.loc[matched_barcodes_list]

print(f"\nFiltered expression data shape: {Final_combined_expression_double_normalized_log_filtered.shape}")
print(f"Filtered cell proportions shape: {cell_proportions_df_filtered.shape}")

# Verify no zero cell proportions remain
remaining_zeros = (cell_proportions_df_filtered.sum(axis=1) == 0).sum()
print(f"Remaining barcodes with zero cell proportions: {remaining_zeros}")

# Display basic statistics for filtered data
print(f"\nFiltered cell proportions statistics:")
for col in cell_proportions_df_filtered.columns:
    non_zero = (cell_proportions_df_filtered[col] > 0).sum()
    mean_val = cell_proportions_df_filtered[col].mean()
    print(f"  {col}: {non_zero}/{len(cell_proportions_df_filtered)} non-zero values, mean = {mean_val:.4f}")

# Save filtered data
cell_proportions_df_filtered.to_csv(os.path.join(output_dir, "Cell_Type_Proportions_Filtered.csv"))
Final_combined_expression_double_normalized_log_filtered.to_csv(os.path.join(output_dir, "Final_combined_expression_double_normalized_log_filtered.csv"))






### XGBoost Training for Selected Important Genes



from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# Combine all selected genes
all_selected_genes = (Tumor_important_marker_genes + T_important_marker_genes + 
                     Stromal_important_marker_genes + Other_Immune_important_marker_genes)

# Create gene to cell type mapping
gene_to_celltype = {}
for gene in Tumor_important_marker_genes:
    gene_to_celltype[gene] = "Tumor"
for gene in T_important_marker_genes:
    gene_to_celltype[gene] = "T"
for gene in Stromal_important_marker_genes:
    gene_to_celltype[gene] = "Stromal"  
for gene in Other_Immune_important_marker_genes:
    gene_to_celltype[gene] = "Other_Immune"


# Load combined dataset (same as old version)
combined_file = "Colorectal_Cancer_HE_patches/Gene_Expression_Prediction/Training_features/Combined_all_celltypes_Virchow2_300genes.pt"
print(f"\nLoading combined dataset: {combined_file}")

data = torch.load(combined_file, weights_only=False)
features = data['embeddings']  # [n_tiles, n_features]
full_marker_expression = data['marker_expression']  # [n_tiles, 300_genes]
individual_ids = data['individual_ids']
tile_ids = data['tile_ids']
full_marker_genes = data['marker_genes']  # 300 genes



print(f"Dataset loaded:")
print(f"  Features shape: {features.shape}")
print(f"  Expression shape: {full_marker_expression.shape}")
print(f"  Number of genes: {len(full_marker_genes)}")
print(f"  Number of tiles: {len(tile_ids)}")
print(f"  Number of individuals: {len(set(individual_ids))}")

### Exclude specific individual
excluded_individual = ["SU-15-27301-B1", "SU-16-02468-B1"] 
valid_indices = [i for i, ind_id in enumerate(individual_ids) if ind_id not in excluded_individual]

# Filter all data to exclude these individuals
features = features[valid_indices]
individual_ids = [individual_ids[i] for i in valid_indices]
tile_ids = [tile_ids[i] for i in valid_indices]

print(f"After excluding individuals:")
print(f"  Features shape: {features.shape}")
print(f"  Number of tiles: {len(tile_ids)}")
print(f"  Number of individuals: {len(set(individual_ids))}")

# Get unique individuals for cross-validation
unique_individuals = sorted(list(set(individual_ids)))

# Find intersected tiles between all datasets
print(f"\nFinding intersected tiles...")
print(f"  Final_combined tiles: {len(Final_combined_expression_double_normalized_log_filtered)}")
print(f"  Cell_proportions tiles: {len(cell_proportions_df_filtered)}")
print(f"  Virchow2 tiles: {len(tile_ids)}")

# Get tile sets
final_combined_tiles = set(Final_combined_expression_double_normalized_log_filtered.index)
cell_proportions_tiles = set(cell_proportions_df_filtered.index)
virchow2_tiles = set(tile_ids)

# Find intersection of all three datasets
intersected_tiles = final_combined_tiles & cell_proportions_tiles & virchow2_tiles
intersected_tiles = sorted(list(intersected_tiles))

print(f"  Intersected tiles: {len(intersected_tiles)}")


# Create mapping from tile_id to index in Virchow2 data
virchow_tile_to_idx = {}
for i, tile_id in enumerate(tile_ids):
    virchow_tile_to_idx[tile_id] = i

# Extract data for intersected tiles only
matched_features = []
matched_individual_ids = []
matched_tile_ids = []

for tile in intersected_tiles:
    if tile in virchow_tile_to_idx:
        virchow_idx = virchow_tile_to_idx[tile]
        matched_features.append(features[virchow_idx])
        matched_individual_ids.append(individual_ids[virchow_idx])
        matched_tile_ids.append(tile)

# Convert to arrays
features = torch.stack(matched_features) if matched_features else torch.empty(0)
individual_ids = matched_individual_ids
tile_ids = matched_tile_ids


# Get expression data for 20 genes (intersected tiles only)

marker_expression_df = Final_combined_expression_double_normalized_log_filtered.loc[
    intersected_tiles, all_selected_genes
]

marker_expression = marker_expression_df.values

# Get cell type proportions for intersected tiles
intersected_cell_proportions = cell_proportions_df_filtered.loc[intersected_tiles]

marker_genes = marker_expression_df.columns.tolist()  # Use deduplicated column names
unique_individuals = sorted(list(set(individual_ids)))

print(f"\nFinal intersected data:")
print(f"  Features shape: {features.shape}")
print(f"  Marker expression shape: {marker_expression.shape}")
print(f"  Cell proportions shape: {intersected_cell_proportions.shape}")
print(f"  Number of genes: {len(marker_genes)}")
print(f"  Number of tiles: {len(tile_ids)}")
print(f"  Number of individuals: {len(unique_individuals)}")

# XGBoost parameters 
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'n_jobs': -1,
    'eta': 0.01,
    'gamma': 0,
    'min_child_weight': 5,
    'colsample_bytree': 0.05,
    'subsample': 0.5,
    'alpha': 0.1,
    'lambda': 0.01,
    'max_depth': 12,
    'seed': 42
}
num_boost_round = 800

# Cell type proportion mapping for partial correlation
celltype_to_proportion = {
    "Tumor": "Cancer_Cells_Proportion",
    "T": "T_Cells_Proportion",
    "Stromal": "Stromal_Proportion",
    "Other_Immune": "Other_Immune_Proportion"
}

# Function to calculate partial correlation with single control variable
def partial_correlation(x, y, control):
    """
    Calculate partial correlation between x and y, controlling for single control variable
    """
    control = control.reshape(-1, 1)
    
    # Regress x on control, get residuals
    reg_x = LinearRegression().fit(control, x)
    residual_x = x - reg_x.predict(control)
    
    # Regress y on control, get residuals  
    reg_y = LinearRegression().fit(control, y)
    residual_y = y - reg_y.predict(control)
    
    # Correlation between residuals = partial correlation
    if len(residual_x) > 1:
        partial_corr, p_value = pearsonr(residual_x, residual_y)
        return partial_corr if not np.isnan(partial_corr) else 0.0
    else:
        return 0.0

# Initialize results DataFrames
results_df_regular = pd.DataFrame(index=unique_individuals, columns=all_selected_genes)
results_df_partial = pd.DataFrame(index=unique_individuals, columns=all_selected_genes)

# Create output directories
xgb_output_dir = os.path.join(output_dir, "xgboost_results")
detailed_results_dir = os.path.join(output_dir, "detailed_gene_results")
os.makedirs(xgb_output_dir, exist_ok=True)
os.makedirs(detailed_results_dir, exist_ok=True)


### Main XGBoost training loop for each gene
for gene_idx, gene_name in enumerate(all_selected_genes):
    gene_celltype = gene_to_celltype[gene_name]
    proportion_col = celltype_to_proportion[gene_celltype]
    
    print(f"\n{'='*60}")
    print(f"Processing gene {gene_idx + 1}/{len(all_selected_genes)}: {gene_name}")
    print(f"Cell type: {gene_celltype}, Proportion column: {proportion_col}")
    print(f"{'='*60}")
    
    gene_predictions = {}
    gene_actuals = {}
    
    ### Leave-one-individual-out cross-validation
    for test_individual in unique_individuals:
        print(f"  Testing on individual: {test_individual}")
        
        # Split data
        train_indices = [i for i, ind_id in enumerate(individual_ids) if ind_id != test_individual]
        test_indices = [i for i, ind_id in enumerate(individual_ids) if ind_id == test_individual]
        
        if len(test_indices) == 0:
            print(f"    Warning: No test data for {test_individual}")
            continue
            
        # Get train/test data
        X_train = features[train_indices]
        y_train = marker_expression[train_indices, gene_idx]
        X_test = features[test_indices]
        y_test = marker_expression[test_indices, gene_idx]
        
        print(f"    Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Train model
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=num_boost_round,
            verbose_eval=False
        )
        
        # Make predictions
        y_pred = model.predict(dtest)
        
        # Store results
        gene_predictions[test_individual] = y_pred
        gene_actuals[test_individual] = y_test
        
        print(f"    Predicted values: min={y_pred.min():.4f}, max={y_pred.max():.4f}, mean={y_pred.mean():.4f}")
        print(f"    Actual values: min={y_test.min():.4f}, max={y_test.max():.4f}, mean={y_test.mean():.4f}")
    
    ### Calculate correlations for each individual
    individual_correlations_regular = {}
    individual_correlations_partial = {}
    
    for individual in unique_individuals:
        if individual in gene_predictions and individual in gene_actuals:
            pred_vals = gene_predictions[individual]
            actual_vals = gene_actuals[individual]
            
            # Get indices for this individual's samples
            individual_indices = [i for i, ind_id in enumerate(individual_ids) if ind_id == individual]
            
            if len(pred_vals) > 1 and len(individual_indices) > 1:  # Need at least 2 points for correlation
                # Regular Pearson correlation
                regular_corr, _ = pearsonr(actual_vals, pred_vals)
                individual_correlations_regular[individual] = regular_corr if not np.isnan(regular_corr) else 0.0
                
                # Partial correlation (controlling for corresponding cell type proportion only)
                individual_cell_props = intersected_cell_proportions.loc[
                    intersected_cell_proportions.index[individual_indices], proportion_col
                ].values
                
                if len(individual_cell_props) == len(pred_vals):
                    partial_corr = partial_correlation(actual_vals, pred_vals, individual_cell_props)
                    individual_correlations_partial[individual] = partial_corr
                else:
                    individual_correlations_partial[individual] = 0.0
            else:
                individual_correlations_regular[individual] = 0.0
                individual_correlations_partial[individual] = 0.0
        else:
            individual_correlations_regular[individual] = np.nan
            individual_correlations_partial[individual] = np.nan
    
    ### Update results DataFrames
    for individual in unique_individuals:
        if individual in individual_correlations_regular:
            results_df_regular.loc[individual, gene_name] = individual_correlations_regular[individual]
        if individual in individual_correlations_partial:
            results_df_partial.loc[individual, gene_name] = individual_correlations_partial[individual]
    
    ### Calculate and print summary statistics
    valid_correlations_regular = [corr for corr in individual_correlations_regular.values() if not np.isnan(corr)]
    valid_correlations_partial = [corr for corr in individual_correlations_partial.values() if not np.isnan(corr)]
    
    print(f"\n  Gene {gene_name} Summary:")
    if valid_correlations_regular:
        mean_regular = np.mean(valid_correlations_regular)
        median_regular = np.median(valid_correlations_regular)
        print(f"    Regular Pearson - Mean: {mean_regular:.4f}, Median: {median_regular:.4f}")
    
    if valid_correlations_partial:
        mean_partial = np.mean(valid_correlations_partial)
        median_partial = np.median(valid_correlations_partial)
        print(f"    Partial correlation - Mean: {mean_partial:.4f}, Median: {median_partial:.4f}")
    
    print(f"    Valid individuals: Regular={len(valid_correlations_regular)}, Partial={len(valid_correlations_partial)}/{len(unique_individuals)}")
    
    # Calculate mediation ratio if both are available
    if valid_correlations_regular and valid_correlations_partial:
        mean_mediation_ratio = (np.mean(valid_correlations_regular) - np.mean(valid_correlations_partial)) / np.mean(valid_correlations_regular) if np.mean(valid_correlations_regular) != 0 else 0
        print(f"    Mediation ratio (cell type effect): {mean_mediation_ratio:.4f}")
    
    ### Save detailed gene results (copied from old version)
    # Collect all predictions and cell proportions for this gene
    all_tile_ids = []
    all_actual = []
    all_predicted = []
    all_cell_props = []
    
    for individual in unique_individuals:
        individual_indices = [i for i, ind_id in enumerate(individual_ids) if ind_id == individual]
        if individual_indices:
            # Get actual values
            actual_vals = marker_expression[individual_indices, gene_idx]
            
            # Get predictions from the trained model
            individual_features = features[individual_indices]
            pred_vals = model.predict(xgb.DMatrix(individual_features.numpy()))
            
            # Get cell proportions
            individual_cell_props_all = intersected_cell_proportions.iloc[individual_indices]
            
            # Store data
            for i, idx in enumerate(individual_indices):
                all_tile_ids.append(tile_ids[idx])
                all_actual.append(actual_vals[i])
                all_predicted.append(pred_vals[i])
                # Get all 5 cell type proportions for this tile
                tile_props = individual_cell_props_all.iloc[i]
                all_cell_props.append([
                    tile_props['Cancer_Cells_Proportion'],
                    tile_props['Normal_Epithelial_Proportion'], 
                    tile_props['T_Cells_Proportion'],
                    tile_props['Stromal_Proportion'],
                    tile_props['Other_Immune_Proportion']
                ])
    
    # Create detailed DataFrame for this gene
    if all_tile_ids:
        detailed_df = pd.DataFrame({
            'Tile_ID': all_tile_ids,
            f'{gene_name}_True': all_actual,
            f'{gene_name}_Predicted': all_predicted
        })
        
        # Add cell type proportions
        cell_type_names = ['Cancer_Cells', 'Normal_Epithelial', 'T_Cells', 'Stromal', 'Other_Immune']
        for i, cell_type in enumerate(cell_type_names):
            detailed_df[f'{cell_type}_Proportion'] = [props[i] for props in all_cell_props]
        
        # Save detailed CSV directly to detailed_gene_results folder (no subfolders)
        detailed_csv_path = os.path.join(detailed_results_dir, f"{gene_name}_detailed_predictions.csv")
        detailed_df.to_csv(detailed_csv_path, index=False)
        print(f"  Saved detailed predictions to: {detailed_csv_path}")
    else:
        print(f"  Warning: No data available for detailed CSV for gene {gene_name}")

# Save results to CSV
regular_output_path = os.path.join(xgb_output_dir, "important_genes_regular_correlations.csv")
partial_output_path = os.path.join(xgb_output_dir, "important_genes_partial_correlations.csv")

results_df_regular.to_csv(regular_output_path)
results_df_partial.to_csv(partial_output_path)

print(f"\n" + "="*80)
print("XGBOOST TRAINING COMPLETED")
print("="*80)
print(f"Regular correlations saved to: {regular_output_path}")
print(f"Partial correlations saved to: {partial_output_path}")
print(f"Results shape: {results_df_regular.shape}")




### Create correlation boxplot visualization
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import wilcoxon

print(f"\nCreating correlation boxplot visualization...")

# Read the correlation results
regular_corr_df = pd.read_csv(regular_output_path, index_col=0)
partial_corr_df = pd.read_csv(partial_output_path, index_col=0)

# Define gene groups and colors
Tumor_genes = Tumor_important_marker_genes
T_genes = T_important_marker_genes  
Stromal_genes = Stromal_important_marker_genes
Other_Immune_genes = Other_Immune_important_marker_genes

gene_to_color = {}
for gene in Tumor_genes:
    gene_to_color[gene] = '#E41A1C'  # Red for Tumor
for gene in T_genes:
    gene_to_color[gene] = '#4DAF4A'  # Green for T
for gene in Stromal_genes:
    gene_to_color[gene] = '#FFFF33'  # Yellow for Stromal  
for gene in Other_Immune_genes:
    gene_to_color[gene] = '#FF7F00'  # Orange for Other Immune

# Get the genes that are actually in our results (after deduplication)
available_genes = [gene for gene in marker_genes if gene in regular_corr_df.columns]
print(f"Available genes for plotting: {len(available_genes)}")

# Create the figure
fig, ax = plt.subplots(figsize=(20, 8))

# Set up positions for genes and boxplots
gene_positions = []
x_labels = []
x_ticks = []

for i, gene in enumerate(available_genes):
    # Each gene gets two positions: regular (left) and partial (right)
    base_pos = i * 3  # Space between gene groups
    reg_pos = base_pos - 0.3  # Regular correlation position (left)
    part_pos = base_pos + 0.3  # Partial correlation position (right)
    
    gene_positions.append((reg_pos, part_pos))
    x_labels.append(gene)
    x_ticks.append(base_pos)  # Center position for gene label
    
    # Get correlation data for this gene
    reg_data = regular_corr_df[gene].dropna()
    part_data = partial_corr_df[gene].dropna()
    
    # Get color for this gene
    color = gene_to_color.get(gene, '#000000')  # Default to black if not found
    
    # Create boxplots
    if len(reg_data) > 0:
        bp_reg = ax.boxplot([reg_data], positions=[reg_pos], widths=0.4, 
                           patch_artist=True, showfliers=False)
        bp_reg['boxes'][0].set_facecolor(color)
        bp_reg['boxes'][0].set_alpha(0.7)
        bp_reg['boxes'][0].set_edgecolor('black')
        bp_reg['boxes'][0].set_linewidth(2)
        # Make other elements bold
        for element in ['whiskers', 'caps', 'medians']:
            for item in bp_reg[element]:
                item.set_color('black')
                item.set_linewidth(2)
        
        # Add individual points (circles for regular) with jitter
        y_reg = reg_data.values
        x_reg = np.random.normal(reg_pos, 0.05, len(y_reg))  # Add jitter
        ax.scatter(x_reg, y_reg, c=color, marker='o', s=40, alpha=0.8, edgecolors='black', linewidth=1.5)
    
    if len(part_data) > 0:
        bp_part = ax.boxplot([part_data], positions=[part_pos], widths=0.4,
                            patch_artist=True, showfliers=False)
        bp_part['boxes'][0].set_facecolor(color)
        bp_part['boxes'][0].set_alpha(0.7)
        bp_part['boxes'][0].set_edgecolor('black')
        bp_part['boxes'][0].set_linewidth(2)
        # Make other elements bold
        for element in ['whiskers', 'caps', 'medians']:
            for item in bp_part[element]:
                item.set_color('black')
                item.set_linewidth(2)
        
        # Add individual points (triangles for partial) with jitter
        y_part = part_data.values
        x_part = np.random.normal(part_pos, 0.05, len(y_part))  # Add jitter
        ax.scatter(x_part, y_part, c=color, marker='^', s=40, alpha=0.8, edgecolors='black', linewidth=1.5)
    
    # Perform Wilcoxon signed-rank test between regular and partial correlations
    if len(reg_data) > 0 and len(part_data) > 0:
        # Get matched pairs (same individuals)
        reg_matched = []
        part_matched = []
        
        for individual in regular_corr_df.index:
            if individual in reg_data.index and individual in part_data.index:
                reg_val = reg_data[individual]
                part_val = part_data[individual]
                if not (np.isnan(reg_val) or np.isnan(part_val)):
                    reg_matched.append(reg_val)
                    part_matched.append(part_val)
        
        if len(reg_matched) >= 5:  # Need at least 5 pairs for meaningful test
            try:
                statistic, p_value = wilcoxon(reg_matched, part_matched, alternative='two-sided')
                
                # Determine significance stars
                if p_value < 0.001:
                    stars = '***'
                elif p_value < 0.01:
                    stars = '**'
                elif p_value < 0.05:
                    stars = '*'
                else:
                    stars = 'ns'
                
                # Add p-value and stars at fixed y=0.9 position (no connecting line)
                y_pos = 0.9
                
                # Add p-value text (stars and p-value only)
                ax.text(base_pos, y_pos, f'{stars}\np={p_value:.3f}', 
                       ha='center', va='center', fontsize=8, fontweight='bold', color='black')
                
            except Exception as e:
                print(f"  Warning: Could not perform Wilcoxon test for {gene}: {e}")
        else:
            print(f"  Warning: Not enough matched pairs for {gene} ({len(reg_matched)} pairs)")
    else:
        print(f"  Warning: Missing data for Wilcoxon test for {gene}")

# Customize the plot
ax.set_ylabel('Pearson Correlation', fontsize=14, fontweight='bold')

# Set x-axis ticks (labels will be added later)
ax.set_xticks(x_ticks)

# Add vertical grid lines between gene groups
for i in range(1, len(available_genes)):
    ax.axvline(x=i*3 - 1.5, color='lightgray', linestyle='--', alpha=0.5)

# Create legends
# Legend 1: Correlation type (shape)
correlation_legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, 
           label='Regular Correlation', markeredgecolor='black'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10,
           label='Partial Correlation', markeredgecolor='black')
]

# Legend 2: Gene type (color)
gene_type_legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#E41A1C', markersize=10,
           label='Tumor Markers', markeredgecolor='black'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#4DAF4A', markersize=10,
           label='T Cell Markers', markeredgecolor='black'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFFF33', markersize=10,
           label='Stromal Markers', markeredgecolor='black'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF7F00', markersize=10,
           label='Other Immune Markers', markeredgecolor='black')
]

# Remove x-axis tick labels and set custom ones
ax.set_xticklabels([])

# Add legends to the right side of the plot (no titles)
leg1 = ax.legend(handles=correlation_legend_elements, 
                loc='center left', bbox_to_anchor=(1.02, 0.75), fontsize=12)
leg2 = ax.legend(handles=gene_type_legend_elements,
                loc='center left', bbox_to_anchor=(1.02, 0.25), fontsize=12)

# Make legend text bold
for text in leg1.get_texts():
    text.set_fontweight('bold')

for text in leg2.get_texts():
    text.set_fontweight('bold')

# Add the first legend back
ax.add_artist(leg1)

# Styling - show all four borders, make them bold and black
ax.grid(True, alpha=0.3)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(2)

# Make y-axis tick labels bold
ax.tick_params(axis='y', labelsize=12, labelcolor='black', width=2, length=6)
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

# Make x-axis ticks bold
ax.tick_params(axis='x', width=2, length=6)

# Set Y-axis range
ax.set_ylim(bottom=ax.get_ylim()[0], top=1.0)

# Add x-axis labels with gene-specific colors (Stromal genes use black for readability)
gene_colors_for_labels = []
for gene in available_genes:
    if gene in Stromal_genes:
        gene_colors_for_labels.append('#000000')  # Black for Stromal genes
    else:
        gene_colors_for_labels.append(gene_to_color.get(gene, '#000000'))

y_min = ax.get_ylim()[0]
y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
for i, (tick, label, color) in enumerate(zip(x_ticks, x_labels, gene_colors_for_labels)):
    ax.text(tick, y_min - 0.08 * y_range, 
            label, ha='center', va='top', rotation=45, fontsize=12, fontweight='bold', color=color)

plt.tight_layout()

# Save the plot
boxplot_output_path = os.path.join(output_dir, "important_genes_correlation_boxplot.png")
plt.savefig(boxplot_output_path, dpi=600, bbox_inches='tight')
plt.close()

print(f"Correlation boxplot saved to: {boxplot_output_path}")
print("Boxplot visualization completed!")





### Generate scatter plots for specific genes 


from sklearn.metrics import r2_score
from scipy.stats import spearmanr


# Define specific genes and their corresponding cell type proportions and colors
target_genes = {
    'S100A6': ('Cancer_Cells_Proportion', '#E41A1C'),     # Tumor gene - Red
    'COL6A2': ('Stromal_Proportion', '#FFFF33'),          # Stromal gene - Yellow
    'IGKC': ('Other_Immune_Proportion', '#FF7F00')        # Other Immune gene - Orange
}

# Create output folder for scatter plots
scatter_plots_folder = os.path.join(output_dir, "specific_gene_scatter_plots")
os.makedirs(scatter_plots_folder, exist_ok=True)

print(f"Target genes: {list(target_genes.keys())}")
print(f"Output folder: {scatter_plots_folder}")

# Process each target gene
for gene_name, (proportion_col, gene_color) in target_genes.items():
    print(f"\nProcessing {gene_name} with {proportion_col} (color: {gene_color})...")
    
    # Look for the detailed CSV file for this gene
    detailed_csv_path = os.path.join(detailed_results_dir, f"{gene_name}_detailed_predictions.csv")
    
    if not os.path.exists(detailed_csv_path):
        print(f"  Warning: CSV file not found for {gene_name}: {detailed_csv_path}")
        continue
    
    # Read the CSV file
    df = pd.read_csv(detailed_csv_path)
    
    # Extract relevant columns
    true_col = f"{gene_name}_True"
    pred_col = f"{gene_name}_Predicted"
    
    if true_col not in df.columns or pred_col not in df.columns or proportion_col not in df.columns:
        print(f"  Warning: Required columns not found for {gene_name}. Skipping...")
        continue
    
    true_expr = df[true_col].values
    pred_expr = df[pred_col].values
    proportion = df[proportion_col].values
    
    # Create 3 separate plots for this gene
    
    ### Plot 1: Predicted vs Actual Expression (X-axis = Actual)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(true_expr, pred_expr, alpha=0.6, s=20, color=gene_color)
    
    # Fit linear regression line
    lr1 = LinearRegression()
    lr1.fit(true_expr.reshape(-1, 1), pred_expr)
    pred_line1 = lr1.predict(true_expr.reshape(-1, 1))
    
    # Calculate Spearman correlation
    rho_1, _ = spearmanr(true_expr, pred_expr)
    
    # Plot regression line
    sorted_indices = np.argsort(true_expr)
    ax1.plot(true_expr[sorted_indices], pred_line1[sorted_indices], 'k-', linewidth=2)
    
    # Styling
    ax1.set_xlabel('Actual Expression', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Predicted Expression', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add Spearman correlation annotation in top-left
    ax1.text(0.05, 0.95, f'Spearman ρ = {rho_1:.3f}', transform=ax1.transAxes, 
             fontsize=12, fontweight='bold', color='black', va='top', ha='left')
    
    # Bold borders
    for spine in ax1.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)
    
    # Bold tick labels
    ax1.tick_params(axis='both', labelsize=12, width=2, length=6)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontweight('bold')
    
    # Save plot 1
    output_plot1_path = os.path.join(scatter_plots_folder, f"{gene_name}_true_vs_predicted.png")
    plt.tight_layout()
    plt.savefig(output_plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    ### Plot 2: Actual Expression vs Cell Type Proportion (X-axis = Proportion)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(proportion, true_expr, alpha=0.6, s=20, color=gene_color)
    
    # Fit linear regression line
    lr2 = LinearRegression()
    lr2.fit(proportion.reshape(-1, 1), true_expr)
    pred_line2 = lr2.predict(proportion.reshape(-1, 1))
    
    # Calculate Spearman correlation
    rho_2, _ = spearmanr(proportion, true_expr)
    
    # Plot regression line
    sorted_indices_prop = np.argsort(proportion)
    ax2.plot(proportion[sorted_indices_prop], pred_line2[sorted_indices_prop], 'k-', linewidth=2)
    
    # Format proportion column name for display
    prop_display = proportion_col.replace('_', ' ').replace('Proportion', 'Proportion')
    
    # Styling
    ax2.set_xlabel(prop_display, fontsize=14, fontweight='bold')
    ax2.set_ylabel('Actual Expression', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add Spearman correlation annotation in top-left
    ax2.text(0.05, 0.95, f'Spearman ρ = {rho_2:.3f}', transform=ax2.transAxes, 
             fontsize=12, fontweight='bold', color='black', va='top', ha='left')
    
    # Bold borders
    for spine in ax2.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)
    
    # Bold tick labels
    ax2.tick_params(axis='both', labelsize=12, width=2, length=6)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontweight('bold')
    
    # Save plot 2
    output_plot2_path = os.path.join(scatter_plots_folder, f"{gene_name}_true_vs_proportion.png")
    plt.tight_layout()
    plt.savefig(output_plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    ### Plot 3: Predicted Expression vs Cell Type Proportion (X-axis = Proportion)
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.scatter(proportion, pred_expr, alpha=0.6, s=20, color=gene_color)
    
    # Fit linear regression line
    lr3 = LinearRegression()
    lr3.fit(proportion.reshape(-1, 1), pred_expr)
    pred_line3 = lr3.predict(proportion.reshape(-1, 1))
    
    # Calculate Spearman correlation
    rho_3, _ = spearmanr(proportion, pred_expr)
    
    # Plot regression line
    ax3.plot(proportion[sorted_indices_prop], pred_line3[sorted_indices_prop], 'k-', linewidth=2)
    
    # Styling
    ax3.set_xlabel(prop_display, fontsize=14, fontweight='bold')
    ax3.set_ylabel('Predicted Expression', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add Spearman correlation annotation in top-left
    ax3.text(0.05, 0.95, f'Spearman ρ = {rho_3:.3f}', transform=ax3.transAxes, 
             fontsize=12, fontweight='bold', color='black', va='top', ha='left')
    
    # Bold borders
    for spine in ax3.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)
    
    # Bold tick labels
    ax3.tick_params(axis='both', labelsize=12, width=2, length=6)
    for label in ax3.get_xticklabels() + ax3.get_yticklabels():
        label.set_fontweight('bold')
    
    # Save plot 3
    output_plot3_path = os.path.join(scatter_plots_folder, f"{gene_name}_predicted_vs_proportion.png")
    plt.tight_layout()
    plt.savefig(output_plot3_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved scatter plots for {gene_name} (color: {gene_color}):")
    print(f"    Actual vs Predicted: {output_plot1_path}")
    print(f"    Proportion vs Actual: {output_plot2_path}")
    print(f"    Proportion vs Predicted: {output_plot3_path}")
    print(f"  Spearman ρ values - Actual vs Pred: {rho_1:.3f}, Proportion vs Actual: {rho_2:.3f}, Proportion vs Pred: {rho_3:.3f}")

print(f"\nSpecific gene scatter plot generation completed!")
print(f"All plots saved to: {scatter_plots_folder}")
print(f"Total plots generated: {len(target_genes) * 3} plots for {len(target_genes)} genes")






### Additional 10 genes processing and merging


# Define the 10 specific genes to extract
additional_genes = ['RGS11', 'VAC14', 'PTPN22', 'YAF2', 'ZFP36L2', 
                   'IFI6', 'ODF3B', 'SLC25A36', 'CUL1', 'NUP155']

print(f"Target genes: {additional_genes}")

# Extract the available genes from Final_combined_expression_normalized

additional_genes_data = Final_combined_expression_normalized[additional_genes].copy()
print(f"Extracted data shape: {additional_genes_data.shape}")

# Apply log2(x*10^4+1) transformation
additional_genes_transformed = np.log2(additional_genes_data * 10000 + 1)
print(f"Applied log2(x*10^4+1) transformation")
print(f"Transformed data range: {additional_genes_transformed.min().min():.3f} to {additional_genes_transformed.max().max():.3f}")

# Get the row index of Final_combined_expression_double_normalized_log_filtered for merging
target_barcodes = Final_combined_expression_double_normalized_log_filtered.index
print(f"Target barcodes (Final_combined_expression_double_normalized_log_filtered): {len(target_barcodes)}")

# Find intersection of barcodes
common_barcodes = additional_genes_transformed.index.intersection(target_barcodes)
print(f"Common barcodes between datasets: {len(common_barcodes)}")


additional_genes_filtered = additional_genes_transformed.loc[common_barcodes]
print(f"Filtered additional genes data shape: {additional_genes_filtered.shape}")



### XGBoost Training for Additional 10 genes 
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# Define the 10 additional genes as our target
additional_all_selected_genes = additional_genes

# Create gene to cell type mapping for additional genes (assume all are general markers)
additional_gene_to_celltype = {}
for gene in additional_all_selected_genes:
    additional_gene_to_celltype[gene] = "General"  # Since these are additional genes, treat as general

# Load combined dataset (same as original version)
combined_file = "Colorectal_Cancer_HE_patches/Gene_Expression_Prediction/Training_features/Combined_all_celltypes_Virchow2_300genes.pt"
print(f"\nLoading combined dataset for additional genes: {combined_file}")

data = torch.load(combined_file, weights_only=False)
additional_features = data['embeddings']  # [n_tiles, n_features]
additional_individual_ids = data['individual_ids']
additional_tile_ids = data['tile_ids']

print(f"Additional genes dataset loaded:")
print(f"  Features shape: {additional_features.shape}")
print(f"  Number of tiles: {len(additional_tile_ids)}")
print(f"  Number of individuals: {len(set(additional_individual_ids))}")

### Exclude specific individual
excluded_individual = ["SU-15-27301-B1", "SU-16-02468-B1"] 
valid_indices = [i for i, ind_id in enumerate(additional_individual_ids) if ind_id not in excluded_individual]

# Filter all data to exclude these individuals
additional_features = additional_features[valid_indices]
additional_individual_ids = [additional_individual_ids[i] for i in valid_indices]
additional_tile_ids = [additional_tile_ids[i] for i in valid_indices]

print(f"After excluding individuals for additional genes:")
print(f"  Features shape: {additional_features.shape}")
print(f"  Number of tiles: {len(additional_tile_ids)}")
print(f"  Number of individuals: {len(set(additional_individual_ids))}")

# Get unique individuals for cross-validation
additional_unique_individuals = sorted(list(set(additional_individual_ids)))

# Find intersected tiles between all datasets
print(f"\nFinding intersected tiles for additional genes...")
print(f"  Additional_genes_filtered tiles: {len(additional_genes_filtered)}")
print(f"  Cell_proportions tiles: {len(cell_proportions_df_filtered)}")
print(f"  Virchow2 tiles: {len(additional_tile_ids)}")

# Get tile sets
additional_genes_tiles = set(additional_genes_filtered.index)
cell_proportions_tiles = set(cell_proportions_df_filtered.index)
additional_virchow2_tiles = set(additional_tile_ids)

# Find intersection of all three datasets
additional_intersected_tiles = additional_genes_tiles & cell_proportions_tiles & additional_virchow2_tiles
additional_intersected_tiles = sorted(list(additional_intersected_tiles))

print(f"  Intersected tiles for additional genes: {len(additional_intersected_tiles)}")

# Create mapping from tile_id to index in Virchow2 data
additional_virchow_tile_to_idx = {}
for i, tile_id in enumerate(additional_tile_ids):
    additional_virchow_tile_to_idx[tile_id] = i

# Extract data for intersected tiles only
additional_matched_features = []
additional_matched_individual_ids = []
additional_matched_tile_ids = []

for tile in additional_intersected_tiles:
    if tile in additional_virchow_tile_to_idx:
        virchow_idx = additional_virchow_tile_to_idx[tile]
        additional_matched_features.append(additional_features[virchow_idx])
        additional_matched_individual_ids.append(additional_individual_ids[virchow_idx])
        additional_matched_tile_ids.append(tile)

# Convert to arrays
additional_features = torch.stack(additional_matched_features) if additional_matched_features else torch.empty(0)
additional_individual_ids = additional_matched_individual_ids
additional_tile_ids = additional_matched_tile_ids

# Get expression data for 10 additional genes (intersected tiles only)
additional_marker_expression_df = additional_genes_filtered.loc[
    additional_intersected_tiles, additional_all_selected_genes
]

additional_marker_expression = additional_marker_expression_df.values

# Get cell type proportions for intersected tiles
additional_intersected_cell_proportions = cell_proportions_df_filtered.loc[additional_intersected_tiles]

additional_marker_genes = additional_marker_expression_df.columns.tolist()
additional_unique_individuals = sorted(list(set(additional_individual_ids)))

print(f"\nFinal intersected data for additional genes:")
print(f"  Features shape: {additional_features.shape}")
print(f"  Marker expression shape: {additional_marker_expression.shape}")
print(f"  Cell proportions shape: {additional_intersected_cell_proportions.shape}")
print(f"  Number of genes: {len(additional_marker_genes)}")
print(f"  Number of tiles: {len(additional_tile_ids)}")
print(f"  Number of individuals: {len(additional_unique_individuals)}")

# XGBoost parameters (same as original)
additional_xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'n_jobs': -1,
    'eta': 0.01,
    'gamma': 0,
    'min_child_weight': 5,
    'colsample_bytree': 0.05,
    'subsample': 0.5,
    'alpha': 0.1,
    'lambda': 0.01,
    'max_depth': 12,
    'seed': 42
}
additional_num_boost_round = 800

# Cell type proportion mapping for partial correlation (use both Cancer and Stromal for additional genes)
additional_celltype_to_proportion_cancer = {
    "General": "Cancer_Cells_Proportion"  # Use Cancer proportion for first partial correlation
}

additional_celltype_to_proportion_stromal = {
    "General": "Stromal_Proportion"  # Use Stromal proportion for second partial correlation
}

# Function to calculate partial correlation with single control variable (same as original)
def additional_partial_correlation(x, y, control):
    """
    Calculate partial correlation between x and y, controlling for single control variable
    """
    control = control.reshape(-1, 1)
    
    # Regress x on control, get residuals
    reg_x = LinearRegression().fit(control, x)
    residual_x = x - reg_x.predict(control)
    
    # Regress y on control, get residuals  
    reg_y = LinearRegression().fit(control, y)
    residual_y = y - reg_y.predict(control)
    
    # Correlation between residuals = partial correlation
    if len(residual_x) > 1:
        partial_corr, p_value = pearsonr(residual_x, residual_y)
        return partial_corr if not np.isnan(partial_corr) else 0.0
    else:
        return 0.0

# Initialize results DataFrames for 3 types of correlations
additional_results_df_regular = pd.DataFrame(index=additional_unique_individuals, columns=additional_all_selected_genes)
additional_results_df_partial_cancer = pd.DataFrame(index=additional_unique_individuals, columns=additional_all_selected_genes)
additional_results_df_partial_stromal = pd.DataFrame(index=additional_unique_individuals, columns=additional_all_selected_genes)

# Create output directories
additional_xgb_output_dir = os.path.join(output_dir, "additional_genes_xgboost_results")
additional_detailed_results_dir = os.path.join(output_dir, "additional_genes_detailed_results")
os.makedirs(additional_xgb_output_dir, exist_ok=True)
os.makedirs(additional_detailed_results_dir, exist_ok=True)

### Main XGBoost training loop for each additional gene
for gene_idx, gene_name in enumerate(additional_all_selected_genes):
    gene_celltype = additional_gene_to_celltype[gene_name]
    proportion_col_cancer = additional_celltype_to_proportion_cancer[gene_celltype]
    proportion_col_stromal = additional_celltype_to_proportion_stromal[gene_celltype]
    
    print(f"\n{'='*60}")
    print(f"Processing additional gene {gene_idx + 1}/{len(additional_all_selected_genes)}: {gene_name}")
    print(f"Cell type: {gene_celltype}")
    print(f"Cancer proportion column: {proportion_col_cancer}")
    print(f"Stromal proportion column: {proportion_col_stromal}")
    print(f"{'='*60}")
    
    gene_predictions = {}
    gene_actuals = {}
    
    ### Leave-one-individual-out cross-validation
    for test_individual in additional_unique_individuals:
        print(f"  Testing on individual: {test_individual}")
        
        # Split data
        train_indices = [i for i, ind_id in enumerate(additional_individual_ids) if ind_id != test_individual]
        test_indices = [i for i, ind_id in enumerate(additional_individual_ids) if ind_id == test_individual]
        
        if len(test_indices) == 0:
            print(f"    Warning: No test data for {test_individual}")
            continue
            
        # Get train/test data
        X_train = additional_features[train_indices]
        y_train = additional_marker_expression[train_indices, gene_idx]
        X_test = additional_features[test_indices]
        y_test = additional_marker_expression[test_indices, gene_idx]
        
        print(f"    Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Train model
        model = xgb.train(
            additional_xgb_params,
            dtrain,
            num_boost_round=additional_num_boost_round,
            verbose_eval=False
        )
        
        # Make predictions
        y_pred = model.predict(dtest)
        
        # Store results
        gene_predictions[test_individual] = y_pred
        gene_actuals[test_individual] = y_test
        
        print(f"    Predicted values: min={y_pred.min():.4f}, max={y_pred.max():.4f}, mean={y_pred.mean():.4f}")
        print(f"    Actual values: min={y_test.min():.4f}, max={y_test.max():.4f}, mean={y_test.mean():.4f}")
    
    ### Calculate correlations for each individual
    individual_correlations_regular = {}
    individual_correlations_partial_cancer = {}
    individual_correlations_partial_stromal = {}
    
    for individual in additional_unique_individuals:
        if individual in gene_predictions and individual in gene_actuals:
            pred_vals = gene_predictions[individual]
            actual_vals = gene_actuals[individual]
            
            # Get indices for this individual's samples
            individual_indices = [i for i, ind_id in enumerate(additional_individual_ids) if ind_id == individual]
            
            if len(pred_vals) > 1 and len(individual_indices) > 1:  # Need at least 2 points for correlation
                # Regular Pearson correlation
                regular_corr, _ = pearsonr(actual_vals, pred_vals)
                individual_correlations_regular[individual] = regular_corr if not np.isnan(regular_corr) else 0.0
                
                # Partial correlation controlling for Cancer cells proportion
                individual_cancer_props = additional_intersected_cell_proportions.loc[
                    additional_intersected_cell_proportions.index[individual_indices], proportion_col_cancer
                ].values
                
                if len(individual_cancer_props) == len(pred_vals):
                    partial_corr_cancer = additional_partial_correlation(actual_vals, pred_vals, individual_cancer_props)
                    individual_correlations_partial_cancer[individual] = partial_corr_cancer
                else:
                    individual_correlations_partial_cancer[individual] = 0.0
                
                # Partial correlation controlling for Stromal cells proportion
                individual_stromal_props = additional_intersected_cell_proportions.loc[
                    additional_intersected_cell_proportions.index[individual_indices], proportion_col_stromal
                ].values
                
                if len(individual_stromal_props) == len(pred_vals):
                    partial_corr_stromal = additional_partial_correlation(actual_vals, pred_vals, individual_stromal_props)
                    individual_correlations_partial_stromal[individual] = partial_corr_stromal
                else:
                    individual_correlations_partial_stromal[individual] = 0.0
            else:
                individual_correlations_regular[individual] = 0.0
                individual_correlations_partial_cancer[individual] = 0.0
                individual_correlations_partial_stromal[individual] = 0.0
        else:
            individual_correlations_regular[individual] = np.nan
            individual_correlations_partial_cancer[individual] = np.nan
            individual_correlations_partial_stromal[individual] = np.nan
    
    ### Update results DataFrames
    for individual in additional_unique_individuals:
        if individual in individual_correlations_regular:
            additional_results_df_regular.loc[individual, gene_name] = individual_correlations_regular[individual]
        if individual in individual_correlations_partial_cancer:
            additional_results_df_partial_cancer.loc[individual, gene_name] = individual_correlations_partial_cancer[individual]
        if individual in individual_correlations_partial_stromal:
            additional_results_df_partial_stromal.loc[individual, gene_name] = individual_correlations_partial_stromal[individual]
    
    ### Calculate and print summary statistics
    valid_correlations_regular = [corr for corr in individual_correlations_regular.values() if not np.isnan(corr)]
    valid_correlations_partial_cancer = [corr for corr in individual_correlations_partial_cancer.values() if not np.isnan(corr)]
    valid_correlations_partial_stromal = [corr for corr in individual_correlations_partial_stromal.values() if not np.isnan(corr)]
    
    print(f"\n  Additional Gene {gene_name} Summary:")
    if valid_correlations_regular:
        mean_regular = np.mean(valid_correlations_regular)
        median_regular = np.median(valid_correlations_regular)
        print(f"    Regular Pearson - Mean: {mean_regular:.4f}, Median: {median_regular:.4f}")
    
    if valid_correlations_partial_cancer:
        mean_partial_cancer = np.mean(valid_correlations_partial_cancer)
        median_partial_cancer = np.median(valid_correlations_partial_cancer)
        print(f"    Partial correlation (Cancer control) - Mean: {mean_partial_cancer:.4f}, Median: {median_partial_cancer:.4f}")
    
    if valid_correlations_partial_stromal:
        mean_partial_stromal = np.mean(valid_correlations_partial_stromal)
        median_partial_stromal = np.median(valid_correlations_partial_stromal)
        print(f"    Partial correlation (Stromal control) - Mean: {mean_partial_stromal:.4f}, Median: {median_partial_stromal:.4f}")
    
    print(f"    Valid individuals: Regular={len(valid_correlations_regular)}, Cancer_Partial={len(valid_correlations_partial_cancer)}, Stromal_Partial={len(valid_correlations_partial_stromal)}/{len(additional_unique_individuals)}")
    
    # Calculate mediation ratios if available
    if valid_correlations_regular and valid_correlations_partial_cancer:
        mean_mediation_ratio_cancer = (np.mean(valid_correlations_regular) - np.mean(valid_correlations_partial_cancer)) / np.mean(valid_correlations_regular) if np.mean(valid_correlations_regular) != 0 else 0
        print(f"    Mediation ratio (Cancer effect): {mean_mediation_ratio_cancer:.4f}")
    
    if valid_correlations_regular and valid_correlations_partial_stromal:
        mean_mediation_ratio_stromal = (np.mean(valid_correlations_regular) - np.mean(valid_correlations_partial_stromal)) / np.mean(valid_correlations_regular) if np.mean(valid_correlations_regular) != 0 else 0
        print(f"    Mediation ratio (Stromal effect): {mean_mediation_ratio_stromal:.4f}")
    
    ### Save detailed gene results (copied from original version)
    # Collect all predictions and cell proportions for this gene
    all_tile_ids = []
    all_actual = []
    all_predicted = []
    all_cell_props = []
    
    for individual in additional_unique_individuals:
        individual_indices = [i for i, ind_id in enumerate(additional_individual_ids) if ind_id == individual]
        if individual_indices:
            # Get actual values
            actual_vals = additional_marker_expression[individual_indices, gene_idx]
            
            # Get predictions from the trained model
            individual_features = additional_features[individual_indices]
            pred_vals = model.predict(xgb.DMatrix(individual_features.numpy()))
            
            # Get cell proportions
            individual_cell_props_all = additional_intersected_cell_proportions.iloc[individual_indices]
            
            # Store data
            for i, idx in enumerate(individual_indices):
                all_tile_ids.append(additional_tile_ids[idx])
                all_actual.append(actual_vals[i])
                all_predicted.append(pred_vals[i])
                # Get all 5 cell type proportions for this tile
                tile_props = individual_cell_props_all.iloc[i]
                all_cell_props.append([
                    tile_props['Cancer_Cells_Proportion'],
                    tile_props['Normal_Epithelial_Proportion'], 
                    tile_props['T_Cells_Proportion'],
                    tile_props['Stromal_Proportion'],
                    tile_props['Other_Immune_Proportion']
                ])
    
    # Create detailed DataFrame for this gene
    if all_tile_ids:
        detailed_df = pd.DataFrame({
            'Tile_ID': all_tile_ids,
            f'{gene_name}_True': all_actual,
            f'{gene_name}_Predicted': all_predicted
        })
        
        # Add cell type proportions
        cell_type_names = ['Cancer_Cells', 'Normal_Epithelial', 'T_Cells', 'Stromal', 'Other_Immune']
        for i, cell_type in enumerate(cell_type_names):
            detailed_df[f'{cell_type}_Proportion'] = [props[i] for props in all_cell_props]
        
        # Save detailed CSV directly to additional_genes_detailed_results folder
        detailed_csv_path = os.path.join(additional_detailed_results_dir, f"{gene_name}_detailed_predictions.csv")
        detailed_df.to_csv(detailed_csv_path, index=False)
        print(f"  Saved detailed predictions to: {detailed_csv_path}")
    else:
        print(f"  Warning: No data available for detailed CSV for gene {gene_name}")

# Save results to CSV (3 output files)
additional_regular_output_path = os.path.join(additional_xgb_output_dir, "additional_genes_regular_correlations.csv")
additional_partial_cancer_output_path = os.path.join(additional_xgb_output_dir, "additional_genes_partial_correlations_cancer.csv")
additional_partial_stromal_output_path = os.path.join(additional_xgb_output_dir, "additional_genes_partial_correlations_stromal.csv")

additional_results_df_regular.to_csv(additional_regular_output_path)
additional_results_df_partial_cancer.to_csv(additional_partial_cancer_output_path)
additional_results_df_partial_stromal.to_csv(additional_partial_stromal_output_path)




