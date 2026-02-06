"""
STPath-COAD: Gene Expression Prediction - Training
===================================================

This script trains XGBoost models to predict gene expression levels from 
histopathology features extracted by foundation models (Virchow2, ResNet50, 
UNI2-h, Virchow, Prov-GigaPath, CONCH).

Author: Saishi Cui
Date: January 2026

Purpose: 
- Process and normalize spatial transcriptomics expression data
- Extract and merge cell type proportions from CARD deconvolution results
- Select marker genes and highly variable genes for prediction
- Train XGBoost models with leave-one-individual-out cross-validation
- Save prediction results for downstream visualization

Note: This script performs model training only. For visualization and analysis,
see COAD_GeneExpression_Prediction_Figures.py

Output files are saved to: Colorectal_Cancer_HE_patches/Gene_Expression_Prediction/
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
cody_data_dir = "/Users/scui2/Desktop/CARD_Need_Files"
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
fh_data_dir = "/Users/scui2/Desktop/FredHutch_Colorectal/CARD_Need_Files"
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
Final_combined_expression_lognormalized = np.log2(Final_combined_expression_normalized * 10000 + 1)


# Save the normalized data
output_dir = "Colorectal_Cancer_HE_patches/Gene_Expression_Prediction"
os.makedirs(output_dir, exist_ok=True)
Final_combined_expression_lognormalized.to_csv(os.path.join(output_dir, "Final_combined_expression_lognormalized.csv"))
print(f"Saved normalized expression data to: {output_dir}")



Final_combined_expression_lognormalized = pd.read_csv("Colorectal_Cancer_HE_patches/Gene_Expression_Prediction/Final_combined_expression_lognormalized.csv", index_col=0)

### Extract Cell Type Proportions for each barcode

# Define the two directories to search for CARD results
card_dirs = [
    "/Users/scui2/Desktop/FredHutch_Colorectal/CARD_Results_Regions",
    "/Users/scui2/Desktop/CARD_Results_Regions"
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
    index=Final_combined_expression_lognormalized.index,
    columns=['Cancer_Cells_Proportion', 'Normal_Epithelial_Proportion', 'Stromal_Proportion', 'T_Cells_Proportion', 'Other_Immune_Proportion']
)
cell_proportions_df = cell_proportions_df.fillna(0.0)

print(f"Initialized cell proportions DataFrame: {cell_proportions_df.shape}")

# Extract unique sample IDs from barcode index
sample_ids = ["_".join(item[1:]) for item in Final_combined_expression_lognormalized.index.str.split("_")]
unique_samples = sorted(list(set(sample_ids)))
print(f"Processing {len(unique_samples)} unique samples...")

# Process each sample
matched_barcodes = 0
total_barcodes = len(Final_combined_expression_lognormalized.index)

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
    for expr_barcode in Final_combined_expression_lognormalized.index:
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



### Filter to keep only matched barcodes (intersected)

# Find barcodes that have non-zero cell proportions (matched barcodes)
cell_prop_sums = cell_proportions_df.sum(axis=1)
matched_barcode_mask = cell_prop_sums > 0.0
matched_barcodes_list = cell_proportions_df.index[matched_barcode_mask]

print(f"Barcodes with cell proportions: {len(matched_barcodes_list)}")
print(f"Barcodes without cell proportions: {len(cell_proportions_df) - len(matched_barcodes_list)}")

# Filter expression data to keep only matched barcodes
Final_combined_expression_normalized_log_filtered = Final_combined_expression_lognormalized.loc[matched_barcodes_list]
cell_proportions_df_filtered = cell_proportions_df.loc[matched_barcodes_list]

print(f"\nFiltered expression data shape: {Final_combined_expression_normalized_log_filtered.shape}")
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




### Marker genes refinement
with open('/Users/scui2/Desktop/scRNAseq_data/final_marker_genes_dict.json', 'r') as f:
    final_marker_genes_dict = json.load(f)

del final_marker_genes_dict["Cancer"]
del final_marker_genes_dict["Normal Epithelia"]



InputDf_for_CARD_SelectedGenes = pd.read_csv('/Users/scui2/Desktop/scRNAseq_data/InputDf_for_CARD_SelectedGenes.csv', index_col=0)
InputDf_for_CARD_meta = pd.read_csv('/Users/scui2/Desktop/scRNAseq_data/InputDf_for_CARD_meta.csv', index_col=0)
InputDf_for_CARD_meta["Cell Type"][InputDf_for_CARD_meta["Cell Type"] == "CD4+ T"] = "T"
InputDf_for_CARD_meta["Cell Type"][InputDf_for_CARD_meta["Cell Type"] == "CD8+ T"] = "T"
InputDf_for_CARD_meta["Cell Type"][InputDf_for_CARD_meta["Cell Type"] == "ASC I"] = "Cancer_Cells"
InputDf_for_CARD_meta["Cell Type"][InputDf_for_CARD_meta["Cell Type"] == "ASC II"] = "Cancer_Cells"
InputDf_for_CARD_meta["Cell Type"][InputDf_for_CARD_meta["Cell Type"] == "ASC III"] = "Cancer_Cells"
InputDf_for_CARD_meta["Cell Type"][InputDf_for_CARD_meta["Cell Type"] == "CSC I"] = "Cancer_Cells"
InputDf_for_CARD_meta["Cell Type"][InputDf_for_CARD_meta["Cell Type"] == "CSC II"] = "Cancer_Cells"
InputDf_for_CARD_meta["Cell Type"][InputDf_for_CARD_meta["Cell Type"] == "CSC III"] = "Cancer_Cells"
InputDf_for_CARD_meta["Cell Type"][InputDf_for_CARD_meta["Cell Type"] == "CSC IV"] = "Cancer_Cells"
InputDf_for_CARD_meta["Cell Type"][InputDf_for_CARD_meta["Cell Type"] == "SSC I"] = "Cancer_Cells"
InputDf_for_CARD_meta["Cell Type"][InputDf_for_CARD_meta["Cell Type"] == "ABS"] = "Normal_Epithelial"
InputDf_for_CARD_meta["Cell Type"][InputDf_for_CARD_meta["Cell Type"] == "CT"] = "Normal_Epithelial"
InputDf_for_CARD_meta["Cell Type"][InputDf_for_CARD_meta["Cell Type"] == "EE"] = "Normal_Epithelial"
InputDf_for_CARD_meta["Cell Type"][InputDf_for_CARD_meta["Cell Type"] == "TUF"] = "Normal_Epithelial"
InputDf_for_CARD_meta["Cell Type"][InputDf_for_CARD_meta["Cell Type"] == "FIB"] = "Stromal"
InputDf_for_CARD_meta["Cell Type"][InputDf_for_CARD_meta["Cell Type"] == "END"] = "Stromal"
InputDf_for_CARD_meta["Cell Type"][InputDf_for_CARD_meta["Cell Type"] == "PLA"] = "Other_Immune"
InputDf_for_CARD_meta["Cell Type"][InputDf_for_CARD_meta["Cell Type"] == "MAS"] = "Other_Immune"
InputDf_for_CARD_meta["Cell Type"][InputDf_for_CARD_meta["Cell Type"] == "MYE"] = "Other_Immune"
InputDf_for_CARD_meta["Cell Type"][InputDf_for_CARD_meta["Cell Type"] == "B"] = "Other_Immune"


InputDF_CounNorm_SelectedGenes = InputDf_for_CARD_SelectedGenes.T.div(InputDf_for_CARD_SelectedGenes.T.sum(axis=1), axis=0)

  ## ranking marker genes based on the log2FC for each cell type
final_ranked_gene_dict = {}
for cell_type in InputDf_for_CARD_meta["Cell Type"].unique():
    
    temp_data = []
    
    for gene in InputDF_CounNorm_SelectedGenes.columns:
        target_cell_GE_mean = InputDF_CounNorm_SelectedGenes.loc[InputDf_for_CARD_meta["Cell Type"] == cell_type, gene].mean()
        non_target_cell_GE_mean = InputDF_CounNorm_SelectedGenes.loc[InputDf_for_CARD_meta["Cell Type"] != cell_type, gene].mean()
        log2FC = np.log2(target_cell_GE_mean+1e-8) - np.log2(non_target_cell_GE_mean+1e-8)
        
        if log2FC >= 1:
            temp_data.append({"Gene": gene, "log2FC": log2FC})
    
    unsorted_df = pd.DataFrame(temp_data, columns=["Gene", "log2FC"])
    sorted_df = unsorted_df.sort_values(by="log2FC", ascending=False)
    selected_genes = sorted_df["Gene"].tolist()
    final_ranked_gene_dict[cell_type] = selected_genes
    print(cell_type, len(final_ranked_gene_dict[cell_type]))




marker_genes_clean_dict = {}
for key, value in final_ranked_gene_dict.items():
    marker_genes_clean_dict[key] = list(set(value).intersection(set(Final_combined_expression_lognormalized.columns)))


# Find all genes and their occurrence counts
all_genes = []
for genes in marker_genes_clean_dict.values():
    all_genes.extend(genes)

from collections import Counter
gene_counts = Counter(all_genes)
duplicate_genes = [gene for gene, count in gene_counts.items() if count > 1]

marker_genes_clean_dict_unique = {}
for cell_type, genes in marker_genes_clean_dict.items():
    unique_genes = [gene for gene in genes if gene not in duplicate_genes]
    marker_genes_clean_dict_unique[cell_type] = unique_genes

marker_genes_clean_dict_unique_top30 = {}
for key, value in marker_genes_clean_dict_unique.items():
    marker_genes_clean_dict_unique_top30[key] = value[:30]


marker_genes_clean_dict_unique_top30["Cancer_Cells"]
marker_genes_clean_dict_unique_top30["Normal_Epithelial"]
marker_genes_clean_dict_unique_top30["T"]
marker_genes_clean_dict_unique_top30["Other_Immune"]
marker_genes_clean_dict_unique_top30["Stromal"]




highly_variable_genes = Final_combined_expression_normalized_log_filtered.std(axis=0).sort_values(ascending=False).index.tolist()
highly_variable_genes_clean = []
all_marker_genes = set()
for key, value in final_ranked_gene_dict.items():
    all_marker_genes.update(value)
for gene in highly_variable_genes:
    if gene not in all_marker_genes:
        highly_variable_genes_clean.append(gene)


highly_variable_genes_clean_top30 = highly_variable_genes_clean[:30]

Predictor_genes = marker_genes_clean_dict_unique_top30["Cancer_Cells"] + marker_genes_clean_dict_unique_top30["Normal_Epithelial"] + marker_genes_clean_dict_unique_top30["T"] + marker_genes_clean_dict_unique_top30["Other_Immune"] + marker_genes_clean_dict_unique_top30["Stromal"] + highly_variable_genes_clean_top30

Final_combined_expression_normalized_log_filtered_Predictor_genes = Final_combined_expression_normalized_log_filtered[Predictor_genes]



Final_combined_expression_normalized_log_filtered_Predictor_genes.to_csv(os.path.join(output_dir, "Final_combined_expression_normalized_log_filtered_Predictor_genes.csv"))




### Merge and deduplicate training features from different cell types (Virchow2)


features_dir = "Colorectal_Cancer_HE_patches/Training_features"
feature_files = [
    "Cancer Cells_training_precomputed_features_Virchow2.pt",
    "Normal Epithelial Cells_training_precomputed_features_Virchow2.pt", 
    "Other Immune Cells_training_precomputed_features_Virchow2.pt",
    "Stromal Cells_training_precomputed_features_Virchow2.pt",
    "T Cells_training_precomputed_features_Virchow2.pt"
]

print(f"Loading features from {len(feature_files)} files...")

# Load all feature files and collect unique tiles
all_tile_ids = set()
all_data = {}
for i, file_name in enumerate(feature_files):
    file_path = os.path.join(features_dir, file_name)

    data = torch.load(file_path, weights_only=False)
    
    # Extract information
    features = data['embeddings']  # [n_tiles, n_features]
    individual_ids = data['individual_ids']
    tile_ids = data['tile_ids']
    

    # Store data for each unique tile
    for j, tile_id in enumerate(tile_ids):
        if tile_id not in all_data:
            all_data[tile_id] = {
                'features': features[j],
                'individual_id': individual_ids[j],
            }
            all_tile_ids.add(tile_id)


# Convert back to arrays
unique_tile_ids = sorted(list(all_tile_ids))
combined_features = []
combined_individual_ids = []
combined_celltype_proportions = None
combined_marker_genes = None

for tile_id in unique_tile_ids:
    tile_data = all_data[tile_id]
    combined_features.append(tile_data['features'])
    combined_individual_ids.append(tile_data['individual_id'])
    
combined_features = torch.stack(combined_features)



# Create mapping from tile_id to expression data index
expr_tile_ids = [idx for idx in Final_combined_expression_normalized_log_filtered_Predictor_genes.index]
expr_tile_ids_set = set(expr_tile_ids)

# Find common tiles
common_tile_ids = []
common_indices_features = []
common_indices_expr = []

for i, tile_id in enumerate(unique_tile_ids):
    if tile_id in expr_tile_ids_set:
        common_tile_ids.append(tile_id)
        common_indices_features.append(i)
        expr_idx = expr_tile_ids.index(tile_id)
        common_indices_expr.append(expr_idx)


# Extract common data
final_features = combined_features[common_indices_features]
final_individual_ids = [combined_individual_ids[i] for i in common_indices_features]
final_tile_ids = common_tile_ids

# Extract corresponding expression data (180 predictor genes)
final_expression_180genes = Final_combined_expression_normalized_log_filtered_Predictor_genes.iloc[common_indices_expr]

# Extract corresponding cell proportions
final_cell_proportions = cell_proportions_df_filtered.iloc[common_indices_expr]

# Create combined dataset
combined_dataset = {
    'tile_ids': final_tile_ids,
    'individual_ids': final_individual_ids,
    'features': final_features,  # Virchow2 features
    'expression_180genes': torch.tensor(final_expression_180genes.values, dtype=torch.float32),
    'expression_180genes_names': list(final_expression_180genes.columns),
    'cell_proportions': torch.tensor(final_cell_proportions.values, dtype=torch.float32),
    'cell_proportion_names': list(final_cell_proportions.columns)
}

len(combined_dataset['tile_ids'])
len(combined_dataset['individual_ids'])
len(combined_dataset['features'])
len(combined_dataset['expression_180genes'])
len(combined_dataset['cell_proportions'])
len(combined_dataset['expression_180genes_names'])
len(combined_dataset['cell_proportion_names'])



# Save combined dataset
output_file = os.path.join(output_dir, "combined_features_Predictor_genes.pt")
torch.save(combined_dataset, output_file)
print(f"\nSaved combined dataset to: {output_file}")



### XGBoost Training for 180 Predictor Genes

print("\n" + "="*80)
print("XGBOOST TRAINING FOR 180 PREDICTOR GENES")
print("="*80)

# Load combined dataset
combined_dataset = torch.load(output_file)
combined_dataset.keys()
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Extract data from combined dataset
features = combined_dataset['features']  # Virchow2 features
individual_ids = combined_dataset['individual_ids']
tile_ids = combined_dataset['tile_ids'] 
expression_180genes = combined_dataset['expression_300genes']  # 300 predictor genes
expression_180genes_names = combined_dataset['expression_300genes_names']
cell_proportions = combined_dataset['cell_proportions']  # 5 cell type proportions
cell_proportion_names = combined_dataset['cell_proportion_names']

# Exclude specific individuals
excluded_individual = ["SU-15-27301-B1", "SU-16-02468-B1"] 
valid_indices = [i for i, ind_id in enumerate(individual_ids) if ind_id not in excluded_individual]

# Filter all data to exclude these individuals
features = features[valid_indices]
individual_ids = [individual_ids[i] for i in valid_indices]
tile_ids = [tile_ids[i] for i in valid_indices]
expression_180genes = expression_180genes[valid_indices]
cell_proportions = cell_proportions[valid_indices]

print(f"After excluding individuals:")
print(f"  Features shape: {features.shape}")
print(f"  Number of tiles: {len(tile_ids)}")
print(f"  Number of individuals: {len(set(individual_ids))}")

# Get unique individuals for cross-validation
unique_individuals = sorted(list(set(individual_ids)))
print(f"Unique individuals: {len(unique_individuals)}")

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

# CLR transformation function for proportions
def clr_transform(proportions):
    """
    Apply centered log-ratio (CLR) transformation to proportions
    CLR(x) = log(x_i / geometric_mean(x))
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-6
    proportions_adj = proportions + epsilon
    
    # Calculate geometric mean
    geometric_mean = np.exp(np.mean(np.log(proportions_adj), axis=1, keepdims=True))
    
    # Apply CLR transformation
    clr_props = np.log(proportions_adj / geometric_mean)
    
    return clr_props

# Function to calculate partial correlation controlling for CLR-transformed cell proportions
def partial_correlation_clr(x, y, cell_props):
    """
    Calculate partial correlation between x and y, controlling for CLR-transformed cell proportions
    """
    # Apply CLR transformation to cell proportions
    clr_props = clr_transform(cell_props.numpy() if torch.is_tensor(cell_props) else cell_props)
    
    # Regress x on CLR-transformed proportions, get residuals
    reg_x = LinearRegression().fit(clr_props, x)
    residual_x = x - reg_x.predict(clr_props)
    
    # Regress y on CLR-transformed proportions, get residuals  
    reg_y = LinearRegression().fit(clr_props, y)
    residual_y = y - reg_y.predict(clr_props)
    
    # Correlation between residuals = partial correlation
    if len(residual_x) > 1:
        partial_corr, p_value = pearsonr(residual_x, residual_y)
        return partial_corr if not np.isnan(partial_corr) else 0.0
    else:
        return 0.0

# Initialize results DataFrames and check for existing results
regular_output_path = os.path.join(output_dir, "180genes_regular_correlations.csv")
partial_output_path = os.path.join(output_dir, "180genes_partial_correlations_clr.csv")

# Try to load existing results to resume from interruption
if os.path.exists(regular_output_path) and os.path.exists(partial_output_path):
    print("Found existing result files, loading...")
    results_df_regular = pd.read_csv(regular_output_path, index_col=0)
    results_df_partial = pd.read_csv(partial_output_path, index_col=0)
    
    # Check which genes are already completed (non-null values)
    completed_genes = []
    for gene in expression_180genes_names:
        if gene in results_df_regular.columns:
            if not results_df_regular[gene].isna().all():  # If not all values are NaN
                completed_genes.append(gene)
    
    print(f"Found {len(completed_genes)} already completed genes")
    print(f"Resuming from gene {len(completed_genes) + 1}")
else:
    print("No existing result files found, starting fresh...")
    results_df_regular = pd.DataFrame(index=unique_individuals, columns=expression_180genes_names)
    results_df_partial = pd.DataFrame(index=unique_individuals, columns=expression_180genes_names)
    completed_genes = []

print(f"\nStarting leave-one-individual-out cross-validation for {len(expression_180genes_names)} genes...")
print(f"Will process {len(expression_180genes_names) - len(completed_genes)} remaining genes...")

# Main XGBoost training loop for each gene
for gene_idx, gene_name in enumerate(expression_180genes_names):
    # Skip if gene is already completed
    if gene_name in completed_genes:
        print(f"\nSkipping gene {gene_idx + 1}/{len(expression_180genes_names)}: {gene_name} (already completed)")
        continue
        
    print(f"\nProcessing gene {gene_idx + 1}/{len(expression_180genes_names)}: {gene_name}")
    
    gene_predictions = {}
    gene_actuals = {}
    
    # Leave-one-individual-out cross-validation
    for test_individual in unique_individuals:
        # Split data
        train_indices = [i for i, ind_id in enumerate(individual_ids) if ind_id != test_individual]
        test_indices = [i for i, ind_id in enumerate(individual_ids) if ind_id == test_individual]
        
        if len(test_indices) == 0:
            continue
            
        # Get train/test data
        X_train = features[train_indices]
        y_train = expression_180genes[train_indices, gene_idx]
        X_test = features[test_indices]
        y_test = expression_180genes[test_indices, gene_idx]
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train.numpy(), label=y_train.numpy())
        dtest = xgb.DMatrix(X_test.numpy(), label=y_test.numpy())
        
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
        gene_actuals[test_individual] = y_test.numpy()
    
    # Calculate correlations for each individual
    for individual in unique_individuals:
        if individual in gene_predictions and individual in gene_actuals:
            pred_vals = gene_predictions[individual]
            actual_vals = gene_actuals[individual]
            
            # Get indices for this individual's samples
            individual_indices = [i for i, ind_id in enumerate(individual_ids) if ind_id == individual]
            
            if len(pred_vals) > 1 and len(individual_indices) > 1:  # Need at least 2 points for correlation
                # Regular Pearson correlation
                regular_corr, _ = pearsonr(actual_vals, pred_vals)
                results_df_regular.loc[individual, gene_name] = regular_corr if not np.isnan(regular_corr) else 0.0
                
                # Partial correlation controlling for CLR-transformed cell proportions
                individual_cell_props = cell_proportions[individual_indices]
                
                if len(individual_cell_props) == len(pred_vals):
                    partial_corr = partial_correlation_clr(actual_vals, pred_vals, individual_cell_props)
                    results_df_partial.loc[individual, gene_name] = partial_corr
                else:
                    results_df_partial.loc[individual, gene_name] = 0.0
            else:
                results_df_regular.loc[individual, gene_name] = 0.0
                results_df_partial.loc[individual, gene_name] = 0.0
        else:
            results_df_regular.loc[individual, gene_name] = np.nan
            results_df_partial.loc[individual, gene_name] = np.nan
    
    # Save results after each gene (for crash recovery)
    results_df_regular.to_csv(regular_output_path)
    results_df_partial.to_csv(partial_output_path)
    
    # Print progress after each gene
    print(f"  Completed gene {gene_idx + 1}/{len(expression_180genes_names)}: {gene_name} - Results saved")




### ResNet50 Feature Processing and Simplified CV for 180 Predictor Genes

print("\n" + "="*80)
print("RESNET50 FEATURE PROCESSING AND SIMPLIFIED CV FOR 180 PREDICTOR GENES")
print("="*80)

# CLR transformation function for proportions (needed for partial correlation)
def clr_transform_resnet50(proportions):
    """
    Apply centered log-ratio (CLR) transformation to proportions
    CLR(x) = log(x_i / geometric_mean(x))
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-6
    proportions_adj = proportions + epsilon
    
    # Calculate geometric mean
    geometric_mean = np.exp(np.mean(np.log(proportions_adj), axis=1, keepdims=True))
    
    # Apply CLR transformation
    clr_props = np.log(proportions_adj / geometric_mean)
    
    return clr_props

# Function to calculate partial correlation controlling for CLR-transformed cell proportions
def partial_correlation_clr_resnet50(x, y, cell_props):
    """
    Calculate partial correlation between x and y, controlling for CLR-transformed cell proportions
    """
    # Apply CLR transformation to cell proportions
    clr_props = clr_transform_resnet50(cell_props.numpy() if torch.is_tensor(cell_props) else cell_props)
    
    # Regress x on CLR-transformed proportions, get residuals
    reg_x = LinearRegression().fit(clr_props, x)
    residual_x = x - reg_x.predict(clr_props)
    
    # Regress y on CLR-transformed proportions, get residuals  
    reg_y = LinearRegression().fit(clr_props, y)
    residual_y = y - reg_y.predict(clr_props)
    
    # Correlation between residuals = partial correlation
    if len(residual_x) > 1:
        partial_corr, p_value = pearsonr(residual_x, residual_y)
        return partial_corr if not np.isnan(partial_corr) else 0.0
    else:
        return 0.0

# Load ResNet50 features
features_dir = "Colorectal_Cancer_HE_patches/Training_features"
feature_files_resnet50 = [
    "Cancer Cells_training_precomputed_features_ResNet50.pt",
    "Normal Epithelial Cells_training_precomputed_features_ResNet50.pt", 
    "Other Immune Cells_training_precomputed_features_ResNet50.pt",
    "Stromal Cells_training_precomputed_features_ResNet50.pt",
    "T Cells_training_precomputed_features_ResNet50.pt"
]

print(f"Loading ResNet50 features from {len(feature_files_resnet50)} files...")

# Load all ResNet50 feature files and collect unique tiles
all_tile_ids_resnet50 = set()
all_data_resnet50 = {}
for i, file_name in enumerate(feature_files_resnet50):
    file_path = os.path.join(features_dir, file_name)

    data = torch.load(file_path, weights_only=False)
    
    # Extract information
    features = data['embeddings']  # [n_tiles, n_features]
    individual_ids = data['individual_ids']
    tile_ids = data['tile_ids']
    

    # Store data for each unique tile
    for j, tile_id in enumerate(tile_ids):
        if tile_id not in all_data_resnet50:
            all_data_resnet50[tile_id] = {
                'features': features[j],
                'individual_id': individual_ids[j],
            }
            all_tile_ids_resnet50.add(tile_id)


# Convert back to arrays
unique_tile_ids_resnet50 = sorted(list(all_tile_ids_resnet50))
combined_features_resnet50 = []
combined_individual_ids_resnet50 = []

for tile_id in unique_tile_ids_resnet50:
    tile_data = all_data_resnet50[tile_id]
    combined_features_resnet50.append(tile_data['features'])
    combined_individual_ids_resnet50.append(tile_data['individual_id'])
    
combined_features_resnet50 = torch.stack(combined_features_resnet50)

# Load Virchow2's combined_dataset to get the processed data
virchow2_combined_dataset = torch.load(os.path.join(output_dir, "combined_features_Predictor_genes.pt"))

# Extract common data by matching with Virchow2's tiles
final_features_resnet50 = []
final_individual_ids_resnet50 = []
final_tile_ids_resnet50 = []

# Create mapping for ResNet50 tiles
resnet50_tile_to_idx = {tile_id: i for i, tile_id in enumerate(unique_tile_ids_resnet50)}

# Use Virchow2's tile order and find matching ResNet50 features
for i, virchow2_tile_id in enumerate(virchow2_combined_dataset['tile_ids']):
    if virchow2_tile_id in resnet50_tile_to_idx:
        resnet50_idx = resnet50_tile_to_idx[virchow2_tile_id]
        final_features_resnet50.append(combined_features_resnet50[resnet50_idx])
        final_individual_ids_resnet50.append(combined_individual_ids_resnet50[resnet50_idx])
        final_tile_ids_resnet50.append(virchow2_tile_id)

final_features_resnet50 = torch.stack(final_features_resnet50)

# Create combined dataset for ResNet50 (using Virchow2's expression data)
# final_features_resnet50 is already aligned with Virchow2's tile order
combined_dataset_resnet50 = {
    'tile_ids': final_tile_ids_resnet50,
    'individual_ids': final_individual_ids_resnet50,
    'features': final_features_resnet50,  # ResNet50 features
    'expression_300genes': virchow2_combined_dataset['expression_300genes'][:len(final_tile_ids_resnet50)],  # Virchow2's expression data
    'expression_300genes_names': virchow2_combined_dataset['expression_300genes_names'],
    'cell_proportions': virchow2_combined_dataset['cell_proportions'][:len(final_tile_ids_resnet50)],
    'cell_proportion_names': virchow2_combined_dataset['cell_proportion_names']
}


# Save ResNet50 combined dataset
output_file_resnet50 = os.path.join(output_dir, "combined_features_Predictor_genes_ResNet50.pt")
torch.save(combined_dataset_resnet50, output_file_resnet50)
print(f"\nSaved ResNet50 combined dataset to: {output_file_resnet50}")

# Load Virchow2 LOIO results to find median correlation samples
print("\nLoading Virchow2 LOIO results...")
virchow2_regular_path = os.path.join(output_dir, "180genes_regular_correlations.csv")


virchow2_regular_df = pd.read_csv(virchow2_regular_path, index_col=0)
# Only take first 180 columns (180 genes)
virchow2_regular_df = virchow2_regular_df.iloc[:, :180]
print(f"Loaded Virchow2 regular correlations (first 180 genes): {virchow2_regular_df.shape}")

# Find median correlation sample for each gene
median_samples_regular = {}
for gene in virchow2_regular_df.columns:
    gene_correlations = virchow2_regular_df[gene].dropna()
    if len(gene_correlations) > 0:
        median_corr = gene_correlations.median()
        # Find the sample closest to median
        closest_sample = (gene_correlations - median_corr).abs().idxmin()
        median_samples_regular[gene] = closest_sample

print(f"Found median correlation samples for {len(median_samples_regular)} genes")


# Extract ResNet50 data from combined dataset
features_resnet50 = combined_dataset_resnet50['features']  # ResNet50 features
individual_ids_resnet50 = combined_dataset_resnet50['individual_ids']
tile_ids_resnet50 = combined_dataset_resnet50['tile_ids'] 
expression_300genes_resnet50 = combined_dataset_resnet50['expression_300genes']  # 300 predictor genes (from Virchow2)
expression_300genes_names_resnet50 = combined_dataset_resnet50['expression_300genes_names']
cell_proportions_resnet50 = combined_dataset_resnet50['cell_proportions']  # 5 cell type proportions
cell_proportion_names_resnet50 = combined_dataset_resnet50['cell_proportion_names']

# Exclude specific individuals
excluded_individual = ["SU-15-27301-B1", "SU-16-02468-B1"] 
valid_indices_resnet50 = [i for i, ind_id in enumerate(individual_ids_resnet50) if ind_id not in excluded_individual]

# Filter all data to exclude these individuals
features_resnet50 = features_resnet50[valid_indices_resnet50]
individual_ids_resnet50 = [individual_ids_resnet50[i] for i in valid_indices_resnet50]
tile_ids_resnet50 = [tile_ids_resnet50[i] for i in valid_indices_resnet50]
expression_300genes_resnet50 = expression_300genes_resnet50[valid_indices_resnet50]
cell_proportions_resnet50 = cell_proportions_resnet50[valid_indices_resnet50]

print(f"ResNet50 data after excluding individuals:")
print(f"  Features shape: {features_resnet50.shape}")
print(f"  Number of tiles: {len(tile_ids_resnet50)}")
print(f"  Number of individuals: {len(set(individual_ids_resnet50))}")

# Get unique individuals for cross-validation
unique_individuals_resnet50 = sorted(list(set(individual_ids_resnet50)))
print(f"Unique individuals: {len(unique_individuals_resnet50)}")

# Initialize results for ResNet50 (1x180 table for regular and partial correlations)
resnet50_results_regular = []
resnet50_results_partial = []

# XGBoost parameters (same as Virchow2)
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

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


print(f"\nStarting simplified CV for ResNet50 with {len(expression_300genes_names_resnet50)} genes...")

# Simplified CV loop for each gene
for gene_idx, gene_name in enumerate(expression_300genes_names_resnet50):
    print(f"\nProcessing gene {gene_idx + 1}/{len(expression_300genes_names_resnet50)}: {gene_name}")
    
    # Find the median correlation sample for this gene from Virchow2 results
    if gene_name in median_samples_regular:
        target_individual = median_samples_regular[gene_name]
        print(f"  Using median correlation sample: {target_individual}")
    else:
        print(f"  No median sample found for {gene_name}, skipping...")
        resnet50_results_regular.append(0.0)
        resnet50_results_partial.append(0.0)
        continue
    
    # Check if target individual exists in ResNet50 data
    if target_individual not in individual_ids_resnet50:
        print(f"  Target individual {target_individual} not found in ResNet50 data, skipping...")
        resnet50_results_regular.append(0.0)
        resnet50_results_partial.append(0.0)
        continue
    
    # Split data: leave out target individual
    train_indices = [i for i, ind_id in enumerate(individual_ids_resnet50) if ind_id != target_individual]
    test_indices = [i for i, ind_id in enumerate(individual_ids_resnet50) if ind_id == target_individual]
    
    if len(test_indices) == 0:
        print(f"  No test samples for {target_individual}, skipping...")
        resnet50_results_regular.append(0.0)
        resnet50_results_partial.append(0.0)
        continue
        
    # Get train/test data
    X_train = features_resnet50[train_indices]
    y_train = expression_300genes_resnet50[train_indices, gene_idx]
    X_test = features_resnet50[test_indices]
    y_test = expression_300genes_resnet50[test_indices, gene_idx]
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train.numpy(), label=y_train.numpy())
    dtest = xgb.DMatrix(X_test.numpy(), label=y_test.numpy())
    
    # Train model using same parameters as Virchow2
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=num_boost_round,
        verbose_eval=False
    )
    
    # Make predictions
    y_pred = model.predict(dtest)
    
    # Calculate regular correlation
    if len(y_test) > 1:
        regular_correlation, _ = pearsonr(y_test.numpy(), y_pred)
        regular_correlation = regular_correlation if not np.isnan(regular_correlation) else 0.0
        
        # Calculate partial correlation controlling for cell proportions
        test_cell_props = cell_proportions_resnet50[test_indices]
        if len(test_cell_props) == len(y_pred):
            partial_correlation = partial_correlation_clr_resnet50(y_test.numpy(), y_pred, test_cell_props)
        else:
            partial_correlation = 0.0
    else:
        regular_correlation = 0.0
        partial_correlation = 0.0
    
    resnet50_results_regular.append(regular_correlation)
    resnet50_results_partial.append(partial_correlation)
    print(f"  Regular Correlation: {regular_correlation:.4f}, Partial Correlation: {partial_correlation:.4f}")

# Save ResNet50 results as 1x300 tables (regular and partial correlations)
resnet50_results_regular_df = pd.DataFrame([resnet50_results_regular], columns=expression_300genes_names_resnet50, index=['ResNet50'])
resnet50_results_partial_df = pd.DataFrame([resnet50_results_partial], columns=expression_300genes_names_resnet50, index=['ResNet50'])

resnet50_regular_output_path = os.path.join(output_dir, "180genes_ResNet50_regular_correlations.csv")
resnet50_partial_output_path = os.path.join(output_dir, "180genes_ResNet50_partial_correlations_clr.csv")

resnet50_results_regular_df.to_csv(resnet50_regular_output_path)
resnet50_results_partial_df.to_csv(resnet50_partial_output_path)

print(f"\nSaved ResNet50 regular correlations to: {resnet50_regular_output_path}")
print(f"Saved ResNet50 partial correlations to: {resnet50_partial_output_path}")




### UNI2h Feature Processing and Simplified CV for 180 Predictor Genes

print("\n" + "="*80)
print("UNI2H FEATURE PROCESSING AND SIMPLIFIED CV FOR 180 PREDICTOR GENES")
print("="*80)

# CLR transformation function for proportions (needed for partial correlation)
def clr_transform_uni2h(proportions):
    """
    Apply centered log-ratio (CLR) transformation to proportions
    CLR(x) = log(x_i / geometric_mean(x))
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-6
    proportions_adj = proportions + epsilon
    
    # Calculate geometric mean
    geometric_mean = np.exp(np.mean(np.log(proportions_adj), axis=1, keepdims=True))
    
    # Apply CLR transformation
    clr_props = np.log(proportions_adj / geometric_mean)
    
    return clr_props

# Function to calculate partial correlation controlling for CLR-transformed cell proportions
def partial_correlation_clr_uni2h(x, y, cell_props):
    """
    Calculate partial correlation between x and y, controlling for CLR-transformed cell proportions
    """
    # Apply CLR transformation to cell proportions
    clr_props = clr_transform_uni2h(cell_props.numpy() if torch.is_tensor(cell_props) else cell_props)
    
    # Regress x on CLR-transformed proportions, get residuals
    reg_x = LinearRegression().fit(clr_props, x)
    residual_x = x - reg_x.predict(clr_props)
    
    # Regress y on CLR-transformed proportions, get residuals  
    reg_y = LinearRegression().fit(clr_props, y)
    residual_y = y - reg_y.predict(clr_props)
    
    # Correlation between residuals = partial correlation
    if len(residual_x) > 1:
        partial_corr, p_value = pearsonr(residual_x, residual_y)
        return partial_corr if not np.isnan(partial_corr) else 0.0
    else:
        return 0.0

# Load UNI2h features
features_dir = "Colorectal_Cancer_HE_patches/Training_features"
feature_files_uni2h = [
    "Cancer Cells_training_precomputed_features_UNI2h.pt",
    "Normal Epithelial Cells_training_precomputed_features_UNI2h.pt", 
    "Other Immune Cells_training_precomputed_features_UNI2h.pt",
    "Stromal Cells_training_precomputed_features_UNI2h.pt",
    "T Cells_training_precomputed_features_UNI2h.pt"
]

print(f"Loading UNI2h features from {len(feature_files_uni2h)} files...")

# Load all UNI2h feature files and collect unique tiles
all_tile_ids_uni2h = set()
all_data_uni2h = {}
for i, file_name in enumerate(feature_files_uni2h):
    file_path = os.path.join(features_dir, file_name)

    data = torch.load(file_path, weights_only=False)
    
    # Extract information
    features = data['embeddings']  # [n_tiles, n_features]
    individual_ids = data['individual_ids']
    tile_ids = data['tile_ids']
    

    # Store data for each unique tile
    for j, tile_id in enumerate(tile_ids):
        if tile_id not in all_data_uni2h:
            all_data_uni2h[tile_id] = {
                'features': features[j],
                'individual_id': individual_ids[j],
            }
            all_tile_ids_uni2h.add(tile_id)


# Convert back to arrays
unique_tile_ids_uni2h = sorted(list(all_tile_ids_uni2h))
combined_features_uni2h = []
combined_individual_ids_uni2h = []

for tile_id in unique_tile_ids_uni2h:
    tile_data = all_data_uni2h[tile_id]
    combined_features_uni2h.append(tile_data['features'])
    combined_individual_ids_uni2h.append(tile_data['individual_id'])
    
combined_features_uni2h = torch.stack(combined_features_uni2h)

# Extract common data by matching with Virchow2's tiles
final_features_uni2h = []
final_individual_ids_uni2h = []
final_tile_ids_uni2h = []

# Create mapping for UNI2h tiles
uni2h_tile_to_idx = {tile_id: i for i, tile_id in enumerate(unique_tile_ids_uni2h)}

# Use Virchow2's tile order and find matching UNI2h features
for i, virchow2_tile_id in enumerate(virchow2_combined_dataset['tile_ids']):
    if virchow2_tile_id in uni2h_tile_to_idx:
        uni2h_idx = uni2h_tile_to_idx[virchow2_tile_id]
        final_features_uni2h.append(combined_features_uni2h[uni2h_idx])
        final_individual_ids_uni2h.append(combined_individual_ids_uni2h[uni2h_idx])
        final_tile_ids_uni2h.append(virchow2_tile_id)

final_features_uni2h = torch.stack(final_features_uni2h)

# Create combined dataset for UNI2h (using Virchow2's expression data)
# final_features_uni2h is already aligned with Virchow2's tile order
combined_dataset_uni2h = {
    'tile_ids': final_tile_ids_uni2h,
    'individual_ids': final_individual_ids_uni2h,
    'features': final_features_uni2h,  # UNI2h features
    'expression_300genes': virchow2_combined_dataset['expression_300genes'][:len(final_tile_ids_uni2h)],  # Virchow2's expression data
    'expression_300genes_names': virchow2_combined_dataset['expression_300genes_names'],
    'cell_proportions': virchow2_combined_dataset['cell_proportions'][:len(final_tile_ids_uni2h)],
    'cell_proportion_names': virchow2_combined_dataset['cell_proportion_names']
}

# Save UNI2h combined dataset
output_file_uni2h = os.path.join(output_dir, "combined_features_Predictor_genes_UNI2h.pt")
torch.save(combined_dataset_uni2h, output_file_uni2h)
print(f"\nSaved UNI2h combined dataset to: {output_file_uni2h}")

# Extract UNI2h data from combined dataset
features_uni2h = combined_dataset_uni2h['features']  # UNI2h features
individual_ids_uni2h = combined_dataset_uni2h['individual_ids']
tile_ids_uni2h = combined_dataset_uni2h['tile_ids'] 
expression_300genes_uni2h = combined_dataset_uni2h['expression_300genes']  # 300 predictor genes (from Virchow2)
expression_300genes_names_uni2h = combined_dataset_uni2h['expression_300genes_names']
cell_proportions_uni2h = combined_dataset_uni2h['cell_proportions']  # 5 cell type proportions
cell_proportion_names_uni2h = combined_dataset_uni2h['cell_proportion_names']

# Exclude specific individuals
excluded_individual = ["SU-15-27301-B1", "SU-16-02468-B1"] 
valid_indices_uni2h = [i for i, ind_id in enumerate(individual_ids_uni2h) if ind_id not in excluded_individual]

# Filter all data to exclude these individuals
features_uni2h = features_uni2h[valid_indices_uni2h]
individual_ids_uni2h = [individual_ids_uni2h[i] for i in valid_indices_uni2h]
tile_ids_uni2h = [tile_ids_uni2h[i] for i in valid_indices_uni2h]
expression_300genes_uni2h = expression_300genes_uni2h[valid_indices_uni2h]
cell_proportions_uni2h = cell_proportions_uni2h[valid_indices_uni2h]

print(f"UNI2h data after excluding individuals:")
print(f"  Features shape: {features_uni2h.shape}")
print(f"  Number of tiles: {len(tile_ids_uni2h)}")
print(f"  Number of individuals: {len(set(individual_ids_uni2h))}")

# Get unique individuals for cross-validation
unique_individuals_uni2h = sorted(list(set(individual_ids_uni2h)))
print(f"Unique individuals: {len(unique_individuals_uni2h)}")

# Initialize results for UNI2h (1x180 table for regular and partial correlations)
uni2h_results_regular = []
uni2h_results_partial = []

# XGBoost parameters (same as Virchow2)
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

print(f"\nStarting simplified CV for UNI2h with {len(expression_300genes_names_uni2h)} genes...")

# Simplified CV loop for each gene
for gene_idx, gene_name in enumerate(expression_300genes_names_uni2h):
    print(f"\nProcessing gene {gene_idx + 1}/{len(expression_300genes_names_uni2h)}: {gene_name}")
    
    # Find the median correlation sample for this gene from Virchow2 results
    if gene_name in median_samples_regular:
        target_individual = median_samples_regular[gene_name]
        print(f"  Using median correlation sample: {target_individual}")
    else:
        print(f"  No median sample found for {gene_name}, skipping...")
        uni2h_results_regular.append(0.0)
        uni2h_results_partial.append(0.0)
        continue
    
    # Check if target individual exists in UNI2h data
    if target_individual not in individual_ids_uni2h:
        print(f"  Target individual {target_individual} not found in UNI2h data, skipping...")
        uni2h_results_regular.append(0.0)
        uni2h_results_partial.append(0.0)
        continue
    
    # Split data: leave out target individual
    train_indices = [i for i, ind_id in enumerate(individual_ids_uni2h) if ind_id != target_individual]
    test_indices = [i for i, ind_id in enumerate(individual_ids_uni2h) if ind_id == target_individual]
    
    if len(test_indices) == 0:
        print(f"  No test samples for {target_individual}, skipping...")
        uni2h_results_regular.append(0.0)
        uni2h_results_partial.append(0.0)
        continue
        
    # Get train/test data
    X_train = features_uni2h[train_indices]
    y_train = expression_300genes_uni2h[train_indices, gene_idx]
    X_test = features_uni2h[test_indices]
    y_test = expression_300genes_uni2h[test_indices, gene_idx]
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train.numpy(), label=y_train.numpy())
    dtest = xgb.DMatrix(X_test.numpy(), label=y_test.numpy())
    
    # Train model using same parameters as Virchow2
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=num_boost_round,
        verbose_eval=False
    )
    
    # Make predictions
    y_pred = model.predict(dtest)
    
    # Calculate regular correlation
    if len(y_test) > 1:
        regular_correlation, _ = pearsonr(y_test.numpy(), y_pred)
        regular_correlation = regular_correlation if not np.isnan(regular_correlation) else 0.0
        
        # Calculate partial correlation controlling for cell proportions
        test_cell_props = cell_proportions_uni2h[test_indices]
        if len(test_cell_props) == len(y_pred):
            partial_correlation = partial_correlation_clr_uni2h(y_test.numpy(), y_pred, test_cell_props)
        else:
            partial_correlation = 0.0
    else:
        regular_correlation = 0.0
        partial_correlation = 0.0
    
    uni2h_results_regular.append(regular_correlation)
    uni2h_results_partial.append(partial_correlation)
    print(f"  Regular Correlation: {regular_correlation:.4f}, Partial Correlation: {partial_correlation:.4f}")

# Save UNI2h results as 1x300 tables (regular and partial correlations)
uni2h_results_regular_df = pd.DataFrame([uni2h_results_regular], columns=expression_300genes_names_uni2h, index=['UNI2h'])
uni2h_results_partial_df = pd.DataFrame([uni2h_results_partial], columns=expression_300genes_names_uni2h, index=['UNI2h'])

uni2h_regular_output_path = os.path.join(output_dir, "180genes_UNI2h_regular_correlations.csv")
uni2h_partial_output_path = os.path.join(output_dir, "180genes_UNI2h_partial_correlations_clr.csv")

uni2h_results_regular_df.to_csv(uni2h_regular_output_path)
uni2h_results_partial_df.to_csv(uni2h_partial_output_path)

print(f"\nSaved UNI2h regular correlations to: {uni2h_regular_output_path}")
print(f"Saved UNI2h partial correlations to: {uni2h_partial_output_path}")




### Virchow Feature Processing and Simplified CV for 180 Predictor Genes

print("\n" + "="*80)
print("VIRCHOW FEATURE PROCESSING AND SIMPLIFIED CV FOR 180 PREDICTOR GENES")
print("="*80)

# CLR transformation function for proportions (needed for partial correlation)
def clr_transform_virchow(proportions):
    """
    Apply centered log-ratio (CLR) transformation to proportions
    CLR(x) = log(x_i / geometric_mean(x))
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-6
    proportions_adj = proportions + epsilon
    
    # Calculate geometric mean
    geometric_mean = np.exp(np.mean(np.log(proportions_adj), axis=1, keepdims=True))
    
    # Apply CLR transformation
    clr_props = np.log(proportions_adj / geometric_mean)
    
    return clr_props

# Function to calculate partial correlation controlling for CLR-transformed cell proportions
def partial_correlation_clr_virchow(x, y, cell_props):
    """
    Calculate partial correlation between x and y, controlling for CLR-transformed cell proportions
    """
    # Apply CLR transformation to cell proportions
    clr_props = clr_transform_virchow(cell_props.numpy() if torch.is_tensor(cell_props) else cell_props)
    
    # Regress x on CLR-transformed proportions, get residuals
    reg_x = LinearRegression().fit(clr_props, x)
    residual_x = x - reg_x.predict(clr_props)
    
    # Regress y on CLR-transformed proportions, get residuals  
    reg_y = LinearRegression().fit(clr_props, y)
    residual_y = y - reg_y.predict(clr_props)
    
    # Correlation between residuals = partial correlation
    if len(residual_x) > 1:
        partial_corr, p_value = pearsonr(residual_x, residual_y)
        return partial_corr if not np.isnan(partial_corr) else 0.0
    else:
        return 0.0

# Load Virchow features
features_dir = "Colorectal_Cancer_HE_patches/Training_features"
feature_files_virchow = [
    "Cancer Cells_training_precomputed_features_Virchow.pt",
    "Normal Epithelial Cells_training_precomputed_features_Virchow.pt", 
    "Other Immune Cells_training_precomputed_features_Virchow.pt",
    "Stromal Cells_training_precomputed_features_Virchow.pt",
    "T Cells_training_precomputed_features_Virchow.pt"
]

print(f"Loading Virchow features from {len(feature_files_virchow)} files...")

# Load all Virchow feature files and collect unique tiles
all_tile_ids_virchow = set()
all_data_virchow = {}
for i, file_name in enumerate(feature_files_virchow):
    file_path = os.path.join(features_dir, file_name)
    data = torch.load(file_path, weights_only=False)
    
    # Extract information
    features = data['embeddings']  # [n_tiles, n_features]
    individual_ids = data['individual_ids']
    tile_ids = data['tile_ids']
    
    # Store data for each unique tile
    for j, tile_id in enumerate(tile_ids):
        if tile_id not in all_data_virchow:
            all_data_virchow[tile_id] = {
                'features': features[j],
                'individual_id': individual_ids[j],
            }
            all_tile_ids_virchow.add(tile_id)

# Convert back to arrays
unique_tile_ids_virchow = sorted(list(all_tile_ids_virchow))
combined_features_virchow = []
combined_individual_ids_virchow = []

for tile_id in unique_tile_ids_virchow:
    tile_data = all_data_virchow[tile_id]
    combined_features_virchow.append(tile_data['features'])
    combined_individual_ids_virchow.append(tile_data['individual_id'])
    
combined_features_virchow = torch.stack(combined_features_virchow)

# Extract common data by matching with Virchow2's tiles
final_features_virchow = []
final_individual_ids_virchow = []
final_tile_ids_virchow = []

# Create mapping for Virchow tiles
virchow_tile_to_idx = {tile_id: i for i, tile_id in enumerate(unique_tile_ids_virchow)}

# Use Virchow2's tile order and find matching Virchow features
for i, virchow2_tile_id in enumerate(virchow2_combined_dataset['tile_ids']):
    if virchow2_tile_id in virchow_tile_to_idx:
        virchow_idx = virchow_tile_to_idx[virchow2_tile_id]
        final_features_virchow.append(combined_features_virchow[virchow_idx])
        final_individual_ids_virchow.append(combined_individual_ids_virchow[virchow_idx])
        final_tile_ids_virchow.append(virchow2_tile_id)

final_features_virchow = torch.stack(final_features_virchow)

# Create combined dataset for Virchow (using Virchow2's expression data)
combined_dataset_virchow = {
    'tile_ids': final_tile_ids_virchow,
    'individual_ids': final_individual_ids_virchow,
    'features': final_features_virchow,  # Virchow features
    'expression_300genes': virchow2_combined_dataset['expression_300genes'][:len(final_tile_ids_virchow)],
    'expression_300genes_names': virchow2_combined_dataset['expression_300genes_names'],
    'cell_proportions': virchow2_combined_dataset['cell_proportions'][:len(final_tile_ids_virchow)],
    'cell_proportion_names': virchow2_combined_dataset['cell_proportion_names']
}

# Save Virchow combined dataset
output_file_virchow = os.path.join(output_dir, "combined_features_Predictor_genes_Virchow.pt")
torch.save(combined_dataset_virchow, output_file_virchow)
print(f"\nSaved Virchow combined dataset to: {output_file_virchow}")

# Extract Virchow data from combined dataset
features_virchow = combined_dataset_virchow['features']
individual_ids_virchow = combined_dataset_virchow['individual_ids']
tile_ids_virchow = combined_dataset_virchow['tile_ids'] 
expression_300genes_virchow = combined_dataset_virchow['expression_300genes']
expression_300genes_names_virchow = combined_dataset_virchow['expression_300genes_names']
cell_proportions_virchow = combined_dataset_virchow['cell_proportions']
cell_proportion_names_virchow = combined_dataset_virchow['cell_proportion_names']

# Exclude specific individuals
excluded_individual = ["SU-15-27301-B1", "SU-16-02468-B1"] 
valid_indices_virchow = [i for i, ind_id in enumerate(individual_ids_virchow) if ind_id not in excluded_individual]

# Filter all data to exclude these individuals
features_virchow = features_virchow[valid_indices_virchow]
individual_ids_virchow = [individual_ids_virchow[i] for i in valid_indices_virchow]
tile_ids_virchow = [tile_ids_virchow[i] for i in valid_indices_virchow]
expression_300genes_virchow = expression_300genes_virchow[valid_indices_virchow]
cell_proportions_virchow = cell_proportions_virchow[valid_indices_virchow]

print(f"Virchow data after excluding individuals:")
print(f"  Features shape: {features_virchow.shape}")
print(f"  Number of tiles: {len(tile_ids_virchow)}")
print(f"  Number of individuals: {len(set(individual_ids_virchow))}")

# Get unique individuals for cross-validation
unique_individuals_virchow = sorted(list(set(individual_ids_virchow)))
print(f"Unique individuals: {len(unique_individuals_virchow)}")

# Initialize results for Virchow (1x180 table for regular and partial correlations)
virchow_results_regular = []
virchow_results_partial = []

# XGBoost parameters (same as Virchow2)
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

print(f"\nStarting simplified CV for Virchow with {len(expression_300genes_names_virchow)} genes...")

# Simplified CV loop for each gene
for gene_idx, gene_name in enumerate(expression_300genes_names_virchow):
    print(f"\nProcessing gene {gene_idx + 1}/{len(expression_300genes_names_virchow)}: {gene_name}")
    
    # Find the median correlation sample for this gene from Virchow2 results
    if gene_name in median_samples_regular:
        target_individual = median_samples_regular[gene_name]
        print(f"  Using median correlation sample: {target_individual}")
    else:
        print(f"  No median sample found for {gene_name}, skipping...")
        virchow_results_regular.append(0.0)
        virchow_results_partial.append(0.0)
        continue
    
    # Check if target individual exists in Virchow data
    if target_individual not in individual_ids_virchow:
        print(f"  Target individual {target_individual} not found in Virchow data, skipping...")
        virchow_results_regular.append(0.0)
        virchow_results_partial.append(0.0)
        continue
    
    # Split data: leave out target individual
    train_indices = [i for i, ind_id in enumerate(individual_ids_virchow) if ind_id != target_individual]
    test_indices = [i for i, ind_id in enumerate(individual_ids_virchow) if ind_id == target_individual]
    
    if len(test_indices) == 0:
        print(f"  No test samples for {target_individual}, skipping...")
        virchow_results_regular.append(0.0)
        virchow_results_partial.append(0.0)
        continue
        
    # Get train/test data
    X_train = features_virchow[train_indices]
    y_train = expression_300genes_virchow[train_indices, gene_idx]
    X_test = features_virchow[test_indices]
    y_test = expression_300genes_virchow[test_indices, gene_idx]
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train.numpy(), label=y_train.numpy())
    dtest = xgb.DMatrix(X_test.numpy(), label=y_test.numpy())
    
    # Train model using same parameters as Virchow2
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=num_boost_round,
        verbose_eval=False
    )
    
    # Make predictions
    y_pred = model.predict(dtest)
    
    # Calculate regular correlation
    if len(y_test) > 1:
        regular_correlation, _ = pearsonr(y_test.numpy(), y_pred)
        regular_correlation = regular_correlation if not np.isnan(regular_correlation) else 0.0
        
        # Calculate partial correlation controlling for cell proportions
        test_cell_props = cell_proportions_virchow[test_indices]
        if len(test_cell_props) == len(y_pred):
            partial_correlation = partial_correlation_clr_virchow(y_test.numpy(), y_pred, test_cell_props)
        else:
            partial_correlation = 0.0
    else:
        regular_correlation = 0.0
        partial_correlation = 0.0
    
    virchow_results_regular.append(regular_correlation)
    virchow_results_partial.append(partial_correlation)
    print(f"  Regular Correlation: {regular_correlation:.4f}, Partial Correlation: {partial_correlation:.4f}")

# Save Virchow results as 1x300 tables (regular and partial correlations)
virchow_results_regular_df = pd.DataFrame([virchow_results_regular], columns=expression_300genes_names_virchow, index=['Virchow'])
virchow_results_partial_df = pd.DataFrame([virchow_results_partial], columns=expression_300genes_names_virchow, index=['Virchow'])

virchow_regular_output_path = os.path.join(output_dir, "180genes_Virchow_regular_correlations.csv")
virchow_partial_output_path = os.path.join(output_dir, "180genes_Virchow_partial_correlations_clr.csv")

virchow_results_regular_df.to_csv(virchow_regular_output_path)
virchow_results_partial_df.to_csv(virchow_partial_output_path)

print(f"\nSaved Virchow regular correlations to: {virchow_regular_output_path}")
print(f"Saved Virchow partial correlations to: {virchow_partial_output_path}")




### ProvGigPath Feature Processing and Simplified CV for 180 Predictor Genes

print("\n" + "="*80)
print("PROVGIGPATH FEATURE PROCESSING AND SIMPLIFIED CV FOR 180 PREDICTOR GENES")
print("="*80)

# CLR transformation function for proportions (needed for partial correlation)
def clr_transform_provgigpath(proportions):
    """
    Apply centered log-ratio (CLR) transformation to proportions
    CLR(x) = log(x_i / geometric_mean(x))
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-6
    proportions_adj = proportions + epsilon
    
    # Calculate geometric mean
    geometric_mean = np.exp(np.mean(np.log(proportions_adj), axis=1, keepdims=True))
    
    # Apply CLR transformation
    clr_props = np.log(proportions_adj / geometric_mean)
    
    return clr_props

# Function to calculate partial correlation controlling for CLR-transformed cell proportions
def partial_correlation_clr_provgigpath(x, y, cell_props):
    """
    Calculate partial correlation between x and y, controlling for CLR-transformed cell proportions
    """
    # Apply CLR transformation to cell proportions
    clr_props = clr_transform_provgigpath(cell_props.numpy() if torch.is_tensor(cell_props) else cell_props)
    
    # Regress x on CLR-transformed proportions, get residuals
    reg_x = LinearRegression().fit(clr_props, x)
    residual_x = x - reg_x.predict(clr_props)
    
    # Regress y on CLR-transformed proportions, get residuals  
    reg_y = LinearRegression().fit(clr_props, y)
    residual_y = y - reg_y.predict(clr_props)
    
    # Correlation between residuals = partial correlation
    if len(residual_x) > 1:
        partial_corr, p_value = pearsonr(residual_x, residual_y)
        return partial_corr if not np.isnan(partial_corr) else 0.0
    else:
        return 0.0

# Load ProvGigPath features
features_dir = "Colorectal_Cancer_HE_patches/Training_features"
feature_files_provgigpath = [
    "Cancer Cells_training_precomputed_features_ProvGigapath.pt",
    "Normal Epithelial Cells_training_precomputed_features_ProvGigapath.pt", 
    "Other Immune Cells_training_precomputed_features_ProvGigapath.pt",
    "Stromal Cells_training_precomputed_features_ProvGigapath.pt",
    "T Cells_training_precomputed_features_ProvGigapath.pt"
]

print(f"Loading ProvGigPath features from {len(feature_files_provgigpath)} files...")

# Load all ProvGigPath feature files and collect unique tiles
all_tile_ids_provgigpath = set()
all_data_provgigpath = {}
for i, file_name in enumerate(feature_files_provgigpath):
    file_path = os.path.join(features_dir, file_name)
    data = torch.load(file_path, weights_only=False)
    
    # Extract information
    features = data['embeddings']  # [n_tiles, n_features]
    individual_ids = data['individual_ids']
    tile_ids = data['tile_ids']
    
    # Store data for each unique tile
    for j, tile_id in enumerate(tile_ids):
        if tile_id not in all_data_provgigpath:
            all_data_provgigpath[tile_id] = {
                'features': features[j],
                'individual_id': individual_ids[j],
            }
            all_tile_ids_provgigpath.add(tile_id)

# Convert back to arrays
unique_tile_ids_provgigpath = sorted(list(all_tile_ids_provgigpath))
combined_features_provgigpath = []
combined_individual_ids_provgigpath = []

for tile_id in unique_tile_ids_provgigpath:
    tile_data = all_data_provgigpath[tile_id]
    combined_features_provgigpath.append(tile_data['features'])
    combined_individual_ids_provgigpath.append(tile_data['individual_id'])
    
combined_features_provgigpath = torch.stack(combined_features_provgigpath)

# Extract common data by matching with Virchow2's tiles
final_features_provgigpath = []
final_individual_ids_provgigpath = []
final_tile_ids_provgigpath = []

# Create mapping for ProvGigPath tiles
provgigpath_tile_to_idx = {tile_id: i for i, tile_id in enumerate(unique_tile_ids_provgigpath)}

# Use Virchow2's tile order and find matching ProvGigPath features
for i, virchow2_tile_id in enumerate(virchow2_combined_dataset['tile_ids']):
    if virchow2_tile_id in provgigpath_tile_to_idx:
        provgigpath_idx = provgigpath_tile_to_idx[virchow2_tile_id]
        final_features_provgigpath.append(combined_features_provgigpath[provgigpath_idx])
        final_individual_ids_provgigpath.append(combined_individual_ids_provgigpath[provgigpath_idx])
        final_tile_ids_provgigpath.append(virchow2_tile_id)

final_features_provgigpath = torch.stack(final_features_provgigpath)

# Create combined dataset for ProvGigPath (using Virchow2's expression data)
combined_dataset_provgigpath = {
    'tile_ids': final_tile_ids_provgigpath,
    'individual_ids': final_individual_ids_provgigpath,
    'features': final_features_provgigpath,  # ProvGigPath features
    'expression_300genes': virchow2_combined_dataset['expression_300genes'][:len(final_tile_ids_provgigpath)],
    'expression_300genes_names': virchow2_combined_dataset['expression_300genes_names'],
    'cell_proportions': virchow2_combined_dataset['cell_proportions'][:len(final_tile_ids_provgigpath)],
    'cell_proportion_names': virchow2_combined_dataset['cell_proportion_names']
}

# Save ProvGigPath combined dataset
output_file_provgigpath = os.path.join(output_dir, "combined_features_Predictor_genes_ProvGigPath.pt")
torch.save(combined_dataset_provgigpath, output_file_provgigpath)
print(f"\nSaved ProvGigPath combined dataset to: {output_file_provgigpath}")

# Extract ProvGigPath data from combined dataset
features_provgigpath = combined_dataset_provgigpath['features']
individual_ids_provgigpath = combined_dataset_provgigpath['individual_ids']
tile_ids_provgigpath = combined_dataset_provgigpath['tile_ids'] 
expression_300genes_provgigpath = combined_dataset_provgigpath['expression_300genes']
expression_300genes_names_provgigpath = combined_dataset_provgigpath['expression_300genes_names']
cell_proportions_provgigpath = combined_dataset_provgigpath['cell_proportions']
cell_proportion_names_provgigpath = combined_dataset_provgigpath['cell_proportion_names']

# Exclude specific individuals
excluded_individual = ["SU-15-27301-B1", "SU-16-02468-B1"] 
valid_indices_provgigpath = [i for i, ind_id in enumerate(individual_ids_provgigpath) if ind_id not in excluded_individual]

# Filter all data to exclude these individuals
features_provgigpath = features_provgigpath[valid_indices_provgigpath]
individual_ids_provgigpath = [individual_ids_provgigpath[i] for i in valid_indices_provgigpath]
tile_ids_provgigpath = [tile_ids_provgigpath[i] for i in valid_indices_provgigpath]
expression_300genes_provgigpath = expression_300genes_provgigpath[valid_indices_provgigpath]
cell_proportions_provgigpath = cell_proportions_provgigpath[valid_indices_provgigpath]

print(f"ProvGigPath data after excluding individuals:")
print(f"  Features shape: {features_provgigpath.shape}")
print(f"  Number of tiles: {len(tile_ids_provgigpath)}")
print(f"  Number of individuals: {len(set(individual_ids_provgigpath))}")

# Get unique individuals for cross-validation
unique_individuals_provgigpath = sorted(list(set(individual_ids_provgigpath)))
print(f"Unique individuals: {len(unique_individuals_provgigpath)}")

# Initialize results for ProvGigPath (1x180 table for regular and partial correlations)
provgigpath_results_regular = []
provgigpath_results_partial = []

# XGBoost parameters (same as Virchow2)
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

print(f"\nStarting simplified CV for ProvGigPath with {len(expression_300genes_names_provgigpath)} genes...")

# Simplified CV loop for each gene
for gene_idx, gene_name in enumerate(expression_300genes_names_provgigpath):
    print(f"\nProcessing gene {gene_idx + 1}/{len(expression_300genes_names_provgigpath)}: {gene_name}")
    
    # Find the median correlation sample for this gene from Virchow2 results
    if gene_name in median_samples_regular:
        target_individual = median_samples_regular[gene_name]
        print(f"  Using median correlation sample: {target_individual}")
    else:
        print(f"  No median sample found for {gene_name}, skipping...")
        provgigpath_results_regular.append(0.0)
        provgigpath_results_partial.append(0.0)
        continue
    
    # Check if target individual exists in ProvGigPath data
    if target_individual not in individual_ids_provgigpath:
        print(f"  Target individual {target_individual} not found in ProvGigPath data, skipping...")
        provgigpath_results_regular.append(0.0)
        provgigpath_results_partial.append(0.0)
        continue
    
    # Split data: leave out target individual
    train_indices = [i for i, ind_id in enumerate(individual_ids_provgigpath) if ind_id != target_individual]
    test_indices = [i for i, ind_id in enumerate(individual_ids_provgigpath) if ind_id == target_individual]
    
    if len(test_indices) == 0:
        print(f"  No test samples for {target_individual}, skipping...")
        provgigpath_results_regular.append(0.0)
        provgigpath_results_partial.append(0.0)
        continue
        
    # Get train/test data
    X_train = features_provgigpath[train_indices]
    y_train = expression_300genes_provgigpath[train_indices, gene_idx]
    X_test = features_provgigpath[test_indices]
    y_test = expression_300genes_provgigpath[test_indices, gene_idx]
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train.numpy(), label=y_train.numpy())
    dtest = xgb.DMatrix(X_test.numpy(), label=y_test.numpy())
    
    # Train model using same parameters as Virchow2
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=num_boost_round,
        verbose_eval=False
    )
    
    # Make predictions
    y_pred = model.predict(dtest)
    
    # Calculate regular correlation
    if len(y_test) > 1:
        regular_correlation, _ = pearsonr(y_test.numpy(), y_pred)
        regular_correlation = regular_correlation if not np.isnan(regular_correlation) else 0.0
        
        # Calculate partial correlation controlling for cell proportions
        test_cell_props = cell_proportions_provgigpath[test_indices]
        if len(test_cell_props) == len(y_pred):
            partial_correlation = partial_correlation_clr_provgigpath(y_test.numpy(), y_pred, test_cell_props)
        else:
            partial_correlation = 0.0
    else:
        regular_correlation = 0.0
        partial_correlation = 0.0
    
    provgigpath_results_regular.append(regular_correlation)
    provgigpath_results_partial.append(partial_correlation)
    print(f"  Regular Correlation: {regular_correlation:.4f}, Partial Correlation: {partial_correlation:.4f}")

# Save ProvGigPath results as 1x300 tables (regular and partial correlations)
provgigpath_results_regular_df = pd.DataFrame([provgigpath_results_regular], columns=expression_300genes_names_provgigpath, index=['ProvGigPath'])
provgigpath_results_partial_df = pd.DataFrame([provgigpath_results_partial], columns=expression_300genes_names_provgigpath, index=['ProvGigPath'])

provgigpath_regular_output_path = os.path.join(output_dir, "180genes_ProvGigPath_regular_correlations.csv")
provgigpath_partial_output_path = os.path.join(output_dir, "180genes_ProvGigPath_partial_correlations_clr.csv")

provgigpath_results_regular_df.to_csv(provgigpath_regular_output_path)
provgigpath_results_partial_df.to_csv(provgigpath_partial_output_path)

print(f"\nSaved ProvGigPath regular correlations to: {provgigpath_regular_output_path}")
print(f"Saved ProvGigPath partial correlations to: {provgigpath_partial_output_path}")




### CONCH Feature Processing and Simplified CV for 180 Predictor Genes

print("\n" + "="*80)
print("CONCH FEATURE PROCESSING AND SIMPLIFIED CV FOR 180 PREDICTOR GENES")
print("="*80)

# CLR transformation function for proportions (needed for partial correlation)
def clr_transform_conch(proportions):
    """
    Apply centered log-ratio (CLR) transformation to proportions
    CLR(x) = log(x_i / geometric_mean(x))
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-6
    proportions_adj = proportions + epsilon
    
    # Calculate geometric mean
    geometric_mean = np.exp(np.mean(np.log(proportions_adj), axis=1, keepdims=True))
    
    # Apply CLR transformation
    clr_props = np.log(proportions_adj / geometric_mean)
    
    return clr_props

# Function to calculate partial correlation controlling for CLR-transformed cell proportions
def partial_correlation_clr_conch(x, y, cell_props):
    """
    Calculate partial correlation between x and y, controlling for CLR-transformed cell proportions
    """
    # Apply CLR transformation to cell proportions
    clr_props = clr_transform_conch(cell_props.numpy() if torch.is_tensor(cell_props) else cell_props)
    
    # Regress x on CLR-transformed proportions, get residuals
    reg_x = LinearRegression().fit(clr_props, x)
    residual_x = x - reg_x.predict(clr_props)
    
    # Regress y on CLR-transformed proportions, get residuals  
    reg_y = LinearRegression().fit(clr_props, y)
    residual_y = y - reg_y.predict(clr_props)
    
    # Correlation between residuals = partial correlation
    if len(residual_x) > 1:
        partial_corr, p_value = pearsonr(residual_x, residual_y)
        return partial_corr if not np.isnan(partial_corr) else 0.0
    else:
        return 0.0

# Load CONCH features
features_dir = "Colorectal_Cancer_HE_patches/Training_features"
feature_files_conch = [
    "Cancer Cells_training_precomputed_features_Conch.pt",
    "Normal Epithelial Cells_training_precomputed_features_Conch.pt", 
    "Other Immune Cells_training_precomputed_features_Conch.pt",
    "Stromal Cells_training_precomputed_features_Conch.pt",
    "T Cells_training_precomputed_features_Conch.pt"
]

print(f"Loading CONCH features from {len(feature_files_conch)} files...")

# Load all CONCH feature files and collect unique tiles
all_tile_ids_conch = set()
all_data_conch = {}
for i, file_name in enumerate(feature_files_conch):
    file_path = os.path.join(features_dir, file_name)
    data = torch.load(file_path, weights_only=False)
    
    # Extract information
    features = data['embeddings']  # [n_tiles, n_features]
    individual_ids = data['individual_ids']
    tile_ids = data['tile_ids']
    
    # Store data for each unique tile
    for j, tile_id in enumerate(tile_ids):
        if tile_id not in all_data_conch:
            all_data_conch[tile_id] = {
                'features': features[j],
                'individual_id': individual_ids[j],
            }
            all_tile_ids_conch.add(tile_id)

# Convert back to arrays
unique_tile_ids_conch = sorted(list(all_tile_ids_conch))
combined_features_conch = []
combined_individual_ids_conch = []

for tile_id in unique_tile_ids_conch:
    tile_data = all_data_conch[tile_id]
    combined_features_conch.append(tile_data['features'])
    combined_individual_ids_conch.append(tile_data['individual_id'])
    
combined_features_conch = torch.stack(combined_features_conch)

# Extract common data by matching with Virchow2's tiles
final_features_conch = []
final_individual_ids_conch = []
final_tile_ids_conch = []

# Create mapping for CONCH tiles
conch_tile_to_idx = {tile_id: i for i, tile_id in enumerate(unique_tile_ids_conch)}

# Use Virchow2's tile order and find matching CONCH features
for i, virchow2_tile_id in enumerate(virchow2_combined_dataset['tile_ids']):
    if virchow2_tile_id in conch_tile_to_idx:
        conch_idx = conch_tile_to_idx[virchow2_tile_id]
        final_features_conch.append(combined_features_conch[conch_idx])
        final_individual_ids_conch.append(combined_individual_ids_conch[conch_idx])
        final_tile_ids_conch.append(virchow2_tile_id)

final_features_conch = torch.stack(final_features_conch)

# Create combined dataset for CONCH (using Virchow2's expression data)
combined_dataset_conch = {
    'tile_ids': final_tile_ids_conch,
    'individual_ids': final_individual_ids_conch,
    'features': final_features_conch,  # CONCH features
    'expression_300genes': virchow2_combined_dataset['expression_300genes'][:len(final_tile_ids_conch)],
    'expression_300genes_names': virchow2_combined_dataset['expression_300genes_names'],
    'cell_proportions': virchow2_combined_dataset['cell_proportions'][:len(final_tile_ids_conch)],
    'cell_proportion_names': virchow2_combined_dataset['cell_proportion_names']
}

# Save CONCH combined dataset
output_file_conch = os.path.join(output_dir, "combined_features_Predictor_genes_CONCH.pt")
torch.save(combined_dataset_conch, output_file_conch)
print(f"\nSaved CONCH combined dataset to: {output_file_conch}")

# Extract CONCH data from combined dataset
features_conch = combined_dataset_conch['features']
individual_ids_conch = combined_dataset_conch['individual_ids']
tile_ids_conch = combined_dataset_conch['tile_ids'] 
expression_300genes_conch = combined_dataset_conch['expression_300genes']
expression_300genes_names_conch = combined_dataset_conch['expression_300genes_names']
cell_proportions_conch = combined_dataset_conch['cell_proportions']
cell_proportion_names_conch = combined_dataset_conch['cell_proportion_names']

# Exclude specific individuals
excluded_individual = ["SU-15-27301-B1", "SU-16-02468-B1"] 
valid_indices_conch = [i for i, ind_id in enumerate(individual_ids_conch) if ind_id not in excluded_individual]

# Filter all data to exclude these individuals
features_conch = features_conch[valid_indices_conch]
individual_ids_conch = [individual_ids_conch[i] for i in valid_indices_conch]
tile_ids_conch = [tile_ids_conch[i] for i in valid_indices_conch]
expression_300genes_conch = expression_300genes_conch[valid_indices_conch]
cell_proportions_conch = cell_proportions_conch[valid_indices_conch]

print(f"CONCH data after excluding individuals:")
print(f"  Features shape: {features_conch.shape}")
print(f"  Number of tiles: {len(tile_ids_conch)}")
print(f"  Number of individuals: {len(set(individual_ids_conch))}")

# Get unique individuals for cross-validation
unique_individuals_conch = sorted(list(set(individual_ids_conch)))
print(f"Unique individuals: {len(unique_individuals_conch)}")

# Initialize results for CONCH (1x180 table for regular and partial correlations)
conch_results_regular = []
conch_results_partial = []

# XGBoost parameters (same as Virchow2)
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

print(f"\nStarting simplified CV for CONCH with {len(expression_300genes_names_conch)} genes...")

# Simplified CV loop for each gene
for gene_idx, gene_name in enumerate(expression_300genes_names_conch):
    print(f"\nProcessing gene {gene_idx + 1}/{len(expression_300genes_names_conch)}: {gene_name}")
    
    # Find the median correlation sample for this gene from Virchow2 results
    if gene_name in median_samples_regular:
        target_individual = median_samples_regular[gene_name]
        print(f"  Using median correlation sample: {target_individual}")
    else:
        print(f"  No median sample found for {gene_name}, skipping...")
        conch_results_regular.append(0.0)
        conch_results_partial.append(0.0)
        continue
    
    # Check if target individual exists in CONCH data
    if target_individual not in individual_ids_conch:
        print(f"  Target individual {target_individual} not found in CONCH data, skipping...")
        conch_results_regular.append(0.0)
        conch_results_partial.append(0.0)
        continue
    
    # Split data: leave out target individual
    train_indices = [i for i, ind_id in enumerate(individual_ids_conch) if ind_id != target_individual]
    test_indices = [i for i, ind_id in enumerate(individual_ids_conch) if ind_id == target_individual]
    
    if len(test_indices) == 0:
        print(f"  No test samples for {target_individual}, skipping...")
        conch_results_regular.append(0.0)
        conch_results_partial.append(0.0)
        continue
        
    # Get train/test data
    X_train = features_conch[train_indices]
    y_train = expression_300genes_conch[train_indices, gene_idx]
    X_test = features_conch[test_indices]
    y_test = expression_300genes_conch[test_indices, gene_idx]
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train.numpy(), label=y_train.numpy())
    dtest = xgb.DMatrix(X_test.numpy(), label=y_test.numpy())
    
    # Train model using same parameters as Virchow2
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=num_boost_round,
        verbose_eval=False
    )
    
    # Make predictions
    y_pred = model.predict(dtest)
    
    # Calculate regular correlation
    if len(y_test) > 1:
        regular_correlation, _ = pearsonr(y_test.numpy(), y_pred)
        regular_correlation = regular_correlation if not np.isnan(regular_correlation) else 0.0
        
        # Calculate partial correlation controlling for cell proportions
        test_cell_props = cell_proportions_conch[test_indices]
        if len(test_cell_props) == len(y_pred):
            partial_correlation = partial_correlation_clr_conch(y_test.numpy(), y_pred, test_cell_props)
        else:
            partial_correlation = 0.0
    else:
        regular_correlation = 0.0
        partial_correlation = 0.0
    
    conch_results_regular.append(regular_correlation)
    conch_results_partial.append(partial_correlation)
    print(f"  Regular Correlation: {regular_correlation:.4f}, Partial Correlation: {partial_correlation:.4f}")

# Save CONCH results as 1x300 tables (regular and partial correlations)
conch_results_regular_df = pd.DataFrame([conch_results_regular], columns=expression_300genes_names_conch, index=['CONCH'])
conch_results_partial_df = pd.DataFrame([conch_results_partial], columns=expression_300genes_names_conch, index=['CONCH'])

conch_regular_output_path = os.path.join(output_dir, "180genes_CONCH_regular_correlations.csv")
conch_partial_output_path = os.path.join(output_dir, "180genes_CONCH_partial_correlations_clr.csv")

conch_results_regular_df.to_csv(conch_regular_output_path)
conch_results_partial_df.to_csv(conch_partial_output_path)

print(f"\nSaved CONCH regular correlations to: {conch_regular_output_path}")
print(f"Saved CONCH partial correlations to: {conch_partial_output_path}")




