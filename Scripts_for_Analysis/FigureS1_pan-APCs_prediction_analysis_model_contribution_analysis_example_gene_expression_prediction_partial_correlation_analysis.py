"""
Figure S1: pan-APCs Prediction Analysis - Model Contribution Analysis, 
Example Gene Expression Prediction, and Partial Correlation Analysis
=====================================================================

This script generates comprehensive supplementary Figure S1 with four panels:

Panel A: Comparison of prediction accuracy
-----------------------------------------
- Compares model performance (MAE and Pearson correlation) across:
  * pan-APC (Other Immune Cells aggregated)
  * B Cells (individual)
  * Myeloid Cells (individual)
- Demonstrates whether aggregating B and Myeloid into pan-APC improves prediction
- 7 models compared: ResNet50, Conch, Prov-GigaPath, UNI2-h, Virchow, Virchow2, Foundation models combined

Panel B: Foundation model contribution analysis
-----------------------------------------------
- Contribution analysis of foundation models to combined XGBoost prediction
- Focus on T cells, pan-APC, and Normal Epithelial cells
- Note: Code is in Figure3_UMAP_Contri_RegressOut.py

Panel C: Example gene expression prediction
--------------------------------------------
- S100A6 gene expression prediction example
- Held-out individual: SU-17-14212-A1
- Leave-One-Individual-Out (LOIO) cross-validation
- Model: Virchow2
- Shows real vs predicted expression scatter plot

Panel D: Partial correlation analysis
--------------------------------------
- Comparison of regular Pearson correlation vs partial Pearson correlation
- Controls for cell type proportions using CLR transformation
- 180 predictor genes analyzed
- Three representative models: Virchow2, UNI2-h, ResNet50
- Gene categories: Tumor, Normal Epithelial, T, pan-APC, Stromal, Highly Variable

Author: Saishi Cui
Date: Feb 2026
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import xgboost as xgb
import os


# Figure S1 (Panel A) Comparison of prediction accuracy of B cells, 
# Myeloid cells, or combining them to one group of pan-APCs.

BASE_PATH = "/xgboost_prediction"

# Map display names to file names
model_file_names = {
    'ResNet50': 'ResNet50',
    'Conch': 'Conch',
    'Prov-GigaPath': 'ProvGigapath',
    'UNI2-h': 'UNI2h', 
    'Virchow': 'Virchow',
    'Virchow2': 'Virchow2',
    'Combined': 'Combined'
}

# Collect all the results
all_results = []

# Cell types to compare

cell_type_configs = {
    "Other Immune Cells": {
        "folder_suffix": "_individual_level_ratio100",
        "metrics_file": "individual_metrics_individual_level.csv"
    },
    "B_Cells": {
        "folder_suffix": "_individual_level",
        "metrics_file": "individual_metrics.csv"
    },
    "Myeloid_Cells": {
        "folder_suffix": "_individual_level",
        "metrics_file": "individual_metrics.csv"
    }
}

# Step 1: First get filtered individuals from Other Immune Cells
# Apply n_test_samples >= 200 and ct_range > 0.3 ONLY to Other Immune Cells
filtered_individuals = set()

print("Step 1: Filtering individuals based on Other Immune Cells criteria...")
for model_display, model_file in model_file_names.items():
    config = cell_type_configs["Other Immune Cells"]
    if model_file == "ResNet50":
        folder_name = f"Other Immune Cells_Resnet50{config['folder_suffix']}"
    else:
        folder_name = f"Other Immune Cells_{model_file}{config['folder_suffix']}"
    
    file_path = f"{BASE_PATH}/{folder_name}/{config['metrics_file']}"
    
    try:
        df = pd.read_csv(file_path)
        for index, row in df.iterrows():
            ct_range = row["Max_celltype_proportion"] - row["Min_celltype_proportion"]
            if row["n_test_samples"] >= 200 and ct_range > 0.3:
                filtered_individuals.add(row["Individual"])
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

print(f"Found {len(filtered_individuals)} individuals passing Other Immune Cells filter")
print(f"Individuals: {sorted(filtered_individuals)}")

# Step 2: Load data for all cell types using the filtered individuals
print("\nStep 2: Loading data for all cell types using filtered individuals...")
for cell_type, config in cell_type_configs.items():
    for model_display, model_file in model_file_names.items():
        # Construct folder path
        # Note: Other Immune Cells uses "Resnet50" (lowercase s) while B/MYE use "ResNet50"
        if cell_type == "Other Immune Cells" and model_file == "ResNet50":
            folder_name = f"{cell_type}_Resnet50{config['folder_suffix']}"
        else:
            folder_name = f"{cell_type}_{model_file}{config['folder_suffix']}"
        
        file_path = f"{BASE_PATH}/{folder_name}/{config['metrics_file']}"
        
        try:
            df = pd.read_csv(file_path)
            
            for index, row in df.iterrows():
                individual = row["Individual"]
                ct_range = row["Max_celltype_proportion"] - row["Min_celltype_proportion"]
                
                # Only include if individual is in the filtered list
                if individual in filtered_individuals:
                    all_results.append({
                        "model_name": model_display, 
                        "cell_type": cell_type, 
                        "MAE": row["MAE"], 
                        "Pearson_correlation": row["Pearson"], 
                        "n_test_samples": row["n_test_samples"], 
                        "Max_celltype_proportion": row["Max_celltype_proportion"], 
                        "Min_celltype_proportion": row["Min_celltype_proportion"], 
                        "ct_range": ct_range,
                        "Individual": individual
                    })
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

# Create the final DataFrame
final_df = pd.DataFrame(all_results)

# Map cell type names for display (Other Immune Cells -> pan-APC)
final_df['cell_type'] = final_df['cell_type'].replace({
    'Other Immune Cells': 'pan-APC',
    'B_Cells': 'B Cells',
    'Myeloid_Cells': 'Myeloid Cells'
})

# Save the filtered results
output_path = f"{BASE_PATH}/B_MYE_panAPC_individual_metrics_filtered.csv"
final_df.to_csv(output_path, index=False)
print(f"Saved filtered results to: {output_path}")

# Print summary
print("\nData summary:")
for cell_type in final_df['cell_type'].unique():
    cell_data = final_df[final_df['cell_type'] == cell_type]
    print(f"{cell_type}: {len(cell_data)} samples")

# Set the figure style
plt.style.use('default')
sns.set_palette("husl")

# Define the color mapping (same as Figure 4)
color_palette = {
    'ResNet50': '#FEF0DE',
    'Conch': '#C43E96', 
    'Prov-GigaPath': '#DEDBEE',
    'UNI2-h': '#06948E',
    'Virchow': '#F3CDCC',
    'Virchow2': '#F0CF7F',
    'Combined': '#FF6B6B'  
}

# Define the metrics to plot
metrics = ['MAE', 'Pearson_correlation']
metric_titles = ['Mean Absolute Error (MAE)', 'Pearson Correlation']

# Define the order of cell types (x-axis)
cell_type_order = ['pan-APC', 'B Cells', 'Myeloid Cells']

# Create a separate figure for each metric
for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
    # Create a separate figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Debug: Check data for current metric
    print(f"\nDEBUG - {metric} data check:")
    for cell_type in cell_type_order:
        cell_data = final_df[final_df['cell_type'] == cell_type]
        print(f"{cell_type}: {len(cell_data)} samples")
        if len(cell_data) > 0:
            print(f"  {metric} range: {cell_data[metric].min():.3f} - {cell_data[metric].max():.3f}")
            print(f"  {metric} std: {cell_data[metric].std():.3f}")
    
    # Create boxplot
    box_plot = sns.boxplot(
        data=final_df, 
        x='cell_type', 
        y=metric, 
        hue='model_name',
        palette=color_palette,
        ax=ax,
        order=cell_type_order,
        showfliers=False,  # Do not show outliers
        linewidth=3,  # Thicker box borders
        legend=False  # Do not show legend
    )
    
    # Add the jitter scatter plot
    # Use stripplot to better show the scatter points in the box
    sns.stripplot(
        data=final_df, 
        x='cell_type', 
        y=metric, 
        hue='model_name',
        palette=color_palette,
        ax=ax,
        order=cell_type_order,
        size=6,  # Same size as FigureS10
        alpha=0.7,  # Match FigureS10 alpha
        dodge=True,  # Let the points with different hue to be displayed separately
        jitter=0.3,  # Add jitter
        edgecolor='black',
        linewidth=0.5,  # Match FigureS10 linewidth
        legend=False  # Do not show the legend of the stripplot
    )
    
    # Set the labels - do not show the horizontal axis title, do not show the total title
    ax.set_xlabel('')  # Do not show the horizontal axis title
    ax.set_ylabel(title, fontsize=22, fontweight='bold')  # Set the vertical axis title to be bold
    
    # Set the tick label style - the horizontal and vertical axis ticks to be bold
    ax.tick_params(axis='x', rotation=25, labelsize=20, labelcolor='black', 
                   width=2, length=6, colors='black')  # rotation=25 makes x-axis labels tilted
    ax.tick_params(axis='y', labelsize=20, labelcolor='black', 
                   width=2, length=6, colors='black')
    
    # Set the tick label font to be bold
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # Set the vertical axis range
    if metric == 'MAE':
        ax.set_ylim(0, 0.25)
    elif metric == 'Pearson_correlation':
        ax.set_ylim(-0.5, 1)
    
    # Add horizontal grid lines only with darker and thicker lines
    ax.grid(True, alpha=0.7, linestyle='--', linewidth=2, axis='y')
    
    # Add thick black borders like hexagon plots
    ax.spines['top'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    
    # Adjust the layout
    plt.tight_layout()
    
    # Save the figure
    output_fig_path = f'/Users/scui2/ST/Colorectal_Cancer_HE_patches/Visual/B_MYE_panAPC_comparison_{metric}_boxplot.png'
    plt.savefig(output_fig_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"\nSaved figure to: {output_fig_path}")
    
    # Show the figure
    plt.show()

print("\nDone!")





### Figure S1 (Panel B) Contribution analysis of foundation models 
## to the combined XGBoost prediction performance for T, pan-APC and normal epithelial cells.

## Please refer to the script: Figure3_UMAP_Contri_RegressOut.py for the code.




### Figure S1 (Panel C) example of S100A6 expression prediction in one held-out individual
## (SU-17-14212-A1) under LOIO cross-validation using Virchow2.



# Load combined dataset for deep dive analysis
output_dir = "/Gene_Expression_Prediction"
combined_dataset_path = os.path.join(output_dir, "combined_features_Predictor_genes.pt")
print(f"Loading combined dataset from: {combined_dataset_path}")
combined_dataset = torch.load(combined_dataset_path, weights_only=False)
print(f"✅ Combined dataset loaded successfully")

# Focus on S100A6 gene
target_gene = 'S100A6'
target_sample = 'SU-17-14212-A1'

print(f"Target gene: {target_gene}")
print(f"Target sample (LOIO): {target_sample}")

# Use data from combined_dataset (loaded from .pt file)
X = combined_dataset['features'].numpy()  # Virchow2 features
expression_180genes = combined_dataset['expression_300genes'].numpy()
expression_180genes_names = combined_dataset['expression_300genes_names']
tile_ids = combined_dataset['tile_ids']  # These are the tile IDs
individual_ids = combined_dataset['individual_ids']  # These are the individual IDs

print(f"Feature matrix shape: {X.shape}")
print(f"Expression matrix shape: {expression_180genes.shape}")
print(f"Number of samples: {len(tile_ids)}")

# Check if the gene and individual exist
if target_gene not in expression_180genes_names:
    print(f"❌ Gene {target_gene} not found in expression data")
    print(f"Available genes: {expression_180genes_names[:10]}...")  # Show first 10
else:
    print(f"✅ Gene {target_gene} found in expression data")

# Check if target individual exists
if target_sample not in individual_ids:
    print(f"❌ Individual {target_sample} not found in individual data")
    unique_individuals = list(set(individual_ids))
    print(f"Available individuals: {unique_individuals[:5]}...")  # Show first 5
else:
    print(f"✅ Individual {target_sample} found in individual data")

# Get target gene index and expression values
target_gene_idx = expression_180genes_names.index(target_gene)
y = expression_180genes[:, target_gene_idx]  # S100A6 expression values

# Find all samples (tiles) belonging to the target individual
target_individual_indices = [i for i, ind_id in enumerate(individual_ids) if ind_id == target_sample]
print(f"Found {len(target_individual_indices)} tiles for individual {target_sample}")

# For LOIO, we need to leave out ALL tiles from the target individual
# Training set: all samples except those from target_individual
train_indices = [i for i in range(len(individual_ids)) if individual_ids[i] != target_sample]
test_indices = target_individual_indices

print(f"Training set: {len(train_indices)} tiles")
print(f"Test set: {len(test_indices)} tiles from individual {target_sample}")

# Prepare training and test data
X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]

print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")

# Train XGBoost model using same parameters as 180-gene training
print(f"Training XGBoost model for {target_gene}...")

# Use same parameters as in the main training loop
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'n_jobs': -1,
    'eta': 0.01,
    'min_child_weight': 3,
    'max_delta_step': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'alpha': 0.1,
    'lambda': 0.01,
    'max_depth': 12,
    'seed': 42
}
num_boost_round = 800

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Train model
xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=num_boost_round,
    verbose_eval=False
)

# Make predictions for test samples
y_pred_test = xgb_model.predict(dtest)
print(f"Real expression range: {y_test.min():.4f} - {y_test.max():.4f}")
print(f"Predicted expression range: {y_pred_test.min():.4f} - {y_pred_test.max():.4f}")

# Calculate correlation for test samples
if len(y_test) > 1:
    test_correlation = np.corrcoef(y_test, y_pred_test)[0, 1]
    test_r2 = test_correlation ** 2
    print(f"Test correlation: {test_correlation:.4f}, R² = {test_r2:.4f}")
else:
    test_correlation = np.nan
    test_r2 = np.nan
    print("Only one test sample - cannot calculate correlation")

# Get predictions for all training samples for visualization
y_pred_train = xgb_model.predict(dtrain)

# Calculate correlation for training samples
if len(y_train) > 1:
    train_correlation = np.corrcoef(y_train, y_pred_train)[0, 1]
    train_r2 = train_correlation ** 2
    print(f"Training correlation: {train_correlation:.4f}, R² = {train_r2:.4f}")
else:
    train_correlation = np.nan
    train_r2 = np.nan
    print("Not enough training samples - cannot calculate correlation")

# Combine all data for plotting
all_real = np.concatenate([y_train, y_test])
all_pred = np.concatenate([y_pred_train, y_pred_test])

# Create sample identifiers for plotting
train_sample_ids = [f"{tile_ids[i]}" for i in train_indices]
test_sample_ids = [f"{tile_ids[i]}" for i in test_indices]
all_samples = train_sample_ids + test_sample_ids
sample_types = ['Training'] * len(train_indices) + ['Test (LOIO)'] * len(test_indices)

# Get tumor proportions for all samples from combined_dataset
cell_proportions = combined_dataset['cell_proportions'].numpy()
cell_proportion_names = combined_dataset['cell_proportion_names']

# Find Cancer_Cells_Proportion index
cancer_cells_idx = cell_proportion_names.index('Cancer_Cells_Proportion')
all_tumor_proportions = cell_proportions[:, cancer_cells_idx]

# Get tumor proportions for plotting samples
tumor_proportions = []
# Training samples
for idx in train_indices:
    tumor_prop = all_tumor_proportions[idx]
    tumor_proportions.append(tumor_prop)
# Test samples
for idx in test_indices:
    tumor_prop = all_tumor_proportions[idx]
    tumor_proportions.append(tumor_prop)

print(f"Tumor proportion range for {target_sample}: {all_tumor_proportions[test_indices].min():.4f} - {all_tumor_proportions[test_indices].max():.4f}")

# Create plotting DataFrame
plot_data = pd.DataFrame({
    'Sample': all_samples,
    'Real_Expression': all_real,
    'Predicted_Expression': all_pred,
    'Sample_Type': sample_types,
    'Tumor_Proportion': tumor_proportions
})

# Plot : Real vs Predicted Expression
plt.figure(figsize=(10, 8))

# Plot training samples
train_data = plot_data[plot_data['Sample_Type'] == 'Training']
plt.scatter(train_data['Predicted_Expression'], train_data['Real_Expression'], 
           alpha=0.7, s=30, color='lightblue', edgecolor='black', linewidth=0.5,
           label=f'Training samples (n={len(train_data)})')

# Plot test individual (LOIO)
test_data = plot_data[plot_data['Sample_Type'] == 'Test (LOIO)']
plt.scatter(test_data['Predicted_Expression'], test_data['Real_Expression'], 
           alpha=1.0, s=50, color='red', edgecolor='black', linewidth=2,
           label=f'Test individual ({target_sample}, n={len(test_data)})', marker='D')

# Add diagonal reference line
min_val = min(plot_data['Real_Expression'].min(), plot_data['Predicted_Expression'].min())
max_val = max(plot_data['Real_Expression'].max(), plot_data['Predicted_Expression'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='Perfect prediction')

# Styling
plt.xlabel('Predicted Gene Expression (log-Normalized)', fontsize=24, fontweight='bold')  # Updated label and increased font size
plt.ylabel('True Gene Expression (log-Normalized)', fontsize=24, fontweight='bold')  # Updated label and increased font size
plt.title(f'{target_gene} Expression Prediction\n(LOIO: {target_sample})', fontsize=20, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)  # Increased tick label size
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontweight('bold')

# Updated legend with larger, bold font
legend = plt.legend(fontsize=18, frameon=True, fancybox=True, shadow=True, markerscale=1.5)
for text in legend.get_texts():
    text.set_fontweight('bold')

# Add Pearson correlation annotations in the upper left corner
if not np.isnan(train_correlation):
    plt.text(0.02, 0.98, f'Training Pearson r = {train_correlation:.3f}', 
             transform=plt.gca().transAxes, fontsize=16, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

if not np.isnan(test_correlation):
    plt.text(0.02, 0.92, f'Testing Pearson r = {test_correlation:.3f}', 
             transform=plt.gca().transAxes, fontsize=16, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.grid(True, alpha=0.3, linewidth=1)
plt.tight_layout()


plot1_path = f"/Visual/{target_gene}_Real_vs_Predicted_LOIO_{target_sample}.png"
plt.savefig(plot1_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()




### Figure S1 (Panel D) Comparison of regular
## Pearson correlation versus partial Pearson correlation controlling for cell type proportions
## across 180 predictor genes for three representative models.

print("\n" + "="*80)
print("STEP 4: Creating correlation scatter plot for 180 genes")
print("="*80)

# Load correlation data
regular_corr_df = pd.read_csv("/Gene_Expression_Prediction/180genes_regular_correlations.csv", index_col=0)
partial_corr_df = pd.read_csv("/Gene_Expression_Prediction/180genes_partial_correlations_clr.csv", index_col=0)

print(f"Regular correlations shape: {regular_corr_df.shape}")
print(f"Partial correlations shape: {partial_corr_df.shape}")

# Get all genes (columns, not rows)
genes_180 = regular_corr_df.columns.tolist()
print(f"Using all genes: {len(genes_180)}")

# Use all genes
regular_corr_180 = regular_corr_df
partial_corr_180 = partial_corr_df

# Calculate median and IQR for each gene (across 23 samples)
regular_median = regular_corr_180.median(axis=0)  # axis=0 for columns (genes)
regular_iqr = regular_corr_180.quantile(0.75, axis=0) - regular_corr_180.quantile(0.25, axis=0)
partial_median = partial_corr_180.median(axis=0)  # axis=0 for columns (genes)
partial_iqr = partial_corr_180.quantile(0.75, axis=0) - partial_corr_180.quantile(0.25, axis=0)

# Create gene groups based on the order from FigureS10
# Based on the script: Cancer_Cells (0-29), Normal_Epithelial (30-59), T (60-89), 
# Other_Immune (90-119), Stromal (120-149), Highly_Variable (150-179)
gene_groups = []
gene_colors = []

# Define colors for each cell type (matching Figure4 color scheme)
color_mapping = {
    'Tumor Markers': '#E41A1C',      # Red (from Figure4)
    'Normal Epithelial Markers': '#377EB8', # Blue (from Figure4)
    'T Markers': '#4DAF4A',           # Green (from Figure4)
    'pan-APC Markers': '#FF7F00',      # Orange (from Figure4)
    'Stromal Markers': '#FFFF33',           # Yellow (from Figure4)
    'Highly Variable Genes': '#666666'    # Dark gray
}

for i, gene in enumerate(genes_180):
    if i < 30:
        gene_groups.append('Tumor Markers')
        gene_colors.append(color_mapping['Tumor Markers'])
    elif i < 60:
        gene_groups.append('Normal Epithelial Markers')
        gene_colors.append(color_mapping['Normal Epithelial Markers'])
    elif i < 90:
        gene_groups.append('T Markers')
        gene_colors.append(color_mapping['T Markers'])
    elif i < 120:
        gene_groups.append('pan-APC Markers')
        gene_colors.append(color_mapping['pan-APC Markers'])
    elif i < 150:
        gene_groups.append('Stromal Markers')
        gene_colors.append(color_mapping['Stromal Markers'])
    else:  # 150-179
        gene_groups.append('Highly Variable Genes')
        gene_colors.append(color_mapping['Highly Variable Genes'])

# Create DataFrame for plotting
plot_df = pd.DataFrame({
    'Gene': genes_180,
    'Regular_Correlation_Median': regular_median,
    'Partial_Correlation_Median': partial_median,
    'Regular_IQR': regular_iqr,
    'Partial_IQR': partial_iqr,
    'Gene_Group': gene_groups,
    'Color': gene_colors
})

# Set fixed point size for all genes (larger size)
plot_df['Point_Size'] = 150  # Larger fixed size for all points

print(f"Using fixed point size: {plot_df['Point_Size'].iloc[0]}")
print(f"Gene group distribution:")
print(plot_df['Gene_Group'].value_counts())

# Create the scatter plot
plt.figure(figsize=(12, 10))

# Plot each group separately for legend
for group in ['Tumor Markers', 'Normal Epithelial Markers', 'T Markers', 'pan-APC Markers', 'Stromal Markers', 'Highly Variable Genes']:
    group_data = plot_df[plot_df['Gene_Group'] == group]
    if len(group_data) > 0:
        # Special case for Highly Variable Genes - show n=30 instead of actual count
        if group == 'Highly Variable Genes':
            label_text = f'{group} (n=30)'
        else:
            label_text = f'{group} (n={len(group_data)})'
            
        plt.scatter(
            group_data['Regular_Correlation_Median'],
            group_data['Partial_Correlation_Median'],
            s=group_data['Point_Size'],
            c=color_mapping[group],
            alpha=0.7,
            label=label_text,
            edgecolors='black',
            linewidth=0.5
        )

# Set labels and title with large fonts (increased by 2 sizes)
plt.xlabel('Regular Pearson Correlation', fontsize=24, fontweight='bold')
plt.ylabel('Partial Pearson Correlation', fontsize=24, fontweight='bold')
plt.title('Gene Expression Prediction (Virchow2)', fontsize=26, fontweight='bold')

# Styling with larger fonts and thicker borders
plt.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontweight('bold')

# Add thick black border around the plot
for spine in plt.gca().spines.values():
    spine.set_linewidth(3)
    spine.set_edgecolor('black')

# Legend with larger font (without reference line) - increased by 2 sizes
legend = plt.legend(loc='upper left', fontsize=16, frameon=True, fancybox=True, shadow=True)
# Make legend text bold
for text in legend.get_texts():
    text.set_fontweight('bold')

# Set axis limits
plt.xlim(-0.1, 0.6)
plt.ylim(-0.1, 0.6)

# Add diagonal reference line (after legend to avoid including it)
plt.plot([-0.1, 0.6], [-0.1, 0.6], 'k--', alpha=0.5, linewidth=2)

# Annotate genes with regular correlation > 0.48
high_corr_genes = plot_df[plot_df['Regular_Correlation_Median'] > 0.48]
print(f"\nGenes with regular correlation > 0.48: {len(high_corr_genes)}")

for i, (idx, row) in enumerate(high_corr_genes.iterrows()):
    gene_name = row['Gene']
    x_pos = row['Regular_Correlation_Median']
    y_pos = row['Partial_Correlation_Median']
    
    # Special positioning for EPCAM gene - move it slightly to the right
    if gene_name == 'EPCAM':
        x_offset = 15  # Move right
        y_offset = -8  # Keep same vertical position
    else:
        x_offset = 0   # Default horizontal position
        y_offset = -8  # Default vertical position
    
    # Annotation moved down to avoid overlapping with points
    plt.annotate(gene_name, (x_pos, y_pos), 
                xytext=(x_offset, y_offset), textcoords='offset points',
                fontsize=12, fontweight='bold', 
                color='black',
                ha='center', va='top')
    
    print(f"  {gene_name}: Regular={x_pos:.3f}, Partial={y_pos:.3f}, Group={row['Gene_Group']}")

# Grid
plt.grid(True, alpha=0.3, linewidth=1)

# Tight layout
plt.tight_layout()

# Save plot
output_path = "/Visual/180_Genes_Correlation_Scatter_Plot_Virchow2.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()




### UNI2h Scatter Plot

print("\n" + "="*80)
print("Creating UNI2h scatter plot")
print("="*80)

# Load UNI2h correlation data (1 row CSV)
uni2h_regular_df = pd.read_csv("/Gene_Expression_Prediction/180genes_UNI2h_regular_correlations.csv", index_col=0)
uni2h_partial_df = pd.read_csv("/Gene_Expression_Prediction/180genes_UNI2h_partial_correlations_clr.csv", index_col=0)

print(f"UNI2h Regular correlations shape: {uni2h_regular_df.shape}")
print(f"UNI2h Partial correlations shape: {uni2h_partial_df.shape}")

# Get correlation values (single row, so use .iloc[0])
uni2h_genes = uni2h_regular_df.columns.tolist()
uni2h_regular_corr = uni2h_regular_df.iloc[0]  # First (and only) row
uni2h_partial_corr = uni2h_partial_df.iloc[0]  # First (and only) row

print(f"Using {len(uni2h_genes)} genes for UNI2h")

# Create gene groups (same logic as Virchow2)
uni2h_gene_groups = []
uni2h_gene_colors = []

for i, gene in enumerate(uni2h_genes):
    if i < 30:
        uni2h_gene_groups.append('Tumor Markers')
        uni2h_gene_colors.append(color_mapping['Tumor Markers'])
    elif i < 60:
        uni2h_gene_groups.append('Normal Epithelial Markers')
        uni2h_gene_colors.append(color_mapping['Normal Epithelial Markers'])
    elif i < 90:
        uni2h_gene_groups.append('T Markers')
        uni2h_gene_colors.append(color_mapping['T Markers'])
    elif i < 120:
        uni2h_gene_groups.append('pan-APC Markers')
        uni2h_gene_colors.append(color_mapping['pan-APC Markers'])
    elif i < 150:
        uni2h_gene_groups.append('Stromal Markers')
        uni2h_gene_colors.append(color_mapping['Stromal Markers'])
    else:  # 150-179
        uni2h_gene_groups.append('Highly Variable Genes')
        uni2h_gene_colors.append(color_mapping['Highly Variable Genes'])

# Create DataFrame for UNI2h plotting
uni2h_plot_df = pd.DataFrame({
    'Gene': uni2h_genes,
    'Regular_Correlation': uni2h_regular_corr,
    'Partial_Correlation': uni2h_partial_corr,
    'Gene_Group': uni2h_gene_groups,
    'Color': uni2h_gene_colors
})

# Set fixed point size (larger)
uni2h_plot_df['Point_Size'] = 150

print(f"UNI2h Gene group distribution:")
print(uni2h_plot_df['Gene_Group'].value_counts())

# Create UNI2h scatter plot
plt.figure(figsize=(12, 10))

# Plot each group separately for legend
for group in ['Tumor Markers', 'Normal Epithelial Markers', 'T Markers', 'pan-APC Markers', 'Stromal Markers', 'Highly Variable Genes']:
    group_data = uni2h_plot_df[uni2h_plot_df['Gene_Group'] == group]
    if len(group_data) > 0:
        # Special case for Highly Variable Genes - show n=30 instead of actual count
        if group == 'Highly Variable Genes':
            label_text = f'{group} (n=30)'
        else:
            label_text = f'{group} (n={len(group_data)})'
            
        plt.scatter(
            group_data['Regular_Correlation'],
            group_data['Partial_Correlation'],
            s=group_data['Point_Size'],
            c=color_mapping[group],
            alpha=0.7,
            label=label_text,
            edgecolors='black',
            linewidth=0.5
        )

# Set labels and title (increased by 2 sizes)
plt.xlabel('Regular Pearson Correlation', fontsize=24, fontweight='bold')
plt.ylabel('Partial Pearson Correlation', fontsize=24, fontweight='bold')
plt.title('Gene Expression Prediction (UNI2-h)', fontsize=26, fontweight='bold')

# Styling with larger fonts and thicker borders
plt.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontweight('bold')

# Add thick black border around the plot
for spine in plt.gca().spines.values():
    spine.set_linewidth(3)
    spine.set_edgecolor('black')

# Legend with bold text (increased by 2 sizes)
legend = plt.legend(loc='upper left', fontsize=16, frameon=True, fancybox=True, shadow=True)
for text in legend.get_texts():
    text.set_fontweight('bold')

# Set axis limits
plt.xlim(-0.1, 0.6)
plt.ylim(-0.1, 0.6)

# Add diagonal reference line
plt.plot([-0.1, 0.6], [-0.1, 0.6], 'k--', alpha=0.5, linewidth=2)

# Annotate specific genes
target_genes = ['ID1', 'S100A6', 'CA2', 'TPM2', 'EPCAM', 'IGKC', 'CDX2']
uni2h_target_genes = uni2h_plot_df[uni2h_plot_df['Gene'].isin(target_genes)]
print(f"\nUNI2h target genes found: {len(uni2h_target_genes)}")

for i, (idx, row) in enumerate(uni2h_target_genes.iterrows()):
    gene_name = row['Gene']
    x_pos = row['Regular_Correlation']
    y_pos = row['Partial_Correlation']
    
    # Special positioning for EPCAM gene
    if gene_name == 'EPCAM':
        x_offset = 15
        y_offset = -8
    else:
        x_offset = 0
        y_offset = -8
    
    plt.annotate(gene_name, (x_pos, y_pos), 
                xytext=(x_offset, y_offset), textcoords='offset points',
                fontsize=12, fontweight='bold', 
                color='black',
                ha='center', va='top')
    
    print(f"  {gene_name}: Regular={x_pos:.3f}, Partial={y_pos:.3f}, Group={row['Gene_Group']}")

# Grid and layout
plt.grid(True, alpha=0.3, linewidth=1)
plt.tight_layout()

# Save UNI2h plot
uni2h_output_path = "Colorectal_Cancer_HE_patches/Visual/180_Genes_Correlation_Scatter_Plot_UNI2h.png"
plt.savefig(uni2h_output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()






print("\n" + "="*80)
print("Creating ResNet50 scatter plot")
print("="*80)

# Load ResNet50 correlation data (1 row CSV)
resnet50_regular_df = pd.read_csv("/Gene_Expression_Prediction/180genes_ResNet50_regular_correlations.csv", index_col=0)
resnet50_partial_df = pd.read_csv("/Gene_Expression_Prediction/180genes_ResNet50_partial_correlations_clr.csv", index_col=0)

print(f"ResNet50 Regular correlations shape: {resnet50_regular_df.shape}")
print(f"ResNet50 Partial correlations shape: {resnet50_partial_df.shape}")

# Get correlation values (single row, so use .iloc[0])
resnet50_genes = resnet50_regular_df.columns.tolist()
resnet50_regular_corr = resnet50_regular_df.iloc[0]  # First (and only) row
resnet50_partial_corr = resnet50_partial_df.iloc[0]  # First (and only) row

print(f"Using {len(resnet50_genes)} genes for ResNet50")

# Create gene groups (same logic as Virchow2)
resnet50_gene_groups = []
resnet50_gene_colors = []

for i, gene in enumerate(resnet50_genes):
    if i < 30:
        resnet50_gene_groups.append('Tumor Markers')
        resnet50_gene_colors.append(color_mapping['Tumor Markers'])
    elif i < 60:
        resnet50_gene_groups.append('Normal Epithelial Markers')
        resnet50_gene_colors.append(color_mapping['Normal Epithelial Markers'])
    elif i < 90:
        resnet50_gene_groups.append('T Markers')
        resnet50_gene_colors.append(color_mapping['T Markers'])
    elif i < 120:
        resnet50_gene_groups.append('pan-APC Markers')
        resnet50_gene_colors.append(color_mapping['pan-APC Markers'])
    elif i < 150:
        resnet50_gene_groups.append('Stromal Markers')
        resnet50_gene_colors.append(color_mapping['Stromal Markers'])
    else:  # 150-179
        resnet50_gene_groups.append('Highly Variable Genes')
        resnet50_gene_colors.append(color_mapping['Highly Variable Genes'])

# Create DataFrame for ResNet50 plotting
resnet50_plot_df = pd.DataFrame({
    'Gene': resnet50_genes,
    'Regular_Correlation': resnet50_regular_corr,
    'Partial_Correlation': resnet50_partial_corr,
    'Gene_Group': resnet50_gene_groups,
    'Color': resnet50_gene_colors
})

# Set fixed point size (larger)
resnet50_plot_df['Point_Size'] = 150

print(f"ResNet50 Gene group distribution:")
print(resnet50_plot_df['Gene_Group'].value_counts())

# Create ResNet50 scatter plot
plt.figure(figsize=(12, 10))

# Plot each group separately for legend
for group in ['Tumor Markers', 'Normal Epithelial Markers', 'T Markers', 'pan-APC Markers', 'Stromal Markers', 'Highly Variable Genes']:
    group_data = resnet50_plot_df[resnet50_plot_df['Gene_Group'] == group]
    if len(group_data) > 0:
        # Special case for Highly Variable Genes - show n=30 instead of actual count
        if group == 'Highly Variable Genes':
            label_text = f'{group} (n=30)'
        else:
            label_text = f'{group} (n={len(group_data)})'
            
        plt.scatter(
            group_data['Regular_Correlation'],
            group_data['Partial_Correlation'],
            s=group_data['Point_Size'],
            c=color_mapping[group],
            alpha=0.7,
            label=label_text,
            edgecolors='black',
            linewidth=0.5
        )

# Set labels and title (increased by 2 sizes)
plt.xlabel('Regular Pearson Correlation', fontsize=24, fontweight='bold')
plt.ylabel('Partial Pearson Correlation', fontsize=24, fontweight='bold')
plt.title('Gene Expression Prediction (ResNet50)', fontsize=26, fontweight='bold')

# Styling with larger fonts and thicker borders
plt.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontweight('bold')

# Add thick black border around the plot
for spine in plt.gca().spines.values():
    spine.set_linewidth(3)
    spine.set_edgecolor('black')

# Legend with bold text (increased by 2 sizes)
legend = plt.legend(loc='upper left', fontsize=16, frameon=True, fancybox=True, shadow=True)
for text in legend.get_texts():
    text.set_fontweight('bold')

# Set axis limits
plt.xlim(-0.1, 0.6)
plt.ylim(-0.1, 0.6)

# Add diagonal reference line
plt.plot([-0.1, 0.6], [-0.1, 0.6], 'k--', alpha=0.5, linewidth=2)

# Annotate specific genes
target_genes = ['ID1', 'S100A6', 'CA2', 'TPM2', 'EPCAM', 'IGKC', 'CDX2']
resnet50_target_genes = resnet50_plot_df[resnet50_plot_df['Gene'].isin(target_genes)]
print(f"\nResNet50 target genes found: {len(resnet50_target_genes)}")

for i, (idx, row) in enumerate(resnet50_target_genes.iterrows()):
    gene_name = row['Gene']
    x_pos = row['Regular_Correlation']
    y_pos = row['Partial_Correlation']
    
    # Special positioning for EPCAM gene
    if gene_name == 'EPCAM':
        x_offset = 15
        y_offset = -8
    else:
        x_offset = 0
        y_offset = -8
    
    plt.annotate(gene_name, (x_pos, y_pos), 
                xytext=(x_offset, y_offset), textcoords='offset points',
                fontsize=12, fontweight='bold', 
                color='black',
                ha='center', va='top')
    
    print(f"  {gene_name}: Regular={x_pos:.3f}, Partial={y_pos:.3f}, Group={row['Gene_Group']}")

# Grid and layout
plt.grid(True, alpha=0.3, linewidth=1)
plt.tight_layout()

# Save ResNet50 plot
resnet50_output_path = "/Visual/180_Genes_Correlation_Scatter_Plot_ResNet50.png"
plt.savefig(resnet50_output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()







