"""
STPath-COAD: Gene Expression Prediction - Visualization
========================================================

This script generates visualizations for gene expression prediction results
from foundation model features.

Author: Saishi Cui
Date: January 2026

Purpose:
- Create correlation scatter plots for Virchow2, UNI2-h, and ResNet50
- Generate boxplot comparing all 6 foundation models
- Create Virchow2 vs ResNet50 comparison scatter plot  
- Perform deep dive analysis for specific genes (e.g., S100A6)
- Visualize partial correlation vs regular correlation

Prerequisites:
- Must run COAD_GeneExpression_Prediction.py first to generate training results
- Results should be in: Colorectal_Cancer_HE_patches/Gene_Expression_Prediction/

Figures are saved to: Colorectal_Cancer_HE_patches/Visual/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import xgboost as xgb
import os


### STEP 4: Create correlation scatter plot for 180 genes

print("\n" + "="*80)
print("STEP 4: Creating correlation scatter plot for 180 genes")
print("="*80)

# Load correlation data
regular_corr_df = pd.read_csv("Colorectal_Cancer_HE_patches/Gene_Expression_Prediction/180genes_regular_correlations.csv", index_col=0)
partial_corr_df = pd.read_csv("Colorectal_Cancer_HE_patches/Gene_Expression_Prediction/180genes_partial_correlations_clr.csv", index_col=0)

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
output_path = "Colorectal_Cancer_HE_patches/Visual/180_Genes_Correlation_Scatter_Plot_Virchow2.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()




### UNI2h Scatter Plot

print("\n" + "="*80)
print("Creating UNI2h scatter plot")
print("="*80)

# Load UNI2h correlation data (1 row CSV)
uni2h_regular_df = pd.read_csv("Colorectal_Cancer_HE_patches/Gene_Expression_Prediction/180genes_UNI2h_regular_correlations.csv", index_col=0)
uni2h_partial_df = pd.read_csv("Colorectal_Cancer_HE_patches/Gene_Expression_Prediction/180genes_UNI2h_partial_correlations_clr.csv", index_col=0)

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




### ResNet50 Scatter Plot

print("\n" + "="*80)
print("Creating ResNet50 scatter plot")
print("="*80)

# Load ResNet50 correlation data (1 row CSV)
resnet50_regular_df = pd.read_csv("Colorectal_Cancer_HE_patches/Gene_Expression_Prediction/180genes_ResNet50_regular_correlations.csv", index_col=0)
resnet50_partial_df = pd.read_csv("Colorectal_Cancer_HE_patches/Gene_Expression_Prediction/180genes_ResNet50_partial_correlations_clr.csv", index_col=0)

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
resnet50_output_path = "Colorectal_Cancer_HE_patches/Visual/180_Genes_Correlation_Scatter_Plot_ResNet50.png"
plt.savefig(resnet50_output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()





