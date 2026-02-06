"""
STPath-COAD: UMAP Analysis with Contribution Regression for Figure 3
====================================================================

This script performs UMAP dimensionality reduction analysis with regression of 
confounding factors for Figure 3. Visualizes feature contributions and corrected embeddings.

Author: Saishi Cui
Date: December 2025

Purpose: Generate UMAP visualizations with contribution analysis, regression corrections,
and comprehensive feature space analysis for Figure 3 (also Figure S5-S8) of the paper.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import umap
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import time
import json
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



### Figure 3 (Panel A) Original UMAP Visualization (with all features)
def create_umap_visualizations(data_path, cell_type, foundation_model, legend_on = False, save_dir="Visual/UMAPs/"):
    """
    Create UMAP dimensionality reduction visualization, generating three differently colored plots
    """
    # Load data
    print(f"Loading data from {data_path}")
    data = torch.load(data_path)
    
    # Extract data
    embeddings = data['embeddings']
    sample_ids = data['sample_ids']
    individual_ids = data['individual_ids']
    celltype_proportions = data['celltype_proportions']
    
    # Convert to numpy arrays (if needed)
    if torch.is_tensor(embeddings):
        embeddings = embeddings.numpy()
    if torch.is_tensor(celltype_proportions):
        celltype_proportions = celltype_proportions.numpy()
    
    print(f"Data shape: {embeddings.shape}")
    print(f"Number of samples: {len(sample_ids)}")
    
    # Perform UMAP dimensionality reduction
    print("Performing UMAP dimensionality reduction...")
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
 
    # Plot 2: colored by individual_id
    print("Creating plot 2: colored by individual_id")
    plt.figure(figsize=(12, 11))  # Increase width to accommodate legend
    
    # Get unique individual_ids and sort
    unique_individual_ids = sorted(list(set(individual_ids)))
    n_individuals = len(unique_individual_ids)
    
    # Create better color mapping
    if n_individuals <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_individuals))
    elif n_individuals <= 15:
        # Use color combinations with stronger contrast
        colors = plt.cm.Set1(np.linspace(0, 1, min(n_individuals, 9)))
        if n_individuals > 9:
            extra_colors = plt.cm.Dark2(np.linspace(0, 1, n_individuals-9))
            colors = np.vstack([colors, extra_colors])
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, n_individuals))
    
    individual_to_color = {ind: colors[i] for i, ind in enumerate(unique_individual_ids)}
    
    # Plot points
    for individual_id in unique_individual_ids:
        mask = np.array([iid == individual_id for iid in individual_ids])
        # Deepen color: multiply original color by 0.8 to make it darker
        deep_color = individual_to_color[individual_id] * 0.8
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[deep_color], label=individual_id, alpha=0.8, s=10, edgecolors='black', linewidths=0.3)
    

    # Add thick black borders like hexagon plots
    for spine in plt.gca().spines.values():
        spine.set_linewidth(5)
        spine.set_color('black')
    plt.xticks([])
    plt.yticks([])
    
    # Add lower left corner coordinate axis indication
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Use Axes coordinate system to ensure arrows and labels are inside the plot and close
    x0, y0 = 0.10, 0.10
    arrow_len = 0.14
    # Horizontal arrow
    ax.annotate('', xy=(x0 + arrow_len, y0), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'),
                xycoords='axes fraction', textcoords='axes fraction')
    ax.text(x0, y0 - 0.02, 'UMAP 1',
            ha='left', va='top', fontsize=30, fontweight='bold', transform=ax.transAxes)
    # Vertical arrow
    ax.annotate('', xy=(x0, y0 + arrow_len), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'),
                xycoords='axes fraction', textcoords='axes fraction')
    ax.text(x0 - 0.02, y0, 'UMAP 2',
            ha='right', va='bottom', fontsize=30, fontweight='bold', rotation=90, transform=ax.transAxes)
    
    # Add legend to the right outside the plot
    if legend_on == True:
        legend = plt.legend(
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            fontsize=14,
            ncol=1,
            framealpha=0.9,
            edgecolor='black',
            markerscale=2
        )
        # Set legend border width
        legend.get_frame().set_linewidth(1)
        # Make legend text bold and black
        for text in legend.get_texts():
            text.set_fontweight('bold')
            text.set_color('black')
    else:
        legend = None
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{foundation_model}_{cell_type}_UMAP_by_individual_id.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: colored by celltype_proportions
    print("Creating plot 3: colored by celltype_proportions")
    plt.figure(figsize=(12, 11))  # Change to square
    
    # Use clearer continuous color mapping
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=celltype_proportions, cmap='plasma', alpha=0.8, s=10, 
                         edgecolors='black', linewidths=0.3)
    
    # Add more detailed colorbar
    if legend_on == True:
        cbar = plt.colorbar(scatter, label=f'{cell_type} Proportion', shrink=0.8)
        cbar.ax.tick_params(labelsize=30, colors='black')
        cbar.set_label(f'{cell_type} Proportion', fontsize=30, fontweight='bold', color='black')
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_color('black')
    else:
        cbar = None

    # Add thick black borders like hexagon plots
    for spine in plt.gca().spines.values():
        spine.set_linewidth(5)
        spine.set_color('black')
    plt.xticks([])
    plt.yticks([])
    
    # Add lower left corner coordinate axis indication
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Use Axes coordinate system to ensure arrows and labels are inside the plot and close
    x0, y0 = 0.10, 0.10
    arrow_len = 0.14
    # Horizontal arrow
    ax.annotate('', xy=(x0 + arrow_len, y0), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'),
                xycoords='axes fraction', textcoords='axes fraction')
    ax.text(x0, y0 - 0.02, 'UMAP 1',
            ha='left', va='top', fontsize=30, fontweight='bold', transform=ax.transAxes)
    # Vertical arrow
    ax.annotate('', xy=(x0, y0 + arrow_len), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'),
                xycoords='axes fraction', textcoords='axes fraction')
    ax.text(x0 - 0.02, y0, 'UMAP 2',
            ha='right', va='bottom', fontsize=30, fontweight='bold', rotation=90, transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{foundation_model}_{cell_type}_UMAP_by_proportions.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return


for cell_type in ["Cancer Cells", "Stromal Cells", "Normal Epithelial Cells", "T Cells", "Other Immune Cells"]:     
    create_umap_visualizations(data_path = f"Training_features/{cell_type}_training_precomputed_features_ResNet50.pt", cell_type = cell_type, foundation_model = "ResNet50")
    create_umap_visualizations(data_path = f"Training_features/{cell_type}_training_precomputed_features_Conch.pt", cell_type = cell_type, foundation_model = "Conch")
    create_umap_visualizations(data_path = f"Training_features/{cell_type}_training_precomputed_features_ProvGigapath.pt", cell_type = cell_type, foundation_model = "Prov-GigaPath")
    create_umap_visualizations(data_path = f"Training_features/{cell_type}_training_precomputed_features_UNI2h.pt", cell_type = cell_type, foundation_model = "UNI2-h")
    create_umap_visualizations(data_path = f"Training_features/{cell_type}_training_precomputed_features_Virchow.pt", cell_type = cell_type, foundation_model = "Virchow")
    create_umap_visualizations(data_path = f"Training_features/{cell_type}_training_precomputed_features_Virchow2.pt", cell_type = cell_type, foundation_model = "Virchow2")


for cell_type in ["Cancer Cells", "Stromal Cells", "Normal Epithelial Cells", "T Cells", "Other Immune Cells"]:     
    create_umap_visualizations(data_path = f"/Training_features/{cell_type}_training_precomputed_features_UNI2h.pt", cell_type = cell_type, foundation_model = "UNI2-h", legend_on = True, save_dir="Colorectal_Cancer_HE_patches/Visual/UMAPs_legend/")





##  Figure 3 (Panel A) UMAP Visualization (with top 10 features)
def create_umap_visualizations_Xgboost(foundation_model, cell_type, if_legend = True, dpi = 600):



    result_dir = f"xgboost_prediction/{cell_type}_{foundation_model}_individual_level_ratio100/xgboost_results_individual_level.pt"
    result_data = torch.load(result_dir)
    
    # Correct way to build feature importance DataFrame
    feature_importance_dfs = []
    for i in range(len(result_data["feature_importances"])):
        df_temp = pd.DataFrame.from_dict(result_data["feature_importances"][i], 
                                       orient='index', columns=[f'fold_{i}'])
        feature_importance_dfs.append(df_temp)
    

    feature_importance_combined = pd.concat(feature_importance_dfs, axis=1)
    feature_importance_combined.fillna(0, inplace=True)

    feature_importance_df = pd.DataFrame({
        'feature': feature_importance_combined.index,
        'importance': feature_importance_combined.mean(axis=1)
    }).reset_index(drop=True)

    feature_importance_df.sort_values(by="importance", ascending=False, inplace=True)



    # 1. Read data
    data_path = f"Training_features/{cell_type}_training_precomputed_features_{foundation_model}.pt"
    print(f"Loading data from {data_path}")
    data = torch.load(data_path)
    embeddings = data['embeddings']
    sample_ids = data['sample_ids']
    individual_ids = data['individual_ids']
    celltype_proportions = data['celltype_proportions']

    # 2. Only keep the top 10 important features
    top10_features = feature_importance_df['feature'].iloc[:10].tolist()
    # feature name like 'f1220', need to convert to int index
    top10_indices = [int(f[1:]) for f in top10_features]
    if torch.is_tensor(embeddings):
        embeddings = embeddings.numpy()
    embeddings_top10 = embeddings[:, top10_indices]

    if torch.is_tensor(celltype_proportions):
        celltype_proportions = celltype_proportions.numpy()

    print(f"Data shape (top 10 features): {embeddings_top10.shape}")
    print(f"Number of samples: {len(sample_ids)}")

    # 3. UMAP dimensionality reduction
    print("Performing UMAP dimensionality reduction...")
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_2d = reducer.fit_transform(embeddings_top10)

    save_dir = "Colorectal_Cancer_HE_patches/Visual/UMAPs_Xgboost/"
    os.makedirs(save_dir, exist_ok=True)

  

    # 4. Plot 2: colored by individual_id
    print("Creating plot 2: colored by individual_id")
    plt.figure(figsize=(12, 11))
    unique_individual_ids = sorted(list(set(individual_ids)))
    n_individuals = len(unique_individual_ids)
    if n_individuals <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_individuals))
    elif n_individuals <= 15:
        colors = plt.cm.Set1(np.linspace(0, 1, min(n_individuals, 9)))
        if n_individuals > 9:
            extra_colors = plt.cm.Dark2(np.linspace(0, 1, n_individuals-9))
            colors = np.vstack([colors, extra_colors])
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, n_individuals))
    individual_to_color = {ind: colors[i] for i, ind in enumerate(unique_individual_ids)}
    for individual_id in unique_individual_ids:
        mask = np.array([iid == individual_id for iid in individual_ids])
        deep_color = individual_to_color[individual_id] * 0.8
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                c=[deep_color], label=individual_id, alpha=0.8, s=10, edgecolors='black', linewidths=0.3)
    # Add thick black borders like hexagon plots
    for spine in plt.gca().spines.values():
        spine.set_linewidth(5)
        spine.set_color('black')
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0, y0 = 0.10, 0.10
    arrow_len = 0.14
    # Horizontal arrow
    ax.annotate('', xy=(x0 + arrow_len, y0), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'),
                xycoords='axes fraction', textcoords='axes fraction')
    ax.text(x0, y0 - 0.02, 'UMAP 1',
            ha='left', va='top', fontsize=30, fontweight='bold', transform=ax.transAxes)
    # Vertical arrow
    ax.annotate('', xy=(x0, y0 + arrow_len), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'),
                xycoords='axes fraction', textcoords='axes fraction')
    ax.text(x0 - 0.02, y0, 'UMAP 2',
            ha='right', va='bottom', fontsize=30, fontweight='bold', rotation=90, transform=ax.transAxes)
    plt.tight_layout()
    if if_legend:
        legend = plt.legend(
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            ncol=1,
            framealpha=0.9,
            edgecolor='black',
            markerscale=2
        )
        legend.get_frame().set_linewidth(1)
        for text in legend.get_texts():
            text.set_fontweight('bold')
            text.set_color('black')
    plt.savefig(f"{save_dir}/{foundation_model}_{cell_type}_UMAP_by_individual_id_top10.png", 
                dpi=dpi, bbox_inches='tight')
    plt.close()

    # ----------- Plot 3: colored by celltype_proportions -----------
    print("Creating plot 3: colored by celltype_proportions")
    plt.figure(figsize=(12, 11))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                        c=celltype_proportions, cmap='plasma', alpha=0.8, s=10, 
                        edgecolors='black', linewidths=0.3)
        # Add more detailed colorbar
    if if_legend == True:
        cbar = plt.colorbar(scatter, label=f'{cell_type} Proportion', shrink=0.8)
        cbar.ax.tick_params(labelsize=30, colors='black')
        cbar.set_label(f'{cell_type} Proportion', fontsize=30, fontweight='bold', color='black')
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_color('black')
    else:
        cbar = None
    # Add thick black borders like hexagon plots
    for spine in plt.gca().spines.values():
        spine.set_linewidth(5)
        spine.set_color('black')
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0, y0 = 0.10, 0.10
    arrow_len = 0.14
    # Horizontal arrow
    ax.annotate('', xy=(x0 + arrow_len, y0), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'),
                xycoords='axes fraction', textcoords='axes fraction')
    ax.text(x0, y0 - 0.02, 'UMAP 1',
            ha='left', va='top', fontsize=30, fontweight='bold', transform=ax.transAxes)
    # Vertical arrow
    ax.annotate('', xy=(x0, y0 + arrow_len), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'),
                xycoords='axes fraction', textcoords='axes fraction')
    ax.text(x0 - 0.02, y0, 'UMAP 2',
            ha='right', va='bottom', fontsize=30, fontweight='bold', rotation=90, transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{foundation_model}_{cell_type}_UMAP_by_proportions_top10.png", 
                dpi=dpi, bbox_inches='tight')
    plt.close()


for cell_type in ["Cancer Cells", "Stromal Cells", "Normal Epithelial Cells", "T Cells", "Other Immune Cells"]:
    for foundation_model in ["ResNet50", "UNI2-h", "Virchow2", "Virchow", "Prov-GigaPath", "Conch"]:
        create_umap_visualizations_Xgboost(foundation_model = foundation_model, cell_type = cell_type, if_legend = False, dpi = 300)








## UNI2h_data, Virchow_data, Virchow2_data, ProvGigapath_data, Conch_data,
## 1536 + 2560 + 2560 + 1536 + 512



### Figure 3 (Panel B) Create barplot of feature importance contribution to the combined model
def Feature_Importance_Xgboost_Barplot(cell_type):


    result_dir = f"/xgboost_prediction/{cell_type}_Combined_individual_level_ratio100/xgboost_results_individual_level.pt"
    result_data = torch.load(result_dir)


    feature_importance_dfs = []
    for i in range(len(result_data["feature_importances"])):
        df_temp = pd.DataFrame.from_dict(result_data["feature_importances"][i], 
                                       orient='index', columns=[f'fold_{i}'])
        feature_importance_dfs.append(df_temp)

    feature_importance_combined = pd.concat(feature_importance_dfs, axis=1)
    feature_importance_combined.fillna(0, inplace=True)

    feature_importance_df = pd.DataFrame({
        'feature': feature_importance_combined.index,
        'importance': feature_importance_combined.mean(axis=1)
    }).reset_index(drop=True)

    feature_importance_df.sort_values(by="importance", ascending=False, inplace=True)


    n_uni2h = 461
    n_virchow = 768
    n_virchow2 = 768
    n_provgigapath = 461
    n_conch = 154


    idx_uni2h = (0, n_uni2h)
    idx_virchow = (n_uni2h, n_uni2h + n_virchow)
    idx_virchow2 = (n_uni2h + n_virchow, n_uni2h + n_virchow + n_virchow2)
    idx_provgigapath = (n_uni2h + n_virchow + n_virchow2, n_uni2h + n_virchow + n_virchow2 + n_provgigapath)
    idx_conch = (n_uni2h + n_virchow + n_virchow2 + n_provgigapath, n_uni2h + n_virchow + n_virchow2 + n_provgigapath + n_conch)


    # calculate the model contribution
    def which_model(idx):
        if idx_uni2h[0] <= idx < idx_uni2h[1]:
            return 'UNI2-h'
        elif idx_virchow[0] <= idx < idx_virchow[1]:
            return 'Virchow'
        elif idx_virchow2[0] <= idx < idx_virchow2[1]:
            return 'Virchow2'
        elif idx_provgigapath[0] <= idx < idx_provgigapath[1]:
            return 'Prov-GigaPath'
        elif idx_conch[0] <= idx < idx_conch[1]:
            return 'Conch'
        else:
            return 'Unknown'

 
    feature_importance_df['idx'] = feature_importance_df['feature'].apply(lambda x: int(x[1:]))
    feature_importance_df['model'] = feature_importance_df['idx'].apply(which_model)

    # calculate the total importance of each model
    model_contrib = feature_importance_df.groupby('model')['importance'].sum()
    total_contrib = model_contrib.sum()
    model_percent = (model_contrib / total_contrib * 100).sort_values(ascending=False)

    model_contrib_normalized = feature_importance_df.groupby('model')['importance'].sum()/np.array([n_conch, n_provgigapath, n_uni2h, n_virchow, n_virchow2])
    total_contrib_normalized = model_contrib_normalized.sum()
    model_percent_normalized = (model_contrib_normalized / total_contrib_normalized * 100).sort_values(ascending=False)




    # fix the model order
    models = ['Conch', 'Prov-GigaPath', 'UNI2-h', 'Virchow', 'Virchow2']

    # make sure the percentage data is in the same order as the y-axis
    orig = model_percent.reindex(models).values
    norm = model_percent_normalized.reindex(models).values
    y = np.arange(len(models))

    plt.figure(figsize=(10, 5))
    bar_height = 0.35
    gap = 0.08

    bars1 = plt.barh(y - bar_height/2 - gap/2, orig, height=bar_height, color='#1f77b4', label='Original Contribution', edgecolor='black', linewidth=1.5)
    bars2 = plt.barh(y + bar_height/2 + gap/2, norm, height=bar_height, color='#ff7f0e', label='Normalized Contribution', edgecolor='black', linewidth=1.5, hatch='///')

    for i, v in enumerate(orig):
        plt.text(v + 1, y[i] - bar_height/2 - gap/2, f'{v:.1f}%', va='center', fontsize=18, fontweight='bold')
    for i, v in enumerate(norm):
        plt.text(v + 1, y[i] + bar_height/2 + gap/2, f'{v:.1f}%', va='center', fontsize=18, fontweight='bold')

    plt.yticks(y, models, fontsize=18, fontweight='bold')
    plt.xticks(fontsize=18, fontweight='bold', color='black')

    plt.xlim(0, max(orig.max(), norm.max()) * 1.2)

    # Remove legend
    # legend = plt.legend(
    #     loc='lower right',
    #     fontsize=14,
    #     frameon=True,
    #     prop={'weight': 'bold', 'size': 14}
    # )
    # legend.get_frame().set_linewidth(1.5)
    # for text in legend.get_texts():
    #     text.set_fontweight('bold')
    #     text.set_color('black')

    # Add thick black borders like hexagon plots
    ax = plt.gca()
    ax.spines['top'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')

    plt.tight_layout()
    plt.savefig(f"/Visual/Barplot_Model_Contribution_{cell_type}.png", dpi=600, bbox_inches='tight')
    plt.close()

    return feature_importance_df



Cancer_cells_feature_importance_df = Feature_Importance_Xgboost_Barplot(cell_type = "Cancer Cells")
Stromal_cells_feature_importance_df = Feature_Importance_Xgboost_Barplot(cell_type = "Stromal Cells")
Normal_Epithelial_cells_feature_importance_df = Feature_Importance_Xgboost_Barplot(cell_type = "Normal Epithelial Cells")
T_cells_feature_importance_df = Feature_Importance_Xgboost_Barplot(cell_type = "T Cells")
Other_Immune_cells_feature_importance_df = Feature_Importance_Xgboost_Barplot(cell_type = "Other Immune Cells")


##### Figure 3 (Panel C) Comparison of predictive performance and computational efficiency across multiple machine learning methods.



# Data paths
XGBOOST_BASE = "/xgboost_prediction"
MODEL_COMPARISON_BASE = "/model_comparison"

# Output path
OUTPUT_DIR = "/Visual"

# Cell types to compare
CELL_TYPES = ["Cancer Cells", "Stromal Cells"]

# ML Models to compare (order: MLP, Lasso, RandomForest, XGBoost)
ML_MODELS = ["MLP", "Lasso", "RandomForest", "XGBoost"]

# Color palette for ML models (distinct colors)
ml_color_palette = {
    'MLP': '#96CEB4',        # Sage green
    'Lasso': '#4ECDC4',      # Teal
    'RandomForest': '#45B7D1',  # Sky blue
    'XGBoost': '#FF6B6B'     # Red
}

def load_individual_metrics():
    """Load and filter individual metrics from all models."""
    all_results = []
    
    for cell_type in CELL_TYPES:
        for ml_model in ML_MODELS:
            # Build path based on model type
            if ml_model == "XGBoost":
                # XGBoost uses ratio100 suffix
                csv_path = f"{XGBOOST_BASE}/{cell_type}_Virchow2_individual_level_ratio100/individual_metrics_individual_level.csv"
            else:
                # Other models in model_comparison folder
                csv_path = f"{MODEL_COMPARISON_BASE}/{ml_model}/{cell_type}_Virchow2_individual_level/individual_metrics_individual_level.csv"
            
            try:
                df = pd.read_csv(csv_path)
                print(f"Loaded {ml_model} - {cell_type}: {len(df)} rows")
                
                # Apply Figure 4 Panel A filtering criteria
                for index, row in df.iterrows():
                    ct_range = row["Max_celltype_proportion"] - row["Min_celltype_proportion"]
                    
                    # Filtering: n_test_samples >= 200 and ct_range > 0.3
                    if row["n_test_samples"] >= 200 and ct_range > 0.3:
                        all_results.append({
                            "ml_model": ml_model,
                            "cell_type": cell_type,
                            "MAE": row["MAE"],
                            "Pearson_correlation": row["Pearson"],
                            "n_test_samples": row["n_test_samples"],
                            "Max_celltype_proportion": row["Max_celltype_proportion"],
                            "Min_celltype_proportion": row["Min_celltype_proportion"],
                            "ct_range": ct_range,
                            "Individual": row["Individual"]
                        })
            except FileNotFoundError:
                print(f"Warning: File not found: {csv_path}")
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
    
    return pd.DataFrame(all_results)


def create_boxplots(final_df):
    """Create boxplots comparing ML models."""
    
    # Set the figure style
    plt.style.use('default')
    
    # Define the metrics to plot
    metrics = ['MAE', 'Pearson_correlation']
    metric_titles = ['Mean Absolute Error (MAE)', 'Pearson Correlation']
    
    # Define ML model order (MLP, Lasso, RandomForest, XGBoost)
    ml_model_order = ['MLP', 'Lasso', 'RandomForest', 'XGBoost']
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Debug: Check data for current metric
        print(f"\nDEBUG - {metric} data check:")
        for cell_type in CELL_TYPES:
            cell_data = final_df[final_df['cell_type'] == cell_type]
            print(f"{cell_type}: {len(cell_data)} samples")
            if len(cell_data) > 0:
                print(f"  {metric} range: {cell_data[metric].min():.3f} - {cell_data[metric].max():.3f}")
        
        # Create boxplot
        box_plot = sns.boxplot(
            data=final_df,
            x='cell_type',
            y=metric,
            hue='ml_model',
            hue_order=ml_model_order,
            palette=ml_color_palette,
            ax=ax,
            showfliers=False,
            linewidth=3,
            legend=False
        )
        
        # Add stripplot for individual points
        sns.stripplot(
            data=final_df,
            x='cell_type',
            y=metric,
            hue='ml_model',
            hue_order=ml_model_order,
            palette=ml_color_palette,
            ax=ax,
            size=6,
            alpha=0.7,
            dodge=True,
            jitter=0.3,
            edgecolor='black',
            linewidth=0.5,
            legend=False
        )
        
        # Set labels
        ax.set_xlabel('')
        ax.set_ylabel(title, fontsize=22, fontweight='bold')
        
        # Set tick label style
        ax.tick_params(axis='x', rotation=0, labelsize=20, labelcolor='black',
                       width=2, length=6, colors='black')
        ax.tick_params(axis='y', labelsize=20, labelcolor='black',
                       width=2, length=6, colors='black')
        
        # Modify x-axis labels
        x_labels = []
        for label in ax.get_xticklabels():
            text = label.get_text().replace(' Cells', '')
            text = text.replace('Cancer', 'Tumor')
            x_labels.append(text)
            label.set_fontweight('bold')
        ax.set_xticklabels(x_labels, fontweight='bold')
        
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        
        # Set y-axis range (0 to 1 for both)
        ax.set_ylim(0, 1)
        
        # Add grid
        ax.grid(True, alpha=0.7, linestyle='--', linewidth=2, axis='y')
        
        # Add thick black borders
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_linewidth(5)
            ax.spines[spine].set_color('black')
        
        # Create legend on the right side (no title)
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=ml_color_palette[m], 
                                   edgecolor='black', linewidth=1.5)
                   for m in ml_model_order]
        legend = ax.legend(handles, ml_model_order,
                          loc='center left',
                          bbox_to_anchor=(1.02, 0.5),
                          fontsize=16,
                          frameon=True,
                          edgecolor='black')
        for text in legend.get_texts():
            text.set_fontweight('bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = f'{OUTPUT_DIR}/ML_model_comparison_{metric}_boxplot.png'
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
        
        plt.close()


def create_summary_table(final_df):
    """Create summary statistics table."""
    summary = final_df.groupby(['cell_type', 'ml_model']).agg({
        'MAE': ['mean', 'std'],
        'Pearson_correlation': ['mean', 'std']
    }).round(4)
    
    summary.columns = ['MAE_mean', 'MAE_std', 'Pearson_mean', 'Pearson_std']
    summary = summary.reset_index()
    
    print("\n=== Summary Statistics ===")
    print(summary.to_string(index=False))
    
    # Save summary
    summary.to_csv(f'{OUTPUT_DIR}/ML_model_comparison_summary.csv', index=False)
    
    return summary



print("Loading individual metrics...")
final_df = load_individual_metrics()

print(f"\nTotal filtered samples: {len(final_df)}")
print(f"Unique individuals: {final_df['Individual'].nunique()}")

# Create boxplots
print("\nCreating boxplots...")
create_boxplots(final_df)

# Create summary table
summary = create_summary_table(final_df)

print("\n=== Done ===")




# (Panel C right) 
FEATURES_DIR = "/Training_features"
OUTPUT_DIR = "/Visual"
CELL_TYPE = "Cancer Cells"
MODEL_NAME = "Virchow2"

# Number of LOO iterations to benchmark
N_BENCHMARK_FOLDS = 3

# Subsample ratio for benchmarking (to speed up)
SUBSAMPLE_RATIO = 0.1  # Use 10% of data for benchmarking

# Best params from grid search
LASSO_PARAMS = {
    'alpha': 7.196856730011514e-05, 
    'max_iter': 10000, 
    'tol': 0.0001, 
    'selection': 'cyclic'
}

RF_PARAMS = {
    'max_depth': None, 
    'max_features': 0.2, 
    'min_samples_leaf': 4, 
    'min_samples_split': 10, 
    'n_estimators': 300,
    'n_jobs': -1,
    'random_state': 42
}

MLP_PARAMS = {
    'batch_size': 256, 
    'dropout_rate': 0.2, 
    'hidden_dims': [512, 256], 
    'learning_rate': 0.0001,
    'epochs': 50
}


class MLPRegressor(nn.Module):
    """Simple MLP for regression."""
    def __init__(self, input_dim, hidden_dims=[512, 256], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()


def load_data():
    """Load features and labels with optional subsampling."""
    pt_file = f"{FEATURES_DIR}/{CELL_TYPE}_training_precomputed_features_{MODEL_NAME}.pt"
    data = torch.load(pt_file, weights_only=False)
    
    features = data['embeddings'].numpy()
    labels = data['celltype_proportions'].numpy()
    individual_ids = np.array(data['individual_ids'])
    
    # Subsample for faster benchmarking (maintain individual structure)
    if SUBSAMPLE_RATIO < 1.0:
        np.random.seed(42)
        unique_inds = np.unique(individual_ids)
        keep_mask = np.zeros(len(features), dtype=bool)
        
        for ind in unique_inds:
            ind_mask = individual_ids == ind
            ind_indices = np.where(ind_mask)[0]
            n_keep = max(1, int(len(ind_indices) * SUBSAMPLE_RATIO))
            keep_indices = np.random.choice(ind_indices, size=n_keep, replace=False)
            keep_mask[keep_indices] = True
        
        features = features[keep_mask]
        labels = labels[keep_mask]
        individual_ids = individual_ids[keep_mask]
    
    return features, labels, individual_ids


def train_lasso(X_train, y_train, X_test):
    """Train Lasso model with best params."""
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Use best params
    model = Lasso(**LASSO_PARAMS)
    model.fit(X_train_scaled, y_train)
    return model.predict(X_test_scaled)


def train_rf(X_train, y_train, X_test):
    """Train Random Forest model with best params."""
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def train_xgboost(X_train, y_train, X_test):
    """Train XGBoost model."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 0
    }
    
    model = xgb.train(params, dtrain, num_boost_round=100)
    return model.predict(dtest)


def train_mlp(X_train, y_train, X_test):
    """Train MLP model with best params."""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    # Create dataloader
    dataset = TensorDataset(X_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=MLP_PARAMS['batch_size'], shuffle=True)
    
    # Create model with best params
    model = MLPRegressor(
        X_train.shape[1], 
        hidden_dims=MLP_PARAMS['hidden_dims'],
        dropout=MLP_PARAMS['dropout_rate']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=MLP_PARAMS['learning_rate'])
    criterion = nn.MSELoss()
    
    # Train
    model.train()
    for epoch in range(MLP_PARAMS['epochs']):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
    
    # Predict
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_t).cpu().numpy()
    
    return predictions


def benchmark_models():
    """Run benchmark for all models."""
    print("Loading data...")
    features, labels, individual_ids = load_data()
    unique_individuals = np.unique(individual_ids)
    
    print(f"Dataset: {len(features)} samples, {len(unique_individuals)} individuals")
    print(f"Benchmarking {N_BENCHMARK_FOLDS} LOO folds...")
    print(f"\nUsing best params from grid search (no hyperparameter tuning)")
    
    # Store times
    times = {
        'Lasso': [],
        'RandomForest': [],
        'XGBoost': [],
        'MLP': []
    }
    
    # Benchmark each fold
    for fold_idx, test_individual in enumerate(unique_individuals[:N_BENCHMARK_FOLDS]):
        print(f"\nFold {fold_idx + 1}/{N_BENCHMARK_FOLDS}: Leave {test_individual} out")
        
        # Split data
        test_mask = individual_ids == test_individual
        train_mask = ~test_mask
        
        X_train = features[train_mask]
        y_train = labels[train_mask]
        X_test = features[test_mask]
        
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Benchmark Lasso
        print("  Training Lasso...", end=" ", flush=True)
        start = time.time()
        train_lasso(X_train, y_train, X_test)
        lasso_time = time.time() - start
        times['Lasso'].append(lasso_time)
        print(f"{lasso_time:.2f}s")
        
        # Benchmark RandomForest
        print("  Training RandomForest...", end=" ", flush=True)
        start = time.time()
        train_rf(X_train, y_train, X_test)
        rf_time = time.time() - start
        times['RandomForest'].append(rf_time)
        print(f"{rf_time:.2f}s")
        
        # Benchmark XGBoost
        print("  Training XGBoost...", end=" ", flush=True)
        start = time.time()
        train_xgboost(X_train, y_train, X_test)
        xgb_time = time.time() - start
        times['XGBoost'].append(xgb_time)
        print(f"{xgb_time:.2f}s")
        
        # Benchmark MLP
        print("  Training MLP...", end=" ", flush=True)
        start = time.time()
        train_mlp(X_train, y_train, X_test)
        mlp_time = time.time() - start
        times['MLP'].append(mlp_time)
        print(f"{mlp_time:.2f}s")
    
    return times


def save_times(times):
    """Save timing data to JSON."""
    # Save raw times
    output_path = f'{OUTPUT_DIR}/ML_model_training_times.json'
    with open(output_path, 'w') as f:
        json.dump(times, f, indent=2)
    print(f"\nSaved timing data: {output_path}")
    
    # Print summary
    models = ['MLP', 'Lasso', 'RandomForest', 'XGBoost']
    print("\n" + "="*50)
    print("Training Time Summary (seconds per LOO fold):")
    print("="*50)
    for m in models:
        print(f"  {m}: {np.mean(times[m]):.2f} ± {np.std(times[m]):.2f}")


def create_barplot(times):
    """Create barplot comparing training times."""
    # Calculate mean and std
    models = ['MLP', 'Lasso', 'RandomForest', 'XGBoost']
    means = [np.mean(times[m]) for m in models]
    stds = [np.std(times[m]) for m in models]
    
    # Colors matching ML model comparison plot
    colors = ['#96CEB4', '#4ECDC4', '#45B7D1', '#FF6B6B']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, 
                  edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + max(means)*0.02,
                f'{mean:.1f}s', ha='center', va='bottom', 
                fontsize=14, fontweight='bold')
    
    # Set labels
    ax.set_xlabel('')
    ax.set_ylabel(f'Training Time per LOO Fold (seconds)\n(10% data, ~7,800 tiles)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=16, fontweight='bold')
    
    # Style
    ax.tick_params(axis='y', labelsize=14)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # Grid
    ax.grid(True, alpha=0.7, linestyle='--', linewidth=2, axis='y')
    
    # Borders
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_linewidth(3)
        ax.spines[spine].set_color('black')
    
    # Set y limit
    ax.set_ylim(0, max(means) * 1.3 + max(stds))
    
    plt.tight_layout()
    
    # Save
    output_path = f'{OUTPUT_DIR}/ML_model_training_time_comparison.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"\nSaved plot: {output_path}")
    
    plt.close()


times = benchmark_models()
save_times(times)
create_barplot(times)









##### Figure 3 (Panel D) Regress out among 5 models (Conch, ProvGigapath, UNI2h, Virchow, Virchow2)


def check_feature_statistics():
    """
    Check the numerical range and distribution of features for four foundation models
    """
    # Data paths and model names
    input_dir = '/Training_features'
    model_names = ["ResNet50", "Conch", "Prov-GigaPath", "UNI2-h", "Virchow", "Virchow2"]
    # Map display names to file names
    model_file_names = {
        'ResNet50': 'ResNet50',
        'Conch': 'Conch', 
        'Prov-GigaPath': 'ProvGigapath',
        'UNI2-h': 'UNI2h',
        'Virchow': 'Virchow',
        'Virchow2': 'Virchow2'
    }
    
    feature_stats = {}
    features_data = {}
    
    print("Loading features from all four foundation models...")
    print("="*80)
    unique_indices_resnet = None  # For recording unique indices of ResNet50
    for model_name in model_names:
        print(f"\nLoading {model_name} features...")
        
        # Get the actual file name for this model
        file_model_name = model_file_names[model_name]
        
        # Construct file path
        feature_file_Cancer = f'Cancer Cells_training_precomputed_features_{file_model_name}.pt'
        feature_file_Stromal = f'Stromal Cells_training_precomputed_features_{file_model_name}.pt'
        feature_file_Normal = f'Normal Epithelial Cells_training_precomputed_features_{file_model_name}.pt'
        feature_file_T = f'T Cells_training_precomputed_features_{file_model_name}.pt'
        feature_file_Other = f'Other Immune Cells_training_precomputed_features_{file_model_name}.pt'
        file_path_Cancer = f"{input_dir}/{feature_file_Cancer}"
        file_path_Stromal = f"{input_dir}/{feature_file_Stromal}"
        file_path_Normal = f"{input_dir}/{feature_file_Normal}"
        file_path_T = f"{input_dir}/{feature_file_T}"
        file_path_Other = f"{input_dir}/{feature_file_Other}"


        # Load data
        data_Cancer = torch.load(file_path_Cancer)
        data_Stromal = torch.load(file_path_Stromal)
        data_Normal = torch.load(file_path_Normal)
        data_T = torch.load(file_path_T)
        data_Other = torch.load(file_path_Other)
        features_Cancer = data_Cancer['embeddings'].numpy()
        features_Stromal = data_Stromal['embeddings'].numpy()
        features_Normal = data_Normal['embeddings'].numpy()
        features_T = data_T['embeddings'].numpy()
        features_Other = data_Other['embeddings'].numpy()

        all_features = np.vstack([
        features_Cancer,
        features_Stromal,
        features_Normal,
        features_T,
        features_Other])

        if model_name == "ResNet50":
            unique_features, unique_indices_resnet = np.unique(all_features, axis=0, return_index=True)
        else:
            unique_features = all_features[unique_indices_resnet]



            
        features_data[model_name] = unique_features
        
        # Calculate statistics
        stats = {
            'shape': unique_features.shape,
            'mean': unique_features.mean(),
            'std': unique_features.std(),
            'min': unique_features.min(),
            'max': unique_features.max(),
            'median': np.median(unique_features),
            'q25': np.percentile(unique_features, 25),
            'q75': np.percentile(unique_features, 75),
            'feature_dim': unique_features.shape[1],
            'n_samples': unique_features.shape[0]
        }
        
        feature_stats[model_name] = stats
        
        print(f"  Shape: {stats['shape']}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Q25-Q75: [{stats['q25']:.4f}, {stats['q75']:.4f}]")
        

    
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    
    comparison_data = []
    for model_name in model_names:
        if model_name in feature_stats:
            stats = feature_stats[model_name]
            comparison_data.append({
                'Model': model_name,
                'Feature_Dim': stats['feature_dim'],
                'N_Samples': stats['n_samples'],
                'Min': stats['min'],
                'Max': stats['max'],
                'Mean': stats['mean'],
                'Std': stats['std'],
                'Range': stats['max'] - stats['min']
            })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.round(4))
        
        # Check scale differences
        print(f"\n SCALE ANALYSIS:")
        means = df_comparison['Mean'].values
        stds = df_comparison['Std'].values
        ranges = df_comparison['Range'].values
        
        print(f"Mean range: {means.min():.4f} to {means.max():.4f} (ratio: {means.max()/abs(means.min()) if means.min() != 0 else 'inf'})")
        print(f"Std range: {stds.min():.4f} to {stds.max():.4f} (ratio: {stds.max()/stds.min():.2f})")
        print(f"Value range: {ranges.min():.4f} to {ranges.max():.4f} (ratio: {ranges.max()/ranges.min():.2f})")
        
        # Standardization recommendation
        max_ratio = max(stds.max()/stds.min(), ranges.max()/ranges.min())
        print(f"\n💡 STANDARDIZATION RECOMMENDATION:")
        if max_ratio > 10:
            print("❌ STRONG recommendation to standardize - scale differences > 10x")
        elif max_ratio > 3:
            print("⚠️  MILD recommendation to standardize - scale differences > 3x")
        else:
            print("✅ Standardization may not be necessary - similar scales")
        
        print(f"   Max scale ratio: {max_ratio:.2f}")
        
        # Visualize distributions
        create_distribution_plots(features_data)
        
        return features_data, feature_stats, df_comparison
    
    else:
        print("No valid feature data loaded!")
        return None, None, None

def create_distribution_plots(features_data):
    """
    Create visualization of feature distributions
    """
    if not features_data:
        return
        
    print(f"\n📈 Creating distribution plots...")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (model_name, features) in enumerate(features_data.items()):
        if idx >= 4:
            break
            
        # Randomly sample some feature dimensions to visualize
        sample_features = features[:, :min(5, features.shape[1])]  # Take first 5 dimensions
        
        ax = axes[idx]
        
        # Plot distribution of each dimension
        for i in range(sample_features.shape[1]):
            ax.hist(sample_features[:, i], bins=50, alpha=0.6, 
                   label=f'Dim {i}', density=True)
        
        ax.set_title(f'{model_name} Feature Distributions', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Add thick black borders like hexagon plots
        ax.spines['top'].set_linewidth(5)
        ax.spines['right'].set_linewidth(5)
        ax.spines['bottom'].set_linewidth(5)
        ax.spines['left'].set_linewidth(5)
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
    
    plt.tight_layout()
    plt.savefig('foundation_model_feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create boxplot comparison
    plt.figure(figsize=(12, 6))
    
    all_means = []
    all_stds = []
    model_labels = []
    
    for model_name, features in features_data.items():
        # Calculate mean and standard deviation for each sample
        sample_means = features.mean(axis=1)
        sample_stds = features.std(axis=1)
        
        all_means.extend(sample_means)
        all_stds.extend(sample_stds)
        model_labels.extend([model_name] * len(sample_means))
    
    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'Model': model_labels,
        'Sample_Mean': all_means,
        'Sample_Std': all_stds
    })
    
    # Plot boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=plot_data, x='Model', y='Sample_Mean')
    plt.title('Sample-wise Feature Means')
    plt.xticks(rotation=45)
    # Add thick black borders
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(5)
    ax1.spines['right'].set_linewidth(5)
    ax1.spines['bottom'].set_linewidth(5)
    ax1.spines['left'].set_linewidth(5)
    ax1.spines['top'].set_color('black')
    ax1.spines['right'].set_color('black')
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_color('black')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=plot_data, x='Model', y='Sample_Std')
    plt.title('Sample-wise Feature Stds')
    plt.xticks(rotation=45)
    # Add thick black borders
    ax2 = plt.gca()
    ax2.spines['top'].set_linewidth(5)
    ax2.spines['right'].set_linewidth(5)
    ax2.spines['bottom'].set_linewidth(5)
    ax2.spines['left'].set_linewidth(5)
    ax2.spines['top'].set_color('black')
    ax2.spines['right'].set_color('black')
    ax2.spines['bottom'].set_color('black')
    ax2.spines['left'].set_color('black')
    
    plt.tight_layout()

    plt.show()


features_data, feature_stats, comparison_df = check_feature_statistics()






def standardize_features():
    models = ['ResNet50', 'Conch', 'Prov-GigaPath', 'UNI2-h', 'Virchow', 'Virchow2']
    # Map display names to file names
    model_file_names = {
        'ResNet50': 'ResNet50',
        'Conch': 'Conch', 
        'Prov-GigaPath': 'ProvGigapath',
        'UNI2-h': 'UNI2h',
        'Virchow': 'Virchow',
        'Virchow2': 'Virchow2'
    }
    input_dir = '/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/Training_features'


    standardized_features = {}
    unique_indices_resnet = None  # For recording unique indices of ResNet50
    for model in models:
        print(f"Processing {model}...")
        
        # Get the actual file name for this model
        file_model_name = model_file_names[model]
        
        # Load original data
        data_Cancer = torch.load(f'{input_dir}/Cancer Cells_training_precomputed_features_{file_model_name}.pt')
        data_Stromal = torch.load(f'{input_dir}/Stromal Cells_training_precomputed_features_{file_model_name}.pt')
        data_Normal = torch.load(f'{input_dir}/Normal Epithelial Cells_training_precomputed_features_{file_model_name}.pt')
        data_T = torch.load(f'{input_dir}/T Cells_training_precomputed_features_{file_model_name}.pt')
        data_Other = torch.load(f'{input_dir}/Other Immune Cells_training_precomputed_features_{file_model_name}.pt')
        features_Cancer = data_Cancer['embeddings'].numpy()
        features_Stromal = data_Stromal['embeddings'].numpy()
        features_Normal = data_Normal['embeddings'].numpy()
        features_T = data_T['embeddings'].numpy()
        features_Other = data_Other['embeddings'].numpy()

        all_features = np.vstack([
        features_Cancer,
        features_Stromal,
        features_Normal,
        features_T,
        features_Other])

        if model == "ResNet50":
            unique_features, unique_indices_resnet = np.unique(all_features, axis=0, return_index=True)
        else:
            unique_features = all_features[unique_indices_resnet]

        # Print original statistics
        print(f"  Original: mean={unique_features.mean():.4f}, std={unique_features.std():.4f}")
        
        # Standardize
        scaler = StandardScaler()
        features_std = scaler.fit_transform(unique_features)
        
        # Print standardized statistics
        print(f"  Standardized: mean={features_std.mean():.4f}, std={features_std.std():.4f}")
        

        standardized_features[model] = features_std
        print()

    print("All 5 models standardized!")
    print(f"Feature shapes: {[(model, feat.shape) for model, feat in standardized_features.items()]}")
    return standardized_features


standardized_features = standardize_features()



def regress_out_features():

    residuals = {}
    regression_results = []

    models = ['ResNet50', 'Conch', 'Prov-GigaPath', 'UNI2-h', 'Virchow', 'Virchow2']
    for y_model in models:
        Y = standardized_features[y_model]
        

        for x_model in models:
            if x_model != y_model:
                X = standardized_features[x_model]
                
                print(f"\nRegressing {y_model} ~ {x_model}")
                print(f"Y shape: {Y.shape}, X shape: {X.shape}")
                
                # Use Ridge regression
                ridge = Ridge(alpha=1.0, random_state=42)
                ridge.fit(X, Y)
                
                # Calculate residuals E = Y - X*B
                Y_pred = ridge.predict(X)
                residual = Y - Y_pred
                
                # Calculate R²
                r2 = r2_score(Y, Y_pred)
                
                # Store residuals
                residual_key = f"{y_model}_minus_{x_model}"
                residuals[residual_key] = residual
                
                # Store results
                regression_results.append({
                    'Y_model': y_model,
                    'X_model': x_model,
                    'R2_score': r2,
                    'Residual_mean': residual.mean(),
                    'Residual_std': residual.std(),
                    'Residual_key': residual_key
                })
                
                print(f"  R² = {r2:.4f}")
                print(f"  Residual stats: mean={residual.mean():.4f}, std={residual.std():.4f}")

    print(f"\n Generated {len(residuals)} residuals (E)!")
    print(f"Residual keys: {list(residuals.keys())}")
    return residuals, regression_results




residuals, regression_results = regress_out_features()







# Create heatmap of R² values

# Create 4x4 matrix to store R² values
models = ['ResNet50', 'Conch', 'Prov-GigaPath', 'UNI2-h', 'Virchow', 'Virchow2']
r2_matrix = np.zeros((6, 6))

# Fill the matrix
for result in regression_results:
    y_idx = models.index(result['Y_model'])
    x_idx = models.index(result['X_model'])
    r2_matrix[y_idx, x_idx] = result['R2_score']

# fill the diagonal with 1
np.fill_diagonal(r2_matrix, 1.0)

# Create DataFrame for heatmap
r2_df = pd.DataFrame(r2_matrix, 
                     index=models, 
                     columns=models)

print("R² Matrix:")
print(r2_df.round(3))

# Create heatmap
plt.figure(figsize=(10, 8))
mask = np.zeros_like(r2_matrix, dtype=bool)
np.fill_diagonal(mask, True)  # Mask diagonal

# Create heatmap
sns.heatmap(r2_df, 
            annot=True, 
            fmt='.3f',
            cmap='RdYlBu_r',  # Red (high R² = low complementarity), Blue (low R² = high complementarity)
            center=0.5,
            vmin=0, 
            vmax=1,
            square=True,
            mask=mask,  # Mask diagonal
            cbar_kws={'label': 'R² Score'},
            annot_kws={'size': 18, 'weight': 'bold', 'color': 'black'})

plt.title(f'Foundation Model Complementarity Matrix\n(Lower R² = Higher Complementarity)', 
          fontsize=16, fontweight='bold', pad=20)
# Remove axis labels
# plt.xlabel('Predictor Model (X)', fontsize=14, fontweight='bold')
# plt.ylabel('Target Model (Y)', fontsize=14, fontweight='bold')

# Beautify x and y axis ticks and colorbar
plt.xticks(fontsize=14, fontweight='bold', color='black', rotation=45, ha='right')
plt.yticks(fontsize=14, fontweight='bold', color='black', rotation=45, ha='right')
cbar = plt.gcf().axes[-1]
cbar.tick_params(labelsize=18, labelcolor='black')
for label in cbar.get_yticklabels():
    label.set_fontweight('bold')
cbar.set_ylabel('R² Score', fontsize=16, fontweight='bold', color='black')

plt.tight_layout()
plt.savefig('/Visual/foundation_model_complementarity_matrix.png', dpi=600, bbox_inches='tight')
plt.close()


