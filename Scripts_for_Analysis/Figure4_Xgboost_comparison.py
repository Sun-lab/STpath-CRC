"""
STPath-COAD: XGBoost Model Comparison Analysis for Figure 4
==========================================================

This script performs comprehensive comparison analysis of different XGBoost models
and foundation model combinations for cell type proportion prediction.

Author: Saishi Cui
Date: Sept 2025

Purpose: Compare performance of different foundation models (ResNet50, Conch, ProvGigapath, 
UNI2h, Virchow, Virchow2) and their combination for cell type proportion prediction.
Generate visualizations and statistical comparisons for Figure 4 of the paper.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


### Panel A, Model Comparison



# Collect all the results
all_results = []

for cell_type in ["Cancer Cells", "Stromal Cells", "Normal Epithelial Cells", "T Cells", "Other Immune Cells"]:
    for model in ["ResNet50", "Conch", "ProvGigapath", "UNI2h", "Virchow", "Virchow2", "Combined"]:

        df = pd.read_csv(f"xgboost_prediction/{cell_type}_{model}_individual_level_ratio100/individual_metrics_individual_level.csv")
        
        for index, row in df.iterrows():
            ct_range = row["Max_celltype_proportion"] - row["Min_celltype_proportion"]
            if row["n_test_samples"] >= 200 and ct_range > 0.3:
                all_results.append({
                    "model_name": model, 
                    "cell_type": cell_type, 
                    "MAE": row["MAE"], 
                    "Pearson_correlation": row["Pearson"], 
                    "n_test_samples": row["n_test_samples"], 
                    "Max_celltype_proportion": row["Max_celltype_proportion"], 
                    "Min_celltype_proportion": row["Min_celltype_proportion"], 
                    "ct_range": ct_range
                })


# Create the final DataFrame
final_df = pd.DataFrame(all_results)
final_df.to_csv(f"xgboost_prediction/individual_metrics_individual_level_filtered.csv", index=False)




# Set the figure style
plt.style.use('default')
sns.set_palette("husl")

# Define the color mapping
color_palette = {
    'ResNet50': '#FEF0DE',
    'Conch': '#C43E96', 
    'ProvGigapath': '#DEDBEE',
    'UNI2h': '#06948E',
    'Virchow': '#F3CDCC',
    'Virchow2': '#F0CF7F',
    'Combined': '#FF6B6B'  
}

# Define the metrics to plot
metrics = ['MAE', 'Pearson_correlation']
metric_titles = ['Mean Absolute Error (MAE)', 'Pearson Correlation']

# Create a separate figure for each metric
for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
    # Create a separate figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Debug: Check data for current metric
    if metric == 'MAE':
        print(f"\nDEBUG - {metric} data check:")
        for cell_type in ["Cancer Cells", "Stromal Cells", "Normal Epithelial Cells", "T Cells", "Other Immune Cells"]:
            cell_data = final_df[final_df['cell_type'] == cell_type]
            print(f"{cell_type}: {len(cell_data)} samples")
            if len(cell_data) > 0:
                print(f"  MAE range: {cell_data['MAE'].min():.3f} - {cell_data['MAE'].max():.3f}")
                print(f"  MAE std: {cell_data['MAE'].std():.3f}")
    
    # Create boxplot
    box_plot = sns.boxplot(
        data=final_df, 
        x='cell_type', 
        y=metric, 
        hue='model_name',
        palette=color_palette,
        ax=ax,
        showfliers=False,  # Do not show outliers, because we will use scatter plot to show all points
        linewidth=1.5,
        legend=False  # Do not show legend
    )
    
    # Add the jitter scatter plot
    # Add the scatter plot for each cell type and model combination
    cell_types = final_df['cell_type'].unique()
    models = final_df['model_name'].unique()
    
    # Use stripplot to better show the scatter points in the box
    sns.stripplot(
        data=final_df, 
        x='cell_type', 
        y=metric, 
        hue='model_name',
        palette=color_palette,
        ax=ax,
        size=6,
        alpha=0.8,
        dodge=True,  # Let the points with different hue to be displayed separately
        jitter=0.3,  # Add jitter
        edgecolor='black',
        linewidth=0.5,
        legend=False  # Do not show the legend of the stripplot
    )
    
    # Set the labels - do not show the horizontal axis title, do not show the total title
    ax.set_xlabel('')  # Do not show the horizontal axis title
    ax.set_ylabel(title, fontsize=22, fontweight='bold')  # Set the vertical axis title to be bold
    
    # Set the tick label style - the horizontal and vertical axis ticks to be bold
    ax.tick_params(axis='x', rotation=0, labelsize=20, labelcolor='black', 
                   width=2, length=6, colors='black')  # rotation=0 makes x-axis labels horizontal
    ax.tick_params(axis='y', labelsize=20, labelcolor='black', 
                   width=2, length=6, colors='black')
    
    # Set the tick label font to be bold, and delete the "Cells" word, and change "Cancer" to "Tumor"
    x_labels = []
    for label in ax.get_xticklabels():
        text = label.get_text().replace(' Cells', '')  # Delete the "Cells" word
        text = text.replace('Cancer', 'Tumor')  # Change "Cancer" to "Tumor"
        x_labels.append(text)
        label.set_fontweight('bold')
    
    # Apply the modified x-axis labels
    ax.set_xticklabels(x_labels, fontweight='bold')
    
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # Set the vertical axis range
    if metric == 'MAE':
        ax.set_ylim(0, 0.4)
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
    
    # Legend removed per user request
    
    # Adjust the layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'Colorectal_Cancer_HE_patches/Visual/model_comparison_{metric}_boxplot.png', 
                dpi=600, bbox_inches='tight', facecolor='white')
    
    # Show the figure
    plt.show()


### Normalization Function for Panel A

def normalize_cell_type_proportions(
    target_cell_type="Cancer Cells",
    input_dir='Colorectal_Cancer_HE_patches/Training_features'
):
    """
    Normalize cell type proportions for target cell type training data.
    
    Args:
        target_cell_type: The cell type whose training data we want to normalize
        input_dir: Directory containing training features
    
    Returns:
        list: Each item contains [tile_id, individual_id, cancer_prop, stromal_prop, normal_prop, t_prop, other_prop]
    """
    import torch
    import xgboost as xgb
    import pickle
    import numpy as np
    import os
    
    # Define all cell types
    all_cell_types = ["Cancer Cells", "Stromal Cells", "Normal Epithelial Cells", "T Cells", "Other Immune Cells"]
    
    # Load target cell type training data (use Virchow2 to get individual info)
    target_feature_name = f'{target_cell_type}_training_precomputed_features_Virchow2.pt'
    target_data_path = os.path.join(input_dir, target_feature_name)
    target_data = torch.load(target_data_path)
    
    # Get individual info
    target_individual_ids = target_data.get('individual_ids', [])
    target_tile_ids = target_data.get('tile_ids', [])
    unique_target_individuals = sorted(list(set(target_individual_ids)))
    
    # Load important features for combined model
    important_features = pickle.load(open(f"xgboost_prediction/important_features_{target_cell_type}.pkl", "rb"))
    
    # Load all foundation model features for target data
    UNI2h_data = torch.load(os.path.join(input_dir, f'{target_cell_type}_training_precomputed_features_UNI2h.pt'))
    Virchow_data = torch.load(os.path.join(input_dir, f'{target_cell_type}_training_precomputed_features_Virchow.pt'))
    Virchow2_data = torch.load(os.path.join(input_dir, f'{target_cell_type}_training_precomputed_features_Virchow2.pt'))
    ProvGigapath_data = torch.load(os.path.join(input_dir, f'{target_cell_type}_training_precomputed_features_ProvGigapath.pt'))
    Conch_data = torch.load(os.path.join(input_dir, f'{target_cell_type}_training_precomputed_features_Conch.pt'))
    
    # Extract important features and combine
    UNI2h_important_indices = [int(f.replace('f', '')) for f in important_features["UNI2h"]]
    Virchow_important_indices = [int(f.replace('f', '')) for f in important_features["Virchow"]]
    Virchow2_important_indices = [int(f.replace('f', '')) for f in important_features["Virchow2"]]
    ProvGigapath_important_indices = [int(f.replace('f', '')) for f in important_features["ProvGigapath"]]
    Conch_important_indices = [int(f.replace('f', '')) for f in important_features["Conch"]]
    
    UNI2h_selected = UNI2h_data['embeddings'][:, UNI2h_important_indices]
    Virchow_selected = Virchow_data['embeddings'][:, Virchow_important_indices]
    Virchow2_selected = Virchow2_data['embeddings'][:, Virchow2_important_indices]
    ProvGigapath_selected = ProvGigapath_data['embeddings'][:, ProvGigapath_important_indices]
    Conch_selected = Conch_data['embeddings'][:, Conch_important_indices]
    
    target_features = torch.cat([UNI2h_selected, Virchow_selected, Virchow2_selected, ProvGigapath_selected, Conch_selected], dim=1)
    
    # Initialize predictions storage
    all_predictions = {}
    for cell_type in all_cell_types:
        all_predictions[cell_type] = np.zeros(len(target_individual_ids))
    
    # For each individual, predict all cell types
    for target_individual in unique_target_individuals:
        # Get indices for this individual
        individual_mask = np.array([iid == target_individual for iid in target_individual_ids])
        individual_indices = np.where(individual_mask)[0]
        individual_features = target_features[individual_indices]
        
        # Predict each cell type
        for cell_type in all_cell_types:
            if cell_type == target_cell_type:
                # For target cell type, use leave-this-individual-out model
                model_path = f"xgboost_prediction/{cell_type}_Combined_individual_level_ratio100/models/xgboost_model_leave_{target_individual}_out.model"
            else:
                # For other cell types, check if individual exists in their training data
                other_feature_name = f'{cell_type}_training_precomputed_features_Virchow2.pt'
                other_data_path = os.path.join(input_dir, other_feature_name)
                other_data = torch.load(other_data_path)
                other_individual_ids = other_data.get('individual_ids', [])
                unique_other_individuals = set(other_individual_ids)
                
                if target_individual in unique_other_individuals:
                    # Individual exists, use leave-this-individual-out model
                    model_path = f"xgboost_prediction/{cell_type}_Combined_individual_level_ratio100/models/xgboost_model_leave_{target_individual}_out.model"
                else:
                    # Individual doesn't exist, use external prediction model
                    model_path = f"xgboost_prediction/{cell_type}_Combined_external_prediction/xgboost_model_{cell_type}_Combined_external_prediction.model"
            
            # Load and use model for prediction
            model = xgb.Booster()
            model.load_model(model_path)
            dtest = xgb.DMatrix(individual_features.numpy())
            predictions = model.predict(dtest)
            all_predictions[cell_type][individual_indices] = predictions
    
    # Create result list
    result_list = []
    for i in range(len(target_individual_ids)):
        # Get predictions for this patch across all cell types
        patch_predictions = np.array([all_predictions[cell_type][i] for cell_type in all_cell_types])
        
        # Ensure non-negative and normalize to sum to 1
        patch_predictions = np.maximum(patch_predictions, 0)
        patch_sum = np.sum(patch_predictions)
        normalized_patch = patch_predictions / patch_sum

        # Create result: [tile_id, individual_id, cancer_prop, stromal_prop, normal_prop, t_prop, other_prop]
        result_item = [
            target_tile_ids[i],
            target_individual_ids[i],
            normalized_patch[0],  # Cancer Cells
            normalized_patch[1],  # Stromal Cells
            normalized_patch[2],  # Normal Epithelial Cells
            normalized_patch[3],  # T Cells
            normalized_patch[4]   # Other Immune Cells
        ]
        result_list.append(result_item)
    
    return result_list


result_cancer_cells_normalized = normalize_cell_type_proportions(target_cell_type="Cancer Cells")
result_stromal_cells_normalized = normalize_cell_type_proportions(target_cell_type="Stromal Cells")
result_normal_epithelial_cells_normalized = normalize_cell_type_proportions(target_cell_type="Normal Epithelial Cells")
result_t_cells_normalized = normalize_cell_type_proportions(target_cell_type="T Cells")
result_other_immune_cells_normalized = normalize_cell_type_proportions(target_cell_type="Other Immune Cells")


def evaluation_metrics_calculation(
    result_normalized,
    cell_type,
    input_dir='Colorectal_Cancer_HE_patches/Training_features'
):
    """
    Calculate evaluation metrics for normalized predictions by individual.
    
    Args:
        result_normalized: List from normalize_cell_type_proportions function
        cell_type: Target cell type
        input_dir: Directory containing training features
    
    Returns:
        pd.DataFrame: Individual-level metrics
    """
    import torch
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr, pearsonr
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import os
    
    # Load original training data to get labels
    target_feature_name = f'{cell_type}_training_precomputed_features_Virchow2.pt'
    target_data_path = os.path.join(input_dir, target_feature_name)
    target_data = torch.load(target_data_path)
    
    # Get labels and metadata
    true_labels = target_data['celltype_proportions'].numpy()
    tile_ids = target_data.get('tile_ids', [])
    individual_ids = target_data.get('individual_ids', [])
    
    # Create mapping from tile_id to true label
    tile_to_label = {}
    for i, tile_id in enumerate(tile_ids):
        tile_to_label[tile_id] = true_labels[i]
    
    # Extract predictions from normalized results
    # result_normalized format: [tile_id, individual_id, cancer_prop, stromal_prop, normal_prop, t_prop, other_prop]
    cell_type_index = {
        "Cancer Cells": 2,
        "Stromal Cells": 3, 
        "Normal Epithelial Cells": 4,
        "T Cells": 5,
        "Other Immune Cells": 6
    }
    
    pred_index = cell_type_index[cell_type]
    
    # Collect data for each individual
    individual_data = {}
    
    for item in result_normalized:
        tile_id = item[0]
        individual_id = item[1]
        predicted_prop = item[pred_index]
        
        # Get true label for this tile
        if tile_id in tile_to_label:
            true_prop = tile_to_label[tile_id]
            
            if individual_id not in individual_data:
                individual_data[individual_id] = {
                    'true_props': [],
                    'pred_props': []
                }
            
            individual_data[individual_id]['true_props'].append(true_prop)
            individual_data[individual_id]['pred_props'].append(predicted_prop)
    
    # Calculate metrics for each individual
    results = []
    
    for individual_id, data in individual_data.items():
        true_props = np.array(data['true_props'])
        pred_props = np.array(data['pred_props'])
        
        # Calculate metrics
        mae = mean_absolute_error(true_props, pred_props)
        rmse = np.sqrt(mean_squared_error(true_props, pred_props))
        spearman_corr, _ = spearmanr(true_props, pred_props)
        pearson_corr, _ = pearsonr(true_props, pred_props)
        
        n_samples = len(true_props)
        min_prop = np.min(true_props)
        max_prop = np.max(true_props)
        
        results.append({
            'Individual': individual_id,
            'MAE': mae,
            'RMSE': rmse,
            'Spearman': spearman_corr,
            'Pearson': pearson_corr,
            'n_test_samples': n_samples,
            'Min_celltype_proportion': min_prop,
            'Max_celltype_proportion': max_prop
        })
    
    # Create DataFrame and sort by Individual
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Individual').reset_index(drop=True)
    
    return results_df


# Calculate metrics for all cell types using normalized results
metrics_cancer = evaluation_metrics_calculation(result_cancer_cells_normalized, "Cancer Cells")
metrics_stromal = evaluation_metrics_calculation(result_stromal_cells_normalized, "Stromal Cells")
metrics_normal = evaluation_metrics_calculation(result_normal_epithelial_cells_normalized, "Normal Epithelial Cells")
metrics_t_cells = evaluation_metrics_calculation(result_t_cells_normalized, "T Cells")
metrics_other_immune = evaluation_metrics_calculation(result_other_immune_cells_normalized, "Other Immune Cells")


metrics_cancer.to_csv("xgboost_prediction/Normalized_metrics_cancer.csv", index=False)
metrics_stromal.to_csv("xgboost_prediction/Normalized_metrics_stromal.csv", index=False)
metrics_normal.to_csv("xgboost_prediction/Normalized_metrics_normal.csv", index=False)
metrics_t_cells.to_csv("xgboost_prediction/Normalized_metrics_t_cells.csv", index=False)
metrics_other_immune.to_csv("xgboost_prediction/Normalized_metrics_other_immune.csv", index=False)




print("Cancer Cells Metrics:")
print(metrics_cancer.head())
print("\nStromal Cells Metrics:")
print(metrics_stromal.head())




####  Panel B, performance by different training data ratio





all_results_panel_b = []


for cell_type in ["Cancer Cells", "Stromal Cells", "Normal Epithelial Cells", "T Cells", "Other Immune Cells"]:
    for training_data_ratio in [0.2, 0.4, 0.6, 0.8, 1.0]:
        model = "Combined"
        training_data_ratio = int(100*training_data_ratio)
        df = pd.read_csv(f"Colorectal_Cancer_HE_patches/xgboost_prediction/{cell_type}_{model}_individual_level_ratio{training_data_ratio}/individual_metrics_individual_level.csv")
        
        for index, row in df.iterrows():
            ct_range = row["Max_celltype_proportion"] - row["Min_celltype_proportion"]
            if row["n_test_samples"] >= 200 and ct_range > 0.3:
                all_results_panel_b.append({
                    "model_name": model, 
                    "cell_type": cell_type, 
                    "MAE": row["MAE"], 
                    "Pearson_correlation": row["Pearson"], 
                    "training_data_ratio": training_data_ratio
                })


# Create the final DataFrame
final_df_panel_b = pd.DataFrame(all_results_panel_b)

# Calculate the average
panel_b_stats = final_df_panel_b.groupby(["model_name", "cell_type", "training_data_ratio"]).mean().reset_index()

# Define the cell type color mapping
cell_type_colors = {
    'Cancer Cells': '#E41A1C',  # Tumor
    'Stromal Cells': '#FFFF33',  # Stromal
    'Normal Epithelial Cells': '#377EB8',  # Normal Epithelial
    'T Cells': '#4DAF4A',  # T
    'Other Immune Cells': '#FF7F00'  # Other Immune
}

# Create the line chart
metrics_panel_b = ['MAE', 'Pearson_correlation']
metric_titles_panel_b = ['Averaged MAE', 'Averaged Pearson Correlation']

for i, (metric, title) in enumerate(zip(metrics_panel_b, metric_titles_panel_b)):
    # Create a separate figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot the line for each cell type
    for cell_type in ["Cancer Cells", "Stromal Cells", "Normal Epithelial Cells", "T Cells", "Other Immune Cells"]:
        # Get the data for the cell type
        cell_data = panel_b_stats[panel_b_stats['cell_type'] == cell_type]
        
        # Prepare the display label (delete "Cells", change "Cancer" to "Tumor")
        display_label = cell_type.replace(' Cells', '').replace('Cancer', 'Tumor')
        
        # Plot the line
        ax.plot(cell_data['training_data_ratio'], cell_data[metric], 
                color=cell_type_colors[cell_type], 
                marker='o', 
                linewidth=4, 
                markersize=10,
                label=display_label,
                markeredgecolor='black',
                markeredgewidth=1)
    
    # Set the labels and title
    ax.set_xlabel('Training Data Ratio (%)', fontsize=16, fontweight='bold')
    ax.set_ylabel(title, fontsize=20, fontweight='bold')
    
    # Set the tick label style
    ax.tick_params(axis='x', labelsize=18, labelcolor='black', 
                   width=2, length=6, colors='black')
    ax.tick_params(axis='y', labelsize=18, labelcolor='black', 
                   width=2, length=6, colors='black')
    
    # Set the tick label font to be bold
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # Set the x-axis ticks
    ax.set_xticks([20, 40, 60, 80, 100])
    
    # Set the vertical axis range
    if metric == 'MAE':
        ax.set_ylim(0, 0.2)
    elif metric == 'Pearson_correlation':
        ax.set_ylim(0, 0.8)
    
    # Add the grid with darker and thicker lines
    ax.grid(True, alpha=0.7, linestyle='--', linewidth=2)
    
    # Add thick black borders like hexagon plots
    ax.spines['top'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    
    # Legend removed per user request
    
    # Adjust the layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'Visual/training_ratio_{metric}_linechart.png', 
                dpi=600, bbox_inches='tight', facecolor='white')
    
    # Show the figure
    plt.show()





###### Panel C, Visualize 5 samples for each cell type


### Marker genes refinement
with open('scRNAseq_data/final_marker_genes_dict.json', 'r') as f:
    final_marker_genes_dict = json.load(f)

del final_marker_genes_dict["Cancer"]
del final_marker_genes_dict["Normal Epithelia"]



InputDf_for_CARD_SelectedGenes = pd.read_csv('scRNAseq_data/InputDf_for_CARD_SelectedGenes.csv', index_col=0)
InputDf_for_CARD_meta = pd.read_csv('/Users/scui2/ST/scRNAseq_data/InputDf_for_CARD_meta.csv', index_col=0)
InputDf_for_CARD_meta.loc[InputDf_for_CARD_meta["Cell Type"] == "CD4+ T", "Cell Type"] = "T"
InputDf_for_CARD_meta.loc[InputDf_for_CARD_meta["Cell Type"] == "CD8+ T", "Cell Type"] = "T"

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







### For Cancer Cells
cell_type = "Cancer Cells"
expression_df = pd.read_csv(f'CARD_Need_Files/6723_KL_1_region0_expression.csv', index_col=0)
Celltype_proportion_df = pd.read_csv(f'CARD_Results_Regions/6723_KL_1_region0_celltype_proportion_modified.csv', index_col=0)
B_matrix_df = pd.read_csv(f'CARD_Results_Regions/6723_KL_1_region0_B_Matrix_modified.csv', index_col=0)
        
cell_type_order = ["ASC I", "ASC II", "ASC III", "CSC I", "CSC II", "CSC III", "CSC IV", "SSC I", 
"ABS", "CT", "EE", "TUF", "T", "PLA", "MAS", "MYE", "B", "FIB", "END"]

Celltype_proportion_df = Celltype_proportion_df[cell_type_order]

B_matrix_df.columns = ['ASC I', 'ASC II', 'ASC III', 'CSC III', 'CSC I', 'CSC IV', "CSC II", 'SSC I', 'ABS', 'CT', 'EE', 'TUF', 'T', 'PLA', 'MAS', 'MYE', 'FIB', 'B', 'END']
B_matrix_df = B_matrix_df[cell_type_order]

kept_barcode = Celltype_proportion_df.index.tolist()
expression_df = expression_df.loc[kept_barcode]

library_size = expression_df.sum(axis=1)
normalized_df = expression_df.copy()
normalized_df = expression_df.div(library_size, axis=0)

# Delete the marker genes that are not in B_matrix_df
marker_genes_clean_dict = {}
for key, value in final_ranked_gene_dict.items():
    marker_genes_clean_dict[key] = list(set(value).intersection(set(B_matrix_df.index)))

# Select the top 5 marker genes for each cell type
marker_genes_clean_dict["Cancer Cells"] = list(set(marker_genes_clean_dict["ASC I"][:5] + marker_genes_clean_dict["ASC II"][:5] +\
                                    marker_genes_clean_dict["ASC III"][:5] + marker_genes_clean_dict["CSC I"][:5] +\
                                    marker_genes_clean_dict["CSC II"][:5] + marker_genes_clean_dict["CSC III"][:5] +\
                                    marker_genes_clean_dict["CSC IV"][:5] + marker_genes_clean_dict["SSC I"][:5]))

marker_genes_clean_dict["Normal Epithelial Cells"] = list(set(marker_genes_clean_dict["ABS"][:10] + marker_genes_clean_dict["CT"][:10] +\
                                            marker_genes_clean_dict["EE"][:10] + marker_genes_clean_dict["TUF"][:10]))

marker_genes_clean_dict["T Cells"] = marker_genes_clean_dict["T"][:40]
marker_genes_clean_dict["Other Immune Cells"] = list(set(marker_genes_clean_dict["PLA"][:10] + marker_genes_clean_dict["MAS"][:10] + marker_genes_clean_dict["MYE"][:10] + marker_genes_clean_dict["B"][:10]))
marker_genes_clean_dict["Stromal Cells"] = list(set(marker_genes_clean_dict["FIB"][:20] + marker_genes_clean_dict["END"][:20]))

for cell_type_inner in ["ASC I", "ASC II", "ASC III", "CSC I", "CSC II", "CSC III", "CSC IV", "SSC I", "ABS", "CT", "EE", "TUF", "T", "PLA", "MAS", "MYE", "B", "FIB", "END"]:
    del marker_genes_clean_dict[cell_type_inner]

# Calculate the relative expression of the marker genes
Cancer_MarkerGE = normalized_df[marker_genes_clean_dict["Cancer Cells"]].div(B_matrix_df.T.loc[["ASC I", "ASC II", "ASC III", "CSC I", "CSC II", "CSC III", "CSC IV", "SSC I"],marker_genes_clean_dict["Cancer Cells"]].mean(axis=0)).mean(axis=1)  
NE_MarkerGE = normalized_df[marker_genes_clean_dict["Normal Epithelial Cells"]].div(B_matrix_df.T.loc[["ABS", "CT", "EE", "TUF"],marker_genes_clean_dict["Normal Epithelial Cells"]].mean(axis=0)).mean(axis=1)
T_MarkerGE = normalized_df[marker_genes_clean_dict["T Cells"]].div(B_matrix_df.T.loc[["T"],marker_genes_clean_dict["T Cells"]].mean(axis=0)).mean(axis = 1)
Other_Immune_MarkerGE = normalized_df[marker_genes_clean_dict["Other Immune Cells"]].div(B_matrix_df.T.loc[["PLA", "MAS", "MYE", "B"],marker_genes_clean_dict["Other Immune Cells"]].mean(axis=0)).mean(axis = 1)
Stromal_MarkerGE = normalized_df[marker_genes_clean_dict["Stromal Cells"]].div(B_matrix_df.T.loc[["FIB", "END"],marker_genes_clean_dict["Stromal Cells"]].mean(axis=0)).mean(axis = 1)
    
marker_genes_expression_df_grouped = pd.concat([Cancer_MarkerGE, NE_MarkerGE, T_MarkerGE, Other_Immune_MarkerGE, Stromal_MarkerGE], axis=1)
marker_genes_expression_df_grouped.columns = ["Cancer Cells", "Normal Epithelial Cells", "T Cells", "Other Immune Cells", "Stromal Cells"]
relative_marker_genes_expression_df_grouped = marker_genes_expression_df_grouped.div(marker_genes_expression_df_grouped.sum(axis=1)+1e-8, axis=0)

Total_Cancer_Celltype_proportion = Celltype_proportion_df.iloc[:,0:8].sum(axis=1)
Total_Normal_Epith_Celltype_proportion = Celltype_proportion_df.iloc[:,8:12].sum(axis=1)
Total_T_Celltype_proportion = Celltype_proportion_df.iloc[:,12]
Total_Other_Immune_Celltype_proportion = Celltype_proportion_df.iloc[:,13:17].sum(axis=1)
Total_Stromal_Celltype_proportion = Celltype_proportion_df.iloc[:,17:].sum(axis=1)
Celltype_proportion_df_grouped = pd.concat([Total_Cancer_Celltype_proportion, Total_Normal_Epith_Celltype_proportion, Total_T_Celltype_proportion, Total_Other_Immune_Celltype_proportion, Total_Stromal_Celltype_proportion], axis=1)
Celltype_proportion_df_grouped.columns = ["Cancer Cells", "Normal Epithelial Cells", "T Cells", "Other Immune Cells", "Stromal Cells"]


focus_celltype_percent = (Celltype_proportion_df_grouped[cell_type]*100).round(4).values.tolist()
focus_relative_marker_genes_expression = (relative_marker_genes_expression_df_grouped[cell_type]*100).round(4).values.tolist()



Spatial_location_df = pd.read_csv('/Users/scui2/ST/CARD_Need_Files/6723_KL_1_region0_spatial.csv', index_col=0)
predicted_data = torch.load("Colorectal_Cancer_HE_patches/xgboost_prediction/Cancer Cells_Combined_individual_level_ratio100/xgboost_results_individual_level.pt")

# 6723_KL_1_region0
Mask_6723_KL_1 = [sample == "6723_KL_1" for sample in predicted_data["sample_ids"]]
Predictions_6723_KL_1 = [round(predicted_data["predictions"][i]*100, 4) for i, mask in enumerate(Mask_6723_KL_1) if mask]
Tile_ids_6723_KL_1 = [predicted_data["tile_ids"][i].split("_")[0] for i, mask in enumerate(Mask_6723_KL_1) if mask]

# Create DataFrame of prediction results with tile_id as index
prediction_df = pd.DataFrame({
    'tile_id': Tile_ids_6723_KL_1,
    'predicted_proportion': Predictions_6723_KL_1
}).set_index('tile_id')

# Match Spatial_location_df with prediction results
# Use left join to keep all spatial locations, fill NaN where no prediction exists
spatial_with_predictions = Spatial_location_df.join(prediction_df, how='left')


# Get corresponding deconvolution and expression data
deconv_data = pd.Series(focus_celltype_percent, index=Spatial_location_df.index)
expression_data = pd.Series(focus_relative_marker_genes_expression, index=Spatial_location_df.index)

showcase_df = pd.concat([
    deconv_data[Spatial_location_df.index],
    spatial_with_predictions['predicted_proportion'],
    expression_data[Spatial_location_df.index],
    Spatial_location_df[['x', 'y']]
], axis=1)
showcase_df.columns = ["Decovoluted_Proportion", "Predicted_Proportion", "Expression", "x", "y"]



### Show the decovoluted proportion of cancer cells
plt.figure(figsize=(10, 8))
ax = plt.gca()

# create the scatter plot with hexagon markers
plt.scatter(
    x=showcase_df['x'],
    y=showcase_df['y'],
    c=showcase_df['Decovoluted_Proportion'],
    cmap='coolwarm',  # the color from light to dark
    marker='h',  # hexagon marker
    s=100,  # marker size
    edgecolors='black',
    linewidth=0.5
    # Color range automatically adjusted based on data
)

# Add colorbar
cbar = plt.colorbar()
cbar.set_label('Tumor cell proportion (%)', fontsize=25, fontweight='bold', color='black')
cbar.ax.tick_params(labelsize=18, colors='black', width=3)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_color('black')

# invert the y axis to match the image coordinate system
ax.invert_yaxis()

# Set four borders as thick black lines
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

# Remove titles and axis labels
plt.title('')
plt.xlabel('')
plt.ylabel('')

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(f"Visual/6723_KL_1_Cancer_cells_deconvoluted_proportion.png", dpi=600, bbox_inches='tight', pad_inches=0.05)
# show the figure
plt.show()


# Show the predicted proportion of cancer cells
plt.figure(figsize=(10, 8))
ax = plt.gca()

# create the scatter plot with hexagon markers for predicted proportion
plt.scatter(
    x=showcase_df['x'],
    y=showcase_df['y'],
    c=showcase_df['Predicted_Proportion'],
    cmap='coolwarm',  # the color from light to dark
    marker='h',  # hexagon marker
    s=100,  # marker size
    edgecolors='black',
    linewidth=0.5
    # Color range automatically adjusted based on data
)

# Add colorbar
cbar = plt.colorbar()
cbar.set_label('Tumor cell proportion (%)', fontsize=25, fontweight='bold', color='black')
cbar.ax.tick_params(labelsize=18, colors='black', width=3)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_color('black')

# invert the y axis to match the image coordinate system
ax.invert_yaxis()

# Set the border of the figure to be black and thick
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

# Remove titles and axis labels
plt.title('')
plt.xlabel('')
plt.ylabel('')

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(f"Visual/6723_KL_1_Cancer_cells_predicted_proportion.png", dpi=600, bbox_inches='tight', pad_inches=0.05)
# show the figure
plt.show()





# Show the relative expression of the marker genes
plt.figure(figsize=(10, 8))
ax = plt.gca()

# create the scatter plot with hexagon markers
plt.scatter(
    x=showcase_df['x'],
    y=showcase_df['y'],
    c=showcase_df['Expression'],
    cmap='coolwarm',  # the color from light to dark
    marker='h',  # hexagon marker
    s=100,  # marker size
    edgecolors='black',
    linewidth=0.5
    # Color range automatically adjusted based on data
)

# Add colorbar
cbar = plt.colorbar()
cbar.set_label('Relative expression (%)', fontsize=25, fontweight='bold', color='black')
cbar.ax.tick_params(labelsize=18, colors='black', width=3)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_color('black')

# invert the y axis to match the image coordinate system
ax.invert_yaxis()

# Set four borders as thick black lines
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

# Remove titles and axis labels
plt.title('')
plt.xlabel('')
plt.ylabel('')

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])

plt.savefig(f"Visual/TENX152_Cancer_marker_genes_relative_expression.png", dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.show()




# Create scatter plot comparing deconvoluted proportion vs predicted proportion
plt.figure(figsize=(10, 8))
ax = plt.gca()

# Remove NaN values for correlation calculation
valid_data = showcase_df.dropna(subset=['Decovoluted_Proportion', 'Predicted_Proportion'])

# Calculate Pearson correlation
correlation, p_value = pearsonr(valid_data['Decovoluted_Proportion'], valid_data['Predicted_Proportion'])

# Create scatter plot with large points
plt.scatter(
    x=valid_data['Decovoluted_Proportion'],
    y=valid_data['Predicted_Proportion'],
    alpha=0.8,
    s=120,  # Large points
    edgecolors='black',
    linewidth=0.5,
    color='steelblue'
)

# Add linear regression line
X = valid_data['Decovoluted_Proportion'].values.reshape(-1, 1)
y = valid_data['Predicted_Proportion'].values
reg = LinearRegression().fit(X, y)

# Create line points
x_line = np.linspace(valid_data['Decovoluted_Proportion'].min(), 
                     valid_data['Decovoluted_Proportion'].max(), 100)
y_line = reg.predict(x_line.reshape(-1, 1))

# Plot regression line
plt.plot(x_line, y_line, 'r-', linewidth=3, alpha=0.8)

# Set labels with large, bold fonts
plt.xlabel('Deconvoluted Proportion (%)', fontsize=28, fontweight='bold')
plt.ylabel('Predicted Proportion (%)', fontsize=28, fontweight='bold')

# Set tick parameters
plt.tick_params(axis='x', labelsize=24, labelcolor='black', 
               width=2, length=6, colors='black')
plt.tick_params(axis='y', labelsize=24, labelcolor='black', 
               width=2, length=6, colors='black')

# Set tick labels to bold
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

# Add Pearson correlation text in upper left corner
plt.text(0.05, 0.95, f'Pearson r = {correlation:.3f}', 
         transform=ax.transAxes, fontsize=26, fontweight='bold',
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add grid with darker and thicker lines (same as line chart)
plt.grid(True, alpha=0.7, linestyle='--', linewidth=2)

# Set the border of the figure to be black and thick (same as line chart)
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

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('Visual/6723_KL_1_Cancer_cells_deconvoluted_vs_predicted_proportion_scatter.png', 
            dpi=600, bbox_inches='tight', facecolor='white')

# Show the figure
plt.show()








### For Stromal Cells
cell_type = "Stromal Cells"
expression_df = pd.read_csv(f'CARD_Need_Files/7003_AS_4_region0_expression.csv', index_col=0)
Celltype_proportion_df = pd.read_csv(f'CARD_Results_Regions/7003_AS_4_region0_celltype_proportion_modified.csv', index_col=0)
B_matrix_df = pd.read_csv(f'CARD_Results_Regions/7003_AS_4_region0_B_Matrix_modified.csv', index_col=0)
        
cell_type_order = ["ASC I", "ASC II", "ASC III", "CSC I", "CSC II", "CSC III", "CSC IV", "SSC I", 
"ABS", "CT", "EE", "TUF", "T", "PLA", "MAS", "MYE", "B", "FIB", "END"]

Celltype_proportion_df = Celltype_proportion_df[cell_type_order]

B_matrix_df.columns = ['ASC I', 'ASC II', 'ASC III', 'CSC III', 'CSC I', 'CSC IV', "CSC II", 'SSC I', 'ABS', 'CT', 'EE', 'TUF', 'T', 'PLA', 'MAS', 'MYE', 'FIB', 'B', 'END']
B_matrix_df = B_matrix_df[cell_type_order]

kept_barcode = Celltype_proportion_df.index.tolist()
expression_df = expression_df.loc[kept_barcode]

library_size = expression_df.sum(axis=1)
normalized_df = expression_df.copy()
normalized_df = expression_df.div(library_size, axis=0)

# Delete the marker genes that are not in B_matrix_df
marker_genes_clean_dict = {}
for key, value in final_ranked_gene_dict.items():
    marker_genes_clean_dict[key] = list(set(value).intersection(set(B_matrix_df.index)))

# Select the top 5 marker genes for each cell type
marker_genes_clean_dict["Cancer Cells"] = list(set(marker_genes_clean_dict["ASC I"][:5] + marker_genes_clean_dict["ASC II"][:5] +\
                                    marker_genes_clean_dict["ASC III"][:5] + marker_genes_clean_dict["CSC I"][:5] +\
                                    marker_genes_clean_dict["CSC II"][:5] + marker_genes_clean_dict["CSC III"][:5] +\
                                    marker_genes_clean_dict["CSC IV"][:5] + marker_genes_clean_dict["SSC I"][:5]))

marker_genes_clean_dict["Normal Epithelial Cells"] = list(set(marker_genes_clean_dict["ABS"][:10] + marker_genes_clean_dict["CT"][:10] +\
                                            marker_genes_clean_dict["EE"][:10] + marker_genes_clean_dict["TUF"][:10]))

marker_genes_clean_dict["T Cells"] = marker_genes_clean_dict["T"][:40]
marker_genes_clean_dict["Other Immune Cells"] = list(set(marker_genes_clean_dict["PLA"][:10] + marker_genes_clean_dict["MAS"][:10] + marker_genes_clean_dict["MYE"][:10] + marker_genes_clean_dict["B"][:10]))
marker_genes_clean_dict["Stromal Cells"] = list(set(marker_genes_clean_dict["FIB"][:20] + marker_genes_clean_dict["END"][:20]))

for cell_type_inner in ["ASC I", "ASC II", "ASC III", "CSC I", "CSC II", "CSC III", "CSC IV", "SSC I", "ABS", "CT", "EE", "TUF", "T", "PLA", "MAS", "MYE", "B", "FIB", "END"]:
    del marker_genes_clean_dict[cell_type_inner]

# Calculate the relative expression of the marker genes
Cancer_MarkerGE = normalized_df[marker_genes_clean_dict["Cancer Cells"]].div(B_matrix_df.T.loc[["ASC I", "ASC II", "ASC III", "CSC I", "CSC II", "CSC III", "CSC IV", "SSC I"],marker_genes_clean_dict["Cancer Cells"]].mean(axis=0)).mean(axis=1)  
NE_MarkerGE = normalized_df[marker_genes_clean_dict["Normal Epithelial Cells"]].div(B_matrix_df.T.loc[["ABS", "CT", "EE", "TUF"],marker_genes_clean_dict["Normal Epithelial Cells"]].mean(axis=0)).mean(axis=1)
T_MarkerGE = normalized_df[marker_genes_clean_dict["T Cells"]].div(B_matrix_df.T.loc[["T"],marker_genes_clean_dict["T Cells"]].mean(axis=0)).mean(axis = 1)
Other_Immune_MarkerGE = normalized_df[marker_genes_clean_dict["Other Immune Cells"]].div(B_matrix_df.T.loc[["PLA", "MAS", "MYE", "B"],marker_genes_clean_dict["Other Immune Cells"]].mean(axis=0)).mean(axis = 1)
Stromal_MarkerGE = normalized_df[marker_genes_clean_dict["Stromal Cells"]].div(B_matrix_df.T.loc[["FIB", "END"],marker_genes_clean_dict["Stromal Cells"]].mean(axis=0)).mean(axis = 1)
    
marker_genes_expression_df_grouped = pd.concat([Cancer_MarkerGE, NE_MarkerGE, T_MarkerGE, Other_Immune_MarkerGE, Stromal_MarkerGE], axis=1)
marker_genes_expression_df_grouped.columns = ["Cancer Cells", "Normal Epithelial Cells", "T Cells", "Other Immune Cells", "Stromal Cells"]
relative_marker_genes_expression_df_grouped = marker_genes_expression_df_grouped.div(marker_genes_expression_df_grouped.sum(axis=1)+1e-8, axis=0)

Total_Cancer_Celltype_proportion = Celltype_proportion_df.iloc[:,0:8].sum(axis=1)
Total_Normal_Epith_Celltype_proportion = Celltype_proportion_df.iloc[:,8:12].sum(axis=1)
Total_T_Celltype_proportion = Celltype_proportion_df.iloc[:,12]
Total_Other_Immune_Celltype_proportion = Celltype_proportion_df.iloc[:,13:17].sum(axis=1)
Total_Stromal_Celltype_proportion = Celltype_proportion_df.iloc[:,17:].sum(axis=1)
Celltype_proportion_df_grouped = pd.concat([Total_Cancer_Celltype_proportion, Total_Normal_Epith_Celltype_proportion, Total_T_Celltype_proportion, Total_Other_Immune_Celltype_proportion, Total_Stromal_Celltype_proportion], axis=1)
Celltype_proportion_df_grouped.columns = ["Cancer Cells", "Normal Epithelial Cells", "T Cells", "Other Immune Cells", "Stromal Cells"]


focus_celltype_percent = (Celltype_proportion_df_grouped[cell_type]*100).round(4).values.tolist()
focus_relative_marker_genes_expression = (relative_marker_genes_expression_df_grouped[cell_type]*100).round(4).values.tolist()



Spatial_location_df = pd.read_csv('/Users/scui2/ST/CARD_Need_Files/7003_AS_4_region0_spatial.csv', index_col=0)
predicted_data = torch.load("Colorectal_Cancer_HE_patches/xgboost_prediction/Stromal Cells_Combined_individual_level_ratio100/xgboost_results_individual_level.pt")

# 7003_AS_4_region0
Mask_7003_AS_4 = [sample == "7003_AS_4" for sample in predicted_data["sample_ids"]]
Predictions_7003_AS_4 = [round(predicted_data["predictions"][i]*100, 4) for i, mask in enumerate(Mask_7003_AS_4) if mask]
Tile_ids_7003_AS_4 = [predicted_data["tile_ids"][i].split("_")[0] for i, mask in enumerate(Mask_7003_AS_4) if mask]

# Create DataFrame of prediction results with tile_id as index
prediction_df = pd.DataFrame({
    'tile_id': Tile_ids_7003_AS_4,
    'predicted_proportion': Predictions_7003_AS_4
}).set_index('tile_id')

# Match Spatial_location_df with prediction results
# Use left join to keep all spatial locations, fill NaN where no prediction exists
spatial_with_predictions = Spatial_location_df.join(prediction_df, how='left')


# Get corresponding deconvolution and expression data
deconv_data = pd.Series(focus_celltype_percent, index=Spatial_location_df.index)
expression_data = pd.Series(focus_relative_marker_genes_expression, index=Spatial_location_df.index)

showcase_df = pd.concat([
    deconv_data[Spatial_location_df.index],
    spatial_with_predictions['predicted_proportion'],
    expression_data[Spatial_location_df.index],
    Spatial_location_df[['x', 'y']]
], axis=1)
showcase_df.columns = ["Decovoluted_Proportion", "Predicted_Proportion", "Expression", "x", "y"]



### Show the decovoluted proportion of cancer cells
plt.figure(figsize=(10, 8))
ax = plt.gca()

# create the scatter plot with hexagon markers
plt.scatter(
    x=showcase_df['x'],
    y=showcase_df['y'],
    c=showcase_df['Decovoluted_Proportion'],
    cmap='coolwarm',  # the color from light to dark
    marker='h',  # hexagon marker
    s=100,  # marker size
    edgecolors='black',
    linewidth=0.5
    # Color range automatically adjusted based on data
)

# Add colorbar
cbar = plt.colorbar()
cbar.set_label('Stromal cell proportion (%)', fontsize=25, fontweight='bold', color='black')
cbar.ax.tick_params(labelsize=18, colors='black', width=3)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_color('black')

# invert the y axis to match the image coordinate system
ax.invert_yaxis()

# Set four borders as thick black lines
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

# Remove titles and axis labels
plt.title('')
plt.xlabel('')
plt.ylabel('')

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(f"Visual/7003_AS_4_Stromal_cells_deconvoluted_proportion.png", dpi=600, bbox_inches='tight', pad_inches=0.05)
# show the figure
plt.show()


# Show the predicted proportion of cancer cells
plt.figure(figsize=(10, 8))
ax = plt.gca()

# create the scatter plot with hexagon markers for predicted proportion
plt.scatter(
    x=showcase_df['x'],
    y=showcase_df['y'],
    c=showcase_df['Predicted_Proportion'],
    cmap='coolwarm',  # the color from light to dark
    marker='h',  # hexagon marker
    s=100,  # marker size
    edgecolors='black',
    linewidth=0.5
    # Color range automatically adjusted based on data
)

# Add colorbar
cbar = plt.colorbar()
cbar.set_label('Stromal cell proportion (%)', fontsize=25, fontweight='bold', color='black')
cbar.ax.tick_params(labelsize=18, colors='black', width=3)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_color('black')

# invert the y axis to match the image coordinate system
ax.invert_yaxis()

# Set the border of the figure to be black and thick
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

# Remove titles and axis labels
plt.title('')
plt.xlabel('')
plt.ylabel('')

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(f"Visual/7003_AS_4_Stromal_cells_predicted_proportion.png", dpi=600, bbox_inches='tight', pad_inches=0.05)
# show the figure
plt.show()





# Show the relative expression of the marker genes
plt.figure(figsize=(10, 8))
ax = plt.gca()

# create the scatter plot with hexagon markers
plt.scatter(
    x=showcase_df['x'],
    y=showcase_df['y'],
    c=showcase_df['Expression'],
    cmap='coolwarm',  # the color from light to dark
    marker='h',  # hexagon marker
    s=100,  # marker size
    edgecolors='black',
    linewidth=0.5
    # Color range automatically adjusted based on data
)

# Add colorbar
cbar = plt.colorbar()
cbar.set_label('Relative expression (%)', fontsize=25, fontweight='bold', color='black')
cbar.ax.tick_params(labelsize=18, colors='black', width=3)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_color('black')

# invert the y axis to match the image coordinate system
ax.invert_yaxis()

# Set four borders as thick black lines
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

# Remove titles and axis labels
plt.title('')
plt.xlabel('')
plt.ylabel('')

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])

plt.savefig(f"Visual/TENX152_Cancer_marker_genes_relative_expression.png", dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.show()




# Create scatter plot comparing deconvoluted proportion vs predicted proportion
plt.figure(figsize=(10, 8))
ax = plt.gca()

# Remove NaN values for correlation calculation
valid_data = showcase_df.dropna(subset=['Decovoluted_Proportion', 'Predicted_Proportion'])

# Calculate Pearson correlation
correlation, p_value = pearsonr(valid_data['Decovoluted_Proportion'], valid_data['Predicted_Proportion'])

# Create scatter plot with large points
plt.scatter(
    x=valid_data['Decovoluted_Proportion'],
    y=valid_data['Predicted_Proportion'],
    alpha=0.8,
    s=120,  # Large points
    edgecolors='black',
    linewidth=0.5,
    color='steelblue'
)

# Add linear regression line
X = valid_data['Decovoluted_Proportion'].values.reshape(-1, 1)
y = valid_data['Predicted_Proportion'].values
reg = LinearRegression().fit(X, y)

# Create line points
x_line = np.linspace(valid_data['Decovoluted_Proportion'].min(), 
                     valid_data['Decovoluted_Proportion'].max(), 100)
y_line = reg.predict(x_line.reshape(-1, 1))

# Plot regression line
plt.plot(x_line, y_line, 'r-', linewidth=3, alpha=0.8)

# Set labels with large, bold fonts
plt.xlabel('Deconvoluted Proportion (%)', fontsize=28, fontweight='bold')
plt.ylabel('Predicted Proportion (%)', fontsize=28, fontweight='bold')

# Set tick parameters
plt.tick_params(axis='x', labelsize=24, labelcolor='black', 
               width=2, length=6, colors='black')
plt.tick_params(axis='y', labelsize=24, labelcolor='black', 
               width=2, length=6, colors='black')

# Set tick labels to bold
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

# Add Pearson correlation text in upper left corner
plt.text(0.05, 0.95, f'Pearson r = {correlation:.3f}', 
         transform=ax.transAxes, fontsize=26, fontweight='bold',
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add grid with darker and thicker lines (same as line chart)
plt.grid(True, alpha=0.7, linestyle='--', linewidth=2)

# Set the border of the figure to be black and thick (same as line chart)
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

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('Visual/7003_AS_4_Stromal_cells_deconvoluted_vs_predicted_proportion_scatter.png', 
            dpi=600, bbox_inches='tight', facecolor='white')

# Show the figure
plt.show()








### For T Cells
cell_type = "T Cells"
expression_df_region1 = pd.read_csv(f'CARD_Need_Files/SH-17-06138-A1_region1_expression.csv', index_col=0)
expression_df_region2 = pd.read_csv(f'CARD_Need_Files/SH-17-06138-A1_region2_expression.csv', index_col=0)

Celltype_proportion_df_region1 = pd.read_csv(f'CARD_Results_Regions/SH-17-06138-A1_region1_celltype_proportion_modified.csv', index_col=0)
Celltype_proportion_df_region2 = pd.read_csv(f'CARD_Results_Regions/SH-17-06138-A1_region2_celltype_proportion_modified.csv', index_col=0)
B_matrix_df = pd.read_csv(f'CARD_Results_Regions/SH-17-06138-A1_region1_B_Matrix_modified.csv', index_col=0)

expression_df = pd.concat([expression_df_region1, expression_df_region2], axis=0)
Celltype_proportion_df = pd.concat([Celltype_proportion_df_region1, Celltype_proportion_df_region2], axis=0)

cell_type_order = ["ASC I", "ASC II", "ASC III", "CSC I", "CSC II", "CSC III", "CSC IV", "SSC I", 
"ABS", "CT", "EE", "TUF", "T", "PLA", "MAS", "MYE", "B", "FIB", "END"]

Celltype_proportion_df = Celltype_proportion_df[cell_type_order]

B_matrix_df.columns = ['ASC I', 'ASC II', 'ASC III', 'CSC III', 'CSC I', 'CSC IV', "CSC II", 'SSC I', 'ABS', 'CT', 'EE', 'TUF', 'T', 'PLA', 'MAS', 'MYE', 'FIB', 'B', 'END']
B_matrix_df = B_matrix_df[cell_type_order]

kept_barcode = Celltype_proportion_df.index.tolist()
expression_df = expression_df.loc[kept_barcode]

library_size = expression_df.sum(axis=1)
normalized_df = expression_df.copy()
normalized_df = expression_df.div(library_size, axis=0)

# Delete the marker genes that are not in B_matrix_df
marker_genes_clean_dict = {}
for key, value in final_ranked_gene_dict.items():
    marker_genes_clean_dict[key] = list(set(value).intersection(set(B_matrix_df.index)))

# Select the top 5 marker genes for each cell type
marker_genes_clean_dict["Cancer Cells"] = list(set(marker_genes_clean_dict["ASC I"][:5] + marker_genes_clean_dict["ASC II"][:5] +\
                                    marker_genes_clean_dict["ASC III"][:5] + marker_genes_clean_dict["CSC I"][:5] +\
                                    marker_genes_clean_dict["CSC II"][:5] + marker_genes_clean_dict["CSC III"][:5] +\
                                    marker_genes_clean_dict["CSC IV"][:5] + marker_genes_clean_dict["SSC I"][:5]))

marker_genes_clean_dict["Normal Epithelial Cells"] = list(set(marker_genes_clean_dict["ABS"][:10] + marker_genes_clean_dict["CT"][:10] +\
                                            marker_genes_clean_dict["EE"][:10] + marker_genes_clean_dict["TUF"][:10]))

marker_genes_clean_dict["T Cells"] = marker_genes_clean_dict["T"][:40]
marker_genes_clean_dict["Other Immune Cells"] = list(set(marker_genes_clean_dict["PLA"][:10] + marker_genes_clean_dict["MAS"][:10] + marker_genes_clean_dict["MYE"][:10] + marker_genes_clean_dict["B"][:10]))
marker_genes_clean_dict["Stromal Cells"] = list(set(marker_genes_clean_dict["FIB"][:20] + marker_genes_clean_dict["END"][:20]))

for cell_type_inner in ["ASC I", "ASC II", "ASC III", "CSC I", "CSC II", "CSC III", "CSC IV", "SSC I", "ABS", "CT", "EE", "TUF", "T", "PLA", "MAS", "MYE", "B", "FIB", "END"]:
    del marker_genes_clean_dict[cell_type_inner]

# Calculate the relative expression of the marker genes
Cancer_MarkerGE = normalized_df[marker_genes_clean_dict["Cancer Cells"]].div(B_matrix_df.T.loc[["ASC I", "ASC II", "ASC III", "CSC I", "CSC II", "CSC III", "CSC IV", "SSC I"],marker_genes_clean_dict["Cancer Cells"]].mean(axis=0)).mean(axis=1)  
NE_MarkerGE = normalized_df[marker_genes_clean_dict["Normal Epithelial Cells"]].div(B_matrix_df.T.loc[["ABS", "CT", "EE", "TUF"],marker_genes_clean_dict["Normal Epithelial Cells"]].mean(axis=0)).mean(axis=1)
T_MarkerGE = normalized_df[marker_genes_clean_dict["T Cells"]].div(B_matrix_df.T.loc[["T"],marker_genes_clean_dict["T Cells"]].mean(axis=0)).mean(axis = 1)
Other_Immune_MarkerGE = normalized_df[marker_genes_clean_dict["Other Immune Cells"]].div(B_matrix_df.T.loc[["PLA", "MAS", "MYE", "B"],marker_genes_clean_dict["Other Immune Cells"]].mean(axis=0)).mean(axis = 1)
Stromal_MarkerGE = normalized_df[marker_genes_clean_dict["Stromal Cells"]].div(B_matrix_df.T.loc[["FIB", "END"],marker_genes_clean_dict["Stromal Cells"]].mean(axis=0)).mean(axis = 1)
    
marker_genes_expression_df_grouped = pd.concat([Cancer_MarkerGE, NE_MarkerGE, T_MarkerGE, Other_Immune_MarkerGE, Stromal_MarkerGE], axis=1)
marker_genes_expression_df_grouped.columns = ["Cancer Cells", "Normal Epithelial Cells", "T Cells", "Other Immune Cells", "Stromal Cells"]
relative_marker_genes_expression_df_grouped = marker_genes_expression_df_grouped.div(marker_genes_expression_df_grouped.sum(axis=1)+1e-8, axis=0)

Total_Cancer_Celltype_proportion = Celltype_proportion_df.iloc[:,0:8].sum(axis=1)
Total_Normal_Epith_Celltype_proportion = Celltype_proportion_df.iloc[:,8:12].sum(axis=1)
Total_T_Celltype_proportion = Celltype_proportion_df.iloc[:,12]
Total_Other_Immune_Celltype_proportion = Celltype_proportion_df.iloc[:,13:17].sum(axis=1)
Total_Stromal_Celltype_proportion = Celltype_proportion_df.iloc[:,17:].sum(axis=1)
Celltype_proportion_df_grouped = pd.concat([Total_Cancer_Celltype_proportion, Total_Normal_Epith_Celltype_proportion, Total_T_Celltype_proportion, Total_Other_Immune_Celltype_proportion, Total_Stromal_Celltype_proportion], axis=1)
Celltype_proportion_df_grouped.columns = ["Cancer Cells", "Normal Epithelial Cells", "T Cells", "Other Immune Cells", "Stromal Cells"]


focus_celltype_percent = (Celltype_proportion_df_grouped[cell_type]*100).round(4).values.tolist()
focus_relative_marker_genes_expression = (relative_marker_genes_expression_df_grouped[cell_type]*100).round(4).values.tolist()



Spatial_location_df_region1 = pd.read_csv('/Users/scui2/Desktop/FredHutch_Colorectal/CARD_Need_Files/SH-17-06138-A1_region1_spatial.csv', index_col=0)
Spatial_location_df_region2 = pd.read_csv('/Users/scui2/Desktop/FredHutch_Colorectal/CARD_Need_Files/SH-17-06138-A1_region2_spatial.csv', index_col=0)
Spatial_location_df = pd.concat([Spatial_location_df_region1, Spatial_location_df_region2], axis=0)

predicted_data = torch.load("Colorectal_Cancer_HE_patches/xgboost_prediction/T Cells_Combined_individual_level_ratio100/xgboost_results_individual_level.pt")




# SH-17-06138-A1
Mask_SH_17_06138_A1 = [sample == "SH-17-06138-A1" for sample in predicted_data["sample_ids"]]
Predictions_SH_17_06138_A1 = [round(predicted_data["predictions"][i]*100, 4) for i, mask in enumerate(Mask_SH_17_06138_A1) if mask]
Tile_ids_SH_17_06138_A1 = [predicted_data["tile_ids"][i].split("_")[0] for i, mask in enumerate(Mask_SH_17_06138_A1) if mask]

# Create DataFrame of prediction results with tile_id as index
prediction_df = pd.DataFrame({
    'tile_id': Tile_ids_SH_17_06138_A1,
    'predicted_proportion': Predictions_SH_17_06138_A1
}).set_index('tile_id')

# Match Spatial_location_df with prediction results
# Use left join to keep all spatial locations, fill NaN where no prediction exists
spatial_with_predictions = Spatial_location_df.join(prediction_df, how='left')


# Get corresponding deconvolution and expression data
deconv_data = pd.Series(focus_celltype_percent, index=Spatial_location_df.index)
expression_data = pd.Series(focus_relative_marker_genes_expression, index=Spatial_location_df.index)

showcase_df = pd.concat([
    deconv_data[Spatial_location_df.index],
    spatial_with_predictions['predicted_proportion'],
    expression_data[Spatial_location_df.index],
    Spatial_location_df[['x', 'y']]
], axis=1)
showcase_df.columns = ["Decovoluted_Proportion", "Predicted_Proportion", "Expression", "x", "y"]



### Show the decovoluted proportion of cancer cells
plt.figure(figsize=(10, 8))
ax = plt.gca()

# create the scatter plot with hexagon markers
plt.scatter(
    x=showcase_df['x'],
    y=showcase_df['y'],
    c=showcase_df['Decovoluted_Proportion'],
    cmap='coolwarm',  # the color from light to dark
    marker='h',  # hexagon marker
    s=100,  # marker size
    edgecolors='black',
    linewidth=0.5
    # Color range automatically adjusted based on data
)

# Add colorbar
cbar = plt.colorbar()
cbar.set_label('T cell proportion (%)', fontsize=25, fontweight='bold', color='black')
cbar.ax.tick_params(labelsize=18, colors='black', width=3)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_color('black')

# invert the y axis to match the image coordinate system
ax.invert_yaxis()

# Set four borders as thick black lines
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

# Remove titles and axis labels
plt.title('')
plt.xlabel('')
plt.ylabel('')

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(f"Visual/SH-17-06138-A1_region1_T_cells_deconvoluted_proportion.png", dpi=600, bbox_inches='tight', pad_inches=0.05)
# show the figure
plt.show()


# Show the predicted proportion of cancer cells
plt.figure(figsize=(10, 8))
ax = plt.gca()

# create the scatter plot with hexagon markers for predicted proportion
plt.scatter(
    x=showcase_df['x'],
    y=showcase_df['y'],
    c=showcase_df['Predicted_Proportion'],
    cmap='coolwarm',  # the color from light to dark
    marker='h',  # hexagon marker
    s=100,  # marker size
    edgecolors='black',
    linewidth=0.5
    # Color range automatically adjusted based on data
)

# Add colorbar
cbar = plt.colorbar()
cbar.set_label('T cell proportion (%)', fontsize=25, fontweight='bold', color='black')
cbar.ax.tick_params(labelsize=18, colors='black', width=3)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_color('black')

# invert the y axis to match the image coordinate system
ax.invert_yaxis()

# Set the border of the figure to be black and thick
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

# Remove titles and axis labels
plt.title('')
plt.xlabel('')
plt.ylabel('')

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(f"Visual/SH-17-06138-A1_region1_T_cells_predicted_proportion.png", dpi=600, bbox_inches='tight', pad_inches=0.05)
# show the figure
plt.show()





# Show the relative expression of the marker genes
plt.figure(figsize=(10, 8))
ax = plt.gca()

# create the scatter plot with hexagon markers
plt.scatter(
    x=showcase_df['x'],
    y=showcase_df['y'],
    c=showcase_df['Expression'],
    cmap='coolwarm',  # the color from light to dark
    marker='h',  # hexagon marker
    s=100,  # marker size
    edgecolors='black',
    linewidth=0.5
    # Color range automatically adjusted based on data
)

# Add colorbar
cbar = plt.colorbar()
cbar.set_label('Relative expression (%)', fontsize=25, fontweight='bold', color='black')
cbar.ax.tick_params(labelsize=18, colors='black', width=3)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_color('black')

# invert the y axis to match the image coordinate system
ax.invert_yaxis()

# Set four borders as thick black lines
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

# Remove titles and axis labels
plt.title('')
plt.xlabel('')
plt.ylabel('')

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])

plt.savefig(f"Visual/SH-17-06138-A1_T_cells_marker_genes_relative_expression.png", dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.show()




# Create scatter plot comparing deconvoluted proportion vs predicted proportion
plt.figure(figsize=(10, 8))
ax = plt.gca()

# Remove NaN values for correlation calculation
valid_data = showcase_df.dropna(subset=['Decovoluted_Proportion', 'Predicted_Proportion'])

# Calculate Pearson correlation
correlation, p_value = pearsonr(valid_data['Decovoluted_Proportion'], valid_data['Predicted_Proportion'])

# Create scatter plot with large points
plt.scatter(
    x=valid_data['Decovoluted_Proportion'],
    y=valid_data['Predicted_Proportion'],
    alpha=0.8,
    s=120,  # Large points
    edgecolors='black',
    linewidth=0.5,
    color='steelblue'
)

# Add linear regression line
X = valid_data['Decovoluted_Proportion'].values.reshape(-1, 1)
y = valid_data['Predicted_Proportion'].values
reg = LinearRegression().fit(X, y)

# Create line points
x_line = np.linspace(valid_data['Decovoluted_Proportion'].min(), 
                     valid_data['Decovoluted_Proportion'].max(), 100)
y_line = reg.predict(x_line.reshape(-1, 1))

# Plot regression line
plt.plot(x_line, y_line, 'r-', linewidth=3, alpha=0.8)

# Set labels with large, bold fonts
plt.xlabel('Deconvoluted Proportion (%)', fontsize=28, fontweight='bold')
plt.ylabel('Predicted Proportion (%)', fontsize=28, fontweight='bold')

# Set tick parameters
plt.tick_params(axis='x', labelsize=24, labelcolor='black', 
               width=2, length=6, colors='black')
plt.tick_params(axis='y', labelsize=24, labelcolor='black', 
               width=2, length=6, colors='black')

# Set tick labels to bold
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

# Add Pearson correlation text in upper left corner
plt.text(0.05, 0.95, f'Pearson r = {correlation:.3f}', 
         transform=ax.transAxes, fontsize=26, fontweight='bold',
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add grid with darker and thicker lines (same as line chart)
plt.grid(True, alpha=0.7, linestyle='--', linewidth=2)

# Set the border of the figure to be black and thick (same as line chart)
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

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('Visual/SH-17-06138-A1_T_cells_deconvoluted_vs_predicted_proportion_scatter.png', 
            dpi=600, bbox_inches='tight', facecolor='white')

# Show the figure
plt.show()
