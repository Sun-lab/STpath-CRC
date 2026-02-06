"""
STPath-COAD: XGBoost Model Comparison Analysis (Colorectal) for Figure 4
==========================================================

This script performs cross-validated performance, gene expression, and spatial agreement of STpath predictions for colorectal cancer,
to generate plots for Figure 4 panel A-C.

Author: Saishi Cui
Date: Feb 2026


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


### Figure 4 (Panel A), Model Comparison


# Collect all the results
all_results = []

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

for cell_type in ["Cancer Cells", "Stromal Cells", "Normal Epithelial Cells", "T Cells", "Other Immune Cells"]:
    for model in ["ResNet50", "Conch", "Prov-GigaPath", "UNI2-h", "Virchow", "Virchow2", "Combined"]:

        file_model_name = model_file_names[model]
        df = pd.read_csv(f"/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/xgboost_prediction/{cell_type}_{file_model_name}_individual_level_ratio100/individual_metrics_individual_level.csv")
        
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

# Map cell type names for display
final_df['cell_type'] = final_df['cell_type'].replace('Other Immune Cells', 'pan-APC')

final_df.to_csv(f"xgboost_prediction/individual_metrics_individual_level_filtered.csv", index=False)




# Set the figure style
plt.style.use('default')
sns.set_palette("husl")

# Define the color mapping
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

# Create a separate figure for each metric
for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
    # Create a separate figure
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    
    # Debug: Check data for current metric
    if metric == 'MAE':
        print(f"\nDEBUG - {metric} data check:")
        for cell_type in ["Cancer Cells", "Stromal Cells", "Normal Epithelial Cells", "T Cells", "pan-APC"]:
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
        linewidth=3,  # Thicker box borders
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
    plt.savefig(f'/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/Visual/model_comparison_{metric}_boxplot.png', 
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




###### Figure 4 (Panel B), Visualize some samples for 3 cell types


### Marker genes refinement
with open('scRNAseq_data/final_marker_genes_dict.json', 'r') as f:
    final_marker_genes_dict = json.load(f)

del final_marker_genes_dict["Cancer"]
del final_marker_genes_dict["Normal Epithelia"]



InputDf_for_CARD_SelectedGenes = pd.read_csv('scRNAseq_data/InputDf_for_CARD_SelectedGenes.csv', index_col=0)
InputDf_for_CARD_meta = pd.read_csv('/Users/scui2/Desktop/scRNAseq_data/InputDf_for_CARD_meta.csv', index_col=0)
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



Spatial_location_df = pd.read_csv('/Users/scui2/Desktop/CARD_Need_Files/6723_KL_1_region0_spatial.csv', index_col=0)
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



Spatial_location_df = pd.read_csv('/Users/scui2/Desktop/CARD_Need_Files/7003_AS_4_region0_spatial.csv', index_col=0)
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



###### Figure 4 (Panel C) Gene expression prediction comparison.

# Define color mapping for gene groups
color_mapping = {
    'Tumor Markers': '#E41A1C',      # Red
    'Normal Epithelial Markers': '#377EB8', # Blue
    'T Markers': '#4DAF4A',           # Green
    'pan-APC Markers': '#FF7F00',      # Orange
    'Stromal Markers': '#FFFF33',           # Yellow
    'Highly Variable Genes': '#984EA3'  # Purple
}


## Boxplot Comparison of 6 Models for Cell Type Marker Gene Prediction

print("\n" + "="*80)
print("STEP 5: Creating boxplot comparison of 6 models for cell type marker gene prediction")
print("="*80)

# Load correlation data for all 6 models (with updated model names)
model_files = {
    'Virchow2': '/Gene_Expression_Prediction/180genes_regular_correlations.csv',
    'ResNet50': '/Gene_Expression_Prediction/180genes_ResNet50_regular_correlations.csv',
    'UNI2-h': '/Gene_Expression_Prediction/180genes_UNI2h_regular_correlations.csv',
    'Virchow': '/Gene_Expression_Prediction/180genes_Virchow_regular_correlations.csv',
    'Prov-GigaPath': '/Gene_Expression_Prediction/180genes_ProvGigPath_regular_correlations.csv',
    'CONCH': '/Gene_Expression_Prediction/180genes_CONCH_regular_correlations.csv'
}

# Load and process data for each model
all_boxplot_data = []

for model_name, file_path in model_files.items():
    print(f"Loading {model_name} data...")
    df = pd.read_csv(file_path, index_col=0)
    
    # Get first 180 genes only (columns)
    genes = df.columns.tolist()[:180]  # Only take first 180 genes
    
    # For Virchow2 (multiple rows), calculate median
    if model_name == 'Virchow2':
        correlations = df.iloc[:, :180].median(axis=0)  # Median across samples, first 180 genes
    else:
        # For other models (single row)
        correlations = df.iloc[0, :180]  # First row, first 180 genes
    
    # Create gene groups (same logic as scatter plots)
    for i, gene in enumerate(genes):
        if i < 30:
            cell_type = 'Tumor Markers'
        elif i < 60:
            cell_type = 'Normal Epithelial Markers'
        elif i < 90:
            cell_type = 'T Markers'
        elif i < 120:
            cell_type = 'pan-APC Markers'
        elif i < 150:
            cell_type = 'Stromal Markers'
        else:  # 150+
            cell_type = 'Highly Variable Genes'
        
        # Include all gene types including Highly Variable Genes
        all_boxplot_data.append({
            'Model': model_name,
            'Cell_Type': cell_type,
            'Gene': gene,
            'Regular_Correlation': correlations[gene]
        })

# Create DataFrame
boxplot_df = pd.DataFrame(all_boxplot_data)
print(f"Boxplot data shape: {boxplot_df.shape}")
print(f"Models: {boxplot_df['Model'].unique()}")
print(f"Cell types: {boxplot_df['Cell_Type'].unique()}")

# Define color palette (similar to Figure4) with updated model names
color_palette = {
    'ResNet50': '#FEF0DE',
    'CONCH': '#C43E96', 
    'Prov-GigaPath': '#DEDBEE',
    'UNI2-h': '#06948E',
    'Virchow': '#F3CDCC',
    'Virchow2': '#F0CF7F'
}

# Define the order of cell types and models for consistent plotting
cell_type_order = ['Tumor Markers', 'Stromal Markers', 'Normal Epithelial Markers', 'T Markers', 'pan-APC Markers', 'Highly Variable Genes']
model_order = ['ResNet50', 'CONCH', 'Prov-GigaPath', 'UNI2-h', 'Virchow', 'Virchow2']

# Create boxplot with flatter aspect ratio
plt.figure(figsize=(18, 8))

# Create boxplot with specified order and thicker lines (no legend)
box_plot = sns.boxplot(
    data=boxplot_df, 
    x='Cell_Type', 
    y='Regular_Correlation', 
    hue='Model',
    order=cell_type_order,
    hue_order=model_order,
    palette=color_palette,
    showfliers=False,  # Do not show outliers
    linewidth=3,  # Thicker box borders
    legend=False  # No legend
)

# Add stripplot for individual points with larger size
sns.stripplot(
    data=boxplot_df, 
    x='Cell_Type', 
    y='Regular_Correlation', 
    hue='Model',
    order=cell_type_order,
    hue_order=model_order,
    palette=color_palette,
    size=6,  # Larger points
    alpha=0.7,
    dodge=True,  # Separate points by hue
    jitter=0.3,  # Add jitter
    edgecolor='black',
    linewidth=0.5,  # Thicker point borders
    legend=False  # Don't show stripplot legend
)

# Styling with larger fonts and title
plt.xlabel('', fontsize=24, fontweight='bold')  # No x-axis label (increased by 2)
plt.ylabel('Pearson Correlation', fontsize=24, fontweight='bold')  # Increased by 2
plt.title('Gene Expression Prediction Comparison', fontsize=26, fontweight='bold')  # Add title

# Set tick parameters with larger fonts
plt.tick_params(axis='x', rotation=15, labelsize=20, labelcolor='black', 
               width=2, length=6, colors='black')  # Increased by 2
plt.tick_params(axis='y', labelsize=20, labelcolor='black', 
               width=2, length=6, colors='black')  # Increased by 2

# Make tick labels bold
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontweight('bold')

# Update x-axis labels (keep "Markers" and change "Highly Variable Genes" to "HVGs")
x_labels = []
for label in plt.gca().get_xticklabels():
    text = label.get_text()
    if text == 'Highly Variable Genes':
        text = 'HVGs'
    x_labels.append(text)
plt.gca().set_xticklabels(x_labels)

# Add thick black border around the plot
for spine in plt.gca().spines.values():
    spine.set_linewidth(3)
    spine.set_edgecolor('black')

# Grid
plt.grid(True, alpha=0.3, linewidth=1)

# Tight layout
plt.tight_layout()

# Save plot
boxplot_output_path = "/Visual/6_Models_Cell_Type_Markers_Boxplot.png"
plt.savefig(boxplot_output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Print summary statistics
print("\nSummary statistics by model and cell type:")
summary_stats = boxplot_df.groupby(['Model', 'Cell_Type'])['Regular_Correlation'].agg(['mean', 'std', 'median', 'count']).round(3)
print(summary_stats)




### Virchow2 vs ResNet50 Comparison Scatter Plot

print("\n" + "="*80)
print("STEP 6: Creating Virchow2 vs ResNet50 comparison scatter plot")
print("="*80)

# Load correlation data for Virchow2 and ResNet50
virchow2_regular_df = pd.read_csv("/Gene_Expression_Prediction/180genes_regular_correlations.csv", index_col=0)
resnet50_regular_df = pd.read_csv("/Gene_Expression_Prediction/180genes_ResNet50_regular_correlations.csv", index_col=0)

# Get first 180 genes for both models
genes_180_comparison = virchow2_regular_df.columns.tolist()[:180]

# Calculate correlations
virchow2_regular_median = virchow2_regular_df.iloc[:, :180].median(axis=0)  # Median for Virchow2
resnet50_regular_corr = resnet50_regular_df.iloc[0, :180]  # Single row for ResNet50

print(f"Using {len(genes_180_comparison)} genes for comparison")

# Create gene groups (same logic as previous scatter plots)
comparison_gene_groups = []
comparison_gene_colors = []

for i, gene in enumerate(genes_180_comparison):
    if i < 30:
        comparison_gene_groups.append('Tumor Markers')
        comparison_gene_colors.append(color_mapping['Tumor Markers'])
    elif i < 60:
        comparison_gene_groups.append('Normal Epithelial Markers')
        comparison_gene_colors.append(color_mapping['Normal Epithelial Markers'])
    elif i < 90:
        comparison_gene_groups.append('T Markers')
        comparison_gene_colors.append(color_mapping['T Markers'])
    elif i < 120:
        comparison_gene_groups.append('pan-APC Markers')
        comparison_gene_colors.append(color_mapping['pan-APC Markers'])
    elif i < 150:
        comparison_gene_groups.append('Stromal Markers')
        comparison_gene_colors.append(color_mapping['Stromal Markers'])
    else:  # 150-179
        comparison_gene_groups.append('Highly Variable Genes')
        comparison_gene_colors.append(color_mapping['Highly Variable Genes'])

# Create DataFrame for comparison plotting
comparison_plot_df = pd.DataFrame({
    'Gene': genes_180_comparison,
    'Virchow2_Correlation': virchow2_regular_median,
    'ResNet50_Correlation': resnet50_regular_corr,
    'Gene_Group': comparison_gene_groups,
    'Color': comparison_gene_colors
})

# Set fixed point size (same as previous scatter plots)
comparison_plot_df['Point_Size'] = 150

print(f"Comparison Gene group distribution:")
print(comparison_plot_df['Gene_Group'].value_counts())

# Create comparison scatter plot
plt.figure(figsize=(12, 10))

# Plot each group separately for legend
for group in ['Tumor Markers', 'Normal Epithelial Markers', 'T Markers', 'pan-APC Markers', 'Stromal Markers', 'Highly Variable Genes']:
    group_data = comparison_plot_df[comparison_plot_df['Gene_Group'] == group]
    if len(group_data) > 0:
        # Special case for Highly Variable Genes - show as HVGs with n=30
        if group == 'Highly Variable Genes':
            label_text = 'HVGs (n=30)'
        else:
            label_text = f'{group} (n={len(group_data)})'
            
        plt.scatter(
            group_data['Virchow2_Correlation'],
            group_data['ResNet50_Correlation'],
            s=group_data['Point_Size'],
            c=color_mapping[group],
            alpha=0.7,
            label=label_text,
            edgecolors='black',
            linewidth=0.5
        )

# Set labels with large fonts (no title)
plt.xlabel('Pearson Correlation (Virchow2)', fontsize=24, fontweight='bold')
plt.ylabel('Pearson Correlation (ResNet50)', fontsize=24, fontweight='bold')

# Styling with larger fonts and thicker borders (same as previous scatter plots)
plt.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontweight('bold')

# Add thick black border around the plot
for spine in plt.gca().spines.values():
    spine.set_linewidth(3)
    spine.set_edgecolor('black')

# Legend with bold text and larger font
legend = plt.legend(loc='upper left', fontsize=18, frameon=True, fancybox=True, shadow=True,  # Increased by 2
                   markerscale=1.5)  # Increase legend marker size
for text in legend.get_texts():
    text.set_fontweight('bold')

# Set axis limits (same as previous scatter plots)
plt.xlim(-0.1, 0.6)
plt.ylim(-0.1, 0.6)

# Add diagonal reference line
plt.plot([-0.1, 0.6], [-0.1, 0.6], 'k--', alpha=0.5, linewidth=2)

# Annotate genes with Virchow2 correlation > 0.45
high_virchow2_genes = comparison_plot_df[comparison_plot_df['Virchow2_Correlation'] > 0.45]
print(f"\nGenes with Virchow2 correlation > 0.45: {len(high_virchow2_genes)}")

for i, (idx, row) in enumerate(high_virchow2_genes.iterrows()):
    gene_name = row['Gene']
    x_pos = row['Virchow2_Correlation']
    y_pos = row['ResNet50_Correlation']
    
    # Special positioning for EPCAM gene (same as previous scatter plots)
    if gene_name == 'EPCAM':
        x_offset = 15
        y_offset = -8
    else:
        x_offset = 0
        y_offset = -8
    
    plt.annotate(gene_name, (x_pos, y_pos), 
                xytext=(x_offset, y_offset), textcoords='offset points',
                fontsize=18, fontweight='black',  # Larger and blacker
                color='black',
                ha='center', va='top')
    
    print(f"  {gene_name}: Virchow2={x_pos:.3f}, ResNet50={y_pos:.3f}, Group={row['Gene_Group']}")

# Grid and layout
plt.grid(True, alpha=0.3, linewidth=1)
plt.tight_layout()

# Save comparison plot
comparison_output_path = "/Visual/Virchow2_vs_ResNet50_Comparison_Scatter_Plot.png"
plt.savefig(comparison_output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()





















