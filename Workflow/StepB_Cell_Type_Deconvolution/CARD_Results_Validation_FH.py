"""
STPath-COAD: CARD Results Validation for Fred Hutchinson Dataset
===============================================================

This script validates CARD deconvolution results specifically for the Fred Hutchinson
cancer center dataset and performs statistical analysis of cell type proportions.

Author: Saishi Cui
Date: Sept 2025

Purpose: Validate CARD deconvolution results on Fred Hutchinson dataset, perform 
statistical comparisons, and generate validation plots for cell type proportion 
estimates in the Fred Hutchinson cancer center dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import json
from scipy import stats
import statsmodels.api as sm
import shutil






#################
# define the path
path_CARD_results_celltype_proportion = "FredHutch_Colorectal/CARD_Results_Regions/"

# get the path list of all .csv files
CARD_results_celltype_proportion_path_list = []
for root, dirs, files in os.walk(path_CARD_results_celltype_proportion):
    for file in files:
        if file.endswith("_celltype_proportion_modified.csv"):
            CARD_results_celltype_proportion_path_list.append(os.path.join(root, file))

CARD_results_celltype_proportion_path_list = sorted(CARD_results_celltype_proportion_path_list)




# extract the sample name from the file path
Region_ID_list = []
Individual_ID_list = []
file_path = CARD_results_celltype_proportion_path_list[0]
for file_path in CARD_results_celltype_proportion_path_list:
    # extract the sample name from the file path
    region_id = "_".join(os.path.basename(file_path).split("_")[0:2])
    Region_ID_list.append(region_id)
    individual_id = os.path.basename(file_path).split("_")[0]
    Individual_ID_list.append(individual_id)


Region_ID_list = sorted(list(set(Region_ID_list)))
Individual_ID_list = sorted(list(set(Individual_ID_list)))





### Marker genes refinement
with open('data/scRNAseq_data/final_marker_genes_dict.json', 'r') as f:
    final_marker_genes_dict = json.load(f)

del final_marker_genes_dict["Cancer"]
del final_marker_genes_dict["Normal Epithelia"]


InputDf_for_CARD_SelectedGenes = pd.read_csv('data/scRNAseq_data/InputDf_for_CARD_SelectedGenes.csv', index_col=0)
InputDf_for_CARD_meta = pd.read_csv('data/scRNAseq_data/InputDf_for_CARD_meta.csv', index_col=0)
InputDf_for_CARD_meta[InputDf_for_CARD_meta["Cell Type"] == "CD4+ T"] = "T"
InputDf_for_CARD_meta[InputDf_for_CARD_meta["Cell Type"] == "CD8+ T"] = "T"


InputDF_CounNorm_SelectedGenes = InputDf_for_CARD_SelectedGenes.T.div(InputDf_for_CARD_SelectedGenes.T.sum(axis=1), axis=0)

  ## take marker genes based on the log2FC for each cell type
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





### Check the relationship between cell type proportion and marker gene expression (Sample region level)

def CTProportion_vs_Relative_expression_Visualization(cell_type, marker_genes_dict=final_ranked_gene_dict):
    
    all_region_ids = []
    all_rhos = []
    all_pvals = []
    all_number_of_spots = []
    all_min_proportion = []
    all_max_proportion = []
    all_min_expression = []
    all_max_expression = []
    

    for Region_ID in Region_ID_list:
        print(f"Processing region: {Region_ID}")
        
        # Load data
        expression_df = pd.read_csv(f'FredHutch_Colorectal/CARD_Need_Files/{Region_ID}_expression.csv', index_col=0)
        Celltype_proportion_df = pd.read_csv(f'FredHutch_Colorectal/CARD_Results_Regions/{Region_ID}_celltype_proportion_modified.csv', index_col=0)
        B_matrix_df = pd.read_csv(f'FredHutch_Colorectal/CARD_Results_Regions/{Region_ID}_B_Matrix_modified.csv', index_col=0)
        
        cell_type_order = ["ASC I", "ASC II", "ASC III", "CSC I", "CSC II", 
        "CSC III", "CSC IV", "SSC I", 
        "ABS", "CT", "EE", "TUF",
        "T", "PLA", "MAS", "MYE", "B", "FIB", "END"]


        Celltype_proportion_df = Celltype_proportion_df[cell_type_order]
        
        B_matrix_df.columns = ['ASC I', 'ASC II', 'ASC III', 'CSC III', 'CSC I', 'CSC IV', "CSC II", 'SSC I', 'ABS', 'CT', 'EE', 'TUF', 'T', 'PLA', 'MAS', 'MYE', 'FIB', 'B', 'END']
        B_matrix_df = B_matrix_df[cell_type_order]
       
        kept_barcode = Celltype_proportion_df.index.tolist()
        expression_df = expression_df.loc[kept_barcode]

        library_size = expression_df.sum(axis=1)
        normalized_df = expression_df.copy()
        normalized_df = expression_df.div(library_size, axis=0)

        # Remove marker genes not in B_matrix_df
        marker_genes_clean_dict = {}
        for key, value in marker_genes_dict.items():
            marker_genes_clean_dict[key] = list(set(value).intersection(set(B_matrix_df.index)))

        # Group top 5 marker genes for each cancer cell and normal epithelial cell type
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

        # Calculate relative expression of marker genes
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
        
        # Create individual plots
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Draw scatter plots and regression lines
        
        # Check if all x values are the same (cannot perform regression analysis)
        if len(set(focus_celltype_percent)) <= 1:
            # If all x values are the same, only draw scatter plot without regression analysis
            scatter = ax.scatter(focus_celltype_percent, focus_relative_marker_genes_expression, 
                                alpha=1.0, s=80, c='black', edgecolor='black', linewidth=1)
            
            # Add explanatory text
            ax.text(0.05, 0.90, "All proportions are identical\nCannot calculate regression",
                    transform=ax.transAxes, fontsize=20, 
                    color='red', fontweight='bold', ha='left',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3'))
            
            all_rhos.append(float('nan'))
            all_pvals.append(float('nan'))
            all_number_of_spots.append(len(focus_celltype_percent))
            all_region_ids.append(Region_ID)
            all_min_proportion.append(round(min(focus_celltype_percent), 4))
            all_max_proportion.append(round(max(focus_celltype_percent), 4))
            all_min_expression.append(round(min(focus_relative_marker_genes_expression), 4))
            all_max_expression.append(round(max(focus_relative_marker_genes_expression), 4))
    

        else:
            scatter = ax.scatter(focus_celltype_percent, focus_relative_marker_genes_expression, 
                                alpha=1.0, s=80, c='black', edgecolor='black', linewidth=1)

            rho, pval = stats.spearmanr(focus_celltype_percent, focus_relative_marker_genes_expression)
                
            slope, intercept, r_value, p_value, std_err = stats.linregress(focus_celltype_percent, focus_relative_marker_genes_expression)
            x_pred = np.linspace(min(focus_celltype_percent), max(focus_celltype_percent), 100)
            y_pred = intercept + slope * x_pred
            ax.plot(x_pred, y_pred, 'r-', lw=5)

                    
            X = sm.add_constant(focus_celltype_percent)
            model = sm.OLS(focus_relative_marker_genes_expression, X).fit()
            predictions = model.get_prediction(sm.add_constant(x_pred))
            pred_df = predictions.summary_frame(alpha=0.05)
            ax.fill_between(x_pred, pred_df['mean_ci_lower'], pred_df['mean_ci_upper'], 
                            color='gray', alpha=0.3, label='95% CI')

                
            # Store data for summary
            all_rhos.append(round(rho, 4))
            all_pvals.append(round(pval, 4))
            all_number_of_spots.append(len(focus_celltype_percent))
            all_region_ids.append(Region_ID)
            all_min_proportion.append(round(min(focus_celltype_percent), 2))
            all_max_proportion.append(round(max(focus_celltype_percent), 2))
            all_min_expression.append(round(min(focus_relative_marker_genes_expression), 2))
            all_max_expression.append(round(max(focus_relative_marker_genes_expression), 2))

            # Add Spearman correlation annotation - only show when calculation is possible
            ax.text(0.05, 0.90, f"Spearman $\\rho$ = {rho:.2f}",
                    transform=ax.transAxes, fontsize=50, 
                    color='black', fontweight='bold')
        
        # Add background grid lines - thick dashed gray
        ax.grid(True, linestyle='--', linewidth=4, color='gray', alpha=0.7)
        
        # Remove axis titles
        ax.set_xlabel("")
        ax.set_ylabel("")
        
        # Adjust tick style - increase size, darken, and bold
        ax.tick_params(axis='both', which='major', labelsize=40, width=3, length=8, colors='black')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_color('black')
        
        # Set left, bottom, and right borders as thick black lines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['right'].set_linewidth(5)
        ax.spines['left'].set_linewidth(5)
        ax.spines['bottom'].set_linewidth(5)
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        

        
        # Adjust layout
        plt.tight_layout()
        
        # Save individual plots
        plt.savefig(f"FredHutch_Colorectal/CARD_Results_Figures/{cell_type}_{Region_ID}.png", dpi=600, bbox_inches='tight')
        plt.close()
    
    # Return summary data
    results_df = pd.DataFrame({
        'Region_ID': all_region_ids,
        'Sample_Size': all_number_of_spots,
        'Cell_Type': [cell_type] * len(all_region_ids),
        'Spearman_rho': all_rhos,
        'Min_DCTP (%)': all_min_proportion,
        'Max_DCTP (%)': all_max_proportion,
        'Min_RMGE (%)': all_min_expression,
        'Max_RMGE (%)': all_max_expression
    })
    
    return results_df


result_df_Cancer_FH = CTProportion_vs_Relative_expression_Visualization(cell_type="Cancer Cells")
result_df_Normal_Epith_FH = CTProportion_vs_Relative_expression_Visualization(cell_type="Normal Epithelial Cells")
result_df_T_FH = CTProportion_vs_Relative_expression_Visualization(cell_type="T Cells")
result_df_Other_Immune_FH = CTProportion_vs_Relative_expression_Visualization(cell_type="Other Immune Cells")
result_df_Stromal_FH = CTProportion_vs_Relative_expression_Visualization(cell_type="Stromal Cells")


result_df_FH = pd.concat([result_df_Cancer_FH, result_df_Normal_Epith_FH, result_df_T_FH, result_df_Other_Immune_FH, result_df_Stromal_FH], axis=0)
result_df_FH.reset_index(drop=True, inplace=True)
result_df_FH.to_csv("FredHutch_Colorectal/CARD_Results_Files/CARD_Results_Validation_FH.csv", index=False)















