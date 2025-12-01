"""
STPath-COAD: Combined CARD Results Validation Analysis
=====================================================

This script performs comprehensive validation of CARD deconvolution results
across multiple datasets and conditions with statistical analysis.

Author: Saishi Cui
Date: December 2025

Purpose: Validate and compare CARD deconvolution results across different
datasets, perform statistical testing, and generate comprehensive validation
reports for the STPath-COAD pipeline.
"""

import pandas as pd
import os
import shutil
from scipy import stats
import numpy as np
import json




### Marker genes refinement
with open('/Users/scui2/Desktop/scRNAseq_data/final_marker_genes_dict.json', 'r') as f:
    final_marker_genes_dict = json.load(f)
final_marker_genes_dict.keys()
del final_marker_genes_dict["Cancer"]
del final_marker_genes_dict["Normal Epithelia"]

# Rename to COAD_Marker_Genes_Dict and save
COAD_Marker_Genes_Dict = final_marker_genes_dict
COAD_Marker_Genes_Dict.keys()
# Save the renamed dictionary
with open('/Users/scui2/Desktop/scRNAseq_data/COAD_Marker_Genes_Dict.json', 'w') as f:
    json.dump(COAD_Marker_Genes_Dict, f, indent=2)



InputDf_for_CARD_SelectedGenes = pd.read_csv('/Users/scui2/Desktop/scRNAseq_data/InputDf_for_CARD_SelectedGenes.csv', index_col=0)
InputDf_for_CARD_meta = pd.read_csv('/Users/scui2/Desktop/scRNAseq_data/InputDf_for_CARD_meta.csv', index_col=0)
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

final_ranked_gene_dict.keys()



### Filter high quality tiles for training

result_df_Cody = pd.read_csv("data/CARD_Results_Files/CARD_Results_Validation_Cody.csv", index_col=0)
result_df_FH = pd.read_csv("FredHutch_Colorectal/CARD_Results_Files/CARD_Results_Validation_FH.csv", index_col=0)
result_df_HEST = pd.read_csv("data/hest_data/CARD_Results_Files/CARD_Results_Validation_HEST.csv", index_col=0)


result_df_combined = pd.concat([result_df_Cody, result_df_FH, result_df_HEST], axis=0)
result_df_combined.to_csv("data/Colorectal_Cancer_HE_patches/Supplementary_File2_CARD_Results_Validation.csv")


def calculate_weighted_average(df, cell_type):
    """Calculate weighted average, handle NaN values"""
    subset = df[df["Cell_Type"] == cell_type]
    if len(subset) == 0:
        return np.nan
    
    # Remove rows containing NaN values
    subset_clean = subset.dropna(subset=["Spearman_rho", "Sample_Size"])
    if len(subset_clean) == 0:
        return np.nan
    
    # Calculate weighted average
    weights = subset_clean["Sample_Size"]
    values = subset_clean["Spearman_rho"]
    
    if weights.sum() == 0:
        return np.nan
    
    weighted_avg = np.sum(values * weights) / weights.sum()
    return weighted_avg

weighted_cancer_rho = calculate_weighted_average(result_df_combined, "Cancer Cells")
weighted_stromal_rho = calculate_weighted_average(result_df_combined, "Stromal Cells")
weighted_normal_epithelial_rho = calculate_weighted_average(result_df_combined, "Normal Epithelial Cells")
weighted_t_rho = calculate_weighted_average(result_df_combined, "T Cells")
weighted_other_immune_rho = calculate_weighted_average(result_df_combined, "Other Immune Cells")

weights_list = [1.1*weighted_cancer_rho, 1.1*weighted_stromal_rho, 1.1*weighted_normal_epithelial_rho, 1.1*weighted_t_rho, 1.1*weighted_other_immune_rho]



def filter_high_quality_tiles_for_training_Cody(cell_type,  marker_genes_dict=final_ranked_gene_dict,
                                          rho_threshold=0.7, target_folder = "data/Colorectal_Cancer_HE_patches/Training_tiles/Cancer"):
    
    path_CARD_results_celltype_proportion = "data/CARD_Results_Regions/"

    # get the path list of all .csv files
    CARD_results_celltype_proportion_path_list = []
    for root, dirs, files in os.walk(path_CARD_results_celltype_proportion):
        for file in files:
            if file.endswith("_celltype_proportion_modified.csv"):
                CARD_results_celltype_proportion_path_list.append(os.path.join(root, file))

    CARD_results_celltype_proportion_path_list = sorted(CARD_results_celltype_proportion_path_list)


    # extract the sample name from the file path
    Region_ID_set = set()

    file_path = CARD_results_celltype_proportion_path_list[0]
    for file_path in CARD_results_celltype_proportion_path_list:
        # extract the sample name from the file path
        file_name = os.path.basename(file_path)
        file_name = file_name.split("_celltype_proportion_modified.csv")[0]
        Region_ID_set.add(file_name)

    Region_ID_list = sorted(list(Region_ID_set))


    os.makedirs(target_folder, exist_ok=True)
    Training_celltype_proportion_df = pd.DataFrame()
    for i in range(0, len(Region_ID_list)):

            
        Region_ID = Region_ID_list[i]
        print(f"Processing region: {Region_ID}")

        # Load data
        expression_df = pd.read_csv(f'data/CARD_Need_Files/{Region_ID}_expression.csv', index_col=0)
        Celltype_proportion_df = pd.read_csv(f'data/CARD_Results_Regions/{Region_ID}_celltype_proportion_modified.csv', index_col=0)
        B_matrix_df = pd.read_csv(f'data/CARD_Results_Regions/{Region_ID}_B_Matrix_modified.csv', index_col=0)
        
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
  

        rho, pval = stats.spearmanr(focus_celltype_percent, focus_relative_marker_genes_expression)

        
        if rho >= rho_threshold:
        
            source_folder = f"data/Colorectal_Cancer_HE_patches/{Region_ID}"
            high_quality_file_names = os.listdir(source_folder)

            high_quality_tile_IDs = [file_name.split(".")[0] for file_name in high_quality_file_names if file_name.endswith(".tif")]
            high_quality_tile_IDs = [file_name.split("_")[0] for file_name in high_quality_tile_IDs]
            high_quality_tile_IDs_intersect = list(set(high_quality_tile_IDs).intersection(set(Celltype_proportion_df_grouped.index.tolist())))
            Celltype_proportion_df_grouped_high_quality = Celltype_proportion_df_grouped.loc[high_quality_tile_IDs_intersect]
            high_quality_tile_IDs_intersect_append_region_ID = [tile_id + "_" + Region_ID for tile_id in high_quality_tile_IDs_intersect]
            Celltype_proportion_df_grouped_high_quality.index = high_quality_tile_IDs_intersect_append_region_ID

            Training_celltype_proportion_df = pd.concat([Training_celltype_proportion_df, Celltype_proportion_df_grouped_high_quality], axis=0)
            
            for file_name in high_quality_file_names:
                
                source_file = os.path.join(source_folder, file_name)
                
            
                target_file = os.path.join(target_folder, file_name)
                
                
                if os.path.exists(source_file):
                    shutil.copy2(source_file, target_file)
                    print(f"Copied and renamed: {source_file} -> {target_file}")
                else:
                    print(f"Source file not found: {source_file}")
        
            
            
    Training_celltype_proportion_df.to_csv(f"data/Colorectal_Cancer_HE_patches/Training_celltype_proportion/{cell_type}_celltype_proportion_Cody.csv")
    print(f"{cell_type} done")
    return None


def filter_high_quality_tiles_for_training_FH(cell_type,  marker_genes_dict=final_ranked_gene_dict, rho_threshold = 0.275, target_folder = "FredHutch_Colorectal/CARD_Results_Files/Training_tiles"):
    
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
    for file_path in CARD_results_celltype_proportion_path_list:
        # extract the sample name from the file path
        region_id = "_".join(os.path.basename(file_path).split("_")[0:2])
        Region_ID_list.append(region_id)

    Region_ID_list = sorted(list(set(Region_ID_list)))



    os.makedirs(target_folder, exist_ok=True)
    Training_celltype_proportion_df = pd.DataFrame()
    for i in range(0, len(Region_ID_list)):

            
        Region_ID = Region_ID_list[i]
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
 

        rho, pval = stats.spearmanr(focus_celltype_percent, focus_relative_marker_genes_expression)

        
        if rho >= rho_threshold:
        
            source_folder = f"FredHutch_Colorectal/HE_patches/{Region_ID}"
            high_quality_file_names = os.listdir(source_folder)

            high_quality_tile_IDs = [file_name.split(".")[0] for file_name in high_quality_file_names if file_name.endswith(".tif")]
            high_quality_tile_IDs = [file_name.split("_")[0] for file_name in high_quality_tile_IDs]
            high_quality_tile_IDs_intersect = list(set(high_quality_tile_IDs).intersection(set(Celltype_proportion_df_grouped.index.tolist())))
            Celltype_proportion_df_grouped_high_quality = Celltype_proportion_df_grouped.loc[high_quality_tile_IDs_intersect]
            high_quality_tile_IDs_intersect_append_region_ID = [tile_id + "_" + Region_ID for tile_id in high_quality_tile_IDs_intersect]
            Celltype_proportion_df_grouped_high_quality.index = high_quality_tile_IDs_intersect_append_region_ID

            Training_celltype_proportion_df = pd.concat([Training_celltype_proportion_df, Celltype_proportion_df_grouped_high_quality], axis=0)
            
            for file_name in high_quality_file_names:
                
                source_file = os.path.join(source_folder, file_name)
                
            
                target_file = os.path.join(target_folder, file_name)
                
                
                if os.path.exists(source_file):
                    shutil.copy2(source_file, target_file)
                    print(f"Copied and renamed: {source_file} -> {target_file}")
                else:
                    print(f"Source file not found: {source_file}")
        
            
            
    Training_celltype_proportion_df.to_csv(f"data/Colorectal_Cancer_HE_patches/Training_celltype_proportion/{cell_type}_celltype_proportion_FH.csv")
    print(f"{cell_type} done")
    return None


def filter_high_quality_tiles_for_training_HEST(cell_type,  marker_genes_dict=final_ranked_gene_dict,
                                          rho_threshold= 0.7, target_folder = "data/Colorectal_Cancer_HE_patches/Training_tiles/Cancer"):
    

    #################
    # define the path
    path_CARD_results_celltype_proportion = "data/hest_data/CARD_Results/"

    # get the path list of all .csv files
    CARD_results_celltype_proportion_path_list = []
    for root, dirs, files in os.walk(path_CARD_results_celltype_proportion):
        for file in files:
            if file.endswith("_celltype_proportion_modified.csv"):
                CARD_results_celltype_proportion_path_list.append(os.path.join(root, file))

    CARD_results_celltype_proportion_path_list = sorted(CARD_results_celltype_proportion_path_list)




    # extract the sample name from the file path
    Region_ID_list = []
    for file_path in CARD_results_celltype_proportion_path_list:
        # extract the sample name from the file path
        region_id = os.path.basename(file_path).split("_")[0]
        Region_ID_list.append(region_id)

    Region_ID_list = sorted(list(set(Region_ID_list)))



    os.makedirs(target_folder, exist_ok=True)
    Training_celltype_proportion_df = pd.DataFrame()
    for i in range(0, len(Region_ID_list)):

            
        Region_ID = Region_ID_list[i]
        print(f"Processing region: {Region_ID}")

        # Load data
        expression_df = pd.read_csv(f'data/hest_data/CARD_Need_Files/{Region_ID}_expression.csv', index_col=0)
        Celltype_proportion_df = pd.read_csv(f'data/hest_data/CARD_Results/{Region_ID}_celltype_proportion_modified.csv', index_col=0)
        B_matrix_df = pd.read_csv(f'data/hest_data/CARD_Results/{Region_ID}_B_Matrix_modified.csv', index_col=0)
        
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


        rho, pval = stats.spearmanr(focus_celltype_percent, focus_relative_marker_genes_expression)

        
        if rho >= rho_threshold:
        
            source_folder = f"data/hest_data/image_patches/{Region_ID}"
            high_quality_file_names = os.listdir(source_folder)

            high_quality_tile_IDs = [file_name.split(".")[0] for file_name in high_quality_file_names if file_name.endswith(".tif")]
            high_quality_tile_IDs_intersect = list(set(high_quality_tile_IDs).intersection(set(Celltype_proportion_df_grouped.index.tolist())))
            Celltype_proportion_df_grouped_high_quality = Celltype_proportion_df_grouped.loc[high_quality_tile_IDs_intersect]
            high_quality_tile_IDs_intersect_append_region_ID = [tile_id + "_" + Region_ID for tile_id in high_quality_tile_IDs_intersect]
            Celltype_proportion_df_grouped_high_quality.index = high_quality_tile_IDs_intersect_append_region_ID

            Training_celltype_proportion_df = pd.concat([Training_celltype_proportion_df, Celltype_proportion_df_grouped_high_quality], axis=0)
            
            for file_name in high_quality_file_names:
                
                source_file = os.path.join(source_folder, file_name)
                target_file = os.path.join(target_folder, file_name.split(".")[0] + "_" + Region_ID + ".tif")
                
                
                if os.path.exists(source_file):
                    shutil.copy2(source_file, target_file)
                    print(f"Copied and renamed: {source_file} -> {target_file}")
                else:
                    print(f"Source file not found: {source_file}")
        
            
            
    Training_celltype_proportion_df.to_csv(f"data/Colorectal_Cancer_HE_patches/Training_celltype_proportion/{cell_type}_celltype_proportion_HEST.csv")
    print(f"{cell_type} done")
    return None









for i, cell_type in enumerate(["Cancer Cells", "Stromal Cells", "Normal Epithelial Cells", "T Cells", "Other Immune Cells"]): 
    filter_high_quality_tiles_for_training_Cody(cell_type = cell_type, marker_genes_dict = final_ranked_gene_dict, rho_threshold = weights_list[i], target_folder = f"data/Colorectal_Cancer_HE_patches/Training_tiles/{cell_type}")
    filter_high_quality_tiles_for_training_FH(cell_type = cell_type, marker_genes_dict = final_ranked_gene_dict, rho_threshold = weights_list[i], target_folder = f"data/Colorectal_Cancer_HE_patches/Training_tiles/{cell_type}")
    filter_high_quality_tiles_for_training_HEST(cell_type = cell_type, marker_genes_dict = final_ranked_gene_dict, rho_threshold = weights_list[i], target_folder = f"data/Colorectal_Cancer_HE_patches/Training_tiles/{cell_type}")






Cancer_Cody = pd.read_csv("data/Colorectal_Cancer_HE_patches/Training_celltype_proportion/Cancer Cells_celltype_proportion_Cody.csv", index_col=0)
Cancer_FH = pd.read_csv("data/Colorectal_Cancer_HE_patches/Training_celltype_proportion/Cancer Cells_celltype_proportion_FH.csv", index_col=0)
Cancer_HEST = pd.read_csv("data/Colorectal_Cancer_HE_patches/Training_celltype_proportion/Cancer Cells_celltype_proportion_HEST.csv", index_col=0)

Stromal_Cody = pd.read_csv("data/Colorectal_Cancer_HE_patches/Training_celltype_proportion/Stromal Cells_celltype_proportion_Cody.csv", index_col=0)
Stromal_FH = pd.read_csv("data/Colorectal_Cancer_HE_patches/Training_celltype_proportion/Stromal Cells_celltype_proportion_FH.csv", index_col=0)
Stromal_HEST = pd.read_csv("data/Colorectal_Cancer_HE_patches/Training_celltype_proportion/Stromal Cells_celltype_proportion_HEST.csv", index_col=0)

Normal_Epith_Cody = pd.read_csv("data/Colorectal_Cancer_HE_patches/Training_celltype_proportion/Normal Epithelial Cells_celltype_proportion_Cody.csv", index_col=0)
Normal_Epith_FH = pd.read_csv("data/Colorectal_Cancer_HE_patches/Training_celltype_proportion/Normal Epithelial Cells_celltype_proportion_FH.csv", index_col=0)
Normal_Epith_HEST = pd.read_csv("data/Colorectal_Cancer_HE_patches/Training_celltype_proportion/Normal Epithelial Cells_celltype_proportion_HEST.csv", index_col=0)

T_Cody = pd.read_csv("data/Colorectal_Cancer_HE_patches/Training_celltype_proportion/T Cells_celltype_proportion_Cody.csv", index_col=0)
T_FH = pd.read_csv("data/Colorectal_Cancer_HE_patches/Training_celltype_proportion/T Cells_celltype_proportion_FH.csv", index_col=0)
T_HEST = pd.read_csv("data/Colorectal_Cancer_HE_patches/Training_celltype_proportion/T Cells_celltype_proportion_HEST.csv", index_col=0)

Other_Immune_Cody = pd.read_csv("data/Colorectal_Cancer_HE_patches/Training_celltype_proportion/Other Immune Cells_celltype_proportion_Cody.csv", index_col=0)
Other_Immune_FH = pd.read_csv("data/Colorectal_Cancer_HE_patches/Training_celltype_proportion/Other Immune Cells_celltype_proportion_FH.csv", index_col=0)
Other_Immune_HEST = pd.read_csv("data/Colorectal_Cancer_HE_patches/Training_celltype_proportion/Other Immune Cells_celltype_proportion_HEST.csv", index_col=0)

Cancer_Combined = pd.concat([Cancer_Cody, Cancer_FH, Cancer_HEST], axis=0)
Stromal_Combined = pd.concat([Stromal_Cody, Stromal_FH, Stromal_HEST], axis=0)
Normal_Epith_Combined = pd.concat([Normal_Epith_Cody, Normal_Epith_FH, Normal_Epith_HEST], axis=0)
T_Combined = pd.concat([T_Cody, T_FH, T_HEST], axis=0)
Other_Immune_Combined = pd.concat([Other_Immune_Cody, Other_Immune_FH, Other_Immune_HEST], axis=0)





Cancer_Combined.to_csv("data/Colorectal_Cancer_HE_patches/Training_celltype_proportion/Cancer Cells_celltype_proportion_Combined.csv")
Stromal_Combined.to_csv("data/Colorectal_Cancer_HE_patches/Training_celltype_proportion/Stromal Cells_celltype_proportion_Combined.csv")
Normal_Epith_Combined.to_csv("data/Colorectal_Cancer_HE_patches/Training_celltype_proportion/Normal Epithelial Cells_celltype_proportion_Combined.csv")
T_Combined.to_csv("data/Colorectal_Cancer_HE_patches/Training_celltype_proportion/T Cells_celltype_proportion_Combined.csv")
Other_Immune_Combined.to_csv("data/Colorectal_Cancer_HE_patches/Training_celltype_proportion/Other Immune Cells_celltype_proportion_Combined.csv")