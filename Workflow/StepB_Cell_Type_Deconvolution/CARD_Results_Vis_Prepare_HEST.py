"""
STPath-COAD: CARD Results Visualization Preparation for HEST Dataset
===================================================================

This script prepares CARD deconvolution results from HEST dataset for visualization
by formatting and organizing cell type proportion data.

Author: Saishi Cui
Date: Sept 2025

Purpose: Process and format CARD deconvolution results from HEST dataset for 
visualization, organize cell type proportion data, and prepare files for 
downstream visualization and analysis.
"""

import pandas as pd
import os






# define the path
path_CARD_results_celltype_proportion = "data/hest_data/CARD_Results/"

# get the path list of all .csv files
CARD_results_celltype_proportion_path_list = []
for root, dirs, files in os.walk(path_CARD_results_celltype_proportion):
    for file in files:
        if file.endswith("celltype_proportion_modified.csv"):  #
            CARD_results_celltype_proportion_path_list.append(os.path.join(root, file))

CARD_results_celltype_proportion_path_list = sorted(CARD_results_celltype_proportion_path_list)




# extract the sample name from the file path
region_id_list = []
for file_path in CARD_results_celltype_proportion_path_list:
    # extract the sample name from the file path
    file_name = os.path.basename(file_path).split("_")[0]
    
    region_id_list.append(file_name)

len(region_id_list)



for i, region_id in enumerate(region_id_list):
    if i == 0:
        input_file_name = os.path.join(path_CARD_results_celltype_proportion, f"{region_id}_celltype_proportion_modified.csv")
        Celltype_proportion_df = pd.read_csv(input_file_name, index_col=0)
        Celltype_proportion_df_mean = pd.DataFrame(Celltype_proportion_df.mean(axis=0))
        Celltype_proportion_df_mean = Celltype_proportion_df_mean*100
        Celltype_proportion_df_mean.columns = ["Proportion_" + region_id]
        Celltype_proportion_df_mean['Cell Type'] = Celltype_proportion_df_mean.index
        Celltype_proportion_df_mean.reset_index(drop=True, inplace=True)
        Celltype_proportion_df_mean = Celltype_proportion_df_mean.iloc[:,[1,0]]
    
    else:
        input_file_name = os.path.join(path_CARD_results_celltype_proportion, f"{region_id}_celltype_proportion_modified.csv")
        Celltype_proportion_df_augmented = pd.read_csv(input_file_name, index_col=0)
        Celltype_proportion_df_mean_augmented = pd.DataFrame(Celltype_proportion_df_augmented.mean(axis=0))
        Celltype_proportion_df_mean_augmented = Celltype_proportion_df_mean_augmented*100
        Celltype_proportion_df_mean_augmented.columns = ["Proportion_" + region_id]
        Celltype_proportion_df_mean_augmented['Cell Type'] = Celltype_proportion_df_mean_augmented.index
        Celltype_proportion_df_mean_augmented.reset_index(drop=True, inplace=True)
        Celltype_proportion_df_mean_augmented = Celltype_proportion_df_mean_augmented.iloc[:,[1,0]]
        Celltype_proportion_df_mean = pd.concat([Celltype_proportion_df_mean, Celltype_proportion_df_mean_augmented.iloc[:,1]], axis=1)



# Melt data and calculate mean and std

Celltype_proportion_df_mean_HEST = Celltype_proportion_df_mean.copy()
Celltype_proportion_df_mean_HEST.to_csv('data/hest_data/CARD_Results_Files/Celltype_proportion_df_mean_HEST.csv')
