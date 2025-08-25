"""
STPath-COAD: CARD Results Visualization Preparation for Cody's Data
==================================================

This script prepares CARD deconvolution results for visualization by formatting
and organizing cell type proportion data.

Author: Saishi Cui
Date: Sept 2025

Purpose: Process and format CARD deconvolution results for visualization,
organize cell type proportion data, and prepare files for downstream 
visualization and analysis in the paper.
"""

import pandas as pd
import os






# define the path
path_CARD_results_celltype_proportion = "data/CARD_Results_Regions"

# get the path list of all .csv files
CARD_results_celltype_proportion_path_list = []
for root, dirs, files in os.walk(path_CARD_results_celltype_proportion):
    for file in files:
        if file.endswith("celltype_proportion_modified.csv"):  #
            CARD_results_celltype_proportion_path_list.append(os.path.join(root, file))

CARD_results_celltype_proportion_path_list = sorted(CARD_results_celltype_proportion_path_list)




# extract the sample name from the file path
Region_ID_set = set()
Sample_ID_set = set()
Individual_ID_set = set()
file_path = CARD_results_celltype_proportion_path_list[0]
for file_path in CARD_results_celltype_proportion_path_list:
    # extract the sample name from the file path
    file_name = os.path.basename(file_path)
    file_name = file_name.split("_celltype_proportion_modified.csv")[0]
    Region_ID_set.add(file_name)
    
    Sample_ID = "_".join(file_name.split("_")[:3])
    Sample_ID_set.add(Sample_ID)

    Individual_ID = Sample_ID.split("_")[0]
    Individual_ID_set.add(Individual_ID)

Region_ID_list = sorted(list(Region_ID_set))
Sample_ID_list = sorted(list(Sample_ID_set))
Individual_ID_list = sorted(list(Individual_ID_set))

len(Region_ID_list)
len(Sample_ID_list)
len(Individual_ID_list)


for i, region_id in enumerate(Region_ID_list):
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




Celltype_proportion_df_mean_Cody = Celltype_proportion_df_mean.copy()
Celltype_proportion_df_mean_Cody.to_csv('data/CARD_Results_Files/Celltype_proportion_df_mean_Cody.csv')

