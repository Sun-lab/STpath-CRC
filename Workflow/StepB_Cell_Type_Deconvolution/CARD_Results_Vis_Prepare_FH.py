import pandas as pd
import os




# define the path
path_CARD_results_celltype_proportion = "/Users/scui2/Desktop/FredHutch_Colorectal/CARD_Results_Regions"

# get the path list of all .csv files
CARD_results_celltype_proportion_path_list = []
for root, dirs, files in os.walk(path_CARD_results_celltype_proportion):
    for file in files:
        if file.endswith("celltype_proportion_modified.csv"):  #
            CARD_results_celltype_proportion_path_list.append(os.path.join(root, file))

CARD_results_celltype_proportion_path_list = sorted(CARD_results_celltype_proportion_path_list)




# extract the sample name from the file path
Region_ID_set = set()
for file_path in CARD_results_celltype_proportion_path_list:
    # extract the sample name from the file path
    file_name = os.path.basename(file_path)
    file_name = file_name.split("_celltype_proportion_modified.csv")[0]
    Region_ID_set.add(file_name)


Region_ID_list = sorted(list(Region_ID_set))
len(Region_ID_list)


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



# Melt data and calculate mean and std

Celltype_proportion_df_mean_FH = Celltype_proportion_df_mean.copy()
Celltype_proportion_df_mean_FH.to_csv('/Users/scui2/Desktop/FredHutch_Colorectal/CARD_Results_Files/Celltype_proportion_df_mean_FH.csv')

