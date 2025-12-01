"""
STPath-COAD: Important Features Extraction for Supplementary File S3
===================================================================

This script extracts and processes important features from trained XGBoost models
to create supplementary tables showing feature importance across different models.

Author: Saishi Cui
Date: December 2025

Purpose: Extract feature importance scores from trained XGBoost models for all
cell types and foundation models, create comprehensive feature importance tables
for Supplementary File S3 of the paper.
"""

import torch
import pandas as pd
import pickle
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows


### Cancer Cells

Conch_Cancer_Cells_data = torch.load(f = "xgboost_prediction/Cancer Cells_Conch_individual_level_ratio100/xgboost_results_individual_level.pt")
ProvGigapath_Cancer_Cells_data = torch.load(f = "xgboost_prediction/Cancer Cells_ProvGigapath_individual_level_ratio100/xgboost_results_individual_level.pt")
Virchow2_Cancer_Cells_data = torch.load(f = "xgboost_prediction/Cancer Cells_Virchow2_individual_level_ratio100/xgboost_results_individual_level.pt")
Virchow_Cancer_Cells_data = torch.load(f = "xgboost_prediction/Cancer Cells_Virchow_individual_level_ratio100/xgboost_results_individual_level.pt")
UNI2h_Cancer_Cells_data = torch.load(f = "xgboost_prediction/Cancer Cells_UNI2h_individual_level_ratio100/xgboost_results_individual_level.pt")


Important_Features_Cancer_Cells = {}

Conch_FI_Cancer_Cells_DF = pd.DataFrame.from_dict(Conch_Cancer_Cells_data["feature_importances"][0], orient='index')
for i in range(1, len(Conch_Cancer_Cells_data["feature_importances"])):
    Conch_FI_Cancer_Cells_DF = pd.concat([Conch_FI_Cancer_Cells_DF, pd.DataFrame.from_dict(Conch_Cancer_Cells_data["feature_importances"][i], orient='index')], axis=1)

Conch_FI_Cancer_Cells_DF.fillna(0, inplace=True)
Conch_FI_Cancer_Cells_DF_mean = Conch_FI_Cancer_Cells_DF.mean(axis=1)
Conch_FI_Cancer_Cells_DF_mean_sorted = Conch_FI_Cancer_Cells_DF_mean.sort_values(ascending=False)
Conch_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage = Conch_FI_Cancer_Cells_DF_mean_sorted.cumsum()/Conch_FI_Cancer_Cells_DF_mean_sorted.sum()
Conch_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(Conch_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_Cancer_Cells["Conch"] = list(Conch_FI_Cancer_Cells_DF_mean_sorted.index[0:round(len(Conch_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_Cancer_Cells["Conch_percentage"] = round(Conch_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(Conch_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)


ProvGigapath_FI_Cancer_Cells_DF = pd.DataFrame.from_dict(ProvGigapath_Cancer_Cells_data["feature_importances"][0], orient='index')
for i in range(1, len(ProvGigapath_Cancer_Cells_data["feature_importances"])):
    ProvGigapath_FI_Cancer_Cells_DF = pd.concat([ProvGigapath_FI_Cancer_Cells_DF, pd.DataFrame.from_dict(ProvGigapath_Cancer_Cells_data["feature_importances"][i], orient='index')], axis=1)
ProvGigapath_FI_Cancer_Cells_DF.fillna(0, inplace=True)
ProvGigapath_FI_Cancer_Cells_DF_mean = ProvGigapath_FI_Cancer_Cells_DF.mean(axis=1)
ProvGigapath_FI_Cancer_Cells_DF_mean_sorted = ProvGigapath_FI_Cancer_Cells_DF_mean.sort_values(ascending=False)
ProvGigapath_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage = ProvGigapath_FI_Cancer_Cells_DF_mean_sorted.cumsum()/ProvGigapath_FI_Cancer_Cells_DF_mean_sorted.sum()
ProvGigapath_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(ProvGigapath_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_Cancer_Cells["ProvGigapath"] = list(ProvGigapath_FI_Cancer_Cells_DF_mean_sorted.index[0:round(len(ProvGigapath_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_Cancer_Cells["ProvGigapath_percentage"] = round(ProvGigapath_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(ProvGigapath_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)


Virchow_FI_Cancer_Cells_DF = pd.DataFrame.from_dict(Virchow_Cancer_Cells_data["feature_importances"][0], orient='index')
for i in range(1, len(Virchow_Cancer_Cells_data["feature_importances"])):
    Virchow_FI_Cancer_Cells_DF = pd.concat([Virchow_FI_Cancer_Cells_DF, pd.DataFrame.from_dict(Virchow_Cancer_Cells_data["feature_importances"][i], orient='index')], axis=1)
Virchow_FI_Cancer_Cells_DF.fillna(0, inplace=True)
Virchow_FI_Cancer_Cells_DF_mean = Virchow_FI_Cancer_Cells_DF.mean(axis=1)
Virchow_FI_Cancer_Cells_DF_mean_sorted = Virchow_FI_Cancer_Cells_DF_mean.sort_values(ascending=False)
Virchow_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage = Virchow_FI_Cancer_Cells_DF_mean_sorted.cumsum()/Virchow_FI_Cancer_Cells_DF_mean_sorted.sum()
Virchow_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(Virchow_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_Cancer_Cells["Virchow"] = list(Virchow_FI_Cancer_Cells_DF_mean_sorted.index[0:round(len(Virchow_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_Cancer_Cells["Virchow_percentage"] = round(Virchow_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(Virchow_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)


Virchow2_FI_Cancer_Cells_DF = pd.DataFrame.from_dict(Virchow2_Cancer_Cells_data["feature_importances"][0], orient='index')
for i in range(1, len(Virchow2_Cancer_Cells_data["feature_importances"])):
    Virchow2_FI_Cancer_Cells_DF = pd.concat([Virchow2_FI_Cancer_Cells_DF, pd.DataFrame.from_dict(Virchow2_Cancer_Cells_data["feature_importances"][i], orient='index')], axis=1)
Virchow2_FI_Cancer_Cells_DF.fillna(0, inplace=True)
Virchow2_FI_Cancer_Cells_DF_mean = Virchow2_FI_Cancer_Cells_DF.mean(axis=1)
Virchow2_FI_Cancer_Cells_DF_mean_sorted = Virchow2_FI_Cancer_Cells_DF_mean.sort_values(ascending=False)
Virchow2_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage = Virchow2_FI_Cancer_Cells_DF_mean_sorted.cumsum()/Virchow2_FI_Cancer_Cells_DF_mean_sorted.sum()
Virchow2_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(Virchow2_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_Cancer_Cells["Virchow2"] = list(Virchow2_FI_Cancer_Cells_DF_mean_sorted.index[0:round(len(Virchow2_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_Cancer_Cells["Virchow2_percentage"] = round(Virchow2_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(Virchow2_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)


UNI2h_FI_Cancer_Cells_DF = pd.DataFrame.from_dict(UNI2h_Cancer_Cells_data["feature_importances"][0], orient='index')
for i in range(1, len(UNI2h_Cancer_Cells_data["feature_importances"])):
    UNI2h_FI_Cancer_Cells_DF = pd.concat([UNI2h_FI_Cancer_Cells_DF, pd.DataFrame.from_dict(UNI2h_Cancer_Cells_data["feature_importances"][i], orient='index')], axis=1)
UNI2h_FI_Cancer_Cells_DF.fillna(0, inplace=True)
UNI2h_FI_Cancer_Cells_DF_mean = UNI2h_FI_Cancer_Cells_DF.mean(axis=1)
UNI2h_FI_Cancer_Cells_DF_mean_sorted = UNI2h_FI_Cancer_Cells_DF_mean.sort_values(ascending=False)
UNI2h_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage = UNI2h_FI_Cancer_Cells_DF_mean_sorted.cumsum()/UNI2h_FI_Cancer_Cells_DF_mean_sorted.sum()
UNI2h_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(UNI2h_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_Cancer_Cells["UNI2h"] = list(UNI2h_FI_Cancer_Cells_DF_mean_sorted.index[0:round(len(UNI2h_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_Cancer_Cells["UNI2h_percentage"] = round(UNI2h_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(UNI2h_FI_Cancer_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)


print(Important_Features_Cancer_Cells["UNI2h_percentage"])
print(Important_Features_Cancer_Cells["Virchow2_percentage"])
print(Important_Features_Cancer_Cells["Virchow_percentage"])
print(Important_Features_Cancer_Cells["ProvGigapath_percentage"])
print(Important_Features_Cancer_Cells["Conch_percentage"])

print(len(Important_Features_Cancer_Cells["UNI2h"]))
print(len(Important_Features_Cancer_Cells["Virchow2"]))
print(len(Important_Features_Cancer_Cells["Virchow"]))
print(len(Important_Features_Cancer_Cells["ProvGigapath"]))
print(len(Important_Features_Cancer_Cells["Conch"]))

pickle.dump(Important_Features_Cancer_Cells, open("xgboost_prediction/important_features_Cancer Cells.pkl", "wb"))



### Stromal Cells

Conch_Stromal_Cells_data = torch.load(f = "xgboost_prediction/Stromal Cells_Conch_individual_level_ratio100/xgboost_results_individual_level.pt")
ProvGigapath_Stromal_Cells_data = torch.load(f = "xgboost_prediction/Stromal Cells_ProvGigapath_individual_level_ratio100/xgboost_results_individual_level.pt")
Virchow2_Stromal_Cells_data = torch.load(f = "xgboost_prediction/Stromal Cells_Virchow2_individual_level_ratio100/xgboost_results_individual_level.pt")
Virchow_Stromal_Cells_data = torch.load(f = "xgboost_prediction/Stromal Cells_Virchow_individual_level_ratio100/xgboost_results_individual_level.pt")
UNI2h_Stromal_Cells_data = torch.load(f = "xgboost_prediction/Stromal Cells_UNI2h_individual_level_ratio100/xgboost_results_individual_level.pt")


Important_Features_Stromal_Cells = {}

Conch_FI_Stromal_Cells_DF = pd.DataFrame.from_dict(Conch_Stromal_Cells_data["feature_importances"][0], orient='index')
for i in range(1, len(Conch_Stromal_Cells_data["feature_importances"])):
    Conch_FI_Stromal_Cells_DF = pd.concat([Conch_FI_Stromal_Cells_DF, pd.DataFrame.from_dict(Conch_Stromal_Cells_data["feature_importances"][i], orient='index')], axis=1)
Conch_FI_Stromal_Cells_DF.fillna(0, inplace=True)
Conch_FI_Stromal_Cells_DF_mean = Conch_FI_Stromal_Cells_DF.mean(axis=1)
Conch_FI_Stromal_Cells_DF_mean_sorted = Conch_FI_Stromal_Cells_DF_mean.sort_values(ascending=False)
Conch_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage = Conch_FI_Stromal_Cells_DF_mean_sorted.cumsum()/Conch_FI_Stromal_Cells_DF_mean_sorted.sum()
Conch_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(Conch_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_Stromal_Cells["Conch"] = list(Conch_FI_Stromal_Cells_DF_mean_sorted.index[0:round(len(Conch_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_Stromal_Cells["Conch_percentage"] = round(Conch_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(Conch_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)


ProvGigapath_FI_Stromal_Cells_DF = pd.DataFrame.from_dict(ProvGigapath_Stromal_Cells_data["feature_importances"][0], orient='index')
for i in range(1, len(ProvGigapath_Stromal_Cells_data["feature_importances"])):
    ProvGigapath_FI_Stromal_Cells_DF = pd.concat([ProvGigapath_FI_Stromal_Cells_DF, pd.DataFrame.from_dict(ProvGigapath_Stromal_Cells_data["feature_importances"][i], orient='index')], axis=1)
ProvGigapath_FI_Stromal_Cells_DF.fillna(0, inplace=True)
ProvGigapath_FI_Stromal_Cells_DF_mean = ProvGigapath_FI_Stromal_Cells_DF.mean(axis=1)
ProvGigapath_FI_Stromal_Cells_DF_mean_sorted = ProvGigapath_FI_Stromal_Cells_DF_mean.sort_values(ascending=False)
ProvGigapath_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage = ProvGigapath_FI_Stromal_Cells_DF_mean_sorted.cumsum()/ProvGigapath_FI_Stromal_Cells_DF_mean_sorted.sum()
ProvGigapath_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(ProvGigapath_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_Stromal_Cells["ProvGigapath"] = list(ProvGigapath_FI_Stromal_Cells_DF_mean_sorted.index[0:round(len(ProvGigapath_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_Stromal_Cells["ProvGigapath_percentage"] = round(ProvGigapath_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(ProvGigapath_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)   


Virchow_FI_Stromal_Cells_DF = pd.DataFrame.from_dict(Virchow_Stromal_Cells_data["feature_importances"][0], orient='index')
for i in range(1, len(Virchow_Stromal_Cells_data["feature_importances"])):
    Virchow_FI_Stromal_Cells_DF = pd.concat([Virchow_FI_Stromal_Cells_DF, pd.DataFrame.from_dict(Virchow_Stromal_Cells_data["feature_importances"][i], orient='index')], axis=1)
Virchow_FI_Stromal_Cells_DF.fillna(0, inplace=True)
Virchow_FI_Stromal_Cells_DF_mean = Virchow_FI_Stromal_Cells_DF.mean(axis=1)
Virchow_FI_Stromal_Cells_DF_mean_sorted = Virchow_FI_Stromal_Cells_DF_mean.sort_values(ascending=False)
Virchow_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage = Virchow_FI_Stromal_Cells_DF_mean_sorted.cumsum()/Virchow_FI_Stromal_Cells_DF_mean_sorted.sum()
Virchow_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(Virchow_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_Stromal_Cells["Virchow"] = list(Virchow_FI_Stromal_Cells_DF_mean_sorted.index[0:round(len(Virchow_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_Stromal_Cells["Virchow_percentage"] = round(Virchow_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(Virchow_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)


Virchow2_FI_Stromal_Cells_DF = pd.DataFrame.from_dict(Virchow2_Stromal_Cells_data["feature_importances"][0], orient='index')
for i in range(1, len(Virchow2_Stromal_Cells_data["feature_importances"])): 
    Virchow2_FI_Stromal_Cells_DF = pd.concat([Virchow2_FI_Stromal_Cells_DF, pd.DataFrame.from_dict(Virchow2_Stromal_Cells_data["feature_importances"][i], orient='index')], axis=1)
Virchow2_FI_Stromal_Cells_DF.fillna(0, inplace=True)
Virchow2_FI_Stromal_Cells_DF_mean = Virchow2_FI_Stromal_Cells_DF.mean(axis=1)
Virchow2_FI_Stromal_Cells_DF_mean_sorted = Virchow2_FI_Stromal_Cells_DF_mean.sort_values(ascending=False)
Virchow2_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage = Virchow2_FI_Stromal_Cells_DF_mean_sorted.cumsum()/Virchow2_FI_Stromal_Cells_DF_mean_sorted.sum()
Virchow2_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(Virchow2_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_Stromal_Cells["Virchow2"] = list(Virchow2_FI_Stromal_Cells_DF_mean_sorted.index[0:round(len(Virchow2_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_Stromal_Cells["Virchow2_percentage"] = round(Virchow2_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(Virchow2_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)

UNI2h_FI_Stromal_Cells_DF = pd.DataFrame.from_dict(UNI2h_Stromal_Cells_data["feature_importances"][0], orient='index')
for i in range(1, len(UNI2h_Stromal_Cells_data["feature_importances"])):
    UNI2h_FI_Stromal_Cells_DF = pd.concat([UNI2h_FI_Stromal_Cells_DF, pd.DataFrame.from_dict(UNI2h_Stromal_Cells_data["feature_importances"][i], orient='index')], axis=1)
UNI2h_FI_Stromal_Cells_DF.fillna(0, inplace=True)
UNI2h_FI_Stromal_Cells_DF_mean = UNI2h_FI_Stromal_Cells_DF.mean(axis=1)
UNI2h_FI_Stromal_Cells_DF_mean_sorted = UNI2h_FI_Stromal_Cells_DF_mean.sort_values(ascending=False)
UNI2h_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage = UNI2h_FI_Stromal_Cells_DF_mean_sorted.cumsum()/UNI2h_FI_Stromal_Cells_DF_mean_sorted.sum()
UNI2h_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(UNI2h_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_Stromal_Cells["UNI2h"] = list(UNI2h_FI_Stromal_Cells_DF_mean_sorted.index[0:round(len(UNI2h_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_Stromal_Cells["UNI2h_percentage"] = round(UNI2h_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(UNI2h_FI_Stromal_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)

print(Important_Features_Stromal_Cells["UNI2h_percentage"])
print(Important_Features_Stromal_Cells["Virchow2_percentage"])
print(Important_Features_Stromal_Cells["Virchow_percentage"])
print(Important_Features_Stromal_Cells["ProvGigapath_percentage"])
print(Important_Features_Stromal_Cells["Conch_percentage"])


print(len(Important_Features_Stromal_Cells["UNI2h"]))
print(len(Important_Features_Stromal_Cells["Virchow2"]))
print(len(Important_Features_Stromal_Cells["Virchow"]))
print(len(Important_Features_Stromal_Cells["ProvGigapath"]))
print(len(Important_Features_Stromal_Cells["Conch"]))

pickle.dump(Important_Features_Stromal_Cells, open("xgboost_prediction/important_features_Stromal Cells.pkl", "wb"))



### Normal Epithelial Cells

Conch_Normal_Epithelial_Cells_data = torch.load(f = "xgboost_prediction/Normal Epithelial Cells_Conch_individual_level_ratio100/xgboost_results_individual_level.pt")
ProvGigapath_Normal_Epithelial_Cells_data = torch.load(f = "xgboost_prediction/Normal Epithelial Cells_ProvGigapath_individual_level_ratio100/xgboost_results_individual_level.pt")
Virchow_Normal_Epithelial_Cells_data = torch.load(f = "xgboost_prediction/Normal Epithelial Cells_Virchow_individual_level_ratio100/xgboost_results_individual_level.pt")
Virchow2_Normal_Epithelial_Cells_data = torch.load(f = "xgboost_prediction/Normal Epithelial Cells_Virchow2_individual_level_ratio100/xgboost_results_individual_level.pt")
UNI2h_Normal_Epithelial_Cells_data = torch.load(f = "xgboost_prediction/Normal Epithelial Cells_UNI2h_individual_level_ratio100/xgboost_results_individual_level.pt")

Important_Features_Normal_Epithelial_Cells = {}

Conch_FI_Normal_Epithelial_Cells_DF = pd.DataFrame.from_dict(Conch_Normal_Epithelial_Cells_data["feature_importances"][0], orient='index')
for i in range(1, len(Conch_Normal_Epithelial_Cells_data["feature_importances"])):
    Conch_FI_Normal_Epithelial_Cells_DF = pd.concat([Conch_FI_Normal_Epithelial_Cells_DF, pd.DataFrame.from_dict(Conch_Normal_Epithelial_Cells_data["feature_importances"][i], orient='index')], axis=1)
Conch_FI_Normal_Epithelial_Cells_DF.fillna(0, inplace=True)
Conch_FI_Normal_Epithelial_Cells_DF_mean = Conch_FI_Normal_Epithelial_Cells_DF.mean(axis=1)
Conch_FI_Normal_Epithelial_Cells_DF_mean_sorted = Conch_FI_Normal_Epithelial_Cells_DF_mean.sort_values(ascending=False)
Conch_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage = Conch_FI_Normal_Epithelial_Cells_DF_mean_sorted.cumsum()/Conch_FI_Normal_Epithelial_Cells_DF_mean_sorted.sum()
Conch_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(Conch_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_Normal_Epithelial_Cells["Conch"] = list(Conch_FI_Normal_Epithelial_Cells_DF_mean_sorted.index[0:round(len(Conch_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_Normal_Epithelial_Cells["Conch_percentage"] = round(Conch_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(Conch_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)


ProvGigapath_FI_Normal_Epithelial_Cells_DF = pd.DataFrame.from_dict(ProvGigapath_Normal_Epithelial_Cells_data["feature_importances"][0], orient='index')
for i in range(1, len(ProvGigapath_Normal_Epithelial_Cells_data["feature_importances"])):
    ProvGigapath_FI_Normal_Epithelial_Cells_DF = pd.concat([ProvGigapath_FI_Normal_Epithelial_Cells_DF, pd.DataFrame.from_dict(ProvGigapath_Normal_Epithelial_Cells_data["feature_importances"][i], orient='index')], axis=1)
ProvGigapath_FI_Normal_Epithelial_Cells_DF.fillna(0, inplace=True)
ProvGigapath_FI_Normal_Epithelial_Cells_DF_mean = ProvGigapath_FI_Normal_Epithelial_Cells_DF.mean(axis=1)
ProvGigapath_FI_Normal_Epithelial_Cells_DF_mean_sorted = ProvGigapath_FI_Normal_Epithelial_Cells_DF_mean.sort_values(ascending=False)
ProvGigapath_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage = ProvGigapath_FI_Normal_Epithelial_Cells_DF_mean_sorted.cumsum()/ProvGigapath_FI_Normal_Epithelial_Cells_DF_mean_sorted.sum()
ProvGigapath_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(ProvGigapath_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_Normal_Epithelial_Cells["ProvGigapath"] = list(ProvGigapath_FI_Normal_Epithelial_Cells_DF_mean_sorted.index[0:round(len(ProvGigapath_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_Normal_Epithelial_Cells["ProvGigapath_percentage"] = round(ProvGigapath_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(ProvGigapath_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)


Virchow_FI_Normal_Epithelial_Cells_DF = pd.DataFrame.from_dict(Virchow_Normal_Epithelial_Cells_data["feature_importances"][0], orient='index')  
for i in range(1, len(Virchow_Normal_Epithelial_Cells_data["feature_importances"])):
    Virchow_FI_Normal_Epithelial_Cells_DF = pd.concat([Virchow_FI_Normal_Epithelial_Cells_DF, pd.DataFrame.from_dict(Virchow_Normal_Epithelial_Cells_data["feature_importances"][i], orient='index')], axis=1)
Virchow_FI_Normal_Epithelial_Cells_DF.fillna(0, inplace=True)
Virchow_FI_Normal_Epithelial_Cells_DF_mean = Virchow_FI_Normal_Epithelial_Cells_DF.mean(axis=1)
Virchow_FI_Normal_Epithelial_Cells_DF_mean_sorted = Virchow_FI_Normal_Epithelial_Cells_DF_mean.sort_values(ascending=False)
Virchow_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage = Virchow_FI_Normal_Epithelial_Cells_DF_mean_sorted.cumsum()/Virchow_FI_Normal_Epithelial_Cells_DF_mean_sorted.sum()
Virchow_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(Virchow_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_Normal_Epithelial_Cells["Virchow"] = list(Virchow_FI_Normal_Epithelial_Cells_DF_mean_sorted.index[0:round(len(Virchow_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_Normal_Epithelial_Cells["Virchow_percentage"] = round(Virchow_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(Virchow_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)


Virchow2_FI_Normal_Epithelial_Cells_DF = pd.DataFrame.from_dict(Virchow2_Normal_Epithelial_Cells_data["feature_importances"][0], orient='index')
for i in range(1, len(Virchow2_Normal_Epithelial_Cells_data["feature_importances"])):
    Virchow2_FI_Normal_Epithelial_Cells_DF = pd.concat([Virchow2_FI_Normal_Epithelial_Cells_DF, pd.DataFrame.from_dict(Virchow2_Normal_Epithelial_Cells_data["feature_importances"][i], orient='index')], axis=1)
Virchow2_FI_Normal_Epithelial_Cells_DF.fillna(0, inplace=True)
Virchow2_FI_Normal_Epithelial_Cells_DF_mean = Virchow2_FI_Normal_Epithelial_Cells_DF.mean(axis=1)
Virchow2_FI_Normal_Epithelial_Cells_DF_mean_sorted = Virchow2_FI_Normal_Epithelial_Cells_DF_mean.sort_values(ascending=False)
Virchow2_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage = Virchow2_FI_Normal_Epithelial_Cells_DF_mean_sorted.cumsum()/Virchow2_FI_Normal_Epithelial_Cells_DF_mean_sorted.sum()
Virchow2_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(Virchow2_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_Normal_Epithelial_Cells["Virchow2"] = list(Virchow2_FI_Normal_Epithelial_Cells_DF_mean_sorted.index[0:round(len(Virchow2_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_Normal_Epithelial_Cells["Virchow2_percentage"] = round(Virchow2_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(Virchow2_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)



UNI2h_FI_Normal_Epithelial_Cells_DF = pd.DataFrame.from_dict(UNI2h_Normal_Epithelial_Cells_data["feature_importances"][0], orient='index')  
for i in range(1, len(UNI2h_Normal_Epithelial_Cells_data["feature_importances"])):
    UNI2h_FI_Normal_Epithelial_Cells_DF = pd.concat([UNI2h_FI_Normal_Epithelial_Cells_DF, pd.DataFrame.from_dict(UNI2h_Normal_Epithelial_Cells_data["feature_importances"][i], orient='index')], axis=1)
UNI2h_FI_Normal_Epithelial_Cells_DF.fillna(0, inplace=True)
UNI2h_FI_Normal_Epithelial_Cells_DF_mean = UNI2h_FI_Normal_Epithelial_Cells_DF.mean(axis=1)
UNI2h_FI_Normal_Epithelial_Cells_DF_mean_sorted = UNI2h_FI_Normal_Epithelial_Cells_DF_mean.sort_values(ascending=False)
UNI2h_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage = UNI2h_FI_Normal_Epithelial_Cells_DF_mean_sorted.cumsum()/UNI2h_FI_Normal_Epithelial_Cells_DF_mean_sorted.sum()
UNI2h_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(UNI2h_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_Normal_Epithelial_Cells["UNI2h"] = list(UNI2h_FI_Normal_Epithelial_Cells_DF_mean_sorted.index[0:round(len(UNI2h_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_Normal_Epithelial_Cells["UNI2h_percentage"] = round(UNI2h_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(UNI2h_FI_Normal_Epithelial_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)


print(Important_Features_Normal_Epithelial_Cells["Conch_percentage"])
print(Important_Features_Normal_Epithelial_Cells["ProvGigapath_percentage"])
print(Important_Features_Normal_Epithelial_Cells["Virchow_percentage"])
print(Important_Features_Normal_Epithelial_Cells["Virchow2_percentage"])
print(Important_Features_Normal_Epithelial_Cells["UNI2h_percentage"])

print(len(Important_Features_Normal_Epithelial_Cells["Conch"]))
print(len(Important_Features_Normal_Epithelial_Cells["ProvGigapath"]))
print(len(Important_Features_Normal_Epithelial_Cells["Virchow"]))
print(len(Important_Features_Normal_Epithelial_Cells["Virchow2"]))
print(len(Important_Features_Normal_Epithelial_Cells["UNI2h"]))

pickle.dump(Important_Features_Normal_Epithelial_Cells, open("Colorectal_Cancer_HE_patches/xgboost_prediction/important_features_Normal Epithelial Cells.pkl", "wb"))



###  T cells

Conch_T_Cells_data = torch.load(f = "xgboost_prediction/T Cells_Conch_individual_level_ratio100/xgboost_results_individual_level.pt")
ProvGigapath_T_Cells_data = torch.load(f = "xgboost_prediction/T Cells_ProvGigapath_individual_level_ratio100/xgboost_results_individual_level.pt")
Virchow_T_Cells_data = torch.load(f = "xgboost_prediction/T Cells_Virchow_individual_level_ratio100/xgboost_results_individual_level.pt")
Virchow2_T_Cells_data = torch.load(f = "xgboost_prediction/T Cells_Virchow2_individual_level_ratio100/xgboost_results_individual_level.pt")
UNI2h_T_Cells_data = torch.load(f = "xgboost_prediction/T Cells_UNI2h_individual_level_ratio100/xgboost_results_individual_level.pt")

Important_Features_T_Cells = {}

Conch_FI_T_Cells_DF = pd.DataFrame.from_dict(Conch_T_Cells_data["feature_importances"][0], orient='index')
for i in range(1, len(Conch_T_Cells_data["feature_importances"])):
    Conch_FI_T_Cells_DF = pd.concat([Conch_FI_T_Cells_DF, pd.DataFrame.from_dict(Conch_T_Cells_data["feature_importances"][i], orient='index')], axis=1)
Conch_FI_T_Cells_DF.fillna(0, inplace=True)
Conch_FI_T_Cells_DF_mean = Conch_FI_T_Cells_DF.mean(axis=1)
Conch_FI_T_Cells_DF_mean_sorted = Conch_FI_T_Cells_DF_mean.sort_values(ascending=False)
Conch_FI_T_Cells_DF_mean_sorted_cumsum_percentage = Conch_FI_T_Cells_DF_mean_sorted.cumsum()/Conch_FI_T_Cells_DF_mean_sorted.sum()
Conch_FI_T_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(Conch_FI_T_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_T_Cells["Conch"] = list(Conch_FI_T_Cells_DF_mean_sorted.index[0:round(len(Conch_FI_T_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_T_Cells["Conch_percentage"] = round(Conch_FI_T_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(Conch_FI_T_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)


ProvGigapath_FI_T_Cells_DF = pd.DataFrame.from_dict(ProvGigapath_T_Cells_data["feature_importances"][0], orient='index')
for i in range(1, len(ProvGigapath_T_Cells_data["feature_importances"])):
    ProvGigapath_FI_T_Cells_DF = pd.concat([ProvGigapath_FI_T_Cells_DF, pd.DataFrame.from_dict(ProvGigapath_T_Cells_data["feature_importances"][i], orient='index')], axis=1)
ProvGigapath_FI_T_Cells_DF.fillna(0, inplace=True)
ProvGigapath_FI_T_Cells_DF_mean = ProvGigapath_FI_T_Cells_DF.mean(axis=1)
ProvGigapath_FI_T_Cells_DF_mean_sorted = ProvGigapath_FI_T_Cells_DF_mean.sort_values(ascending=False)
ProvGigapath_FI_T_Cells_DF_mean_sorted_cumsum_percentage = ProvGigapath_FI_T_Cells_DF_mean_sorted.cumsum()/ProvGigapath_FI_T_Cells_DF_mean_sorted.sum()
ProvGigapath_FI_T_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(ProvGigapath_FI_T_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_T_Cells["ProvGigapath"] = list(ProvGigapath_FI_T_Cells_DF_mean_sorted.index[0:round(len(ProvGigapath_FI_T_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_T_Cells["ProvGigapath_percentage"] = round(ProvGigapath_FI_T_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(ProvGigapath_FI_T_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)


Virchow_FI_T_Cells_DF = pd.DataFrame.from_dict(Virchow_T_Cells_data["feature_importances"][0], orient='index')
for i in range(1, len(Virchow_T_Cells_data["feature_importances"])):
    Virchow_FI_T_Cells_DF = pd.concat([Virchow_FI_T_Cells_DF, pd.DataFrame.from_dict(Virchow_T_Cells_data["feature_importances"][i], orient='index')], axis=1)
Virchow_FI_T_Cells_DF.fillna(0, inplace=True)
Virchow_FI_T_Cells_DF_mean = Virchow_FI_T_Cells_DF.mean(axis=1)
Virchow_FI_T_Cells_DF_mean_sorted = Virchow_FI_T_Cells_DF_mean.sort_values(ascending=False)
Virchow_FI_T_Cells_DF_mean_sorted_cumsum_percentage = Virchow_FI_T_Cells_DF_mean_sorted.cumsum()/Virchow_FI_T_Cells_DF_mean_sorted.sum()
Virchow_FI_T_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(Virchow_FI_T_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_T_Cells["Virchow"] = list(Virchow_FI_T_Cells_DF_mean_sorted.index[0:round(len(Virchow_FI_T_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_T_Cells["Virchow_percentage"] = round(Virchow_FI_T_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(Virchow_FI_T_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)


Virchow2_FI_T_Cells_DF = pd.DataFrame.from_dict(Virchow2_T_Cells_data["feature_importances"][0], orient='index')
for i in range(1, len(Virchow2_T_Cells_data["feature_importances"])):
    Virchow2_FI_T_Cells_DF = pd.concat([Virchow2_FI_T_Cells_DF, pd.DataFrame.from_dict(Virchow2_T_Cells_data["feature_importances"][i], orient='index')], axis=1)
Virchow2_FI_T_Cells_DF.fillna(0, inplace=True)
Virchow2_FI_T_Cells_DF_mean = Virchow2_FI_T_Cells_DF.mean(axis=1)
Virchow2_FI_T_Cells_DF_mean_sorted = Virchow2_FI_T_Cells_DF_mean.sort_values(ascending=False)
Virchow2_FI_T_Cells_DF_mean_sorted_cumsum_percentage = Virchow2_FI_T_Cells_DF_mean_sorted.cumsum()/Virchow2_FI_T_Cells_DF_mean_sorted.sum()
Virchow2_FI_T_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(Virchow2_FI_T_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_T_Cells["Virchow2"] = list(Virchow2_FI_T_Cells_DF_mean_sorted.index[0:round(len(Virchow2_FI_T_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_T_Cells["Virchow2_percentage"] = round(Virchow2_FI_T_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(Virchow2_FI_T_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)


UNI2h_FI_T_Cells_DF = pd.DataFrame.from_dict(UNI2h_T_Cells_data["feature_importances"][0], orient='index')
for i in range(1, len(UNI2h_T_Cells_data["feature_importances"])):
    UNI2h_FI_T_Cells_DF = pd.concat([UNI2h_FI_T_Cells_DF, pd.DataFrame.from_dict(UNI2h_T_Cells_data["feature_importances"][i], orient='index')], axis=1)
UNI2h_FI_T_Cells_DF.fillna(0, inplace=True)
UNI2h_FI_T_Cells_DF_mean = UNI2h_FI_T_Cells_DF.mean(axis=1)
UNI2h_FI_T_Cells_DF_mean_sorted = UNI2h_FI_T_Cells_DF_mean.sort_values(ascending=False)
UNI2h_FI_T_Cells_DF_mean_sorted_cumsum_percentage = UNI2h_FI_T_Cells_DF_mean_sorted.cumsum()/UNI2h_FI_T_Cells_DF_mean_sorted.sum()
UNI2h_FI_T_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(UNI2h_FI_T_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_T_Cells["UNI2h"] = list(UNI2h_FI_T_Cells_DF_mean_sorted.index[0:round(len(UNI2h_FI_T_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_T_Cells["UNI2h_percentage"] = round(UNI2h_FI_T_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(UNI2h_FI_T_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)


print(Important_Features_T_Cells["Conch_percentage"])
print(Important_Features_T_Cells["ProvGigapath_percentage"])
print(Important_Features_T_Cells["Virchow_percentage"])
print(Important_Features_T_Cells["Virchow2_percentage"])
print(Important_Features_T_Cells["UNI2h_percentage"])

print(len(Important_Features_T_Cells["Conch"]))
print(len(Important_Features_T_Cells["ProvGigapath"]))
print(len(Important_Features_T_Cells["Virchow"]))
print(len(Important_Features_T_Cells["Virchow2"]))
print(len(Important_Features_T_Cells["UNI2h"]))

pickle.dump(Important_Features_T_Cells, open("Colorectal_Cancer_HE_patches/xgboost_prediction/important_features_T Cells.pkl", "wb"))



### Other Immune Cells

Conch_Other_Immune_Cells_data = torch.load(f = "xgboost_prediction/Other Immune Cells_Conch_individual_level_ratio100/xgboost_results_individual_level.pt")
ProvGigapath_Other_Immune_Cells_data = torch.load(f = "xgboost_prediction/Other Immune Cells_ProvGigapath_individual_level_ratio100/xgboost_results_individual_level.pt")
Virchow_Other_Immune_Cells_data = torch.load(f = "xgboost_prediction/Other Immune Cells_Virchow_individual_level_ratio100/xgboost_results_individual_level.pt")
Virchow2_Other_Immune_Cells_data = torch.load(f = "xgboost_prediction/Other Immune Cells_Virchow2_individual_level_ratio100/xgboost_results_individual_level.pt")
UNI2h_Other_Immune_Cells_data = torch.load(f = "xgboost_prediction/Other Immune Cells_UNI2h_individual_level_ratio100/xgboost_results_individual_level.pt")

Important_Features_Other_Immune_Cells = {}

Conch_FI_Other_Immune_Cells_DF = pd.DataFrame.from_dict(Conch_Other_Immune_Cells_data["feature_importances"][0], orient='index')
for i in range(1, len(Conch_Other_Immune_Cells_data["feature_importances"])):
    Conch_FI_Other_Immune_Cells_DF = pd.concat([Conch_FI_Other_Immune_Cells_DF, pd.DataFrame.from_dict(Conch_Other_Immune_Cells_data["feature_importances"][i], orient='index')], axis=1)
Conch_FI_Other_Immune_Cells_DF.fillna(0, inplace=True)
Conch_FI_Other_Immune_Cells_DF_mean = Conch_FI_Other_Immune_Cells_DF.mean(axis=1)
Conch_FI_Other_Immune_Cells_DF_mean_sorted = Conch_FI_Other_Immune_Cells_DF_mean.sort_values(ascending=False)
Conch_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage = Conch_FI_Other_Immune_Cells_DF_mean_sorted.cumsum()/Conch_FI_Other_Immune_Cells_DF_mean_sorted.sum()
Conch_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(Conch_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_Other_Immune_Cells["Conch"] = list(Conch_FI_Other_Immune_Cells_DF_mean_sorted.index[0:round(len(Conch_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_Other_Immune_Cells["Conch_percentage"] = round(Conch_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(Conch_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)

ProvGigapath_FI_Other_Immune_Cells_DF = pd.DataFrame.from_dict(ProvGigapath_Other_Immune_Cells_data["feature_importances"][0], orient='index')
for i in range(1, len(ProvGigapath_Other_Immune_Cells_data["feature_importances"])):
    ProvGigapath_FI_Other_Immune_Cells_DF = pd.concat([ProvGigapath_FI_Other_Immune_Cells_DF, pd.DataFrame.from_dict(ProvGigapath_Other_Immune_Cells_data["feature_importances"][i], orient='index')], axis=1)
ProvGigapath_FI_Other_Immune_Cells_DF.fillna(0, inplace=True)
ProvGigapath_FI_Other_Immune_Cells_DF_mean = ProvGigapath_FI_Other_Immune_Cells_DF.mean(axis=1)
ProvGigapath_FI_Other_Immune_Cells_DF_mean_sorted = ProvGigapath_FI_Other_Immune_Cells_DF_mean.sort_values(ascending=False)
ProvGigapath_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage = ProvGigapath_FI_Other_Immune_Cells_DF_mean_sorted.cumsum()/ProvGigapath_FI_Other_Immune_Cells_DF_mean_sorted.sum()
ProvGigapath_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(ProvGigapath_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_Other_Immune_Cells["ProvGigapath"] = list(ProvGigapath_FI_Other_Immune_Cells_DF_mean_sorted.index[0:round(len(ProvGigapath_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_Other_Immune_Cells["ProvGigapath_percentage"] = round(ProvGigapath_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(ProvGigapath_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)

Virchow_FI_Other_Immune_Cells_DF = pd.DataFrame.from_dict(Virchow_Other_Immune_Cells_data["feature_importances"][0], orient='index')
for i in range(1, len(Virchow_Other_Immune_Cells_data["feature_importances"])):
    Virchow_FI_Other_Immune_Cells_DF = pd.concat([Virchow_FI_Other_Immune_Cells_DF, pd.DataFrame.from_dict(Virchow_Other_Immune_Cells_data["feature_importances"][i], orient='index')], axis=1)
Virchow_FI_Other_Immune_Cells_DF.fillna(0, inplace=True)
Virchow_FI_Other_Immune_Cells_DF_mean = Virchow_FI_Other_Immune_Cells_DF.mean(axis=1)
Virchow_FI_Other_Immune_Cells_DF_mean_sorted = Virchow_FI_Other_Immune_Cells_DF_mean.sort_values(ascending=False)
Virchow_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage = Virchow_FI_Other_Immune_Cells_DF_mean_sorted.cumsum()/Virchow_FI_Other_Immune_Cells_DF_mean_sorted.sum()
Virchow_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(Virchow_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_Other_Immune_Cells["Virchow"] = list(Virchow_FI_Other_Immune_Cells_DF_mean_sorted.index[0:round(len(Virchow_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_Other_Immune_Cells["Virchow_percentage"] = round(Virchow_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(Virchow_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)


Virchow2_FI_Other_Immune_Cells_DF = pd.DataFrame.from_dict(Virchow2_Other_Immune_Cells_data["feature_importances"][0], orient='index')
for i in range(1, len(Virchow2_Other_Immune_Cells_data["feature_importances"])):
    Virchow2_FI_Other_Immune_Cells_DF = pd.concat([Virchow2_FI_Other_Immune_Cells_DF, pd.DataFrame.from_dict(Virchow2_Other_Immune_Cells_data["feature_importances"][i], orient='index')], axis=1)
Virchow2_FI_Other_Immune_Cells_DF.fillna(0, inplace=True)
Virchow2_FI_Other_Immune_Cells_DF_mean = Virchow2_FI_Other_Immune_Cells_DF.mean(axis=1)
Virchow2_FI_Other_Immune_Cells_DF_mean_sorted = Virchow2_FI_Other_Immune_Cells_DF_mean.sort_values(ascending=False)
Virchow2_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage = Virchow2_FI_Other_Immune_Cells_DF_mean_sorted.cumsum()/Virchow2_FI_Other_Immune_Cells_DF_mean_sorted.sum()
Virchow2_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(Virchow2_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_Other_Immune_Cells["Virchow2"] = list(Virchow2_FI_Other_Immune_Cells_DF_mean_sorted.index[0:round(len(Virchow2_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_Other_Immune_Cells["Virchow2_percentage"] = round(Virchow2_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(Virchow2_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)


UNI2h_FI_Other_Immune_Cells_DF = pd.DataFrame.from_dict(UNI2h_Other_Immune_Cells_data["feature_importances"][0], orient='index')
for i in range(1, len(UNI2h_Other_Immune_Cells_data["feature_importances"])):
    UNI2h_FI_Other_Immune_Cells_DF = pd.concat([UNI2h_FI_Other_Immune_Cells_DF, pd.DataFrame.from_dict(UNI2h_Other_Immune_Cells_data["feature_importances"][i], orient='index')], axis=1)
UNI2h_FI_Other_Immune_Cells_DF.fillna(0, inplace=True)  
UNI2h_FI_Other_Immune_Cells_DF_mean = UNI2h_FI_Other_Immune_Cells_DF.mean(axis=1)
UNI2h_FI_Other_Immune_Cells_DF_mean_sorted = UNI2h_FI_Other_Immune_Cells_DF_mean.sort_values(ascending=False)
UNI2h_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage = UNI2h_FI_Other_Immune_Cells_DF_mean_sorted.cumsum()/UNI2h_FI_Other_Immune_Cells_DF_mean_sorted.sum()
UNI2h_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage.iloc[int(len(UNI2h_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage)*0.3)]
Important_Features_Other_Immune_Cells["UNI2h"] = list(UNI2h_FI_Other_Immune_Cells_DF_mean_sorted.index[0:round(len(UNI2h_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage)*0.3)])
Important_Features_Other_Immune_Cells["UNI2h_percentage"] = round(UNI2h_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage.iloc[round(len(UNI2h_FI_Other_Immune_Cells_DF_mean_sorted_cumsum_percentage)*0.3)], 3)


print(Important_Features_Other_Immune_Cells["Conch_percentage"])
print(Important_Features_Other_Immune_Cells["ProvGigapath_percentage"])
print(Important_Features_Other_Immune_Cells["Virchow_percentage"])
print(Important_Features_Other_Immune_Cells["Virchow2_percentage"])
print(Important_Features_Other_Immune_Cells["UNI2h_percentage"])

print(len(Important_Features_Other_Immune_Cells["Conch"]))
print(len(Important_Features_Other_Immune_Cells["ProvGigapath"]))
print(len(Important_Features_Other_Immune_Cells["Virchow"]))
print(len(Important_Features_Other_Immune_Cells["Virchow2"]))
print(len(Important_Features_Other_Immune_Cells["UNI2h"]))

pickle.dump(Important_Features_Other_Immune_Cells, open("xgboost_prediction/important_features_Other Immune Cells.pkl", "wb"))





# Create multi-sheet Excel with important features for all cell types


Important_Features_Cancer_Cells = pickle.load(open("xgboost_prediction/important_features_Cancer Cells.pkl", "rb"))
Important_Features_Stromal_Cells = pickle.load(open("xgboost_prediction/important_features_Stromal Cells.pkl", "rb"))
Important_Features_Normal_Epithelial_Cells = pickle.load(open("xgboost_prediction/important_features_Normal Epithelial Cells.pkl", "rb"))
Important_Features_T_Cells = pickle.load(open("xgboost_prediction/important_features_T Cells.pkl", "rb"))
Important_Features_Other_Immune_Cells = pickle.load(open("xgboost_prediction/important_features_Other Immune Cells.pkl", "rb"))


wb = Workbook()
# Remove default sheet
wb.remove(wb.active)

# Define cell types and their corresponding feature dictionaries in fixed order
cell_types_order = ['Cancer_Cells', 'Stromal_Cells', 'Normal_Epithelial_Cells', 'T_Cells', 'Other_Immune_Cells']
cell_types_data = {
    'Cancer_Cells': Important_Features_Cancer_Cells,
    'Stromal_Cells': Important_Features_Stromal_Cells, 
    'Normal_Epithelial_Cells': Important_Features_Normal_Epithelial_Cells,
    'T_Cells': Important_Features_T_Cells,
    'Other_Immune_Cells': Important_Features_Other_Immune_Cells
}

# Model order and their corresponding keys in dictionaries
models = ['Conch', 'ProvGigapath', 'Virchow', 'Virchow2', 'UNI2h']

for cell_type in cell_types_order:
    features_dict = cell_types_data[cell_type]
    print(f"Processing {cell_type}...")
    
    # Create worksheet for this cell type
    ws = wb.create_sheet(title=cell_type)
    
    # Create header row
    headers = []
    for model in models:
        headers.extend([f'{model}_Features', f'{model}_Importance'])
    
    # Write headers
    for col_idx, header in enumerate(headers, 1):
        ws.cell(row=1, column=col_idx, value=header)
    
    # Get features and importance scores for each model
    model_data = {}
    max_features = 0
    
    for model in models:
        # Get feature list and importance scores  
        feature_key = model
        percentage_key = f'{model}_percentage'
        
        if feature_key in features_dict:
            features = features_dict[feature_key]
            
            # Load the corresponding data to get importance scores
            # Convert cell_type key to the actual folder name format
            folder_name = cell_type.replace('_', ' ')
            data_file = f"xgboost_prediction/{folder_name}_{model}_individual_level_ratio100/xgboost_results_individual_level.pt"
            print(f"  Loading: {data_file}")
            model_results = torch.load(data_file)
            
            # Calculate mean importance scores (same logic as before)
            feature_importances = model_results["feature_importances"]
            feature_names = sorted(list(feature_importances[0].keys()))  # Sort for consistent order
            n_folds = len(feature_importances)
            
            # Create DataFrame with feature names as index and folds as columns
            importance_df = pd.DataFrame(index=feature_names, columns=[f'Fold_{i+1}' for i in range(n_folds)])
            
            # Fill the DataFrame with importance scores from each fold
            for fold_idx in range(n_folds):
                fold_importance = feature_importances[fold_idx]
                for feature_name in feature_names:
                    importance_df.loc[feature_name, f'Fold_{fold_idx+1}'] = fold_importance.get(feature_name, 0.0)
            
            # Convert to float and calculate mean
            importance_df = importance_df.astype(float)
            mean_importance = importance_df.mean(axis=1)
            mean_importance_sorted = mean_importance.sort_values(ascending=False)
            
            # Get importance scores for selected features in the same order
            importance_scores = [mean_importance_sorted[feature] for feature in features]
            
            model_data[model] = {
                'features': features,
                'scores': importance_scores
            }
            
            max_features = max(max_features, len(features))
        else:
            model_data[model] = {
                'features': [],
                'scores': []
            }
    
    # Fill the worksheet
    for row_idx in range(max_features):
        excel_row = row_idx + 2  # +2 because Excel is 1-indexed and we have header
        
        col_idx = 1
        for model in models:
            # Feature name column
            if row_idx < len(model_data[model]['features']):
                feature_name = model_data[model]['features'][row_idx]
                importance_score = model_data[model]['scores'][row_idx]
            else:
                feature_name = ""
                importance_score = ""
            
            ws.cell(row=excel_row, column=col_idx, value=feature_name)
            ws.cell(row=excel_row, column=col_idx + 1, value=importance_score)
            
            col_idx += 2  # Move to next model (skip 2 columns)
    
    print(f"  Added {max_features} rows for {cell_type}")

# Save the Excel file
output_excel_path = "xgboost_prediction/Important_Features_All_CellTypes.xlsx"
wb.save(output_excel_path)
print(f"\n✅ Multi-sheet Excel file saved to: {output_excel_path}")

# Print summary
print(f"\nSummary:")
print(f"- 5 sheets (one per cell type)")
print(f"- 10 columns per sheet (5 models  2 columns each)")
print(f"- Features sorted by importance score (highest first)")
print(f"- Empty cells filled with blanks for alignment")
