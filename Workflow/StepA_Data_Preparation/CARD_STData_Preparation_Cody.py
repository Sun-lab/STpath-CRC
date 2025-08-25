"""
STPath-COAD: CARD Spatial Transcriptomics Data Preparation
=========================================================

This script prepares spatial transcriptomics data for CARD deconvolution analysis
by formatting expression matrices and spatial coordinates.

Author: Saishi Cui
Date: Sept 2025

Purpose: Process and format spatial transcriptomics data for downstream cell type (CARD) 
deconvolution analysis. Prepares expression matrices and coordinate files 
required for cell type proportion estimation.
"""

import scanpy as sc
import pandas as pd
import numpy as np
import os
import re
from scipy.sparse import issparse
import glob

# specify the path
path_ST_data = "data/ST_data_for_deconv/"

# get all .h5ad files in the path
ST_data_path_list = []
for root, dirs, files in os.walk(path_ST_data):
    for file in files:
        if file.endswith(".h5ad"):  #
            ST_data_path_list.append(os.path.join(root, file))

ST_data_path_list = sorted(ST_data_path_list)

# print all .h5ad files
for file in ST_data_path_list:
    print(file)

len(ST_data_path_list)


# get all .csv files in the path
path_ST_spatial = "data/Scaled_Spatial_coordinates/"
ST_spatial_path_list = []
for root, dirs, files in os.walk(path_ST_spatial):
    for file in files:
        if file.endswith(".csv"):  #
            ST_spatial_path_list.append(os.path.join(root, file))

ST_spatial_path_list = sorted(ST_spatial_path_list)

# print all .csv files
for file in ST_spatial_path_list:
    print(file)

len(ST_spatial_path_list)






# assume h5ad_files_sorted is your file path list
extracted_parts = []

for file_path in ST_data_path_list:
    # extract the file name
    file_name = os.path.basename(file_path)
    
    # use regular expression to match xxxx_xx_x format
    match = re.match(r'(\d{4}_\w{2}_\d+)', file_name)
    if match:
        extracted_parts.append(match.group(1))

# print the extracted parts
for part in extracted_parts:
    print(part)




##### Concatenate expression data from multiple files "7794_AS_1","7794_AS_2", "8578_AS_1", "8578_AS_2", "8578_AS_3"

 ## 7794_AS_1
ST_file_path1 = 'data/ST_data_for_deconv/7794_AS_1_filtered_trimmed_WD33479.h5ad'
ST_file_path2 = 'data/ST_data_for_deconv/7794_AS_1_filtered_trimmed_WD33476.h5ad'

adata1 = sc.read_h5ad(ST_file_path1)
if issparse(adata1.X):
    dense_matrix = adata1.X.toarray()
else:
    dense_matrix = adata1.X
expression_df1 = pd.DataFrame(dense_matrix, index=adata1.obs_names, columns=adata1.var_names).transpose()

adata2 = sc.read_h5ad(ST_file_path2)
if issparse(adata2.X):
    dense_matrix = adata2.X.toarray()
else:
    dense_matrix = adata2.X
expression_df2 = pd.DataFrame(dense_matrix, index=adata2.obs_names, columns=adata2.var_names).transpose()

intersected_rows = expression_df1.index.intersection(expression_df2.index)
expression_df1 = expression_df1.loc[intersected_rows]
expression_df2 = expression_df2.loc[intersected_rows]
Concated_expression_df = pd.concat([expression_df1, expression_df2], axis=1).T

Concated_expression_df.to_csv('data/ST_data_for_deconv/7794_AS_1_expression.csv', index=True)



 ## 7794_AS_2
ST_file_path1 = 'data/ST_data_for_deconv/7794_AS_2_filtered_trimmed_WD33468.h5ad'
ST_file_path2 = 'data/ST_data_for_deconv/7794_AS_2_filtered_trimmed_WD33475.h5ad'

adata1 = sc.read_h5ad(ST_file_path1)
if issparse(adata1.X):
    dense_matrix = adata1.X.toarray()
else:
    dense_matrix = adata1.X
expression_df1 = pd.DataFrame(dense_matrix, index=adata1.obs_names, columns=adata1.var_names).transpose()

adata2 = sc.read_h5ad(ST_file_path2)
if issparse(adata2.X):
    dense_matrix = adata2.X.toarray()
else:
    dense_matrix = adata2.X
expression_df2 = pd.DataFrame(dense_matrix, index=adata2.obs_names, columns=adata2.var_names).transpose()

intersected_rows = expression_df1.index.intersection(expression_df2.index)
expression_df1 = expression_df1.loc[intersected_rows]
expression_df2 = expression_df2.loc[intersected_rows]
Concated_expression_df = pd.concat([expression_df1, expression_df2], axis=1).T

Concated_expression_df.to_csv('data/ST_data_for_deconv/7794_AS_2_expression.csv', index=True)



 ## 8578_AS_1
ST_file_path1 = 'data/ST_data_for_deconv/8578_AS_1_filtered_trimmed_WD33469.h5ad'
ST_file_path2 = 'data/ST_data_for_deconv/8578_AS_1_filtered_trimmed_WD33473.h5ad'
ST_file_path3 = 'data/ST_data_for_deconv/8578_AS_1_filtered_trimmed_WD33474.h5ad'

adata1 = sc.read_h5ad(ST_file_path1)
if issparse(adata1.X):
    dense_matrix = adata1.X.toarray()
else:
    dense_matrix = adata1.X
expression_df1 = pd.DataFrame(dense_matrix, index=adata1.obs_names, columns=adata1.var_names).transpose()

adata2 = sc.read_h5ad(ST_file_path2)
if issparse(adata2.X):
    dense_matrix = adata2.X.toarray()
else:
    dense_matrix = adata2.X
expression_df2 = pd.DataFrame(dense_matrix, index=adata2.obs_names, columns=adata2.var_names).transpose()

adata3 = sc.read_h5ad(ST_file_path3)
if issparse(adata3.X):
    dense_matrix = adata3.X.toarray()
else:
    dense_matrix = adata3.X
expression_df3 = pd.DataFrame(dense_matrix, index=adata3.obs_names, columns=adata3.var_names).transpose()


intersected_rows = expression_df1.index.intersection(expression_df2.index).intersection(expression_df3.index)
expression_df1 = expression_df1.loc[intersected_rows]
expression_df2 = expression_df2.loc[intersected_rows]
expression_df3 = expression_df3.loc[intersected_rows]
Concated_expression_df = pd.concat([expression_df1, expression_df2, expression_df3], axis=1).T

Concated_expression_df.to_csv('data/ST_data_for_deconv/8578_AS_1_expression.csv', index=True)



 ## 8578_AS_2
ST_file_path1 = 'data/ST_data_for_deconv/8578_AS_2_filtered_trimmed_WD84216.h5ad'
ST_file_path2 = 'data/ST_data_for_deconv/8578_AS_2_filtered_trimmed_WD84226.h5ad'
ST_file_path3 = 'data/ST_data_for_deconv/8578_AS_2_filtered_trimmed_WD84594.h5ad'

adata1 = sc.read_h5ad(ST_file_path1)
if issparse(adata1.X):
    dense_matrix = adata1.X.toarray()
else:
    dense_matrix = adata1.X
expression_df1 = pd.DataFrame(dense_matrix, index=adata1.obs_names, columns=adata1.var_names).transpose()

adata2 = sc.read_h5ad(ST_file_path2)    
if issparse(adata2.X):
    dense_matrix = adata2.X.toarray()
else:
    dense_matrix = adata2.X
expression_df2 = pd.DataFrame(dense_matrix, index=adata2.obs_names, columns=adata2.var_names).transpose()

adata3 = sc.read_h5ad(ST_file_path3)
if issparse(adata3.X):
    dense_matrix = adata3.X.toarray()
else:
    dense_matrix = adata3.X
expression_df3 = pd.DataFrame(dense_matrix, index=adata3.obs_names, columns=adata3.var_names).transpose()


intersected_rows = expression_df1.index.intersection(expression_df2.index).intersection(expression_df3.index)
expression_df1 = expression_df1.loc[intersected_rows]
expression_df2 = expression_df2.loc[intersected_rows]
expression_df3 = expression_df3.loc[intersected_rows]
Concated_expression_df = pd.concat([expression_df1, expression_df2, expression_df3], axis=1).T

Concated_expression_df.to_csv('data/ST_data_for_deconv/8578_AS_2_expression.csv', index=True)


 ## 8578_AS_3
ST_file_path1 = 'data/ST_data_for_deconv/8578_AS_3_filtered_trimmed_WD84221.h5ad'
ST_file_path2 = 'data/ST_data_for_deconv/8578_AS_3_filtered_trimmed_WD84596.h5ad'

adata1 = sc.read_h5ad(ST_file_path1)
if issparse(adata1.X):
    dense_matrix = adata1.X.toarray()
else:
    dense_matrix = adata1.X
expression_df1 = pd.DataFrame(dense_matrix, index=adata1.obs_names, columns=adata1.var_names).transpose()

adata2 = sc.read_h5ad(ST_file_path2)
if issparse(adata2.X):
    dense_matrix = adata2.X.toarray()
else:
    dense_matrix = adata2.X
expression_df2 = pd.DataFrame(dense_matrix, index=adata2.obs_names, columns=adata2.var_names).transpose()

intersected_rows = expression_df1.index.intersection(expression_df2.index)
expression_df1 = expression_df1.loc[intersected_rows]
expression_df2 = expression_df2.loc[intersected_rows]
Concated_expression_df = pd.concat([expression_df1, expression_df2], axis=1).T

Concated_expression_df.to_csv('data/ST_data_for_deconv/8578_AS_3_expression.csv', index=True)




### For other samples, we don't need to concatenate the expression data, because there are only one file for each sample

one_sample_list = [sample_id for sample_id in extracted_parts if sample_id not in ["7794_AS_1", "7794_AS_2", "8578_AS_1", "8578_AS_2", "8578_AS_3"]]

for sample_id in one_sample_list:
    ST_file_path = f'data/ST_data_for_deconv/{sample_id}_filtered_trimmed.h5ad'
    adata = sc.read_h5ad(ST_file_path)
    if issparse(adata.X):
        dense_matrix = adata.X.toarray()
    else:
        dense_matrix = adata.X
    expression_df = pd.DataFrame(dense_matrix, index=adata.obs_names, columns=adata.var_names)
    expression_df.to_csv(f'data/ST_data_for_deconv/{sample_id}_expression.csv', index=True)






