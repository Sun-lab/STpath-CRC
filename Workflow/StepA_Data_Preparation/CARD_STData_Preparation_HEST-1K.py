"""
STPath-COAD: CARD Spatial Transcriptomics Data Preparation for HEST-1K
======================================================================

This script prepares HEST-1K spatial transcriptomics data for CARD deconvolution
analysis by formatting expression matrices and spatial coordinates.

Author: Saishi Cui
Date: December 2025

Purpose: Process and format HEST-1K spatial transcriptomics data for downstream 
CARD deconvolution analysis. Prepares expression matrices and coordinate files 
required for cell type proportion estimation on HEST dataset.
"""

import h5py
import datasets
import pandas as pd
import scanpy as sc
import os
from PIL import Image
import numpy as np
local_dir='data/hest_data' # hest will be dowloaded to this folder

meta_df = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_1_0.csv")

meta_df = meta_df[(meta_df['organ'] == 'Bowel') & (meta_df['oncotree_code'] == 'COAD') & (meta_df['st_technology'] == 'Visium') & (meta_df["dataset_title"] != "COLON MAP: Colon Molecular Atlas Project")]
meta_df.reset_index(drop=True, inplace=True)
meta_df.to_csv("FredHutch_Colorectal/meta_df_Bowel_COAD_Visium_final.csv")


ids_to_query = meta_df['id'].values
list_patterns = [f"*{id}[_.]**" for id in ids_to_query]
dataset = datasets.load_dataset(
    'MahmoodLab/hest', 
    cache_dir=local_dir,
    patterns=list_patterns,
    trust_remote_code=True 
)



### Prepare CARD Need dataset 


sample_files = os.listdir("data/hest_data/st")
sample_id_list = [x.split(".")[0] for x in sample_files]


for sample_id in sample_id_list:

    st_file = f"data/hest_data/st/{sample_id}.h5ad"
    adata = sc.read_h5ad(st_file)
    if hasattr(adata.X, 'toarray'):
        X_dense = adata.X.toarray()
    else:
        X_dense = adata.X
    ST_df = pd.DataFrame(X_dense, index=adata.obs_names, columns=adata.var_names)



    h5_file = f"data/hest_data/patches/{sample_id}.h5"

    f = h5py.File(h5_file, 'r')
    spatial_coords = pd.DataFrame(f["coords"][:])
    spatial_coords.columns = ["x", "y"]
    barcodes = [x[0].decode('utf-8') for x in f["barcode"][:]]
    spatial_coords.index = barcodes

    intersect_barcodes = set(ST_df.index) & set(barcodes)
    intersect_barcodes = list(intersect_barcodes)


    final_spatial_coords = spatial_coords.loc[intersect_barcodes]
    final_ST_df = ST_df.loc[intersect_barcodes]


    final_spatial_coords.to_csv(f"data/hest_data/CARD_Need_Files/{sample_id}_spatial.csv")
    final_ST_df.to_csv(f"data/hest_data/CARD_Need_Files/{sample_id}_expression.csv")
    
    # Save image patches
    os.makedirs(f"data/hest_data/image_patches/{sample_id}", exist_ok=True)
    images = f["img"][:]
    
    for i, barcode in enumerate(barcodes):
        if barcode in intersect_barcodes:
            img_array = images[i]
            # Ensure data type is uint8
            if img_array.dtype != np.uint8:
                img_array = (img_array * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            img.save(f"data/hest_data/image_patches/{sample_id}/{barcode}.tif")
    
    f.close()



