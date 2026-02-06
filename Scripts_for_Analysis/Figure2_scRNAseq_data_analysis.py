"""
STPath-COAD: Single-cell RNA-seq Data Analysis for Figure 2
==========================================================

This script analyzes single-cell RNA-seq data to identify cell types and generate 
visualizations for Figure 2 Panel A-B. Includes cell type annotation, marker gene identification,
and UMAP visualizationc.

This script also generates plots for cell type deconvolution analysis (Figure 2 Panel C-E). Including average proportions by different 
cell types for all colorectal samples, correlation analysis between deconvoluted cell type proportions and
marker gene expression–based relative expression, and a representative H&E image 
with corresponding spatial hexagonal binned heatmaps of deconvoluted tumor cell proportions and
marker gene–based evidence, highlighting consistent spatial enrichment patterns.

Author: Saishi Cui
Date: Feb 2026


"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from anndata import AnnData
import seaborn as sns
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
import json




# Read h5ad data
scRNAseq_data_path = "scRNAseq_data/VUMC_COMBINED.h5ad"
scData_original = sc.read_h5ad(scRNAseq_data_path)


######## Scanpy re-clustering ##############################################################

scData_Recluster = scData_original.copy()

# filter cells and genes
sc.pp.filter_cells(scData_Recluster, min_genes= np.round(0.01*scData_Recluster.n_vars,0)) # Make sure for each cell at least 165 genes are expressed
sc.pp.filter_genes(scData_Recluster, min_cells= np.round(0.01*scData_Recluster.n_obs,0))  # Make sure for each gene at least 226 cells express the gene

# calculate quality control metrics
scData_Recluster.var['mt'] = scData_Recluster.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(scData_Recluster, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# filter out cells with high mitochondrial content and low gene numbers
scData_Recluster = scData_Recluster[scData_Recluster.obs.n_genes_by_counts < 2500, :]
scData_Recluster = scData_Recluster[scData_Recluster.obs.pct_counts_mt < 5, :]

# normalize and log transform
sc.pp.normalize_total(scData_Recluster, target_sum=1e4)
sc.pp.log1p(scData_Recluster)

# identify highly variable genes
sc.pp.highly_variable_genes(scData_Recluster, min_mean=0.0125, max_mean=3, min_disp=0.5)
scData_Recluster = scData_Recluster[:, scData_Recluster.var.highly_variable]
scData_Recluster_before_scale = scData_Recluster.copy()

# scale data
sc.pp.scale(scData_Recluster, max_value=10)

# PCA
sc.tl.pca(scData_Recluster, svd_solver='arpack')


# calculate neighbors graph
sc.pp.neighbors(scData_Recluster, n_neighbors=10, n_pcs=40)

# use Leiden algorithm for clustering
sc.tl.leiden(scData_Recluster, resolution=0.5)




# calculate UMAP
sc.tl.umap(scData_Recluster)


# plot UMAP, color by clustering results, and add labels
# Set global font settings for bold text

# Set figure size
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.figsize'] = [8, 8]

sc.settings.figdir = 'Visual/Sup/'

# Create the UMAP plot
fig, ax = plt.subplots(figsize=(10, 8))
sc.pl.umap(
    scData_Recluster,
    color=['leiden'],
    title='Leiden Clustering of scRNAseq data',
    size=25,  # Larger point size
    alpha=0.7,  # Adjust point transparency
    legend_loc='on data',  # Label on the plot
    legend_fontsize=18,  # Larger label font size
    legend_fontweight='bold',  # Adjust label font weight
    show=False,  # Don't show yet, we'll customize it
    ax=ax
)

# Customize the plot
ax.set_title('Leiden Clustering of scRNAseq data', fontsize=24, fontweight='bold', pad=20)

# Make tick labels larger and bold
ax.tick_params(axis='both', which='major', labelsize=16, width=3, length=6)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# Make axis labels larger and bold
ax.set_xlabel('UMAP1', fontsize=20, fontweight='bold')
ax.set_ylabel('UMAP2', fontsize=20, fontweight='bold')

# Make borders thicker and black
for spine in ax.spines.values():
    spine.set_linewidth(5)
    spine.set_color('black')

plt.tight_layout()
plt.savefig('Visual/Sup/Sup_scRNAseq-Leiden_UMAP.png', 
            dpi=600, bbox_inches='tight')
plt.show()



# Create the UMAP plot for Cell Type
fig, ax = plt.subplots(figsize=(10, 8))
sc.pl.umap(
    scData_Recluster,
    color=['Cell_Type'],
    title='Previous Cell Type Annotation of scRNAseq data',
    size=25,  # Larger point size
    alpha=0.7,  # Adjust point transparency
    legend_loc='on data',  # Label on the plot
    legend_fontsize=18,  # Larger label font size
    legend_fontweight='bold',  # Adjust label font weight
    show=False,  # Don't show yet, we'll customize it
    ax=ax
)

# Customize the plot
ax.set_title('Previous Cell Type Annotation of scRNAseq data', fontsize=24, fontweight='bold', pad=20)

# Make tick labels larger and bold
ax.tick_params(axis='both', which='major', labelsize=16, width=3, length=6)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# Make axis labels larger and bold
ax.set_xlabel('UMAP1', fontsize=20, fontweight='bold')
ax.set_ylabel('UMAP2', fontsize=20, fontweight='bold')

# Make borders thicker and black
for spine in ax.spines.values():
    spine.set_linewidth(5)
    spine.set_color('black')

plt.tight_layout()
plt.savefig('Visual/Sup/Sup_scRNAseq-Previous_Ann_UMAP.png', 
            dpi=600, bbox_inches='tight')
plt.show()







# plot UMAP, color by CD4 and CD8A gene expression

fig, ax = plt.subplots(figsize=(10, 8))
sc.pl.umap(
    scData_Recluster,
    color=['CD8A'],  # Use gene names to color the UMAP
    title='CD8A',
    size=25,  # Larger point size
    alpha=0.9,  # Adjust point transparency
    legend_loc='on data',  # Label on the plot
    legend_fontsize=18,  # Larger label font size
    legend_fontweight='bold',  # Adjust label font weight
    show=False,  # Don't show yet, we'll customize it
    ax=ax
)

# Customize the plot
ax.set_title('CD8A', fontsize=24, fontweight='bold', pad=20)

# Make tick labels larger and bold
ax.tick_params(axis='both', which='major', labelsize=16, width=3, length=6)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# Make axis labels larger and bold
ax.set_xlabel('UMAP1', fontsize=20, fontweight='bold')
ax.set_ylabel('UMAP2', fontsize=20, fontweight='bold')

# Make borders thicker and black
for spine in ax.spines.values():
    spine.set_linewidth(5)
    spine.set_color('black')

# Add colorbar title
cbar = plt.gcf().axes[-1]  # Get the colorbar
cbar.set_ylabel('Log-Normalized Expression', fontsize=20, fontweight='bold', color='black')
cbar.tick_params(labelsize=16, colors='black', width=3)
for label in cbar.get_yticklabels():
    label.set_fontweight('bold')

plt.tight_layout()
plt.savefig('Visual/Sup/Sup_scRNAseq-CD8A_UMAP.png', 
            dpi=600, bbox_inches='tight')
plt.show()


# create cross-tabulation tables

leiden_labels = scData_Recluster.obs['leiden']
previous_labels = scData_Recluster.obs['Cell_Type']
sample_id = scData_Recluster.obs['HTAN Specimen ID']

crosstab_ReCluster_VS_preLabeled = pd.crosstab(leiden_labels, previous_labels, rownames=['Leiden Cluster'], colnames=['Previous Label'])
crosstab_ReCluster_VS_sample = pd.crosstab(leiden_labels, sample_id, rownames=['Leiden Cluster'], colnames=['Sample ID']).iloc[[7,16,17,18, 3, 15, 10],]
kept_sample_id_list = crosstab_ReCluster_VS_sample.sum(axis=0).sort_values(ascending=False)[0:39].index.tolist()
crosstab_ReCluster_VS_sample = crosstab_ReCluster_VS_sample.loc[:,kept_sample_id_list]


### Visualize the cross-tabulation table for Re-clustering vs sample ID
yticklabels = [
    'Cluster 7 (Subtype of CSC)', 
    'Cluster 16 (Subtype of CSC)', 
    'Cluster 17 (Subtype of CSC)', 
    'Cluster 18 (Subtype of CSC)', 
    'Cluster 3 (Subtype of ASC)', 
    'Cluster 15 (Subtype of ASC)', 
    'Cluster 10 (SSC)'
]

plt.figure(figsize=(20, 8))
sns.heatmap(
    crosstab_ReCluster_VS_sample, 
    cmap='coolwarm', 
    annot=True, 
    fmt='d', 
    cbar_kws={'label': 'Count'},
    annot_kws={"size": 8, "weight": "bold"},
    linecolor='black',  # Set grid line color to black
    linewidths=0.5      # Set grid line width
)  
plt.yticks(ticks=np.arange(len(yticklabels))+0.5, labels=yticklabels, rotation=0, fontsize=9, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9, fontweight='bold')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.show()



plt.figure(figsize=(12, 10))
sns.heatmap(
    crosstab_ReCluster_VS_preLabeled, 
    cmap='coolwarm', 
    annot=True, 
    fmt='d', 
    cbar_kws={'label': 'Count'},
    annot_kws={"size": 12, "weight": "bold"},
    linecolor='black',  # Set grid line color to black
    linewidths=0.5      # Set grid line width
)  
plt.xticks(rotation=45, ha='right', fontsize=16, fontweight='bold')

# Set y-axis tick labels
plt.yticks(ticks=[i + 0.5 for i in range(20)], labels=[f'Cluster {i}' for i in range(20)], fontsize=16, fontweight='bold', rotation=0)

# Customize colorbar
cbar = plt.gcf().axes[-1]
cbar.set_ylabel('Count', fontsize=20, fontweight='bold', color='black')
cbar.tick_params(labelsize=16, colors='black', width=3)
for label in cbar.get_yticklabels():
    label.set_fontweight('bold')

plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.tight_layout()
plt.savefig('Visual/Sup/Sup_CrossTable_DoubleConfirm.png', dpi=600)
plt.show()






#### Make cells double-confirmed by previous annotation #########################################################

double_confirmed_df = pd.DataFrame(scData_Recluster.obs[['Cell_Type','leiden']])
double_confirmed_df["double_confirmed"] = ["Unknown"] * len(double_confirmed_df)

double_confirmed_df["double_confirmed"].loc[(double_confirmed_df["Cell_Type"] == "T") & (double_confirmed_df["leiden"] == "0")] = "CD8+ T"
double_confirmed_df["double_confirmed"].loc[(double_confirmed_df["Cell_Type"] == "T") & (double_confirmed_df["leiden"] == "3")] = "CD4+ T"
double_confirmed_df["double_confirmed"].loc[(double_confirmed_df["Cell_Type"] == "PLA") & (double_confirmed_df["leiden"] == "2")] = "PLA"
double_confirmed_df["double_confirmed"].loc[(double_confirmed_df["Cell_Type"] == "ASC") & (double_confirmed_df["leiden"] == "1")] = "ASC I"
double_confirmed_df["double_confirmed"].loc[(double_confirmed_df["Cell_Type"] == "ASC") & (double_confirmed_df["leiden"] == "4")] = "ASC II"
double_confirmed_df["double_confirmed"].loc[(double_confirmed_df["Cell_Type"] == "B") & (double_confirmed_df["leiden"] == "5")] = "B"
double_confirmed_df["double_confirmed"].loc[(double_confirmed_df["Cell_Type"] == "CSC") & (double_confirmed_df["leiden"] == "6")] = "CSC I"
double_confirmed_df["double_confirmed"].loc[(double_confirmed_df["Cell_Type"] == "MYE") & (double_confirmed_df["leiden"] == "7")] = "MYE"
double_confirmed_df["double_confirmed"].loc[(double_confirmed_df["Cell_Type"] == "FIB") & (double_confirmed_df["leiden"] == "8")] = "FIB"
double_confirmed_df["double_confirmed"].loc[(double_confirmed_df["Cell_Type"] == "SSC") & (double_confirmed_df["leiden"] == "9")] = "SSC I"
double_confirmed_df["double_confirmed"].loc[(double_confirmed_df["Cell_Type"] == "MAS") & (double_confirmed_df["leiden"] == "10")] = "MAS"
double_confirmed_df["double_confirmed"].loc[(double_confirmed_df["Cell_Type"] == "TUF") & (double_confirmed_df["leiden"] == "11")] = "TUF"
double_confirmed_df["double_confirmed"].loc[(double_confirmed_df["Cell_Type"] == "END") & (double_confirmed_df["leiden"] == "12")] = "END"
double_confirmed_df["double_confirmed"].loc[(double_confirmed_df["Cell_Type"] == "CT") & (double_confirmed_df["leiden"] == "13")] = "CT"
double_confirmed_df["double_confirmed"].loc[(double_confirmed_df["Cell_Type"] == "ASC") & (double_confirmed_df["leiden"] == "14")] = "ASC III"
double_confirmed_df["double_confirmed"].loc[(double_confirmed_df["Cell_Type"] == "CSC") & (double_confirmed_df["leiden"] == "15")] = "CSC II"
double_confirmed_df["double_confirmed"].loc[(double_confirmed_df["Cell_Type"] == "CSC") & (double_confirmed_df["leiden"] == "16")] = "CSC III"
double_confirmed_df["double_confirmed"].loc[(double_confirmed_df["Cell_Type"] == "CSC") & (double_confirmed_df["leiden"] == "17")] = "CSC IV"
double_confirmed_df["double_confirmed"].loc[(double_confirmed_df["Cell_Type"] == "EE") & (double_confirmed_df["leiden"] == "18")] = "EE"
double_confirmed_df["double_confirmed"].loc[(double_confirmed_df["Cell_Type"] == "ABS") & (double_confirmed_df["leiden"] == "19")] = "ABS"


double_confirmed_df["double_confirmed"].value_counts().sum()


double_confirmed_df = double_confirmed_df[double_confirmed_df["double_confirmed"] != "Unknown"]




X_scData_DoubleConfirmed = scData_Recluster_before_scale[scData_Recluster_before_scale.obs_names.isin(double_confirmed_df.index.tolist()), :].X
scData_DoubleConfirmed = scData_Recluster[scData_Recluster.obs_names.isin(double_confirmed_df.index.tolist()), :]
scData_DoubleConfirmed.X = X_scData_DoubleConfirmed
scData_DoubleConfirmed.obs["double_confirmed"] = double_confirmed_df["double_confirmed"]

scData_DoubleConfirmed.write_h5ad('scRNAseq_data/scData_DoubleConfirmed.h5ad')





######### Maker genes finding ##############################################################


scData_DoubleConfirmed = sc.read_h5ad('scRNAseq_data/scData_DoubleConfirmed.h5ad')

scData_DoubleConfirmed_Norm = scData_original[scData_DoubleConfirmed.obs_names.tolist(), :].copy()
scData_DoubleConfirmed_Norm.obs = scData_DoubleConfirmed.obs

InputDf_for_CARD_Original = pd.DataFrame(scData_DoubleConfirmed_Norm.X, index=scData_DoubleConfirmed_Norm.obs_names, columns=scData_DoubleConfirmed_Norm.var_names).T

InputDf_for_CARD_Original.to_csv('scRNAseq_data/InputDf_for_CARD_Original.csv')


sc.pp.normalize_total(scData_DoubleConfirmed_Norm, target_sum=1e4)

cell_types = scData_DoubleConfirmed_Norm.obs["double_confirmed"].unique().tolist()
len(cell_types)

def find_de_genes(Norm_adata, cell_type1_list, cell_type2_list):
    results = []
    
    
    mask1 = Norm_adata.obs["double_confirmed"].isin(cell_type1_list)
    mask2 = Norm_adata.obs["double_confirmed"].isin(cell_type2_list)

    # test each gene
    for i, gene in enumerate(Norm_adata.var_names):
        g1_exp_Norm = Norm_adata[mask1, gene].X.toarray().flatten()
        g2_exp_Norm = Norm_adata[mask2, gene].X.toarray().flatten()
        
        # Wilcoxon rank-sum test 
        statistic, pval = ranksums(g1_exp_Norm, g2_exp_Norm)
        
        # calculate the log-transformed mean of two groups
        mean1 = np.log2(np.mean(g1_exp_Norm +1))
        mean2 = np.log2(np.mean(g2_exp_Norm +1))
        
        # calculate the log2FC (from natural logarithm difference to log2 difference)
        log2fc = (mean1 - mean2)
        
        # calculate the percentage of cells expressing the gene
        pct = np.mean(g1_exp_Norm > 0) * 100
        
        results.append({
            'gene': gene,
            'log2fc': log2fc,
            'pval': pval,
            'pct': pct,    
            'statistic': statistic
        })
    
    results_df = pd.DataFrame(results)
    results_df['padj'] = multipletests(results_df['pval'], method='fdr_bh')[1]
    results_df = results_df[
        (results_df['padj'] < 0.05) & 
        (results_df['log2fc'] > 1) &
        (results_df['pct'] > 25)
    ].sort_values('log2fc', ascending=False).reset_index(drop=True)
    return results_df




finding_marker_genes_dict = {}
for cell_type in cell_types:
    
    if cell_type != "CD4+ T" and cell_type != "CD8+ T":
        cell_types_rest = cell_types.copy()
        cell_types_rest.remove(cell_type)
        finding_marker_genes_dict[cell_type] = find_de_genes(scData_DoubleConfirmed_Norm, [cell_type], cell_types_rest)
    elif cell_type == "CD4+ T":
        cell_types_rest = cell_types.copy()
        cell_types_rest.remove(cell_type)
        cell_types_rest.remove("CD8+ T")
        finding_marker_genes_dict[cell_type] = find_de_genes(scData_DoubleConfirmed_Norm, [cell_type], cell_types_rest)
    elif cell_type == "CD8+ T":
        cell_types_rest = cell_types.copy()
        cell_types_rest.remove(cell_type)
        cell_types_rest.remove("CD4+ T")
        finding_marker_genes_dict[cell_type] = find_de_genes(scData_DoubleConfirmed_Norm, [cell_type], cell_types_rest)
        
    print(cell_type)



finding_marker_genes_extended_dict = {}
for cell_type in ["ASC I", "ASC II", "ASC III", "CSC I", "CSC II", "CSC III", "CSC IV", "SSC I"]:
    cell_type_list1 = [cell_type]
    cell_type_list2 = ["ABS", "CT", "EE", "TUF"]
    finding_marker_genes_extended_dict[cell_type] = find_de_genes(scData_DoubleConfirmed_Norm, cell_type_list1, cell_type_list2)
    print(cell_type)

for cell_type in ["ABS", "CT", "EE", "TUF"]:
    cell_type_list1 = [cell_type]
    cell_type_list2 = ["ASC I", "ASC II", "ASC III", "CSC I", "CSC II", "CSC III", "CSC IV", "SSC I"]
    finding_marker_genes_extended_dict[cell_type] = find_de_genes(scData_DoubleConfirmed_Norm, cell_type_list1, cell_type_list2)
    print(cell_type)

finding_marker_genes_extended_dict["CD4+ T"] = find_de_genes(scData_DoubleConfirmed_Norm, ["CD4+ T"], ["CD8+ T"])
finding_marker_genes_extended_dict["CD8+ T"] = find_de_genes(scData_DoubleConfirmed_Norm, ["CD8+ T"], ["CD4+ T"])
finding_marker_genes_extended_dict["FIB"] = find_de_genes(scData_DoubleConfirmed_Norm, ["FIB"], ["END"])
finding_marker_genes_extended_dict["END"] = find_de_genes(scData_DoubleConfirmed_Norm, ["END"], ["FIB"])



final_marker_genes_dict = {}
for cell_type in cell_types:
    if cell_type in finding_marker_genes_extended_dict.keys():
        a= finding_marker_genes_dict[cell_type]["gene"].head(50).tolist()
        b= finding_marker_genes_extended_dict[cell_type]["gene"].head(40).tolist()
        final_marker_genes_dict[cell_type] = list(set(a + b))
    else:
        final_marker_genes_dict[cell_type] = finding_marker_genes_dict[cell_type]["gene"].head(50).tolist()


for key, value in finding_marker_genes_dict.items():
    print(key, len(value))

for key, value in finding_marker_genes_extended_dict.items():
    print(key, len(value))

for key, value in final_marker_genes_dict.items():
    print(key, len(value))


final_marker_genes_dict["CD4+ T"]
final_marker_genes_dict["CD8+ T"]



final_marker_genes_dict["Cancer"] = list(set(final_marker_genes_dict["ASC I"]).union(set(final_marker_genes_dict["ASC II"])).union(set(final_marker_genes_dict["ASC III"])).union(set(final_marker_genes_dict["CSC I"])).union(set(final_marker_genes_dict["CSC II"])).union(set(final_marker_genes_dict["CSC III"])).union(set(final_marker_genes_dict["CSC IV"])).union(set(final_marker_genes_dict["SSC I"])))
final_marker_genes_dict["Normal Epithelia"] = list(set(final_marker_genes_dict["ABS"]).union(set(final_marker_genes_dict["CT"])).union(set(final_marker_genes_dict["EE"])).union(set(final_marker_genes_dict["TUF"])))



final_marker_genes_list = []
for key, value in final_marker_genes_dict.items():
    final_marker_genes_list += value
final_marker_genes_list = list(set(final_marker_genes_list))



scData_input_for_CARD_SelectedGenes = scData_original[scData_DoubleConfirmed.obs_names.tolist(), final_marker_genes_list]
InputDf_for_CARD_SelectedGenes = pd.DataFrame(
    scData_input_for_CARD_SelectedGenes.X.T,  # Transpose to have genes as rows and cells as columns
    index=scData_input_for_CARD_SelectedGenes.var_names,  # Gene names as row index
    columns=scData_input_for_CARD_SelectedGenes.obs_names  # Cell barcodes as column names
)

InputDf_for_CARD_SelectedGenes.to_csv('scRNAseq_data/InputDf_for_CARD_SelectedGenes.csv')



InputDf_for_CARD_meta = pd.DataFrame({
    'Cell Type': scData_DoubleConfirmed.obs['double_confirmed'],
    'HTAN Specimen ID': scData_DoubleConfirmed.obs['HTAN Specimen ID']
}, index=scData_DoubleConfirmed.obs_names)


InputDf_for_CARD_meta.to_csv('scRNAseq_data/InputDf_for_CARD_meta.csv')








###### Figure 2 (Panel A) (Final Umap) ######
scData_DoubleConfirmed = sc.read_h5ad('scRNAseq_data/scData_DoubleConfirmed.h5ad')

# Extract UMAP coordinates and labels
umap_coords = scData_DoubleConfirmed.obsm['X_umap']
cell_types = scData_DoubleConfirmed.obs['double_confirmed']

# Create custom UMAP plot
plt.figure(figsize=(12, 12))  # Square
ax = plt.gca()

# Set colors for each cell type
unique_cell_types = cell_types.unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cell_types)))
color_dict = dict(zip(unique_cell_types, colors))

# Draw scatter plot
for cell_type in unique_cell_types:
    mask = cell_types == cell_type
    plt.scatter(umap_coords[mask, 0], umap_coords[mask, 1], 
                c=[color_dict[cell_type]], 
                label=cell_type, 
                s=15, 
                alpha=0.9)

# Add cell type labels to data points - enlarged bold text, no border
for cell_type in unique_cell_types:
    mask = cell_types == cell_type
    center_x = np.mean(umap_coords[mask, 0])
    center_y = np.mean(umap_coords[mask, 1])
    plt.text(center_x, center_y, cell_type, 
             fontsize=24, fontweight='bold', 
             ha='center', va='center')

# Set left and bottom borders as thick black lines, add arrows
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(5)
ax.spines['bottom'].set_linewidth(5)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

# Add arrows to coordinate axes - enlarged arrow heads
# X-axis arrow
ax.annotate('', xy=(1.04, 0), xytext=(1.0, 0),
           xycoords='axes fraction', textcoords='axes fraction',
           arrowprops=dict(arrowstyle='-|>', lw=5, color='black', 
                          mutation_scale=20))

# Y-axis arrow  
ax.annotate('', xy=(0, 1.04), xytext=(0, 1.0),
           xycoords='axes fraction', textcoords='axes fraction',
           arrowprops=dict(arrowstyle='-|>', lw=5, color='black',
                          mutation_scale=20))

# Remove axis titles and overall title
plt.xlabel("")
plt.ylabel("")
plt.title("")

# Remove axis ticks
ax.set_xticks([])
ax.set_yticks([])

# Adjust layout and save
plt.tight_layout()
plt.savefig('Visual/Sup_scRNAseq-DoubleConfrim_UMAP.png', dpi=600, bbox_inches='tight')
plt.close()




# Check the marker genes for some clusters
  ## Note: double confirmed data is not scaled 
scData_DoubleConfirmed = sc.read_h5ad('scRNAseq_data/scData_DoubleConfirmed.h5ad')
scData_DoubleConfirmed_raw = scData_original[scData_DoubleConfirmed.obs_names.tolist(), :]
scData_DoubleConfirmed_raw.obs = scData_DoubleConfirmed.obs
scData_DoubleConfirmed_raw.uns = scData_DoubleConfirmed.uns
scData_DoubleConfirmed_raw.obsm = scData_DoubleConfirmed.obsm
sc.pp.normalize_total(scData_DoubleConfirmed_raw, target_sum=1e4)
sc.pp.scale(scData_DoubleConfirmed_raw, max_value=10)



a=pd.DataFrame(scData_DoubleConfirmed_raw.X, index = scData_DoubleConfirmed_raw.obs_names, columns = scData_DoubleConfirmed_raw.var_names)
a["HIST1H3F"].sort_values(ascending=False).head(10)


marker_genes = ["HIST1H3F"]


# Plot UMAP, colored by multiple marker genes with specified color range
sc.pl.umap(
    scData_DoubleConfirmed_raw,
    color=marker_genes,
    title=[f'{gene}' for gene in marker_genes],
    size=11,  
    alpha=0.7,  
    legend_loc='on data',  
    legend_fontsize=12,  
    legend_fontweight='bold',  
    ncols=2,
    color_map='coolwarm',
    vmin=0,  
    vmax=2   
)




for key, value in final_marker_genes_dict.items():
    print(key, len(value))



### Figure 2 (Panel B) (Heatmap of selected marker genes)

InputDf_for_CARD_meta = pd.read_csv('scRNAseq_data/InputDf_for_CARD_meta.csv', index_col= 0)
InputDf_for_CARD_Original = pd.read_csv('scRNAseq_data/InputDf_for_CARD_Original.csv', index_col=0)
InputDf_for_CARD_Original = InputDf_for_CARD_Original.T

InputDf_for_CARD_LogNorm = np.log2(InputDf_for_CARD_Original.div(InputDf_for_CARD_Original.sum(axis=1), axis=0)*1e4 + 1)
InputDf_for_CARD_LogNorm = InputDf_for_CARD_LogNorm.loc[InputDf_for_CARD_meta.index, final_marker_genes_list]
InputDf_for_CARD_LogNorm_Z = (InputDf_for_CARD_LogNorm - InputDf_for_CARD_LogNorm.mean()) / InputDf_for_CARD_LogNorm.std(ddof=0)



cell_types = [
    "ASC I", "ASC II", "ASC III", "CSC I", "CSC II", "CSC III", "CSC IV",
    "SSC I",  "ABS", "CT", "EE", "TUF", "CD4+ T", "CD8+ T", "PLA", "B",
    "MAS", "MYE", "FIB", "END"
]

Number_of_marker_genes = []

for cell_type in cell_types:
    Number_of_marker_genes.append(len(final_marker_genes_dict[cell_type]))


cell_type_order = {cell: i for i, cell in enumerate(cell_types)}


InputDf_for_CARD_meta_sorted = InputDf_for_CARD_meta.copy()
InputDf_for_CARD_meta_sorted['sort_order'] = InputDf_for_CARD_meta['Cell Type'].map(cell_type_order)
InputDf_for_CARD_meta_sorted = InputDf_for_CARD_meta_sorted.sort_values('sort_order').drop('sort_order', axis=1)

ordered_genes = []
for cell_type in cell_types:
    if cell_type != "Cancer" and cell_type != "Normal Epithelia":
        ordered_genes += final_marker_genes_dict[cell_type]


marker_genes_heatmap_df = pd.concat([InputDf_for_CARD_LogNorm_Z.loc[InputDf_for_CARD_meta_sorted.index, ordered_genes], InputDf_for_CARD_meta_sorted['Cell Type']], axis=1)
marker_genes_heatmap_df_groupmean = marker_genes_heatmap_df.groupby('Cell Type').mean()
marker_genes_heatmap_df_groupmean_sorted = marker_genes_heatmap_df_groupmean.reindex(cell_type_order)



plt.figure(figsize=(20, 10))

# Set global font settings for bold and black text
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelcolor'] = 'black'

ax = sns.heatmap(
    marker_genes_heatmap_df_groupmean_sorted,
    cmap='coolwarm',
    linewidths=0.0,
    annot=False,
    cbar_kws={'label': 'Scaled Log-Normalized Gene Expression'},
    vmin = -1,
    vmax= 1,
    xticklabels=False  # Remove x-axis tick labels
)

# Add thick black border
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(5)
    spine.set_color('black')

# Set y-axis labels with larger, bold, black font
ax.set_yticklabels(marker_genes_heatmap_df_groupmean_sorted.index, 
                   rotation=0, fontsize=17, fontweight='bold', color='black')  # One size larger
ax.set_xlabel('')
ax.set_ylabel('')

# Make colorbar label bold and black 
cbar = ax.collections[0].colorbar
cbar.set_label('Scaled Log-Normalized Gene Expression', fontsize=20, fontweight='bold', color='black')  # Increased from 14 to 16
cbar.ax.tick_params(labelsize=16, colors='black', width=3) 
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_color('black')

# Add horizontal separators
for i in range(1, len(marker_genes_heatmap_df_groupmean_sorted.index)):
    ax.axhline(i, color='black', linewidth=1.5)

# Calculate the position of vertical separators
marker_counts = Number_of_marker_genes
line_positions = []
cumulative_pos = 0

for count in marker_counts:
    cumulative_pos += count
    line_positions.append(cumulative_pos)

# Add vertical separators with thicker lines
for pos in line_positions[:-1]:  # The last position does not need a separator
    ax.axvline(x=pos, color='black', linewidth=1.5)

# Calculate the position of each cell type label (the center position of each group of genes)
label_positions = []
start_pos = 0
for count in marker_counts:
    center_pos = start_pos + count/2
    label_positions.append(center_pos)
    start_pos += count

# Add x-axis labels with larger, bold, black font
plt.xticks(label_positions, cell_types, rotation=45, fontsize=17, fontweight='bold', color='black')  # One size larger

plt.tight_layout()
# Adjust subplot position to shift chart to the right
plt.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.15)
plt.savefig('Visual/Heatmap_scRNAseq_marker_genes.png', dpi=600, bbox_inches='tight')
plt.close()


 ### Grouped heatmap
Number_of_marker_genes_grouped = [379,257,45,46,40,40,40,40,65,70]

cell_types_grouped = [
    "Cancer", "Normal Epithelia", "CD4+ T", "CD8+ T", "PLA", "B",
    "MAS", "MYE", "FIB", "END"
]

cell_type_order_grouped = {cell: i for i, cell in enumerate(cell_types_grouped)}


cell_types_mapping = {
    "ASC I": "Cancer",
    "ASC II": "Cancer",
    "ASC III": "Cancer",
    "CSC I": "Cancer",
    "CSC II": "Cancer",
    "CSC III": "Cancer",
    "CSC IV": "Cancer",
    "CSC V": "Cancer",
    "SSC": "Cancer",
    "NE Mixture": "Normal Epithelia",
    "CT": "Normal Epithelia",
    "EE": "Normal Epithelia",
    "GOB": "Normal Epithelia",
    "TUF": "Normal Epithelia",
    "CD4+ T": "CD4+ T",
    "CD8+ T": "CD8+ T",
    "PLA": "PLA",
    "B": "B",
    "MAS": "MAS",
    "MYE": "MYE",
    "FIB": "FIB",
    "END": "END"
}



InputDf_for_CARD_meta_grouped_sorted = InputDf_for_CARD_meta.copy()
InputDf_for_CARD_meta_grouped_sorted['Cell Type'] = InputDf_for_CARD_meta['Cell Type'].map(cell_types_mapping)
InputDf_for_CARD_meta_grouped_sorted['sort_order'] = InputDf_for_CARD_meta_grouped_sorted['Cell Type'].map(cell_type_order_grouped)
InputDf_for_CARD_meta_grouped_sorted = InputDf_for_CARD_meta_grouped_sorted.sort_values('sort_order').drop('sort_order', axis=1)


ordered_genes_grouped = []
for cell_type in cell_types_grouped:
        ordered_genes_grouped += final_marker_genes_dict[cell_type]


marker_genes_heatmap_df_grouped = pd.concat([InputDf_for_CARD_LogNorm_Z.loc[InputDf_for_CARD_meta_grouped_sorted.index, ordered_genes_grouped], InputDf_for_CARD_meta_grouped_sorted['Cell Type']], axis=1)
marker_genes_heatmap_df_groupmean_grouped = marker_genes_heatmap_df_grouped.groupby('Cell Type').mean()
marker_genes_heatmap_df_groupmean_sorted_grouped = marker_genes_heatmap_df_groupmean_grouped.reindex(cell_type_order_grouped)




plt.figure(figsize=(20, 10))
ax = sns.heatmap(
    marker_genes_heatmap_df_groupmean_sorted_grouped,
    cmap='coolwarm',
    linewidths=0.0,
    annot=False,
    cbar_kws={'label': 'Scaled Log-Normalized Gene Expression'},
    vmin = -1,
    vmax= 1,
    xticklabels=False  # Remove x-axis tick labels
)

# Set y-axis labels
ax.set_yticklabels(marker_genes_heatmap_df_groupmean_sorted_grouped.index, 
                   rotation=0, fontsize=10, fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('')

# Add horizontal separators
for i in range(1, len(marker_genes_heatmap_df_groupmean_sorted_grouped.index)):
    ax.axhline(i, color='black', linewidth=1.2)

# Calculate the position of vertical separators
marker_counts = Number_of_marker_genes_grouped
line_positions = []
cumulative_pos = 0

for count in marker_counts:
    cumulative_pos += count
    line_positions.append(cumulative_pos)

# Add red vertical separators
for pos in line_positions[:-1]:  # The last position does not need a separator
    ax.axvline(x=pos, color='black', linewidth=1.2)

# Calculate the position of each cell type label (the center position of each group of genes)
label_positions = []
start_pos = 0
for count in marker_counts:
    center_pos = start_pos + count/2
    label_positions.append(center_pos)
    start_pos += count

# Add x-axis labels
plt.xticks(label_positions, cell_types_grouped, rotation=45, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()



### sampled 50 single cells heatmap


marker_genes_heatmap_df_sampled100 = marker_genes_heatmap_df.groupby('Cell Type').apply(lambda x: x.sample(n=100, random_state=12345)).reset_index(drop=True)
marker_genes_heatmap_df_sampled100['Cell Type'] = pd.Categorical(
    marker_genes_heatmap_df_sampled100['Cell Type'], 
    categories=cell_types, 
    ordered=True
)

marker_genes_heatmap_df_sampled100_sorted = marker_genes_heatmap_df_sampled100.sort_values('Cell Type')


marker_genes_heatmap_df_sampled100_regrouped = marker_genes_heatmap_df_sampled100_sorted.copy()


plt.figure(figsize=(20, 10))
ax = sns.heatmap(
    marker_genes_heatmap_df_sampled100_sorted.drop(columns='Cell Type'),
    cmap='coolwarm',
    linewidths=0.0,
    annot=False,
    cbar_kws={'label': 'Scaled Log-Normalized Expression'},
    vmin=-1,
    vmax=1,
    xticklabels=False
)

# Calculate the position of y-axis tick labels and labels
cell_types_labels = marker_genes_heatmap_df_sampled100_sorted['Cell Type'].values
y_ticks = np.arange(50, len(cell_types_labels), 100)  # From 25, every 50 positions
y_labels = [cell_types_labels[i] for i in y_ticks]   # Get the labels at the corresponding positions

# Set y-axis tick labels and labels
plt.yticks(ticks=y_ticks, labels=y_labels, fontsize=10, fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('')

# Add horizontal separators
for i in range(0, len(marker_genes_heatmap_df_sampled100_sorted.index), 100):
    ax.axhline(i, color='black', linewidth=0.7)


# Calculate the position of vertical separators
marker_counts = Number_of_marker_genes
line_positions = []
cumulative_pos = 0

for count in marker_counts:
    cumulative_pos += count
    line_positions.append(cumulative_pos)

# 
for pos in line_positions[:-1]:  # The last position does not need a separator
    ax.axvline(x=pos, color='black', linewidth=1.2)

# Calculate the position of each cell type label (the center position of each group of genes)
label_positions = []
start_pos = 0
for count in marker_counts:
    center_pos = start_pos + count/2
    label_positions.append(center_pos)
    start_pos += count

# Add x-axis labels
plt.xticks(label_positions, cell_types, rotation=45, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()


### Grouped sampled 50 single cell heatmap


marker_genes_heatmap_df_sampled100_grouped= marker_genes_heatmap_df_grouped.groupby('Cell Type').apply(lambda x: x.sample(n=100, random_state=12345)).reset_index(drop=True)
marker_genes_heatmap_df_sampled100_grouped['Cell Type'] = pd.Categorical(
    marker_genes_heatmap_df_sampled100_grouped['Cell Type'], 
    categories=cell_types_grouped, 
    ordered=True
)

marker_genes_heatmap_df_sampled100_grouped_sorted = marker_genes_heatmap_df_sampled100_grouped.sort_values('Cell Type')


marker_genes_heatmap_df_sampled100_grouped_regrouped = marker_genes_heatmap_df_sampled100_grouped_sorted.copy()


plt.figure(figsize=(20, 10))
ax = sns.heatmap(
    marker_genes_heatmap_df_sampled100_grouped_sorted.drop(columns='Cell Type'),
    cmap='coolwarm',
    linewidths=0.0,
    annot=False,
    cbar_kws={'label': 'Scaled Log-Normalized Expression'},
    vmin=-1,
    vmax=1,
    xticklabels=False
)

# Calculate the position of y-axis tick labels and labels
cell_types_labels = marker_genes_heatmap_df_sampled100_grouped_sorted['Cell Type'].values
y_ticks = np.arange(50, len(cell_types_labels), 100)  # From 50, every 100 positions
y_labels = [cell_types_labels[i] for i in y_ticks]   # Get the labels at the corresponding positions

# Set y-axis tick labels and labels
plt.yticks(ticks=y_ticks, labels=y_labels, fontsize=10, fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('')

# Add horizontal separators
for i in range(0, len(marker_genes_heatmap_df_sampled100_grouped_sorted.index), 100):
    ax.axhline(i, color='black', linewidth=0.7)


# Calculate the position of vertical separators
marker_counts = Number_of_marker_genes_grouped
line_positions = []
cumulative_pos = 0

for count in marker_counts:
    cumulative_pos += count
    line_positions.append(cumulative_pos)

# 
for pos in line_positions[:-1]:  # The last position does not need a separator
    ax.axvline(x=pos, color='black', linewidth=1.2)

# Calculate the position of each cell type label (the center position of each group of genes)
label_positions = []
start_pos = 0
for count in marker_counts:
    center_pos = start_pos + count/2
    label_positions.append(center_pos)
    start_pos += count

# Add x-axis labels
plt.xticks(label_positions, cell_types_grouped, rotation=45, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()









# Save the final marker genes dictionary to a JSON file
with open('data/scRNAseq_data/final_marker_genes_dict.json', 'w') as f:
    json.dump(final_marker_genes_dict, f)

with open('data/scRNAseq_data/final_marker_genes_dict.json', 'r') as f:
    final_marker_genes_dict = json.load(f)



#### Prepare data for deconvolution #########################################################

InputDf_for_CARD_Original = pd.read_csv('scRNAseq_data/InputDf_for_CARD_Original.csv', index_col=0)
InputDf_for_CARD_SelectedGenes = pd.read_csv('scRNAseq_data/InputDf_for_CARD_SelectedGenes.csv', index_col=0)
InputDf_for_CARD_meta = pd.read_csv('scRNAseq_data/InputDf_for_CARD_meta.csv', index_col=0)







### Figure 2 (C)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


Celltype_proportion_df_mean_Cody = pd.read_csv('/CARD_Results_Files/Celltype_proportion_df_mean_Cody.csv', index_col=0)
Celltype_proportion_df_mean_FH = pd.read_csv('/FredHutch_Colorectal/CARD_Results_Files/Celltype_proportion_df_mean_FH.csv', index_col=0)
Celltype_proportion_df_mean_HEST = pd.read_csv('/hest_data/CARD_Results_Files/Celltype_proportion_df_mean_HEST.csv', index_col=0)

# Fix the total dataframe creation - merge all proportion columns while keeping Cell Type
# First get the Cell Type column from any of the dataframes (they should all be the same)
cell_types_df = Celltype_proportion_df_mean_Cody[['Cell Type']].copy()

# Get only the proportion columns from each dataframe (excluding Cell Type)
cody_props = Celltype_proportion_df_mean_Cody.drop('Cell Type', axis=1)
fh_props = Celltype_proportion_df_mean_FH.drop('Cell Type', axis=1)
hest_props = Celltype_proportion_df_mean_HEST.drop('Cell Type', axis=1)

# Combine all proportion columns
all_props = pd.concat([cody_props, fh_props, hest_props], axis=1)

# Add Cell Type back as the first column
Celltype_proportion_df_mean_total = pd.concat([cell_types_df, all_props], axis=1)

Celltype_proportion_df_mean_total.to_csv('Supplementary_File1_CellType_Proportion_Mean.csv')


def CARD_results_vis(df = Celltype_proportion_df_mean_total, group = True):
    
    df = df.copy()


    if group == True:
        category_mapping = {'ASC I': 'Tumor',
                            'ASC II': 'Tumor',
                            'ASC III': 'Tumor',
                            'CSC I': 'Tumor',
                            'CSC II': 'Tumor',
                            'CSC III': 'Tumor',
                            'CSC IV': 'Tumor',
                            'SSC I': 'Tumor',
                            'CT': 'Normal Epithelial',
                            'EE': 'Normal Epithelial',
                            'TUF': 'Normal Epithelial',
                            'ABS': 'Normal Epithelial',
                            'T': 'T',
                            'B': 'pan-APC',
                            'PLA': 'pan-APC',
                            'MAS': 'pan-APC',
                            'MYE': 'pan-APC',
                            'FIB': 'Stromal',
                            'END': 'Stromal'}
        df['Category'] = df['Cell Type'].map(category_mapping)
        grouped_df = df.groupby('Category').sum().reset_index()
        grouped_df = grouped_df.drop(grouped_df.columns[1], axis=1)
        df = grouped_df.copy()
        df.columns.values[0] = 'Cell Type'




    melted_df = df.melt(id_vars='Cell Type', var_name='Sample', value_name='Proportion')
    mean_proportions = melted_df.groupby('Cell Type')['Proportion'].mean().reset_index()
    std_proportions = melted_df.groupby('Cell Type')['Proportion'].std().reset_index()
    # Merge mean and std
    mean_proportions['std'] = std_proportions['Proportion']


    if group == True:
        cell_type_order = ["Tumor", "Normal Epithelial", "T", "pan-APC", "Stromal"]
    else:
        cell_type_order = ["ASC I", "ASC II", "ASC III", "CSC I", "CSC II", "CSC III", "CSC IV", "SSC I", "ABS", "CT", "EE", "TUF", "T", "B", "MAS", "MYE", "PLA", "FIB", "END"]


    # Convert the 'Cell Type' column to a categorical type and set the order
    melted_df['Cell Type'] = pd.Categorical(melted_df['Cell Type'], categories=cell_type_order, ordered=True)
    mean_proportions['Cell Type'] = pd.Categorical(mean_proportions['Cell Type'], categories=cell_type_order, ordered=True)
    mean_proportions = mean_proportions.sort_values('Cell Type')



    # Define color palette
    # Set a better color palette - original cell types (22 categories)
    if group == False:
        palette = {
        # Tumor cells - red
        'ASC I': '#E41A1C',        # light red
        'ASC II': '#F43C37',       # light red
        'ASC III': '#F7675F',      # light red
        'CSC I': '#A50F15',        # dark red
        'CSC II': '#CB181D',       # dark red
        'CSC III': '#EF3B2C',      # orange red
        'CSC IV': '#FB6A4A',       # coral red
        'SSC I': '#FCBBA1',          # light pink
        
        # Normal Epithelia - blue
        'ABS': '#377EB8',   # blue
        'CT': '#4292C6',           # medium blue
        'EE': '#6BAED6',           # light blue
        'GOB': '#9ECAE1',          # light blue
        'TUF': '#C6DBEF',          # very light blue
        
        # Immune cells - green and other colors
        'T': '#4DAF4A',       # green
        'PLA': '#FF7F00',          # orange
        'MAS': '#FFFF33',          # yellow
        'MYE': '#A65628',          # brown
        'B': '#F781BF',            # pink
        
        # Stromal cells - gray and green
        'FIB': '#999999',          # gray
        'END': '#66C2A5'           # green
    }
    elif group == True:
        palette= {
                'Tumor': '#E41A1C',               # red
                'Normal Epithelial': '#377EB8',    # blue
                'T': '#4DAF4A',              # green
                'pan-APC': '#FF7F00',                 # orange
                'Stromal': '#FFFF33',                 # yellow
                  }



    # Create bar plot with error bars and scatter points
    plt.figure(figsize=(10, 6))  # reduce figure width to make bars closer

    # define bar width for consistent spacing
    if group == True:
        bar_width = 0.4
    else:
        bar_width = 0.6
    
    # draw the bar plot
    bars = sns.barplot(
        data=mean_proportions,
        x='Cell Type',
        y='Proportion',
        palette=palette,
        edgecolor='black',  # set the rectangle border to black
        linewidth=6,  # the width of the rectangle border - bold
        errorbar=None,  # disable the default error bar
        width=bar_width  # make bars narrower
    )


    # draw the error bar on each bar
    for i, (mean, std) in enumerate(zip(mean_proportions['Proportion'], mean_proportions['std'])):
        # draw the error bar
        plt.plot(
            [i, i],  # x 
            [mean, mean + 0.5*std],  # y 
            color='black',  # the color of the error bar
            linewidth=5  # the width of the error bar - same thickness as border
        )
        # draw the error bar cap with width matching bar width
        cap_half_width = bar_width / 2
        plt.plot(
            [i - cap_half_width, i + cap_half_width],  # x coordinates matching bar width
            [mean + 0.5*std, mean + 0.5*std],  # y 
            color='black',  # the color of the error bar cap
            linewidth=5  # the width of the error bar cap - same thickness as border
        )


    # draw the scatter plot on the bar plot
    if group == True:
        dot_size = 10
    else:
        dot_size = 5


    sns.stripplot(
        data=melted_df,
        x='Cell Type',
        y='Proportion',
        color='black',  # the color of the scatter plot
        jitter= True,  # enable the jitter
        size= dot_size,  # the size of the scatter plot
        alpha=0.7  # opaque, same as scatter plot
    )



    # add the horizontal dashed line - grid lines same as scatter plot
    plt.grid(True, axis='y', linestyle='--', linewidth=4, color='gray', alpha=0.7)


    # set the style of the coordinate axis
    if group == True:
        plt.gca().xaxis.set_tick_params(labelsize=24, width=3, rotation=25)  # increased rotation angle
    else:
        plt.gca().xaxis.set_tick_params(labelsize=22, width=3, rotation=45)  # increase font size
    plt.gca().yaxis.set_tick_params(labelsize=20, width=3)  # increase font size
    plt.gca().tick_params(axis='both', which='both', color='black', direction='out')  # set the style of the tick
    
    # Make x-axis labels bold and black
    for label in plt.gca().get_xticklabels():
        label.set_fontweight('bold')
        label.set_color('black')
    
    # Make y-axis labels bold and black
    for label in plt.gca().get_yticklabels():
        label.set_fontweight('bold')
        label.set_color('black')
    
    # Set left, bottom, and right borders as thick black lines
    ax = plt.gca()
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

    # remove the horizontal coordinate axis title
    plt.xlabel(None)

    # remove the vertical coordinate axis title
    plt.ylabel(None)
    # remove the title
    plt.title(None)

    # show the figure
    plt.tight_layout()
    if group == True:
        plt.savefig('/Visual/CARD_Results_Vis_Combined_Grouped.png', dpi=600)
    else:
        plt.savefig('Visual/CARD_Results_Vis_Combined_Ungrouped.png', dpi=600)
    plt.close()



CARD_results_vis(df = Celltype_proportion_df_mean_total, group = True)
CARD_results_vis(df = Celltype_proportion_df_mean_total, group = False)





### Figure 2 (Panel D)
## For the visualization code, please refer to:
## 1. STpath_COAD/Workflow/StepB_Cell_Type_Deconvolution/CARD_Results_Validation_Cody.py
## 2. STpath_COAD/Workflow/StepB_Cell_Type_Deconvolution/CARD_Results_Validation_FH.py
## 3. STpath_COAD/Workflow/StepB_Cell_Type_Deconvolution/CARD_Results_Validation_HEST.py



### Figure 2 (Panel E)


### Marker genes refinement
with open('data/scRNAseq_data/final_marker_genes_dict.json', 'r') as f:
    final_marker_genes_dict = json.load(f)

del final_marker_genes_dict["Cancer"]
del final_marker_genes_dict["Normal Epithelia"]



InputDf_for_CARD_SelectedGenes = pd.read_csv('scRNAseq_data/InputDf_for_CARD_SelectedGenes.csv', index_col=0)
InputDf_for_CARD_meta = pd.read_csv('scRNAseq_data/InputDf_for_CARD_meta.csv', index_col=0)
InputDf_for_CARD_meta["Cell Type"][InputDf_for_CARD_meta["Cell Type"] == "CD4+ T"] = "T"
InputDf_for_CARD_meta["Cell Type"][InputDf_for_CARD_meta["Cell Type"] == "CD8+ T"] = "T"

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




cell_type = "Cancer Cells"
Region_ID = "6723_KL_1_region0"
expression_df = pd.read_csv(f'CARD_Need_Files/{Region_ID}_expression.csv', index_col=0)
Celltype_proportion_df = pd.read_csv(f'CARD_Results_Regions/{Region_ID}_celltype_proportion_modified.csv', index_col=0)
B_matrix_df = pd.read_csv(f'data/CARD_Results_Regions/{Region_ID}_B_Matrix_modified.csv', index_col=0)
        
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

# Remove marker genes not in B_matrix_df
marker_genes_clean_dict = {}
for key, value in final_ranked_gene_dict.items():
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



Spatial_location_df = pd.read_csv('CARD_Need_Files/6723_KL_1_region0_spatial.csv', index_col=0)
showcase_df = pd.concat([pd.Series(focus_celltype_percent), pd.Series(focus_relative_marker_genes_expression), Spatial_location_df[["x", "y"]].reset_index(drop=True)], axis=1)
showcase_df.columns = ["Proportion", "Expression", "x", "y"]


plt.figure(figsize=(10, 8))
ax = plt.gca()

# create the scatter plot with hexagon markers
plt.scatter(
    x=showcase_df['x'],
    y=showcase_df['y'],
    c=showcase_df['Proportion'],
    cmap='coolwarm',  # the color from light to dark
    marker='h',  # hexagon marker
    s=100,  # marker size
    edgecolors='black',
    linewidth=0.5
)

# Add colorbar
cbar = plt.colorbar()
cbar.set_label('Relative Markers GE (%)', fontsize=25, fontweight='bold', color='black')
cbar.ax.tick_params(labelsize=18, colors='black', width=3)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_color('black')

# Invert the y axis to match the image coordinate system
ax.invert_yaxis()

# Set all four borders as thick black lines
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

# Remove title and axis labels
plt.title('')
plt.xlabel('')
plt.ylabel('')

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(f"Visual/6723_KL_1_region0_Cancer_cells_proportion.png", dpi=600, bbox_inches='tight')
# Show the figure
plt.show()


# Set the figure size
plt.figure(figsize=(10, 8))
ax = plt.gca()

# Create the scatter plot with hexagon markers
plt.scatter(
    x=showcase_df['x'],
    y=showcase_df['y'],
    c=showcase_df['Expression'],
    cmap='coolwarm',  # the color from light to dark
    marker='h',  # hexagon marker
    s=100,  # marker size
    edgecolors='black',
    linewidth=0.5
)

# Add colorbar
cbar = plt.colorbar()
cbar.set_label('Relative Marker Gene Expression (%)', fontsize=25, fontweight='bold', color='black')
cbar.ax.tick_params(labelsize=18, colors='black', width=3)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_color('black')

# Invert the y axis to match the image coordinate system
ax.invert_yaxis()

# Set all four borders as thick black lines
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

# Remove title and axis labels
plt.title('')
plt.xlabel('')
plt.ylabel('')

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])

plt.savefig(f"Visual/6723_KL_1_region0_Cancer_marker_genes_relative_expression.png", dpi=600, bbox_inches='tight')
plt.show()



