#
# STPath-COAD: CARD Deconvolution Analysis
# ========================================
#
# This script performs cell type deconvolution using CARD (Conditional Autoregressive-based 
# Deconvolution) method on spatial transcriptomics data.
#
# Author: Saishi Cui
# Date: Sept 2025
#
# Purpose: Estimate cell type proportions in spatial transcriptomics data using CARD
# deconvolution method with single-cell RNA-seq reference data.
#

packages <- c("car", "caret", "concaveman", "cowplot", "cvms", "data.table", "dplyr", "ggbeeswarm", "ggcorrplot", "ggplot2", "ggpointdensity", "gprofiler2", "gtable", "grid", "gridExtra", "knitr", "likert", "Matrix", "MCMCpack", "multiROC", "patchwork", "RANN", "RColorBrewer", "readr", "reshape2", "scales", "sf", "sp", "stringr", "styler", "tidyverse", "viridis", "yarrr")

install_if_missing <- function(p) {
  if (!p %in% installed.packages()) {
    install.packages(p)
  }
}


invisible(sapply(packages, install_if_missing))

BiocManager::install("SingleCellExperiment")
BiocManager::install("SummarizedExperiment")
BiocManager::install("TOAST")
devtools::install_github('xuranw/MuSiC')
devtools::install_github('YMa-lab/CARD')
BiocManager::install("biomaRt")
BiocManager::install("org.Hs.eg.db")
BiocManager::install("EnsDb.Hsapiens.v79")
remotes::install_github("satijalab/seurat", "seurat5", quiet = TRUE)


Sys.setenv(FC = "/opt/homebrew/bin/gfortran")
Sys.setenv(F77 = "/opt/homebrew/bin/gfortran")
Sys.setenv(FLIBS = "-L/opt/homebrew/lib")
Sys.getenv("FC")
Sys.getenv("F77")


############

# Load necessary libraries
library(ggplot2)
library(data.table)
library(MuSiC)
library(stringr)
library(Matrix)
library(readr)
library(tidyverse)
library(biomaRt)
library(gprofiler2)
library(org.Hs.eg.db)
library(EnsDb.Hsapiens.v79)
library(dplyr)
library(grid)


remove.packages("CARD")
# 2. Check if the package has been uninstalled
"CARD" %in% installed.packages()[,"Package"]  # If it returns FALSE, it means it has been uninstalled successfully

library(devtools)
install_local("~/CARD", force = TRUE) ## Install the modified CARD package
library(CARD)
?createCARDObject

## Load the scRNAseq data
sc_count_SelectedGenes <- read.csv('data/scRNAseq_data/InputDf_for_CARD_SelectedGenes.csv', header = TRUE, sep = ",")
rownames(sc_count_SelectedGenes) <- sc_count_SelectedGenes[, 1]
sc_count_SelectedGenes <- sc_count_SelectedGenes[, -1]
sc_count_SelectedGenes <- as(sc_count_SelectedGenes, "sparseMatrix")
sc_count_SelectedGenes <- as(sc_count_SelectedGenes, "CsparseMatrix")



## Load the scRNAseq metadata
sc_meta <- read.csv('data/scRNAseq_data/InputDf_for_CARD_meta.csv', header = TRUE, sep = ",")
sc_meta$X <- gsub("-", ".", sc_meta$X)
rownames(sc_meta) <- sc_meta[, 1]
sc_meta <- sc_meta[, -1]
colnames(sc_meta)

sc_meta$Cell.Type[sc_meta$Cell.Type == "CD4+ T"] = "T"
sc_meta$Cell.Type[sc_meta$Cell.Type == "CD8+ T"] = "T"







## Run deconvolution


run_deconvolution <- function(expression_file, spatial_file, sc_count, sc_meta, sample_name) {
  
  st_count  <- read.csv(expression_file, header = TRUE, sep = ",")
  rownames(st_count) <- st_count[, 1]
  st_count <- st_count[, -1]
  st_count <- as(st_count, "sparseMatrix")
  st_count <- as(st_count, "CsparseMatrix")
  st_count <- t(st_count)


  spatial_location <- read.csv(spatial_file, header = TRUE, sep = ",", row.names = 1)
  colnames(spatial_location) = c("x", "y")
  

  CARD_obj <- createCARDObject(
    sc_count = sc_count,
    sc_meta = sc_meta,
    spatial_count = st_count,
    spatial_location = spatial_location,
    ct.varname = "Cell.Type",
    ct.select = unique(sc_meta$Cell.Type),
    sample.varname = sample_name,
    minCountGene = 100,
    minCountSpot = 5
  )

  gc()
  
  # Run deconvolution
  CARD_obj <- CARD_deconvolution(CARD_object = CARD_obj)

  # Save the CARD object

  CARD_obj_file_name <- paste0("data/CARD_Results_Regions/", sub("_expression$", "", tools::file_path_sans_ext(basename(expression_file))), "_cardobj_modified.rds")
  Celltype_file_name <- paste0("data/CARD_Results_Regions/", sub("_expression$", "", tools::file_path_sans_ext(basename(expression_file))), "_celltype_proportion_modified.csv")
  B_Matrix_file_name <- paste0("data/CARD_Results_Regions/", sub("_expression$", "", tools::file_path_sans_ext(basename(expression_file))), "_B_Matrix_modified.csv")
  Basis_file_name <- paste0("data/CARD_Results_Regions/", sub("_expression$", "", tools::file_path_sans_ext(basename(expression_file))), "_Basis_modified.csv")


  df_celltype <- round(CARD_obj@Proportion_CARD,4)
  B_Matrix_df <- data.frame(as.matrix(CARD_obj@algorithm_matrix$B))
  Basis_df <- data.frame(as.matrix(CARD_obj@algorithm_matrix$Basis))
  write.csv(df_celltype, file = Celltype_file_name)
  write.csv(B_Matrix_df, file = B_Matrix_file_name)
  write.csv(Basis_df, file = Basis_file_name)
  saveRDS(CARD_obj, file = CARD_obj_file_name)

}




# Loop through each file
expression_files <- list.files("data/CARD_Need_Files", pattern = "_expression.csv", full.names = TRUE)
spatial_files <- list.files("data/CARD_Need_Files", pattern = "_spatial.csv", full.names = TRUE)
length(expression_files)
for (i in seq_along(expression_files)) {
  run_deconvolution(expression_files[i], spatial_files[i], sc_count_SelectedGenes, sc_meta, sample_name = NULL)
  print(i)
}


