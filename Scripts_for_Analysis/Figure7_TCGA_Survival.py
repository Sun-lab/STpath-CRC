#!/usr/bin/env python3
"""
STPath-COAD: TCGA Survival Analysis for Figure 7
===============================================

This script performs survival analysis using TCGA data and STPath predictions.
Correlates STPath cell type predictions with clinical outcomes and survival data.

Author: Saishi Cui
Date: Sept 2025

Purpose: Correlate STPath-COAD predictions with clinical outcomes and survival data
from TCGA colorectal cancer patients. Generate survival analysis plots and statistics
for Figure 7 of the STPath-COAD manuscript.
"""

import pandas as pd

### Read TCGA Clinical Data Resource (CDR) supplemental table
print("Reading TCGA CDR supplemental table...")
TCGA_CDR_file_path = "Colorectal_Cancer_HE_patches/TCGA-CDR-SupplementalTableS1.xlsx"

# Read the first sheet of the Excel file with first column as index
TCGA_CDR_df = pd.read_excel(TCGA_CDR_file_path, sheet_name=0, index_col=0)

### Read TCGA Features Complete with Clinical data
TCGA_features_file_path = "Colorectal_Cancer_HE_patches/TCGA_Features_Complete_WithClinical.csv"

# Read the CSV file
TCGA_features_df = pd.read_csv(TCGA_features_file_path)

### Left merge TCGA_features_df with selected columns from TCGA_CDR_df
# Select specific columns from TCGA_CDR_df for merging
cdr_columns_to_merge = ['bcr_patient_barcode', 'age_at_initial_pathologic_diagnosis', 'gender', 
                        'ajcc_pathologic_tumor_stage', 'OS', 'OS.time', 'PFI', 'PFI.time']

TCGA_CDR_selected = TCGA_CDR_df[cdr_columns_to_merge].copy()

# Perform left merge: keep all rows from TCGA_features_df
merged_df = TCGA_features_df.merge(
    TCGA_CDR_selected, 
    left_on='Patient_ID', 
    right_on='bcr_patient_barcode', 
    how='left'
)

# Drop the redundant bcr_patient_barcode column (since we have Patient_ID)
merged_df.dropna(inplace=True)
merged_df.drop('bcr_patient_barcode', axis=1, inplace=True)
merged_df.drop(["age_at_initial_pathologic_diagnosis", "gender"], axis=1, inplace=True)

merged_df.to_csv("Colorectal_Cancer_HE_patches/TCGA_Features_Complete_WithClinical_CDR.csv", index=False)



### Read TCGA Features Complete with Clinical data
TCGA_merged_features_file_path = "Colorectal_Cancer_HE_patches/TCGA_Features_Complete_WithClinical_CDR.csv"

# Read the CSV file
TCGA_merged_features_df = pd.read_csv(TCGA_merged_features_file_path)

TCGA_merged_features_df["ajcc_pathologic_tumor_stage"].value_counts()

# Remove [Discrepancy] and [Not Available] entries

TCGA_merged_features_df = TCGA_merged_features_df[
    ~TCGA_merged_features_df['ajcc_pathologic_tumor_stage'].isin(['[Discrepancy]', '[Not Available]'])
].copy()
print(f"After removing discrepancy/missing: {len(TCGA_merged_features_df)}")

# Group stages into 4 main categories
def group_stage(stage):
    if pd.isna(stage):
        return None
    stage = str(stage).upper()
    
    # Check Stage IV first (most specific)
    if stage.startswith('STAGE IV'):
        return 'Stage_IV'
    # Check Stage III (including IIIA, IIIB, IIIC)
    elif stage.startswith('STAGE III'):
        return 'Stage_III'
    # Check Stage II (including IIA, IIB, IIC)
    elif stage.startswith('STAGE II'):
        return 'Stage_II'
    # Check Stage I (including IA)
    elif stage.startswith('STAGE I') and not stage.startswith('STAGE II'):
        return 'Stage_I'
    else:
        return None

# Apply grouping
TCGA_merged_features_df['Stage_Group'] = TCGA_merged_features_df['ajcc_pathologic_tumor_stage'].apply(group_stage)


print(TCGA_merged_features_df['Stage_Group'].value_counts())

# Create dummy variables for the 4 stage groups (0/1 format)
stage_dummies = pd.get_dummies(TCGA_merged_features_df['Stage_Group'], prefix='Stage', dtype=int)

# Add dummy variables to the dataframe
TCGA_merged_features_df = pd.concat([TCGA_merged_features_df, stage_dummies], axis=1)
TCGA_merged_features_df.drop("Stage_Group", axis=1, inplace=True)
TCGA_merged_features_df.drop("ajcc_pathologic_tumor_stage", axis=1, inplace=True)

TCGA_merged_features_df.to_csv("Colorectal_Cancer_HE_patches/TCGA_Survival_Analysis.csv", index=False)



### Cox Proportional Hazards Regression Analysis (Marginal / PFI)
print("\n" + "="*70)
print("COX PROPORTIONAL HAZARDS REGRESSION ANALYSIS")
print("="*70)

from lifelines import CoxPHFitter
import numpy as np

survival_df = pd.read_csv("Colorectal_Cancer_HE_patches/TCGA_Survival_Analysis.csv")

### Convert proportion variables to percentages (multiply by 100)
print("Converting proportion variables to percentages...")
proportion_columns = [col for col in survival_df.columns if 'Overall_Proportion' in col]
print(f"Found {len(proportion_columns)} proportion variables: {proportion_columns}")

for col in proportion_columns:
    survival_df[col] = survival_df[col] * 100
    print(f"  {col}: converted to percentage (0-100 scale)")

print("Proportion conversion completed.")

# Remove excluded columns for marginal analysis
excluded_cols = ['OS', 'OS.time', 'Sample_ID', 'Patient_ID']
feature_cols = [col for col in survival_df.columns if col not in excluded_cols + ['PFI', 'PFI.time']]


### Marginal Cox Regression for each variable

marginal_cox_results = []
cph = CoxPHFitter()

# Separate stage variables for joint analysis
stage_vars = ['Stage_Stage_II', 'Stage_Stage_III', 'Stage_Stage_IV'] 
non_stage_vars = [col for col in feature_cols if col not in ['Stage_Stage_I', 'Stage_Stage_II', 'Stage_Stage_III', 'Stage_Stage_IV']]

# 1. Individual marginal analysis for non-stage variables
for i, var_name in enumerate(non_stage_vars):
    print(f"\nProcessing variable {i+1}/{len(non_stage_vars)}: {var_name}")
    
    # Prepare data for this variable
    cox_data = survival_df[['PFI', 'PFI.time', var_name]].copy()
    
    # Fit Cox model
    cph.fit(cox_data, duration_col='PFI.time', event_col='PFI')
    
    # Extract results
    coef = cph.params_[var_name]
    hazard_ratio = np.exp(coef)
    p_value = cph.summary.loc[var_name, 'p']
    ci_lower = cph.confidence_intervals_.loc[var_name, '95% lower-bound']
    ci_upper = cph.confidence_intervals_.loc[var_name, '95% upper-bound']
    hr_ci_lower = np.exp(ci_lower)
    hr_ci_upper = np.exp(ci_upper)
    concordance = cph.concordance_index_
    
    # Store results
    marginal_cox_results.append({
        'Variable': var_name,
        'Coefficient': coef,
        'Hazard_Ratio': hazard_ratio,
        'HR_CI_Lower_95': hr_ci_lower,
        'HR_CI_Upper_95': hr_ci_upper,
        'P_Value': p_value,
        'Concordance_Index': concordance,
        'N_Observations': len(cox_data),
        'N_Events': cox_data['PFI'].sum()
    })
    
    print(f"  HR: {hazard_ratio:.3f} [{hr_ci_lower:.3f}-{hr_ci_upper:.3f}], p: {p_value:.4f}, C-index: {concordance:.3f}")

# 2. Joint marginal analysis for Stage variables (as one group, using Stage I as reference)
print(f"\nProcessing Stage variables as one group (reference: Stage I)...")

# Prepare data for stage analysis
stage_cox_data = survival_df[['PFI', 'PFI.time'] + stage_vars].copy()

# Fit Cox model with all stage variables together
cph_stage = CoxPHFitter()
cph_stage.fit(stage_cox_data, duration_col='PFI.time', event_col='PFI')

print(f"Stage group concordance index: {cph_stage.concordance_index_:.3f}")

# Extract results for each stage variable and add to main results
for var_name in stage_vars:
    coef = cph_stage.params_[var_name]
    hazard_ratio = np.exp(coef)
    p_value = cph_stage.summary.loc[var_name, 'p']
    ci_lower = cph_stage.confidence_intervals_.loc[var_name, '95% lower-bound']
    ci_upper = cph_stage.confidence_intervals_.loc[var_name, '95% upper-bound']
    hr_ci_lower = np.exp(ci_lower)
    hr_ci_upper = np.exp(ci_upper)
    
    # Store results in the same list as other variables
    marginal_cox_results.append({
        'Variable': var_name,
        'Coefficient': coef,
        'Hazard_Ratio': hazard_ratio,
        'HR_CI_Lower_95': hr_ci_lower,
        'HR_CI_Upper_95': hr_ci_upper,
        'P_Value': p_value,
        'Concordance_Index': cph_stage.concordance_index_,
        'N_Observations': len(stage_cox_data),
        'N_Events': stage_cox_data['PFI'].sum()
    })
    
    # Interpret stage names for display
    stage_name = var_name.replace('Stage_Stage_', 'Stage ')
    print(f"  {stage_name} vs Stage I: HR = {hazard_ratio:.3f} [{hr_ci_lower:.3f}-{hr_ci_upper:.3f}], p = {p_value:.4f}")

# Create results DataFrame
cox_results_df = pd.DataFrame(marginal_cox_results)


# Sort by p-value
cox_results_sorted = cox_results_df.sort_values('P_Value').reset_index(drop=True)


# Add significance indicators
cox_results_sorted['Significance'] = cox_results_sorted['P_Value'].apply(
    lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else ''
)

# Save results
output_path = "Colorectal_Cancer_HE_patches/Marginal_CoxPH_Regression_PFI_Results.csv"
cox_results_sorted.to_csv(output_path, index=False)
print(f"Marginal PFI results saved to: {output_path}")


### Cox Proportional Hazards Regression Analysis (Marginal / OS)
print("\n" + "="*70)
print("COX PROPORTIONAL HAZARDS REGRESSION ANALYSIS - MARGINAL OS")
print("="*70)

marginal_os_results = []
cph_os = CoxPHFitter()

# 1. Individual marginal analysis for non-stage variables (OS)
for i, var_name in enumerate(non_stage_vars):
    print(f"\nProcessing variable {i+1}/{len(non_stage_vars)}: {var_name}")
    
    # Prepare data for this variable
    cox_data = survival_df[['OS', 'OS.time', var_name]].copy()
    
    # Fit Cox model
    cph_os.fit(cox_data, duration_col='OS.time', event_col='OS')
    
    # Extract results
    coef = cph_os.params_[var_name]
    hazard_ratio = np.exp(coef)
    p_value = cph_os.summary.loc[var_name, 'p']
    ci_lower = cph_os.confidence_intervals_.loc[var_name, '95% lower-bound']
    ci_upper = cph_os.confidence_intervals_.loc[var_name, '95% upper-bound']
    hr_ci_lower = np.exp(ci_lower)
    hr_ci_upper = np.exp(ci_upper)
    concordance = cph_os.concordance_index_
    
    # Store results
    marginal_os_results.append({
        'Variable': var_name,
        'Coefficient': coef,
        'Hazard_Ratio': hazard_ratio,
        'HR_CI_Lower_95': hr_ci_lower,
        'HR_CI_Upper_95': hr_ci_upper,
        'P_Value': p_value,
        'Concordance_Index': concordance,
        'N_Observations': len(cox_data),
        'N_Events': cox_data['OS'].sum()
    })
    
    print(f"  HR: {hazard_ratio:.3f} [{hr_ci_lower:.3f}-{hr_ci_upper:.3f}], p: {p_value:.4f}, C-index: {concordance:.3f}")

# 2. Joint marginal analysis for Stage variables (OS, using Stage I as reference)
print(f"\nProcessing Stage variables as one group (reference: Stage I)...")

# Prepare data for stage analysis
stage_os_data = survival_df[['OS', 'OS.time'] + stage_vars].copy()

# Fit Cox model with all stage variables together
cph_stage_os = CoxPHFitter()
cph_stage_os.fit(stage_os_data, duration_col='OS.time', event_col='OS')

print(f"Stage group concordance index: {cph_stage_os.concordance_index_:.3f}")

# Extract results for each stage variable and add to main results
for var_name in stage_vars:
    coef = cph_stage_os.params_[var_name]
    hazard_ratio = np.exp(coef)
    p_value = cph_stage_os.summary.loc[var_name, 'p']
    ci_lower = cph_stage_os.confidence_intervals_.loc[var_name, '95% lower-bound']
    ci_upper = cph_stage_os.confidence_intervals_.loc[var_name, '95% upper-bound']
    hr_ci_lower = np.exp(ci_lower)
    hr_ci_upper = np.exp(ci_upper)
    
    # Store results in the same list as other variables
    marginal_os_results.append({
        'Variable': var_name,
        'Coefficient': coef,
        'Hazard_Ratio': hazard_ratio,
        'HR_CI_Lower_95': hr_ci_lower,
        'HR_CI_Upper_95': hr_ci_upper,
        'P_Value': p_value,
        'Concordance_Index': cph_stage_os.concordance_index_,
        'N_Observations': len(stage_os_data),
        'N_Events': stage_os_data['OS'].sum()
    })
    
    # Interpret stage names for display
    stage_name = var_name.replace('Stage_Stage_', 'Stage ')
    print(f"  {stage_name} vs Stage I: HR = {hazard_ratio:.3f} [{hr_ci_lower:.3f}-{hr_ci_upper:.3f}], p = {p_value:.4f}")

# Create OS results DataFrame
cox_os_df = pd.DataFrame(marginal_os_results)
cox_os_sorted = cox_os_df.sort_values('P_Value').reset_index(drop=True)
cox_os_sorted['Significance'] = cox_os_sorted['P_Value'].apply(
    lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else ''
)

# Save OS results
output_path_os = "Colorectal_Cancer_HE_patches/Marginal_CoxPH_Regression_OS_Results.csv"
cox_os_sorted.to_csv(output_path_os, index=False)
print(f"Marginal OS results saved to: {output_path_os}")






### Cox Proportional Hazards Regression Analysis (Joint / PFI)
print("\n" + "="*70)
print("COX PROPORTIONAL HAZARDS REGRESSION ANALYSIS - JOINT PFI")
print("="*70)

# Filter variables based on marginal model concordance index > 0.5
print("Filtering variables based on marginal PFI concordance index > 0.5...")

good_pfi_vars = []
for _, row in cox_results_sorted.iterrows():
    if row['Concordance_Index'] > 0.5:
        good_pfi_vars.append(row['Variable'])
        print(f"  {row['Variable']}: C-index = {row['Concordance_Index']:.3f}")

print(f"\nFound {len(good_pfi_vars)} variables with concordance > 0.5 for PFI")

# Always include stage variables (essential for adjustment)
stage_vars_essential = [var for var in stage_vars if var in good_pfi_vars]
if len(stage_vars_essential) == 0:
    # If no stage vars pass threshold, add them anyway for proper adjustment
    stage_vars_essential = stage_vars
    good_pfi_vars.extend(stage_vars)
    print(f"Added essential stage variables: {stage_vars}")

# Remove Normal_Epithelial_Overall_Proportion if present (use as reference)
reference_var = 'Normal_Epithelial_Overall_Proportion'
if reference_var in good_pfi_vars:
    good_pfi_vars.remove(reference_var)
    print(f"Removed {reference_var} as reference category")

print(f"Final variables for joint PFI analysis: {len(good_pfi_vars)} variables")
print(f"Variables: {good_pfi_vars}")

# Final data for joint analysis
joint_pfi_data_final = survival_df[['PFI', 'PFI.time'] + good_pfi_vars].copy()

# Fit joint Cox model for PFI
print("Fitting joint PFI Cox model with filtered variables...")
cph_joint_pfi = CoxPHFitter()
cph_joint_pfi.fit(joint_pfi_data_final, duration_col='PFI.time', event_col='PFI')

# Update filtered_vars for consistency
filtered_vars = good_pfi_vars

print(f"Joint PFI model concordance index: {cph_joint_pfi.concordance_index_:.3f}")

# Extract results for joint model
joint_pfi_results = []
for var_name in filtered_vars:
    coef = cph_joint_pfi.params_[var_name]
    hazard_ratio = np.exp(coef)
    p_value = cph_joint_pfi.summary.loc[var_name, 'p']
    ci_lower = cph_joint_pfi.confidence_intervals_.loc[var_name, '95% lower-bound']
    ci_upper = cph_joint_pfi.confidence_intervals_.loc[var_name, '95% upper-bound']
    hr_ci_lower = np.exp(ci_lower)
    hr_ci_upper = np.exp(ci_upper)
    
    joint_pfi_results.append({
        'Variable': var_name,
        'Coefficient': coef,
        'Hazard_Ratio': hazard_ratio,
        'HR_CI_Lower_95': hr_ci_lower,
        'HR_CI_Upper_95': hr_ci_upper,
        'P_Value': p_value,
        'Concordance_Index': cph_joint_pfi.concordance_index_,
        'N_Observations': len(joint_pfi_data_final),
        'N_Events': joint_pfi_data_final['PFI'].sum()
    })

# Create joint PFI results DataFrame
joint_pfi_df = pd.DataFrame(joint_pfi_results)
joint_pfi_sorted = joint_pfi_df.sort_values('P_Value').reset_index(drop=True)
joint_pfi_sorted['Significance'] = joint_pfi_sorted['P_Value'].apply(
    lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else ''
)

# Save joint PFI results
output_path_joint_pfi = "Colorectal_Cancer_HE_patches/Joint_CoxPH_Regression_PFI_Results.csv"
joint_pfi_sorted.to_csv(output_path_joint_pfi, index=False)
print(f"Joint PFI results saved to: {output_path_joint_pfi}")


### Cox Proportional Hazards Regression Analysis (Joint / OS)
print("\n" + "="*70)
print("COX PROPORTIONAL HAZARDS REGRESSION ANALYSIS - JOINT OS")
print("="*70)

# Filter variables based on marginal OS model concordance index > 0.5
print("Filtering variables based on marginal OS concordance index > 0.5...")

good_os_vars = []
for _, row in cox_os_sorted.iterrows():
    if row['Concordance_Index'] > 0.5:
        good_os_vars.append(row['Variable'])
        print(f"  {row['Variable']}: C-index = {row['Concordance_Index']:.3f}")

print(f"\nFound {len(good_os_vars)} variables with concordance > 0.5 for OS")

# Always include stage variables (essential for adjustment)
stage_vars_essential_os = [var for var in stage_vars if var in good_os_vars]
if len(stage_vars_essential_os) == 0:
    # If no stage vars pass threshold, add them anyway for proper adjustment
    stage_vars_essential_os = stage_vars
    good_os_vars.extend(stage_vars)
    print(f"Added essential stage variables: {stage_vars}")

# Remove Normal_Epithelial_Overall_Proportion if present (use as reference)
reference_var = 'Normal_Epithelial_Overall_Proportion'
if reference_var in good_os_vars:
    good_os_vars.remove(reference_var)
    print(f"Removed {reference_var} as reference category")

print(f"Final variables for joint OS analysis: {len(good_os_vars)} variables")
print(f"Variables: {good_os_vars}")

# Use original data (no transformation needed)
joint_os_data_final = survival_df[['OS', 'OS.time'] + good_os_vars].copy()

# Fit joint Cox model for OS
print("Fitting joint OS Cox model with filtered variables...")
cph_joint_os = CoxPHFitter()
cph_joint_os.fit(joint_os_data_final, duration_col='OS.time', event_col='OS')

# Update filtered_vars for OS results
filtered_vars_os = good_os_vars

print(f"Joint OS model concordance index: {cph_joint_os.concordance_index_:.3f}")

# Extract results for joint model
joint_os_results = []
for var_name in filtered_vars_os:
    coef = cph_joint_os.params_[var_name]
    hazard_ratio = np.exp(coef)
    p_value = cph_joint_os.summary.loc[var_name, 'p']
    ci_lower = cph_joint_os.confidence_intervals_.loc[var_name, '95% lower-bound']
    ci_upper = cph_joint_os.confidence_intervals_.loc[var_name, '95% upper-bound']
    hr_ci_lower = np.exp(ci_lower)
    hr_ci_upper = np.exp(ci_upper)
    
    joint_os_results.append({
        'Variable': var_name,
        'Coefficient': coef,
        'Hazard_Ratio': hazard_ratio,
        'HR_CI_Lower_95': hr_ci_lower,
        'HR_CI_Upper_95': hr_ci_upper,
        'P_Value': p_value,
        'Concordance_Index': cph_joint_os.concordance_index_,
        'N_Observations': len(joint_os_data_final),
        'N_Events': joint_os_data_final['OS'].sum()
    })

# Create joint OS results DataFrame
joint_os_df = pd.DataFrame(joint_os_results)
joint_os_sorted = joint_os_df.sort_values('P_Value').reset_index(drop=True)
joint_os_sorted['Significance'] = joint_os_sorted['P_Value'].apply(
    lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else ''
)

# Save joint OS results
output_path_joint_os = "Colorectal_Cancer_HE_patches/Joint_CoxPH_Regression_OS_Results.csv"
joint_os_sorted.to_csv(output_path_joint_os, index=False)
print(f"Joint OS results saved to: {output_path_joint_os}")







### Kaplan-Meier Survival Curves
print("\n" + "="*70)
print("KAPLAN-MEIER SURVIVAL CURVE ANALYSIS")
print("="*70)

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

# Create stage groups for KM analysis
print("Creating stage groups for KM analysis...")

# Create a combined stage variable
survival_df['Stage_Group'] = 'Unknown'
survival_df.loc[survival_df['Stage_Stage_II'] == 1, 'Stage_Group'] = 'Stage II'
survival_df.loc[survival_df['Stage_Stage_III'] == 1, 'Stage_Group'] = 'Stage III'
survival_df.loc[survival_df['Stage_Stage_IV'] == 1, 'Stage_Group'] = 'Stage IV'
# Stage I is reference (all stage dummies = 0)
stage_i_mask = (survival_df['Stage_Stage_II'] == 0) & (survival_df['Stage_Stage_III'] == 0) & (survival_df['Stage_Stage_IV'] == 0)
survival_df.loc[stage_i_mask, 'Stage_Group'] = 'Stage I'

print(f"Stage distribution:")
print(survival_df['Stage_Group'].value_counts())


### Figure 7 (C)
### 1. KM Curves stratified by Stage (PFI only)
print("\n1. KM Curves stratified by Stage (PFI only)...")

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# PFI by Stage
kmf = KaplanMeierFitter()
stage_groups = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
colors = ['green', 'blue', 'orange', 'red']

for i, stage in enumerate(stage_groups):
    stage_data = survival_df[survival_df['Stage_Group'] == stage]
    if len(stage_data) > 0:
        kmf.fit(stage_data['PFI.time'], event_observed=stage_data['PFI'], label=f'{stage} (n={len(stage_data)})')
        kmf.plot_survival_function(ax=ax, color=colors[i], linewidth=3, show_censors=True, ci_show=False)

# Remove title
# ax.set_title('Progression-Free Interval by Stage', fontsize=16, fontweight='bold')

# Bold and larger axis labels and ticks
ax.set_xlabel('Time (days)', fontsize=24, fontweight='bold')
ax.set_ylabel('Progression-Free Survival Probability', fontsize=24, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=20, width=3, length=6, colors='black')

# Make tick labels bold
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# Bold legend in bottom right, larger font, bold lines
legend = ax.legend(fontsize=26, loc='lower right', 
                  frameon=True, fancybox=False, shadow=False)
for line in legend.get_lines():
    line.set_linewidth(3)
# Make legend text bold
for text in legend.get_texts():
    text.set_fontweight('bold')

# Bold border
for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')

# Remove grid
ax.grid(False)

# Overall log-rank test and annotate P-value on plot
from lifelines.statistics import multivariate_logrank_test
import pandas as pd
import numpy as np

# Prepare data for multivariate log-rank test
all_durations = []
all_events = []
all_groups = []

for i, stage in enumerate(['Stage I', 'Stage II', 'Stage III', 'Stage IV']):
    stage_data = survival_df[survival_df['Stage_Group'] == stage]
    if len(stage_data) > 0:
        all_durations.extend(stage_data['PFI.time'].values)
        all_events.extend(stage_data['PFI'].values)
        all_groups.extend([i] * len(stage_data))

if len(set(all_groups)) > 1:
    overall_result = multivariate_logrank_test(all_durations, all_groups, all_events, alpha=0.05)
    
    # Annotate P-value in top right corner
    if overall_result.p_value < 0.001:
        p_text = f"Log-rank p-value < 0.001"
    else:
        p_text = f"Log-rank p-value = {overall_result.p_value:.3f}"
    
    ax.text(0.98, 0.98, p_text, transform=ax.transAxes, fontsize=24, fontweight='bold',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1))

plt.tight_layout()
plt.savefig('Colorectal_Cancer_HE_patches/Visual/KM_Curves_PFI_by_Stage.png', dpi=300, bbox_inches='tight')
plt.show()






### 2. KM Curves stratified by Tumor_to_Other_Immune_Distance_Avg (Quartiles, PFI only)
print("\n2. KM Curves stratified by Tumor_to_Other_Immune_Distance_Avg (Quartiles, PFI only)...")

# Create quartiles for Tumor_to_Other_Immune_Distance_Avg
distance_var = 'Tumor_to_Other_Immune_Distance_Avg'
quartiles = survival_df[distance_var].quantile([0.25, 0.5, 0.75]).values
print(f"Quartile cutoffs for {distance_var}: {quartiles}")

def assign_quartile(value, quartiles):
    if value <= quartiles[0]:
        return 'Q1'
    elif value <= quartiles[1]:
        return 'Q2'
    elif value <= quartiles[2]:
        return 'Q3'
    else:
        return 'Q4'

survival_df['Distance_Quartile'] = survival_df[distance_var].apply(lambda x: assign_quartile(x, quartiles))

print(f"Distance quartile distribution:")
print(survival_df['Distance_Quartile'].value_counts())

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# PFI by Distance Quartiles with value ranges
quartile_groups = ['Q1', 'Q2', 'Q3', 'Q4']
colors_q = ['green', 'blue', 'orange', 'red']

# Define quartile ranges for labels
quartile_ranges = [
    f"≤{quartiles[0]:.1f}",
    f"{quartiles[0]:.1f}-{quartiles[1]:.1f}",
    f"{quartiles[1]:.1f}-{quartiles[2]:.1f}",
    f">{quartiles[2]:.1f}"
]

for i, quartile in enumerate(quartile_groups):
    q_data = survival_df[survival_df['Distance_Quartile'] == quartile]
    if len(q_data) > 0:
        # Include value range in label
        label_text = f'{quartile}: {quartile_ranges[i]} (n={len(q_data)})'
        kmf.fit(q_data['PFI.time'], event_observed=q_data['PFI'], label=label_text)
        kmf.plot_survival_function(ax=ax, color=colors_q[i], linewidth=3, show_censors=True, ci_show=False)

# Remove title
# ax.set_title('Progression-Free Interval by Tumor-Other Immune Distance', fontsize=16, fontweight='bold')

# Bold and larger axis labels and ticks
ax.set_xlabel('Time (days)', fontsize=24, fontweight='bold')
ax.set_ylabel('Progression-Free Survival Probability', fontsize=24, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=20, width=3, length=6, colors='black')

# Make tick labels bold
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# Bold legend in bottom right, larger font, bold lines
legend = ax.legend(fontsize=26, loc='lower right', 
                  frameon=True, fancybox=False, shadow=False)
for line in legend.get_lines():
    line.set_linewidth(3)
# Make legend text bold
for text in legend.get_texts():
    text.set_fontweight('bold')

# Bold border
for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')

# Remove grid
ax.grid(False)

# Overall log-rank test and annotate P-value on plot
# Prepare data for multivariate log-rank test
all_durations_q = []
all_events_q = []
all_groups_q = []

for i, quartile in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    quartile_data = survival_df[survival_df['Distance_Quartile'] == quartile]
    if len(quartile_data) > 0:
        all_durations_q.extend(quartile_data['PFI.time'].values)
        all_events_q.extend(quartile_data['PFI'].values)
        all_groups_q.extend([i] * len(quartile_data))

if len(set(all_groups_q)) > 1:
    overall_result_q = multivariate_logrank_test(all_durations_q, all_groups_q, all_events_q, alpha=0.05)
    
    # Annotate P-value in top right corner
    if overall_result_q.p_value < 0.001:
        p_text_q = f"Log-rank p-value < 0.001"
    else:
        p_text_q = f"Log-rank p-value = {overall_result_q.p_value:.3f}"
    
    ax.text(0.98, 0.98, p_text_q, transform=ax.transAxes, fontsize=24, fontweight='bold',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1))

plt.tight_layout()
plt.savefig('Colorectal_Cancer_HE_patches/Visual/KM_Curves_PFI_by_Tumor_to_Other_Immune_Distance_Quartiles.png', dpi=300, bbox_inches='tight')
plt.show()


### 3. KM Curves stratified by Stromal_to_Other_Immune_Distance_Avg (Quartiles, PFI only)
print("\n3. KM Curves stratified by Stromal_to_Other_Immune_Distance_Avg (Quartiles, PFI only)...")

# Create quartiles for Stromal_to_Other_Immune_Distance_Avg
stromal_distance_var = 'Stromal_to_Other_Immune_Distance_Avg'
stromal_quartiles = survival_df[stromal_distance_var].quantile([0.25, 0.5, 0.75]).values
print(f"Stromal distance quartile cutoffs for {stromal_distance_var}: {stromal_quartiles}")

def assign_stromal_quartile(value, quartiles):
    if value <= quartiles[0]:
        return 'Q1'
    elif value <= quartiles[1]:
        return 'Q2'
    elif value <= quartiles[2]:
        return 'Q3'
    else:
        return 'Q4'

survival_df['Stromal_Distance_Quartile'] = survival_df[stromal_distance_var].apply(lambda x: assign_stromal_quartile(x, stromal_quartiles))

print(f"Stromal distance quartile distribution:")
print(survival_df['Stromal_Distance_Quartile'].value_counts())

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# PFI by Stromal Distance Quartiles with value ranges
stromal_quartile_groups = ['Q1', 'Q2', 'Q3', 'Q4']
colors_stromal = ['green', 'blue', 'orange', 'red']

# Define stromal quartile ranges for labels
stromal_quartile_ranges = [
    f"≤{stromal_quartiles[0]:.1f}",
    f"{stromal_quartiles[0]:.1f}-{stromal_quartiles[1]:.1f}",
    f"{stromal_quartiles[1]:.1f}-{stromal_quartiles[2]:.1f}",
    f">{stromal_quartiles[2]:.1f}"
]

for i, quartile in enumerate(stromal_quartile_groups):
    q_data = survival_df[survival_df['Stromal_Distance_Quartile'] == quartile]
    if len(q_data) > 0:
        # Include value range in label
        label_text = f'{quartile}: {stromal_quartile_ranges[i]} (n={len(q_data)})'
        kmf.fit(q_data['PFI.time'], event_observed=q_data['PFI'], label=label_text)
        kmf.plot_survival_function(ax=ax, color=colors_stromal[i], linewidth=3, show_censors=True, ci_show=False)

# Bold and larger axis labels and ticks
ax.set_xlabel('Time (days)', fontsize=24, fontweight='bold')
ax.set_ylabel('Progression-Free Survival Probability', fontsize=24, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=20, width=3, length=6, colors='black')

# Make tick labels bold
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# Bold legend in bottom right, larger font, bold lines
legend = ax.legend(fontsize=26, loc='lower right', 
                  frameon=True, fancybox=False, shadow=False)
for line in legend.get_lines():
    line.set_linewidth(3)
# Make legend text bold
for text in legend.get_texts():
    text.set_fontweight('bold')

# Bold border
for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')

# Remove grid
ax.grid(False)

# Overall log-rank test and annotate P-value on plot
# Prepare data for multivariate log-rank test
all_durations_stromal = []
all_events_stromal = []
all_groups_stromal = []

for i, quartile in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    quartile_data = survival_df[survival_df['Stromal_Distance_Quartile'] == quartile]
    if len(quartile_data) > 0:
        all_durations_stromal.extend(quartile_data['PFI.time'].values)
        all_events_stromal.extend(quartile_data['PFI'].values)
        all_groups_stromal.extend([i] * len(quartile_data))

if len(set(all_groups_stromal)) > 1:
    overall_result_stromal = multivariate_logrank_test(all_durations_stromal, all_groups_stromal, all_events_stromal, alpha=0.05)
    
    # Annotate P-value in top right corner
    if overall_result_stromal.p_value < 0.001:
        p_text_stromal = f"Log-rank p-value < 0.001"
    else:
        p_text_stromal = f"Log-rank p-value = {overall_result_stromal.p_value:.3f}"
    
    ax.text(0.98, 0.98, p_text_stromal, transform=ax.transAxes, fontsize=24, fontweight='bold',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1))

plt.tight_layout()
plt.savefig('Colorectal_Cancer_HE_patches/Visual/KM_Curves_PFI_by_Stromal_to_Other_Immune_Distance_Quartiles.png', dpi=300, bbox_inches='tight')
plt.show()



### Overall Survival (OS) Analysis
print("\n" + "="*70)
print("OVERALL SURVIVAL (OS) KAPLAN-MEIER CURVE ANALYSIS")
print("="*70)

### 1. OS KM Curves stratified by Stage
print("\n1. OS KM Curves stratified by Stage...")

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# OS by Stage
kmf = KaplanMeierFitter()
stage_groups = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
colors = ['green', 'blue', 'orange', 'red']

for i, stage in enumerate(stage_groups):
    stage_data = survival_df[survival_df['Stage_Group'] == stage]
    if len(stage_data) > 0:
        kmf.fit(stage_data['OS.time'], event_observed=stage_data['OS'], label=f'{stage} (n={len(stage_data)})')
        kmf.plot_survival_function(ax=ax, color=colors[i], linewidth=3, show_censors=True, ci_show=False)

# Bold and larger axis labels and ticks
ax.set_xlabel('Time (days)', fontsize=24, fontweight='bold')
ax.set_ylabel('Overall Survival Probability', fontsize=24, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=20, width=3, length=6, colors='black')

# Bold legend in bottom right, larger font, bold lines
legend = ax.legend(fontsize=26, loc='lower right', 
                  frameon=True, fancybox=False, shadow=False)
for line in legend.get_lines():
    line.set_linewidth(3)
# Make legend text bold
for text in legend.get_texts():
    text.set_fontweight('bold')

# Bold border
for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')

# Remove grid
ax.grid(False)

# Overall log-rank test and annotate P-value on plot
# Prepare data for multivariate log-rank test
all_durations_os = []
all_events_os = []
all_groups_os = []

for i, stage in enumerate(['Stage I', 'Stage II', 'Stage III', 'Stage IV']):
    stage_data = survival_df[survival_df['Stage_Group'] == stage]
    if len(stage_data) > 0:
        all_durations_os.extend(stage_data['OS.time'].values)
        all_events_os.extend(stage_data['OS'].values)
        all_groups_os.extend([i] * len(stage_data))

if len(set(all_groups_os)) > 1:
    overall_result_os = multivariate_logrank_test(all_durations_os, all_groups_os, all_events_os, alpha=0.05)
    
    # Annotate P-value in top right corner
    if overall_result_os.p_value < 0.001:
        p_text_os = f"Log-rank p-value < 0.001"
    else:
        p_text_os = f"Log-rank p-value = {overall_result_os.p_value:.3f}"
    
    ax.text(0.98, 0.98, p_text_os, transform=ax.transAxes, fontsize=24, fontweight='bold',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1))

plt.tight_layout()
plt.savefig('Colorectal_Cancer_HE_patches/Visual/KM_Curves_OS_by_Stage.png', dpi=300, bbox_inches='tight')
plt.show()



### 2. OS KM Curves stratified by Age (Quartiles)
print("\n2. OS KM Curves stratified by Age (Quartiles)...")

# Check if age variable exists in the dataframe
age_var = 'age_at_initial_pathologic_diagnosis'
if age_var not in survival_df.columns:
    print(f"Warning: {age_var} not found in dataframe. Available columns:")
    print([col for col in survival_df.columns if 'age' in col.lower()])
    # Try to find age variable
    age_columns = [col for col in survival_df.columns if 'age' in col.lower()]
    if age_columns:
        age_var = age_columns[0]
        print(f"Using {age_var} instead")
    else:
        print("No age variable found. Skipping age analysis.")
        age_var = None

if age_var and age_var in survival_df.columns:
    # Create quartiles for Age
    age_quartiles = survival_df[age_var].quantile([0.25, 0.5, 0.75]).values
    print(f"Age quartile cutoffs: {age_quartiles}")

    def assign_age_quartile(value, quartiles):
        if pd.isna(value):
            return None
        if value <= quartiles[0]:
            return 'Q1'
        elif value <= quartiles[1]:
            return 'Q2'
        elif value <= quartiles[2]:
            return 'Q3'
        else:
            return 'Q4'

    survival_df['Age_Quartile'] = survival_df[age_var].apply(lambda x: assign_age_quartile(x, age_quartiles))
    
    # Remove NaN values
    survival_df_age = survival_df.dropna(subset=['Age_Quartile'])
    
    print(f"Age quartile distribution:")
    print(survival_df_age['Age_Quartile'].value_counts())

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # OS by Age Quartiles with value ranges
    age_quartile_groups = ['Q1', 'Q2', 'Q3', 'Q4']
    colors_age = ['green', 'blue', 'orange', 'red']

    # Define age quartile ranges for labels
    age_quartile_ranges = [
        f"≤{age_quartiles[0]:.0f}",
        f"{age_quartiles[0]:.0f}-{age_quartiles[1]:.0f}",
        f"{age_quartiles[1]:.0f}-{age_quartiles[2]:.0f}",
        f">{age_quartiles[2]:.0f}"
    ]

    for i, quartile in enumerate(age_quartile_groups):
        q_data = survival_df_age[survival_df_age['Age_Quartile'] == quartile]
        if len(q_data) > 0:
            # Include value range in label
            label_text = f'{quartile}: {age_quartile_ranges[i]} years (n={len(q_data)})'
            kmf.fit(q_data['OS.time'], event_observed=q_data['OS'], label=label_text)
            kmf.plot_survival_function(ax=ax, color=colors_age[i], linewidth=3, show_censors=True, ci_show=False)

    # Bold and larger axis labels and ticks
    ax.set_xlabel('Time (days)', fontsize=24, fontweight='bold')
    ax.set_ylabel('Overall Survival Probability', fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=20, width=3, length=6, colors='black')

    # Bold legend in bottom right, larger font, bold lines
    legend = ax.legend(fontsize=26, loc='lower right', 
                      frameon=True, fancybox=False, shadow=False)
    for line in legend.get_lines():
        line.set_linewidth(3)
    # Make legend text bold
    for text in legend.get_texts():
        text.set_fontweight('bold')

    # Bold border
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    # Remove grid
    ax.grid(False)

    # Overall log-rank test and annotate P-value on plot
    # Prepare data for multivariate log-rank test
    all_durations_age = []
    all_events_age = []
    all_groups_age = []

    for i, quartile in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        quartile_data = survival_df_age[survival_df_age['Age_Quartile'] == quartile]
        if len(quartile_data) > 0:
            all_durations_age.extend(quartile_data['OS.time'].values)
            all_events_age.extend(quartile_data['OS'].values)
            all_groups_age.extend([i] * len(quartile_data))

    if len(set(all_groups_age)) > 1:
        overall_result_age = multivariate_logrank_test(all_durations_age, all_groups_age, all_events_age, alpha=0.05)
        
        # Annotate P-value in top right corner
        if overall_result_age.p_value < 0.001:
            p_text_age = f"Log-rank p-value < 0.001"
        else:
            p_text_age = f"Log-rank p-value = {overall_result_age.p_value:.3f}"
        
        ax.text(0.98, 0.98, p_text_age, transform=ax.transAxes, fontsize=24, fontweight='bold',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1))

    plt.tight_layout()
    plt.savefig('Colorectal_Cancer_HE_patches/Visual/KM_Curves_OS_by_Age_Quartiles.png', dpi=300, bbox_inches='tight')
    plt.show()

