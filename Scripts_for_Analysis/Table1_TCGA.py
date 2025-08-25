"""
STPath-COAD: TCGA Summary Statistics for Table 1
===============================================

This script generates summary statistics and demographics for TCGA colorectal cancer
cohort used in STPath-COAD validation. Creates comprehensive patient characteristics table.

Author: Saishi Cui
Date: Sept 2025

Purpose: Generate comprehensive summary table of TCGA patient characteristics,
clinical variables, and molecular features for Table 1 of the STPath-COAD manuscript.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load MAF file
maf_df = pd.read_csv("TCGA_data/mc3.v0.2.8.CONTROLLED.maf", sep='\t')

print(f"MAF Shape: {maf_df.shape}")

# Extract patient ID
maf_df['Patient_ID'] = maf_df['Tumor_Sample_Barcode'].str[:12]

# Calculate burden using all mutations (no filtering)
print("Calculating mutation burden...")
all_burden = maf_df.groupby('Patient_ID').size()
print(f"Number of patients with mutations: {len(all_burden)}")

# Read features file
print("\nReading features file...")
features_df = pd.read_csv("Colorectal_Cancer_HE_patches/TCGA_Features_Complete_WithClinical.csv")
print(f"Features shape: {features_df.shape}")

# Merge burden data
print("Merging mutation burden...")
burden_df = all_burden.reset_index()
burden_df.columns = ['Patient_ID', 'Mutation_Burden']

# Merge to features
features_with_burden = features_df.merge(burden_df, on='Patient_ID', how='left')

# Patients without mutation data, set burden to 0
features_with_burden['Mutation_Burden'] = features_with_burden['Mutation_Burden'].fillna(0).astype(int)

print(f"Merged shape: {features_with_burden.shape}")
print(f"Patients with mutation burden data: {(features_with_burden['Mutation_Burden'] > 0).sum()}")
print(f"Patients with burden = 0: {(features_with_burden['Mutation_Burden'] == 0).sum()}")

# Remove patients without mutation data
print("\nRemoving patients without mutation data...")
features_final = features_with_burden[features_with_burden['Mutation_Burden'] > 0].copy()

print(f"Final dataset shape: {features_final.shape}")
print(f"Mutation burden range: {features_final['Mutation_Burden'].min()} - {features_final['Mutation_Burden'].max()}")
print(f"Average mutation burden: {features_final['Mutation_Burden'].mean():.1f}")
print(f"Median mutation burden: {features_final['Mutation_Burden'].median():.1f}")

# Remove rows where any value in columns 7-30 equals 0
print(f"Before filtering: {features_final.shape}")
mask_no_zeros = ~(features_final.iloc[:,7:31] == 0).any(axis=1)
features_final = features_final[mask_no_zeros].reset_index(drop=True)
print(f"After removing rows with 0 values in columns 7-30: {features_final.shape}")

features_final.to_csv("Colorectal_Cancer_HE_patches/TCGA_Features_Final_WithMutationBurden.csv", index=False)






# ===== MARGINAL STATISTICAL MODEL ANALYSIS =====
print("\n" + "="*70)
print("MARGINAL STATISTICAL MODEL ANALYSIS")
print("="*70)

import statsmodels.api as sm
import numpy as np

# Prepare dependent variable: log(Mutation_Burden)
log_mutation_burden = np.log(features_final['Mutation_Burden'])
print(f"Log-transformed mutation burden - mean: {log_mutation_burden.mean():.3f}, std: {log_mutation_burden.std():.3f}")

# Convert proportions to percentages
features_final.iloc[:, 2:7] = features_final.iloc[:, 2:7]*100

# Get feature columns (column 3 to 33, i.e., index 3:34)
feature_columns = features_final.columns[2:33]
print(f"Analyzing {len(feature_columns)} variables from column 3 to 33")

# Initialize results list
marginal_results = []

# Fit marginal model for each variable
for i, var_name in enumerate(feature_columns):
    print(f"\nProcessing variable {i+1}/{len(feature_columns)}: {var_name}")
    
    # Get the variable data
    X = features_final[var_name].values
    y = log_mutation_burden.values
    
    # Check for missing values
    valid_mask = ~(np.isnan(X) | np.isnan(y))
    if valid_mask.sum() < 10:  # Need at least 10 valid observations
        print(f"  Skipping {var_name}: insufficient valid data ({valid_mask.sum()} observations)")
        continue
    
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    
    # Add constant for intercept
    X_with_const = sm.add_constant(X_valid)
    
    try:
        # Fit OLS model
        model = sm.OLS(y_valid, X_with_const).fit()
        
        # Extract results for the variable (index 1, since 0 is intercept)
        coef = model.params[1]
        pvalue = model.pvalues[1]
        conf_int = model.conf_int()  # 95% CI
        ci_lower = conf_int[1, 0]  # row 1 (variable), column 0 (lower bound)
        ci_upper = conf_int[1, 1]  # row 1 (variable), column 1 (upper bound)
        std_err = model.bse[1]
        
        # Store results
        marginal_results.append({
            'Variable': var_name,
            'Coefficient': coef,
            'Std_Error': std_err,
            'P_Value': pvalue,
            'CI_Lower_95': ci_lower,
            'CI_Upper_95': ci_upper,
            'N_Observations': valid_mask.sum(),
            'R_Squared': model.rsquared
        })
        
        print(f"  Coefficient: {coef:.4f}, P-value: {pvalue:.4e}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
    except Exception as e:
        print(f"  Error fitting model for {var_name}: {str(e)}")
        continue

# Create results DataFrame
output_df = pd.DataFrame(marginal_results)

print(f"\n" + "="*70)
print("MARGINAL MODEL RESULTS SUMMARY")
print("="*70)
print(f"Successfully analyzed {len(output_df)} variables")

# Sort by p-value for easier interpretation
output_df_sorted = output_df.sort_values('P_Value').reset_index(drop=True)

print(f"\nTop 10 most significant associations (sorted by p-value):")
print(output_df_sorted[['Variable', 'Coefficient', 'P_Value', 'CI_Lower_95', 'CI_Upper_95']].head(10))

# Add significance indicators
output_df_sorted['Significance'] = output_df_sorted['P_Value'].apply(
    lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else ''
)


# Save results
output_path = "Colorectal_Cancer_HE_patches/Marginal_Regression_Results.csv"
output_df_sorted.to_csv(output_path, index=False)
print(f"\nMarginal regression results saved to: {output_path}")







# ===== JOINT MODEL WITH T CELLS TO OTHER DISTANCE VARIABLES, PROPORTIONS, AND CLINICAL VARIABLES =====
print("\n" + "="*70)
print("JOINT MODEL WITH T CELLS TO OTHER DISTANCE VARIABLES, PROPORTIONS, AND CLINICAL VARIABLES")
print("="*70)

# Select variables for joint model
# 1. T cells to other cell types distance variables (both _Avg and _Std)
t_cell_distance_vars = [col for col in feature_columns if 'Distance' in col and col.startswith('T_cells_to_')]

# 2. Clinical variables
clinical_vars = [col for col in feature_columns if col in ['Patient_Age', 'Patient_Gender']]

# 3. Proportion variables (exclude Cancer to use tumor as reference category)
proportion_vars = [col for col in feature_columns if 'Overall_Proportion' in col and 'Cancer' not in col]

# Combine selected variables
selected_vars = t_cell_distance_vars + clinical_vars + proportion_vars

print(f"Selected variables for joint model:")
print(f"T cells to other distance variables ({len(t_cell_distance_vars)}): {t_cell_distance_vars}")
print(f"Clinical variables ({len(clinical_vars)}): {clinical_vars}")
print(f"Proportion variables ({len(proportion_vars)}): {proportion_vars}")
print(f"Total variables: {len(selected_vars)}")
print(f"\nNote: Using T cells to other cell types distance variables, cell type proportions, and clinical variables")

print(f"\nAll variables for joint model:")
for i, var in enumerate(selected_vars, 1):
    print(f"  {i}. {var}")

# Prepare data for joint model
X_joint = features_final[selected_vars].copy()
y_joint = log_mutation_burden.copy()

# Check for missing values
valid_mask_joint = ~(X_joint.isna().any(axis=1) | y_joint.isna())
X_joint_valid = X_joint[valid_mask_joint]
y_joint_valid = y_joint[valid_mask_joint]

print(f"\nJoint model dataset: {len(X_joint_valid)} samples, {len(selected_vars)} variables")
print(f"Samples with complete data: {len(X_joint_valid)}/{len(features_final)} ({len(X_joint_valid)/len(features_final)*100:.1f}%)")

# Add constant for intercept
X_joint_with_const = sm.add_constant(X_joint_valid)

try:
    # Fit joint OLS model
    print(f"\nFitting joint OLS model...")
    joint_model = sm.OLS(y_joint_valid, X_joint_with_const).fit()
    
    print("\n" + "="*80)
    print("T CELLS TO OTHER DISTANCE VARIABLES JOINT MODEL RESULTS")
    print("="*80)
    print(joint_model.summary())
    
    # Extract joint model results
    joint_results = []
    for i, var_name in enumerate(selected_vars):
        coef = joint_model.params[i+1]  # +1 to skip intercept
        pvalue = joint_model.pvalues[i+1]
        conf_int = joint_model.conf_int()
        ci_lower = conf_int[i+1, 0]
        ci_upper = conf_int[i+1, 1]
        std_err = joint_model.bse[i+1]
        
        joint_results.append({
            'Variable': var_name,
            'Coefficient': coef,
            'Std_Error': std_err,
            'P_Value': pvalue,
            'CI_Lower_95': ci_lower,
            'CI_Upper_95': ci_upper
        })
    
    # Create joint results DataFrame
    joint_df = pd.DataFrame(joint_results)
    joint_df['Significance'] = joint_df['P_Value'].apply(
        lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else ''
    )
    
    print(f"\n" + "="*70)
    print("T CELLS TO OTHER DISTANCE VARIABLES JOINT MODEL COEFFICIENTS")
    print("="*70)
    print(joint_df[['Variable', 'Coefficient', 'P_Value', 'CI_Lower_95', 'CI_Upper_95', 'Significance']])
    
    # Model fit statistics
    print(f"\n" + "="*70)
    print("MODEL FIT STATISTICS")
    print("="*70)
    print(f"R-squared: {joint_model.rsquared:.4f}")
    print(f"Adjusted R-squared: {joint_model.rsquared_adj:.4f}")
    print(f"F-statistic: {joint_model.fvalue:.2f}")
    print(f"F-statistic p-value: {joint_model.f_pvalue:.4e}")
    print(f"AIC: {joint_model.aic:.2f}")
    print(f"BIC: {joint_model.bic:.2f}")
    print(f"Log-likelihood: {joint_model.llf:.2f}")
    print(f"Number of observations: {joint_model.nobs}")
    print(f"Degrees of freedom: {joint_model.df_resid}")
    
    # Count significant variables in joint model
    joint_significant = joint_df[joint_df['P_Value'] < 0.05]
    print(f"\nSignificant variables in joint model (p < 0.05): {len(joint_significant)}")
    if len(joint_significant) > 0:
        print("Significant variables:")
        for _, row in joint_significant.iterrows():
            print(f"  {row['Variable']}: β = {row['Coefficient']:.4f}, p = {row['P_Value']:.4e}")
    
    # Group results by variable type for interpretation
    print(f"\n" + "="*70)
    print("RESULTS BY VARIABLE TYPE")
    print("="*70)
    
    # T cells to other distance variables
    dist_results = joint_df[joint_df['Variable'].str.contains('T_cells_to_')]
    if len(dist_results) > 0:
        print(f"T Cells to Other Cell Types Distance Variables:")
        for _, row in dist_results.iterrows():
            sig = row['Significance']
            print(f"  {row['Variable']:30} β = {row['Coefficient']:7.4f} {sig:3} (p = {row['P_Value']:.4e})")
    
    # Proportion variables
    prop_results = joint_df[joint_df['Variable'].str.contains('Overall_Proportion')]
    if len(prop_results) > 0:
        print(f"\nCell Type Proportion Variables:")
        for _, row in prop_results.iterrows():
            sig = row['Significance']
            print(f"  {row['Variable']:30} β = {row['Coefficient']:7.4f} {sig:3} (p = {row['P_Value']:.4e})")
    
    # Clinical variables
    clin_results = joint_df[joint_df['Variable'].isin(['Patient_Age', 'Patient_Gender'])]
    if len(clin_results) > 0:
        print(f"\nClinical Variables:")
        for _, row in clin_results.iterrows():
            sig = row['Significance']
            print(f"  {row['Variable']:30} β = {row['Coefficient']:7.4f} {sig:3} (p = {row['P_Value']:.4e})")
    
    # Save joint model results
    joint_output_path = "Colorectal_Cancer_HE_patches/T_Cells_Distance_Joint_Regression_Results.csv"
    joint_df.to_csv(joint_output_path, index=False)
    print(f"\nT cells distance variables joint regression results saved to: {joint_output_path}")
    
    # Save model summary
    summary_path = "Colorectal_Cancer_HE_patches/T_Cells_Distance_Joint_Model_Summary.txt"
    with open(summary_path, 'w') as f:
        f.write(str(joint_model.summary()))
    print(f"Model summary saved to: {summary_path}")
    
except Exception as e:
    print(f"Error fitting joint model: {str(e)}")
    print("This might be due to multicollinearity or insufficient data")


