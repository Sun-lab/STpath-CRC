"""
STPath-COAD: XGBoost Model Training and Prediction for Colorectal Cancer
======================================================================

This script trains XGBoost models for predicting cell type proportions in colorectal cancer
H&E patches using features extracted from multiple foundation models.

Author: Saishi Cui
Date: December 2025

Purpose: Train and evaluate XGBoost models for cell type proportion prediction using
both tile-level and individual-level cross-validation strategies. Supports feature
selection and hyperparameter optimization.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import xgboost as xgb
from datetime import datetime
from scipy import stats
from scipy.optimize import minimize
from matplotlib.patches import Patch
from umap import UMAP
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from joblib import Parallel, delayed
import pickle


def calibrate_predictions(y_true, y_pred, cell_type, n_quantiles=100, outlier_percentile=99.5):
    """
    Calibrate model predictions using global quantile mapping with linear interpolation.
    For certain cell types (T Cells, Other Immune Cells, Normal Epithelial Cells),
    first preprocess true proportions by removing outliers and scaling.
    
    Preprocessing steps (for specific cell types):
    1. Remove outliers: Keep only values <= outlier_percentile (default 99.5%)
    2. Scale: Divide all values by the max (99.5th percentile value)
    3. Apply quantile mapping on processed data
    
    This method:
    1. Computes n_quantiles percentiles of both y_pred and y_true_processed
    2. Creates a lookup table mapping y_pred quantiles to y_true_processed quantiles
    3. For new predictions, uses linear interpolation (np.interp) between quantile points
    
    Args:
        y_true: True cell type proportions (deconvoluted) - can be torch tensor or numpy array
        y_pred: Raw model predictions - can be torch tensor or numpy array
        cell_type: Name of cell type (determines if preprocessing is needed)
        n_quantiles: Number of quantiles to store (default: 100, i.e., percentiles)
        outlier_percentile: Percentile threshold for outlier removal (default: 99.5)
    
    Returns:
        tuple: (lookup_table, calibrated_predictions, y_true_processed)
        - lookup_table: Dict with quantile info and scaling parameters
        - calibrated_predictions: Calibrated values for current data
        - y_true_processed: Processed true values (for plotting)
    """
    
    # Convert torch tensors to numpy if needed
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    # Ensure numpy arrays
    y_true_original = np.asarray(y_true).flatten()
    y_pred_original = np.asarray(y_pred).flatten()
    
    print(f"  Original data: {len(y_true_original)} samples")
    print(f"  Pred range: [{y_pred_original.min():.4f}, {y_pred_original.max():.4f}]")
    print(f"  True range: [{y_true_original.min():.4f}, {y_true_original.max():.4f}]")
    
    # Check if this cell type needs true proportion preprocessing
    needs_preprocessing = cell_type in ["T Cells", "Other Immune Cells", "Normal Epithelial Cells"]
    
    if needs_preprocessing:
        print(f"\n  🔧 Preprocessing TRUE proportions for {cell_type}:")
        
        # Step 1: Remove top 0.5% outliers (keep bottom 99.5%)
        outlier_threshold = np.percentile(y_true_original, outlier_percentile)
        mask = y_true_original < outlier_threshold  # Strict inequality to remove exactly top 0.5%
        removed_count = np.sum(~mask)
        
        print(f"     ① Outlier removal: Remove top {100-outlier_percentile}%")
        print(f"        Threshold (99.5th percentile): {outlier_threshold:.4f}")
        print(f"        Removed {removed_count} outliers (≥{outlier_threshold:.4f})")
        
        # Apply mask to both arrays
        y_true_filtered = y_true_original[mask]
        y_pred_filtered = y_pred_original[mask]
        
        # Step 2: Find max value in the REMAINING 99.5% data and use it as scaling factor
        max_in_remaining = y_true_filtered.max()
        scale_factor = max_in_remaining
        y_true_scaled = y_true_filtered / scale_factor
        
        print(f"     ② Scaling: max in remaining 99.5% data = {max_in_remaining:.4f}")
        print(f"        Scale factor = {scale_factor:.4f}")
        print(f"        Scaled TRUE range: [{y_true_scaled.min():.4f}, {y_true_scaled.max():.4f}]")
        print(f"        Samples after preprocessing: {len(y_true_scaled)}")
        
        # Use processed data for calibration
        y_true_for_calibration = y_true_scaled
        y_pred_for_calibration = y_pred_filtered
        
        scaling_info = {
            'needs_preprocessing': True,
            'outlier_threshold': outlier_threshold,
            'scale_factor': scale_factor,
            'max_in_remaining': max_in_remaining,
            'removed_count': removed_count,
            'original_samples': len(y_true_original),
            'filtered_samples': len(y_true_scaled),
            'outlier_percentile': outlier_percentile
        }
    else:
        print(f"  ℹ️  No preprocessing for {cell_type}")
        y_true_for_calibration = y_true_original
        y_pred_for_calibration = y_pred_original
        scaling_info = {'needs_preprocessing': False}
    
    print(f"\n  Using global quantile mapping with {n_quantiles} quantiles")
    
    # Compute quantiles
    quantile_levels = np.linspace(0, 100, n_quantiles)
    y_pred_quantiles = np.percentile(y_pred_for_calibration, quantile_levels)
    y_true_quantiles = np.percentile(y_true_for_calibration, quantile_levels)
    
    # Create lookup table
    lookup_table = {
        'y_pred_quantiles': y_pred_quantiles,
        'y_true_quantiles': y_true_quantiles,
        'quantile_levels': quantile_levels,
        'scaling_info': scaling_info,
        'calibration_method': 'quantile_mapping'
    }
    
    print(f"  Lookup table:")
    print(f"    y_pred range: [{y_pred_quantiles[0]:.4f}, {y_pred_quantiles[-1]:.4f}]")
    print(f"    y_true range: [{y_true_quantiles[0]:.4f}, {y_true_quantiles[-1]:.4f}]")
    
    # Apply calibration to ORIGINAL full y_pred (with linear interpolation)
    calibrated_predictions = np.interp(y_pred_original, y_pred_quantiles, y_true_quantiles)
    calibrated_predictions = np.clip(calibrated_predictions, 0, 1)
    
    n_clipped = np.sum((calibrated_predictions == 0) | (calibrated_predictions == 1))
    if n_clipped > 0:
        print(f"  Clipped: {n_clipped} values ({n_clipped/len(y_pred_original)*100:.1f}%)")
    
    print(f"  ✅ Calibration completed")
    
    return lookup_table, calibrated_predictions, y_true_for_calibration


def plot_calibration_scatter(y_true, y_pred_raw, y_pred_calibrated, y_true_processed,
                            calibration_table=None, dummy=None, output_path=None, model_name=None, cell_type=None):
    """
    Create 2x2 plots:
    - Top-left: Raw predicted vs Raw true
    - Top-right: Raw predicted vs Calibrated true (scaled/processed)
    - Bottom-left: Calibrated predicted vs Calibrated true
    - Bottom-right: Calibration curve (quantile mapping visualization)
    
    Args:
        y_true: Original true cell type proportions
        y_pred_raw: Raw model predictions
        y_pred_calibrated: Calibrated predictions
        y_true_processed: Processed true proportions (scaled/filtered for certain cell types)
        calibration_table: The lookup table with quantile info
        dummy: Placeholder for compatibility
        output_path: Path to save the plot
        model_name: Name of the model
        cell_type: Cell type name
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    from scipy.stats import spearmanr, pearsonr
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    # Handle processed data
    # If no preprocessing was done, y_true_processed is the same as y_true
    scaling_info = calibration_table.get('scaling_info', {'needs_preprocessing': False})
    needs_preprocessing = scaling_info.get('needs_preprocessing', False)
    
    if needs_preprocessing:
        # Need to filter y_pred and y_true to match y_true_processed
        outlier_threshold = scaling_info['outlier_threshold']
        mask = y_true <= outlier_threshold
        y_pred_filtered = y_pred_raw[mask]
        y_true_filtered = y_true[mask]
        
        # Scale true values
        scale_factor = scaling_info['scale_factor']
        y_true_scaled = y_true_filtered / scale_factor
    else:
        y_pred_filtered = y_pred_raw
        y_true_scaled = y_true
        y_true_filtered = y_true
    
    # Create 1x3 subplot layout
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    x_line = np.array([0, 1])
    
    ##########################################################################
    ### Left: Raw predicted vs Raw true
    ##########################################################################
    ax = axes[0]
    mae1 = mean_absolute_error(y_true, y_pred_raw)
    rmse1 = np.sqrt(mean_squared_error(y_true, y_pred_raw))
    spearman1, _ = spearmanr(y_true, y_pred_raw)
    pearson1, _ = pearsonr(y_true, y_pred_raw)
    slope1, intercept1, r1, _, _ = stats.linregress(y_true, y_pred_raw)
    
    ax.scatter(y_true, y_pred_raw, alpha=0.4, s=10, color='blue', edgecolor='none')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Identity')
    ax.plot(x_line, intercept1 + slope1 * x_line, 'r-', linewidth=2, alpha=0.7, label='Fit')
    
    textstr1 = f'MAE: {mae1:.4f}\nRMSE: {rmse1:.4f}\n'
    textstr1 += f'Spearman: {spearman1:.4f}\nPearson: {pearson1:.4f}\n'
    textstr1 += f'Slope: {slope1:.4f}\nR²: {r1**2:.4f}'
    ax.text(0.05, 0.95, textstr1, transform=ax.transAxes, fontsize=11, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Raw True Proportion', fontsize=13, fontweight='bold')
    ax.set_ylabel('Raw Predicted Proportion', fontsize=13, fontweight='bold')
    ax.set_title(f'(A) Raw Predicted vs Raw True\n{model_name} - {cell_type}', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='lower right')
    
    ##########################################################################
    ### Middle: Raw predicted vs Calibrated (scaled) true
    ##########################################################################
    ax = axes[1]
    mae2 = mean_absolute_error(y_true_scaled, y_pred_filtered)
    rmse2 = np.sqrt(mean_squared_error(y_true_scaled, y_pred_filtered))
    spearman2, _ = spearmanr(y_true_scaled, y_pred_filtered)
    pearson2, _ = pearsonr(y_true_scaled, y_pred_filtered)
    slope2, intercept2, r2, _, _ = stats.linregress(y_true_scaled, y_pred_filtered)
    
    ax.scatter(y_true_scaled, y_pred_filtered, alpha=0.4, s=10, color='green', edgecolor='none')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Identity')
    ax.plot(x_line, intercept2 + slope2 * x_line, 'r-', linewidth=2, alpha=0.7, label='Fit')
    
    textstr2 = f'MAE: {mae2:.4f}\nRMSE: {rmse2:.4f}\n'
    textstr2 += f'Spearman: {spearman2:.4f}\nPearson: {pearson2:.4f}\n'
    textstr2 += f'Slope: {slope2:.4f}\nR²: {r2**2:.4f}'
    if needs_preprocessing:
        textstr2 += f'\n\n✂️ Outliers removed\n📏 Scaled by {scaling_info["scale_factor"]:.4f}'
    ax.text(0.05, 0.95, textstr2, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax.set_xlabel('Calibrated True Proportion', fontsize=13, fontweight='bold')
    ax.set_ylabel('Raw Predicted Proportion', fontsize=13, fontweight='bold')
    ax.set_title(f'(B) Raw Predicted vs Calibrated True\n{model_name} - {cell_type}', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='lower right')
    
    ##########################################################################
    ### Right: Quantile Mapping Curve (100 points connected by lines)
    ##########################################################################
    ax = axes[2]
    
    if calibration_table is not None:
        # Get the quantile points
        y_pred_quantiles = calibration_table['y_pred_quantiles']
        y_true_quantiles = calibration_table['y_true_quantiles']
        
        # Plot the quantile mapping as a line connecting points
        ax.plot(y_pred_quantiles, y_true_quantiles, 'b-o', linewidth=2, markersize=4, 
                alpha=0.7, label='Quantile Mapping')
        
        # Plot identity line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Identity')
        
        # Add text info
        n_quantiles = len(y_pred_quantiles)
        textstr_qm = f'Quantile Mapping\n'
        textstr_qm += f'{n_quantiles} quantiles\n'
        textstr_qm += f'Linear interpolation\n\n'
        textstr_qm += f'y_pred range:\n[{y_pred_quantiles[0]:.4f}, {y_pred_quantiles[-1]:.4f}]\n\n'
        textstr_qm += f'y_true range:\n[{y_true_quantiles[0]:.4f}, {y_true_quantiles[-1]:.4f}]'
        
        if scaling_info.get('needs_preprocessing', False):
            textstr_qm += f'\n\n✂️ Preprocessing applied'
        
        ax.text(0.05, 0.95, textstr_qm, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.set_xlabel('Raw Predicted Proportion', fontsize=13, fontweight='bold')
        ax.set_ylabel('Calibrated Predicted Proportion', fontsize=13, fontweight='bold')
        ax.set_title(f'(C) Quantile Mapping Curve\n{model_name} - {cell_type}', fontsize=14, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='lower right')
    else:
        ax.text(0.5, 0.5, 'Calibration table not available', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
    
    plt.suptitle(f'Calibration Analysis: {cell_type}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Calibration plot (1x3) saved: {output_path}")


### Train XGBoost model with leave-some-out cross-validation


def train_xgboost_tile_level(
    input_dir='Colorectal_Cancer_HE_patches/Training_features',
    cell_type="Cancer Cells",
    model_name='UNI2h',
    output_dir=None,
    params=None,
    num_boost_round=500,
    seed=42,
    if_combined=False
):
    """
    Tile-level 5-fold cross-validation
    
    Args:
        input_dir: Feature input directory
        cell_type: Cell type for prediction
        model_name: Model name for output directory naming
        output_dir: Output directory, if None it will be automatically generated
        params: XGBoost parameters, if None default parameters will be used
        num_boost_round: Fixed number of training rounds
        seed: Random seed
        if_combined: Whether to use combined features from multiple models
    
    Returns:
        dict: Training results and evaluation metrics
    """
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_dir is None:
        output_dir = f'Colorectal_Cancer_HE_patches/xgboost_prediction/{cell_type}_{model_name}_tile_level'
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default XGBoost parameters with CPU optimization
    if params is None:
        # Optimized CPU parameters - consistent with hyperparameter_grid_search
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',  # Fast CPU method
            'n_jobs': -1,  # Use all CPU cores
            'eta': 0.01,  # Fixed - consistent with hyperparameter_grid_search
            'gamma': 0,  # Fixed - consistent with hyperparameter_grid_search
            'min_child_weight': 5,  # Fixed - consistent with hyperparameter_grid_search
            'colsample_bytree': 0.05,  # Fixed - consistent with hyperparameter_grid_search
            'subsample': 0.5,  # Fixed - consistent with hyperparameter_grid_search
            'alpha': 0.1,  # Fixed - consistent with hyperparameter_grid_search
            'lambda': 0.01,  # Fixed - consistent with hyperparameter_grid_search
            'max_depth': 12,  # Default value, can be optimized through hyperparameter search
            'seed': seed
        }
        print("✅ Using XGBoost with optimized CPU acceleration (params consistent with hyperparameter_grid_search)")
    
    # Load feature data
    if if_combined:
        UNI2h_feature_name = f'{cell_type}_training_precomputed_features_UNI2h.pt'
        Virchow_feature_name = f'{cell_type}_training_precomputed_features_Virchow.pt'
        Virchow2_feature_name = f'{cell_type}_training_precomputed_features_Virchow2.pt'
        ProvGigapath_feature_name = f'{cell_type}_training_precomputed_features_ProvGigapath.pt'
        Conch_feature_name = f'{cell_type}_training_precomputed_features_Conch.pt'
        
        UNI2h_data = torch.load(input_dir + "/" + UNI2h_feature_name)
        Virchow_data = torch.load(input_dir + "/" + Virchow_feature_name)
        Virchow2_data = torch.load(input_dir + "/" + Virchow2_feature_name)
        ProvGigapath_data = torch.load(input_dir + "/" + ProvGigapath_feature_name)
        Conch_data = torch.load(input_dir + "/" + Conch_feature_name)

        features = torch.cat([UNI2h_data['embeddings'], Virchow_data['embeddings'], 
                             Virchow2_data['embeddings'], ProvGigapath_data['embeddings'], 
                             Conch_data['embeddings']], dim=1)
        data = UNI2h_data
        labels = data['celltype_proportions']
    else:
        feature_name = f'{cell_type}_training_precomputed_features_{model_name}.pt'
        data_path = input_dir + "/" + feature_name
        print(f"Loading data: {data_path}")
        data = torch.load(data_path)
        features = data['embeddings']
        labels = data['celltype_proportions']

    # Get tile IDs
    tile_ids = data.get('tile_ids', [])
    
    print(f"Dataset contains: {len(tile_ids)} tiles")
    
    # Randomly split tiles into 5 folds
    tile_indices = list(range(len(tile_ids)))
    np.random.shuffle(tile_indices)
    fold_size = len(tile_indices) // 5
    fold_indices = []
    for i in range(5):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < 4 else len(tile_indices)
        test_indices = tile_indices[start_idx:end_idx]
        train_indices = [idx for idx in tile_indices if idx not in test_indices]
        fold_indices.append((train_indices, test_indices))
    
    # Initialize variables to store prediction results
    all_predictions = np.zeros_like(labels)
    all_importances = []
    is_test = np.zeros(len(labels), dtype=bool)
    fold_metrics = []
    
    # Iterate through each fold
    for fold_idx, (train_indices, test_indices) in enumerate(fold_indices):
        print(f"\n{'-'*80}")
        print(f"Starting Tile-level fold {fold_idx+1}/5 prediction...")
        print(f"{'-'*80}")
        
        # Update test mask
        is_test[test_indices] = True
        
        # Prepare training and test data
        X_train, y_train = features[train_indices], labels[train_indices]
        X_test, y_test = features[test_indices], labels[test_indices]
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Train XGBoost model
        print(f"Training XGBoost model...")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Train model with fixed number of iterations
        model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=num_boost_round,
            evals=[(dtrain, 'train')],
            verbose_eval=100
        )
        
        # Feature importance
        importance = model.get_score(importance_type='gain')
        all_importances.append(importance)
        
        # Predict test samples
        print(f"Predicting test samples...")
        predictions = model.predict(dtest)
        
        # Store predictions
        all_predictions[test_indices] = predictions
        
        # Calculate metrics for this fold
        fold_mae = mean_absolute_error(y_test, predictions)
        fold_rmse = np.sqrt(mean_squared_error(y_test, predictions))
        fold_spearman, _ = spearmanr(y_test, predictions)
        fold_pearson, _ = pearsonr(y_test, predictions)
        
        # Calculate min and max celltype proportion for this fold
        fold_celltype_min = round(y_test.min().item(), 3)
        fold_celltype_max = round(y_test.max().item(), 3)
        
        # Store fold metrics
        fold_metrics.append({
            'Fold': fold_idx + 1,
            'Test_Element': f"Fold_{fold_idx+1}",
            'MAE': fold_mae,
            'RMSE': fold_rmse,
            'Spearman': fold_spearman,
            'Pearson': fold_pearson,
            'n_test_samples': len(test_indices),
            'Min_celltype_proportion': fold_celltype_min,
            'Max_celltype_proportion': fold_celltype_max
        })
        
        print(f"Fold {fold_idx+1} metrics:")
        print(f"  MAE={fold_mae:.4f}, RMSE={fold_rmse:.4f}, Spearman={fold_spearman:.4f}, Pearson={fold_pearson:.4f}")
        print(f"  Celltype proportion range: [{fold_celltype_min:.3f}, {fold_celltype_max:.3f}]")
        
        # Save model
        model_dir = os.path.join(output_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        model.save_model(os.path.join(model_dir, f'xgboost_model_fold_{fold_idx+1}.model'))
    
    # Calculate average fold metrics
    fold_metrics_df = pd.DataFrame(fold_metrics)
    average_metrics = {
        'MAE': fold_metrics_df['MAE'].mean(),
        'RMSE': fold_metrics_df['RMSE'].mean(),
        'Spearman': fold_metrics_df['Spearman'].mean(),
        'Pearson': fold_metrics_df['Pearson'].mean(),
        'MAE_std': fold_metrics_df['MAE'].std(),
        'RMSE_std': fold_metrics_df['RMSE'].std(),
        'Spearman_std': fold_metrics_df['Spearman'].std(),
        'Pearson_std': fold_metrics_df['Pearson'].std()
    }
    
    print("\nAverage fold metrics:")
    print(f"  MAE={average_metrics['MAE']:.4f} ± {average_metrics['MAE_std']:.4f}")
    print(f"  RMSE={average_metrics['RMSE']:.4f} ± {average_metrics['RMSE_std']:.4f}")
    print(f"  Spearman={average_metrics['Spearman']:.4f} ± {average_metrics['Spearman_std']:.4f}")
    print(f"  Pearson={average_metrics['Pearson']:.4f} ± {average_metrics['Pearson_std']:.4f}")
    
    # Save fold metrics
    fold_metrics_csv_path = os.path.join(output_dir, 'fold_metrics_tile_level.csv')
    fold_metrics_df.to_csv(fold_metrics_csv_path, index=False)
    print(f"Fold metrics saved to: {fold_metrics_csv_path}")
    
    # Calculate overall metrics
    overall_mae = mean_absolute_error(labels, all_predictions)
    overall_rmse = np.sqrt(mean_squared_error(labels, all_predictions))
    overall_spearman, _ = spearmanr(labels, all_predictions)
    overall_pearson, _ = pearsonr(labels, all_predictions)
    
    print("\nOverall metrics:")
    print(f"  MAE={overall_mae:.4f}, RMSE={overall_rmse:.4f}, Spearman={overall_spearman:.4f}, Pearson={overall_pearson:.4f}")
    
    # Create overall scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(labels, all_predictions, alpha=0.7, s=15, color='blue')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
    
    # Calculate and draw regression line
    slope, intercept, _, _, _ = stats.linregress(labels, all_predictions)
    plt.plot([0, 1], [intercept, intercept + slope], 'r-', alpha=0.7)
    
    textstr = f'MAE: {overall_mae:.4f}\nRMSE: {overall_rmse:.4f}\nSpearman: {overall_spearman:.4f}\nPearson: {overall_pearson:.4f}\nSlope: {slope:.4f}\nn: {len(labels)}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.title(f'Tile-level Performance ({model_name})', fontsize=14, fontweight='bold')
    plt.xlabel('True Cell Type Proportion', fontsize=12)
    plt.ylabel('Predicted Cell Type Proportion', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save overall scatter plot
    plt.savefig(os.path.join(output_dir, 'overall_performance_tile_level.png'), dpi=300)
    plt.close()
    
    # Save prediction results
    results = {
        'predictions': all_predictions,
        'labels': labels,
        'tile_ids': tile_ids,
        'is_test': is_test,
        'feature_importances': all_importances,
        'params': params,
        'model_name': model_name,
        'cell_type': cell_type,
        'level': 'tile',
        'timestamp': timestamp,
        'output_dir': output_dir,
        'metrics': {
            'overall_mae': overall_mae,
            'overall_rmse': overall_rmse,
            'overall_spearman': overall_spearman,
            'overall_pearson': overall_pearson,
        },
        'fold_metrics': fold_metrics,
        'average_metrics': average_metrics
    }
    
    # Save results
    results_path = os.path.join(output_dir, 'xgboost_results_tile_level.pt')
    torch.save(results, results_path)
    print(f"All prediction results saved to: {results_path}")
    
    return results


def train_xgboost_individual_level(
    input_dir='Colorectal_Cancer_HE_patches/Training_features',
    cell_type="Cancer Cells",
    model_name='UNI2h',
    output_dir=None,
    params=None,
    num_boost_round=500,
    seed=42,
    if_combined=False,
    training_data_ratio=1.0,
    save_files=True
):
    """
    Individual-level leave-one-out training with individual-level testing
    
    Args:
        input_dir: Feature input directory
        cell_type: Cell type for prediction
        model_name: Model name for output directory naming
        output_dir: Output directory, if None it will be automatically generated
        params: XGBoost parameters, if None default parameters will be used
        num_boost_round: Fixed number of training rounds
        seed: Random seed
        if_combined: Whether to use combined features from multiple models
        training_data_ratio: Ratio of training data to use (0.1=10%, 1.0=100%)
        save_files: Whether to save model files and detailed outputs to disk
    
    Returns:
        dict: Training results and evaluation metrics
    """
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set output directory (only if saving files)
    if save_files:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if output_dir is None:
            ratio_str = f"ratio{int(training_data_ratio*100)}"
            output_dir = f'Colorectal_Cancer_HE_patches/xgboost_prediction/{cell_type}_{model_name}_individual_level_{ratio_str}'
        os.makedirs(output_dir, exist_ok=True)
    
    # Set default XGBoost parameters with CPU optimization
    # Use same fixed parameters as hyperparameter_grid_search, only max_depth set to default value
    if params is None:
        # Optimized CPU parameters - consistent with hyperparameter_grid_search
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',  
            'n_jobs': -1,  
            'eta': 0.01,  
            'gamma': 0,  
            'min_child_weight': 5,  
            'colsample_bytree': 0.05,  
            'subsample': 0.5,  
            'alpha': 0.1,  
            'lambda': 0.01,  
            'max_depth': 12,  
            'seed': seed
        }
        print("✅ Using XGBoost with optimized CPU acceleration (params consistent with hyperparameter_grid_search)")
    
    # Load feature data
    if if_combined:
        # Load important features dictionary
        important_features_path = f"/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/xgboost_prediction/important_features_{cell_type}.pkl"
        important_features = pickle.load(open(important_features_path, "rb"))
        
        UNI2h_feature_name = f'{cell_type}_training_precomputed_features_UNI2h.pt'
        Virchow_feature_name = f'{cell_type}_training_precomputed_features_Virchow.pt'
        Virchow2_feature_name = f'{cell_type}_training_precomputed_features_Virchow2.pt'
        ProvGigapath_feature_name = f'{cell_type}_training_precomputed_features_ProvGigapath.pt'
        Conch_feature_name = f'{cell_type}_training_precomputed_features_Conch.pt'
        
        UNI2h_data = torch.load(input_dir + "/" + UNI2h_feature_name)
        Virchow_data = torch.load(input_dir + "/" + Virchow_feature_name)
        Virchow2_data = torch.load(input_dir + "/" + Virchow2_feature_name)
        ProvGigapath_data = torch.load(input_dir + "/" + ProvGigapath_feature_name)
        Conch_data = torch.load(input_dir + "/" + Conch_feature_name)

        # Select only important features for each model
        UNI2h_important_indices = [int(f.replace('f', '')) for f in important_features["UNI2h"]]
        Virchow_important_indices = [int(f.replace('f', '')) for f in important_features["Virchow"]]
        Virchow2_important_indices = [int(f.replace('f', '')) for f in important_features["Virchow2"]]
        ProvGigapath_important_indices = [int(f.replace('f', '')) for f in important_features["ProvGigapath"]]
        Conch_important_indices = [int(f.replace('f', '')) for f in important_features["Conch"]]
        
        UNI2h_selected = UNI2h_data['embeddings'][:, UNI2h_important_indices]
        Virchow_selected = Virchow_data['embeddings'][:, Virchow_important_indices]
        Virchow2_selected = Virchow2_data['embeddings'][:, Virchow2_important_indices]
        ProvGigapath_selected = ProvGigapath_data['embeddings'][:, ProvGigapath_important_indices]
        Conch_selected = Conch_data['embeddings'][:, Conch_important_indices]

        features = torch.cat([UNI2h_selected, Virchow_selected, Virchow2_selected, ProvGigapath_selected, Conch_selected], dim=1)
        data = UNI2h_data
        labels = data['celltype_proportions']
        
        print(f"✅ Using important features: UNI2h({len(UNI2h_important_indices)}), Virchow({len(Virchow_important_indices)}), Virchow2({len(Virchow2_important_indices)}), ProvGigapath({len(ProvGigapath_important_indices)}), Conch({len(Conch_important_indices)})")
        print(f"Total selected features: {features.shape[1]}")
    else:
        feature_name = f'{cell_type}_training_precomputed_features_{model_name}.pt'
        data_path = input_dir + "/" + feature_name
        print(f"Loading data: {data_path}")
        data = torch.load(data_path)
        features = data['embeddings']
        labels = data['celltype_proportions']

    # Get ID information
    tile_ids = data.get('tile_ids', [])
    individual_ids = data.get('individual_ids', [])
    sample_ids = data.get('sample_ids', [])
    
    # Get unique individuals
    unique_individuals = sorted(list(set(individual_ids)))
    print(f"Dataset contains: {len(unique_individuals)} individuals")
    
    # Initialize variables to store prediction results
    all_predictions = np.zeros_like(labels)
    all_importances = []
    is_test = np.zeros(len(labels), dtype=bool)
    fold_metrics = []
    
    # Iterate through each individual
    for ind_idx, ind in enumerate(unique_individuals):
        print(f"\n{'-'*80}")
        print(f"Starting Individual-level prediction, leaving individual {ind} out ({ind_idx+1}/{len(unique_individuals)})...")
        print(f"{'-'*80}")
        
        # Get all samples for this individual
        individual_mask = np.array([iid == ind for iid in individual_ids])
        individual_sample_ids = [sample_ids[i] for i in range(len(sample_ids)) if individual_mask[i]]
        unique_individual_samples = sorted(list(set(individual_sample_ids)))
        
        print(f"Individual {ind} has {len(unique_individual_samples)} samples: {unique_individual_samples}")
        
        # Train data: all other individuals (exclude the entire individual)
        train_mask = np.array([iid != ind for iid in individual_ids])
        all_train_indices = np.where(train_mask)[0]
        
        # Subsample training data if ratio < 1.0
        if training_data_ratio < 1.0:
            # Set random state for reproducibility
            np.random.seed(seed + ind_idx)  # Different seed for each individual to avoid bias
            n_samples = int(len(all_train_indices) * training_data_ratio)
            train_indices = np.random.choice(all_train_indices, size=n_samples, replace=False)
            train_indices = np.sort(train_indices)  # Sort for consistency
        else:
            train_indices = all_train_indices
        
        # Prepare training data
        X_train, y_train = features[train_indices], labels[train_indices]
        
        print(f"Training XGBoost model for individual {ind}...")
        print(f"Available training samples: {len(all_train_indices)}, Using: {len(X_train)} ({training_data_ratio*100:.1f}%)")
        
        # Train XGBoost model once for this individual
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Train model with fixed number of iterations
        model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=num_boost_round,
            evals=[(dtrain, 'train')],
            verbose_eval=100
        )
        
        # Feature importance
        importance = model.get_score(importance_type='gain')
        all_importances.append(importance)
        
        # Save model (once per individual) - only if save_files is True
        if save_files:
            model_dir = os.path.join(output_dir, 'models')
            os.makedirs(model_dir, exist_ok=True)
            model.save_model(os.path.join(model_dir, f'xgboost_model_leave_{ind}_out.model'))
        
        # Test on the entire individual (all samples of this individual)
        test_mask = np.array([iid == ind for iid in individual_ids])
        test_indices = np.where(test_mask)[0]
        
        # Update test mask
        is_test[test_indices] = True
        
        # Prepare test data
        X_test, y_test = features[test_indices], labels[test_indices]
        
        print(f"  Testing on individual {ind}: {len(X_test)} samples")
        
        # Predict test samples
        dtest = xgb.DMatrix(X_test, label=y_test)
        predictions = model.predict(dtest)
        
        # Store predictions
        all_predictions[test_indices] = predictions
        
        # Calculate metrics for this individual
        individual_mae = mean_absolute_error(y_test, predictions)
        individual_rmse = np.sqrt(mean_squared_error(y_test, predictions))
        individual_spearman, _ = spearmanr(y_test, predictions)
        individual_pearson, _ = pearsonr(y_test, predictions)
        
        # Calculate min and max celltype proportion for this individual
        individual_celltype_min = round(y_test.min().item(), 3)
        individual_celltype_max = round(y_test.max().item(), 3)
        
        # Store individual metrics
        fold_metrics.append({
            'Fold': ind,
            'Test_Element': ind,
            'Individual': ind,
            'MAE': individual_mae,
            'RMSE': individual_rmse,
            'Spearman': individual_spearman,
            'Pearson': individual_pearson,
            'n_test_samples': len(test_indices),
            'Min_celltype_proportion': individual_celltype_min,
            'Max_celltype_proportion': individual_celltype_max
        })
        
        print(f"    Individual {ind} metrics: MAE={individual_mae:.4f}, RMSE={individual_rmse:.4f}, Spearman={individual_spearman:.4f}, Pearson={individual_pearson:.4f}")
        print(f"    Celltype proportion range: [{individual_celltype_min:.3f}, {individual_celltype_max:.3f}]")
    
    # Calculate average individual metrics
    fold_metrics_df = pd.DataFrame(fold_metrics)
    average_metrics = {
        'MAE': fold_metrics_df['MAE'].mean(),
        'RMSE': fold_metrics_df['RMSE'].mean(),
        'Spearman': fold_metrics_df['Spearman'].mean(),
        'Pearson': fold_metrics_df['Pearson'].mean(),
        'MAE_std': fold_metrics_df['MAE'].std(),
        'RMSE_std': fold_metrics_df['RMSE'].std(),
        'Spearman_std': fold_metrics_df['Spearman'].std(),
        'Pearson_std': fold_metrics_df['Pearson'].std()
    }
    
    print("\nAverage individual metrics:")
    print(f"  MAE={average_metrics['MAE']:.4f} ± {average_metrics['MAE_std']:.4f}")
    print(f"  RMSE={average_metrics['RMSE']:.4f} ± {average_metrics['RMSE_std']:.4f}")
    print(f"  Spearman={average_metrics['Spearman']:.4f} ± {average_metrics['Spearman_std']:.4f}")
    print(f"  Pearson={average_metrics['Pearson']:.4f} ± {average_metrics['Pearson_std']:.4f}")
    
    # Calculate overall metrics
    overall_mae = mean_absolute_error(labels, all_predictions)
    overall_rmse = np.sqrt(mean_squared_error(labels, all_predictions))
    overall_spearman, _ = spearmanr(labels, all_predictions)
    overall_pearson, _ = pearsonr(labels, all_predictions)
    
    print("\nOverall metrics (raw predictions):")
    print(f"  MAE={overall_mae:.4f}, RMSE={overall_rmse:.4f}, Spearman={overall_spearman:.4f}, Pearson={overall_pearson:.4f}")
    
    # Perform calibration
    print(f"\n{'='*80}")
    print(f"Performing cross-model calibration")
    print(f"{'='*80}\n")
    
    calibration_table, calibrated_predictions, labels_processed = calibrate_predictions(labels, all_predictions, cell_type)
    
    # Calculate metrics for calibrated predictions
    cal_mae = mean_absolute_error(labels, calibrated_predictions)
    cal_rmse = np.sqrt(mean_squared_error(labels, calibrated_predictions))
    cal_spearman, _ = spearmanr(labels, calibrated_predictions)
    cal_pearson, _ = pearsonr(labels, calibrated_predictions)
    
    print("\nOverall metrics (calibrated predictions):")
    print(f"  MAE={cal_mae:.4f}, RMSE={cal_rmse:.4f}, Spearman={cal_spearman:.4f}, Pearson={cal_pearson:.4f}")
    print(f"  Improvement: MAE Δ={overall_mae-cal_mae:+.4f}, RMSE Δ={overall_rmse-cal_rmse:+.4f}")
    
    # Only save files if save_files is True
    if save_files:
        # Save individual metrics
        fold_metrics_csv_path = os.path.join(output_dir, 'individual_metrics_individual_level.csv')
        fold_metrics_df.to_csv(fold_metrics_csv_path, index=False)
        print(f"Individual metrics saved to: {fold_metrics_csv_path}")
        
        # Create overall scatter plot
        plt.figure(figsize=(8, 8))
        plt.scatter(labels, all_predictions, alpha=0.7, s=15, color='blue')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
        
        # Calculate and draw regression line
        slope, intercept, _, _, _ = stats.linregress(labels, all_predictions)
        plt.plot([0, 1], [intercept, intercept + slope], 'r-', alpha=0.7)
        
        textstr = f'MAE: {overall_mae:.4f}\nRMSE: {overall_rmse:.4f}\nSpearman: {overall_spearman:.4f}\nPearson: {overall_pearson:.4f}\nSlope: {slope:.4f}\nn: {len(labels)}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)
        
        plt.title(f'Individual-level Performance ({model_name}, Training Ratio: {training_data_ratio*100:.0f}%)', fontsize=14, fontweight='bold')
        plt.xlabel('True Cell Type Proportion', fontsize=12)
        plt.ylabel('Predicted Cell Type Proportion', fontsize=12)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save overall scatter plot
        plt.savefig(os.path.join(output_dir, 'overall_performance_individual_level.png'), dpi=300)
        plt.close()
        
        # Create and save calibration comparison plot
        print("\nCreating calibration comparison plot...")
        calibration_plot_path = os.path.join(output_dir, 'calibration_comparison.png')
        plot_calibration_scatter(
            y_true=labels,
            y_pred_raw=all_predictions,
            y_pred_calibrated=calibrated_predictions,
            y_true_processed=labels_processed,
            calibration_table=calibration_table,
            dummy=None,
            output_path=calibration_plot_path,
            model_name=model_name,
            cell_type=cell_type
        )
        
        # Save detailed tile-level predictions to CSV (Most important output!)
        print("\nSaving detailed tile-level predictions CSV...")
        tile_predictions_df = pd.DataFrame({
            'tile_id': tile_ids,
            'sample_id': sample_ids,
            'individual_id': individual_ids,
            'deconvoluted_proportion': labels,
            'predicted_proportion_raw': all_predictions,
            'predicted_proportion_calibrated': calibrated_predictions
        })
        
        tile_predictions_csv_path = os.path.join(output_dir, 'tile_level_predictions_detailed.csv')
        tile_predictions_df.to_csv(tile_predictions_csv_path, index=False)
        print(f"Tile-level predictions saved to: {tile_predictions_csv_path}")
        print(f"  Total tiles: {len(tile_predictions_df):,}")
        print(f"  Columns: {', '.join(tile_predictions_df.columns)}")
    
    # Prepare results dictionary
    results = {
        'predictions': all_predictions,
        'predictions_calibrated': calibrated_predictions,
        'labels': labels,
        'tile_ids': tile_ids,
        'individual_ids': individual_ids,
        'sample_ids': sample_ids,
        'is_test': is_test,
        'feature_importances': all_importances,
        'params': params,
        'model_name': model_name,
        'cell_type': cell_type,
        'level': 'individual',
        'training_data_ratio': training_data_ratio,
        'output_dir': output_dir if save_files else None,
        'metrics': {
            'overall_mae': overall_mae,
            'overall_rmse': overall_rmse,
            'overall_spearman': overall_spearman,
            'overall_pearson': overall_pearson,
        },
        'calibration': {
            'lookup_table': calibration_table,
            'calibrated_mae': cal_mae,
            'calibrated_rmse': cal_rmse,
            'calibrated_spearman': cal_spearman,
            'calibrated_pearson': cal_pearson
        },
        'individual_metrics': fold_metrics,
        'average_metrics': average_metrics
    }
    
    # Save prediction results to file (only if save_files is True)
    if save_files and output_dir:
        results_path = os.path.join(output_dir, 'xgboost_results_individual_level.pt')
        torch.save(results, results_path)
        print(f"All prediction results saved to: {results_path}")
    
    return results



## Hyperparameter Grid Search
def hyperparameter_grid_search():
    """
    Fixed parameters grid search for XGBoost with CPU optimization
    Fixed: eta=0.01, min_child_weight=5, gamma=0, colsample_bytree=0.05, subsample=0.5, alpha=0.1, lambda=0.01
    Search: max_depth
    """
    
    # Optimized CPU fixed parameters
    fixed_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',  # Fast CPU method
        'n_jobs': -1,  # Use all CPU cores
        'eta': 0.01,  # Fixed
        'gamma': 0,  # Fixed
        'min_child_weight': 5,  # Fixed
        "colsample_bytree": 0.05,  #Fixed
        "subsample": 0.5,  #Fixed
        "alpha": 0.1,  #Fixed
        'lambda': 0.01,  # Fixed
        'seed': 42
    }
    print("✅ Using optimized CPU acceleration for hyperparameter search")
    
    # Fixed num_boost_round
    fixed_num_boost_round = 300
    
    # Grid search parameters
    search_grid = {
        'max_depth': [11, 12, 13, 14, 15],
    }
    
    import itertools
    import time
    
    # Generate all parameter combinations
    param_names = list(search_grid.keys())
    param_values = list(search_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"🔍 Starting hyperparameter grid search...")
    print(f"📊 Fixed parameters: {fixed_params}")
    print(f"🎛️ Search space: {len(param_combinations)} combinations")
    print(f"⏱️ Estimated time: ~{len(param_combinations) * 2:.0f} minutes")
    
    # Prepare output file
    output_dir = 'Colorectal_Cancer_HE_patches/xgboost_hyperparam_search'
    os.makedirs(output_dir, exist_ok=True)
    results_csv_path = os.path.join(output_dir, 'grid_search_results.csv')
    
    results = []
    
    for i, param_combo in enumerate(param_combinations):
        print(f"\n{'='*80}")
        print(f"🔄 Combination {i+1}/{len(param_combinations)}")
        
        # Create complete parameter dict (including num_boost_round)
        current_params = fixed_params.copy()
        
        for param_name, param_value in zip(param_names, param_combo):
            current_params[param_name] = param_value
                

        start_time = time.time()
        
        try:
            # Use Cancer Cells as representative cell type for quick evaluation
            results_dict = train_xgboost_individual_level(
                input_dir='Colorectal_Cancer_HE_patches/Training_features',
                cell_type="Cancer Cells",
                model_name="Combined",
                output_dir=None,  # No output directory needed
                params=current_params,
                num_boost_round= fixed_num_boost_round,
                seed=42,
                if_combined=True,
                training_data_ratio=0.5,  # Use 50% data for faster evaluation
                save_files=False  # Don't save individual files during grid search
            )
            
            # Extract individual-level metrics for averaging and median calculation
            individual_metrics = results_dict['individual_metrics']
            individual_maes = [metric['MAE'] for metric in individual_metrics]
            individual_pearsons = [metric['Pearson'] for metric in individual_metrics]
            
            # Calculate averaged and median metrics across individuals
            avg_mae = np.mean(individual_maes)
            avg_pearson = np.mean(individual_pearsons)
            median_mae = np.median(individual_maes)
            median_pearson = np.median(individual_pearsons)
            
            elapsed_time = time.time() - start_time
            
            # Store results with the requested format
            result_entry = {
                'max_depth': current_params['max_depth'],
                'averaged_MAE': avg_mae,
                'averaged_Pearson': avg_pearson,
                'median_MAE': median_mae,
                'median_Pearson': median_pearson,
                'elapsed_time': elapsed_time
            }
            
            results.append(result_entry)
            
            # Save immediately after each combination
            import pandas as pd
            current_df = pd.DataFrame(results)
            current_df = current_df.sort_values('averaged_Pearson', ascending=False)
            current_df.to_csv(results_csv_path, index=False)
            
            print(f"✅ Results: Averaged Pearson={avg_pearson:.4f}, Averaged MAE={avg_mae:.4f}")
            print(f"📊 Median: Pearson={median_pearson:.4f}, MAE={median_mae:.4f}")
            print(f"⏱️ Time: {elapsed_time:.1f}s | 💾 Saved to CSV")
            
        except Exception as e:
            print(f"❌ Error in combination {i+1}: {str(e)}")
            continue
    
    # Final DataFrame (already saved during the loop)
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('averaged_Pearson', ascending=False)
 
    return results_df


## Run Hyperparameter Search
results_df = hyperparameter_grid_search()



for model in ["Resnet50", 'UNI2h', 'Virchow', 'Virchow2', 'ProvGigapath', 'Conch']:
    for cell_type in ["Cancer Cells", "Stromal Cells", "Normal Epithelial Cells", "T Cells", "Other Immune Cells"]:
        results = train_xgboost_individual_level(
            input_dir='Colorectal_Cancer_HE_patches/Training_features',
            cell_type=cell_type,
            model_name=model,
            output_dir=None,
            params=None,
            num_boost_round=800,
            seed=42,
            if_combined=False,
            training_data_ratio=1.0,
            save_files=True
        )








for cell_type in ["Cancer Cells", "Stromal Cells", "Normal Epithelial Cells", "T Cells", "Other Immune Cells"]:
    model = "Combined"
    for training_data_ratio in [0.2, 0.4, 0.6, 0.8, 1.0]:
        results = train_xgboost_individual_level(
            input_dir='Colorectal_Cancer_HE_patches/Training_features',
            cell_type=cell_type,
            model_name=model,
            output_dir= None,
            params=None,
            num_boost_round=800,
            seed=42,
            if_combined=True,
            training_data_ratio=training_data_ratio,
            save_files=True
        )




def train_xgboost_for_external_prediction(
    input_dir='Colorectal_Cancer_HE_patches/Training_features',
    cell_type="Cancer Cells",
    model_name='Combined',
    output_dir=None,
    params=None,
    num_boost_round=800,
    seed=42,
    if_combined=True
):
    """
    Train XGBoost model on all available data for external prediction
    No cross-validation, just train on everything and save the final model for external prediction
    
    Args:
        input_dir: Feature input directory
        cell_type: Cell type for prediction
        model_name: Model name for output directory naming
        output_dir: Output directory, if None it will be automatically generated
        params: XGBoost parameters, if None default optimized parameters will be used
        num_boost_round: Number of training rounds
        seed: Random seed
        if_combined: Whether to use combined features from multiple models
    
    Returns:
        dict: Training results with model path
    """
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set output directory
    if output_dir is None:
        output_dir = f'Colorectal_Cancer_HE_patches/xgboost_prediction/{cell_type}_{model_name}_external_prediction'
    os.makedirs(output_dir, exist_ok=True)
    
    # Set fixed optimized XGBoost parameters
    if params is None:
        # Use the same optimized parameters as in individual-level training
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',  
            'n_jobs': -1,  
            'eta': 0.01,  
            'gamma': 0,  
            'min_child_weight': 5,  
            'colsample_bytree': 0.05,  
            'subsample': 0.5,  
            'alpha': 0.1,  
            'lambda': 0.01,  
            'max_depth': 12,  
            'seed': seed
        }
        print("✅ Using optimized XGBoost parameters for external prediction")
    
    # Load feature data
    if if_combined:
        # Load important features dictionary
        # Convert cell_type spaces to underscores for file naming
        important_features_path = f"/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/xgboost_prediction/important_features_{cell_type}.pkl"
        important_features = pickle.load(open(important_features_path, "rb"))
        
        UNI2h_feature_name = f'{cell_type}_training_precomputed_features_UNI2h.pt'
        Virchow_feature_name = f'{cell_type}_training_precomputed_features_Virchow.pt'
        Virchow2_feature_name = f'{cell_type}_training_precomputed_features_Virchow2.pt'
        ProvGigapath_feature_name = f'{cell_type}_training_precomputed_features_ProvGigapath.pt'
        Conch_feature_name = f'{cell_type}_training_precomputed_features_Conch.pt'
        
        print("Loading feature data...")
        UNI2h_data = torch.load(input_dir + "/" + UNI2h_feature_name)
        Virchow_data = torch.load(input_dir + "/" + Virchow_feature_name)
        Virchow2_data = torch.load(input_dir + "/" + Virchow2_feature_name)
        ProvGigapath_data = torch.load(input_dir + "/" + ProvGigapath_feature_name)
        Conch_data = torch.load(input_dir + "/" + Conch_feature_name)

        # Select only important features for each model
        print("Selecting important features...")
        UNI2h_important_indices = [int(f.replace('f', '')) for f in important_features["UNI2h"]]
        Virchow_important_indices = [int(f.replace('f', '')) for f in important_features["Virchow"]]
        Virchow2_important_indices = [int(f.replace('f', '')) for f in important_features["Virchow2"]]
        ProvGigapath_important_indices = [int(f.replace('f', '')) for f in important_features["ProvGigapath"]]
        Conch_important_indices = [int(f.replace('f', '')) for f in important_features["Conch"]]
        
        UNI2h_selected = UNI2h_data['embeddings'][:, UNI2h_important_indices]
        Virchow_selected = Virchow_data['embeddings'][:, Virchow_important_indices]
        Virchow2_selected = Virchow2_data['embeddings'][:, Virchow2_important_indices]
        ProvGigapath_selected = ProvGigapath_data['embeddings'][:, ProvGigapath_important_indices]
        Conch_selected = Conch_data['embeddings'][:, Conch_important_indices]

        features = torch.cat([UNI2h_selected, Virchow_selected, Virchow2_selected, ProvGigapath_selected, Conch_selected], dim=1)
        data = UNI2h_data
        labels = data['celltype_proportions']
        
        print(f"✅ Using important features: UNI2h({len(UNI2h_important_indices)}), Virchow({len(Virchow_important_indices)}), Virchow2({len(Virchow2_important_indices)}), ProvGigapath({len(ProvGigapath_important_indices)}), Conch({len(Conch_important_indices)})")
        print(f"Total selected features: {features.shape[1]}")
    else:
        feature_name = f'{cell_type}_training_precomputed_features_{model_name}.pt'
        data_path = input_dir + "/" + feature_name
        print(f"Loading data: {data_path}")
        data = torch.load(data_path)
        features = data['embeddings']
        labels = data['celltype_proportions']
        print(f"Total features: {features.shape[1]}")

    print(f"Training data: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"Target: {cell_type} proportions")
    
    # Train XGBoost model on ALL data
    print(f"\n🚀 Training XGBoost model on all {features.shape[0]} samples...")
    dtrain = xgb.DMatrix(features, label=labels)
    
    # Train model
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=num_boost_round,
        evals=[(dtrain, 'train')],
        verbose_eval=100
    )
    
    # Save the trained model
    model_path = os.path.join(output_dir, f'xgboost_model_{cell_type.replace(" ", "_")}_{model_name}_external_prediction.model')
    model.save_model(model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Save feature information for external prediction
    feature_info = {
        'cell_type': cell_type,
        'model_name': model_name,
        'if_combined': if_combined,
        'total_features': features.shape[1],
        'training_samples': features.shape[0],
        'params': params,
        'num_boost_round': num_boost_round,
        'seed': seed
    }
    
    if if_combined:
        feature_info.update({
            'UNI2h_feature_count': len(UNI2h_important_indices),
            'Virchow_feature_count': len(Virchow_important_indices),
            'Virchow2_feature_count': len(Virchow2_important_indices),
            'ProvGigapath_feature_count': len(ProvGigapath_important_indices),
            'Conch_feature_count': len(Conch_important_indices),
            'important_features_file': f"important_features_{cell_type}.pkl"
        })
    
    # Save feature info
    info_path = os.path.join(output_dir, 'model_info.pt')
    torch.save(feature_info, info_path)
    print(f"✅ Model info saved to: {info_path}")
    
    # Return results
    results = {
        'model_path': model_path,
        'info_path': info_path,
        'output_dir': output_dir,
        'feature_info': feature_info,
        'training_completed': True
    }
    
    print(f"\n🎉 External prediction model training completed!")
    print(f"📁 Output directory: {output_dir}")
    print(f"🤖 Model file: {model_path}")
    print(f"📋 Info file: {info_path}")
    
    return results



if __name__ == "__main__":
    
    ##############################################################################
    ### STAGE 1: Training - Train models and save .model files and tile CSVs
    ##############################################################################
    
    CELL_TYPES = ["Stromal Cells", "Cancer Cells", "Normal Epithelial Cells", "T Cells", "Other Immune Cells"]
    BASE_OUTPUT_DIR = '/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/xgboost_prediction_with_calibration'
    INPUT_DIR = '/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/Training_features'
    
    print(f"\n{'#'*80}")
    print(f"STAGE 1: XGBoost Model Training")
    print(f"Cell Types: {CELL_TYPES}")
    print(f"Output Directory: {BASE_OUTPUT_DIR}")
    print(f"{'#'*80}\n")
    
    # Train models for all 5 cell types
    for idx, cell_type in enumerate(CELL_TYPES):
        print(f"\n{'='*80}")
        print(f"Training Cell Type {idx+1}/{len(CELL_TYPES)}: {cell_type}")
        print(f"{'='*80}\n")
        
        # Set cell-type specific output directory
        cell_type_folder = cell_type.replace(" ", "_")
        output_dir = f"{BASE_OUTPUT_DIR}/{cell_type_folder}_Combined_individual_level"
        
        # Train with individual-level cross-validation
        results = train_xgboost_individual_level(
            input_dir=INPUT_DIR,
            cell_type=cell_type,
            model_name='Combined',
            output_dir=output_dir,
            params=None,  # Use default optimized params
            num_boost_round=800,
            seed=42,
            if_combined=True,
            training_data_ratio=1.0,  # Use 100% training data
            save_files=True
        )
        
        print(f"\n✅ Completed: {cell_type}")
        print(f"   📁 Output: {output_dir}")
        print(f"   📊 Raw MAE: {results['metrics']['overall_mae']:.4f}")
        print(f"   📊 Calibrated MAE: {results['calibration']['calibrated_mae']:.4f}")
    
    print(f"\n{'#'*80}")
    print(f"✅ STAGE 1 COMPLETED: All models trained!")
    print(f"📁 Results saved in: {BASE_OUTPUT_DIR}")
    print(f"   - Model files (.model)")
    print(f"   - Tile-level prediction CSVs")
    print(f"{'#'*80}\n")
    
    
    ##############################################################################
    ### STAGE 2: Calibration - Apply calibration and save .pkl and plots
    ##############################################################################
    
    print(f"\n{'#'*80}")
    print(f"STAGE 2: Applying Quantile Mapping Calibration")
    print(f"{'#'*80}\n")
    
    BASE_DIR = '/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/xgboost_prediction_with_calibration'
    
    CALIBRATION_CELL_TYPES = [
        "Cancer Cells",
        "Stromal Cells", 
        "Normal Epithelial Cells",
        "T Cells",
        "Other Immune Cells"
    ]
    
    for cell_type in CALIBRATION_CELL_TYPES:
        print(f"\n{'='*80}")
        print(f"Calibrating: {cell_type}")
        print(f"{'='*80}")
        
        # Construct path
        cell_type_dir_name = cell_type.replace(" ", "_") + "_Combined_individual_level"
        cell_type_dir = os.path.join(BASE_DIR, cell_type_dir_name)
        csv_path = os.path.join(cell_type_dir, "tile_level_predictions_detailed.csv")
        
        if not os.path.exists(csv_path):
            print(f"⚠️  File not found: {csv_path}")
            continue
        
        # Read CSV
        print(f"\n📂 Reading: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"   Loaded {len(df):,} tiles")
        
        # Extract data
        y_true = df['deconvoluted_proportion'].values
        y_pred_raw = df['predicted_proportion_raw'].values
        
        # Apply quantile mapping calibration
        calibration_table, y_pred_calibrated, y_true_processed = calibrate_predictions(y_true, y_pred_raw, cell_type)
        
        # Calculate calibration metrics
        cal_mae = mean_absolute_error(y_true, y_pred_calibrated)
        cal_rmse = np.sqrt(mean_squared_error(y_true, y_pred_calibrated))
        cal_spearman, _ = spearmanr(y_true, y_pred_calibrated)
        cal_pearson, _ = pearsonr(y_true, y_pred_calibrated)
        
        # Save calibration parameters (lookup table for external use)
        calibration_data = {
            'lookup_table': calibration_table,
            'raw_mae': mean_absolute_error(y_true, y_pred_raw),
            'raw_rmse': np.sqrt(mean_squared_error(y_true, y_pred_raw)),
            'calibrated_mae': cal_mae,
            'calibrated_rmse': cal_rmse,
            'calibrated_spearman': cal_spearman,
            'calibrated_pearson': cal_pearson
        }
        
        calib_params_path = os.path.join(cell_type_dir, 'calibration_parameters.pkl')
        with open(calib_params_path, 'wb') as f:
            pickle.dump(calibration_data, f)
        print(f"✅ Calibration parameters saved: {calib_params_path}")
        print(f"   - Quantile mapping lookup table (for external data)")
        
        # Save calibrated predictions CSV
        df['predicted_proportion_calibrated'] = y_pred_calibrated
        output_csv = os.path.join(cell_type_dir, "tile_level_predictions_calibrated.csv")
        df.to_csv(output_csv, index=False)
        print(f"💾 Calibrated CSV saved: {output_csv}")
        
        # Generate calibration comparison plot
        output_plot = os.path.join(cell_type_dir, "calibration_comparison.png")
        plot_calibration_scatter(
            y_true=y_true,
            y_pred_raw=y_pred_raw,
            y_pred_calibrated=y_pred_calibrated,
            y_true_processed=y_true_processed,
            calibration_table=calibration_table,
            dummy=None,
            output_path=output_plot,
            model_name="Combined",
            cell_type=cell_type
        )
        
        # Generate additional scatter plot: y_true vs y_calibrated
        print(f"\n📈 Creating y_true vs y_calibrated scatter plot...")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Scatter plot
        ax.scatter(y_true, y_pred_calibrated, alpha=0.5, s=15, color='blue', edgecolor='none')
        
        # Identity line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Identity (y=x)')
        
        # Fit line
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred_calibrated)
        x_line = np.array([0, 1])
        ax.plot(x_line, intercept + slope * x_line, 'r-', linewidth=2, alpha=0.7, label='Fit line')
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred_calibrated)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_calibrated))
        spearman_corr, _ = spearmanr(y_true, y_pred_calibrated)
        pearson_corr, _ = pearsonr(y_true, y_pred_calibrated)
        
        # Add text box with metrics
        textstr = f'MAE: {mae:.4f}\n'
        textstr += f'RMSE: {rmse:.4f}\n'
        textstr += f'Spearman: {spearman_corr:.4f}\n'
        textstr += f'Pearson: {pearson_corr:.4f}\n'
        textstr += f'Slope: {slope:.4f}\n'
        textstr += f'R²: {r_value**2:.4f}\n'
        textstr += f'n: {len(y_true):,}'
        
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Labels and styling
        ax.set_xlabel('True Cell Type Proportion (y_true)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Calibrated Predicted Proportion (y_calibrated)', fontsize=14, fontweight='bold')
        ax.set_title(f'Calibrated Predictions vs Ground Truth\n{cell_type} - Quantile Mapping Calibration', 
                     fontsize=16, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='lower right')
        ax.tick_params(labelsize=12, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        plt.tight_layout()
        
        output_scatter = os.path.join(cell_type_dir, "y_true_vs_y_calibrated_scatter.png")
        plt.savefig(output_scatter, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Scatter plot saved: {output_scatter}")
        
        # Print summary
        raw_mae = mean_absolute_error(y_true, y_pred_raw)
        
        print(f"\n📊 Calibration Summary:")
        print(f"   Raw MAE:        {raw_mae:.6f}")
        print(f"   Calibrated MAE: {cal_mae:.6f}")
        print(f"   Improvement:    {((raw_mae - cal_mae) / raw_mae * 100):.2f}%")
    
    print(f"\n{'#'*80}")
    print(f"✅ STAGE 2 COMPLETED: All calibrations done!")
    print(f"📁 Outputs in each cell type folder:")
    print(f"   - calibration_parameters.pkl (lookup table)")
    print(f"   - calibration_comparison.png (visualization)")
    print(f"   - tile_level_predictions_calibrated.csv")
    print(f"{'#'*80}\n")
    
    
    ##############################################################################
    ### Inspect Calibration Parameters (Cancer Cells Example)
    ##############################################################################
    
    print(f"\n{'#'*80}")
    print(f"📦 Inspecting Calibration Parameters: Cancer Cells")
    print(f"{'#'*80}\n")
    
    pkl_path = '/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/xgboost_prediction_with_calibration/Cancer_Cells_Combined_individual_level/calibration_parameters.pkl'
    
    with open(pkl_path, 'rb') as f:
        calib_params = pickle.load(f)
    
    print(f"📋 Keys in calibration_parameters.pkl:")
    for key in calib_params.keys():
        print(f"   - {key}")
    
    print(f"\n{'='*80}")
    print(f"📊 Lookup Table Details:")
    print(f"{'='*80}")
    
    lookup = calib_params['lookup_table']
    print(f"\nLookup table keys: {list(lookup.keys())}")
    
    for key, value in lookup.items():
        if isinstance(value, np.ndarray):
            print(f"\n🔹 {key}:")
            print(f"   Shape: {value.shape}")
            print(f"   Dtype: {value.dtype}")
            print(f"   Range: [{value.min():.6f}, {value.max():.6f}]")
            print(f"   First 10 values: {value[:10]}")
            print(f"   Last 10 values: {value[-10:]}")
        else:
            print(f"\n🔹 {key}: {value}")
    
    print(f"\n{'='*80}")
    print(f"📈 Performance Metrics:")
    print(f"{'='*80}")
    
    print(f"\n🔴 Raw Predictions:")
    print(f"   MAE:  {calib_params.get('raw_mae', 'N/A'):.6f}")
    print(f"   RMSE: {calib_params.get('raw_rmse', 'N/A'):.6f}")
    
    print(f"\n🟢 Calibrated Predictions:")
    print(f"   MAE:      {calib_params.get('calibrated_mae', 'N/A'):.6f}")
    print(f"   RMSE:     {calib_params.get('calibrated_rmse', 'N/A'):.6f}")
    print(f"   Spearman: {calib_params.get('calibrated_spearman', 'N/A'):.6f}")
    print(f"   Pearson:  {calib_params.get('calibrated_pearson', 'N/A'):.6f}")
    
    raw_mae = calib_params.get('raw_mae', 0)
    cal_mae = calib_params.get('calibrated_mae', 0)
    if raw_mae > 0:
        improvement = ((raw_mae - cal_mae) / raw_mae * 100)
        print(f"\n✨ MAE Improvement: {improvement:.2f}%")
    
    print(f"\n{'#'*80}")
    print(f"✅ Inspection Complete!")
    print(f"{'#'*80}\n")
