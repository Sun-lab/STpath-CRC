"""
STPath-COAD: XGBoost Model Training and Prediction for Colorectal Cancer
======================================================================

This script trains XGBoost models for predicting cell type proportions in colorectal cancer
H&E patches using features extracted from multiple foundation models.

Author: Saishi Cui
Date: Sept 2025

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
from matplotlib.patches import Patch
from umap import UMAP
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from joblib import Parallel, delayed
import pickle





### Train XGBoost model with leave-some-out cross-validation
### 2 levels: tile, individual


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
        important_features = pickle.load(open(f"Colorectal_Cancer_HE_patches/xgboost_prediction/important_features_{cell_type}.pkl", "rb"))
        
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
    
    print("\nOverall metrics:")
    print(f"  MAE={overall_mae:.4f}, RMSE={overall_rmse:.4f}, Spearman={overall_spearman:.4f}, Pearson={overall_pearson:.4f}")
    
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
    
    # Prepare results dictionary
    results = {
        'predictions': all_predictions,
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
        important_features = pickle.load(open(f"Colorectal_Cancer_HE_patches/xgboost_prediction/important_features_{cell_type}.pkl", "rb"))
        
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
    for cell_type in ["Cancer Cells", "Stromal Cells", "Normal Epithelial Cells", "T Cells", "Other Immune Cells"]:
        results = train_xgboost_for_external_prediction(
            input_dir='Colorectal_Cancer_HE_patches/Training_features',
            cell_type=cell_type,
            model_name='Combined',
            output_dir=None,
            params=None,
            num_boost_round=800,
            seed=42,
            if_combined=True
        )