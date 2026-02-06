"""
XGBoost for B Cells and Myeloid Cells
=====================================

This script trains XGBoost models for B_Cells and Myeloid_Cells prediction.
For combined model, it reads important features from the Excel file.

This script also includes functionality to split Other Immune Cells features
into B Cells and Myeloid Cells features.

Author: Saishi Cui
Date: January 2026
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import xgboost as xgb
from datetime import datetime
from scipy import stats
import pickle

# Path to important features Excel
IMPORTANT_FEATURES_EXCEL = (
    '/Users/scui2/ST/Colorectal_Cancer_HE_patches/xgboost_prediction/'
    'Important_Features_COAD.xlsx'
)


def split_other_immune_features_to_b_mye(
    features_path='/Users/scui2/ST/Colorectal_Cancer_HE_patches/Training_features',
    csv_path='/Users/scui2/ST/Colorectal_Cancer_HE_patches/Training_celltype_proportion',
    foundation_models=None
):
    """
    Split Other Immune Cells features into B Cells and Myeloid Cells.
    
    This function reads the precomputed features for Other Immune Cells and splits them
    into separate feature files for B Cells and Myeloid Cells by replacing the 
    celltype_proportions with the corresponding values from the split CSV files.
    
    Args:
        features_path: Path to the directory containing feature .pt files
        csv_path: Path to the directory containing celltype proportion CSV files
        foundation_models: List of foundation model names to process
    
    Returns:
        None. Creates new .pt files for B_Cells and Myeloid_Cells.
    """
    if foundation_models is None:
        foundation_models = ["Virchow", "Virchow2", "Conch", "ProvGigapath", "UNI2h", "ResNet50"]
    
    print("\n" + "="*60)
    print("SPLITTING OTHER IMMUNE CELLS FEATURES")
    print("="*60 + "\n")
    
    # Load all B and MYE CSV files once
    print("Loading B Cells CSV files...")
    b_cody = pd.read_csv(os.path.join(csv_path, "B_Cells_celltype_proportion_Cody.csv"), index_col=0)
    b_fh = pd.read_csv(os.path.join(csv_path, "B_Cells_celltype_proportion_FH.csv"), index_col=0)
    b_hest = pd.read_csv(os.path.join(csv_path, "B_Cells_celltype_proportion_HEST.csv"), index_col=0)
    
    print("Loading Myeloid Cells CSV files...")
    mye_cody = pd.read_csv(os.path.join(csv_path, "Myeloid_Cells_celltype_proportion_Cody.csv"), index_col=0)
    mye_fh = pd.read_csv(os.path.join(csv_path, "Myeloid_Cells_celltype_proportion_FH.csv"), index_col=0)
    mye_hest = pd.read_csv(os.path.join(csv_path, "Myeloid_Cells_celltype_proportion_HEST.csv"), index_col=0)
    
    # Combine all CSVs into single lookup dictionaries
    print("\nBuilding lookup dictionaries...")
    
    # Build B cells lookup
    b_lookup = {}
    for idx in b_cody.index:
        b_lookup[idx] = b_cody.loc[idx, 'B Cells']
    for idx in b_fh.index:
        b_lookup[idx] = b_fh.loc[idx, 'B Cells']
    for idx in b_hest.index:
        b_lookup[idx] = b_hest.loc[idx, 'B Cells']
    
    # Build MYE cells lookup
    mye_lookup = {}
    for idx in mye_cody.index:
        mye_lookup[idx] = mye_cody.loc[idx, 'Myeloid Cells']
    for idx in mye_fh.index:
        mye_lookup[idx] = mye_fh.loc[idx, 'Myeloid Cells']
    for idx in mye_hest.index:
        mye_lookup[idx] = mye_hest.loc[idx, 'Myeloid Cells']
    
    print(f"B cells lookup size: {len(b_lookup)}")
    print(f"MYE cells lookup size: {len(mye_lookup)}")
    
    # Process each foundation model
    for model_name in foundation_models:
        print(f"\n{'='*60}")
        print(f"Processing {model_name}...")
        print(f"{'='*60}")
        
        input_pt_file = os.path.join(features_path, f"Other Immune Cells_training_precomputed_features_{model_name}.pt")
        
        if not os.path.exists(input_pt_file):
            print(f"Warning: File not found: {input_pt_file}")
            continue
        
        # Load the original .pt file
        print(f"Loading {model_name} feature file...")
        data = torch.load(input_pt_file, weights_only=False)
        
        print(f"Original embeddings shape: {data['embeddings'].shape}")
        print(f"Original number of tiles: {len(data['tile_ids'])}")
        
        # Filter tiles - only keep tiles that exist in the lookup
        print("\nFiltering tiles and looking up cell type proportions...")
        valid_indices = []
        b_proportions = []
        mye_proportions = []
        valid_tile_ids = []
        valid_sample_ids = []
        valid_individual_ids = []
        
        for i, tile_id in enumerate(tqdm(data['tile_ids'])):
            if tile_id in b_lookup:
                valid_indices.append(i)
                b_proportions.append(b_lookup[tile_id])
                mye_proportions.append(mye_lookup[tile_id])
                valid_tile_ids.append(tile_id)
                valid_sample_ids.append(data['sample_ids'][i])
                valid_individual_ids.append(data['individual_ids'][i])
        
        removed_count = len(data['tile_ids']) - len(valid_indices)
        print(f"Removed {removed_count} tiles not found in CSV lookup")
        print(f"Remaining tiles: {len(valid_indices)}")
        
        # Filter embeddings
        valid_indices_tensor = torch.tensor(valid_indices)
        filtered_embeddings = data['embeddings'][valid_indices_tensor]
        
        # Convert proportions to tensors
        b_proportions_tensor = torch.tensor(b_proportions, dtype=torch.float32)
        mye_proportions_tensor = torch.tensor(mye_proportions, dtype=torch.float32)
        
        # Create new data dictionaries
        b_data = {
            'embeddings': filtered_embeddings,
            'celltype_proportions': b_proportions_tensor,
            'tile_ids': valid_tile_ids,
            'sample_ids': valid_sample_ids,
            'individual_ids': valid_individual_ids
        }
        
        mye_data = {
            'embeddings': filtered_embeddings,
            'celltype_proportions': mye_proportions_tensor,
            'tile_ids': valid_tile_ids,
            'sample_ids': valid_sample_ids,
            'individual_ids': valid_individual_ids
        }
        
        # Save new .pt files
        b_output_file = os.path.join(features_path, f"B_Cells_training_precomputed_features_{model_name}.pt")
        mye_output_file = os.path.join(features_path, f"Myeloid_Cells_training_precomputed_features_{model_name}.pt")
        
        print(f"\nSaving B Cells features to: {b_output_file}")
        torch.save(b_data, b_output_file)
        
        print(f"Saving Myeloid Cells features to: {mye_output_file}")
        torch.save(mye_data, mye_output_file)
        
        # Print statistics
        print(f"\nB Cells proportions statistics:")
        print(f"  Mean: {b_proportions_tensor.mean():.6f}")
        print(f"  Max: {b_proportions_tensor.max():.6f}")
        print(f"  Min: {b_proportions_tensor.min():.6f}")
        
        print(f"\nMyeloid Cells proportions statistics:")
        print(f"  Mean: {mye_proportions_tensor.mean():.6f}")
        print(f"  Max: {mye_proportions_tensor.max():.6f}")
        print(f"  Min: {mye_proportions_tensor.min():.6f}")
    
    print(f"\n{'='*60}")
    print("Feature splitting complete!")
    print(f"{'='*60}\n")


def load_important_features_from_excel(cell_type):
    """
    Load important features from Excel file for a given cell type.
    
    Args:
        cell_type: 'B_Cells' or 'Myeloid_Cells'
    
    Returns:
        dict: {model_name: list of feature indices}
    """
    print(f"Loading important features from Excel: {cell_type}")
    
    # Read the sheet for this cell type
    df = pd.read_excel(IMPORTANT_FEATURES_EXCEL, sheet_name=cell_type)
    
    # Extract features for each model
    models = ['UNI2h', 'Virchow', 'Virchow2', 'ProvGigapath', 'Conch']
    important_features = {}
    
    for model in models:
        col_name = f'{model}_Features'
        if col_name in df.columns:
            # Get non-empty feature names
            features = df[col_name].dropna().tolist()
            # Convert feature names (f0, f1, ...) to indices
            indices = [int(f.replace('f', '')) for f in features]
            important_features[model] = indices
            print(f"  {model}: {len(indices)} features")
        else:
            important_features[model] = []
            print(f"  {model}: column not found!")
    
    return important_features


def train_xgboost_individual_level(
    input_dir='/Users/scui2/ST/Colorectal_Cancer_HE_patches/Training_features',
    cell_type="B_Cells",
    model_name='UNI2h',
    output_dir=None,
    params=None,
    num_boost_round=800,
    seed=42,
    if_combined=False,
    save_files=True
):
    """Train XGBoost with leave-one-individual-out CV."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if save_files:
        if output_dir is None:
            base = '/Users/scui2/ST/Colorectal_Cancer_HE_patches'
            sub = f'{cell_type}_{model_name}_individual_level'
            output_dir = f'{base}/xgboost_prediction/{sub}'
        os.makedirs(output_dir, exist_ok=True)
    
    if params is None:
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
        print("Using XGBoost with CPU acceleration")
    
    if if_combined:
        # Load important features from Excel
        important_features = load_important_features_from_excel(cell_type)
        
        ct = cell_type
        f1 = f'{ct}_training_precomputed_features_UNI2h.pt'
        f2 = f'{ct}_training_precomputed_features_Virchow.pt'
        f3 = f'{ct}_training_precomputed_features_Virchow2.pt'
        f4 = f'{ct}_training_precomputed_features_ProvGigapath.pt'
        f5 = f'{ct}_training_precomputed_features_Conch.pt'
        
        print("\nLoading features from foundation models...")
        d1 = torch.load(input_dir + "/" + f1, weights_only=False)
        d2 = torch.load(input_dir + "/" + f2, weights_only=False)
        d3 = torch.load(input_dir + "/" + f3, weights_only=False)
        d4 = torch.load(input_dir + "/" + f4, weights_only=False)
        d5 = torch.load(input_dir + "/" + f5, weights_only=False)
        
        # Select only important features for each model
        print("\nSelecting important features...")
        uni2h_idx = important_features['UNI2h']
        virchow_idx = important_features['Virchow']
        virchow2_idx = important_features['Virchow2']
        provgiga_idx = important_features['ProvGigapath']
        conch_idx = important_features['Conch']
        
        uni2h_sel = d1['embeddings'][:, uni2h_idx]
        virchow_sel = d2['embeddings'][:, virchow_idx]
        virchow2_sel = d3['embeddings'][:, virchow2_idx]
        provgiga_sel = d4['embeddings'][:, provgiga_idx]
        conch_sel = d5['embeddings'][:, conch_idx]
        
        # Concatenate selected features
        features = torch.cat([
            uni2h_sel, virchow_sel, virchow2_sel, provgiga_sel, conch_sel
        ], dim=1)
        
        data = d1
        labels = data['celltype_proportions']
        
        print(f"\nSelected features per model:")
        print(f"  UNI2h: {len(uni2h_idx)}")
        print(f"  Virchow: {len(virchow_idx)}")
        print(f"  Virchow2: {len(virchow2_idx)}")
        print(f"  ProvGigapath: {len(provgiga_idx)}")
        print(f"  Conch: {len(conch_idx)}")
        print(f"  Total combined: {features.shape[1]}")
    else:
        fn = f'{cell_type}_training_precomputed_features_{model_name}.pt'
        data_path = input_dir + "/" + fn
        print(f"Loading: {data_path}")
        data = torch.load(data_path, weights_only=False)
        features = data['embeddings']
        labels = data['celltype_proportions']
    
    tile_ids = data.get('tile_ids', [])
    individual_ids = data.get('individual_ids', [])
    sample_ids = data.get('sample_ids', [])
    
    unique_individuals = sorted(list(set(individual_ids)))
    n_ind = len(unique_individuals)
    n_tiles = len(tile_ids)
    print(f"\nDataset: {n_ind} individuals, {n_tiles} tiles")
    
    all_predictions = np.zeros(len(labels))
    all_importances = []
    is_test = np.zeros(len(labels), dtype=bool)
    fold_metrics = []
    
    for ind_idx, ind in enumerate(unique_individuals):
        print(f"\n{'-'*60}")
        msg = f"Leave-one-out: {ind} ({ind_idx+1}/{n_ind})"
        print(msg)
        print(f"{'-'*60}")
        
        train_mask = np.array([iid != ind for iid in individual_ids])
        train_indices = np.where(train_mask)[0]
        
        X_train = features[train_indices]
        y_train = labels[train_indices]
        
        print(f"Training samples: {len(X_train)}")
        
        if torch.is_tensor(X_train):
            X_train_np = X_train.numpy()
        else:
            X_train_np = X_train
        if torch.is_tensor(y_train):
            y_train_np = y_train.numpy()
        else:
            y_train_np = y_train
        
        dtrain = xgb.DMatrix(X_train_np, label=y_train_np)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, 'train')],
            verbose_eval=100
        )
        
        importance = model.get_score(importance_type='gain')
        all_importances.append(importance)
        
        if save_files:
            model_dir = os.path.join(output_dir, 'models')
            os.makedirs(model_dir, exist_ok=True)
            mpath = os.path.join(model_dir, f'model_leave_{ind}_out.model')
            model.save_model(mpath)
        
        test_mask = np.array([iid == ind for iid in individual_ids])
        test_indices = np.where(test_mask)[0]
        
        is_test[test_indices] = True
        
        X_test = features[test_indices]
        y_test = labels[test_indices]
        
        print(f"Testing: {len(X_test)} samples")
        
        if torch.is_tensor(X_test):
            X_test_np = X_test.numpy()
        else:
            X_test_np = X_test
        
        dtest = xgb.DMatrix(X_test_np)
        predictions = model.predict(dtest)
        
        all_predictions[test_indices] = predictions
        
        if torch.is_tensor(y_test):
            y_test_np = y_test.numpy()
        else:
            y_test_np = y_test
        
        ind_mae = mean_absolute_error(y_test_np, predictions)
        ind_rmse = np.sqrt(mean_squared_error(y_test_np, predictions))
        ind_spearman, _ = spearmanr(y_test_np, predictions)
        ind_pearson, _ = pearsonr(y_test_np, predictions)
        
        ind_min = round(float(y_test_np.min()), 3)
        ind_max = round(float(y_test_np.max()), 3)
        
        fold_metrics.append({
            'Fold': ind,
            'Test_Element': ind,
            'Individual': ind,
            'MAE': ind_mae,
            'RMSE': ind_rmse,
            'Spearman': ind_spearman,
            'Pearson': ind_pearson,
            'n_test_samples': len(test_indices),
            'Min_celltype_proportion': ind_min,
            'Max_celltype_proportion': ind_max
        })
        
        msg = f"  MAE={ind_mae:.4f}, RMSE={ind_rmse:.4f}"
        msg += f", Spearman={ind_spearman:.4f}, Pearson={ind_pearson:.4f}"
        print(msg)
    
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
    
    print("\n" + "="*60)
    print("Average individual metrics:")
    avg_mae = average_metrics['MAE']
    avg_mae_std = average_metrics['MAE_std']
    print(f"  MAE={avg_mae:.4f} +/- {avg_mae_std:.4f}")
    avg_rmse = average_metrics['RMSE']
    avg_rmse_std = average_metrics['RMSE_std']
    print(f"  RMSE={avg_rmse:.4f} +/- {avg_rmse_std:.4f}")
    avg_sp = average_metrics['Spearman']
    avg_sp_std = average_metrics['Spearman_std']
    print(f"  Spearman={avg_sp:.4f} +/- {avg_sp_std:.4f}")
    avg_pe = average_metrics['Pearson']
    avg_pe_std = average_metrics['Pearson_std']
    print(f"  Pearson={avg_pe:.4f} +/- {avg_pe_std:.4f}")
    
    if torch.is_tensor(labels):
        labels_np = labels.numpy()
    else:
        labels_np = labels
    
    overall_mae = mean_absolute_error(labels_np, all_predictions)
    overall_rmse = np.sqrt(mean_squared_error(labels_np, all_predictions))
    overall_spearman, _ = spearmanr(labels_np, all_predictions)
    overall_pearson, _ = pearsonr(labels_np, all_predictions)
    
    print("\nOverall metrics:")
    msg = f"  MAE={overall_mae:.4f}, RMSE={overall_rmse:.4f}"
    msg += f", Spearman={overall_spearman:.4f}, Pearson={overall_pearson:.4f}"
    print(msg)
    
    if save_files:
        csv_path = os.path.join(output_dir, 'individual_metrics.csv')
        fold_metrics_df.to_csv(csv_path, index=False)
        print(f"Metrics saved to: {csv_path}")
        
        plt.figure(figsize=(8, 8))
        plt.scatter(labels_np, all_predictions, alpha=0.7, s=15)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
        
        slope, intercept, _, _, _ = stats.linregress(labels_np, all_predictions)
        plt.plot([0, 1], [intercept, intercept + slope], 'r-', alpha=0.7)
        
        txt = f'MAE: {overall_mae:.4f}\n'
        txt += f'RMSE: {overall_rmse:.4f}\n'
        txt += f'Spearman: {overall_spearman:.4f}\n'
        txt += f'Pearson: {overall_pearson:.4f}\n'
        txt += f'Slope: {slope:.4f}\n'
        txt += f'n: {len(labels_np)}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax = plt.gca()
        plt.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)
        
        title = f'Performance ({cell_type} - {model_name})'
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('True Proportion', fontsize=12)
        plt.ylabel('Predicted Proportion', fontsize=12)
        max_val = max(labels_np.max(), all_predictions.max()) * 1.1
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        fig_path = os.path.join(output_dir, 'performance.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        print("\nSaving tile predictions...")
        tile_df = pd.DataFrame({
            'tile_id': tile_ids,
            'sample_id': sample_ids,
            'individual_id': individual_ids,
            'deconvoluted_proportion': labels_np,
            'predicted_proportion': all_predictions
        })
        
        tile_csv = os.path.join(output_dir, 'tile_predictions.csv')
        tile_df.to_csv(tile_csv, index=False)
        print(f"Tiles saved: {tile_csv}")
        print(f"  Total: {len(tile_df):,}")
    
    results = {
        'predictions': all_predictions,
        'labels': labels_np,
        'tile_ids': tile_ids,
        'individual_ids': individual_ids,
        'sample_ids': sample_ids,
        'is_test': is_test,
        'feature_importances': all_importances,
        'params': params,
        'model_name': model_name,
        'cell_type': cell_type,
        'level': 'individual',
        'timestamp': timestamp,
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
    
    if save_files and output_dir:
        res_path = os.path.join(output_dir, 'results.pt')
        torch.save(results, res_path)
        print(f"Results saved: {res_path}")
    
    return results


if __name__ == "__main__":
    
    # =================================================================
    # OPTIONAL: Split Other Immune Cells features first
    # =================================================================
    # Uncomment the following lines to run feature splitting before training
    # This will create B_Cells and Myeloid_Cells feature files from 
    # Other Immune Cells features
    #
    # split_other_immune_features_to_b_mye(
    #     features_path='/Users/scui2/ST/Colorectal_Cancer_HE_patches/Training_features',
    #     csv_path='/Users/scui2/ST/Colorectal_Cancer_HE_patches/Training_celltype_proportion',
    #     foundation_models=["Virchow", "Virchow2", "Conch", "ProvGigapath", "UNI2h", "ResNet50"]
    # )
    
    # =================================================================
    # ONLY TRAIN COMBINED MODELS FOR B_Cells and Myeloid_Cells
    # =================================================================
    
    CELL_TYPES = ["B_Cells", "Myeloid_Cells"]
    INPUT_DIR = '/Users/scui2/ST/Colorectal_Cancer_HE_patches/Training_features'
    base = '/Users/scui2/ST/Colorectal_Cancer_HE_patches'
    BASE_OUTPUT = base + '/xgboost_prediction'
    
    print(f"\n{'#'*60}")
    print(f"XGBoost Combined Model Training")
    print(f"Cell Types: {CELL_TYPES}")
    print(f"Using important features from Excel")
    print(f"{'#'*60}\n")
    
    all_results = []
    
    for cell_type in CELL_TYPES:
        print(f"\n{'='*60}")
        print(f"Training {cell_type} with Combined (Important Features)")
        print(f"{'='*60}\n")
        
        out_dir = f"{BASE_OUTPUT}/{cell_type}_Combined_individual_level"
        
        results = train_xgboost_individual_level(
            input_dir=INPUT_DIR,
            cell_type=cell_type,
            model_name='Combined',
            output_dir=out_dir,
            params=None,
            num_boost_round=800,
            seed=42,
            if_combined=True,
            save_files=True
        )
        
        all_results.append({
            'cell_type': cell_type,
            'model_name': 'Combined',
            'MAE': results['metrics']['overall_mae'],
            'RMSE': results['metrics']['overall_rmse'],
            'Spearman': results['metrics']['overall_spearman'],
            'Pearson': results['metrics']['overall_pearson']
        })
        
        mae = results['metrics']['overall_mae']
        pearson = results['metrics']['overall_pearson']
        print(f"\nDone: {cell_type} - Combined")
        print(f"   MAE: {mae:.4f}, Pearson: {pearson:.4f}")
    
    print(f"\n{'#'*60}")
    print(f"SUMMARY")
    print(f"{'#'*60}\n")
    
    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))
    
    print(f"\n{'#'*60}")
    print(f"ALL DONE!")
    print(f"{'#'*60}\n")
    
    # =================================================================
    # FULL TRAINING (Individual + Combined) - Uncomment to run all
    # =================================================================
    # INDIVIDUAL_MODELS = ['UNI2h', 'Virchow', 'Virchow2', 
    #                      'ProvGigapath', 'Conch', 'ResNet50']
    # 
    # for cell_type in CELL_TYPES:
    #     for model_name in INDIVIDUAL_MODELS:
    #         out_dir = f"{BASE_OUTPUT}/{cell_type}_{model_name}_individual_level"
    #         results = train_xgboost_individual_level(
    #             input_dir=INPUT_DIR,
    #             cell_type=cell_type,
    #             model_name=model_name,
    #             output_dir=out_dir,
    #             params=None,
    #             num_boost_round=800,
    #             seed=42,
    #             if_combined=False,
    #             save_files=True
    #         )
