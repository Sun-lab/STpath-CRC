"""
STPath-COAD: Other ML Models Training and Prediction for Colorectal Cancer
===========================================================================

This script trains Lasso, MLP, and Random Forest models for predicting 
cell type proportions in colorectal cancer H&E patches using features 
extracted from foundation models.

Author: Saishi Cui
Date: January 2026

Purpose: 
1. Grid search for best hyperparameters using Cancer Cells with 50% data
2. Train and evaluate models using individual-level CV
3. Output individual-level metrics CSV (same format as XGBoost)
4. Only train for Cancer Cells and Stromal Cells with Virchow2
5. Support for Lasso, MLP, and Random Forest models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, ParameterGrid
from datetime import datetime
from scipy import stats
import pickle
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
INPUT_DIR = '/Users/scui2/ST/Colorectal_Cancer_HE_patches/Training_features'
BASE_OUTPUT_DIR = '/Users/scui2/ST/Colorectal_Cancer_HE_patches/model_comparison'

# Only train for these combinations
CELL_TYPES = ["Cancer Cells", "Stromal Cells"]
FOUNDATION_MODELS = ['Virchow2']  # Only Virchow2
RANDOM_STATE = 42

# Which models to train (set to True to train that model)
TRAIN_LASSO = True
TRAIN_MLP = True
TRAIN_RANDOM_FOREST = True

# Set device for MLP
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Apple Silicon)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# =============================================================================
# MLP Model Definition
# =============================================================================
class MLP(nn.Module):
    """Multi-Layer Perceptron for regression."""
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.3):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Constrain output to [0, 1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


# =============================================================================
# Data Loading Functions
# =============================================================================
def load_features(input_dir, cell_type, model_name):
    """Load features from a single foundation model."""
    feature_name = f'{cell_type}_training_precomputed_features_{model_name}.pt'
    data_path = os.path.join(input_dir, feature_name)
    print(f"  Loading {model_name}: {data_path}")
    data = torch.load(data_path, weights_only=False)
    
    features = data['embeddings']
    labels = data['celltype_proportions']
    
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    return {
        'embeddings': features,
        'celltype_proportions': labels,
        'tile_ids': data.get('tile_ids', []),
        'individual_ids': data.get('individual_ids', []),
        'sample_ids': data.get('sample_ids', [])
    }


# =============================================================================
# LASSO Functions
# =============================================================================
def grid_search_lasso(input_dir, cell_type="Cancer Cells", 
                      model_name="Virchow2", subsample_ratio=0.5):
    """Grid search for Lasso hyperparameters."""
    print("\n" + "="*80)
    print("LASSO HYPERPARAMETER GRID SEARCH")
    print(f"Cell Type: {cell_type}, Model: {model_name}")
    print(f"Subsample Ratio: {subsample_ratio*100:.0f}%")
    print("="*80 + "\n")
    
    data = load_features(input_dir, cell_type, model_name)
    features = data['embeddings'].numpy()
    labels = data['celltype_proportions'].numpy()
    
    np.random.seed(RANDOM_STATE)
    n_samples = int(len(features) * subsample_ratio)
    indices = np.random.choice(len(features), size=n_samples, replace=False)
    X = features[indices]
    y = labels[indices]
    
    print(f"Using {n_samples} samples for grid search")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    alphas = np.logspace(-5, 1, 50)
    print(f"Searching over {len(alphas)} alpha values...")
    
    lasso_cv = LassoCV(
        alphas=alphas, cv=5, max_iter=10000, tol=1e-4,
        n_jobs=-1, random_state=RANDOM_STATE, verbose=True
    )
    
    lasso_cv.fit(X_scaled, y)
    
    best_params = {
        'alpha': lasso_cv.alpha_,
        'max_iter': 10000,
        'tol': 1e-4,
        'selection': 'cyclic'
    }
    
    print(f"\nBest Alpha: {lasso_cv.alpha_:.6f}")
    print(f"Non-zero coefficients: {np.sum(lasso_cv.coef_ != 0)} / {len(lasso_cv.coef_)}")
    
    return best_params


def train_lasso_individual_level(input_dir, cell_type, model_name, output_dir, params):
    """Train Lasso model using individual-level CV."""
    print("\n" + "="*80)
    print(f"TRAINING LASSO - {cell_type} - {model_name}")
    print("="*80 + "\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    data = load_features(input_dir, cell_type, model_name)
    features = data['embeddings'].numpy()
    labels = data['celltype_proportions'].numpy()
    tile_ids = data.get('tile_ids', [])
    individual_ids = data.get('individual_ids', [])
    sample_ids = data.get('sample_ids', [])
    
    unique_individuals = sorted(list(set(individual_ids)))
    print(f"Dataset: {len(features)} samples, {len(unique_individuals)} individuals")
    
    all_predictions = np.zeros_like(labels)
    fold_metrics = []
    
    for ind_idx, ind in enumerate(unique_individuals):
        print(f"\nFold {ind_idx+1}/{len(unique_individuals)}: Leaving {ind} out")
        
        train_mask = np.array([iid != ind for iid in individual_ids])
        test_mask = np.array([iid == ind for iid in individual_ids])
        
        X_train, y_train = features[train_mask], labels[train_mask]
        X_test, y_test = features[test_mask], labels[test_mask]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        lasso = Lasso(
            alpha=params.get('alpha', 0.001),
            max_iter=params.get('max_iter', 10000),
            tol=params.get('tol', 1e-4),
            selection=params.get('selection', 'cyclic'),
            random_state=RANDOM_STATE
        )
        
        lasso.fit(X_train_scaled, y_train)
        predictions = lasso.predict(X_test_scaled)
        predictions = np.clip(predictions, 0, 1)
        all_predictions[test_mask] = predictions
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        spearman, _ = spearmanr(y_test, predictions)
        pearson, _ = pearsonr(y_test, predictions)
        
        fold_metrics.append({
            'Fold': ind, 'Test_Element': ind, 'Individual': ind,
            'MAE': mae, 'RMSE': rmse,
            'Spearman': spearman, 'Pearson': pearson,
            'n_test_samples': len(X_test),
            'Min_celltype_proportion': round(float(y_test.min()), 3),
            'Max_celltype_proportion': round(float(y_test.max()), 3)
        })
        
        print(f"  MAE: {mae:.4f}, Pearson: {pearson:.4f}")
    
    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(os.path.join(output_dir, 'individual_metrics_individual_level.csv'), index=False)
    
    overall_mae = mean_absolute_error(labels, all_predictions)
    overall_pearson, _ = pearsonr(labels, all_predictions)
    
    print(f"\nOverall MAE: {overall_mae:.4f}, Pearson: {overall_pearson:.4f}")
    
    # Save scatter plot
    save_scatter_plot(labels, all_predictions, cell_type, model_name, 'Lasso', output_dir)
    
    results = {
        'predictions': all_predictions, 'labels': labels,
        'tile_ids': tile_ids, 'individual_ids': individual_ids,
        'sample_ids': sample_ids, 'params': params,
        'cell_type': cell_type, 'model_name': model_name, 'model': 'Lasso',
        'overall_metrics': {'MAE': overall_mae, 'Pearson': overall_pearson},
        'fold_metrics': fold_metrics
    }
    
    torch.save(results, os.path.join(output_dir, 'lasso_results_individual_level.pt'))
    return results


# =============================================================================
# MLP Functions
# =============================================================================
def train_mlp_model(model, train_loader, val_loader, n_epochs=100, lr=0.001, patience=10):
    """Train MLP model with early stopping."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(X_batch)
        
        train_loss /= len(train_loader.dataset)
        
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * len(X_batch)
            
            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model


def grid_search_mlp(input_dir, cell_type="Cancer Cells", 
                    model_name="Virchow2", subsample_ratio=0.5):
    """Grid search for MLP hyperparameters."""
    print("\n" + "="*80)
    print("MLP HYPERPARAMETER GRID SEARCH")
    print(f"Cell Type: {cell_type}, Model: {model_name}, Device: {DEVICE}")
    print("="*80 + "\n")
    
    data = load_features(input_dir, cell_type, model_name)
    features = data['embeddings'].numpy()
    labels = data['celltype_proportions'].numpy()
    individual_ids = data.get('individual_ids', [])
    
    np.random.seed(RANDOM_STATE)
    n_samples = int(len(features) * subsample_ratio)
    indices = np.random.choice(len(features), size=n_samples, replace=False)
    X = features[indices]
    y = labels[indices]
    ind_ids = [individual_ids[i] for i in indices]
    
    input_dim = X.shape[1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    param_grid = {
        'hidden_dims': [[512, 256], [512, 256, 128], [1024, 512, 256]],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'dropout_rate': [0.2, 0.3, 0.4],
        'batch_size': [64, 128, 256]
    }
    
    # Split for grid search
    unique_inds = sorted(list(set(ind_ids)))
    n_val_inds = max(1, len(unique_inds) // 5)
    np.random.shuffle(unique_inds)
    val_inds = set(unique_inds[:n_val_inds])
    
    train_mask = np.array([iid not in val_inds for iid in ind_ids])
    val_mask = ~train_mask
    
    X_train, y_train = X_scaled[train_mask], y[train_mask]
    X_val, y_val = X_scaled[val_mask], y[val_mask]
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    best_score = float('inf')
    best_params = None
    
    param_list = list(ParameterGrid(param_grid))
    print(f"Searching {len(param_list)} parameter combinations...")
    
    for idx, params in enumerate(param_list):
        if (idx + 1) % 10 == 0:
            print(f"[{idx+1}/{len(param_list)}]")
        
        torch.manual_seed(RANDOM_STATE)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        
        model = MLP(
            input_dim=input_dim,
            hidden_dims=params['hidden_dims'],
            dropout_rate=params['dropout_rate']
        ).to(DEVICE)
        
        model = train_mlp_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            n_epochs=100, lr=params['learning_rate'], patience=10
        )
        
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t.to(DEVICE)).cpu().numpy()
        
        val_mse = mean_squared_error(y_val, val_preds)
        
        if val_mse < best_score:
            best_score = val_mse
            best_params = params.copy()
    
    print(f"\nBest Parameters: {best_params}")
    print(f"Best Val RMSE: {np.sqrt(best_score):.6f}")
    
    return best_params


def train_mlp_individual_level(input_dir, cell_type, model_name, output_dir, params):
    """Train MLP model using individual-level CV."""
    print("\n" + "="*80)
    print(f"TRAINING MLP - {cell_type} - {model_name}")
    print("="*80 + "\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    data = load_features(input_dir, cell_type, model_name)
    features = data['embeddings'].numpy()
    labels = data['celltype_proportions'].numpy()
    tile_ids = data.get('tile_ids', [])
    individual_ids = data.get('individual_ids', [])
    sample_ids = data.get('sample_ids', [])
    
    input_dim = features.shape[1]
    unique_individuals = sorted(list(set(individual_ids)))
    print(f"Dataset: {len(features)} samples, {len(unique_individuals)} individuals")
    
    all_predictions = np.zeros_like(labels)
    fold_metrics = []
    
    for ind_idx, ind in enumerate(unique_individuals):
        print(f"\nFold {ind_idx+1}/{len(unique_individuals)}: Leaving {ind} out")
        
        train_mask = np.array([iid != ind for iid in individual_ids])
        test_mask = np.array([iid == ind for iid in individual_ids])
        
        X_train, y_train = features[train_mask], labels[train_mask]
        X_test, y_test = features[test_mask], labels[test_mask]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_t = torch.FloatTensor(X_train_scaled)
        y_train_t = torch.FloatTensor(y_train)
        X_test_t = torch.FloatTensor(X_test_scaled)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        
        torch.manual_seed(RANDOM_STATE + ind_idx)
        model = MLP(
            input_dim=input_dim,
            hidden_dims=params['hidden_dims'],
            dropout_rate=params['dropout_rate']
        ).to(DEVICE)
        
        model = train_mlp_model(
            model=model, train_loader=train_loader, val_loader=None,
            n_epochs=100, lr=params['learning_rate'], patience=20
        )
        
        model.eval()
        with torch.no_grad():
            predictions = model(X_test_t.to(DEVICE)).cpu().numpy()
        
        all_predictions[test_mask] = predictions
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        spearman, _ = spearmanr(y_test, predictions)
        pearson, _ = pearsonr(y_test, predictions)
        
        fold_metrics.append({
            'Fold': ind, 'Test_Element': ind, 'Individual': ind,
            'MAE': mae, 'RMSE': rmse,
            'Spearman': spearman, 'Pearson': pearson,
            'n_test_samples': len(X_test),
            'Min_celltype_proportion': round(float(y_test.min()), 3),
            'Max_celltype_proportion': round(float(y_test.max()), 3)
        })
        
        print(f"  MAE: {mae:.4f}, Pearson: {pearson:.4f}")
    
    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(os.path.join(output_dir, 'individual_metrics_individual_level.csv'), index=False)
    
    overall_mae = mean_absolute_error(labels, all_predictions)
    overall_pearson, _ = pearsonr(labels, all_predictions)
    
    print(f"\nOverall MAE: {overall_mae:.4f}, Pearson: {overall_pearson:.4f}")
    
    save_scatter_plot(labels, all_predictions, cell_type, model_name, 'MLP', output_dir)
    
    results = {
        'predictions': all_predictions, 'labels': labels,
        'tile_ids': tile_ids, 'individual_ids': individual_ids,
        'sample_ids': sample_ids, 'params': params,
        'cell_type': cell_type, 'model_name': model_name, 'model': 'MLP',
        'overall_metrics': {'MAE': overall_mae, 'Pearson': overall_pearson},
        'fold_metrics': fold_metrics
    }
    
    torch.save(results, os.path.join(output_dir, 'mlp_results_individual_level.pt'))
    return results


# =============================================================================
# Random Forest Functions
# =============================================================================
def grid_search_random_forest(input_dir, cell_type="Cancer Cells", 
                               model_name="Virchow2", subsample_ratio=0.5):
    """Grid search for Random Forest hyperparameters."""
    print("\n" + "="*80)
    print("RANDOM FOREST HYPERPARAMETER GRID SEARCH")
    print(f"Cell Type: {cell_type}, Model: {model_name}")
    print("="*80 + "\n")
    
    data = load_features(input_dir, cell_type, model_name)
    features = data['embeddings'].numpy()
    labels = data['celltype_proportions'].numpy()
    
    np.random.seed(RANDOM_STATE)
    n_samples = int(len(features) * subsample_ratio)
    indices = np.random.choice(len(features), size=n_samples, replace=False)
    X = features[indices]
    y = labels[indices]
    
    print(f"Using {n_samples} samples for grid search")
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.1, 0.2]
    }
    
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=3,
        scoring='neg_mean_squared_error', n_jobs=-1, verbose=2
    )
    
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    
    print(f"\nBest Parameters: {best_params}")
    print(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.6f}")
    
    return best_params


def train_rf_individual_level(input_dir, cell_type, model_name, output_dir, params):
    """Train Random Forest model using individual-level CV."""
    print("\n" + "="*80)
    print(f"TRAINING RANDOM FOREST - {cell_type} - {model_name}")
    print("="*80 + "\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    data = load_features(input_dir, cell_type, model_name)
    features = data['embeddings'].numpy()
    labels = data['celltype_proportions'].numpy()
    tile_ids = data.get('tile_ids', [])
    individual_ids = data.get('individual_ids', [])
    sample_ids = data.get('sample_ids', [])
    
    unique_individuals = sorted(list(set(individual_ids)))
    print(f"Dataset: {len(features)} samples, {len(unique_individuals)} individuals")
    
    all_predictions = np.zeros_like(labels)
    fold_metrics = []
    
    for ind_idx, ind in enumerate(unique_individuals):
        print(f"\nFold {ind_idx+1}/{len(unique_individuals)}: Leaving {ind} out")
        
        train_mask = np.array([iid != ind for iid in individual_ids])
        test_mask = np.array([iid == ind for iid in individual_ids])
        
        X_train, y_train = features[train_mask], labels[train_mask]
        X_test, y_test = features[test_mask], labels[test_mask]
        
        rf = RandomForestRegressor(
            n_estimators=params.get('n_estimators', 200),
            max_depth=params.get('max_depth', 15),
            min_samples_split=params.get('min_samples_split', 5),
            min_samples_leaf=params.get('min_samples_leaf', 2),
            max_features=params.get('max_features', 'sqrt'),
            random_state=RANDOM_STATE, n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)
        all_predictions[test_mask] = predictions
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        spearman, _ = spearmanr(y_test, predictions)
        pearson, _ = pearsonr(y_test, predictions)
        
        fold_metrics.append({
            'Fold': ind, 'Test_Element': ind, 'Individual': ind,
            'MAE': mae, 'RMSE': rmse,
            'Spearman': spearman, 'Pearson': pearson,
            'n_test_samples': len(X_test),
            'Min_celltype_proportion': round(float(y_test.min()), 3),
            'Max_celltype_proportion': round(float(y_test.max()), 3)
        })
        
        print(f"  MAE: {mae:.4f}, Pearson: {pearson:.4f}")
    
    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(os.path.join(output_dir, 'individual_metrics_individual_level.csv'), index=False)
    
    overall_mae = mean_absolute_error(labels, all_predictions)
    overall_pearson, _ = pearsonr(labels, all_predictions)
    
    print(f"\nOverall MAE: {overall_mae:.4f}, Pearson: {overall_pearson:.4f}")
    
    save_scatter_plot(labels, all_predictions, cell_type, model_name, 'RandomForest', output_dir)
    
    results = {
        'predictions': all_predictions, 'labels': labels,
        'tile_ids': tile_ids, 'individual_ids': individual_ids,
        'sample_ids': sample_ids, 'params': params,
        'cell_type': cell_type, 'model_name': model_name, 'model': 'RandomForest',
        'overall_metrics': {'MAE': overall_mae, 'Pearson': overall_pearson},
        'fold_metrics': fold_metrics
    }
    
    torch.save(results, os.path.join(output_dir, 'rf_results_individual_level.pt'))
    return results


# =============================================================================
# Utility Functions
# =============================================================================
def save_scatter_plot(labels, predictions, cell_type, model_name, ml_model, output_dir):
    """Save scatter plot of predictions vs true values."""
    overall_mae = mean_absolute_error(labels, predictions)
    overall_rmse = np.sqrt(mean_squared_error(labels, predictions))
    overall_spearman, _ = spearmanr(labels, predictions)
    overall_pearson, _ = pearsonr(labels, predictions)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(labels, predictions, alpha=0.5, s=10, c='blue')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Identity')
    
    slope, intercept, _, _, _ = stats.linregress(labels, predictions)
    ax.plot([0, 1], [intercept, intercept + slope], 'r-', alpha=0.7, label='Fit')
    
    textstr = (f'MAE: {overall_mae:.4f}\n'
               f'RMSE: {overall_rmse:.4f}\n'
               f'Spearman: {overall_spearman:.4f}\n'
               f'Pearson: {overall_pearson:.4f}\n'
               f'n: {len(labels):,}')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('True Proportion', fontsize=12)
    ax.set_ylabel('Predicted Proportion', fontsize=12)
    ax.set_title(f'{ml_model} - {cell_type} - {model_name}\nIndividual-Level CV', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_plot.png'), dpi=300)
    plt.close()


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    print("\n" + "#"*80)
    print("OTHER ML MODELS COMPARISON")
    print("STPath-COAD: Cell Type Proportion Prediction")
    print("Training: Cancer Cells + Stromal Cells with Virchow2 only")
    print(f"Models: Lasso={TRAIN_LASSO}, MLP={TRAIN_MLP}, RF={TRAIN_RANDOM_FOREST}")
    print("#"*80 + "\n")
    
    all_results_summary = []
    
    # ==========================================================================
    # LASSO
    # ==========================================================================
    if TRAIN_LASSO:
        print("\n" + "#"*80)
        print("LASSO REGRESSION")
        print("#"*80)
        
        lasso_output_dir = os.path.join(BASE_OUTPUT_DIR, 'Lasso')
        os.makedirs(lasso_output_dir, exist_ok=True)
        
        print("\nStep 1: Grid Search")
        best_lasso_params = grid_search_lasso(INPUT_DIR, "Cancer Cells", "Virchow2", 0.5)
        
        with open(os.path.join(lasso_output_dir, 'best_params.pkl'), 'wb') as f:
            pickle.dump(best_lasso_params, f)
        
        print("\nStep 2: Training")
        for cell_type in CELL_TYPES:
            for model_name in FOUNDATION_MODELS:
                cell_output_dir = os.path.join(
                    lasso_output_dir, f"{cell_type}_{model_name}_individual_level"
                )
                results = train_lasso_individual_level(
                    INPUT_DIR, cell_type, model_name, cell_output_dir, best_lasso_params
                )
                all_results_summary.append({
                    'ML_Model': 'Lasso',
                    'Cell_Type': cell_type,
                    'Foundation_Model': model_name,
                    'MAE': results['overall_metrics']['MAE'],
                    'Pearson': results['overall_metrics']['Pearson']
                })
    
    # ==========================================================================
    # MLP
    # ==========================================================================
    if TRAIN_MLP:
        print("\n" + "#"*80)
        print("MULTI-LAYER PERCEPTRON (MLP)")
        print("#"*80)
        
        mlp_output_dir = os.path.join(BASE_OUTPUT_DIR, 'MLP')
        os.makedirs(mlp_output_dir, exist_ok=True)
        
        print("\nStep 1: Grid Search")
        best_mlp_params = grid_search_mlp(INPUT_DIR, "Cancer Cells", "Virchow2", 0.5)
        
        with open(os.path.join(mlp_output_dir, 'best_params.pkl'), 'wb') as f:
            pickle.dump(best_mlp_params, f)
        
        print("\nStep 2: Training")
        for cell_type in CELL_TYPES:
            for model_name in FOUNDATION_MODELS:
                cell_output_dir = os.path.join(
                    mlp_output_dir, f"{cell_type}_{model_name}_individual_level"
                )
                results = train_mlp_individual_level(
                    INPUT_DIR, cell_type, model_name, cell_output_dir, best_mlp_params
                )
                all_results_summary.append({
                    'ML_Model': 'MLP',
                    'Cell_Type': cell_type,
                    'Foundation_Model': model_name,
                    'MAE': results['overall_metrics']['MAE'],
                    'Pearson': results['overall_metrics']['Pearson']
                })
    
    # ==========================================================================
    # RANDOM FOREST
    # ==========================================================================
    if TRAIN_RANDOM_FOREST:
        print("\n" + "#"*80)
        print("RANDOM FOREST")
        print("#"*80)
        
        rf_output_dir = os.path.join(BASE_OUTPUT_DIR, 'RandomForest')
        os.makedirs(rf_output_dir, exist_ok=True)
        
        print("\nStep 1: Grid Search")
        best_rf_params = grid_search_random_forest(INPUT_DIR, "Cancer Cells", "Virchow2", 0.5)
        
        with open(os.path.join(rf_output_dir, 'best_params.pkl'), 'wb') as f:
            pickle.dump(best_rf_params, f)
        
        print("\nStep 2: Training")
        for cell_type in CELL_TYPES:
            for model_name in FOUNDATION_MODELS:
                cell_output_dir = os.path.join(
                    rf_output_dir, f"{cell_type}_{model_name}_individual_level"
                )
                results = train_rf_individual_level(
                    INPUT_DIR, cell_type, model_name, cell_output_dir, best_rf_params
                )
                all_results_summary.append({
                    'ML_Model': 'RandomForest',
                    'Cell_Type': cell_type,
                    'Foundation_Model': model_name,
                    'MAE': results['overall_metrics']['MAE'],
                    'Pearson': results['overall_metrics']['Pearson']
                })
    
    # ==========================================================================
    # Final Summary
    # ==========================================================================
    print("\n" + "#"*80)
    print("TRAINING COMPLETE - SUMMARY")
    print("#"*80 + "\n")
    
    if all_results_summary:
        summary_df = pd.DataFrame(all_results_summary)
        summary_df.to_csv(os.path.join(BASE_OUTPUT_DIR, 'all_models_summary.csv'), index=False)
        
        print(summary_df.to_string(index=False))
        print(f"\nAll results saved to: {BASE_OUTPUT_DIR}")
    
    print("\n" + "#"*80)
    print("DONE!")
    print("#"*80 + "\n")
