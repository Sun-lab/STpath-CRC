"""
STPath-COAD: Within-Sample XGBoost Model Training and Evaluation
===============================================================

This script trains XGBoost models for predicting cell type proportions in colorectal cancer
H&E patches using features extracted from multiple foundation models with within-sample
80/20 train/test split instead of Leave-One-Individual-Out cross-validation.

Author: Saishi Cui
Date: December 2025

Purpose: Train and evaluate XGBoost models for cell type proportion prediction using
80% training and 20% testing split for all 5 cell types and 6 foundation models.
"""

import torch
import numpy as np
import pandas as pd
import os
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def train_xgboost_within_sample(
    input_dir='/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/Training_features',
    cell_type="Cancer Cells",
    model_name='UNI2h',
    params=None,
    num_boost_round=800,
    seed=42,
    test_size=0.2
):
    """
    Train XGBoost model with 80/20 train/test split (within-sample validation)
    
    Args:
        input_dir: Feature input directory
        cell_type: Cell type for prediction
        model_name: Model name ('ResNet50', 'Conch', 'Prov-GigaPath', 'UNI2-h', 'Virchow', 'Virchow2')
        params: XGBoost parameters, if None default parameters will be used
        num_boost_round: Number of training rounds
        seed: Random seed
        test_size: Proportion of dataset to include in the test split
    
    Returns:
        dict: Training results with Pearson correlation and MAE
    """
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set default XGBoost parameters (same as original script)
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
    
    # Map display names to file names
    model_file_names = {
        'ResNet50': 'Resnet50',
        'Conch': 'Conch',
        'Prov-GigaPath': 'ProvGigapath',
        'UNI2-h': 'UNI2h',
        'Virchow': 'Virchow',
        'Virchow2': 'Virchow2'
    }
    
    file_model_name = model_file_names[model_name]
    
    # Load feature data
    feature_name = f'{cell_type}_training_precomputed_features_{file_model_name}.pt'
    data_path = os.path.join(input_dir, feature_name)
    print(f"Loading data: {data_path}")
    
    data = torch.load(data_path)
    features = data['embeddings'].numpy()
    labels = data['celltype_proportions'].numpy()
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Split data into train and test sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=seed, shuffle=True
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Train model
    print(f"Training XGBoost model for {cell_type} using {model_name}...")
    xgb_model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        verbose_eval=False
    )
    
    # Make predictions on test set
    y_pred = xgb_model.predict(dtest)
    
    # Calculate metrics
    pearson_corr, _ = pearsonr(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Results for {model_name} - {cell_type}:")
    print(f"  Pearson Correlation: {pearson_corr:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    return {
        'Model': model_name,
        'Cell_Type': cell_type,
        'Pearson_Correlation': pearson_corr,
        'MAE': mae,
        'n_train': len(y_train),
        'n_test': len(y_test)
    }


def create_barplots(results_df, output_dir, timestamp):
    """
    Create bar plots to compare model performance across cell types
    Similar to Figure4 Panel A but using bar plots instead of box plots
    """
    
    print("\n" + "="*80)
    print("Creating bar plots for model comparison...")
    print("="*80)
    
    # Map cell type names for display (same as Figure4)
    results_df = results_df.copy()
    results_df['Cell_Type'] = results_df['Cell_Type'].replace('Other Immune Cells', 'pan-APC')
    
    # Set figure style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Define color mapping (same as Figure4)
    color_palette = {
        'ResNet50': '#FEF0DE',
        'Conch': '#C43E96', 
        'Prov-GigaPath': '#DEDBEE',
        'UNI2-h': '#06948E',
        'Virchow': '#F3CDCC',
        'Virchow2': '#F0CF7F'
    }
    
    # Define model order (same as Figure4)
    model_order = ['ResNet50', 'Conch', 'Prov-GigaPath', 'UNI2-h', 'Virchow', 'Virchow2']
    
    # Define cell type order (same as Figure4)
    cell_type_order = ['Cancer Cells', 'Stromal Cells', 'Normal Epithelial Cells', 'T Cells', 'pan-APC']
    
    # Define metrics to plot
    metrics = [
        ('Pearson_Correlation', 'Pearson Correlation'),
        ('MAE', 'Mean Absolute Error (MAE)')
    ]
    
    # Create bar plots for each metric
    for metric_col, metric_title in metrics:
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(18, 8))
        
        # Prepare data for plotting
        plot_data = []
        x_positions = []
        colors = []
        x_labels = []
        
        bar_width = 0.13  # Width of each bar (6 models)
        x_base = np.arange(len(cell_type_order))
        
        for cell_idx, cell_type in enumerate(cell_type_order):
            for model_idx, model in enumerate(model_order):
                # Get the value for this model-cell type combination
                value = results_df[(results_df['Model'] == model) & 
                                 (results_df['Cell_Type'] == cell_type)][metric_col].values
                
                if len(value) > 0 and not np.isnan(value[0]):
                    plot_data.append(value[0])
                    x_pos = x_base[cell_idx] + (model_idx - 2.5) * bar_width
                    x_positions.append(x_pos)
                    colors.append(color_palette[model])
                else:
                    # Handle missing data
                    plot_data.append(0)
                    x_pos = x_base[cell_idx] + (model_idx - 2.5) * bar_width
                    x_positions.append(x_pos)
                    colors.append('lightgray')
        
        # Create bar plot with thicker black borders
        bars = ax.bar(x_positions, plot_data, width=bar_width, color=colors, 
                     edgecolor='black', linewidth=3, alpha=0.8)  # Increased linewidth to 3
        
        # Customize the plot (same style as Figure4)
        ax.set_xlabel('', fontsize=22, fontweight='bold')  # No x-axis title
        ax.set_ylabel(metric_title, fontsize=22, fontweight='bold')
        
        # Set Y-axis limits based on metric type
        if metric_col == 'Pearson_Correlation':
            ax.set_ylim(0, 1)  # Pearson correlation: 0-1
        elif metric_col == 'MAE':
            ax.set_ylim(0, 0.2)  # MAE: 0-0.2
        
        # Set x-axis ticks and labels
        ax.set_xticks(x_base)
        # Modify cell type labels (same as Figure4)
        x_tick_labels = []
        for cell_type in cell_type_order:
            label = cell_type.replace(' Cells', '').replace('Cancer', 'Tumor')
            x_tick_labels.append(label)
        
        ax.set_xticklabels(x_tick_labels, fontsize=20, fontweight='bold', rotation=25)
        
        # Style ticks
        ax.tick_params(axis='y', labelsize=20, labelcolor='black', width=2, length=6)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        
        # Add grid
        ax.grid(True, alpha=0.7, linestyle='--', linewidth=2, axis='y')
        
        # Add thick black borders (same as Figure4)
        ax.spines['top'].set_linewidth(5)
        ax.spines['right'].set_linewidth(5)
        ax.spines['bottom'].set_linewidth(5)
        ax.spines['left'].set_linewidth(5)
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        
        # Create legend outside the plot area (right side)
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_palette[model], 
                                       edgecolor='black', linewidth=1, label=model) 
                         for model in model_order]
        
        legend = ax.legend(handles=legend_elements, 
                          loc='center left', 
                          bbox_to_anchor=(1.02, 0.5),  # Position outside the plot area
                          fontsize=16, 
                          frameon=True, 
                          fancybox=True, 
                          shadow=True,
                          borderpad=1,
                          handlelength=1.5,
                          handletextpad=0.5)
        
        # Style the legend
        for text in legend.get_texts():
            text.set_fontweight('bold')
        legend.get_frame().set_linewidth(2)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.95)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'WithinSample_Barplot_{metric_col}_{timestamp}.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=600, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✅ Bar plot saved: {plot_path}")
    
    print("="*80)
    print("Bar plots creation completed!")
    print("="*80)

if __name__ == "__main__":
    # Define models and cell types
    models = ['ResNet50', 'Conch', 'Prov-GigaPath', 'UNI2-h', 'Virchow', 'Virchow2']
    cell_types = ["Cancer Cells", "Stromal Cells", "Normal Epithelial Cells", "T Cells", "Other Immune Cells"]

    # Initialize results list
    all_results = []

    print("="*80)
    print("STPath-COAD: Within-Sample XGBoost Training and Evaluation")
    print("="*80)
    print(f"Models: {models}")
    print(f"Cell Types: {cell_types}")
    print(f"Total combinations: {len(models)} × {len(cell_types)} = {len(models) * len(cell_types)}")
    print("="*80)

    # Run training for all combinations
    for model_idx, model in enumerate(models):
        for cell_idx, cell_type in enumerate(cell_types):
            combination_num = model_idx * len(cell_types) + cell_idx + 1
            print(f"\n[{combination_num}/{len(models) * len(cell_types)}] Processing {model} - {cell_type}")
            print("-" * 60)
            
            try:
                result = train_xgboost_within_sample(
                    input_dir='/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/Training_features',
                    cell_type=cell_type,
                    model_name=model,
                    params=None,
                    num_boost_round=800,
                    seed=42,
                    test_size=0.2
                )
                all_results.append(result)
                
            except Exception as e:
                print(f"❌ Error processing {model} - {cell_type}: {str(e)}")
                # Add a result with NaN values for failed cases
                all_results.append({
                    'Model': model,
                    'Cell_Type': cell_type,
                    'Pearson_Correlation': np.nan,
                    'MAE': np.nan,
                    'n_train': np.nan,
                    'n_test': np.nan
                })

    # Create DataFrame with results
    results_df = pd.DataFrame(all_results)

    # Reorder columns for better readability
    results_df = results_df[['Model', 'Cell_Type', 'Pearson_Correlation', 'MAE', 'n_train', 'n_test']]

    # Create output directory
    output_dir = '/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/xgboost_prediction/WithinSample_Results'
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'WithinSample_Results_{timestamp}.csv')
    results_df.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    print(f"Results saved to: {output_path}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Mean Pearson Correlation: {results_df['Pearson_Correlation'].mean():.4f} ± {results_df['Pearson_Correlation'].std():.4f}")
    print(f"Mean MAE: {results_df['MAE'].mean():.4f} ± {results_df['MAE'].std():.4f}")
    
    # Create bar plots for comparison
    create_barplots(results_df, output_dir, timestamp)


