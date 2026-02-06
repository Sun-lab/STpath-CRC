"""
Figure 7: XGBoost Model Comparison (Breast cancer)
========================================

This script generates comparison visualizations for breast cancer XGBoost models:
1. MAE boxplot comparing 7 models (5 foundation models + 1 resnet50 model + 1 foundation models combined)
2. Pearson correlation boxplot comparing 7 models  
3. BRCA vs COAD feature importance scatter plots for UNI2-h and Virchow2


Author: Saishi Cui
Date: Feb 2026
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
from collections import defaultdict
from adjustText import adjust_text

# Configuration
BRCA_RESULTS_DIR = '/BRCA_XGBoost_Results/New_common_samples'
COAD_RESULTS_DIR = '/xgboost_prediction'
OUTPUT_DIR = '/BRCA_XGBoost_Results/New_common_samples/Visualizations'

os.makedirs(OUTPUT_DIR, exist_ok=True)

CELL_TYPES = ['Tumor', 'Stromal', 'pan_APC', 'T_cell', 'Normal_Epithelial']
MODELS = ['ResNet50', 'Conch', 'ProvGigapath', 'UNI2h', 'Virchow', 'Virchow2', 'Combined']

# Display names mapping
MODEL_DISPLAY = {
    'ResNet50': 'ResNet50',
    'Conch': 'Conch',
    'ProvGigapath': 'Prov-GigaPath',
    'UNI2h': 'UNI2-h',
    'Virchow': 'Virchow',
    'Virchow2': 'Virchow2',
    'Combined': 'Combined'
}

CELL_TYPE_DISPLAY = {
    'Tumor': 'Tumor',
    'Stromal': 'Stromal',
    'pan_APC': 'pan-APC',
    'T_cell': 'T',
    'Normal_Epithelial': 'Normal Epithelial'
}

# Color palette (following COAD Figure4 style)
color_palette = {
    'ResNet50': '#FEF0DE',
    'Conch': '#C43E96', 
    'Prov-GigaPath': '#DEDBEE',
    'UNI2-h': '#06948E',
    'Virchow': '#F3CDCC',
    'Virchow2': '#F0CF7F',
    'Combined': '#FF6B6B'
}

# Feature importance comparison configuration
FEATURE_IMPORTANCE_MODELS = {
    'UNI2-h': {'brca_name': 'UNI2h', 'coad_name': 'UNI2h', 'n_features': 1536},
    'Virchow2': {'brca_name': 'Virchow2', 'coad_name': 'Virchow2', 'n_features': 2560}
}

BRCA_THRESHOLD = 1.0  # 1.0% for BRCA (blue)
COAD_THRESHOLD = 0.5  # 0.5% for COAD (red)


def generate_boxplots():
    """Generate MAE and Pearson correlation boxplots"""
    print("\n" + "="*80)
    print("Generating Boxplots")
    print("="*80)
    
    # Collect all results
    all_results = []
    
    for cell_type in CELL_TYPES:
        for model in MODELS:
            csv_path = f"{BRCA_RESULTS_DIR}/{cell_type}_{model}_individual_level/individual_metrics_individual_level.csv"
            
            if not os.path.exists(csv_path):
                print(f"Warning: {csv_path} not found")
                continue
            
            df = pd.read_csv(csv_path)
            
            for index, row in df.iterrows():
                all_results.append({
                    "model_name": MODEL_DISPLAY[model],
                    "cell_type": CELL_TYPE_DISPLAY[cell_type],
                    "MAE": row["MAE"],
                    "Pearson": row["Pearson"],
                    "Individual": row["Individual"]
                })
    
    df_all = pd.DataFrame(all_results)
    print(f"Collected {len(df_all)} results")
    
    # Define model order
    model_order = ['ResNet50', 'Conch', 'Prov-GigaPath', 'UNI2-h', 'Virchow', 'Virchow2', 'Combined']
    cell_type_order = ['Tumor', 'Stromal', 'pan-APC', 'T', 'Normal Epithelial']
    
    # Generate both MAE and Pearson boxplots
    for metric in ['MAE', 'Pearson']:
        print(f"\nGenerating {metric} boxplot...")
        
        fig, ax = plt.subplots(figsize=(18, 8))
        
        # Create boxplot
        sns.boxplot(
            data=df_all,
            x='cell_type',
            y=metric,
            hue='model_name',
            hue_order=model_order,
            order=cell_type_order,
            palette=color_palette,
            showfliers=False,
            linewidth=3,
            ax=ax
        )
        
        # Add stripplot
        sns.stripplot(
            data=df_all,
            x='cell_type',
            y=metric,
            hue='model_name',
            hue_order=model_order,
            order=cell_type_order,
            palette=color_palette,
            size=6,
            alpha=0.7,
            dodge=True,
            jitter=0.3,
            edgecolor='black',
            linewidth=0.5,
            ax=ax,
            legend=False
        )
        
        # Styling
        ax.set_xlabel('', fontsize=24, fontweight='bold')
        ylabel = 'Mean Absolute Error (MAE)' if metric == 'MAE' else 'Pearson Correlation'
        ax.set_ylabel(ylabel, fontsize=24, fontweight='bold')
        ax.set_title(f'BRCA Model Comparison: {ylabel}', fontsize=26, fontweight='bold')
        
        # Set y-axis limits
        if metric == 'MAE':
            ax.set_ylim(0, 0.4)
        else:  # Pearson
            ax.set_ylim(-0.5, 1.0)
        
        # Tick styling
        ax.tick_params(axis='x', rotation=15, labelsize=20, labelcolor='black', 
                      width=2, length=6, colors='black')
        ax.tick_params(axis='y', labelsize=20, labelcolor='black', 
                      width=2, length=6, colors='black')
        
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        # Legend
        handles, labels = ax.get_legend_handles_labels()
        n_models = len(model_order)
        ax.legend(handles[:n_models], labels[:n_models], 
                 loc='upper right', fontsize=18, framealpha=0.9)
        legend = ax.get_legend()
        for text in legend.get_texts():
            text.set_fontweight('bold')
        
        # Grid
        ax.grid(True, alpha=0.3, linewidth=1)
        
        # Thick borders
        for spine in ax.spines.values():
            spine.set_linewidth(5)
            spine.set_color('black')
        
        plt.tight_layout()
        
        # Save
        output_path = f'{OUTPUT_DIR}/BRCA_model_comparison_{metric}_boxplot_with_Combined.png'
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
        
        plt.close()
    
    print("\n✅ Boxplots created successfully!")


def load_feature_importance(pt_file_path, n_features):
    """Load feature importance and compute average across folds as percentage"""
    data = torch.load(pt_file_path, weights_only=False)
    feature_importances = data['feature_importances']
    
    # Aggregate importances across all folds
    aggregated = defaultdict(list)
    for fold_importance in feature_importances:
        for feat, val in fold_importance.items():
            aggregated[feat].append(val)
    
    # Compute mean importance for each feature
    mean_importance = {}
    for feat, vals in aggregated.items():
        mean_importance[feat] = np.mean(vals)
    
    # Fill in missing features with 0
    for i in range(n_features):
        feat_name = f'f{i}'
        if feat_name not in mean_importance:
            mean_importance[feat_name] = 0.0
    
    # Convert to percentage (sum to 100%)
    total = sum(mean_importance.values())
    if total > 0:
        for feat in mean_importance:
            mean_importance[feat] = (mean_importance[feat] / total) * 100
    
    return mean_importance


def create_scatter_plot(brca_importance, coad_importance, model_name, n_features, output_path):
    """Create scatter plot comparing BRCA vs COAD feature importance"""
    
    # Prepare data
    features = [f'f{i}' for i in range(n_features)]
    brca_vals = np.array([brca_importance.get(f, 0) for f in features])
    coad_vals = np.array([coad_importance.get(f, 0) for f in features])
    
    # Categorize features with different thresholds
    normal_mask = (brca_vals <= BRCA_THRESHOLD) & (coad_vals <= COAD_THRESHOLD)
    brca_high_mask = (brca_vals > BRCA_THRESHOLD) & (coad_vals <= COAD_THRESHOLD)
    coad_high_mask = (coad_vals > COAD_THRESHOLD) & (brca_vals <= BRCA_THRESHOLD)
    both_high_mask = (brca_vals > BRCA_THRESHOLD) & (coad_vals > COAD_THRESHOLD)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot normal features (grey)
    ax.scatter(brca_vals[normal_mask], coad_vals[normal_mask], 
               c='grey', alpha=0.5, s=50, label='Normal features', zorder=1)
    
    # Plot BRCA high features (blue) - threshold 1.0%
    ax.scatter(brca_vals[brca_high_mask], coad_vals[brca_high_mask], 
               c='blue', alpha=0.8, s=80, label=f'BRCA >{BRCA_THRESHOLD}%', zorder=2)
    
    # Plot COAD high features (red) - threshold 0.5%
    ax.scatter(brca_vals[coad_high_mask], coad_vals[coad_high_mask], 
               c='red', alpha=0.8, s=80, label=f'COAD >{COAD_THRESHOLD}%', zorder=2)
    
    # Plot both high features (purple) - using both thresholds
    ax.scatter(brca_vals[both_high_mask], coad_vals[both_high_mask], 
               c='purple', alpha=0.9, s=120, label=f'Both (BRCA>{BRCA_THRESHOLD}%, COAD>{COAD_THRESHOLD}%)', 
               edgecolor='black', linewidth=1.5, zorder=3)
    
    # Annotate features above thresholds
    texts = []
    for i, (brca_val, coad_val) in enumerate(zip(brca_vals, coad_vals)):
        if brca_val > BRCA_THRESHOLD or coad_val > COAD_THRESHOLD:
            texts.append(ax.text(brca_val, coad_val, features[i], 
                                fontsize=18, ha='center', va='bottom'))
    
    # Adjust text to avoid overlap (no arrows)
    if texts:
        adjust_text(texts, ax=ax)
    
    # Diagonal line y=x
    max_val = max(brca_vals.max(), coad_vals.max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=2, label='y=x', zorder=0)
    
    # Styling
    ax.set_xlabel('Breast Tumor Cells Feature Importance (%)', fontsize=22, fontweight='bold')
    ax.set_ylabel('Colorectal Tumor Cells Feature Importance (%)', fontsize=22, fontweight='bold')
    ax.set_title(f'Feature Importance: BRCA vs COAD ({model_name})', 
                fontsize=24, fontweight='bold')
    
    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(20)
    
    # Add legend with bold font
    legend = ax.legend(loc='upper right', fontsize=18, framealpha=0.9)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Thick borders
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {output_path}")
    
    # Print statistics
    print(f"  Statistics for {model_name}:")
    print(f"    Normal features: {np.sum(normal_mask)}")
    print(f"    BRCA >{BRCA_THRESHOLD}%: {np.sum(brca_high_mask)}")
    print(f"    COAD >{COAD_THRESHOLD}%: {np.sum(coad_high_mask)}")
    print(f"    Both (BRCA>{BRCA_THRESHOLD}%, COAD>{COAD_THRESHOLD}%): {np.sum(both_high_mask)}")


def generate_feature_importance_scatter():
    """Generate BRCA vs COAD feature importance scatter plots"""
    print("\n" + "="*80)
    print("BRCA vs COAD Feature Importance Comparison")
    print("="*80)
    
    for model_display_name, config in FEATURE_IMPORTANCE_MODELS.items():
        print(f"\nProcessing {model_display_name}...")
        
        # Paths
        brca_pt = f"{BRCA_RESULTS_DIR}/Tumor_{config['brca_name']}_individual_level/xgboost_results_individual_level.pt"
        coad_pt = f"{COAD_RESULTS_DIR}/Cancer Cells_{config['coad_name']}_individual_level_ratio100/xgboost_results_individual_level.pt"
        
        # Check files exist
        if not os.path.exists(brca_pt):
            print(f"  Warning: BRCA file not found: {brca_pt}")
            continue
        if not os.path.exists(coad_pt):
            print(f"  Warning: COAD file not found: {coad_pt}")
            continue
        
        # Load feature importances
        print(f"  Loading BRCA importance...")
        brca_importance = load_feature_importance(brca_pt, config['n_features'])
        
        print(f"  Loading COAD importance...")
        coad_importance = load_feature_importance(coad_pt, config['n_features'])
        
        # Create scatter plot
        filename_safe = model_display_name.replace('-', '_')
        output_path = f"{OUTPUT_DIR}/BRCA_vs_COAD_{filename_safe}_feature_importance.png"
        create_scatter_plot(brca_importance, coad_importance, model_display_name, 
                           config['n_features'], output_path)
    
    print("\n" + "="*80)
    print("✅ Scatter plots created!")
    print("="*80)


def main():
    """Main execution"""
    print("="*80)
    print("Figure 7: BRCA XGBoost Model Comparison")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}\n")
    
    # Generate boxplots
    generate_boxplots()
    
    # Generate feature importance scatter plots
    generate_feature_importance_scatter()
    
    print("\n" + "="*80)
    print("✅ All visualizations complete!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  1. BRCA_model_comparison_MAE_boxplot_with_Combined.png")
    print(f"  2. BRCA_model_comparison_Pearson_boxplot_with_Combined.png")
    print(f"  3. BRCA_vs_COAD_UNI2_h_feature_importance.png")
    print(f"  4. BRCA_vs_COAD_Virchow2_feature_importance.png")
    print(f"\nAll files saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
