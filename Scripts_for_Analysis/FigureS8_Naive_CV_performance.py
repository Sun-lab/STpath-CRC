"""
Compare feature importance between BRCA and COAD Cancer Cells using Virchow2 models
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr

def load_and_average_importance(model_dir, model_pattern="*.model"):
    """
    Load all XGBoost models from directory and calculate average feature importance
    """
    model_files = glob.glob(os.path.join(model_dir, model_pattern))
    print(f"Found {len(model_files)} model files in {model_dir}")
    

    all_importances = []
    
    for model_file in model_files:
        # Load XGBoost model
        model = xgb.Booster()
        model.load_model(model_file)
        
        # Get feature importance (gain)
        importance_dict = model.get_score(importance_type='gain')
        
        # Convert to array (features are named f0, f1, f2, ...)
        max_feature_idx = max([int(k[1:]) for k in importance_dict.keys()])
        importance_array = np.zeros(max_feature_idx + 1)
        
        for feature_name, importance in importance_dict.items():
            feature_idx = int(feature_name[1:])  # Remove 'f' prefix
            importance_array[feature_idx] = importance
        
        all_importances.append(importance_array)


    # Calculate mean importance across all models
    mean_importance = np.mean(all_importances, axis=0)
    
    print(f"Successfully loaded {len(all_importances)} models")
    print(f"Feature dimension: {len(mean_importance)}")
    
    return mean_importance

def create_comparison_dataframe(brca_importance, coad_importance):
    """
    Create comparison dataframe with normalized BRCA and COAD importance values
    """
    # Ensure both arrays have the same length
    min_length = min(len(brca_importance), len(coad_importance))
    
    # Normalize importance by dividing by sum (relative importance)
    brca_norm = brca_importance[:min_length] / brca_importance[:min_length].sum()
    coad_norm = coad_importance[:min_length] / coad_importance[:min_length].sum()
    
    df = pd.DataFrame({
        'Feature_Index': range(1, min_length + 1),  # 1-based indexing
        'BRCA_Virchow2_Importance': brca_norm,
        'COAD_Virchow2_Importance': coad_norm
    })
    
    return df

def create_scatter_plot(df, output_path):
    """
    Create scatter plot comparing normalized BRCA and COAD feature importance
    """
    plt.figure(figsize=(12, 10))
    
    # Set font properties for bold, large text
    plt.rcParams.update({
        'font.size': 20,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'axes.linewidth': 3,
        'xtick.major.width': 3,
        'ytick.major.width': 3,
        'xtick.color': 'black',
        'ytick.color': 'black',
        'axes.edgecolor': 'black'
    })
    
    # Convert to percentage
    brca_pct = df['BRCA_Virchow2_Importance'] * 100
    coad_pct = df['COAD_Virchow2_Importance'] * 100
    
    # Identify high importance features (>0.5%)
    brca_high = brca_pct > 0.5
    coad_high = coad_pct > 0.5
    
    # Create scatter plot with different colors
    # Default points (gray)
    normal_mask = ~(brca_high | coad_high)
    plt.scatter(brca_pct[normal_mask], coad_pct[normal_mask], 
               alpha=0.6, s=60, color='dimgray', label='Normal features')
    
    # BRCA high importance (blue)
    brca_only_high = brca_high & ~coad_high
    plt.scatter(brca_pct[brca_only_high], coad_pct[brca_only_high], 
               alpha=0.8, s=80, color='blue', label='BRCA >0.5%')
    
    # COAD high importance (red)
    coad_only_high = coad_high & ~brca_high
    plt.scatter(brca_pct[coad_only_high], coad_pct[coad_only_high], 
               alpha=0.8, s=80, color='red', label='COAD >0.5%')
    
    # Both high importance (orange - more visible than purple)
    both_high = brca_high & coad_high
    if both_high.any():  # Only plot if there are any points
        plt.scatter(brca_pct[both_high], coad_pct[both_high], 
                   alpha=0.8, s=80, color='orange')  # No label for legend
    
    # Annotate high importance features with smart positioning to avoid overlap
    import numpy as np
    
    # Collect all high importance features for smart positioning
    all_high_indices = []
    all_high_indices.extend(df.index[brca_only_high].tolist())
    all_high_indices.extend(df.index[coad_only_high].tolist())
    all_high_indices.extend(df.index[both_high].tolist())
    
    # Custom offset positions for specific features (Virchow2 optimized)
    custom_offsets = {
        'f525': (0, -12),      # 下移
        'f1024': (15, 8),      # 右上
        'f512': (-20, 10),     # 左上
        'f256': (12, -15),     # 右下
        'f768': (-15, -12),    # 左下
        'f1536': (8, 18),      # 上方
        'f128': (20, 5),       # 右侧
        'f64': (-25, 8),       # 左上
        'f32': (10, -20),      # 右下
        'f16': (-18, -8),      # 左下
        'f8': (5, 15),         # 上方
        'f4': (-12, 15),       # 左上
        'f2': (18, -8),        # 右下
        'f1': (-8, -18),       # 左下
        'f0': (25, 12),        # 右上
        'f213': (-20, -15),    # 正左平移 + 正下方平移
        'f728': (0, -15)       # 正下方平移
    }
    
    # Default offset positions for other features
    default_offsets = [(8, 8), (10, -12), (-15, 8), (-16, -12), 
                      (15, 2), (-22, 2), (2, 15), (2, -18),
                      (12, -6), (-18, 6), (6, 18), (-10, -15)]
    
    def get_offset_for_feature(feature_name, index):
        if feature_name in custom_offsets:
            return custom_offsets[feature_name]
        else:
            return default_offsets[index % len(default_offsets)]
    
    for i, idx in enumerate(df.index[brca_only_high]):
        feature_name = f"f{df.loc[idx, 'Feature_Index']-1}"  # Convert to 0-based for f naming
        offset = get_offset_for_feature(feature_name, i)
        plt.annotate(feature_name, (brca_pct.iloc[idx], coad_pct.iloc[idx]), 
                    xytext=offset, textcoords='offset points', 
                    fontsize=16, fontweight='bold', color='blue')
    
    for i, idx in enumerate(df.index[coad_only_high]):
        feature_name = f"f{df.loc[idx, 'Feature_Index']-1}"
        offset = get_offset_for_feature(feature_name, i + len(df.index[brca_only_high]))
        plt.annotate(feature_name, (brca_pct.iloc[idx], coad_pct.iloc[idx]), 
                    xytext=offset, textcoords='offset points', 
                    fontsize=16, fontweight='bold', color='red')
    
    for i, idx in enumerate(df.index[both_high]):
        feature_name = f"f{df.loc[idx, 'Feature_Index']-1}"
        offset = get_offset_for_feature(feature_name, i + len(df.index[brca_only_high]) + len(df.index[coad_only_high]))
        plt.annotate(feature_name, (brca_pct.iloc[idx], coad_pct.iloc[idx]), 
                    xytext=offset, textcoords='offset points', 
                    fontsize=16, fontweight='bold', color='orange')
    
    # Add diagonal line for reference
    plt.plot([0, 1.25], [0, 1.25], 'k--', alpha=0.5, linewidth=2)
    
    # Set axis limits
    plt.xlim(0, 1.25)
    plt.ylim(0, 1.25)
    
    # Formatting with bold, large fonts
    plt.xlabel('BRCA Tumor Cells Feature Importance (%)', fontsize=22, fontweight='bold', color='black')
    plt.ylabel('COAD Tumor Cells Feature Importance (%)', fontsize=22, fontweight='bold', color='black')
    plt.title(f'Virchow2 (N features = {len(df)})', fontsize=26, fontweight='bold', color='black')
    
    # Make tick labels bold and black
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_color('black')
    
    
    # Add legend
    plt.legend(loc='upper right', fontsize=16, prop={'weight': 'bold'})
    
    # Grid and styling
    plt.grid(True, alpha=0.3, linewidth=1)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Scatter plot saved to: {output_path}")
    print(f"📊 Total features: {len(df)}")
    print(f"📊 BRCA high importance features (>0.5%): {brca_high.sum()}")
    print(f"📊 COAD high importance features (>0.5%): {coad_high.sum()}")
    print(f"📊 Both high importance features (>0.5%): {both_high.sum()}")



if __name__ == "__main__":
    # Define paths
    brca_model_dir = "/Users/scui2/Desktop/BRCA_XGBoost_Results/Cancer.Epithelial_UNI2h/models"
    coad_model_dir = "/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/xgboost_prediction/Cancer Cells_UNI2h_individual_level_ratio100/models"

    output_dir = "/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/Visual"

    brca_importance = load_and_average_importance(brca_model_dir)

    coad_importance = load_and_average_importance(coad_model_dir)

    comparison_df = create_comparison_dataframe(brca_importance, coad_importance)

    # Create scatter plot
    plot_output = os.path.join(output_dir, "BRCA_COAD_Cancer_UNI2h_Feature_Importance_Scatter.png")
    create_scatter_plot(comparison_df, plot_output)

