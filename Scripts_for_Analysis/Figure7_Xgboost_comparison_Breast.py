"""
FigureS11: Cell Type Proportion Distribution Analysis
======================================================
Analyze cell type proportion distributions across all TCGA BRCA samples

Input: All prediction CSV files from TCGA_Predictions_CSV folder
Output:
1. Distribution plots for each of the 5 merged cell types
2. Violin plot comparing all 5 cell types

Cell Type Merging (8 → 5):
- Cancer Cells (unchanged)
- Normal Epithelial Cells (unchanged)
- T Cells (unchanged)
- Stromal Cells = CAFs + Endothelial
- pan-APC = B Cells + Myeloid + Plasma

Author: Saishi Cui
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from tqdm import tqdm


def load_and_merge_all_predictions(predictions_dir):
    """
    Load all prediction CSVs, merge cell types, and concatenate into one DataFrame
    
    Args:
        predictions_dir (str): Directory containing prediction CSV files
    
    Returns:
        pd.DataFrame: Combined DataFrame with all tiles from all samples
    """
    
    print(f"\n{'='*80}")
    print(f"Loading and merging prediction CSVs")
    print(f"{'='*80}\n")
    
    # Find all prediction CSV files
    csv_files = sorted(glob.glob(os.path.join(predictions_dir, "*_predictions.csv")))
    print(f"Found {len(csv_files)} prediction CSV files\n")
    
    if len(csv_files) == 0:
        raise ValueError(f"No prediction CSV files found in {predictions_dir}")
    
    all_data = []
    
    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # Extract sample name
        sample_name = os.path.basename(csv_file).replace('_predictions.csv', '')
        df['sample_id'] = sample_name
        
        # Merge cell types
        # Stromal = CAFs + Endothelial
        df['Stromal Cells'] = df['Cancer-Associated Fibroblasts'] + df['Endothelial Cells']
        
        # pan-APC = B + Myeloid + Plasma
        df['pan-APC'] = df['B Cells'] + df['Myeloid Cells'] + df['Plasma Cells']
        
        # Keep the 5 merged cell types + sample info
        df_merged = df[[
            'tile_name', 'sample_id',
            'Cancer Cells', 'Normal Epithelial Cells', 'T Cells', 
            'Stromal Cells', 'pan-APC'
        ]].copy()
        
        # Remove white tiles (NaN values)
        df_merged = df_merged.dropna()
        
        all_data.append(df_merged)
    
    # Concatenate all samples
    df_combined = pd.concat(all_data, ignore_index=True)
    
    print(f"\n✅ Data loaded and merged!")
    print(f"   Total samples: {len(csv_files)}")
    print(f"   Total valid tiles: {len(df_combined):,}")
    print(f"   Tiles per sample (avg): {len(df_combined) / len(csv_files):.0f}")
    
    return df_combined


def plot_cell_type_distributions(df_combined, output_dir):
    """
    Create distribution plots for each cell type
    
    Args:
        df_combined (pd.DataFrame): Combined data from all samples
        output_dir (str): Directory to save plots
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    cell_types = ['Cancer Cells', 'Normal Epithelial Cells', 'T Cells', 
                  'Stromal Cells', 'pan-APC']
    
    print(f"\n{'='*80}")
    print(f"Creating distribution plots")
    print(f"{'='*80}\n")
    
    for cell_type in cell_types:
        print(f"  Plotting: {cell_type}")
        
        proportions = df_combined[cell_type] * 100  # Convert to percentage
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ### Subplot 1: Histogram with KDE
        axes[0].hist(proportions, bins=50, density=True, alpha=0.7, 
                    color='skyblue', edgecolor='black', linewidth=1.5)
        
        # Add KDE
        from scipy import stats
        try:
            kde = stats.gaussian_kde(proportions)
            x_range = np.linspace(proportions.min(), proportions.max(), 200)
            axes[0].plot(x_range, kde(x_range), 'r-', linewidth=3, label='KDE')
        except:
            pass
        
        # Add vertical lines for min and max
        min_val = proportions.min()
        max_val = proportions.max()
        ymax = axes[0].get_ylim()[1]
        
        axes[0].axvline(min_val, color='green', linestyle='--', linewidth=2.5, 
                       label=f'Min: {min_val:.2f}%')
        axes[0].axvline(max_val, color='purple', linestyle='--', linewidth=2.5, 
                       label=f'Max: {max_val:.2f}%')
        
        axes[0].legend(fontsize=12, loc='upper right', frameon=True, fancybox=True)
        
        axes[0].set_xlabel('Proportion (%)', fontsize=16, fontweight='bold')
        axes[0].set_ylabel('Density', fontsize=16, fontweight='bold')
        axes[0].set_title(f'{cell_type} Distribution', fontsize=18, fontweight='bold')
        axes[0].tick_params(labelsize=14, width=2)
        for spine in axes[0].spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
        axes[0].grid(True, alpha=0.3)
        
        ### Subplot 2: Box plot with statistics
        bp = axes[1].boxplot([proportions], widths=0.6, patch_artist=True,
                            boxprops=dict(facecolor='lightblue', linewidth=2),
                            medianprops=dict(color='red', linewidth=3),
                            whiskerprops=dict(linewidth=2),
                            capprops=dict(linewidth=2))
        
        # Add statistics text
        mean_val = proportions.mean()
        median_val = proportions.median()
        std_val = proportions.std()
        min_val = proportions.min()
        max_val = proportions.max()
        q25 = proportions.quantile(0.25)
        q75 = proportions.quantile(0.75)
        
        stats_text = f'Mean: {mean_val:.2f}%\n'
        stats_text += f'Median: {median_val:.2f}%\n'
        stats_text += f'Std: {std_val:.2f}%\n'
        stats_text += f'Min: {min_val:.2f}%\n'
        stats_text += f'Max: {max_val:.2f}%\n'
        stats_text += f'Q25: {q25:.2f}%\n'
        stats_text += f'Q75: {q75:.2f}%\n'
        stats_text += f'N: {len(proportions):,}'
        
        axes[1].text(1.3, mean_val, stats_text,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=14, fontweight='bold', verticalalignment='center')
        
        axes[1].set_ylabel('Proportion (%)', fontsize=16, fontweight='bold')
        axes[1].set_title(f'{cell_type} Summary', fontsize=18, fontweight='bold')
        axes[1].set_xticks([])
        axes[1].tick_params(labelsize=14, width=2)
        for spine in axes[1].spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, 
                                  f"{cell_type.replace(' ', '_')}_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Saved: {output_path}")
    
    print(f"\n✅ All distribution plots saved!")


def plot_violin_comparison(df_combined, output_dir):
    """
    Create violin plot comparing all 5 cell types
    
    Args:
        df_combined (pd.DataFrame): Combined data from all samples
        output_dir (str): Directory to save plot
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Creating violin plot comparison")
    print(f"{'='*80}\n")
    
    cell_types = ['Cancer Cells', 'Normal Epithelial Cells', 'T Cells', 
                  'Stromal Cells', 'pan-APC']
    
    # Prepare data for violin plot (convert to long format)
    data_long = []
    for cell_type in cell_types:
        proportions = df_combined[cell_type] * 100  # Convert to percentage
        for prop in proportions:
            data_long.append({
                'Cell Type': cell_type,
                'Proportion (%)': prop
            })
    
    df_long = pd.DataFrame(data_long)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors for each cell type
    colors = {
        'Cancer Cells': '#E41A1C',           # Red
        'Stromal Cells': '#FFFF33',          # Yellow
        'Normal Epithelial Cells': '#377EB8', # Blue
        'T Cells': '#4DAF4A',                # Green
        'pan-APC': '#FF7F00'                 # Orange
    }
    
    palette = [colors[ct] for ct in cell_types]
    
    # Create violin plot
    parts = ax.violinplot(
        [df_combined[ct] * 100 for ct in cell_types],
        positions=range(len(cell_types)),
        widths=0.7,
        showmeans=True,
        showmedians=True,
        showextrema=True
    )
    
    # Color the violin plots
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(palette[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(2)
    
    # Style the median, mean, and extrema lines
    parts['cmedians'].set_edgecolor('red')
    parts['cmedians'].set_linewidth(3)
    parts['cmeans'].set_edgecolor('blue')
    parts['cmeans'].set_linewidth(2)
    parts['cbars'].set_edgecolor('black')
    parts['cbars'].set_linewidth(2)
    parts['cmaxes'].set_edgecolor('black')
    parts['cmaxes'].set_linewidth(2)
    parts['cmins'].set_edgecolor('black')
    parts['cmins'].set_linewidth(2)
    
    # Add box plots on top for quartiles
    bp = ax.boxplot(
        [df_combined[ct] * 100 for ct in cell_types],
        positions=range(len(cell_types)),
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor='white', alpha=0.5, linewidth=2),
        medianprops=dict(color='red', linewidth=2),
        whiskerprops=dict(linewidth=1.5, linestyle='--'),
        capprops=dict(linewidth=1.5),
        showfliers=False
    )
    
    # Set labels and title
    ax.set_xticks(range(len(cell_types)))
    ax.set_xticklabels(cell_types, rotation=45, ha='right', 
                       fontsize=14, fontweight='bold')
    ax.set_ylabel('Cell Type Proportion (%)', fontsize=18, fontweight='bold')
    ax.set_title('Cell Type Proportion Distribution Across All Samples', 
                fontsize=20, fontweight='bold', pad=20)
    
    # Style the axes
    ax.tick_params(labelsize=14, width=2)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color('black')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend for median and mean
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=3, label='Median'),
        Line2D([0], [0], color='blue', linewidth=2, label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14, 
             frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, "Cell_Type_Violin_Comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Violin plot saved: {output_path}\n")
    
    # Print summary statistics
    print(f"Summary Statistics:")
    print(f"{'='*80}")
    for cell_type in cell_types:
        proportions = df_combined[cell_type] * 100
        print(f"\n{cell_type}:")
        print(f"  Mean:   {proportions.mean():.2f}%")
        print(f"  Median: {proportions.median():.2f}%")
        print(f"  Std:    {proportions.std():.2f}%")
        print(f"  Min:    {proportions.min():.2f}%")
        print(f"  Max:    {proportions.max():.2f}%")
        print(f"  Q25:    {proportions.quantile(0.25):.2f}%")
        print(f"  Q75:    {proportions.quantile(0.75):.2f}%")


### Main execution
if __name__ == "__main__":
    
    ### Configuration
    PREDICTIONS_DIR = "/Users/scui2/Desktop/TCGA_Predictions_CSV"
    OUTPUT_DIR = "/Users/scui2/Desktop/TCGA_Cell_Type_Distributions"
    
    ### Step 1: Load and merge all prediction CSVs
    df_combined = load_and_merge_all_predictions(PREDICTIONS_DIR)
    
    ### Step 2: Create distribution plots for each cell type
    plot_cell_type_distributions(df_combined, OUTPUT_DIR)
    
    ### Step 3: Create violin plot comparing all cell types
    plot_violin_comparison(df_combined, OUTPUT_DIR)
    
    print(f"\n{'#'*80}")
    print(f"✅ Analysis Complete!")
    print(f"All plots saved to: {OUTPUT_DIR}")
    print(f"{'#'*80}\n")

