"""
Figure 6: TCGA COAD Analysis Visualizations
============================================

This script generates hexagonal heatmap visualizations for TCGA COAD data analysis:
1. Cell type proportion hexagon heatmaps (5 cell types)
2. Hard classification hexagon maps (4 cell types)
3. Minimal directional distance hexagon heatmaps (12 directional pairs)

Prerequisites:
- Must run TCGA_COAD_Analysis.py first to generate:
  * Grouped & normalized predictions CSVs
  
Author: Saishi Cui
Date: Feb 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm

# Configuration
INPUT_DIR = "/TCGA/TCGA_COAD/TCGA_COAD_Grouped_Normalized_Predictions"
OUTPUT_HEXAGON_DIR = "/TCGA/TCGA_COAD/TCGA_COAD_Hexagon_Heatmaps_Figure6"
OUTPUT_CLASSIFICATION_DIR = "/TCGA/TCGA_COAD/TCGA_COAD_Classification_Maps_Figure6"
OUTPUT_DISTANCE_DIR = "/TCGA/TCGA_COAD/TCGA_COAD_Distance_Heatmaps_Figure6"

# Cell types
CELL_TYPES_ALL = ['Cancer', 'Stromal', 'pan-APC', 'T_Cells', 'Normal_Epithelial']
CELL_TYPES_DISTANCE = ['Cancer', 'Stromal', 'pan-APC', 'T_Cells']

# Classification thresholds
CLASSIFICATION_THRESHOLDS = {
    'Cancer': 0.30,
    'Stromal': 0.40,
    'pan-APC': 0.30,
    'T_Cells': 0.10
}

# Cell type colors
CELL_TYPE_COLORS = {
    'Cancer': '#E41A1C',
    'Stromal': '#FFFF33',
    'pan-APC': '#FF7F00',
    'T_Cells': '#4DAF4A'
}


def create_output_directories():
    """Create all output directories"""
    os.makedirs(OUTPUT_HEXAGON_DIR, exist_ok=True)
    os.makedirs(OUTPUT_CLASSIFICATION_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DISTANCE_DIR, exist_ok=True)
    
    # Hexagon proportion subdirectories (5 cell types)
    hexagon_subdirs = {}
    for i, cell_type in enumerate(CELL_TYPES_ALL, 1):
        subdir = os.path.join(OUTPUT_HEXAGON_DIR, f"{i:02d}_{cell_type}")
        os.makedirs(subdir, exist_ok=True)
        hexagon_subdirs[cell_type] = subdir
    
    # Classification subdirectories (4 cell types, no Normal_Epithelial)
    classif_subdirs = {}
    for cell_type in CELL_TYPES_DISTANCE:
        subdir = os.path.join(OUTPUT_CLASSIFICATION_DIR, cell_type)
        os.makedirs(subdir, exist_ok=True)
        classif_subdirs[cell_type] = subdir
    
    # Distance subdirectories (12 directional pairs)
    directional_pairs = []
    for source in CELL_TYPES_DISTANCE:
        for target in CELL_TYPES_DISTANCE:
            if source != target:
                directional_pairs.append((source, target))
    
    distance_subdirs = {}
    for source, target in directional_pairs:
        subdir = os.path.join(OUTPUT_DISTANCE_DIR, f"From_{source}_To_{target}")
        os.makedirs(subdir, exist_ok=True)
        distance_subdirs[(source, target)] = subdir
    
    return hexagon_subdirs, classif_subdirs, distance_subdirs, directional_pairs


def calculate_marker_size(max_row, max_col):
    """Calculate marker size based on grid dimensions"""
    estimated_grid_size = max(max_row, max_col)
    base_size = 150
    size_factor = (40 / estimated_grid_size) ** 1.5
    marker_size = max(20, int(base_size * size_factor))
    return marker_size


def generate_proportion_heatmaps(df, sample_name, hexagon_subdirs):
    """Generate hexagon heatmaps for each cell type proportion"""
    print(f"\n  Generating proportion hexagon heatmaps...")
    
    # Get grid dimensions
    max_row = int(df['row'].max())
    max_col = int(df['col'].max())
    marker_size = calculate_marker_size(max_row, max_col)
    
    # Filter valid tiles (non-white)
    valid_mask = df[[f'{ct}_normalized' for ct in CELL_TYPES_ALL]].notna().all(axis=1)
    white_mask = ~valid_mask
    
    for cell_type in CELL_TYPES_ALL:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Prepare plot data (exclude white tiles)
        plot_data = []
        for idx, row_data in df.iterrows():
            if pd.isna(row_data['row']) or pd.isna(row_data['col']):
                continue
            if white_mask[idx]:
                continue
            
            proportion = row_data[f'{cell_type}_normalized']
            if pd.notna(proportion):
                plot_data.append({
                    'x': row_data['col'],
                    'y': row_data['row'],
                    'proportion': proportion
                })
        
        if len(plot_data) == 0:
            plt.close()
            print(f"    ⚠️  No valid data for {cell_type}, skipping...")
            continue
        
        df_plot = pd.DataFrame(plot_data)
        
        # Get min and max values
        vmin_val = df_plot['proportion'].min()
        vmax_val = df_plot['proportion'].max()
        
        # Create scatter plot
        scatter = ax.scatter(
            x=df_plot['x'],
            y=df_plot['y'],
            c=df_plot['proportion'],
            cmap='RdBu_r',
            marker='h',
            s=marker_size,
            edgecolors='black',
            linewidth=0.5,
            vmin=vmin_val,
            vmax=vmax_val
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Proportion', fontsize=28, fontweight='black', color='black')
        cbar.ax.tick_params(labelsize=26, width=2, length=6, color='black', labelcolor='black')
        for label in cbar.ax.get_yticklabels():
            label.set_fontsize(26)
            label.set_fontweight('black')
            label.set_color('black')
        plt.setp(cbar.ax.get_yticklabels(), fontweight='black', fontsize=26, color='black')
        cbar.outline.set_linewidth(2)
        cbar.outline.set_edgecolor('black')
        
        # Invert y-axis
        ax.invert_yaxis()
        
        # Set axis style
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3)
            spine.set_color('black')
        
        # Remove labels and ticks
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(hexagon_subdirs[cell_type], f"{sample_name}_{cell_type}_heatmap.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        display_name = cell_type.replace('_', ' ')
        print(f"    ✅ {display_name}: saved")


def generate_classification_maps(df, sample_name, classif_subdirs):
    """Generate classification hexagon maps for each cell type"""
    print(f"\n  Generating classification hexagon maps...")
    
    # Filter valid tiles
    valid_mask = df[[f'{ct}_normalized' for ct in CELL_TYPES_ALL]].notna().all(axis=1)
    df_valid = df[valid_mask].copy()
    
    if len(df_valid) == 0:
        print(f"    ⚠️  No valid tiles, skipping...")
        return
    
    # Get grid dimensions
    max_row = int(df_valid['row'].max())
    max_col = int(df_valid['col'].max())
    marker_size = calculate_marker_size(max_row, max_col)
    
    # Perform hard classification
    classified_tiles = {ct: [] for ct in CELL_TYPES_DISTANCE}
    for idx, row_data in df_valid.iterrows():
        for cell_type in CELL_TYPES_DISTANCE:
            if row_data[f'{cell_type}_normalized'] > CLASSIFICATION_THRESHOLDS[cell_type]:
                classified_tiles[cell_type].append(idx)
    
    # Generate map for each cell type
    for cell_type in CELL_TYPES_DISTANCE:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot all valid tiles in light gray
        ax.scatter(
            x=df_valid['col'],
            y=df_valid['row'],
            c='#D3D3D3',
            marker='h',
            s=marker_size,
            edgecolors='black',
            linewidth=0.3,
            alpha=0.5,
            zorder=1
        )
        
        # Overlay classified tiles
        if len(classified_tiles[cell_type]) > 0:
            classified_df = df_valid.loc[classified_tiles[cell_type]]
            ax.scatter(
                x=classified_df['col'],
                y=classified_df['row'],
                c=CELL_TYPE_COLORS[cell_type],
                marker='h',
                s=marker_size,
                edgecolors='black',
                linewidth=0.5,
                alpha=0.8,
                zorder=2,
                label=f'{cell_type} (n={len(classified_tiles[cell_type])})'
            )
        
        # Invert y-axis
        ax.invert_yaxis()
        
        # Set axis style
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3)
            spine.set_color('black')
        
        # Remove labels and ticks
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add legend
        ax.legend(fontsize=12, loc='upper right')
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(classif_subdirs[cell_type], f"{sample_name}_{cell_type}_classification.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ {cell_type} classification map saved")


def generate_distance_heatmaps(df, sample_name, distance_subdirs, directional_pairs):
    """Generate distance hexagon heatmaps for all directional pairs"""
    print(f"\n  Generating distance hexagon heatmaps...")
    
    # Filter valid tiles
    valid_mask = df[[f'{ct}_normalized' for ct in CELL_TYPES_ALL]].notna().all(axis=1)
    df_valid = df[valid_mask].copy()
    
    if len(df_valid) == 0:
        print(f"    ⚠️  No valid tiles, skipping...")
        return
    
    # Get grid dimensions
    max_row = int(df_valid['row'].max())
    max_col = int(df_valid['col'].max())
    marker_size = calculate_marker_size(max_row, max_col)
    
    # Classify tiles for each cell type
    classified_tiles = {ct: [] for ct in CELL_TYPES_DISTANCE}
    for idx, row_data in df_valid.iterrows():
        for cell_type in CELL_TYPES_DISTANCE:
            if row_data[f'{cell_type}_normalized'] > CLASSIFICATION_THRESHOLDS[cell_type]:
                classified_tiles[cell_type].append(idx)
    
    # Generate heatmap for each directional pair
    for source, target in directional_pairs:
        if len(classified_tiles[source]) == 0 or len(classified_tiles[target]) == 0:
            print(f"    ⚠️  {source} → {target}: No classified tiles, skipping...")
            continue
        
        # Get source and target tiles
        source_tiles_df = df_valid.loc[classified_tiles[source]]
        target_tiles_df = df_valid.loc[classified_tiles[target]]
        
        # Calculate minimum distance for each source tile
        tile_min_distances = {}
        for idx_source, source_tile in source_tiles_df.iterrows():
            source_row = source_tile['row']
            source_col = source_tile['col']
            source_proportion = source_tile[f'{source}_normalized']
            
            # Find minimum distance to any target tile
            min_distance = np.inf
            for idx_target, target_tile in target_tiles_df.iterrows():
                target_row = target_tile['row']
                target_col = target_tile['col']
                distance = np.sqrt((target_row - source_row)**2 + (target_col - source_col)**2)
                if distance < min_distance:
                    min_distance = distance
            
            # Weight by source proportion
            weighted_min_dist = min_distance * source_proportion
            tile_min_distances[(source_row, source_col)] = weighted_min_dist
        
        # Prepare plot data
        plot_data_hex = []
        for (row, col), dist in tile_min_distances.items():
            plot_data_hex.append({'x': col, 'y': row, 'distance': dist})
        
        if len(plot_data_hex) == 0:
            print(f"    ⚠️  {source} → {target}: No data, skipping...")
            continue
        
        df_plot_hex = pd.DataFrame(plot_data_hex)
        
        # Get min and max distance
        vmin_dist = df_plot_hex['distance'].min()
        vmax_dist = df_plot_hex['distance'].max()
        
        # Create hexagon heatmap
        fig_hex, ax_hex = plt.subplots(figsize=(10, 8))
        
        # Plot non-source tiles in gray
        source_tile_coords = set(tile_min_distances.keys())
        other_tiles = []
        for idx, row_data in df_valid.iterrows():
            coord = (row_data['row'], row_data['col'])
            if coord not in source_tile_coords:
                other_tiles.append({'x': row_data['col'], 'y': row_data['row']})
        
        if len(other_tiles) > 0:
            df_other = pd.DataFrame(other_tiles)
            ax_hex.scatter(
                x=df_other['x'],
                y=df_other['y'],
                c='#D3D3D3',
                marker='h',
                s=marker_size,
                edgecolors='black',
                linewidth=0.3,
                alpha=0.5,
                zorder=1
            )
        
        # Plot source tiles with distance coloring
        scatter_hex = ax_hex.scatter(
            x=df_plot_hex['x'],
            y=df_plot_hex['y'],
            c=df_plot_hex['distance'],
            cmap='RdBu',
            marker='h',
            s=marker_size,
            edgecolors='black',
            linewidth=0.5,
            vmin=vmin_dist,
            vmax=vmax_dist,
            zorder=2
        )
        
        # Add colorbar
        cbar_hex = plt.colorbar(scatter_hex, ax=ax_hex, fraction=0.046, pad=0.04)
        cbar_hex.set_label('Minimum Distance', fontsize=28, fontweight='black', color='black')
        cbar_hex.ax.tick_params(labelsize=26, width=2, length=6, color='black', labelcolor='black')
        for label in cbar_hex.ax.get_yticklabels():
            label.set_fontsize(26)
            label.set_fontweight('black')
            label.set_color('black')
        plt.setp(cbar_hex.ax.get_yticklabels(), fontweight='black', fontsize=26, color='black')
        cbar_hex.outline.set_linewidth(2)
        cbar_hex.outline.set_edgecolor('black')
        
        # Invert y-axis
        ax_hex.invert_yaxis()
        
        # Set axis style
        for spine in ax_hex.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3)
            spine.set_color('black')
        
        # Remove labels and ticks
        ax_hex.set_xlabel('')
        ax_hex.set_ylabel('')
        ax_hex.set_xticks([])
        ax_hex.set_yticks([])
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(distance_subdirs[(source, target)], 
                                   f"{sample_name}_{source}_to_{target}_heatmap.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ {source} → {target}: saved")


def main():
    """Main execution"""
    print("="*80)
    print("Figure 6: TCGA COAD Hexagonal Visualization Generation")
    print("="*80)
    
    # Create output directories
    print("\nCreating output directories...")
    hexagon_subdirs, classif_subdirs, distance_subdirs, directional_pairs = create_output_directories()
    print(f"✅ Output directories created")
    print(f"  - Proportion heatmaps: {OUTPUT_HEXAGON_DIR}")
    print(f"  - Classification maps: {OUTPUT_CLASSIFICATION_DIR}")
    print(f"  - Distance heatmaps: {OUTPUT_DISTANCE_DIR}")
    
    # Get all grouped & normalized CSV files
    csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_grouped_normalized.csv")))
    print(f"\n✅ Found {len(csv_files)} CSV files to process\n")
    
    if len(csv_files) == 0:
        print(f"❌ No CSV files found in {INPUT_DIR}")
        print(f"Please run TCGA_COAD_Analysis.py first to generate input files.")
        return
    
    # Process each sample
    for i, csv_file in enumerate(csv_files, 1):
        sample_name = os.path.basename(csv_file).replace('_grouped_normalized.csv', '')
        
        print(f"\n{'='*80}")
        print(f"Processing {i}/{len(csv_files)}: {sample_name}")
        print(f"{'='*80}")
        
        try:
            # Load data
            df = pd.read_csv(csv_file)
            print(f"  Loaded {len(df)} tiles")
            
            # Generate all visualizations
            generate_proportion_heatmaps(df, sample_name, hexagon_subdirs)
            generate_classification_maps(df, sample_name, classif_subdirs)
            generate_distance_heatmaps(df, sample_name, distance_subdirs, directional_pairs)
            
            print(f"\n✅ Sample {sample_name} complete!")
            
        except Exception as e:
            print(f"❌ Error processing {sample_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"✅ All visualizations complete!")
    print(f"  - Proportion heatmaps: {len(CELL_TYPES_ALL)} cell types × {len(csv_files)} samples")
    print(f"  - Classification maps: {len(CELL_TYPES_DISTANCE)} cell types × {len(csv_files)} samples")
    print(f"  - Distance heatmaps: {len(directional_pairs)} pairs × {len(csv_files)} samples")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
