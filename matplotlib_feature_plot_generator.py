#!/usr/bin/env python3
"""
Matplotlib Feature Plot Generator

Standalone script to generate feature plots using matplotlib with Seurat-like styling.
Generates high-quality feature plots similar to Seurat's FeaturePlot function.

Usage:
    python matplotlib_feature_plot_generator.py --genes APOE C1QC HLA-DRA CD74 ISG20 GZMK IL32
    python matplotlib_feature_plot_generator.py --genes APOE --cell_type "T cell"
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import scanpy as sc
from pathlib import Path

# Add scchatbot to Python path for importing functions
sys.path.append(os.path.join(os.path.dirname(__file__), 'scchatbot'))

try:
    from scchatbot.cell_types.utils import get_h5ad
    from scchatbot.cell_types.standardization import unified_cell_type_handler
except ImportError as e:
    print(f"Warning: Could not import scchatbot modules: {e}")
    print("Some functionality may be limited")


class MatplotlibFeaturePlotGenerator:
    def __init__(self, h5ad_path=None, umap_csv_path=None, annotation_csv_path=None):
        """
        Initialize the Matplotlib Feature Plot Generator
        """
        self.h5ad_path = h5ad_path or self.find_h5ad_file()
        self.umap_csv_path = umap_csv_path or "glioblastoma_umap_data.csv"
        self.annotation_csv_path = annotation_csv_path or "glioblastoma_cell_annotation.csv"
        self.adata = None
        self.umap_data = None
        self.annotation_data = None
        
        # Seurat-like color palette
        self.seurat_colors = {
            'background': '#E6E6E6',  # Light gray for low/zero expression
            'low': '#FEE5D9',         # Light peach
            'mid_low': '#FCAE91',     # Light orange
            'mid': '#FB6A4A',         # Orange-red
            'mid_high': '#DE2D26',    # Red
            'high': '#A50F15'         # Dark red
        }
        
    def find_h5ad_file(self):
        """Find h5ad file in the project directory"""
        try:
            h5ad_file = get_h5ad("media", ".h5ad")
            if h5ad_file and os.path.exists(h5ad_file):
                return h5ad_file
        except:
            pass
            
        # Fallback search
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(".h5ad"):
                    return os.path.join(root, file)
        return None
    
    def load_data(self):
        """Load all required data files"""
        print("📊 Loading data files...")
        
        # Load h5ad file with gene expression
        if not self.h5ad_path or not os.path.exists(self.h5ad_path):
            raise FileNotFoundError(f"H5AD file not found: {self.h5ad_path}")
        
        print(f"📁 Loading AnnData from: {self.h5ad_path}")
        self.adata = sc.read_h5ad(self.h5ad_path)
        print(f"✅ Loaded AnnData: {self.adata.shape[0]} cells, {self.adata.shape[1]} genes")
        
        # Apply preprocessing steps to ensure proper expression values
        self.preprocess_data()
        
        # Load UMAP coordinates
        if not os.path.exists(self.umap_csv_path):
            raise FileNotFoundError(f"UMAP CSV not found: {self.umap_csv_path}")
        
        print(f"📁 Loading UMAP data from: {self.umap_csv_path}")
        self.umap_data = pd.read_csv(self.umap_csv_path)
        print(f"✅ Loaded UMAP data: {len(self.umap_data)} cells")
        
        # Load cell annotations (optional)
        if os.path.exists(self.annotation_csv_path):
            print(f"📁 Loading annotations from: {self.annotation_csv_path}")
            self.annotation_data = pd.read_csv(self.annotation_csv_path)
            print(f"✅ Loaded annotations: {len(self.annotation_data)} cells")
        else:
            print(f"⚠️ Annotation file not found: {self.annotation_csv_path}")
    
    def preprocess_data(self):
        """
        Apply standard preprocessing steps to ensure proper expression values for plotting
        """
        print("🔬 Applying preprocessing steps...")
        
        # Check if data is already preprocessed by looking for common indicators
        has_raw = self.adata.raw is not None
        has_counts_layer = 'counts' in self.adata.layers if hasattr(self.adata, 'layers') else False
        
        # If data appears to be already preprocessed, skip preprocessing
        if has_raw or has_counts_layer:
            print("✅ Data appears to be already preprocessed, using existing processed values")
            return
        
        print("📊 Data needs preprocessing, applying standard steps...")
        
        # Filter cells and genes (basic quality control)
        print("🔍 Filtering cells (min 100 genes) and genes (min 3 cells)...")
        sc.pp.filter_cells(self.adata, min_genes=100)
        sc.pp.filter_genes(self.adata, min_cells=3)
        
        # Calculate mitochondrial gene percentage
        print("🧬 Calculating mitochondrial gene metrics...")
        self.adata.var['mt'] = self.adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(self.adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        
        # Filter cells with high mitochondrial percentage (>20%)
        print("🔍 Filtering cells with high mitochondrial content (>20%)...")
        initial_cells = self.adata.shape[0]
        self.adata = self.adata[self.adata.obs.pct_counts_mt < 20]
        print(f"📊 Filtered from {initial_cells} to {self.adata.shape[0]} cells")
        
        # Remove mitochondrial genes for downstream analysis
        print("🗑️ Removing mitochondrial genes...")
        non_mt_mask = ~self.adata.var['mt']
        self.adata = self.adata[:, non_mt_mask].copy()
        
        # Store raw counts before normalization
        print("💾 Storing raw counts...")
        self.adata.layers["counts"] = self.adata.X.copy()
        
        # Normalize to 10,000 reads per cell
        print("⚖️ Normalizing total counts to 10,000 per cell...")
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        
        # Log transform (log1p)
        print("📈 Applying log(1+x) transformation...")
        sc.pp.log1p(self.adata)
        
        # Ensure no negative values after log transformation (due to numerical precision)
        if hasattr(self.adata.X, 'min'):
            min_val = self.adata.X.min()
            if min_val < 0:
                print(f"⚠️ Found negative values after log transformation (min: {min_val}), clipping to 0")
                self.adata.X = np.clip(self.adata.X, 0, None)
        
        # Store processed data in raw
        print("💾 Storing processed data...")
        self.adata.raw = self.adata
        
        print(f"✅ Preprocessing complete: {self.adata.shape[0]} cells, {self.adata.shape[1]} genes")
    
    def prepare_expression_data(self, genes, cell_type_filter=None):
        """
        Prepare gene expression data matched with UMAP coordinates
        """
        print(f"🧬 Preparing expression data for genes: {genes}")
        
        # Get cell barcodes that are present in both datasets
        adata_barcodes = set(self.adata.obs.index)
        
        # The UMAP CSV doesn't have barcodes, but the annotation CSV does
        # We need to merge them using row index matching
        if self.annotation_data is not None and 'barcode' in self.annotation_data.columns:
            # Check if we can match by row order (both CSVs should have same cell order)
            if len(self.umap_data) == len(self.annotation_data):
                print("📊 Matching UMAP and annotation data by row order")
                
                # Debug: Check individual CSV structures
                print(f"🔍 Debug: UMAP data shape: {self.umap_data.shape}, columns: {list(self.umap_data.columns)}")
                print(f"🔍 Debug: Annotation data shape: {self.annotation_data.shape}, columns: {list(self.annotation_data.columns)}")
                
                # Check if UMAP CSV already has barcodes (if so, use annotation data as primary)
                if 'barcode' in self.umap_data.columns:
                    print("🔍 UMAP data already contains barcodes, using it directly")
                    merged_csv = self.umap_data.copy()
                    # Add cell_type from annotation data if available
                    if 'cell_type' in self.annotation_data.columns:
                        merged_csv['cell_type'] = self.annotation_data['cell_type']
                else:
                    print("🔍 Merging UMAP and annotation data by row order")
                    # Merge UMAP and annotation data by index - ensure we get all columns including cell_type
                    merged_csv = pd.concat([
                        self.umap_data.reset_index(drop=True), 
                        self.annotation_data.reset_index(drop=True)
                    ], axis=1)
                
                # Debug: Check the merged CSV structure
                print(f"🔍 Debug: Merged CSV shape: {merged_csv.shape}")
                print(f"🔍 Debug: Merged CSV columns: {list(merged_csv.columns)}")
                print(f"🔍 Debug: First 3 barcode values: {merged_csv['barcode'].head(3).tolist()}")
                
                # Now find common barcodes with h5ad
                csv_barcodes = set(merged_csv['barcode'].dropna())
                print(f"🔍 Debug: Total CSV barcodes found: {len(csv_barcodes)}")
                common_barcodes = adata_barcodes.intersection(csv_barcodes)
                print(f"📊 Found {len(common_barcodes)} common cells between h5ad and CSV data")
                
                if len(common_barcodes) == 0:
                    # Try alternative barcode matching strategies
                    print("🔍 Trying alternative barcode matching...")
                    
                    # Check if h5ad barcodes need suffix/prefix modification
                    sample_h5ad = list(adata_barcodes)[:5]
                    sample_csv = list(csv_barcodes)[:5]
                    print(f"H5AD sample barcodes: {sample_h5ad}")
                    print(f"CSV sample barcodes: {sample_csv}")
                    
                    # Try removing suffixes from CSV barcodes
                    csv_barcodes_clean = set()
                    for barcode in csv_barcodes:
                        if pd.notna(barcode):  # Skip NaN values
                            # Remove common suffixes like "-1-0", "-1", etc.
                            clean_barcode = str(barcode).split('-')[0]
                            csv_barcodes_clean.add(clean_barcode)
                    
                    common_barcodes_clean = adata_barcodes.intersection(csv_barcodes_clean)
                    
                    if len(common_barcodes_clean) > 0:
                        print(f"📊 Found {len(common_barcodes_clean)} matches after cleaning barcode suffixes")
                        # Update the merged CSV with clean barcodes for matching
                        merged_csv['barcode_clean'] = merged_csv['barcode'].astype(str).str.split('-').str[0]
                        barcode_col = 'barcode_clean'
                        common_barcodes = common_barcodes_clean
                    else:
                        raise ValueError("No common cells found between h5ad and UMAP data after trying various matching strategies")
                else:
                    barcode_col = 'barcode'
            else:
                raise ValueError(f"UMAP data ({len(self.umap_data)} cells) and annotation data ({len(self.annotation_data)} cells) have different lengths")
        else:
            raise ValueError("Cannot match datasets: annotation data missing or no barcode column")
        
        # Filter genes that exist in the dataset
        available_genes = []
        missing_genes = []
        
        for gene in genes:
            if gene in self.adata.var_names:
                available_genes.append(gene)
            else:
                missing_genes.append(gene)
        
        if missing_genes:
            print(f"⚠️ Genes not found in dataset: {missing_genes}")
        
        if not available_genes:
            raise ValueError(f"None of the requested genes found in dataset: {genes}")
        
        print(f"✅ Using genes: {available_genes}")
        
        # Create merged dataset
        merged_data = []
        
        for barcode in common_barcodes:
            # Get UMAP coordinates from merged CSV
            csv_row = merged_csv[merged_csv[barcode_col] == barcode]
                
            if len(csv_row) == 0:
                continue
                
            csv_row = csv_row.iloc[0]
            
            # Get gene expression values - need to map barcode for h5ad lookup
            h5ad_barcode = barcode
            
            # Get gene expression values
            cell_data = {
                'barcode': barcode,
                'UMAP_1': csv_row['UMAP_1'],
                'UMAP_2': csv_row['UMAP_2'],
                'cell_type': csv_row.get('cell_type', 'Unknown')
            }
            
            # Add expression values for each gene
            for gene in available_genes:
                if h5ad_barcode in self.adata.obs.index:
                    # Get expression value (handle both sparse and dense matrices)
                    gene_idx = self.adata.var_names.get_loc(gene)
                    expr_value = self.adata.X[self.adata.obs.index.get_loc(h5ad_barcode), gene_idx]
                    
                    # Convert sparse matrix element to scalar
                    if hasattr(expr_value, 'A1'):
                        expr_value = expr_value.A1[0]  # For sparse matrices
                    elif hasattr(expr_value, 'item'):
                        expr_value = expr_value.item()  # For numpy scalars
                    
                    cell_data[f'{gene}_expression'] = float(expr_value)
                else:
                    cell_data[f'{gene}_expression'] = 0.0
            
            merged_data.append(cell_data)
        
        result_df = pd.DataFrame(merged_data)
        
        # Apply cell type filter if specified
        if cell_type_filter:
            print(f"🔍 Applying cell type filter: '{cell_type_filter}'")
            print(f"📊 Available cell types in dataset: {sorted(result_df['cell_type'].unique())}")
            
            # Try multiple matching strategies
            original_count = len(result_df)
            
            # Strategy 1: Exact match
            exact_match = result_df[result_df['cell_type'] == cell_type_filter]
            
            # Strategy 2: Case-insensitive exact match
            case_insensitive_match = result_df[result_df['cell_type'].str.lower() == cell_type_filter.lower()]
            
            # Strategy 3: Contains match
            contains_match = result_df[result_df['cell_type'].str.contains(cell_type_filter, case=False, na=False)]
            
            # Strategy 4: Try unified cell type handler if available
            standardized_filter = None
            if 'unified_cell_type_handler' in globals():
                try:
                    standardized_filter = unified_cell_type_handler(cell_type_filter)
                    std_match = result_df[result_df['cell_type'].str.contains(standardized_filter, case=False, na=False)]
                    print(f"📊 Standardized cell type: '{standardized_filter}'")
                except:
                    std_match = pd.DataFrame()
            else:
                std_match = pd.DataFrame()
            
            # Use the best match (prefer exact > case-insensitive > contains > standardized)
            if len(exact_match) > 0:
                result_df = exact_match
                print(f"✅ Used exact match")
            elif len(case_insensitive_match) > 0:
                result_df = case_insensitive_match
                print(f"✅ Used case-insensitive match")
            elif len(contains_match) > 0:
                result_df = contains_match
                print(f"✅ Used contains match")
            elif len(std_match) > 0:
                result_df = std_match
                print(f"✅ Used standardized match")
            else:
                print(f"❌ No cells found matching '{cell_type_filter}'")
                print(f"💡 Try one of these available cell types:")
                for ct in sorted(result_df['cell_type'].unique())[:10]:  # Show first 10
                    print(f"   - {ct}")
                if len(result_df['cell_type'].unique()) > 10:
                    print(f"   ... and {len(result_df['cell_type'].unique()) - 10} more")
            
            print(f"🔍 Filtered to {len(result_df)} cells matching cell type: {cell_type_filter}")
        
        print(f"📊 Final dataset: {len(result_df)} cells with expression data")
        return result_df, available_genes
    
    def create_seurat_colormap(self):
        """Create a Seurat-like colormap"""
        colors = ['#E6E6E6', '#FEE5D9', '#FCAE91', '#FB6A4A', '#DE2D26', '#A50F15']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('seurat', colors, N=n_bins)
        return cmap
    
    def plot_single_feature(self, ax, expression_data, gene, point_size=1.0, alpha=0.8):
        """
        Plot a single feature on the given axis with Seurat-like styling
        """
        expr_col = f'{gene}_expression'
        
        if expr_col not in expression_data.columns:
            ax.text(0.5, 0.5, f'No data for {gene}', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            return
        
        # Get expression values
        expr_values = expression_data[expr_col]
        x_coords = expression_data['UMAP_1']
        y_coords = expression_data['UMAP_2']
        
        # Ensure expression values are non-negative (clip any negative values to 0)
        expr_values = expr_values.clip(lower=0)
        if expr_values.min() < 0:
            print(f"⚠️ Clipped {(expression_data[expr_col] < 0).sum()} negative values to 0 for {gene}")
        
        # Create colormap
        cmap = self.create_seurat_colormap()
        
        # Handle cases where all expression is zero
        if expr_values.max() == 0:
            print(f"⚠️ No expression detected for {gene} in filtered cells")
            colors = ['#E6E6E6'] * len(expr_values)
            scatter = ax.scatter(x_coords, y_coords, c=colors, s=point_size, alpha=alpha, 
                               edgecolors='none', rasterized=True)
        else:
            # Sort points so high expression is plotted on top
            sort_idx = np.argsort(expr_values)
            x_sorted = x_coords.iloc[sort_idx]
            y_sorted = y_coords.iloc[sort_idx]
            expr_sorted = expr_values.iloc[sort_idx]
            
            # Create scatter plot with proper scaling
            vmax_val = max(expr_values.quantile(0.95), expr_values.max() * 0.1)  # Ensure reasonable vmax
            scatter = ax.scatter(x_sorted, y_sorted, c=expr_sorted, s=point_size, alpha=alpha,
                               cmap=cmap, edgecolors='none', rasterized=True,
                               vmin=0, vmax=vmax_val)  # Always use 0 as minimum
        
        # Set title and labels
        ax.set_title(gene, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('UMAP_1', fontsize=12)
        ax.set_ylabel('UMAP_2', fontsize=12)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set aspect ratio to be equal
        ax.set_aspect('equal', adjustable='box')
        
        return scatter
    
    def create_feature_plots(self, genes, expression_data, output_dir="feature_plot", 
                           figsize_per_plot=(4, 4), point_size=1.0, alpha=0.8):
        """
        Create feature plots for multiple genes with Seurat-like styling
        """
        n_genes = len(genes)
        
        # Calculate grid dimensions (similar to your example: 4 columns)
        n_cols = min(4, n_genes)
        n_rows = int(np.ceil(n_genes / n_cols))
        
        # Calculate figure size
        fig_width = n_cols * figsize_per_plot[0]
        fig_height = n_rows * figsize_per_plot[1]
        
        print(f"🎨 Creating feature plots: {n_genes} genes in {n_rows}x{n_cols} grid")
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        
        # Handle case where we only have one row or one column
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot each gene
        scatters = []
        for idx, gene in enumerate(genes):
            row = idx // n_cols
            col = idx % n_cols
            
            if n_rows == 1:
                ax = axes[col] if n_cols > 1 else axes[0]
            else:
                ax = axes[row, col]
            
            scatter = self.plot_single_feature(ax, expression_data, gene, point_size, alpha)
            scatters.append(scatter)
        
        # Hide empty subplots
        for idx in range(n_genes, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            if n_rows == 1:
                ax = axes[col] if n_cols > 1 else axes[0]
            else:
                ax = axes[row, col]
            ax.set_visible(False)
        
        # Adjust layout
        plt.tight_layout(pad=2.0)
        
        # Save plots
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as PNG (high resolution)
        png_path = os.path.join(output_dir, f"feature_plots_{'_'.join(genes[:4])}.png")
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='png')
        print(f"📷 Feature plots saved as PNG: {png_path}")
        
        # Save as PDF (vector format)
        pdf_path = os.path.join(output_dir, f"feature_plots_{'_'.join(genes[:4])}.pdf")
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='pdf')
        print(f"📄 Feature plots saved as PDF: {pdf_path}")
        
        plt.show()
        plt.close()
        
        return png_path, pdf_path
    
    def create_individual_plots(self, genes, expression_data, output_dir="feature_plot",
                              figsize=(6, 5), point_size=1.5, alpha=0.8):
        """
        Create individual feature plots for each gene
        """
        print(f"🎨 Creating individual feature plots for {len(genes)} genes")
        
        saved_files = []
        
        for gene in genes:
            # Create figure for individual gene
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            
            # Plot the feature
            scatter = self.plot_single_feature(ax, expression_data, gene, point_size, alpha)
            
            # Add colorbar for individual plots
            expr_col = f'{gene}_expression'
            if expr_col in expression_data.columns and expression_data[expr_col].max() > 0:
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
                cbar.set_label('Expression', rotation=270, labelpad=15)
            
            plt.tight_layout()
            
            # Save individual plot
            os.makedirs(output_dir, exist_ok=True)
            
            png_path = os.path.join(output_dir, f"feature_plot_{gene}.png")
            plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', 
                       edgecolor='none', format='png')
            
            pdf_path = os.path.join(output_dir, f"feature_plot_{gene}.pdf")
            plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', 
                       edgecolor='none', format='pdf')
            
            saved_files.extend([png_path, pdf_path])
            print(f"💾 Individual plot saved: {gene}")
            
            plt.close()
        
        return saved_files
    
    def generate_feature_plots(self, genes, cell_type_filter=None, output_dir="feature_plot",
                             plot_type="both", point_size=1.0, alpha=0.8):
        """
        Main function to generate feature plots using matplotlib
        """
        try:
            # Load data if not already loaded
            if self.adata is None:
                self.load_data()
            
            # Prepare expression data
            expression_data, available_genes = self.prepare_expression_data(genes, cell_type_filter)
            
            if len(available_genes) == 0:
                return {"error": "No genes available for plotting"}
            
            results = {
                "genes": available_genes,
                "cell_count": len(expression_data),
                "saved_files": []
            }
            
            # Generate combined plot
            if plot_type in ["combined", "both"]:
                print("🎨 Creating combined feature plot...")
                png_path, pdf_path = self.create_feature_plots(
                    available_genes, expression_data, output_dir, 
                    point_size=point_size, alpha=alpha
                )
                results["saved_files"].extend([png_path, pdf_path])
            
            # Generate individual plots
            if plot_type in ["individual", "both"]:
                print("🎨 Creating individual feature plots...")
                individual_files = self.create_individual_plots(
                    available_genes, expression_data, output_dir,
                    point_size=point_size, alpha=alpha
                )
                results["saved_files"].extend(individual_files)
            
            print(f"✅ Feature plot generation complete: {len(results['saved_files'])} files saved")
            return results
            
        except Exception as e:
            error_msg = f"Error generating feature plots: {e}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}


def main():
    """Command line interface for feature plot generation"""
    parser = argparse.ArgumentParser(description="Generate feature plots using matplotlib with Seurat-like styling")
    parser.add_argument("--genes", nargs="+", required=True, help="Gene names to plot")
    parser.add_argument("--cell_type", type=str, help="Cell type to filter for")
    parser.add_argument("--plot_type", choices=["individual", "combined", "both"], default="both",
                       help="Type of plots to generate")
    parser.add_argument("--point_size", type=float, default=1.0, help="Size of points in scatter plot")
    parser.add_argument("--alpha", type=float, default=0.8, help="Transparency of points")
    parser.add_argument("--output_dir", default="feature_plot", help="Output directory")
    parser.add_argument("--h5ad_path", help="Path to h5ad file")
    parser.add_argument("--umap_csv", help="Path to UMAP CSV file")
    parser.add_argument("--annotation_csv", help="Path to annotation CSV file")
    
    args = parser.parse_args()
    
    print(f"🧬 Generating matplotlib feature plots for genes: {args.genes}")
    
    # Create generator
    generator = MatplotlibFeaturePlotGenerator(
        h5ad_path=args.h5ad_path,
        umap_csv_path=args.umap_csv,
        annotation_csv_path=args.annotation_csv
    )
    
    # Generate plots
    results = generator.generate_feature_plots(
        genes=args.genes,
        cell_type_filter=args.cell_type,
        output_dir=args.output_dir,
        plot_type=args.plot_type,
        point_size=args.point_size,
        alpha=args.alpha
    )
    
    if "error" in results:
        print(f"❌ {results['error']}")
        return 1
    
    print(f"✅ Successfully generated plots for {len(results['genes'])} genes")
    print(f"📊 Processed {results['cell_count']} cells")
    print(f"📁 Files saved to: {args.output_dir}")
    print(f"💾 Total files created: {len(results['saved_files'])}")
    
    return 0


if __name__ == "__main__":
    exit(main())