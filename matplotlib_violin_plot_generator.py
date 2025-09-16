#!/usr/bin/env python3
"""
Matplotlib Violin Plot Generator

Standalone script to generate violin plots showing gene expression across leiden clusters
with pre/post treatment comparison.

Usage:
    python matplotlib_violin_plot_generator.py --genes APOE C1QC HLA-DRA CD74 --cell_type "T cell"
    python matplotlib_violin_plot_generator.py --genes LAG3 CTLA4 EOMES --cluster "t_cluster"
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


class MatplotlibViolinPlotGenerator:
    def __init__(self, h5ad_path=None, umap_csv_path=None, annotation_csv_path=None):
        """
        Initialize the Matplotlib Violin Plot Generator
        """
        self.h5ad_path = h5ad_path or self.find_h5ad_file()
        self.umap_csv_path = umap_csv_path or "glioblastoma_umap_data.csv"
        self.annotation_csv_path = annotation_csv_path or "glioblastoma_cell_annotation.csv"
        self.adata = None
        self.umap_data = None
        self.annotation_data = None
        
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
        """Load all required data files and apply essential preprocessing steps"""
        print("📊 Loading data files...")
        
        # Load h5ad file with gene expression
        if not self.h5ad_path or not os.path.exists(self.h5ad_path):
            raise FileNotFoundError(f"H5AD file not found: {self.h5ad_path}")
        
        print(f"📁 Loading AnnData from: {self.h5ad_path}")
        self.adata = sc.read_h5ad(self.h5ad_path)
        print(f"✅ Loaded AnnData: {self.adata.shape[0]} cells, {self.adata.shape[1]} genes")
        
        # Apply essential preprocessing steps only
        print("🔄 Applying essential preprocessing steps...")
        
        # Filter cells and genes
        sc.pp.filter_cells(self.adata, min_genes=100)
        sc.pp.filter_genes(self.adata, min_cells=3)
        
        # Calculate mitochondrial gene metrics
        self.adata.var['mt'] = self.adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(self.adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        
        # Filter out high mitochondrial content cells
        self.adata = self.adata[self.adata.obs.pct_counts_mt < 20]
        
        # Remove mitochondrial genes
        non_mt_mask = ~self.adata.var['mt']
        self.adata = self.adata[:, non_mt_mask].copy()
        
        # Store raw counts
        self.adata.layers["counts"] = self.adata.X.copy()
        
        # Normalize and log-transform
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        
        # Store normalized data as raw
        self.adata.raw = self.adata
        
        print("✅ Applied essential preprocessing: filtering, normalization, and log1p transformation")
        
        # Load UMAP coordinates with leiden and treatment info
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
    
    def prepare_violin_data(self, genes, cell_type_filter=None, cluster_filter=None):
        """
        Prepare gene expression data for violin plots with leiden clusters and treatment groups.
        
        Expression values are log1p-transformed (log(count + 1)) from the preprocessing pipeline,
        providing normalized expression levels suitable for visualization.
        """
        print(f"🧬 Preparing violin plot data for genes: {genes}")
        
        # Get cell barcodes that are present in both datasets
        adata_barcodes = set(self.adata.obs.index)
        
        # Both files have barcodes, so merge them directly by barcode
        if self.annotation_data is not None and 'barcode' in self.annotation_data.columns and 'barcode' in self.umap_data.columns:
            print("📊 Merging annotation and UMAP data by barcode")
            
            # Merge the two dataframes on barcode to get all columns
            merged_csv = pd.merge(
                self.annotation_data, 
                self.umap_data[['barcode', 'Exp_sample_category', 'UMAP_1', 'UMAP_2']], 
                on='barcode', 
                how='inner'
            )
            
            print(f"📊 Merged data: {len(merged_csv)} cells with complete information")
            
            # Now find common barcodes with h5ad
            csv_barcodes = set(merged_csv['barcode'])
            common_barcodes = adata_barcodes.intersection(csv_barcodes)
            print(f"📊 Found {len(common_barcodes)} common cells between h5ad and merged CSV data")
            
            barcode_col = 'barcode'
            if len(common_barcodes) == 0:
                # Try alternative barcode matching strategies
                print("🔍 Trying alternative barcode matching...")
                
                sample_h5ad = list(adata_barcodes)[:5]
                sample_csv = list(csv_barcodes)[:5]
                print(f"H5AD sample barcodes: {sample_h5ad}")
                print(f"CSV sample barcodes: {sample_csv}")
                
                # Try removing suffixes from CSV barcodes
                csv_barcodes_clean = set()
                for barcode in csv_barcodes:
                    clean_barcode = barcode.split('-')[0]
                    csv_barcodes_clean.add(clean_barcode)
                
                common_barcodes_clean = adata_barcodes.intersection(csv_barcodes_clean)
                
                if len(common_barcodes_clean) > 0:
                    print(f"📊 Found {len(common_barcodes_clean)} matches after cleaning barcode suffixes")
                    merged_csv['barcode_clean'] = merged_csv['barcode'].str.split('-').str[0]
                    barcode_col = 'barcode_clean'
                    common_barcodes = common_barcodes_clean
                else:
                    raise ValueError("No common cells found between h5ad and merged CSV data")
        else:
            raise ValueError("Required data missing: need both annotation and UMAP data with barcode columns")
        
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
            # Get UMAP coordinates and cluster info from merged CSV
            csv_row = merged_csv[merged_csv[barcode_col] == barcode]
                
            if len(csv_row) == 0:
                continue
                
            csv_row = csv_row.iloc[0]
            
            # Extract treatment information from Exp_sample_category
            exp_sample = csv_row.get('Exp_sample_category', '')
            if '_pre' in str(exp_sample):
                treatment = 'pre'
            elif '_post' in str(exp_sample):
                treatment = 'post'
            else:
                treatment = 'unknown'
            
            # Get gene expression values
            h5ad_barcode = barcode
            
            # Get gene expression values including hierarchical_leiden
            cell_data = {
                'barcode': barcode,
                'leiden': str(csv_row.get('leiden', 'Unknown')),
                'hierarchical_leiden': str(csv_row.get('hierarchical_leiden', 'Unknown')),
                'treatment': treatment,
                'cell_type': csv_row.get('cell_type', 'Unknown'),
                'exp_sample_category': exp_sample
            }
            
            # Add expression values for each gene (using log-transformed data)
            for gene in available_genes:
                if h5ad_barcode in self.adata.obs.index:
                    # Get log-transformed expression value from main X matrix (after preprocessing)
                    gene_idx = self.adata.var_names.get_loc(gene)
                    expr_value = self.adata.X[self.adata.obs.index.get_loc(h5ad_barcode), gene_idx]
                    
                    # Convert sparse matrix element to scalar
                    if hasattr(expr_value, 'A1'):
                        expr_value = expr_value.A1[0]  # For sparse matrices
                    elif hasattr(expr_value, 'item'):
                        expr_value = expr_value.item()  # For numpy scalars
                    
                    # Expression value is now log1p-transformed (log(count + 1))
                    cell_data[f'{gene}_expression'] = float(expr_value)
                else:
                    cell_data[f'{gene}_expression'] = 0.0
            
            merged_data.append(cell_data)
        
        result_df = pd.DataFrame(merged_data)
        
        # Apply cluster filter if specified
        if cluster_filter:
            print(f"🔍 Applying cluster filter: '{cluster_filter}'")
            print(f"📊 Available hierarchical clusters in dataset: {sorted(result_df['hierarchical_leiden'].unique())}")
            
            # Filter cells that contain the cluster prefix (e.g., "t_cluster")
            cluster_match = result_df[result_df['hierarchical_leiden'].str.contains(cluster_filter, case=False, na=False)]
            
            if len(cluster_match) > 0:
                result_df = cluster_match
                print(f"✅ Filtered to {len(result_df)} cells matching cluster pattern: {cluster_filter}")
                
                # Extract cluster numbers from hierarchical_leiden for plotting
                # e.g., "t_cluster_2" -> "2"
                def extract_cluster_number(hierarchical_cluster):
                    # Find the last underscore and extract number after it
                    if '_' in hierarchical_cluster:
                        return hierarchical_cluster.split('_')[-1]
                    return hierarchical_cluster
                
                result_df['cluster_number'] = result_df['hierarchical_leiden'].apply(extract_cluster_number)
                print(f"📊 Extracted cluster numbers: {sorted(result_df['cluster_number'].unique())}")
                
                # Replace leiden column with cluster_number for plotting
                result_df['leiden'] = result_df['cluster_number']
            else:
                print(f"❌ No cells found matching cluster pattern '{cluster_filter}'")
                print(f"💡 Available cluster patterns:")
                unique_patterns = set()
                for cluster in result_df['hierarchical_leiden'].unique():
                    if '_' in str(cluster):
                        pattern = '_'.join(str(cluster).split('_')[:-1])
                        unique_patterns.add(pattern)
                for pattern in sorted(unique_patterns):
                    print(f"   - {pattern}")
        
        # Apply cell type filter if specified (keep original functionality)
        elif cell_type_filter:
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
        
        # Print summary of data structure
        print(f"📊 Final dataset: {len(result_df)} cells")
        print(f"📊 Leiden clusters: {sorted(result_df['leiden'].unique())}")
        print(f"📊 Treatment groups: {sorted(result_df['treatment'].unique())}")
        print(f"📊 Cells per treatment: {result_df['treatment'].value_counts().to_dict()}")
        
        return result_df, available_genes
    
    def create_violin_plot(self, data, gene, output_dir="violin_plots"):
        """
        Create violin plot for a single gene showing expression across leiden clusters
        with pre/post treatment comparison
        """
        expr_col = f'{gene}_expression'
        
        if expr_col not in data.columns:
            print(f"⚠️ No expression data for {gene}")
            return None
        
        # Filter out cells with unknown treatment
        plot_data = data[data['treatment'].isin(['pre', 'post'])].copy()
        
        if len(plot_data) == 0:
            print(f"⚠️ No pre/post treatment data available for {gene}")
            return None
        
        # Note: After log1p transformation, all values should be non-negative
        # No need to filter negative values, but we can still filter for quality
        if expr_col in plot_data.columns:
            # Log-transformed data should already be >= 0 after log1p(count)
            negative_count = (plot_data[expr_col] < 0).sum()
            if negative_count > 0:
                print(f"⚠️ Found {negative_count} unexpected negative values in log-transformed data")
                plot_data = plot_data[plot_data[expr_col] >= 0].copy()
        
        # Create figure (more square-shaped)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sort leiden clusters numerically for better visualization
        leiden_order = sorted(plot_data['leiden'].unique(), key=lambda x: int(x) if x.isdigit() else float('inf'))
        
        # Create violin plot with separate violins (not split)
        sns.violinplot(
            data=plot_data,
            x='leiden',
            y=expr_col,
            hue='treatment',
            split=False,
            inner='quartile',
            palette={'pre': '#FF6347', 'post': '#4682B4'},
            order=leiden_order,
            ax=ax
        )
        
        # Customize plot
        ax.set_title(f'{gene}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Leiden Cluster', fontsize=14)
        ax.set_ylabel('Expression Level', fontsize=14)
        
        # Customize legend
        legend = ax.get_legend()
        if legend:
            legend.set_title('Treatment')
            legend.get_title().set_fontweight('bold')
            for text in legend.get_texts():
                text.set_fontsize(11)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)
        
        # Rotate x-axis labels if there are many clusters
        if len(leiden_order) > 10:
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save plots
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as PNG
        png_path = os.path.join(output_dir, f"violin_plot_{gene}.png")
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📷 Violin plot saved as PNG: {png_path}")
        
        # Save as PDF
        pdf_path = os.path.join(output_dir, f"violin_plot_{gene}.pdf")
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f"📄 Violin plot saved as PDF: {pdf_path}")
        
        plt.close()
        
        return png_path, pdf_path
    
    def create_combined_violin_plot(self, data, genes, output_dir="violin_plots"):
        """
        Create a combined violin plot for multiple genes in subplots
        """
        n_genes = len(genes)
        
        # Calculate grid dimensions
        n_cols = min(2, n_genes)
        n_rows = int(np.ceil(n_genes / n_cols))
        
        # Create figure with subplots (more square-shaped)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 8 * n_rows))
        
        # Handle case where we only have one gene
        if n_genes == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        # Filter out cells with unknown treatment
        plot_data = data[data['treatment'].isin(['pre', 'post'])].copy()
        
        # Quality check for log-transformed expression values
        for gene in genes:
            expr_col = f'{gene}_expression'
            if expr_col in plot_data.columns:
                negative_count = (plot_data[expr_col] < 0).sum()
                if negative_count > 0:
                    print(f"⚠️ Found {negative_count} unexpected negative values for {gene} in log-transformed data")
                    plot_data = plot_data[plot_data[expr_col] >= 0].copy()
        
        leiden_order = sorted(plot_data['leiden'].unique(), key=lambda x: int(x) if x.isdigit() else float('inf'))
        
        for idx, gene in enumerate(genes):
            ax = axes[idx]
            expr_col = f'{gene}_expression'
            
            if expr_col in plot_data.columns:
                # Create violin plot with separate violins (not split)
                sns.violinplot(
                    data=plot_data,
                    x='leiden',
                    y=expr_col,
                    hue='treatment',
                    split=False,
                    inner='quartile',
                    palette={'pre': '#FF6347', 'post': '#4682B4'},
                    order=leiden_order,
                    ax=ax
                )
                
                # Customize subplot
                ax.set_title(f'{gene}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Leiden Cluster', fontsize=12)
                ax.set_ylabel('Expression Level', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_axisbelow(True)
                
                # Set y-axis to start from 0
                ax.set_ylim(bottom=0)
                
                # Handle legend (only show on first subplot)
                if idx == 0:
                    legend = ax.get_legend()
                    if legend:
                        legend.set_title('Treatment')
                        legend.get_title().set_fontweight('bold')
                else:
                    ax.get_legend().remove() if ax.get_legend() else None
                
                # Rotate x-axis labels if there are many clusters
                if len(leiden_order) > 8:
                    ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, f'No data for {gene}', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(gene, fontsize=14)
        
        # Hide empty subplots
        for idx in range(n_genes, len(axes)):
            axes[idx].set_visible(False)
        
        # Add overall title
        fig.suptitle('Gene Expression', fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save plots
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as PNG
        png_path = os.path.join(output_dir, f"violin_plots_combined_{'_'.join(genes[:3])}.png")
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📷 Combined violin plots saved as PNG: {png_path}")
        
        # Save as PDF
        pdf_path = os.path.join(output_dir, f"violin_plots_combined_{'_'.join(genes[:3])}.pdf")
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f"📄 Combined violin plots saved as PDF: {pdf_path}")
        
        plt.close()
        
        return png_path, pdf_path
    
    def generate_violin_plots(self, genes, cell_type_filter=None, cluster_filter=None, 
                            output_dir="violin_plots", plot_type="both"):
        """
        Main function to generate violin plots
        """
        try:
            # Load data if not already loaded
            if self.adata is None:
                self.load_data()
            
            # Prepare violin plot data
            violin_data, available_genes = self.prepare_violin_data(genes, cell_type_filter, cluster_filter)
            
            if len(available_genes) == 0:
                return {"error": "No genes available for plotting"}
            
            results = {
                "genes": available_genes,
                "cell_count": len(violin_data),
                "saved_files": []
            }
            
            # Generate individual violin plots
            if plot_type in ["individual", "both"]:
                print("🎻 Creating individual violin plots...")
                for gene in available_genes:
                    files = self.create_violin_plot(violin_data, gene, output_dir)
                    if files:
                        results["saved_files"].extend(files)
            
            # Generate combined violin plot
            if plot_type in ["combined", "both"] and len(available_genes) > 1:
                print("🎻 Creating combined violin plot...")
                files = self.create_combined_violin_plot(violin_data, available_genes, output_dir)
                if files:
                    results["saved_files"].extend(files)
            
            print(f"✅ Violin plot generation complete: {len(results['saved_files'])} files saved")
            return results
            
        except Exception as e:
            error_msg = f"Error generating violin plots: {e}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}


def main():
    """Command line interface for violin plot generation"""
    parser = argparse.ArgumentParser(description="Generate violin plots showing gene expression across leiden clusters with pre/post treatment comparison")
    parser.add_argument("--genes", nargs="+", required=True, help="Gene names to plot")
    parser.add_argument("--cell_type", type=str, help="Cell type to filter for")
    parser.add_argument("--cluster", type=str, help="Cluster pattern to filter for (e.g., 't_cluster')")
    parser.add_argument("--plot_type", choices=["individual", "combined", "both"], default="both",
                       help="Type of plots to generate")
    parser.add_argument("--output_dir", default="violin_plots", help="Output directory")
    parser.add_argument("--h5ad_path", help="Path to h5ad file")
    parser.add_argument("--umap_csv", help="Path to UMAP CSV file")
    parser.add_argument("--annotation_csv", help="Path to annotation CSV file")
    
    args = parser.parse_args()
    
    print(f"🎻 Generating violin plots for genes: {args.genes}")
    
    # Create generator
    generator = MatplotlibViolinPlotGenerator(
        h5ad_path=args.h5ad_path,
        umap_csv_path=args.umap_csv,
        annotation_csv_path=args.annotation_csv
    )
    
    # Generate plots
    results = generator.generate_violin_plots(
        genes=args.genes,
        cell_type_filter=args.cell_type,
        cluster_filter=args.cluster,
        output_dir=args.output_dir,
        plot_type=args.plot_type
    )
    
    if "error" in results:
        print(f"❌ {results['error']}")
        return 1
    
    print(f"✅ Successfully generated violin plots for {len(results['genes'])} genes")
    print(f"📊 Processed {results['cell_count']} cells")
    print(f"📁 Files saved to: {args.output_dir}")
    print(f"💾 Total files created: {len(results['saved_files'])}")
    
    return 0


if __name__ == "__main__":
    exit(main())