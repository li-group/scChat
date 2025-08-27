import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from ..cell_types.standardization import unified_cell_type_handler
import plotly.io as pio
import os
import glob


def _find_enrichment_file(analysis: str, cell_type: str, condition: str = None) -> str:
    """
    Unified file discovery function for enrichment analyses.
    Used by all enrichment visualization functions.
    """
    base = analysis.lower()
    
    # If specific condition is provided, use that folder
    if condition:
        folder = f"scchatbot/enrichment/{condition}"
        fname = f"results_summary_{cell_type}.csv"
    else:
        # Use default folder structure based on analysis type
        if base == "go":
            # For GO, we need to handle domain-specific subfolders
            # This is a simplified version - the calling function should handle domain logic
            folder = f"scchatbot/enrichment/go_bp"  # Default to BP
            fname = f"results_summary_{cell_type}.csv"
        elif base.startswith("go_"):
            folder = f"scchatbot/enrichment/{base}"
            fname = f"results_summary_{cell_type}.csv"
        else:
            folder = f"scchatbot/enrichment/{base}"
            fname = f"results_summary_{cell_type}.csv"
    
    path = os.path.join(folder, fname)
    
    # Check if file exists, if not try alternative naming
    if not os.path.exists(path):
        fname_alt = f"{base}_results_summary_{cell_type}.csv"
        path_alt = os.path.join(folder, fname_alt)
        if os.path.exists(path_alt):
            return path_alt
    
    # If still not found for GSEA, try glob patterns
    if base == "gsea" and not os.path.exists(path):
        search_patterns = [
            f"scchatbot/enrichment/gsea*/results_summary_{cell_type}.csv",
            f"scchatbot/enrichment/gsea/results_summary_*.csv",
            f"scchatbot/enrichment/gsea*/results_summary_*.csv"
        ]
        
        for pattern in search_patterns:
            found_files = glob.glob(pattern)
            if found_files:
                return found_files[0]
    
    return path if os.path.exists(path) else None


def display_enrichment_visualization(
    analysis: str,
    cell_type: str,
    plot_type: str = "both",  # "bar", "dot", or "both"
    top_n: int = 10,
    domain: str = None,
    condition: str = None
) -> str:
    """
    Unified enrichment visualization function that can generate bar plots, dot plots, or both.
    
    Args:
        analysis: "reactome", "kegg", "gsea", or "go"
        cell_type: The cell type to analyze
        plot_type: "bar", "dot", or "both" 
        top_n: Number of top terms to display
        domain: Required if analysis=="go" (one of "BP","MF","CC")
        condition: Optional specific condition folder
    
    Returns:
        HTML string containing the plot(s)
    """
    
    # Handle GO domain logic
    if analysis.lower() == "go":
        if domain is None:
            return "Error: `domain` must be provided for GO analysis (BP, MF, or CC)."
        analysis = f"go_{domain.lower()}"
    
    # Find the data file
    file_path = _find_enrichment_file(analysis, cell_type, condition)
    
    if not file_path:
        return f"Enrichment data not available for {analysis} analysis of {cell_type}. Please run the enrichment analysis first."
    
    try:
        # Load and process data
        df = pd.read_csv(file_path)
        df["minus_log10_p"] = -np.log10(df["p_value"])
        top = df.nsmallest(top_n, "p_value")
        
        plots_html = []
        
        # Generate bar plot if requested
        if plot_type in ["bar", "both"]:
            print(f"📊 Generating bar plot with {len(top)} terms")
            bar_fig = px.bar(
                top,
                x="gene_ratio",
                y="Term",
                orientation="h",
                title=f"Top {top_n} {analysis.upper()} Terms - Bar Plot ({cell_type})",
                labels={"gene_ratio": "Gene Ratio", "Term": "Term"},
            )
            bar_fig.update_layout(
                height=top_n * 40 + 200,
                yaxis={"categoryorder": "total ascending"},
                margin=dict(l=100, r=50, t=50, b=50),  # Add margins to prevent cropping
                autosize=True
            )
            bar_html = pio.to_html(bar_fig, full_html=False, include_plotlyjs="cdn")
            plots_html.append(bar_html)
            print(f"✅ Bar plot HTML generated: {len(bar_html)} characters")
        
        # Generate dot plot if requested
        if plot_type in ["dot", "both"]:
            print(f"🔴 Generating dot plot with {len(top)} terms")
            print(f"📊 Data sample: gene_ratio range: {top['gene_ratio'].min():.3f}-{top['gene_ratio'].max():.3f}")
            dot_fig = px.scatter(
                top,
                x="gene_ratio",
                y="Term",
                size="intersection_size",
                color="minus_log10_p",
                title=f"Top {top_n} {analysis.upper()} Terms - Dot Plot ({cell_type})",
                labels={
                    "gene_ratio": "Gene Ratio",
                    "minus_log10_p": "-log10(p-value)",
                    "Term": "Term",
                    "intersection_size": "Gene Count"
                },
                size_max=20,
                orientation="h"
            )
            # Ensure points are visible by setting minimum sizes and proper ranges
            dot_fig.update_traces(
                marker=dict(
                    sizemin=4,  # Minimum point size
                    sizeref=2. * max(top['intersection_size']) / (15 ** 2),  # Scale factor
                    sizemode='area'
                )
            )
            dot_fig.update_layout(
                height=top_n * 40 + 200,
                yaxis={"categoryorder": "total ascending"},
                margin=dict(l=100, r=50, t=50, b=50),  # Add margins to prevent cropping
                autosize=True
            )
            dot_html = pio.to_html(dot_fig, full_html=False, include_plotlyjs="cdn")
            plots_html.append(dot_html)
            print(f"✅ Dot plot HTML generated: {len(dot_html)} characters")
        
        # Handle multiple plots if both requested
        if plot_type == "both":
            print(f"🔗 Creating separate plot objects for {len(plots_html)} plots")
            print(f"📊 Bar plot size: {len(plots_html[0])} chars")
            print(f"🔴 Dot plot size: {len(plots_html[1])} chars") 
            
            # Return multiple plots structure for backend processing
            multiple_plots_result = {
                "multiple_plots": True,
                "plots": [
                    {
                        "type": "bar",
                        "title": f"Top {top_n} {analysis.upper()} Terms - Bar Plot ({cell_type})",
                        "html": plots_html[0]
                    },
                    {
                        "type": "dot", 
                        "title": f"Top {top_n} {analysis.upper()} Terms - Dot Plot ({cell_type})",
                        "html": plots_html[1]
                    }
                ]
            }
            
            # Add legacy combined HTML for backward compatibility
            combined_html = f"""
            <div style="margin-bottom: 30px;">
                <h3>Bar Plot</h3>
                {plots_html[0]}
            </div>
            <div>
                <h3>Dot Plot</h3>
                {plots_html[1]}
            </div>
            """
            multiple_plots_result["legacy_combined_html"] = combined_html
            
            print(f"🎯 Created multiple plots structure with {len(multiple_plots_result['plots'])} individual plots")
            return multiple_plots_result
        else:
            return plots_html[0]
            
    except Exception as e:
        return f"Error generating {analysis} {plot_type} plot: {e}"






def display_dotplot(cell_type: str = "Overall cells") -> str:
    """
    Creates a dotplot visualization from CSV data and returns an HTML snippet
    containing the Plotly figure. Uses dynamic file discovery instead of hardcoded paths.
    """
    import os
    import glob
    
    # Use dynamic file discovery
    cell_type_formatted = unified_cell_type_handler(cell_type)
    
    # Try multiple possible locations for dot plot data
    possible_paths = [
        f"scchatbot/runtime_data/basic_data/{cell_type_formatted}_dot_plot_data.csv",
        f"scchatbot/runtime_data/basic_data/Overall cells_dot_plot_data.csv",
        f"process_cell_data/{cell_type_formatted}_dot_plot_data.csv"
    ]
    
    # Use glob patterns to find any dot plot data files
    search_patterns = [
        f"scchatbot/runtime_data/basic_data/*dot_plot_data.csv",
        f"process_cell_data/*dot_plot_data.csv"
    ]
    
    dot_plot_file = None
    
    # First try specific paths
    for path in possible_paths:
        if os.path.exists(path):
            dot_plot_file = path
            break
    
    # If not found, use glob patterns
    if not dot_plot_file:
        for pattern in search_patterns:
            found_files = glob.glob(pattern)
            if found_files:
                dot_plot_file = found_files[0]  # Use first found file
                print(f"📁 Using dot plot file: {dot_plot_file}")
                break
    
    if not dot_plot_file:
        return "Dot plot data is not available. Please ensure the analysis has been run and data files are generated."
    
    try:
        dot_plot_data = pd.read_csv(dot_plot_file)
        fig = px.scatter(
            dot_plot_data,
            x='gene',
            y='leiden',
            size='expression',
            color='expression',
            title=f'Dot Plot ({cell_type_formatted})',
            labels={'gene': 'Gene', 'leiden': 'Cluster', 'expression': 'Expression Level'},
            color_continuous_scale='Blues'
        )
        fig.update_traces(marker=dict(opacity=0.8))
        fig.update_layout(width=1200, height=800, autosize=True)
        # Return the HTML snippet for embedding the interactive figure
        plot_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        return plot_html
    except Exception as e:
        return f"Error generating dot plot: {e}"


def display_processed_umap(cell_type: str) -> str:
    import os
    import pandas as pd
    import plotly.express as px
    import plotly.io as pio

    # standardize your cell type into the form used by process_cells
    std = unified_cell_type_handler(cell_type)  # e.g. "Monocytes"
    
    # Use hierarchical UMAP file lookup strategy
    umap_path, filter_for_descendants = _find_hierarchical_umap_file(std)
    
    if not umap_path:
        print(f"Warning: could not find any UMAP file for '{cell_type}' (standardized: '{std}')")
        return f"UMAP data not available for {cell_type}. Please ensure the cell type has been processed or its parent cell type has been analyzed."

    print(f"📊 Loading UMAP data from: {umap_path} (filter_for_descendants: {filter_for_descendants})")

    try:
        # load UMAP data
        umap_data = pd.read_csv(umap_path)
        
        # Apply filtering if we're using a parent file
        if filter_for_descendants:
            filtered_data = _filter_umap_for_descendants(umap_data, std)
            title_suffix = f"{std} and Subtypes"
        else:
            # Use data as-is (direct file match)
            filtered_data = umap_data
            title_suffix = f"{std} (annotated)"
        
        if filtered_data.empty:
            return f"No {cell_type} cells found in the UMAP data."
        
        print(f"📊 Plotting {len(filtered_data)} cells for {title_suffix}")
        
        # create scatter plot
        fig = px.scatter(
            filtered_data,
            x="UMAP_1",
            y="UMAP_2",
            color="cell_type",
            title=f"{title_suffix} UMAP Plot",
            labels={"UMAP_1": "UMAP 1", "UMAP_2": "UMAP 2"}
        )
        fig.update_traces(marker=dict(size=5, opacity=0.8))
        fig.update_layout(
            width=800, 
            height=600, 
            autosize=False, 
            showlegend=True,
            margin=dict(l=50, r=50, t=80, b=50)
        )

        # return html snippet
        return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        
    except Exception as e:
        print(f"Error loading UMAP data from {umap_path}: {e}")
        return f"Error generating UMAP plot for {cell_type}: {e}"


def _find_hierarchical_umap_file(cell_type: str) -> tuple:
    """
    Find the appropriate UMAP file using hierarchical lookup strategy with Neo4j integration.
    
    Returns:
        (file_path, filter_for_descendants): 
        - file_path: path to the UMAP CSV file to use
        - filter_for_descendants: whether to filter for descendants of the cell type
    """
    import os
    from ..cell_types.utils import get_subtypes
    
    # Strategy 1: Try exact match first (cell type was directly processed)
    exact_path = f'umaps/annotated/{cell_type}_umap_data.csv'
    if os.path.exists(exact_path):
        print(f"🎯 Found exact UMAP file for {cell_type}")
        return exact_path, True  # Still filter to show descendants
    
    # Strategy 2: Check if this is a root cell type by querying Neo4j
    try:
        subtypes = get_subtypes(cell_type)
        if subtypes and len(subtypes) > 0:
            # This cell type has subtypes in Neo4j, so it could be a root
            # Use "Overall cells" file for root cell types
            overall_path = 'umaps/annotated/Overall cells_umap_data.csv'
            if os.path.exists(overall_path):
                print(f"🌟 Using Overall cells file for root cell type: {cell_type}")
                return overall_path, True
    except Exception as e:
        print(f"⚠️ Error querying subtypes for {cell_type}: {e}")
    
    # Strategy 3: Find parent cell type that was processed and contains this cell type
    # Use hierarchy system to find the ancestor that was actually processed
    parent_path = _find_parent_umap_file(cell_type)
    if parent_path:
        return parent_path, True
    
    # Strategy 4: Fallback to "Overall cells" if it exists
    overall_path = 'umaps/annotated/Overall cells_umap_data.csv'
    if os.path.exists(overall_path):
        print(f"🔄 Fallback to Overall cells file for {cell_type}")
        return overall_path, True
    
    return None, False


def _find_parent_umap_file(target_cell_type: str) -> str:
    """
    Find a parent UMAP file that contains the target cell type using the hierarchy system.
    """
    import os
    import glob
    import pandas as pd
    
    # Search all available UMAP files
    umap_files = glob.glob('umaps/annotated/*_umap_data.csv')
    
    # Check each file to see if it contains the target cell type
    for umap_file in umap_files:
        try:
            # Skip if it's the exact match (already tried)
            parent_type = os.path.basename(umap_file).replace('_umap_data.csv', '')
            if parent_type == target_cell_type:
                continue
                
            # Check if this file contains the target cell type
            if _file_contains_cell_type(umap_file, target_cell_type):
                print(f"🔍 Found {target_cell_type} in parent file: {parent_type}")
                return umap_file
                
        except Exception as e:
            print(f"⚠️ Error checking file {umap_file}: {e}")
            continue
    
    return None


def _file_contains_cell_type(file_path: str, target_cell_type: str) -> bool:
    """Check if a UMAP file contains the target cell type."""
    try:
        import pandas as pd
        df = pd.read_csv(file_path)
        if 'cell_type' not in df.columns:
            return False
        
        # Check for exact matches and partial matches
        cell_types = df['cell_type'].unique()
        target_lower = target_cell_type.lower()
        
        for ct in cell_types:
            ct_lower = str(ct).lower()
            # Check if target is contained in this cell type or vice versa
            if target_lower in ct_lower or ct_lower in target_lower:
                return True
        return False
    except Exception:
        return False


def _filter_umap_for_descendants(umap_data, target_cell_type: str):
    """Filter UMAP data to show target cell type and its descendants."""
    if 'cell_type' not in umap_data.columns:
        return umap_data
    
    # Get all cell types that contain the target cell type name
    # This will include the target type and its subtypes
    target_lower = target_cell_type.lower()
    
    # Create masks for different matching strategies
    contains_mask = umap_data['cell_type'].str.lower().str.contains(target_lower, na=False)
    exact_mask = umap_data['cell_type'].str.lower() == target_lower
    
    # Also check for cells where target is a substring (e.g., "T cell" matches "CD4+ T cell")
    substring_mask = umap_data['cell_type'].str.lower().str.contains(
        target_lower.split()[0] if ' ' in target_lower else target_lower, na=False
    )
    
    # Combine all masks
    combined_mask = contains_mask | exact_mask | substring_mask
    
    filtered_data = umap_data[combined_mask]
    
    if filtered_data.empty:
        # If no matches, try a broader search by splitting the cell type name
        print(f"🔍 No direct matches for '{target_cell_type}', trying broader search...")
        words = target_cell_type.split()
        for word in words:
            if len(word) > 2:  # Avoid very short words
                word_mask = umap_data['cell_type'].str.contains(word, case=False, na=False)
                if word_mask.any():
                    filtered_data = umap_data[word_mask]
                    break
    
    return filtered_data


def display_cell_count_comparison(cell_types_data: dict, plot_type: str = "stacked") -> str:
    """
    Create stacked bar chart comparing cell counts across conditions for multiple cell types.
    
    Args:
        cell_types_data: Dictionary where keys are cell types and values are count results
        Example: {
            "T cell": [{"category": "pre", "cell_count": 150, "description": "..."}, ...],
            "B cell": [{"category": "pre", "cell_count": 75, "description": "..."}, ...],
            "Macrophages": [{"category": "pre", "cell_count": 200, "description": "..."}, ...]
        }
        plot_type: "stacked" or "grouped" (default: "stacked")
    
    Returns:
        HTML string containing the plot
    """
    import plotly.express as px
    import plotly.io as pio
    
    try:
        # Validate input
        if not cell_types_data or not isinstance(cell_types_data, dict):
            return "Error: No cell count data provided for visualization."
        
        # Convert aggregated data to plotting format
        plot_data = []
        all_conditions = set()
        
        for cell_type, count_results in cell_types_data.items():
            if not isinstance(count_results, list):
                continue
                
            for result in count_results:
                if isinstance(result, dict) and 'category' in result and 'cell_count' in result:
                    plot_data.append({
                        'cell_type': cell_type,
                        'condition': result['category'],
                        'cell_count': int(result['cell_count']),
                        'description': result.get('description', result['category'])
                    })
                    all_conditions.add(result['category'])
        
        if not plot_data:
            return "Error: No valid cell count data found for visualization."
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(plot_data)
        
        print(f"📊 Creating cell count comparison with {len(df)} data points")
        print(f"📊 Cell types: {df['cell_type'].unique()}")
        print(f"📊 Conditions: {df['condition'].unique()}")
        
        # Create the visualization based on plot type
        if plot_type == "stacked":
            # Create stacked bar chart
            fig = px.bar(
                df,
                x='cell_type',
                y='cell_count',
                color='condition',
                title='Cell Type Count Comparison Across Conditions',
                labels={
                    'cell_type': 'Cell Type',
                    'cell_count': 'Cell Count',
                    'condition': 'Condition'
                },
                hover_data=['description'],
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            
            # Update layout for better readability
            fig.update_layout(
                height=600,
                width=1000,
                xaxis_title="Cell Type",
                yaxis_title="Cell Count",
                legend_title="Condition",
                xaxis={'categoryorder': 'total descending'},
                margin=dict(l=80, r=50, t=80, b=100),
                font=dict(size=12)
            )
            
            # Rotate x-axis labels if needed
            fig.update_xaxes(tickangle=45)
            
        elif plot_type == "grouped":
            # Create grouped bar chart
            fig = px.bar(
                df,
                x='cell_type',
                y='cell_count',
                color='condition',
                barmode='group',
                title='Cell Type Count Comparison Across Conditions (Grouped)',
                labels={
                    'cell_type': 'Cell Type',
                    'cell_count': 'Cell Count',
                    'condition': 'Condition'
                },
                hover_data=['description'],
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            
            fig.update_layout(
                height=600,
                width=1000,
                xaxis_title="Cell Type",
                yaxis_title="Cell Count",
                legend_title="Condition",
                xaxis={'categoryorder': 'total descending'},
                margin=dict(l=80, r=50, t=80, b=100),
                font=dict(size=12)
            )
            
            fig.update_xaxes(tickangle=45)
            
        else:
            return f"Error: Unsupported plot type '{plot_type}'. Use 'stacked' or 'grouped'."
        
        # Convert to HTML
        plot_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        
        print(f"✅ Cell count comparison visualization generated: {len(plot_html)} characters")
        return plot_html
        
    except Exception as e:
        return f"Error generating cell count comparison plot: {e}"


def display_dea_heatmap(cell_type: str, top_n_genes: int = 20, cluster_genes: bool = True, cluster_samples: bool = True) -> str:
    """
    Create dual heatmap visualization from DEA (Differential Expression Analysis) results for a single cell type.
    Generates separate heatmaps for upregulated and downregulated genes.
    
    Args:
        cell_type: The cell type to visualize DEA results for
        top_n_genes: Number of top differentially expressed genes to display per direction (default: 20)
        cluster_genes: Whether to apply hierarchical clustering to genes (default: True)
        cluster_samples: Whether to apply hierarchical clustering to samples/conditions (default: True)
    
    Returns:
        HTML string containing both interactive heatmap plots (upregulated and downregulated)
    """
    import plotly.graph_objects as go
    import plotly.io as pio
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist
    from ..cell_types.standardization import unified_cell_type_handler
    
    def create_single_heatmap(genes_data, direction, cell_type, conditions, cluster_genes, cluster_samples):
        """Helper function to create a single heatmap for either up or downregulated genes"""
        import numpy as np
        
        if not genes_data:
            return f"<p>No {direction} genes found for {cell_type}</p>"
        
        # Create expression matrix
        expression_matrix = []
        gene_labels = []
        
        for gene in genes_data:
            gene_row = []
            for condition in conditions:
                logfc = genes_data[gene].get(condition, 0)  # Use 0 if gene not found in condition
                gene_row.append(logfc)
            expression_matrix.append(gene_row)
            gene_labels.append(gene)
        
        if not expression_matrix:
            return f"<p>No expression data to visualize for {direction} genes in {cell_type}</p>"
        
        # Convert to numpy array for clustering
        matrix = np.array(expression_matrix)
        
        # Apply clustering if requested
        gene_order = list(range(len(gene_labels)))
        condition_order = list(range(len(conditions)))
        
        if cluster_genes and len(gene_labels) > 1:
            try:
                gene_distances = pdist(matrix, metric='euclidean')
                gene_linkage = linkage(gene_distances, method='ward')
                gene_dendro = dendrogram(gene_linkage, no_plot=True)
                gene_order = gene_dendro['leaves']
            except Exception as e:
                print(f"⚠️ Gene clustering failed for {direction}: {e}")
        
        if cluster_samples and len(conditions) > 1:
            try:
                condition_distances = pdist(matrix.T, metric='euclidean')
                condition_linkage = linkage(condition_distances, method='ward')
                condition_dendro = dendrogram(condition_linkage, no_plot=True)
                condition_order = condition_dendro['leaves']
            except Exception as e:
                print(f"⚠️ Condition clustering failed for {direction}: {e}")
        
        # Reorder matrix based on clustering
        clustered_matrix = matrix[np.ix_(gene_order, condition_order)]
        clustered_genes = [gene_labels[i] for i in gene_order]
        clustered_conditions = [conditions[i] for i in condition_order]
        
        # Choose appropriate colorscale based on direction
        if direction == "upregulated":
            colorscale = 'Reds'
            color_range = [0, max(clustered_matrix.flatten()) if len(clustered_matrix.flatten()) > 0 else 1]
        else:
            colorscale = 'Blues_r'
            color_range = [min(clustered_matrix.flatten()) if len(clustered_matrix.flatten()) > 0 else -1, 0]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=clustered_matrix,
            x=clustered_conditions,
            y=clustered_genes,
            colorscale=colorscale,
            zmin=color_range[0],
            zmax=color_range[1],
            colorbar=dict(
                title="Log Fold Change",
                titleside="right"
            ),
            hoverongaps=False,
            hovertemplate='<b>Gene:</b> %{y}<br>' +
                         '<b>Condition:</b> %{x}<br>' +
                         '<b>Log FC:</b> %{z:.2f}<br>' +
                         '<extra></extra>'
        ))
        
        # Update layout with frontend-friendly settings
        fig.update_layout(
            title=f'{direction.title()} Genes - {cell_type}<br><sub>Top {len(clustered_genes)} {direction} genes</sub>',
            xaxis_title="Conditions",
            yaxis_title="Genes",
            width=None,  # Let container control width
            height=max(400, len(clustered_genes) * 25 + 150),
            font=dict(size=10),
            margin=dict(l=150, r=50, t=100, b=50),
            xaxis=dict(
                tickangle=45,
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                tickfont=dict(size=9),
                autorange='reversed'
            ),
            # Additional settings to prevent overlap
            autosize=True,
            showlegend=False
        )
        
        # Generate HTML with improved configuration
        return pio.to_html(
            fig, 
            full_html=False, 
            include_plotlyjs='cdn',
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                'responsive': True
            },
            div_id=f"heatmap_{direction}_{cell_type.replace(' ', '_')}"
        )

    try:
        # Standardize cell type name
        cell_type = unified_cell_type_handler(cell_type)
        
        # Look for DEA results files - try multiple possible locations
        import glob
        import os
        
        # Try different possible paths for DEA files
        possible_paths = [
            f"scchatbot/deg_res/{cell_type}_markers_*.csv",
            f"deg_res/{cell_type}_markers_*.csv", 
            f"scchatbot/{cell_type}_markers_*.csv",
            f"{cell_type}_markers_*.csv"
        ]
        
        dea_files = []
        for path_pattern in possible_paths:
            found_files = glob.glob(path_pattern)
            if found_files:
                dea_files = found_files
                print(f"📊 Found DEA files at: {path_pattern}")
                break
        
        if not dea_files:
            return f"Error: No DEA results found for {cell_type}. Please run DEA analysis first using dea_split_by_condition."
        
        print(f"📊 Found {len(dea_files)} DEA result files for {cell_type}")
        
        # Separate storage for upregulated and downregulated genes
        upregulated_genes = {}  # gene -> {condition: logfc}
        downregulated_genes = {}  # gene -> {condition: logfc}
        conditions = []
        
        for file_path in dea_files:
            # Extract condition from filename
            filename = os.path.basename(file_path)
            if "_bulk.csv" in filename:
                condition = "bulk"
            else:
                # Extract condition name from filename like "T cell_markers_condition1.csv"
                condition = filename.replace(f"{cell_type}_markers_", "").replace(".csv", "")
            
            conditions.append(condition)
            
            try:
                df = pd.read_csv(file_path)
                print(f"📊 Loading DEA data from {condition}: {len(df)} genes")
                
                # Check column names in the CSV
                print(f"📊 CSV columns: {list(df.columns)}")
                
                # Handle different column name formats
                logfc_col = None
                pval_col = None
                gene_col = None
                direction_col = None
                
                # Find the correct column names
                for col in df.columns:
                    if 'logfoldchange' in col.lower() or 'logfc' in col.lower() or col == 'log_fc':
                        logfc_col = col
                    elif 'pvals_adj' in col.lower() or 'padj' in col.lower() or 'fdr' in col.lower():
                        pval_col = col
                    elif 'names' in col.lower() or 'gene' in col.lower() or col == 'index':
                        gene_col = col
                    elif 'direction' in col.lower():
                        direction_col = col
                
                if not all([logfc_col, pval_col, gene_col]):
                    print(f"⚠️ Missing required columns in {file_path}")
                    print(f"   logfc_col: {logfc_col}, pval_col: {pval_col}, gene_col: {gene_col}")
                    continue
                
                # Get significant genes and separate by direction
                df_filtered = df[
                    (df[pval_col] < 0.05) & 
                    (abs(df[logfc_col]) > 1)
                ]
                
                if len(df_filtered) == 0:
                    print(f"⚠️ No significant genes found in {condition}")
                    continue
                
                # Separate upregulated and downregulated genes
                if direction_col and direction_col in df_filtered.columns:
                    # Use direction column if available
                    df_up = df_filtered[df_filtered[direction_col] == 'upregulated'].copy()
                    df_down = df_filtered[df_filtered[direction_col] == 'downregulated'].copy()
                else:
                    # Fall back to logFC sign
                    df_up = df_filtered[df_filtered[logfc_col] > 0].copy()
                    df_down = df_filtered[df_filtered[logfc_col] < 0].copy()
                
                # Sort and select top genes for each direction
                df_up = df_up.reindex(df_up[logfc_col].sort_values(ascending=False).index).head(top_n_genes)
                df_down = df_down.reindex(df_down[logfc_col].abs().sort_values(ascending=False).index).head(top_n_genes)
                
                # Store upregulated genes
                for _, row in df_up.iterrows():
                    gene = row[gene_col]
                    logfc = row[logfc_col]
                    if gene not in upregulated_genes:
                        upregulated_genes[gene] = {}
                    upregulated_genes[gene][condition] = logfc
                
                # Store downregulated genes
                for _, row in df_down.iterrows():
                    gene = row[gene_col]
                    logfc = row[logfc_col]
                    if gene not in downregulated_genes:
                        downregulated_genes[gene] = {}
                    downregulated_genes[gene][condition] = logfc
                    
            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")
                continue
        
        if not upregulated_genes and not downregulated_genes:
            return f"Error: No significant genes found in DEA results for {cell_type}."
        
        # Clean up conditions list
        conditions = sorted(set(conditions))  # Remove duplicates and sort
        
        # Select top genes for each direction based on maximum absolute fold change across conditions
        def select_top_genes(genes_dict, n_genes):
            gene_max_fc = {}
            for gene, condition_data in genes_dict.items():
                max_abs_fc = max([abs(fc) for fc in condition_data.values()])
                gene_max_fc[gene] = max_abs_fc
            return dict(list(sorted(gene_max_fc.items(), key=lambda x: x[1], reverse=True))[:n_genes])
        
        # Get top genes for each direction
        top_upregulated = select_top_genes(upregulated_genes, top_n_genes) if upregulated_genes else {}
        top_downregulated = select_top_genes(downregulated_genes, top_n_genes) if downregulated_genes else {}
        
        # Filter the gene dictionaries to only include top genes
        filtered_upregulated = {gene: data for gene, data in upregulated_genes.items() if gene in top_upregulated}
        filtered_downregulated = {gene: data for gene, data in downregulated_genes.items() if gene in top_downregulated}
        
        print(f"📊 Creating dual heatmaps: {len(filtered_upregulated)} upregulated, {len(filtered_downregulated)} downregulated genes")
        
        # Create both heatmaps
        upregulated_html = create_single_heatmap(filtered_upregulated, "upregulated", cell_type, conditions, cluster_genes, cluster_samples)
        downregulated_html = create_single_heatmap(filtered_downregulated, "downregulated", cell_type, conditions, cluster_genes, cluster_samples)
        
        # Check if we have valid heatmaps to display
        valid_plots = []
        if upregulated_html and not upregulated_html.startswith("<p>No"):
            valid_plots.append({
                "type": "upregulated_heatmap",
                "title": f"Upregulated Genes - {cell_type}",
                "html": upregulated_html
            })
        
        if downregulated_html and not downregulated_html.startswith("<p>No"):
            valid_plots.append({
                "type": "downregulated_heatmap", 
                "title": f"Downregulated Genes - {cell_type}",
                "html": downregulated_html
            })
        
        if not valid_plots:
            return f"No significant genes found for DEA heatmap visualization in {cell_type}."
        
        # If we have multiple valid plots, return multiple plots structure (like enrichment analysis)
        if len(valid_plots) > 1:
            print(f"🔗 Creating separate DEA heatmap objects for {len(valid_plots)} plots")
            print(f"📊 Upregulated heatmap size: {len(valid_plots[0]['html'])} chars")
            print(f"🔴 Downregulated heatmap size: {len(valid_plots[1]['html'])} chars")
            
            # Return multiple plots structure for backend processing (same as enrichment)
            multiple_plots_result = {
                "multiple_plots": True,
                "plots": valid_plots
            }
            
            # Add legacy combined HTML for backward compatibility
            combined_html = f"""
            <div style="
                width: 100%; 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px;
                box-sizing: border-box;
                overflow: hidden;
                position: relative;
            ">
                <div style="
                    width: 100%;
                    margin-bottom: 50px;
                    padding: 20px;
                    background-color: #fafafa;
                    border-radius: 8px;
                    border: 1px solid #e0e0e0;
                    box-sizing: border-box;
                    overflow: hidden;
                    position: relative;
                    clear: both;
                ">
                    <h2 style="
                        text-align: center; 
                        color: #d62728; 
                        margin: 0 0 20px 0;
                        padding: 10px;
                        font-family: Arial, sans-serif;
                        font-size: 18px;
                        border-bottom: 2px solid #d62728;
                    ">Upregulated Genes</h2>
                    <div style="
                        width: 100%;
                        overflow: hidden;
                        position: relative;
                    ">
                        {upregulated_html}
                    </div>
                </div>
                
                <div style="
                    width: 100%;
                    margin-top: 50px;
                    padding: 20px;
                    background-color: #fafafa;
                    border-radius: 8px;
                    border: 1px solid #e0e0e0;
                    box-sizing: border-box;
                    overflow: hidden;
                    position: relative;
                    clear: both;
                ">
                    <h2 style="
                        text-align: center; 
                        color: #1f77b4; 
                        margin: 0 0 20px 0;
                        padding: 10px;
                        font-family: Arial, sans-serif;
                        font-size: 18px;
                        border-bottom: 2px solid #1f77b4;
                    ">Downregulated Genes</h2>
                    <div style="
                        width: 100%;
                        overflow: hidden;
                        position: relative;
                    ">
                        {downregulated_html}
                    </div>
                </div>
            </div>
            """
            multiple_plots_result["legacy_combined_html"] = combined_html
            
            print(f"🎯 Created multiple DEA heatmap structure with {len(multiple_plots_result['plots'])} individual plots")
            return multiple_plots_result
        else:
            # Single plot case - return the HTML directly
            print(f"✅ Single DEA heatmap generated for {cell_type}")
            return valid_plots[0]["html"]
        
    except Exception as e:
        return f"Error generating DEA heatmap for {cell_type}: {e}"


