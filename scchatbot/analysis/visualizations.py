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



