import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import json
from plotly.utils import PlotlyJSONEncoder
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
        
        # Combine plots if both requested
        if plot_type == "both":
            print(f"🔗 Combining {len(plots_html)} plots")
            print(f"📊 Bar plot size: {len(plots_html[0])} chars")
            print(f"🔴 Dot plot size: {len(plots_html[1])} chars") 
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
            print(f"🎯 Combined HTML size: {len(combined_html)} characters")
            return combined_html
        else:
            return plots_html[0]
            
    except Exception as e:
        return f"Error generating {analysis} {plot_type} plot: {e}"


def display_enrichment_barplot(
    analysis: str,
    cell_type: str,
    top_n: int = 10,
    domain: str = None,
    condition: str = None
) -> str:
    """
    Generic barplot of top_n enriched terms for a given analysis and cell type.
    This is now a wrapper function that calls the unified visualization function.
    
    analysis: "reactome", "kegg", "gsea", or "go"
    domain: required if analysis=="go" (one of "BP","MF","CC")
    condition: optional specific condition folder (e.g., "kegg_p1_post")
    """
    return display_enrichment_visualization(
        analysis=analysis,
        cell_type=cell_type,
        plot_type="bar",
        top_n=top_n,
        domain=domain,
        condition=condition
    )


def display_enrichment_dotplot(
    analysis: str,
    cell_type: str,
    top_n: int = 10,
    domain: str = None,
    condition: str = None
) -> str:
    """
    Generic dotplot of avg_log2fc vs. top enriched terms for a given analysis.
    This is now a wrapper function that calls the unified visualization function.
    
    analysis: "reactome", "kegg", "gsea", or "go"
    domain: required if analysis=="go" (one of "BP","MF","CC")
    condition: optional specific condition folder (e.g., "kegg_p1_post")
    """
    return display_enrichment_visualization(
        analysis=analysis,
        cell_type=cell_type,
        plot_type="dot",
        top_n=top_n,
        domain=domain,
        condition=condition
    )


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

def display_umap(cell_type: str) -> str:
    cell_type_formatted = unified_cell_type_handler(cell_type)
    umap_data = pd.read_csv(f'scchatbot/runtime_data/process_cell_data/{cell_type_formatted}_umap_data.csv')
    if cell_type_formatted != "Overall cells":
        umap_data['original_cell_type'] = umap_data['cell_type']
        umap_data['cell_type'] = 'Unknown'
    fig = px.scatter(
        umap_data,
        x="UMAP_1",
        y="UMAP_2",
        color="leiden",
        symbol="patient_name",
        title="T Cells UMAP Plot",
        labels={"UMAP_1": "UMAP 1", "UMAP_2": "UMAP 2"}
    )
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    fig.update_layout(width=1200, height=800, autosize=True, showlegend=False)
    custom_legend = go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='rgba(0,0,0,0)'),
        legendgroup="Unknown",
        showlegend=True,
        name="Unknown"
    )
    fig.add_trace(custom_legend)
    # Generate HTML snippet of the interactive plot using Plotly's to_html
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    return plot_html


def display_processed_umap(cell_type: str) -> str:
    import os
    import pandas as pd
    import plotly.express as px
    import plotly.io as pio

    # standardize your cell type into the form used by process_cells
    std = unified_cell_type_handler(cell_type)  # e.g. "Monocytes"
    umap_path = f'umaps/annotated/{std}_umap_data.csv'
    if not os.path.exists(umap_path):
        print(f"Warning: could not find an annotated UMAP file for '{cell_type}' at {umap_path}")
        return ""

    # load and plot
    umap_data = pd.read_csv(umap_path)
    fig = px.scatter(
        umap_data,
        x="UMAP_1",
        y="UMAP_2",
        color="cell_type",
        title=f"{std} (annotated) UMAP Plot",
        labels={"UMAP_1": "UMAP 1", "UMAP_2": "UMAP 2"}
    )
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    fig.update_layout(width=1200, height=800, autosize=True, showlegend=True)

    # return html snippet
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

def display_gsea_dotplot(cell_type: str = "overall", condition: str = None, top_n: int = 20) -> str:
    """
    Creates a GSEA dot plot using the unified enrichment visualization function.
    This is now a wrapper function that calls the unified visualization function.
    """
    return display_enrichment_visualization(
        analysis="gsea",
        cell_type=cell_type,
        plot_type="dot",
        top_n=top_n,
        domain=None,
        condition=condition
    )

def display_cell_type_composition() -> str:
    """
    Generates a dendrogram of cell type composition using dynamic file discovery.
    """
    # Try multiple possible locations for dendrogram data
    possible_paths = [
        "scchatbot/runtime_data/basic_data/dendrogram_data.csv",
        "basic_data/dendrogram_data.csv",
        "scchatbot/runtime_data/basic_data/Overall cells_dendrogram_data.csv"
    ]
    
    # Use glob patterns to find any dendrogram data files
    search_patterns = [
        "scchatbot/runtime_data/basic_data/*dendrogram_data.csv",
        "basic_data/*dendrogram_data.csv"
    ]
    
    dendrogram_file = None
    
    # First try specific paths
    for path in possible_paths:
        if os.path.exists(path):
            dendrogram_file = path
            break
    
    # If not found, use glob patterns
    if not dendrogram_file:
        for pattern in search_patterns:
            found_files = glob.glob(pattern)
            if found_files:
                dendrogram_file = found_files[0]  # Use first found file
                print(f"📁 Using dendrogram file: {dendrogram_file}")
                break
    
    if not dendrogram_file:
        return "Cell type composition data is not available. Please ensure the analysis has been run and data files are generated."
    
    try:
        dendrogram_data = pd.read_csv(dendrogram_file)
        fig = ff.create_dendrogram(dendrogram_data.values, orientation='left')
        fig.update_layout(title='Dendrogram', xaxis_title='Distance', yaxis_title='Clusters',
                          width=1200, height=800, autosize=True)
        return fig.to_json()
    except Exception as e:
        return f"Error generating cell type composition plot: {e}"

def generate_umap_plot(data: pd.DataFrame) -> str:
    """
    Generates a UMAP projection scatter plot from the provided data and returns the JSON representation.
    """
    fig = px.scatter(
        data,
        x='umap1',
        y='umap2',
        hover_data={'label': True},
        title='UMAP Projection'
    )
    fig.update_layout(hovermode='closest', xaxis_title='Component 1', yaxis_title='Component 2')
    graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    return graph_json