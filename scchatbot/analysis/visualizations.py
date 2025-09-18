import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from ..cell_types.standardization import unified_cell_type_handler
import plotly.io as pio
import os
import glob
from typing import List, Optional
from .palettes.seurat import seurat_blue_to_lightred, seurat_continuous_lightgrey_red, seurat_discrete


def _seurat_colorscale_to_plotly():
    """Convert Seurat blue-to-lightred colormap to Plotly colorscale format."""
    cmap = seurat_blue_to_lightred()
    n_colors = 10
    colors = []
    for i in range(n_colors):
        position = i / (n_colors - 1)
        rgba = cmap(position)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        )
        colors.append([position, hex_color])
    return colors


def _seurat_feature_colorscale_to_plotly():
    """Convert Seurat lightgrey-to-red colormap to Plotly colorscale format for feature plots."""
    cmap = seurat_continuous_lightgrey_red()
    n_colors = 10
    colors = []
    for i in range(n_colors):
        position = i / (n_colors - 1)
        rgba = cmap(position)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        )
        colors.append([position, hex_color])
    return colors


def _save_plot_as_pdf(fig, filename: str) -> None:
    """Save a Plotly figure as PDF in the scchatbot/plots directory."""
    try:
        plots_dir = "scchatbot/plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        kaleido_available = False
        try:
            import kaleido
            kaleido_available = True
        except ImportError:
            print(f"⚠️ Kaleido not available for PDF export. Install with: pip install kaleido")
        
        pdf_path = os.path.join(plots_dir, f"{filename}.pdf")
        png_path = os.path.join(plots_dir, f"{filename}.png")
        html_path = os.path.join(plots_dir, f"{filename}.html")
        
        if kaleido_available:
            try:
                fig.write_image(
                    png_path, 
                    format="png",
                    width=1200,
                    height=800,
                    scale=2  # High resolution for print quality
                )
                print(f"📷 Plot saved as high-res PNG: {png_path}")
                
                try:
                    fig.write_image(
                        pdf_path, 
                        format="pdf",
                        width=1200,
                        height=800,
                        scale=1
                    )
                    print(f"📄 Plot also saved as PDF (with vector interpolation): {pdf_path}")
                except Exception as pdf_error:
                    print(f"⚠️ PDF save failed: {pdf_error}")
                
                return
            except Exception as png_error:
                print(f"⚠️ PNG save failed: {png_error}")
                try:
                    fig.write_image(
                        pdf_path, 
                        format="pdf",
                        width=1200,
                        height=800,
                        scale=1
                    )
                    print(f"📄 Plot saved as PDF (fallback, may have interpolation): {pdf_path}")
                    return
                except Exception as pdf_error2:
                    print(f"⚠️ PDF fallback also failed: {pdf_error2}")
        
        try:
            fig.write_html(html_path)
            print(f"🌐 Plot saved as HTML (interactive format): {html_path}")
        except Exception as html_error:
            print(f"⚠️ HTML save failed: {html_error}")
            
    except Exception as e:
        print(f"⚠️ Could not save plot {filename}: {e}")


def _find_enrichment_file(analysis: str, cell_type: str, condition: str = None) -> str:
    """
    Unified file discovery function for enrichment analyses.
    Used by all enrichment visualization functions.
    """
    base = analysis.lower()
    
    if condition:
        folder = f"scchatbot/enrichment/{condition}"
        fname = f"results_summary_{cell_type}.csv"
    else:
        if base == "go":
            folder = f"scchatbot/enrichment/go_bp"  # Default to BP
            fname = f"results_summary_{cell_type}.csv"
        elif base.startswith("go_"):
            folder = f"scchatbot/enrichment/{base}"
            fname = f"results_summary_{cell_type}.csv"
        else:
            folder = f"scchatbot/enrichment/{base}"
            fname = f"results_summary_{cell_type}.csv"
    
    path = os.path.join(folder, fname)
    
    if not os.path.exists(path):
        fname_alt = f"{base}_results_summary_{cell_type}.csv"
        path_alt = os.path.join(folder, fname_alt)
        if os.path.exists(path_alt):
            return path_alt
    
    if base == "gsea" and not os.path.exists(path):
        search_patterns = [
            f"scchatbot/enrichment/gsea_*/results_summary_{cell_type}_*.csv",
            f"scchatbot/enrichment/gsea*/results_summary_{cell_type}_*.csv",
            f"scchatbot/enrichment/gsea_*/results_summary_{cell_type}.csv",
            f"scchatbot/enrichment/gsea*/results_summary_{cell_type}.csv",
            f"scchatbot/enrichment/gsea/results_summary_*.csv",
            f"scchatbot/enrichment/gsea*/results_summary_*.csv"
        ]

        for pattern in search_patterns:
            found_files = glob.glob(pattern)
            if found_files:
                print(f"🔍 Found GSEA results at: {found_files[0]}")
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
    
    if analysis.lower() == "go":
        if domain is None:
            return "Error: `domain` must be provided for GO analysis (BP, MF, or CC)."
        analysis = f"go_{domain.lower()}"
    
    file_path = _find_enrichment_file(analysis, cell_type, condition)
    
    if not file_path:
        return f"Enrichment data not available for {analysis} analysis of {cell_type}. Please run the enrichment analysis first."
    
    try:
        df = pd.read_csv(file_path)
        df["minus_log10_p"] = -np.log10(df["p_value"])
        top = df.nsmallest(top_n, "p_value")
        
        plots_html = []
        
        if plot_type in ["bar", "both"]:
            print(f"📊 Generating bar plot with {len(top)} terms")
            bar_fig = px.bar(
                top,
                x="gene_ratio",
                y="Term",
                color="minus_log10_p",
                orientation="h",
                title=f"Top {top_n} {analysis.upper()} Terms - Bar Plot ({cell_type})",
                labels={"gene_ratio": "Gene Ratio", "Term": "Term", "minus_log10_p": "-log10(p-value)"},
                color_continuous_scale=_seurat_colorscale_to_plotly()
            )
            bar_fig.update_layout(
                height=top_n * 40 + 200,
                yaxis={"categoryorder": "total ascending"},
                margin=dict(l=100, r=50, t=50, b=50),  # Add margins to prevent cropping
                autosize=True
            )

            bar_fig.update_coloraxes(
                colorbar_title="-log10(p-value)",
                colorbar_thickness=15,
                colorbar_len=0.8
            )
            
            _save_plot_as_pdf(bar_fig, f"{analysis}_{cell_type}_bar_plot")
            
            bar_html = pio.to_html(bar_fig, full_html=False, include_plotlyjs="cdn")
            plots_html.append(bar_html)
            print(f"✅ Bar plot HTML generated: {len(bar_html)} characters")
        
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
                orientation="h",
                color_continuous_scale=_seurat_colorscale_to_plotly()
            )
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

            dot_fig.update_coloraxes(
                colorbar_title="-log10(p-value)",
                colorbar_thickness=15,
                colorbar_len=0.8
            )
            
            _save_plot_as_pdf(dot_fig, f"{analysis}_{cell_type}_dot_plot")
            
            dot_html = pio.to_html(dot_fig, full_html=False, include_plotlyjs="cdn")
            plots_html.append(dot_html)
            print(f"✅ Dot plot HTML generated: {len(dot_html)} characters")
        
        if plot_type == "both":
            print(f"🔗 Creating separate plot objects for {len(plots_html)} plots")
            print(f"📊 Bar plot size: {len(plots_html[0])} chars")
            print(f"🔴 Dot plot size: {len(plots_html[1])} chars") 
            
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
        
        # Save dot plot as PDF
        _save_plot_as_pdf(fig, f"dotplot_{cell_type_formatted}")
        
        # Return the HTML snippet for embedding the interactive figure
        plot_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        return plot_html
    except Exception as e:
        return f"Error generating dot plot: {e}"


def display_processed_umap(cell_type: str) -> str:
    # standardize your cell type into the form used by process_cells
    std = unified_cell_type_handler(cell_type)  # e.g. "Monocytes"

    # Use hierarchical UMAP file lookup strategy with intelligent filtering
    result = _find_hierarchical_umap_file(std)
    if len(result) == 3:
        umap_path, filter_for_descendants, show_mode = result
    else:
        # Backward compatibility - old function returned 2 values
        umap_path, filter_for_descendants = result
        show_mode = 'legacy'

    if not umap_path:
        print(f"Warning: could not find any UMAP file for '{cell_type}' (standardized: '{std}')")
        return f"UMAP data not available for {cell_type}. Please ensure the cell type has been processed or its parent cell type has been analyzed."

    print(f"📊 Loading UMAP data from: {umap_path} (filter_for_descendants: {filter_for_descendants}, mode: {show_mode})")

    try:
        # load UMAP data
        umap_data = pd.read_csv(umap_path)

        # Apply filtering logic based on the discovery mode
        if filter_for_descendants:
            filtered_data = _filter_umap_for_descendants(umap_data, std)
            if show_mode == 'parent':
                title_suffix = f"{std} (from parent cell type)"
            elif show_mode == 'root':
                title_suffix = f"{std} and Subtypes"
            else:
                title_suffix = f"{std} and Subtypes"
        else:
            # Use data as-is (exact file match - show all subtypes in this file)
            filtered_data = umap_data
            if show_mode == 'exact':
                title_suffix = f"{std} Subtypes"
            else:
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

        # Save UMAP plot as PDF
        _save_plot_as_pdf(fig, f"UMAP_{std}")

        # return html snippet
        return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

    except Exception as e:
        print(f"Error loading UMAP data from {umap_path}: {e}")
        return f"Error generating UMAP plot for {cell_type}: {e}"


def display_leiden_umap(cell_type: str) -> str:
    """
    Generate UMAP plot colored by leiden clusters instead of cell types.
    Uses the same hierarchical file discovery logic as display_processed_umap.

    Args:
        cell_type: Cell type to analyze (used for file discovery)

    Returns:
        HTML string containing interactive Plotly UMAP plot colored by leiden clusters
    """

    # standardize your cell type into the form used by process_cells
    std = unified_cell_type_handler(cell_type)  # e.g. "Monocytes"

    # Use hierarchical UMAP file lookup strategy with intelligent filtering
    result = _find_hierarchical_umap_file(std)
    if len(result) == 3:
        umap_path, filter_for_descendants, show_mode = result
    else:
        # Backward compatibility - old function returned 2 values
        umap_path, filter_for_descendants = result
        show_mode = 'legacy'

    if not umap_path:
        print(f"Warning: could not find any UMAP file for '{cell_type}' (standardized: '{std}')")
        return f"UMAP data not available for {cell_type}. Please ensure the cell type has been processed or its parent cell type has been analyzed."

    print(f"📊 Loading UMAP data for leiden clusters from: {umap_path} (filter_for_descendants: {filter_for_descendants}, mode: {show_mode})")

    try:
        # load UMAP data
        umap_data = pd.read_csv(umap_path)

        # Apply filtering logic based on the discovery mode
        if filter_for_descendants:
            filtered_data = _filter_umap_for_descendants(umap_data, std)
            if show_mode == 'parent':
                title_suffix = f"{std} Leiden Clusters (from parent cell type)"
            elif show_mode == 'root':
                title_suffix = f"{std} Leiden Clusters"
            else:
                title_suffix = f"{std} Leiden Clusters"
        else:
            # Use data as-is (exact file match - show all subtypes in this file)
            filtered_data = umap_data
            if show_mode == 'exact':
                title_suffix = f"{std} Leiden Clusters"
            else:
                title_suffix = f"{std} Leiden Clusters"

        if filtered_data.empty:
            return f"No {cell_type} cells found in the UMAP data."

        # Check if leiden column exists
        if 'leiden' not in filtered_data.columns:
            return f"Error: No leiden clustering information found in the data for {cell_type}."

        print(f"📊 Plotting {len(filtered_data)} cells for {title_suffix}")

        # Convert leiden to string to ensure proper discrete coloring
        filtered_data = filtered_data.copy()
        filtered_data['leiden'] = filtered_data['leiden'].astype(str)

        # Get Seurat discrete colors for leiden clusters
        unique_clusters = sorted(filtered_data['leiden'].unique(), key=lambda x: int(x) if x.isdigit() else float('inf'))
        n_clusters = len(unique_clusters)
        seurat_colors = seurat_discrete(n_clusters)
        seurat_colors_hex = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in seurat_colors]

        # Create color mapping
        color_map = {cluster: color for cluster, color in zip(unique_clusters, seurat_colors_hex)}

        # create scatter plot colored by leiden clusters
        fig = px.scatter(
            filtered_data,
            x="UMAP_1",
            y="UMAP_2",
            color="leiden",
            title=f"{title_suffix} UMAP Plot",
            labels={"UMAP_1": "UMAP 1", "UMAP_2": "UMAP 2", "leiden": "Leiden Cluster"},
            color_discrete_map=color_map
        )
        fig.update_traces(marker=dict(size=5, opacity=0.8))
        fig.update_layout(
            width=800,
            height=600,
            autosize=False,
            showlegend=True,
            margin=dict(l=50, r=50, t=80, b=50)
        )

        # Save UMAP plot as PDF
        _save_plot_as_pdf(fig, f"UMAP_leiden_{std}")

        # return html snippet
        return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

    except Exception as e:
        print(f"Error loading UMAP data from {umap_path}: {e}")
        return f"Error generating leiden UMAP plot for {cell_type}: {e}"


def display_overall_umap(color_mode: str = "cell_type") -> str:
    """
    Generate overall UMAP plot with global view using the comprehensive Overall cells dataset.
    Provides both biological (cell type) and computational (accumulative leiden) perspectives.

    Args:
        color_mode: "cell_type" for biological annotation or "accumulative_leiden" for global clusters

    Returns:
        HTML string containing interactive Plotly UMAP plot
    """
    import os
    import pandas as pd
    import plotly.express as px
    import plotly.io as pio

    # Always use the comprehensive Overall cells file for global view
    umap_path = 'umaps/annotated/Overall cells_umap_data.csv'

    if not os.path.exists(umap_path):
        return f"Error: Overall cells UMAP data not found at {umap_path}. Please ensure initial annotation has been completed."

    print(f"📊 Loading global UMAP data from: {umap_path} (color_mode: {color_mode})")

    try:
        # Load the comprehensive dataset
        umap_data = pd.read_csv(umap_path)

        # Validate required columns
        required_cols = ['UMAP_1', 'UMAP_2']
        if color_mode == "cell_type":
            required_cols.append('cell_type')
        elif color_mode == "accumulative_leiden":
            required_cols.append('accumulative_leiden')
        else:
            return f"Error: Invalid color_mode '{color_mode}'. Use 'cell_type' or 'accumulative_leiden'."

        missing_cols = [col for col in required_cols if col not in umap_data.columns]
        if missing_cols:
            return f"Error: Missing required columns in UMAP data: {missing_cols}"

        print(f"📊 Plotting {len(umap_data)} cells in global view (color_mode: {color_mode})")

        # Prepare data based on color mode
        if color_mode == "cell_type":
            color_column = "cell_type"
            title = "Overall Cell Types UMAP (Global View)"
            legend_title = "Cell Type"

            # Get unique cell types and assign Seurat discrete colors
            unique_types = sorted(umap_data['cell_type'].unique())
            n_types = len(unique_types)
            seurat_colors = seurat_discrete(n_types)
            seurat_colors_hex = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in seurat_colors]
            color_map = {cell_type: color for cell_type, color in zip(unique_types, seurat_colors_hex)}

        else:  # accumulative_leiden
            color_column = "consolidated_leiden"
            title = "Consolidated Leiden Clusters UMAP (Cell Type Unified)"
            legend_title = "Consolidated Cluster"

            # Consolidate leiden clusters by cell type
            umap_data = umap_data.copy()

            # Create mapping from cell type to consolidated leiden number
            unique_cell_types = sorted(umap_data['cell_type'].unique())
            cell_type_to_leiden = {cell_type: i for i, cell_type in enumerate(unique_cell_types)}

            # Apply consolidation mapping
            umap_data['consolidated_leiden'] = umap_data['cell_type'].map(cell_type_to_leiden).astype(str)

            print(f"🔄 Consolidated {len(umap_data['accumulative_leiden'].unique())} original leiden clusters into {len(unique_cell_types)} cell-type-based clusters")

            # Create color mapping for consolidated clusters
            unique_clusters = sorted(umap_data['consolidated_leiden'].unique(),
                                   key=lambda x: int(x) if x.isdigit() else float('inf'))
            n_clusters = len(unique_clusters)
            seurat_colors = seurat_discrete(n_clusters)
            seurat_colors_hex = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in seurat_colors]
            color_map = {cluster: color for cluster, color in zip(unique_clusters, seurat_colors_hex)}

        # Create interactive scatter plot
        fig = px.scatter(
            umap_data,
            x="UMAP_1",
            y="UMAP_2",
            color=color_column,
            title=title,
            labels={
                "UMAP_1": "UMAP 1",
                "UMAP_2": "UMAP 2",
                color_column: legend_title
            },
            color_discrete_map=color_map
        )

        # Update plot styling
        fig.update_traces(marker=dict(size=3, opacity=0.7))
        fig.update_layout(
            width=700,
            height=500,
            autosize=False,
            showlegend=True,
            margin=dict(l=50, r=120, t=70, b=50),  # Proportionally smaller margins
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.01
            )
        )

        # Save plot as PDF
        plot_name = f"Overall_UMAP_{color_mode}"
        _save_plot_as_pdf(fig, plot_name)

        print(f"✅ Generated global UMAP plot with {len(unique_types if color_mode == 'cell_type' else unique_clusters)} {legend_title.lower()}s")

        # Return HTML snippet
        return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

    except Exception as e:
        print(f"Error loading global UMAP data from {umap_path}: {e}")
        return f"Error generating overall UMAP plot: {e}"


def _find_hierarchical_umap_file(cell_type: str) -> tuple:
    """
    Find the appropriate UMAP file using hierarchical lookup strategy with intelligent filtering.

    Returns:
        (file_path, filter_for_descendants, show_mode):
        - file_path: path to the UMAP CSV file to use
        - filter_for_descendants: whether to filter for descendants of the cell type
        - show_mode: 'exact' | 'parent' | 'root' for title generation
    """
    from ..cell_types.utils import get_subtypes

    # Strategy 1: Try exact match first (cell type was directly processed)
    exact_path = f'umaps/annotated/{cell_type}_umap_data.csv'
    if os.path.exists(exact_path):
        print(f"🎯 Found exact UMAP file for {cell_type}")
        # For exact matches, determine if we should show subtypes or filter for the specific type
        # If the user requested a specific cell type and we have its file, show all subtypes in that file
        return exact_path, False, 'exact'  # Don't filter - show all cells in this file

    # Strategy 2: Find parent cell type that was processed and contains this cell type
    # Use hierarchy system to find the ancestor that was actually processed
    parent_path = _find_parent_umap_file(cell_type)
    if parent_path:
        print(f"🔍 Found parent UMAP file for {cell_type}: {parent_path}")
        # When using parent file, we need to filter for the specific cell type and its descendants
        return parent_path, True, 'parent'

    # Strategy 3: Check if this is a root cell type by querying Neo4j (only if no parent found)
    try:
        subtypes = get_subtypes(cell_type)
        if subtypes and len(subtypes) > 0:
            # This cell type has subtypes in Neo4j, so it could be a root
            # Use "Overall cells" file for root cell types
            overall_path = 'umaps/annotated/Overall cells_umap_data.csv'
            if os.path.exists(overall_path):
                print(f"🌟 Using Overall cells file for root cell type: {cell_type}")
                return overall_path, True, 'root'
    except Exception as e:
        print(f"⚠️ Error querying subtypes for {cell_type}: {e}")

    # Strategy 4: Fallback to "Overall cells" if it exists
    overall_path = 'umaps/annotated/Overall cells_umap_data.csv'
    if os.path.exists(overall_path):
        print(f"🔄 Fallback to Overall cells file for {cell_type}")
        return overall_path, True, 'root'

    return None, False, None


def _find_parent_umap_file(target_cell_type: str) -> str:
    """
    Find a parent UMAP file that contains the target cell type using the hierarchy system.
    """
    
    # Search all available UMAP files
    umap_files = glob.glob('umaps/annotated/*_umap_data.csv')

    # Sort files to prioritize more specific parents over "Overall cells"
    # Put "Overall cells" last so more specific parents are found first
    umap_files = sorted(umap_files, key=lambda x: ("Overall" in x, x))

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
        
        # Save cell count comparison plot as PDF
        _save_plot_as_pdf(fig, f"cell_count_comparison_{plot_type}")
        
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
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist
    
    def create_single_heatmap(genes_data, direction, cell_type, conditions, cluster_genes, cluster_samples):
        """Helper function to create a single heatmap for either up or downregulated genes"""
        
        if not genes_data:
            return f"<p>No {direction} genes found for {cell_type}</p>"
        
        # Create expression matrix
        expression_matrix = []
        gene_labels = []
        
        for gene in genes_data:
            gene_row = []
            for condition in conditions:
                logfc = genes_data[gene].get(condition, 0)  # Use 0 if gene not found in condition
                # Ensure we have a valid numeric value
                if isinstance(logfc, (int, float)) and not np.isnan(logfc):
                    gene_row.append(float(logfc))
                else:
                    gene_row.append(0.0)
            expression_matrix.append(gene_row)
            gene_labels.append(str(gene))
        
        if not expression_matrix:
            return f"<p>No expression data to visualize for {direction} genes in {cell_type}</p>"
        
        # Convert to numpy array for clustering and ensure proper data types
        try:
            matrix = np.array(expression_matrix, dtype=float)
            if matrix.size == 0:
                return f"<p>No valid expression data for {direction} genes in {cell_type}</p>"
        except Exception as e:
            print(f"⚠️ Error creating matrix for {direction} genes: {e}")
            return f"<p>Error processing expression data for {direction} genes in {cell_type}</p>"
        
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
        
        # Validate clustered matrix
        if clustered_matrix.size == 0:
            return f"<p>No data available after clustering for {direction} genes in {cell_type}</p>"
            
        # Ensure all values are finite
        clustered_matrix = np.nan_to_num(clustered_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Choose appropriate colorscale based on direction
        if direction == "upregulated":
            colorscale = 'Reds'
            max_val = np.max(clustered_matrix) if clustered_matrix.size > 0 else 1
            color_range = [0, max(max_val, 0.1)]  # Ensure positive range
        else:
            colorscale = 'Blues_r'
            min_val = np.min(clustered_matrix) if clustered_matrix.size > 0 else -1
            color_range = [min(min_val, -0.1), 0]  # Ensure negative range
        
        # Create heatmap with forced discrete rendering
        try:
            fig = go.Figure(data=go.Heatmap(
                z=clustered_matrix.tolist(),  # Convert to list for better compatibility
                x=[str(c) for c in clustered_conditions],  # Ensure strings
                y=[str(g) for g in clustered_genes],  # Ensure strings
                colorscale=colorscale,
                zmin=color_range[0],
                zmax=color_range[1],
                colorbar=dict(title="Log Fold Change"),
                hoverongaps=False,
                # Force discrete rendering parameters
                connectgaps=False,  # Prevent interpolation between gaps
                showscale=True,
                transpose=False,  # Ensure proper orientation
                hovertemplate='<b>Gene:</b> %{y}<br>' +
                             '<b>Condition:</b> %{x}<br>' +
                             '<b>Log FC:</b> %{z:.2f}<br>' +
                             '<extra></extra>'
            ))
        except Exception as e:
            print(f"⚠️ Error creating heatmap figure for {direction}: {e}")
            return f"<p>Error creating heatmap figure for {direction} genes in {cell_type}: {e}</p>"
        
        # Update layout with consistent rendering settings
        fig.update_layout(
            title=f'{direction.title()} Genes - {cell_type}<br><sub>Top {len(clustered_genes)} {direction} genes</sub>',
            xaxis_title="Conditions",
            yaxis_title="Genes",
            width=1200,  # Fixed width for consistency across formats
            height=max(400, len(clustered_genes) * 25 + 150),
            font=dict(size=10),
            margin=dict(l=150, r=50, t=100, b=50),
            xaxis=dict(
                tickangle=45,
                tickfont=dict(size=10),
                type='category'  # Ensure categorical axis
            ),
            yaxis=dict(
                tickfont=dict(size=9),
                autorange='reversed',
                type='category'  # Ensure categorical axis
            ),
            # Settings for consistent rendering
            autosize=False,  # Use fixed dimensions
            showlegend=False,
            plot_bgcolor='white',  # Consistent background
            paper_bgcolor='white'
        )
        
        # Save heatmap as PDF (non-blocking)
        try:
            _save_plot_as_pdf(fig, f"DEA_{cell_type}_{direction}_heatmap")
        except Exception as e:
            print(f"⚠️ PDF save failed for {direction} heatmap: {e}")
        
        # Generate HTML with improved configuration
        try:
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
        except Exception as e:
            print(f"⚠️ Error generating HTML for {direction} heatmap: {e}")
            return f"<p>Error generating HTML for {direction} genes in {cell_type}: {e}</p>"

    try:
        # Standardize cell type name
        cell_type = unified_cell_type_handler(cell_type)
        
        # Look for DEA results files - try multiple possible locations
        
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
        
        if upregulated_html and not (upregulated_html.startswith("<p>No") or upregulated_html.startswith("Error") or len(upregulated_html) < 100):
            valid_plots.append({
                "type": "upregulated_heatmap",
                "title": f"Upregulated Genes - {cell_type}",
                "html": upregulated_html
            })
        
        if downregulated_html and not (downregulated_html.startswith("<p>No") or downregulated_html.startswith("Error") or len(downregulated_html) < 100):
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


def display_feature_plot(adata, genes: List[str], cell_type_filter: Optional[str] = None) -> str:
    """
    Generate interactive feature plots showing gene expression on UMAP using workflow data.

    Args:
        adata: Live AnnData object from workflow (contains current expression data)
        genes: List of gene names to plot
        cell_type_filter: Optional cell type filter for subset visualization

    Returns:
        HTML string containing interactive Plotly feature plots
    """
    print(f"🧬 Creating feature plots for genes: {genes}")

    try:
        # Always use Overall cells UMAP for the complete landscape
        umap_path = 'umaps/annotated/Overall cells_umap_data.csv'

        if not os.path.exists(umap_path):
            return f"UMAP data not available. Please ensure cell processing has been completed."

        # Load UMAP coordinates and metadata
        print(f"📊 Loading UMAP coordinates from: {umap_path}")
        umap_data = pd.read_csv(umap_path)
        print(f"✅ Loaded UMAP data: {len(umap_data)} cells")

        # Filter genes that exist in the dataset
        available_genes = [gene for gene in genes if gene in adata.var_names]
        missing_genes = [gene for gene in genes if gene not in adata.var_names]

        if missing_genes:
            print(f"⚠️ Genes not found in dataset: {missing_genes}")

        if not available_genes:
            return f"Error: None of the requested genes found in dataset: {genes}"

        print(f"✅ Using genes: {available_genes}")

        # Get cell barcodes that exist in both adata and UMAP data
        adata_barcodes = set(adata.obs.index)

        # Handle barcode matching between adata and CSV
        if 'barcode' in umap_data.columns:
            csv_barcodes = set(umap_data['barcode'].dropna())
            barcode_col = 'barcode'
        else:
            # If no barcode column, assume row order matches
            csv_barcodes = set(umap_data.index.astype(str))
            barcode_col = None

        # Find common cells
        if barcode_col:
            common_barcodes = adata_barcodes.intersection(csv_barcodes)
            if len(common_barcodes) == 0:
                # Try cleaning barcode suffixes
                csv_barcodes_clean = {str(bc).split('-')[0] for bc in csv_barcodes if pd.notna(bc)}
                common_barcodes = adata_barcodes.intersection(csv_barcodes_clean)
                if len(common_barcodes) > 0:
                    umap_data['barcode_clean'] = umap_data['barcode'].astype(str).str.split('-').str[0]
                    barcode_col = 'barcode_clean'
        else:
            # Use first N cells that match between datasets
            common_barcodes = list(adata_barcodes)[:len(umap_data)]

        if len(common_barcodes) == 0:
            return "Error: No common cells found between expression data and UMAP coordinates."

        print(f"📊 Found {len(common_barcodes)} common cells for plotting")

        # Prepare plotting data
        plot_data = []

        for i, (_, umap_row) in enumerate(umap_data.iterrows()):
            if barcode_col and barcode_col in umap_row:
                barcode = str(umap_row[barcode_col])
            else:
                # Use row index matching
                if i < len(adata.obs):
                    barcode = adata.obs.index[i]
                else:
                    continue

            if barcode not in adata.obs.index:
                continue

            # Get cell metadata
            cell_data = {
                'barcode': barcode,
                'UMAP_1': umap_row['UMAP_1'],
                'UMAP_2': umap_row['UMAP_2'],
                'cell_type': umap_row.get('cell_type', 'Unknown')
            }

            # Get expression values for each gene from live adata
            cell_idx = adata.obs.index.get_loc(barcode)
            for gene in available_genes:
                gene_idx = adata.var_names.get_loc(gene)
                expr_value = adata.X[cell_idx, gene_idx]

                # Handle sparse matrices
                if hasattr(expr_value, 'A1'):
                    expr_value = expr_value.A1[0]
                elif hasattr(expr_value, 'item'):
                    expr_value = expr_value.item()

                # Ensure non-negative expression values
                cell_data[f'{gene}_expression'] = max(0.0, float(expr_value))

            plot_data.append(cell_data)

        if not plot_data:
            return "Error: No valid data for plotting after processing."

        plot_df = pd.DataFrame(plot_data)

        # Apply cell type filter if specified
        if cell_type_filter:
            print(f"🔍 Applying cell type filter: '{cell_type_filter}'")
            original_count = len(plot_df)

            # Try multiple matching strategies
            filter_mask = (
                (plot_df['cell_type'].str.lower() == cell_type_filter.lower()) |
                (plot_df['cell_type'].str.contains(cell_type_filter, case=False, na=False))
            )
            plot_df = plot_df[filter_mask]

            print(f"🔍 Filtered to {len(plot_df)} cells matching '{cell_type_filter}' (from {original_count})")

        if len(plot_df) == 0:
            return f"No cells found matching the specified criteria."

        # Create feature plots
        n_genes = len(available_genes)

        if n_genes == 1:
            # Single gene plot
            gene = available_genes[0]
            expr_col = f'{gene}_expression'

            fig = px.scatter(
                plot_df,
                x='UMAP_1',
                y='UMAP_2',
                color=expr_col,
                title=f"Feature Plot: {gene}",
                labels={
                    'UMAP_1': 'UMAP 1',
                    'UMAP_2': 'UMAP 2',
                    expr_col: 'Expression'
                },
                color_continuous_scale=_seurat_feature_colorscale_to_plotly(),
                hover_data=['cell_type']
            )

            fig.update_traces(marker=dict(size=3, opacity=0.8))
            fig.update_layout(
                width=600,
                height=500,
                showlegend=False
            )

            # Update colorbar
            fig.update_coloraxes(
                colorbar_title="Expression",
                colorbar_thickness=15,
                colorbar_len=0.8
            )

            # Save plot
            _save_plot_as_pdf(fig, f"feature_plot_{gene}")

            plot_html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
            print(f"✅ Single feature plot generated for {gene}")
            return plot_html

        else:
            # Multiple genes subplot layout
            from plotly.subplots import make_subplots

            # Calculate grid dimensions
            n_cols = min(3, n_genes)
            n_rows = int(np.ceil(n_genes / n_cols))

            # Create subplots
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=available_genes,
                horizontal_spacing=0.08,
                vertical_spacing=0.12
            )

            for i, gene in enumerate(available_genes):
                row = (i // n_cols) + 1
                col = (i % n_cols) + 1
                expr_col = f'{gene}_expression'

                # Create scatter trace
                scatter = go.Scatter(
                    x=plot_df['UMAP_1'],
                    y=plot_df['UMAP_2'],
                    mode='markers',
                    marker=dict(
                        color=plot_df[expr_col],
                        colorscale=_seurat_feature_colorscale_to_plotly(),
                        size=3,
                        opacity=0.8,
                        colorbar=dict(
                            title="Expression",
                            x=1.02 if col == n_cols else None,
                            len=0.8
                        ) if i == len(available_genes) - 1 else None
                    ),
                    text=plot_df['cell_type'],
                    hovertemplate='<b>%{text}</b><br>UMAP 1: %{x}<br>UMAP 2: %{y}<br>Expression: %{marker.color}<extra></extra>',
                    showlegend=False
                )

                fig.add_trace(scatter, row=row, col=col)

                # Update subplot axes
                fig.update_xaxes(title_text="UMAP 1", row=row, col=col)
                fig.update_yaxes(title_text="UMAP 2", row=row, col=col)

            # Update layout
            fig.update_layout(
                title=f"Feature Plots: {', '.join(available_genes[:3])}{'...' if len(available_genes) > 3 else ''}",
                width=300 * n_cols,
                height=250 * n_rows + 100,
                showlegend=False
            )

            # Save plot
            _save_plot_as_pdf(fig, f"feature_plots_{'_'.join(available_genes[:3])}")

            plot_html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
            print(f"✅ Multiple feature plots generated for {len(available_genes)} genes")
            return plot_html

    except Exception as e:
        error_msg = f"Error generating feature plots: {e}"
        print(f"❌ {error_msg}")
        return error_msg


def display_violin_plot(adata, cell_type: str, genes: List[str]) -> str:
    """
    Generate interactive violin plots using live adata directly (no CSV files needed).

    Args:
        adata: Live AnnData object from workflow
        cell_type: Cell type to analyze
        genes: List of gene names to plot
    Returns:
        HTML string containing interactive Plotly violin plots
    """
    print(f"🎻 Creating violin plots for {cell_type} with genes: {genes}")

    try:
        # Filter cells directly from live adata for the specified cell type
        if 'cell_type' not in adata.obs.columns:
            return f"Error: No cell_type information found in the dataset."

        # Filter adata for the target cell type
        cell_mask = adata.obs['cell_type'] == cell_type
        if cell_mask.sum() == 0:
            return f"No {cell_type} cells found in the dataset."

        print(f"✅ Found {cell_mask.sum()} {cell_type} cells in the dataset")

        # Filter genes that exist in the dataset
        available_genes = [gene for gene in genes if gene in adata.var_names]
        missing_genes = [gene for gene in genes if gene not in adata.var_names]

        if missing_genes:
            print(f"⚠️ Genes not found in dataset: {missing_genes}")

        if not available_genes:
            return f"Error: None of the requested genes found in dataset: {genes}"

        print(f"✅ Using genes: {available_genes}")

        # Get subset of data for target cell type
        cell_subset = adata[cell_mask]

        # Prepare plotting data
        plot_data = []

        for gene in available_genes:
            gene_idx = adata.var_names.get_loc(gene)

            # Get expression values for this gene across all cells of this type
            gene_expression = cell_subset.X[:, gene_idx]

            # Handle sparse matrices
            if hasattr(gene_expression, 'toarray'):
                gene_expression = gene_expression.toarray().flatten()
            elif hasattr(gene_expression, 'A1'):
                gene_expression = gene_expression.A1

            # Get metadata for these cells
            cell_leiden = cell_subset.obs.get('leiden', ['Unknown'] * len(cell_subset))
            cell_conditions = cell_subset.obs.get('Exp_sample_category', ['Unknown'] * len(cell_subset))

            # Add data for each cell
            for i, expr_val in enumerate(gene_expression):
                leiden_val = cell_leiden.iloc[i] if hasattr(cell_leiden, 'iloc') else cell_leiden[i] if i < len(cell_leiden) else 'Unknown'
                condition_val = cell_conditions.iloc[i] if hasattr(cell_conditions, 'iloc') else cell_conditions[i] if i < len(cell_conditions) else 'Unknown'

                plot_data.append({
                    'gene': gene,
                    'expression': float(expr_val),
                    'leiden': str(leiden_val),
                    'condition': str(condition_val),
                    'cell_type': cell_type
                })

        if not plot_data:
            return f"No expression data available for {cell_type} with genes {genes}"

        # Convert to DataFrame for plotting
        plot_df = pd.DataFrame(plot_data)
        print(f"📊 Prepared {len(plot_df)} data points for violin plots")

        # Create violin plot for each gene
        figures = []
        for gene in available_genes:
            gene_data = plot_df[plot_df['gene'] == gene]

            # Define a set of distinct colors for any conditions
            predefined_colors = [
                '#4682B4',  # Steel Blue
                '#FF6347',  # Tomato Red
                '#32CD32',  # Lime Green
                '#FFD700',  # Gold
                '#9370DB',  # Medium Purple
                '#FF69B4',  # Hot Pink
                '#20B2AA',  # Light Sea Green
                '#F4A460'   # Sandy Brown
            ]

            # Get unique conditions and map them to colors
            unique_conditions = sorted(gene_data['condition'].unique())
            color_map = {}
            for i, condition in enumerate(unique_conditions):
                color_map[condition] = predefined_colors[i % len(predefined_colors)]

            print(f"📊 Conditions: {unique_conditions}")
            print(f"📊 Color mapping: {color_map}")

            # Create figure manually using go.Violin for better control

            fig = go.Figure()

            # Get sorted leiden clusters for proper x-axis ordering
            leiden_clusters = sorted(gene_data['leiden'].unique(), key=lambda x: int(x) if x.isdigit() else float('inf'))

            # Create violin for each condition with proper grouping
            for condition in unique_conditions:
                condition_data = gene_data[gene_data['condition'] == condition]
                color = color_map[condition]

                # Create one violin trace per condition (let Plotly handle the grouping by leiden)
                fig.add_trace(go.Violin(
                    y=condition_data['expression'],
                    x=condition_data['leiden'],
                    name=condition,
                    line_color='black',
                    fillcolor=color,
                    opacity=0.7,
                    box_visible=False,  # Don't show box plots
                    meanline_visible=True,  # Show only mean line
                    points=False,
                    spanmode='hard'  # Don't extend beyond data range
                ))

            # Update layout to match matplotlib style
            fig.update_layout(
                title=f'{gene} Expression in {cell_type} (by Leiden Cluster)',
                xaxis_title='Leiden Cluster',
                yaxis_title=f'{gene} Expression',
                violinmode='group',  # Group violins side by side
                template='plotly_white',
                xaxis=dict(type='category')  # Treat leiden as categorical
            )

            # Set y-axis minimum to 0
            fig.update_yaxes(range=[0, None])

            # Save plot as PDF
            _save_plot_as_pdf(fig, f"violin_plot_{gene}_{cell_type}")

            figures.append(fig)

        # Combine all figures into one HTML
        html_parts = []
        for fig in figures:
            html_parts.append(pio.to_html(fig, full_html=False, include_plotlyjs="cdn"))

        print(f"✅ Violin plots generated for {len(available_genes)} genes")
        return "<div>" + "".join(html_parts) + "</div>"

    except Exception as e:
        print(f"❌ Error creating violin plot: {e}")
        return f"Error creating violin plot: {e}"


def display_cell_count_stacked_plot(adata, cell_types: List[str]) -> str:
    """
    Create stacked bar plot showing cell counts across conditions for specified cell types.
    Based on create_cell_count_plot.py implementation.

    Args:
        adata: Live AnnData object from workflow
        cell_types: List of cell type names to include in the plot

    Returns:
        HTML string containing interactive Plotly stacked bar plot
    """
    from plotly.subplots import make_subplots

    print(f"📊 Creating stacked cell count plot for cell types: {cell_types}")

    try:
        # Check if required metadata exists
        if 'cell_type' not in adata.obs.columns:
            return f"Error: No cell_type information found in the dataset."

        if 'Exp_sample_category' not in adata.obs.columns:
            return f"Error: No Exp_sample_category information found in the dataset."

        # Prepare data for plotting
        plot_data = []

        # Extract sample/patient and treatment information
        for idx, row in adata.obs.iterrows():
            cell_type = row.get('cell_type', 'Unknown')
            exp_sample = str(row.get('Exp_sample_category', ''))

            # Skip if not one of the requested cell types
            if cell_type not in cell_types:
                continue

            # Extract patient and treatment information
            patient = 'Unknown'
            treatment = 'Unknown'

            # Parse sample category to extract patient and treatment
            if 'patient' in exp_sample.lower() or 'p' in exp_sample.lower():
                # Try to extract patient number
                import re
                patient_match = re.search(r'[pP](?:atient)?[_\s]*(\d+)', exp_sample)
                if patient_match:
                    patient = f"Patient {patient_match.group(1)}"

                # Extract treatment
                if '_pre' in exp_sample.lower() or 'pre' in exp_sample.lower():
                    treatment = 'Pre-treatment'
                elif '_post' in exp_sample.lower() or 'post' in exp_sample.lower():
                    treatment = 'Post-treatment'

            plot_data.append({
                'cell_type': cell_type,
                'patient': patient,
                'treatment': treatment,
                'exp_sample': exp_sample
            })

        if not plot_data:
            return f"No data found for cell types: {cell_types}"

        # Convert to DataFrame and aggregate counts
        df = pd.DataFrame(plot_data)

        # Count cells by cell type, patient, and treatment
        count_df = df.groupby(['cell_type', 'patient', 'treatment']).size().reset_index(name='count')

        print(f"📊 Found data for {len(count_df)} cell type-patient-treatment combinations")

        # Map cell types to display names (consistent with create_cell_count_plot.py)
        cell_type_mapping = {
            'Macrophage': 'Macrophage',
            'Macrophages': 'Macrophage',
            'B cell': 'B cell',
            'B Cells': 'B cell',
            'T cell': 'T cell',
            'T Cells': 'T cell',
            'Microglial cell': 'Microglial cell',
            'Microglial Cells': 'Microglial cell',
        }

        # Standardize cell type names
        count_df['display_cell_type'] = count_df['cell_type'].map(
            lambda x: cell_type_mapping.get(x, x)
        )

        # Get unique patients and create condition combinations
        patients = sorted([p for p in count_df['patient'].unique() if p != 'Unknown'])
        treatments = ['Pre-treatment', 'Post-treatment']

        # Create condition labels (p1_pre, p1_post, etc.)
        conditions = []
        condition_labels = []
        for patient in patients:
            for treatment in treatments:
                patient_num = patient.split()[-1] if 'Patient' in patient else patient
                treatment_short = 'Pre' if treatment == 'Pre-treatment' else 'Post'
                condition_key = f"p{patient_num}_{treatment_short.lower()}"
                conditions.append(condition_key)
                condition_labels.append(f"{patient} {treatment_short}")

        # Get unique display cell types and sort them
        display_cell_types = sorted(count_df['display_cell_type'].unique())

        # Prepare data matrix: cell_types x conditions
        data_matrix = np.zeros((len(display_cell_types), len(conditions)))

        # Fill the data matrix
        for i, cell_type in enumerate(display_cell_types):
            for j, condition in enumerate(conditions):
                # Parse condition to get patient and treatment
                condition_parts = condition.split('_')
                patient_num = condition_parts[0][1:]  # Remove 'p' prefix
                treatment = 'Pre-treatment' if condition_parts[1] == 'pre' else 'Post-treatment'
                patient = f"Patient {patient_num}"

                # Get count from dataframe
                mask = (
                    (count_df['display_cell_type'] == cell_type) &
                    (count_df['patient'] == patient) &
                    (count_df['treatment'] == treatment)
                )
                cell_data = count_df[mask]
                if not cell_data.empty:
                    data_matrix[i, j] = cell_data['count'].iloc[0]

        # Get Seurat discrete palette for conditions
        seurat_colors = seurat_discrete(len(conditions))
        # Convert RGB tuples to hex strings for Plotly compatibility
        seurat_colors_hex = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in seurat_colors]

        # Create stacked bar plot
        fig = go.Figure()

        # Add each condition as a layer in the stack
        for j, (condition, color, label) in enumerate(zip(conditions, seurat_colors_hex, condition_labels)):
            fig.add_trace(go.Bar(
                name=label,
                x=display_cell_types,
                y=data_matrix[:, j],
                marker_color=color,
                opacity=0.8
            ))

        # Update layout
        fig.update_layout(
            title="Cell Type Count Comparison Across Conditions",
            xaxis_title="Cell Type",
            yaxis_title="Cell Count",
            barmode='stack',
            template='plotly_white',
            font=dict(size=12),
            width=800,
            height=600,
            plot_bgcolor='#f0f0f0'
        )

        # Set y-axis to start from 0
        fig.update_yaxes(range=[0, None])

        # Save plot as PDF
        _save_plot_as_pdf(fig, f"cell_count_stacked_plot_{'_'.join(display_cell_types[:3])}")

        # Return HTML
        html_output = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
        print(f"✅ Cell count stacked plot generated for {len(display_cell_types)} cell types")
        return html_output

    except Exception as e:
        print(f"❌ Error creating cell count plot: {e}")
        return f"Error creating cell count plot: {e}"
