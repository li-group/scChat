import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import json
from plotly.utils import PlotlyJSONEncoder
import numpy as np

# def display_dotplot() -> str:
#     """
#     Creates a dotplot visualization from CSV data and returns its JSON representation.
#     """
#     dot_plot_data = pd.read_csv("/Users/ashleyjvarghese/Desktop/scChat-2/schatbot/runtime_data/basic_data/Overall cells_dot_plot_data.csv")
#     fig = px.scatter(
#         dot_plot_data,
#         x='gene',
#         y='leiden',
#         size='expression',
#         color='expression',
#         title='Dot Plot',
#         labels={'gene': 'Gene', 'leiden': 'Cluster', 'expression': 'Expression Level'},
#         color_continuous_scale='Blues'
#     )
#     fig.update_traces(marker=dict(opacity=0.8))
#     fig.update_layout(width=1200, height=800, autosize=True)
#     return fig.to_json()

import pandas as pd
import plotly.express as px
import plotly.io as pio
import os

# def display_reactome_barplot(cell_type: str, top_n: int = 10) -> str:
#     """
#     Reads reactome summary CSV, picks top_n by p-value, and returns
#     a barplot HTML snippet (-log10 p-value).
#     """
#     folder = "schatbot/enrichment/reactome"
#     csv_path = os.path.join(
#         folder, f"results_summary_{cell_type}.csv"
#     )
#     df = pd.read_csv(csv_path)
#     # add an -log10(p_value) column for plotting
#     df["minus_log10_p"] = -np.log10(df["p_value"])
#     top_df = df.nsmallest(top_n, "p_value")
#     fig = px.bar(
#         top_df,
#         x="minus_log10_p",
#         y="Term",
#         orientation="h",
#         title=f"Top {top_n} Reactome Pathways ({cell_type})",
#         labels={"minus_log10_p":"-log10(p-value)", "Term":"Pathway"},
#     )
#     fig.update_layout(height=top_n*50 + 200, yaxis={"categoryorder":"total ascending"})
#     return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

def display_enrichment_barplot(
    analysis: str,
    cell_type: str,
    top_n: int = 10,
    domain: str = None
) -> str:
    """
    Generic barplot of top_n enriched terms for a given analysis and cell type.
    analysis: "reactome", "kegg", "gsea", or "go"
    domain: required if analysis=="go" (one of "BP","MF","CC")
    """
    base = analysis.lower()
    if base == "go":
        if domain is None:
            raise ValueError("`domain` must be provided for GO.")
        sub = domain.lower()
        folder = f"schatbot/enrichment/go"
        # fname = f"go_{sub}_results_summary_{cell_type}.csv"
        fname = f"results_summary_{cell_type}.csv"
    else:
        folder = f"schatbot/enrichment/{base}"
        # fname = f"{base}_results_summary_{cell_type}.csv"
        fname = f"results_summary_{cell_type}.csv"
    path = os.path.join(folder, fname)
    df = pd.read_csv(path)
    df["minus_log10_p"] = -np.log10(df["p_value"])
    top = df.nsmallest(top_n, "p_value")

    fig = px.bar(
        top,
        x="minus_log10_p",
        y="Term",
        orientation="h",
        title=f"Top {top_n} {analysis.capitalize()} Terms ({cell_type})",
        labels={"minus_log10_p": "-log10(p-value)", "Term": "Term"},
    )
    fig.update_layout(
        height=top_n * 40 + 200,
        yaxis={"categoryorder": "total ascending"}
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")


def display_enrichment_dotplot(
    analysis: str,
    cell_type: str,
    top_n: int = 10,
    domain: str = None
) -> str:
    """
    Generic dotplot of avg_log2fc vs. top enriched terms for a given analysis.
    analysis: "reactome", "kegg", "gsea", or "go"
    domain: required if analysis=="go" (one of "BP","MF","CC")
    """
    base = analysis.lower()
    if base == "go":
        if domain is None:
            raise ValueError("`domain` must be provided for GO.")
        folder = "schatbot/enrichment/go"
        fname = f"go_{domain.lower()}_results_summary_{cell_type}.csv"
    else:
        folder = f"schatbot/enrichment/{base}"
        # fname = f"{base}_results_summary_{cell_type}.csv"
        fname  = f"results_summary_{cell_type}.csv"
    path = os.path.join(folder, fname)
    df = pd.read_csv(path)
    df["minus_log10_p"] = -np.log10(df["p_value"])
    top = df.nsmallest(top_n, "p_value")

    fig = px.scatter(
        top,
        x="avg_log2fc",
        y="Term",
        size="intersection_size",
        color="minus_log10_p",
        title=f"Top {top_n} {analysis.capitalize()} Terms ({cell_type})",
        labels={
            "avg_log2fc": "Avg log₂ FC",
            "minus_log10_p": "-log10(p-value)",
            "Term": "Term",
            "intersection_size": "Gene Count"
        },
        size_max=20,
        orientation="h"
    )
    fig.update_layout(
        height=top_n * 40 + 200,
        yaxis={"categoryorder": "total ascending"}
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")


def display_dotplot() -> str:
    """
    Creates a dotplot visualization from CSV data and returns an HTML snippet
    containing the Plotly figure.
    """
    dot_plot_data = pd.read_csv("/Users/ashleyjvarghese/Desktop/scChat-2/schatbot/runtime_data/basic_data/Overall cells_dot_plot_data.csv")
    fig = px.scatter(
        dot_plot_data,
        x='gene',
        y='leiden',
        size='expression',
        color='expression',
        title='Dot Plot',
        labels={'gene': 'Gene', 'leiden': 'Cluster', 'expression': 'Expression Level'},
        color_continuous_scale='Blues'
    )
    fig.update_traces(marker=dict(opacity=0.8))
    fig.update_layout(width=1200, height=800, autosize=True)
    # Return the HTML snippet for embedding the interactive figure
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    return plot_html


# def display_cell_population_change(cell_type: str) -> str:
#     """
#     Reads cell population change CSV data for the specified cell type
#     and returns a Plotly bar chart JSON.
#     """
#     cell_type_formatted = cell_type.split()[0].capitalize() + " cells"
#     filename = f'schatbot/cell_population_change/{cell_type_formatted}_cell_population_change.csv'
#     cell_counts = pd.read_csv(filename)
#     fig = px.bar(
#         cell_counts,
#         x="patient_name",
#         y="percentage",
#         color="cell_type",
#         title="Cell Population Change",
#         labels={"patient_name": "Patient Name", "percentage": "Percentage of Cell Type"}
#     )
#     fig.update_layout(width=1200, height=800, autosize=True, showlegend=True)
#     return fig.to_json()

def display_cell_population_change(cell_type: str) -> str:
    """
    Reads cell population change CSV data for the specified cell type
    and returns an HTML snippet of a Plotly bar chart.
    """
    cell_type_formatted = cell_type.split()[0].capitalize() + " cells"
    filename = f'schatbot/cell_population_change/{cell_type_formatted}_cell_population_change.csv'
    cell_counts = pd.read_csv(filename)
    fig = px.bar(
        cell_counts,
        x="patient_name",
        y="percentage",
        color="cell_type",
        title="Cell Population Change",
        labels={"patient_name": "Patient Name", "percentage": "Percentage of Cell Type"}
    )
    fig.update_layout(width=1200, height=800, autosize=True, showlegend=True)
    # Return the HTML snippet for embedding the interactive figure
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    return plot_html

# def display_umap(cell_type: str) -> str:
#     """
#     Reads UMAP data for the specified cell type from process_cell_data and returns a scatter plot JSON.
#     """
#     print ("T1")
#     cell_type_formatted = cell_type.split()[0].capitalize() + " cells"
#     print (cell_type_formatted)
#     umap_data = pd.read_csv(f'schatbot/runtime_data/process_cell_data/{cell_type_formatted}_umap_data.csv')
#     print ("T2")
#     if cell_type_formatted != "Overall cells":
#         print ("T3")
#         umap_data['original_cell_type'] = umap_data['cell_type']
#         umap_data['cell_type'] = 'Unknown'
#         print ("T4")
#     fig = px.scatter(
#         umap_data,
#         x="UMAP_1",
#         y="UMAP_2",
#         color="leiden",
#         symbol="patient_name",
#         title="T Cells UMAP Plot",
#         labels={"UMAP_1": "UMAP 1", "UMAP_2": "UMAP 2"}
#     )
#     print ("T5")
#     fig.update_traces(marker=dict(size=5, opacity=0.8))
#     print ("T6")
#     fig.update_layout(width=1200, height=800, autosize=True, showlegend=False)
#     print ("T7")
#     custom_legend = go.Scatter(
#         x=[None], y=[None],
#         mode='markers',
#         marker=dict(size=10, color='rgba(0,0,0,0)'),
#         legendgroup="Unknown",
#         showlegend=True,
#         name="Unknown"
#     )
#     print ("T8")
#     fig.add_trace(custom_legend)
#     print ("T9")
#     return fig.to_json()
import plotly.io as pio


# def display_umap(cell_type: str) -> str:
#     cell_type_formatted = cell_type.split()[0].capitalize() + " cells"
#     umap_data = pd.read_csv(f'schatbot/runtime_data/process_cell_data/{cell_type_formatted}_umap_data.csv')
#     if cell_type_formatted != "Overall cells":
#         umap_data['original_cell_type'] = umap_data['cell_type']
#         umap_data['cell_type'] = 'Unknown'
#     fig = px.scatter(
#         umap_data,
#         x="UMAP_1",
#         y="UMAP_2",
#         color="leiden",
#         symbol="patient_name",
#         title="T Cells UMAP Plot",
#         labels={"UMAP_1": "UMAP 1", "UMAP_2": "UMAP 2"}
#     )
#     fig.update_traces(marker=dict(size=5, opacity=0.8))
#     fig.update_layout(width=1200, height=800, autosize=True, showlegend=False)
#     custom_legend = go.Scatter(
#         x=[None], y=[None],
#         mode='markers',
#         marker=dict(size=10, color='rgba(0,0,0,0)'),
#         legendgroup="Unknown",
#         showlegend=True,
#         name="Unknown"
#     )
#     fig.add_trace(custom_legend)
    
#     # Use plotly.io.to_html instead of pyo.plot
#     plot_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
#     return plot_html

def display_umap(cell_type: str) -> str:
    cell_type_formatted = cell_type.split()[0].capitalize() + " cells"
    umap_data = pd.read_csv(f'schatbot/runtime_data/process_cell_data/{cell_type_formatted}_umap_data.csv')
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



# def display_processed_umap(cell_type: str) -> str:
#     """
#     Reads annotated UMAP data for the specified cell type from the umaps folder and returns a scatter plot JSON.
#     """
#     cell_type_cell = cell_type.split()[0].capitalize() + " cell"
#     cell_type_formatted = cell_type.split()[0].capitalize() + " cells"
#     import os
#     if os.path.exists(f'umaps/{cell_type_formatted}_annotated_umap_data.csv'):
#         umap_data = pd.read_csv(f'umaps/{cell_type_formatted}_annotated_umap_data.csv')
#     else:
#         umap_data = pd.read_csv(f'umaps/{cell_type_cell}_annotated_umap_data.csv')
#     fig = px.scatter(
#         umap_data,
#         x="UMAP_1",
#         y="UMAP_2",
#         color="cell_type",
#         symbol="patient_name",
#         title=f'{cell_type_formatted} UMAP Plot',
#         labels={"UMAP_1": "UMAP 1", "UMAP_2": "UMAP 2"}
#     )
#     fig.update_traces(marker=dict(size=5, opacity=0.8))
#     fig.update_layout(width=1200, height=800, autosize=True, showlegend=True)
#     return fig.to_json()

# def display_processed_umap(cell_type: str) -> str:
#     cell_type_cell = cell_type.split()[0].capitalize() + " cell"
#     cell_type_formatted = cell_type.split()[0].capitalize() + " cells"
#     import os
#     if os.path.exists(f'umaps/annotated/{cell_type_formatted}_umap_data.csv'):
#         print (f'umaps/annotated/{cell_type_formatted}_umap_data.csv')
#         umap_data = pd.read_csv(f'umaps/annotated/{cell_type_formatted}_umap_data.csv')
#     else:
#         print (f'umaps/annotated/{cell_type_formatted}_umap_data.csv')
#         umap_data = pd.read_csv(f'umaps/annotated/{cell_type_cell}_umap_data.csv')
#     fig = px.scatter(
#         umap_data,
#         x="UMAP_1",
#         y="UMAP_2",
#         color="cell_type",
#         # symbol="patient_name",
#         title=f'{cell_type_formatted} UMAP Plot',
#         labels={"UMAP_1": "UMAP 1", "UMAP_2": "UMAP 2"}
#     )
#     fig.update_traces(marker=dict(size=5, opacity=0.8))
#     fig.update_layout(width=1200, height=800, autosize=True, showlegend=True)
#     # Return an HTML snippet of the Plotly figure
#     plot_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
#     return plot_html
from .sc_analysis import unified_cell_type_handler
def display_processed_umap(cell_type: str) -> str:
    import os
    import pandas as pd
    import plotly.express as px
    import plotly.io as pio

    # standardize your cell type into the form used by process_cells
    std = unified_cell_type_handler(cell_type)  # e.g. "Monocytes"

    # possible filenames in order
    candidates = [
        f'umaps/annotated/{std}_umap_data.csv',         # new pattern
        f'umaps/annotated/{std} cells_umap_data.csv',   # old plural
        f'umaps/annotated/{std} cell_umap_data.csv'     # old singular
    ]

    # pick the first one that exists
    umap_path = next((p for p in candidates if os.path.exists(p)), None)

    # if none found, scan the directory for any "{std}*_umap_data.csv"
    if umap_path is None:
        directory = 'umaps/annotated'
        if os.path.isdir(directory):
            for fn in os.listdir(directory):
                if fn.startswith(std) and fn.endswith('_umap_data.csv'):
                    umap_path = os.path.join(directory, fn)
                    break

    if umap_path is None:
        print(f"Warning: could not find an annotated UMAP file for '{cell_type}'")
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

def display_gsea_dotplot() -> str:
    """
    Creates a GSEA dot plot from CSV data and returns its JSON representation.
    """
    column_names = ['Term', 'fdr', 'es', 'nes', 'Rank Metric', 'Enrichment Score']
    dot_plot_data = pd.read_csv("gsea_plots/gsea_plot_data.csv", header=None, names=column_names, skiprows=1)
    dot_plot_data = dot_plot_data.dropna(subset=['Rank Metric', 'Enrichment Score'])
    dot_plot_data['Rank Metric'] = pd.to_numeric(dot_plot_data['Rank Metric'])
    dot_plot_data['Enrichment Score'] = pd.to_numeric(dot_plot_data['Enrichment Score'])
    dot_plot_data['Enrichment Score'] = dot_plot_data['Enrichment Score'].apply(lambda x: max(abs(x), 1))
    fig = px.scatter(
        dot_plot_data,
        x='Rank Metric',
        y='Enrichment Score',
        size='Enrichment Score',
        color='Enrichment Score',
        title='GSEA Dot Plot',
        labels={'Rank Metric': 'Rank Metric', 'Enrichment Score': 'Enrichment Score'},
        color_continuous_scale='Blues'
    )
    fig.update_traces(marker=dict(opacity=0.8))
    fig.update_layout(width=1200, height=800, autosize=True,
                      xaxis=dict(title="Rank Metric"),
                      yaxis=dict(title="Enrichment Score"))
    return fig.to_json()

def display_cell_type_composition() -> str:
    """
    Generates a dendrogram of cell type composition and returns its JSON representation.
    """
    dendrogram_data = pd.read_csv("basic_data/dendrogram_data.csv")
    fig = ff.create_dendrogram(dendrogram_data.values, orientation='left')
    fig.update_layout(title='Dendrogram', xaxis_title='Distance', yaxis_title='Clusters',
                      width=1200, height=800, autosize=True)
    return fig.to_json()

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