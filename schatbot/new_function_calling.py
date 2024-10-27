#done 23 Aug
import os
import json
import openai
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import pickle
import os
import json
import openai
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, ChatMessage
import scvi
import scanpy as sc
from matplotlib.pyplot import rc_context
sc.set_figure_params(dpi=100)
scvi._settings.seed = 0
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import scvi
from scvi.model import SCVI
import plotly.express as px
import json
import pandas as pd
import plotly.graph_objects as go
import ast
import matplotlib
import warnings
import matplotlib
import scanpy as sc
from scvi.model import SCVI
from matplotlib import rc_context
import pandas as pd
import numpy as np
# Suppress all warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

current_adata = None
base_annotated_adata = None
adata = None
resolution = 0.5
sample_mapping = None
SGP = None
global function_flag
function_flag = False
global display_flag
display_flag = False

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")



data = pd.DataFrame({
    'feature1': [0.1, 0.2, 0.3, 0.4],
    'feature2': [0.2, 0.1, 0.4, 0.3],
    'label': ['A', 'B', 'C', 'D']
})

import os
import shutil

def clear_directory(directory_path):
    if not os.path.exists(directory_path):
        return
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file or symbolic link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the directory and all its contents
        except Exception as e:
            print ("Not found")


def find_and_load_sample_mapping(directory):
    global sample_mapping

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'sample_mapping.json':
                file_path = os.path.join(root, file)
                
                with open(file_path, 'r') as f:
                    sample_mapping = json.load(f)
                
                print(f"'sample_mapping.json' found and loaded from {file_path}")
                return sample_mapping
    
    # If the file wasn't found
    return None


    

        
def get_umap_data():
    try:
        # Dummy data for illustration; replace with your actual data
        data = pd.DataFrame({
            'feature1': [0.1, 0.2, 0.3, 0.4],
            'feature2': [0.2, 0.1, 0.4, 0.3],
            'label': ['A', 'B', 'C', 'D']
        })

        # Set n_neighbors to a sensible value based on the size of your dataset
        n_neighbors = min(15, len(data) - 1)

        reducer = umap.UMAP(n_neighbors=n_neighbors)
        embedding = reducer.fit_transform(data[['feature1', 'feature2']])
        data['umap1'] = embedding[:, 0]
        data['umap2'] = embedding[:, 1]

        return data
    except Exception as e:
        raise




def display_cell_population_change(cell_type):
    # Load the CSV file
    global display_flag
    display_flag = True

    cell_type = cell_type.split()[0].capitalize() + " cells"    
    filename = f'schatbot/cell_population_change/{cell_type}_cell_population_change.csv'
    cell_counts = pd.read_csv(filename)

    # Create the Plotly plot
    fig = px.bar(
        cell_counts,
        x="patient_name",
        y="percentage",
        color="cell_type",
        title=f"Cell Population Change",
        labels={"patient_name": "Patient Name", "percentage": "Percentage of Cell Type"}
    )

    # Update layout for better visualization
    fig.update_layout(
        width=1200,  # Set the width of the plot
        height=800,  # Set the height of the plot
        autosize=True,
        showlegend=True  # Show the legend
    )

    # Convert the plot to JSON for frontend display
    fig_json = fig.to_json()
    return fig_json




def display_umap(cell_type):
    global display_flag
    display_flag = True

    cell_type = cell_type.split()[0].capitalize() + " cells"    
    umap_data = pd.read_csv(f'process_cell_data/{cell_type}_umap_data.csv')
    if cell_type != "Overall cells":
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
    fig.update_layout(
        width=1200,
        height=800,
        autosize=True,
        showlegend=False
    )

    custom_legend = go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='rgba(0,0,0,0)'),
        legendgroup="Unknown",
        showlegend=True,
        name="Unknown"
    )

    fig.add_trace(custom_legend)
    fig_json = fig.to_json()
    return fig_json

def display_processed_umap(cell_type):
    global display_flag
    display_flag = True
    cell_type2 = cell_type.split()[0].capitalize() + " cell"        
    cell_type = cell_type.split()[0].capitalize() + " cells"        
    umap_data = None
    if os.path.exists(f'umaps/{cell_type}_annotated_umap_data.csv'):
        umap_data = pd.read_csv(f'umaps/{cell_type}_annotated_umap_data.csv')
    else:
        umap_data = pd.read_csv(f'umaps/{cell_type2}_annotated_umap_data.csv')
    # if cell_type != "Overall cells":
    #     umap_data['original_cell_type'] = umap_data['cell_type']
    #     umap_data['cell_type'] = 'Unknown'

    fig = px.scatter(
        umap_data,
        x="UMAP_1",
        y="UMAP_2",
        color="cell_type",
        symbol="patient_name",
        title=f'{cell_type} UMAP Plot',
        labels={"UMAP_1": "UMAP 1", "UMAP_2": "UMAP 2"}
    )

    fig.update_traces(marker=dict(size=5, opacity=0.8))
    fig.update_layout(
        width=1200,
        height=800,
        autosize=True,
        showlegend=True
    )

    fig_json = fig.to_json()
    return fig_json




def display_dotplot():
    import plotly.express as px
    import pandas as pd
    
    dot_plot_data = pd.read_csv("basic_data/dot_plot_data.csv")

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
    fig.update_layout(
        width=1200,  
            height=800, 
            autosize=True,
    )
    return fig.to_json()


def display_gsea_dotplot():
    import plotly.express as px
    import pandas as pd

    # Manually define the correct column names based on your CSV's structure
    column_names = ['Term', 'fdr', 'es', 'nes', 'Rank Metric', 'Enrichment Score']

    # Load the data with the correct column names
    dot_plot_data = pd.read_csv("gsea_plots/gsea_plot_data.csv", header=None, names=column_names, skiprows=1)

    # Drop rows with missing values in 'Rank Metric' and 'Enrichment Score' (they are likely to be the actual data rows)
    dot_plot_data = dot_plot_data.dropna(subset=['Rank Metric', 'Enrichment Score'])

    # Convert necessary columns to numeric types
    dot_plot_data['Rank Metric'] = pd.to_numeric(dot_plot_data['Rank Metric'])
    dot_plot_data['Enrichment Score'] = pd.to_numeric(dot_plot_data['Enrichment Score'])

    # Filter out invalid sizes (non-positive numbers) or set a minimum size value
    dot_plot_data['Enrichment Score'] = dot_plot_data['Enrichment Score'].apply(lambda x: max(abs(x), 1))

    # Create the Plotly scatter plot
    fig = px.scatter(
        dot_plot_data,
        x='Rank Metric',
        y='Enrichment Score',
        size='Enrichment Score',  # Using filtered/enforced positive values for size
        color='Enrichment Score',
        title='GSEA Dot Plot',
        labels={'Rank Metric': 'Rank Metric', 'Enrichment Score': 'Enrichment Score'},
        color_continuous_scale='Blues'
    )

    fig.update_traces(marker=dict(opacity=0.8))
    fig.update_layout(
        width=1200,
        height=800,
        autosize=True,
        xaxis=dict(title="Rank Metric"),
        yaxis=dict(title="Enrichment Score")
    )

    return fig.to_json()


def display_cell_type_composition():
    import plotly.figure_factory as ff
    import pandas as pd
    
    dendrogram_data = pd.read_csv("basic_data/dendrogram_data.csv")

    fig = ff.create_dendrogram(dendrogram_data.values, orientation='left')
    fig.update_layout(title='Dendrogram', xaxis_title='Distance', yaxis_title='Clusters')
    fig.update_layout(
        
        width=1200,  
            height=800, 
            autosize=True,
    )
    # fig.show()
    return fig.to_json()

annotated_adata = None
global conversation_history2

def label_clusters(cell_type):
    global adata
    global annotated_adata
    global conversation_history2
    global current_adata
    global base_annotated_adata
    adata2 = adata.copy()
    standardized_cell_type3 = cell_type.split()[0].capitalize()     
    standardized_cell_type2 = cell_type.split()[0].capitalize() + " cell"        
    standardized_cell_type = cell_type.split()[0].capitalize() + " cells"        
    last_message = conversation_history2[-2]['content']
    try:
        start_idx = last_message.find("{")
        end_idx = last_message.rfind("}") + 1
        str_map = last_message[start_idx:end_idx]        
        map2 = ast.literal_eval(str_map)
        map2 = {str(key): value for key, value in map2.items()}        
        if standardized_cell_type == "Overall cells" or standardized_cell_type2 == "Overall cell":
            adata2.obs['cell_type'] = 'Unknown'
            for group, cell_type in map2.items():
                adata2.obs.loc[adata2.obs['leiden'] == group, 'cell_type'] = cell_type
            adata2.obs[['UMAP_1', 'UMAP_2', 'leiden', 'patient_name', 'cell_type']].to_csv(f'umaps/{standardized_cell_type}_annotated_umap_data.csv', index=False)
            annotated_adata = adata2.copy()
            fname = f'annotated_adata/Overall cells_annotated_adata.pkl'
            with open(fname, "wb") as file:
                pickle.dump(annotated_adata, file)
            base_annotated_adata = adata2
        else:
            adata3 = base_annotated_adata.copy()
            specific_cells = adata3[adata3.obs['cell_type'].isin([standardized_cell_type])].copy()
            if specific_cells.shape[0] == 0:
                specific_cells = adata3[adata3.obs['cell_type'].isin([standardized_cell_type2])].copy()
            if specific_cells.shape[0] == 0:
                specific_cells = adata3[adata3.obs['cell_type'].isin([standardized_cell_type3])].copy()
            sc.tl.pca(specific_cells, svd_solver='arpack')
            sc.pp.neighbors(specific_cells)
            sc.tl.umap(specific_cells)
            sc.tl.leiden(specific_cells, resolution=resolution)

            specific_cells.obs['cell_type'] = 'Unknown'
            for group, cell_type in map2.items():
                specific_cells.obs.loc[specific_cells.obs['leiden'] == group, 'cell_type'] = cell_type
            
            specific_cells.obs[['UMAP_1', 'UMAP_2', 'leiden', 'patient_name', 'cell_type']].to_csv(f'umaps/{standardized_cell_type}_annotated_umap_data.csv', index=False)
            
            annotated_adata = specific_cells.copy()
            fname = f'annotated_adata/{standardized_cell_type}_annotated_adata.pkl'
            with open(fname, "wb") as file:
                pickle.dump(annotated_adata, file)

    
    except (SyntaxError, ValueError) as e:
        print(f"Error in parsing the map: {e}")
        # Handle the error or return an appropriate response

    return "Repeat 'Annotation of clusters is complete'"


def convert_into_labels(conversation_history):
    global adata
    global annotated_adata
    # Extract the last two messages from the conversation history
    last_messages_arr = conversation_history[-1]
    last_message = last_messages_arr['content']
    
    str_map = ""
    map = {}
    flag = False
    for i in last_message:
        if i == "{":
            flag = True
        if i == "}":
            str_map += i
            flag = False
        if flag:
            str_map += i
    
    map = ast.literal_eval(str_map)
    map2 = {str(key): value for key, value in map.items()}
    
    adata.obs['cell_type'] = 'Unknown'
    ctr = 0
    for group, cell_type in map2.items():
        adata.obs.loc[adata.obs['leiden'] == group, 'cell_type'] = cell_type
    adata.obs[['UMAP_1', 'UMAP_2', 'leiden', 'patient_name', 'cell_type']].to_csv("annotated_umap_data.csv", index=False)
    global annotated_adata
    annotated_adata = adata.copy()
    with open('overall_annotated_adata.pkl', 'wb') as file:
        
        pickle.dump(annotated_adata, file)

    
    
def extract_top_genes_stats(adata, groupby='leiden', n_genes=25):
    result = adata.uns['rank_genes_groups']
    gene_names = result['names']
    pvals = result['pvals']
    pvals_adj = result['pvals_adj']
    logfoldchanges = result['logfoldchanges']
    
    top_genes_stats = {group: {} for group in gene_names.dtype.names}
    
    for group in gene_names.dtype.names:
        top_genes_stats[group]['gene'] = gene_names[group][:n_genes]
        top_genes_stats[group]['pval'] = pvals[group][:n_genes]
        top_genes_stats[group]['pval_adj'] = pvals_adj[group][:n_genes]
        top_genes_stats[group]['logfoldchange'] = logfoldchanges[group][:n_genes]
    
    top_genes_stats_df = pd.concat({group: pd.DataFrame(top_genes_stats[group])
                                    for group in top_genes_stats}, axis=0)

    top_genes_stats_df = top_genes_stats_df.reset_index()
    
    top_genes_stats_df = top_genes_stats_df.rename(columns={'level_0': 'cluster', 'level_1': 'index'})
    return top_genes_stats_df

# Define the statistical extraction and API interaction functions
def calculate_cluster_statistics(adata, category, n_genes=25):
    #adding
    global sample_mapping
    base_markers = get_rag_and_markers(False)
    markers = []
    for cell_type, cell_data in base_markers.items():
        print (cell_type)
        print ('--')
        print (cell_data)
        markers += cell_data['genes']
    
    print ("MARKERS BEFORE FILTER2 ", markers)
    markers = filter_existing_genes(adata, markers)
    print ("MARKERS FINAL2 ", markers)
    markers = list(set(markers))

    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon', n_genes=n_genes)
    top_genes_df = extract_top_genes_stats(adata, groupby='leiden', n_genes=25)
    second_dataset = True

    if sample_mapping:
        sc.tl.dendrogram(adata, groupby='leiden', use_rep='X_scVI')
        # sc.tl.dendrogram(adata, groupby='leiden')
    else:
        sc.tl.dendrogram(adata, groupby='leiden')

    marker_expression = sc.get.obs_df(adata, keys=['leiden'] + markers, use_raw=True)
    marker_expression.set_index('leiden', inplace=True)
    # Calculating mean and proportion of expression per cluster
    mean_expression = marker_expression.groupby('leiden').mean()
    expression_proportion = marker_expression.gt(0).groupby('leiden').mean()
    global global_top_genes_df, global_mean_expression, global_expression_proportion
    global_top_genes_df = top_genes_df
    global_mean_expression = mean_expression
    global_expression_proportion = expression_proportion
    return global_top_genes_df, global_mean_expression, global_expression_proportion

def retreive_stats():
    with open("basic_data/mean_expression.json", 'r') as file:
        mean_expression = json.load(file)
    with open("basic_data/expression_proportion.json", 'r') as file:
        expression_proportion = json.load(file)
    global adata
    global_top_genes_df, global_mean_expression, global_expression_proportion = calculate_cluster_statistics(adata, 'overall')
    # myeloid_markers = get_myeloid_markers()
    # t_cell_markers = get_t_markers()
    # overall_markers = get_overall_markers()    
    markers = get_rag_and_markers(False)
    markers = ', '.join(markers)
    explanation = "Please analyze the clustering statistics and classify each cluster based on the following data: Top Genes:Mean Expression: Expression Proportion: , based on statistical data: 1. top_genes_df: 25 top genes expression within each clusters, with it's p_val, p_val_adj, and logfoldchange; 2. mean_expression of the marker genes: specific marker genes mean expression within each cluster; 3. expression_proportion of the marker genes: every cluster each gene expression fraction within each cluster, and give back the mapping dictionary in the format like this group_to_cell_type = {'0': 'Myeloid cells','1': 'T cells','2': 'Myeloid cells','3': 'Myeloid cells','4': 'T cells'} without further explanation or comment.  I only want the summary map in the response, do not give me any explanation or comment or repeat my input, i dont want any redundant information other than the summary map"
    top_genes_summary = []
    mean_expression_str = ", ".join([f"{k}: {v}" for k, v in mean_expression.items()])
    expression_proportion_str = ", ".join([f"{k}: {v}" for k, v in expression_proportion.items()])
    # myeloid_markers_str = ", ".join(myeloid_markers)
    # t_cell_markers_str = ", ".join(t_cell_markers)
    # overall_markers_str = ", ".join(overall_markers)
    summary = (
        f"Explanation: {explanation}. "
        f"Mean expression data: {mean_expression_str}. "
        f"Expression proportion data: {expression_proportion_str}. "
        f"Top genes details: {global_top_genes_df}. "
        f"markers: {markers}. "
    )
    return summary




#tag True for RAG, False for marker genes list
def get_rag_and_markers(tag):
    specification = None
    # file_path = "../media/specification.json"
    file_path = "media/specification.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            specification = json.load(file)
            # print("BASE:", specification)
    else:
        print ("specification not found")
        return "-"

    base_file_path = os.path.join("schatbot/scChat_RAG", specification['marker'].lower())
    file_paths = []
    
    for tissue in specification['tissue']:
        file_path = os.path.join(base_file_path, tissue.lower(), specification['condition'] + '.json')
        file_paths.append(file_path)
    
    print("Constructed file paths:", file_paths)

    for file_path in file_paths:
        if os.path.exists(file_path):
            print(f"File found: {file_path}")
            with open(file_path, 'r') as file:
                data = json.load(file)
                # print(data)
        else:
            print(f"File not found: {file_path}")
            continue
    
    combined_data = {}

    # Iterate through the file paths
    for file_path in file_paths:
        if os.path.exists(file_path):
            print(f"File found: {file_path}")
            with open(file_path, 'r') as file:
                data = json.load(file)

                if tag:  # If tag is true, combine all data from the files
                    for cell_type, cell_data in data.items():
                        if cell_type not in combined_data:
                            combined_data[cell_type] = cell_data
                        else:
                            combined_data[cell_type]['markers'].extend(cell_data['markers'])

                else:  # If tag is false, retrieve only marker name + list of genes
                    for cell_type, cell_data in data.items():
                        if cell_type not in combined_data:
                            combined_data[cell_type] = {'genes': []}
                        combined_data[cell_type]['genes'].extend([marker['gene'] for marker in cell_data['markers']])
        
        else:
            print(f"File not found: {file_path}")

    fptr = open("testop.txt", "w")
    fptr.write(json.dumps(combined_data, indent=4))
    # print("Combined data:", json.dumps(combined_data, indent=4))
    return combined_data

def filter_existing_genes(adata, gene_list):
    existing_genes = [gene for gene in gene_list if gene in adata.raw.var_names]
    return existing_genes

def find_file_with_extension(directory_path, extension):
    # Iterate over all files and subdirectories in the directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(extension):
                # Return the full path to the file
                return os.path.join(root, file)
    return None


globalGroupToCellType = None
def generate_umap():
    global sample_mapping
    second_dataset = True
    matplotlib.use('Agg')
    global adata
    global resolution
    path = find_file_with_extension("media", ".h5ad")
    if not path:
        return ".h5ad file isn't given, unable to generate UMAP."
    adata = sc.read_h5ad(path)

    global current_adata
    current_adata = adata

    # Data preprocessing
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 20]
    adata.layers['counts'] = adata.X.copy()  # used by scVI-tools

    # Normalization
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata

    find_and_load_sample_mapping("media")
    # Variable genes
    if sample_mapping:
        sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True, layer='counts', flavor="seurat_v3", batch_key="Sample")
    else:
        sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True, layer='counts', flavor="seurat_v3")

    # Setup and load scVI model
    if sample_mapping:
        SCVI.setup_anndata(adata, layer="counts", categorical_covariate_keys=["Sample"], continuous_covariate_keys=['pct_counts_mt', 'total_counts'])
        model = SCVI.load(dir_path="schatbot/glioma_scvi_model", adata=adata)
        latent = model.get_latent_representation()
        adata.obsm['X_scVI'] = latent
        adata.layers['scvi_normalized'] = model.get_normalized_expression(library_size=1e4)
    
    # Clustering and UMAP
    if sample_mapping:
        sc.pp.neighbors(adata, use_rep='X_scVI')
    else:
        sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution)


    if sample_mapping:
        adata.obs['patient_name'] = adata.obs['Sample'].map(sample_mapping)

    # UMAP plot
    umap_df = adata.obsm['X_umap']
    adata.obs['UMAP_1'] = umap_df[:, 0]
    adata.obs['UMAP_2'] = umap_df[:, 1]

    #take this from MARK
    # markers = get_overall_markers()
    base_markers = get_rag_and_markers(False)
    markers = []
    for cell_type, cell_data in base_markers.items():
        print (cell_type)
        print ('--')
        print (cell_data)
        markers += cell_data['genes']
    
    # print ("MARKERS BEFORE FILTER ", markers)
    markers = filter_existing_genes(adata, markers)
    # print ("MARKERS FINAL ", markers)
    markers = list(set(markers))
    # Calculate statistics to feed into GPT
    statistic_data = sc.get.obs_df(adata, keys=['leiden'] + markers, use_raw=True)
    statistic_data.set_index('leiden', inplace=True)
    mean_expression = statistic_data.groupby('leiden').mean()
    pd_mean_expression = pd.DataFrame(mean_expression)
    pd_mean_expression.to_csv("basic_data/mean_expression.csv")
    pd_mean_expression.to_json("basic_data/mean_expression.json")

    expression_proportion = statistic_data.gt(0).groupby('leiden').mean()
    pd_expression_proportion = pd.DataFrame(expression_proportion)
    pd_expression_proportion.to_csv("basic_data/expression_proportion.csv")
    pd_expression_proportion.to_json("basic_data/expression_proportion.json")
    # Dot plot
    if sample_mapping == None:
        sc.tl.dendrogram(adata, groupby='leiden')
        with rc_context({'figure.figsize': (10, 10)}):
            sc.pl.dotplot(adata, markers, groupby='leiden', swap_axes=True, use_raw=True, standard_scale='var', dendrogram=True, color_map="Blues", save="dotplot.png")
            plt.close()
    else:
        sc.tl.dendrogram(adata, groupby='leiden', use_rep='X_scVI')
        with rc_context({'figure.figsize': (10, 10)}):
            sc.pl.dotplot(adata, markers, groupby='leiden', swap_axes=True, use_raw=True, standard_scale='var', dendrogram=True, color_map="Blues", save="dotplot.png")
            plt.close()
    
    adata.obs['cell_type'] = 'Unknown'
    adata.obs[['UMAP_1', 'UMAP_2', 'leiden', 'patient_name']].to_csv("basic_data/Overall cells_umap_data.csv", index=False)
    adata.obs[['UMAP_1', 'UMAP_2', 'leiden', 'patient_name']].to_csv("process_cell_data/Overall cells_umap_data.csv", index=False)

    # Save dot plot data for Plotly
    dot_plot_data = statistic_data.reset_index().melt(id_vars='leiden', var_name='gene', value_name='expression')
    dot_plot_data.to_csv("basic_data/dot_plot_data.csv", index=False)

    # Save dendrogram data for Plotly
    dendrogram_data = adata.uns['dendrogram_leiden']
    pd_dendrogram_linkage = pd.DataFrame(dendrogram_data['linkage'], columns=['source', 'target', 'distance', 'count'])
    pd_dendrogram_linkage.to_csv("basic_data/dendrogram_data.csv", index=False)
    # rag_data = load_RAG("Overall cells")
    rag_data = get_rag_and_markers(True)
    rag_data_str = ', '.join(rag_data)
                #   f"RAG Data : {', '.join(rag_data_str)}. " \

    summary = f"UMAP analysis completed. Data summary: {adata}, " \
                f"RAG Data : {str(rag_data)}. " \
              f"Cell counts details are provided. " \
              f"Additional data file generated: preface.txt."
    retrieve_stats_summary = retreive_stats()
    final_summary = f"{summary} {retrieve_stats_summary}"
    current_adata = adata

    return final_summary
    
def load_RAG(cell_type):
    # Construct the file name based on the cell type
    file_path = f"schatbot/markers/{cell_type}.json"

    # Check if the file exists
    if os.path.exists(file_path):
        # Open and return the contents of the JSON file
        with open(file_path, 'r') as f:
            return f.read()
    else:
        # Return an empty string if the file doesn't exist
        return ""



def process_cells(cell_type):
    global base_annotated_adata
    global resolution
    adata2 = base_annotated_adata.copy()
    cell_type = cell_type.split()[0]
    cell_type = cell_type.lower()
    cell_type = cell_type.capitalize()
    cell_type3 = cell_type
    cell_type2 = cell_type + " " + "cell"
    cell_type = cell_type + " " + "cells"
    
    filtered_cells = adata2[adata2.obs['cell_type'].isin([cell_type])].copy()
    if filtered_cells.shape[0] == 0:
        filtered_cells = adata2[adata2.obs['cell_type'].isin([cell_type2])].copy()
        # return f"No cells found for the specified cell type: {cell_type}. Please check your input."
    if filtered_cells.shape[0] == 0:
        filtered_cells = adata2[adata2.obs['cell_type'].isin([cell_type3])].copy()

    sc.tl.pca(filtered_cells, svd_solver='arpack')
    sc.pp.neighbors(filtered_cells)
    sc.tl.umap(filtered_cells)
    sc.tl.leiden(filtered_cells, resolution=resolution)
    umap_data = filtered_cells.obsm['X_umap']
    umap_df = pd.DataFrame(umap_data, columns=['UMAP_1', 'UMAP_2'])
    umap_df['cell_type'] = filtered_cells.obs['cell_type'].values
    umap_df['patient_name'] = filtered_cells.obs['patient_name'].values
    umap_df['leiden'] = filtered_cells.obs['leiden'].values

    umap_df.to_csv(f'process_cell_data/{cell_type}_umap_data.csv', index=False)
    explanation =  f'Please analyze the {cell_type} clustering statistics and classify each cluster based on the following data into more in depth cell types, based on statistical data: we prepare 1. {cell_type}_top_genes_df: 25 top genes expression within each clusters, with its p_val, p_val_adj, and logfoldchange; 2. {cell_type}_mean_expression of the marker genes: specific marker genes mean expression within each cluster; 3. {cell_type}_expression_proportion of the marker genes: every cluster each gene expression fraction within each cluster, and give back the mapping dictionary in the python dictionary format string corresponding to string without further explanation or comment.  I only want the summary map in the response, do not give me any explanation or comment or repeat my input, i dont want any redundant information other than the summary map.'
    global_top_genes_df, global_mean_expression, global_expression_proportion = calculate_cluster_statistics(filtered_cells, cell_type)
    # cell_markers = get_markers(cell_type=cell_type)
    # rag_data = load_RAG(cell_type=cell_type)
    cell_markers = get_rag_and_markers(False)
    cell_markers = ', '.join(cell_markers)
    # rag_data = get_rag_and_markers(True)
    # rag_data = ', '.join(rag_data)
    summary2 = (
    f"Explanation: {explanation}, "
    f"{cell_type} key marker genes include: {(cell_markers)}. "
    # f"{cell_type} RAG data {rag_data}"
    f"{cell_type} top genes {global_top_genes_df}"
    f"{cell_type} mean expression {str(global_mean_expression)}"
    f"{cell_type} expression proportion {str(global_expression_proportion)}"
    )
    fname = f'{cell_type} adata_processed.pkl'
    with open(fname, 'wb') as file:
        pickle.dump(umap_df, file)
    return summary2



def calculate_cell_population_change(cell_type):
    # Load the annotated adata from a pickle file
    cpc_adata = None
    cell_type = cell_type.split()[0].capitalize() + " cells"

    if cell_type == 'Overall cells':
        with open('annotated_adata/Overall cells_annotated_adata.pkl', 'rb') as file:
            cpc_adata = pd.read_pickle(file)
    else:
        fname = f'annotated_adata/{cell_type}_annotated_adata.pkl'
        with open(fname, 'rb') as file:
            cpc_adata = pd.read_pickle(file)

    cell_counts = cpc_adata.obs.groupby(['patient_name', 'cell_type']).size().reset_index(name='counts')
    total_counts = cell_counts.groupby('patient_name')['counts'].transform('sum')
    cell_counts['percentage'] = (cell_counts['counts'] / total_counts) * 100
    output_filename = f'schatbot/cell_population_change/{cell_type}_cell_population_change.csv'
    cell_counts.to_csv(output_filename, index=False)
    summary = "Can you tell me the cell population change for each cell type / patient from this data? do not tell me how to do it, just tell me"
    summary2 = (
        f"Explanation: {summary}, "
        + cell_counts.to_string(index=False)
    )
    return summary2


def sample_differential_expression_genes_comparison(cell_type, sample_1, sample_2):
    global adata
    global sample_mapping
    adata2 = None
    cell_type3 = cell_type.split()[0].capitalize()
    cell_type2 = cell_type.split()[0].capitalize() + " cell"
    cell_type = cell_type.split()[0].capitalize() + " cells"
    
    with open(f'annotated_adata/Overall cells_annotated_adata.pkl', 'rb') as file:
        adata2 = pd.read_pickle(file)
    
    filtered_cells = adata2[adata2.obs['cell_type'].isin([cell_type])].copy()
    if filtered_cells.shape[0] == 0:
        filtered_cells = adata2[adata2.obs['cell_type'].isin([cell_type2])].copy()
    if filtered_cells.shape[0] == 0:
        filtered_cells = adata2[adata2.obs['cell_type'].isin([cell_type3])].copy()

    adata_filtered = filtered_cells[filtered_cells.obs['patient_name'].isin([sample_1, sample_2])].copy()
    unique_patients = adata_filtered.obs['patient_name'].astype(str).unique()
    if sample_1 not in unique_patients or sample_2 not in unique_patients:
        return f"Error: One or both patients ({sample_1}, {sample_2}) not found in the dataset for cell type '{cell_type}'."
    

    sc.tl.rank_genes_groups(adata_filtered, groupby='patient_name', groups=[sample_2], reference=sample_1, method='wilcoxon')

    results_post = {
        'genes': adata_filtered.uns['rank_genes_groups']['names'][sample_2],
        'logfoldchanges': adata_filtered.uns['rank_genes_groups']['logfoldchanges'][sample_2],
        'pvals': adata_filtered.uns['rank_genes_groups']['pvals'][sample_2],
        'pvals_adj': adata_filtered.uns['rank_genes_groups']['pvals_adj'][sample_2]
    }
    
    df_post = pd.DataFrame(results_post)

    significant_genes_post = df_post[df_post['pvals_adj'] < 0.05]

    significant_genes_post = significant_genes_post[abs(significant_genes_post['logfoldchanges']) > 1]
    significant_genes_post.to_csv('SGP.csv', index=False)

    summary = (
        f"Reference Sample: {sample_1}, Comparison Sample: {sample_2}\n"
        "Explanation: DO NOT GIVE PYTHON CODE. JUST COMPARE AND EXPLAIN. This function is designed to perform a differential gene expression analysis for a specified cell type between two sample conditions (pre and post-treatment or two different conditions). "
        "The reference patient condition is patient_1. The differential expression analysis is performed by comparing the gene expression levels of sample_2 against those of sample_1. Provide a comparison with normal formatting.\n"
        "Significant Genes Data: \n"
        f"{str(significant_genes_post)}\n"
        "Explanation of attributes:\n"
        "Genes: The names of the genes analyzed, providing insight into which genes are tested for differential expression between the two conditions.\n"
        "Log Fold Changes: Values showing how gene expression levels differ between the two conditions. Positive values indicate upregulation in sample_2, and negative values indicate downregulation in sample_2.\n"
        "P-values: These values help determine the statistical significance of the observed changes in gene expression. Lower p-values suggest that the changes are less likely to have occurred by chance.\n"
        "Adjusted P-values: These values provide a more stringent measure of significance by controlling for the false discovery rate. Significant adjusted p-values (e.g., < 0.05) indicate that the changes in gene expression are statistically robust even after adjusting for multiple comparisons."
    )
    

    return summary



import os
import gseapy as gp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regex as re
out_df = None
matplotlib.use('Agg')

def safe_filename(term):
    # Replace spaces with underscores and remove any special characters
    term_safe = re.sub(r'[^a-zA-Z0-9_]', '_', term)
    # Truncate the term if it exceeds a certain length (e.g., 50 characters)
    return term_safe[:50]

def gsea_analysis():
    try:
        global SGP
        SGP = pd.read_csv('SGP.csv')
        print("SGP.csv loaded successfully")
    except Exception as e:
        print(f"Error loading SGP.csv: {e}")
        return {"status": "error", "message": f"Error loading SGP.csv: {e}"}

    try:
        significant_genes_post = SGP
        significant_genes_post['rank'] = -np.log10(significant_genes_post.pvals_adj) * significant_genes_post.logfoldchanges
        significant_genes_post = significant_genes_post.sort_values('rank', ascending=False).reset_index(drop=True)
        ranking = significant_genes_post[['genes', 'rank']]
        
        gene_list = ranking['genes'].str.strip().to_list()
        print("Gene list prepared")
    except Exception as e:
        print(f"Error preparing gene list: {e}")
        return {"status": "error", "message": f"Error preparing gene list: {e}"}

    try:
        libraries = gp.get_library_name()
        pre_res = gp.prerank(rnk=ranking, 
                             gene_sets=["KEGG_2021_Human", "GO_Biological_Process_2023", "GO_Molecular_Function_2023", 
                                        "GO_Cellular_Component_2023", "Reactome_2022", "MSigDB_Hallmark_2020", 
                                        "MSigDB_Oncogenic_Signatures", "Cancer_Cell_Line_Encyclopedia", 
                                        "Human_Phenotype_Ontology", "Disease_Signatures_from_GEO_down_2014", 
                                        "Disease_Signatures_from_GEO_up_2014", "Disease_Perturbations_from_GEO_down", 
                                        "Disease_Perturbations_from_GEO_up"],
                             seed=6)
        print("GSEA analysis completed")
    except Exception as e:
        print(f"Error in GSEA analysis: {e}")
        return {"status": "error", "message": f"Error in GSEA analysis: {e}"}

    out = []
    for term in pre_res.results:
        fdr = pre_res.results[term]['fdr']
        es = pre_res.results[term]['es']
        nes = pre_res.results[term]['nes']
        if fdr <= 0.05:
            out.append([term, fdr, es, nes])

    global out_df

    try:
        out_df = pd.DataFrame(out, columns=['Term', 'fdr', 'es', 'nes']).sort_values('fdr').reset_index(drop=True)
        print("Filtered significant terms")
    except Exception as e:
        print(f"Error filtering significant terms: {e}")
        return {"status": "error", "message": f"Error filtering significant terms: {e}"}

    try:
        os.makedirs('gsea_plots', exist_ok=True)
        print("Directory for plots ensured")
    except Exception as e:
        print(f"Error ensuring plot directory: {e}")
        return {"status": "error", "message": f"Error ensuring plot directory: {e}"}

    terms_to_plot = out_df['Term']
    try:
        axs = pre_res.plot(terms=terms_to_plot, show_ranking=False, legend_kws={'loc': (1.05, 0)})
        plt.title("GSEA Enrichment Scores for Significant Terms (FDR ≤ 0.05)")
        plt.xlabel("Rank in Ordered Dataset")
        plt.ylabel("Enrichment Score (ES)")
        plt.savefig('gsea_plots/enrichment_scores.png')
        plt.close()
        print("Enrichment scores plot saved")
    except Exception as e:
        print(f"Failed to plot all terms together: {e}")

    try:
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            x=out_df['nes'],
            y=out_df['Term'],
            s=(out_df['es'].abs() * 500),  # Increase the size scaling factor
            c=out_df['fdr'],
            cmap='RdBu_r',
            alpha=0.7
        )

        cbar = plt.colorbar(scatter)
        cbar.set_label('FDR')

        plt.title('GSEA Dot Plot')
        plt.xlabel('Normalized Enrichment Score (NES)')
        plt.ylabel('Term')
        plt.tight_layout()
        plt.savefig("gsea_plots/gsea_dot_plot.png")
        plt.close()
        print("GSEA dot plot saved")
    except Exception as e:
        print(f"Error saving GSEA dot plot: {e}")

    try:
        output_file = 'gsea_plots/gsea_plot_data.csv'
        if not out_df.empty:
            all_output_data = []
            for index, row in out_df.iterrows():
                term_to_graph = row['Term']
                term_details = pre_res.results[term_to_graph].copy()

                rank_metric = pre_res.ranking
                es_profile = term_details['RES']

                gsea_data = pd.DataFrame({
                    'Rank Metric': rank_metric,
                    'Enrichment Score': es_profile
                })

                term_info = {
                    'Term': term_to_graph,
                    'fdr': term_details['fdr'],
                    'es': term_details['es'],
                    'nes': term_details['nes'],
                    'Rank Metric': '',
                    'Enrichment Score': ''
                }

                term_info_df = pd.DataFrame([term_info])
                output_df = pd.concat([term_info_df, gsea_data], ignore_index=True)
                all_output_data.append(output_df)

            final_output_df = pd.concat(all_output_data, ignore_index=True)
            final_output_df.to_csv(output_file, index=False)
            print(f"GSEA plot data saved to '{output_file}'")
        else:
            print("No significant gene sets found with FDR <= 0.05.")
    except Exception as e:
        print(f"Error saving GSEA plot data: {e}")

    try:
        ranking_gene_list = significant_genes_post[['genes', 'rank']]
        ranking_gene_list.to_csv('ranking_gene_list.csv', index=False)
        print("Ranking gene list saved to 'ranking_gene_list.csv'")
    except Exception as e:
        print(f"Error saving ranking gene list: {e}")

    response = {
        "status": "success",
        "message": "GSEA analysis completed successfully",
        "enrichment_scores_plot": 'gsea_plots/enrichment_scores.png',
        "dot_plot": 'gsea_plots/gsea_dot_plot.png',
        "gsea_plot_data": 'gsea_plot_data.csv',
        "ranking_gene_list": 'ranking_gene_list.csv'
    }
    # When returning the response, serialize it to JSON if needed
    response_json = json.dumps(response)

    # Ensure that when passing this response back, it is treated as an object
    return response_json

def read_image():
    import base64
    import requests

    # OpenAI API Key
    api_key = os.getenv("OPENAI_API_KEY")


    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Find the first JPG image in the folder (modify the directory if needed)
    image_path = find_file_with_extension("media", ".jpg")

    if not image_path:
        return "No image found in the folder."

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What’s in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    try:
        return response.json()['choices'][0]['message']['content']
    except KeyError:
        return "Error in processing the image."


function_descriptions = [
    {
        "name": "generate_umap",
        "description": "Used to Generate UMAP for unsupervised clustering for RNA analysis. Generates a UMAP visualization based on the given RNA sequencing data",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    
    {
        "name": "display_dotplot",
        "description": "Displays the dotplot for the sample",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "display_cell_type_composition",
        "description": "Displays the cell type composition for the sample",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
   
    
    {
        "name": "read_image",
        "description": "Reads and processes an image from the 'media' folder and returns a description of what's in the image.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "process_cells",
        "description": "process cells",
        "parameters": {
            "type": "object",
            "properties": 
                {"cell_type":{
                    "type":"string","description":"the cell type"}
                 },
            "required": ["cell_type"],
        },
    },
    {
        "name": "display_umap",
        "description": "displays umap that is NOT annotated. This function should be called whenever the user asks for a umap that is not annotated. In the case that the user does not specify cell type, use overall cells. This function can be called multiple times. This function should not be called when asked to GENERATE umap.",
        "parameters": {
            "type": "object",
            "properties":
                {"cell_type":{
                    "type":"string","description":"the cell type"}
                 },
            "required": ["cell_type"],
        },
    },
    {
        "name": "display_processed_umap",
        "description": "displays umap that IS annotated. This function should be called whenever the user asks for a umap that IS annotated. In the case that the user does not specify cell type, use overall cells. This function can be called multiple times. This function should not be called when asked to GENERATE umap.",
        "parameters": {
            "type": "object",
            "properties":
                {"cell_type":{
                    "type":"string","description":"the cell type"}
                 },
            "required": ["cell_type"],
        },
    },
    {
        "name": "display_cell_population_change",
        "description": "displays cell population change graph",
        "parameters": {
            "type": "object",
            "properties": 
                {"cell_type":{
                    "type":"string","description":"the cell type"}
                 },
            "required": ["cell_type"],
        },
    },
    
    {
    "name": "sample_differential_expression_genes_comparison",
    "description": "Function is designed to perform a differential gene expression analysis for a specified cell type between two patient conditions (pre and post-treatment or two different conditions)",
    "parameters": {
        "type": "object",
        "properties": {
            "cell_type": {
                "type": "string",
                "description": "The type of cell to be compared"
            },
            "sample_1": {
                "type": "string",
                "description": "Identifier for the first patient"
            },
            "sample_2": {
                "type": "string",
                "description": "Identifier for the second patient"
            }
        },
        "required": ["cell_type", "sample_1", "sample_2"]
    }
    },

    {
        "name": "calculate_cell_population_change",
        "description": "calculate_cell population change for percentage changes in cell populations or samples before and after treatment. This calculation can be done for any cell type. This is to see the changes in the population before and after treatment.",
        "parameters": {
            "type": "object",
            "properties": 
                {"cell_type":{
                    "type":"string","description":"the cell type"}
                 },
            "required": ["cell_type"],
        },
    },
    {
        "name": "gsea_analysis",
        "description": "Performs Gene Set Enrichment Analysis (GSEA) on a dataset of significant genes. This function ranks the genes based on their adjusted p-values and log-fold changes, performs the GSEA analysis using multiple gene set libraries, filters the results for significant terms (FDR ≤ 0.05), and generates several output files including enrichment score plots, a dot plot, and CSV files with the GSEA results and ranked gene list.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "label_clusters",
        "description": "This function can be called multiple times. this function is to label and or annotate clusters. It can be done for any type of cells that is mentioned by the user. If the user does not mention the cell type use overall cells. This function can be called multiple times.",
        "parameters": {
            "type": "object",
            "properties": 
                {"cell_type":{
                    "type":"string","description":"the cell type"}
                 },
            "required": ["cell_type"],
        },
    },
]


def start_chat2(file_path):
    conversation_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        conversation_history.append({"role": "user", "content": user_input})

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=conversation_history,
            functions=function_descriptions,
            function_call="auto"  
        )

        output = response.choices[0].message
        ai_response = output.content
        print ("received ai response : ", ai_response)
        if output.function_call:
            print(f"Making a function call to: {output.function_call.name}")
            function_name = output.function_call.name
            print("Assistant:", "All values returned from " + output.function_call.name + " have been received")
            summary = globals()[function_name]()  # Assuming no arguments are needed
            if summary:
                conversation_history.append({"role": "user", "content": summary})
        else:
            print("Assistant:", ai_response)
            conversation_history.append({"role": "assistant", "content": ai_response})
        





global first_try
global clear_data
first_try = True
clear_data = True
def start_chat2_web(user_input, conversation_history):
    global function_flag, display_flag, first_try, clear_data
    scchat_context1 = " You are a chatbot for helping in Single Cell RNA Analysis, you can call the functions generate_umap, process_cells, label_clusters, display_umap, display_processed_umap and more multiple times. DO NOT FORGET THIS. respond with a greeting."
    scchat_context2 = " you should decide if you want to call the functions from this message. Functions include generate_umap, process_cells, label_clusters, display_umap, display_processed_umap and more multiple times. DO NOT FORGET THIS."
    # base_conversation_history = [{"role": "user", "content": scchat_context1}]
    base_conversation_history = []
    if first_try and clear_data:
        # Clear preexisting directory data
        clear_directory("annotated_adata")
        clear_directory("basic_data")
        clear_directory("umaps")
        clear_directory("process_cell_data")
        research_context_path = find_file_with_extension("media", ".txt")
        if research_context_path:
            rcptr = open(research_context_path, "r")
            research_context = rcptr.read()
            conversation_history.append({"role": "user", "content": research_context})
        # clear_directory("media")
        conversation_history.append({"role": "user", "content": scchat_context1})
        first_try = False
        

    display_flag = False
    # user_input += scchat_context2
    conversation_history.append({"role": "user", "content": user_input})
    
    global conversation_history2
    conversation_history2 = conversation_history
    # print(conversation_history)
    openai.api_key = os.getenv("OPENAI_API_KEY")

    base_conversation_history.append({"role": "user", "content": user_input})
    # print (conversation_history)
    # print (base_conversation_history)
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=base_conversation_history,
        functions=function_descriptions,
        function_call="auto",
        temperature=0.2,
        top_p=0.4
        # max_tokens=300
    )
    output = response.choices[0].message
    main_flag = False
    if response and output.function_call:
        main_flag = True

    if main_flag == False:
        # print(conversation_history)
        fin_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=conversation_history,
            functions=function_descriptions,
            function_call="auto",
            temperature=0.2,
            top_p=0.4
            # max_tokens=300
        )
        output = fin_response.choices[0].message
        ai_response = output.content

    if output.function_call and main_flag:
        function_name = output.function_call.name
        function_args = output.function_call.arguments
        function_flag = True

        try:
            function_response = "Function did not execute."  # Placeholder response
            print (f"Making a function call to: {function_name}")
            if function_args is None:
                function_response = globals()[function_name]()
            else:
                function_args = json.loads(function_args)
                print(f"Parsed function arguments: {function_args}")

                function_response = globals()[function_name](**function_args)
            
            # Check if display_flag is set to True, return immediately if it is
            if display_flag:
                return function_response, conversation_history, display_flag
            
            # Handle label_clusters differently to avoid re-generating responses
            if function_name == "label_clusters":
                function_result_message = {"role": "assistant", "content": "Annotation is complete."}
                conversation_history.append(function_result_message)  
                final_response = "Annotation is complete."
            else:
                function_result_message = {"role": "user", "content": function_response}
                conversation_history.append(function_result_message)

                # Generate a new response if the function isn't label_clusters
                new_response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=conversation_history,
                    temperature=0.2,
                    top_p=0.4
                    # max_tokens=300
                )
                final_response = new_response.choices[0].message.content if new_response.choices[0] else "Interesting"
                conversation_history.append({"role": "assistant", "content": final_response})

            function_flag = False
            return final_response, conversation_history, display_flag

        except KeyError as e:
            print(f"Function {function_name} not found: {e}")
            return f"Function {function_name} not found.", conversation_history, display_flag

    else:
        conversation_history.append({"role": "assistant", "content": ai_response})
        return ai_response, conversation_history, display_flag




if __name__ == "__main__":
    start_chat2()

