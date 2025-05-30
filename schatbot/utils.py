import os
import shutil
import json
import regex as re
import logging
import ast
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import scipy
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

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
            print("Not found")


def get_url(gene_name):
    gene_name = str(gene_name).upper()
    url_format = "https://www.ncbi.nlm.nih.gov/gene/?term="
    return gene_name+": "+url_format+gene_name

def explain_gene(gene_dict, marker_tree, annotation_result):
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )
    
    input_msg =(f"Top genes details: {gene_dict}. "
                f"Markers: {marker_tree}. "
                f"Annotation results: {annotation_result}")
    
    messages = [
        SystemMessage(content="""
            You are a bioinformatics researcher that can do the cell annotation.
            The following are the data and its decription that you will receive.
            * 1. Gene list: top 25 cells arragned from the highest to lowest expression levels in each cluster.
            * 2. Marker tree: marker genes that can help annotating cell type. 
            * 3. Cell annotation results.
            
            Basing on the given information you will have to return a gene list with top 3 possible genes that you use to do the cell annotation.
            This mean you have to give a gene list with 5 most possible genes for each cluster that be used for cell annotation.
            
            Provide your output in the following example format:
            {'0': ['gene 1', 'gene 2', ...],'1': ['gene 1', 'gene 2', ...],'2': ['gene 1', 'gene 2', ...],'3': ['gene 1', 'gene 2', ...],'4': ['gene 1', 'gene 2', ...], ...}.
            
            Strictly adhere to follwing rules:
            * 1. Adhere to the dictionary format and do not include any additional words or explanations.
            * 2. The cluster number in the result dictionary must be arranged with raising power.            
            """),
        HumanMessage(content=input_msg)
    ]
    results = model.invoke(messages)
    gene = results.content
    gene = ast.literal_eval(gene)

    url_clusters = {}
    for cluster_id, gene_list in gene.items():
        url_clusters[cluster_id] = [get_url(gene) for gene in gene_list]
        
    input_msg =(f"Annotation results: {annotation_result}"
                f"Top marker genes: {gene}"
                f"URL for top marker genes: {url_clusters}")
    messages = [
        SystemMessage(content="""
            You are a bioinformatics researcher that can do the cell annotation.
            The following are the data and its decription that you will receive.
            * 1. Cell annotation results.
            * 2. Top marker genes that have been use to do cell annotation.
            * 3. URL for top marker genes, which are the references for each marker genes in each cluster.
            
            Basing on the given information you will have to explain why these marker genes have been used to do cell annotation while with reference URL with it.
            This means you will have to expalin why those top marker genes are used to represent cell in cell annotation results.
            After explanation you will have to attach genes' corresponding URL. 
            
            The response is supposed to follow the following format:
            ### Cluster 0: Cells
            - **Gene 1**: Explanation.
            - **Gene 2**: Explanation.
            - **Gene 3**: Explanation.
            - **Gene 4**: Explanation.
            - **Gene 5**: Explanation.

            **URLs**:
            - Gene 1: (URL 1)
            - Gene 2: (URL 2)
            - Gene 3: (URL 3)
            - Gene 4: (URL 4)
            - Gene 5: (URL 5)
            ...
            
            ** All the clusters should be included.
            ** If there is same cell appears in different clusters you can combine them together.
            """),
        HumanMessage(content=input_msg)
    ]    
    results = model.invoke(messages)
    explanation = results.content
    
    return explanation

def get_mapping(directory):
    sample_mapping = None
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'sample_mapping.json':
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        sample_mapping = json.load(f)
                    return sample_mapping  # Return immediately when found
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return sample_mapping  # Return None if no valid file is found

def get_h5ad(directory_path, extension):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(extension):
                return os.path.join(root, file)
    return None

def extract_genes(data):
    genes = []
    if isinstance(data, dict):
        # Check if this is a cell type dictionary with a 'markers' key
        if 'markers' in data:
            # If markers is directly a list of gene names
            if isinstance(data['markers'], list):
                genes.extend(data['markers'])
        
        # Recurse into nested dictionaries
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                genes.extend(extract_genes(value))
                
    elif isinstance(data, list):
        # Process each item in the list
        for item in data:
            genes.extend(extract_genes(item))
            
    return genes

def get_rag():
    """Retrieve marker genes using Neo4j graph database."""
    from neo4j import GraphDatabase
    uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "37754262"))
    specification = None
    file_path = "media/specification_graph.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            specification = json.load(file)
    else:
        print("specification not found")
        return "-"
    database = specification['database']
    system = specification['system']
    organ = specification['organ']
    combined_data = {}
    try:
        with driver.session(database=database) as session:
            query = """
            MATCH (s:System {name: $system})-[:HAS_ORGAN]->(o:Organ {name: $organ})-[:HAS_CELL]->(c:CellType)-[:HAS_MARKER]->(m:Marker)
            RETURN c.name as cell_name, m.markers as marker_list
            """
            result = session.run(query, system=system, organ=organ)
            for record in result:
                cell_name = record["cell_name"]
                marker_genes = record["marker_list"]
                if cell_name not in combined_data:
                    combined_data[cell_name] = {"markers": []}
                for marker in marker_genes:
                    combined_data[cell_name]["markers"].append(marker)
    except Exception as e:
        print(f"Error accessing Neo4j: {e}")
        return {}
    finally:
        driver.close()
    return combined_data

def get_subtypes(cell_type):
    """Get subtypes of a given cell type using Neo4j."""
    from neo4j import GraphDatabase
    uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "37754262"))
    specification = None
    file_path = "media/specification_graph.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            specification = json.load(file)
    database = specification['database']
    subtypes_data = {}
    try:
        with driver.session(database=database) as session:
            query = """
            MATCH (parent:CellType {name: $parent_cell})-[:DEVELOPS_TO]->(c:CellType)-[:HAS_MARKER]->(m:Marker)
            RETURN c.name as cell_name, m.markers as marker_list
            """
            result = session.run(query, parent_cell=cell_type)
            for record in result:
                subtype_name = record["cell_name"]
                marker_genes = record["marker_list"]
                if subtype_name not in subtypes_data:
                    subtypes_data[subtype_name] = {"markers": []}
                for marker in marker_genes:
                    subtypes_data[subtype_name]["markers"].append(marker)
    except Exception as e:
        print(f"Error accessing Neo4j: {e}")
        return {}
    finally:
        driver.close()
    return subtypes_data

def filter_existing_genes(adata, gene_list):
    """Filter genes to only those present in the dataset, handling non-unique indices."""
    if hasattr(adata, 'raw') and adata.raw is not None:
        var_names = adata.raw.var_names
    else:
        var_names = adata.var_names
    existing_genes = [gene for gene in gene_list if gene in var_names]
    return existing_genes

def save_analysis_results(adata, prefix, leiden_key='leiden', save_umap=True, 
                         save_dendrogram=True, save_dotplot=False, markers=None):
    """Save analysis results to files with consistent naming."""
    out_dir = os.path.dirname(prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if save_umap:
        umap_cols = ['UMAP_1', 'UMAP_2', leiden_key]
        if 'patient_name' in adata.obs.columns:
            umap_cols.append('patient_name')
        if 'cell_type' in adata.obs.columns:
            umap_cols.append('cell_type')
        adata.obs[umap_cols].to_csv(f"{prefix}_umap_data.csv", index=False)
    if save_dendrogram and f'dendrogram_{leiden_key}' in adata.uns:
        dendrogram_data = adata.uns[f'dendrogram_{leiden_key}']
        pd_dendrogram_linkage = pd.DataFrame(
            dendrogram_data['linkage'],
            columns=['source', 'target', 'distance', 'count']
        )
        pd_dendrogram_linkage.to_csv(f"{prefix}_dendrogram_data.csv", index=False)
    if save_dotplot and markers:
        statistic_data = sc.get.obs_df(adata, keys=[leiden_key] + markers, use_raw=True)
        statistic_data.set_index(leiden_key, inplace=True)
        dot_plot_data = statistic_data.reset_index().melt(
            id_vars=leiden_key, var_name='gene', value_name='expression'
        )
        dot_plot_data.to_csv(f"{prefix}_dot_plot_data.csv", index=False)

def dea_split_by_condition(adata, cell_type, n_genes=100, logfc_threshold=1, pval_threshold=0.05, save_csv=True):
    try:
        adata_modified = adata.copy()
        pre_mask = adata_modified.obs["patient_name"].str.contains("_pre")
        post_mask = ~pre_mask
        adata_pre = adata_modified[pre_mask].copy()
        adata_post = adata_modified[post_mask].copy()
        if len(adata_pre) == 0 or len(adata_post) == 0:
            print("Error: One of the conditions (pre or post) has no cells.")
            return adata_pre, adata_post, [], []
        adata_pre.obs["cell_type_group"] = "Other"
        cell_type_mask_pre = adata_pre.obs["cell_type"] == str(cell_type)
        adata_pre.obs.loc[cell_type_mask_pre, "cell_type_group"] = str(cell_type)
        adata_post.obs["cell_type_group"] = "Other"
        cell_type_mask_post = adata_post.obs["cell_type"] == str(cell_type)
        adata_post.obs.loc[cell_type_mask_post, "cell_type_group"] = str(cell_type)
        if sum(cell_type_mask_pre) == 0:
            print(f"Error: No {cell_type} cells found in pre condition.")
            return adata_pre, adata_post, [], []
        if sum(cell_type_mask_post) == 0:
            print(f"Error: No {cell_type} cells found in post condition.")
            return adata_pre, adata_post, [], []
        pre_key_name = f"{cell_type}_markers_pre_only"
        post_key_name = f"{cell_type}_markers_post_only"
        sc.tl.rank_genes_groups(adata_pre, groupby="cell_type_group", 
                            groups=[str(cell_type)], reference="Other",
                            method="wilcoxon", n_genes=n_genes, 
                            key_added=pre_key_name, use_raw_=False)
        sc.tl.rank_genes_groups(adata_post, groupby="cell_type_group", 
                            groups=[str(cell_type)], reference="Other",
                            method="wilcoxon", n_genes=n_genes, 
                            key_added=post_key_name, use_raw_=False)
        pre_data = sc.get.rank_genes_groups_df(adata_pre, group=str(cell_type), key=pre_key_name)
        pre_significant_genes = list(pre_data.loc[(pre_data['pvals_adj'] < pval_threshold) & 
                            (abs(pre_data['logfoldchanges']) > logfc_threshold), 'names'])
        post_data = sc.get.rank_genes_groups_df(adata_post, group=str(cell_type), key=post_key_name)
        post_significant_genes = list(post_data.loc[(post_data['pvals_adj'] < pval_threshold) & 
                                    (abs(post_data['logfoldchanges']) > logfc_threshold), 'names'])
        if save_csv:
            pre_file_name = f"{cell_type}_markers_pre_only.csv"
            pre_data.to_csv(pre_file_name, index=False)
            print(f"Pre condition {cell_type} marker results saved to {pre_file_name}")
            post_file_name = f"{cell_type}_markers_post_only.csv"  
            post_data.to_csv(post_file_name, index=False)
            print(f"Post condition {cell_type} marker results saved to {post_file_name}")
        return adata_pre, adata_post, pre_significant_genes, post_significant_genes
    except Exception as e:
        print(f"Error in analysis: {e}")
        return None, None, [], []

def compare_cell_count(adata, cell_type, sample1, sample2):
    mask_sample1 = adata.obs["patient_name"] == sample1
    mask_sample2 = adata.obs["patient_name"] == sample2
    mask_cells = adata.obs["cell_type"] == cell_type
    combined_mask1 = mask_sample1 & mask_cells
    combined_mask2 = mask_sample2 & mask_cells
    adata_sample1 = adata[combined_mask1]
    adata_sample2 = adata[combined_mask2]
    sample1_num = adata_sample1.shape[0]
    sample2_num = adata_sample2.shape[0]
    counts_comp = {f"{sample1}": sample1_num, f"{sample2}": sample2_num}
    return counts_comp

def repeat():
    return "Hello"

