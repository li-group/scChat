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
import mygene
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
    import json
    import os
    
    uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "37754262"))
    specification = None
    file_path = "media/specification_graph.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            specification = json.load(file)
    else:
        print("specification not found")
        return {}
    
    database = specification['database']
    system = specification['system']
    organ = specification['organ']
    
    print(f"🔍 Querying Level 1 markers for organ: {organ}")
    
    combined_data = {}
    try:
        with driver.session(database=database) as session:
            query = """
            MATCH (s:System {name: $system})-[:HAS_ORGAN]->(o:Organ {name: $organ})-[:HAS_CELL]->(c:CellType)
            WHERE c.level = 1 AND c.organ = $organ
            RETURN c.name as cell_name, c.markers as marker_list
            """
            result = session.run(query, system=system, organ=organ)
            
            for record in result:
                cell_name = record["cell_name"]  # ✅ Now matches query alias
                marker_genes = record["marker_list"] or []  # ✅ Now matches query alias, handle null
                
                if cell_name not in combined_data:
                    combined_data[cell_name] = {"markers": []}
            
                combined_data[cell_name]["markers"] = marker_genes   
    except Exception as e:
        print(f"Error accessing Neo4j: {e}")
        return {}
    finally:
        driver.close()
    return combined_data

def get_subtypes(cell_type):
    """Get subtypes of a given cell type using Neo4j."""
    from neo4j import GraphDatabase
    import json
    import os
    
    uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "37754262"))
    specification = None
    file_path = "media/specification_graph.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            specification = json.load(file)
    else:
        print("specification not found")
        return {}
    
    database = specification['database']
    organ = specification['organ']  # ✅ ADDED: Extract organ from specification
        
    subtypes_data = {}
    try:
        with driver.session(database=database) as session:
            query = """
            MATCH (parent:CellType {name: $parent_cell, organ: $organ})-[:DEVELOPS_TO {organ: $organ}]->(child:CellType)
            WHERE child.organ = $organ
            RETURN child.name as cell_name, child.markers as marker_list
            """
            # ✅ FIXED: Added missing organ parameter
            result = session.run(query, parent_cell=cell_type, organ=organ)
            
            for record in result:
                subtype_name = record["cell_name"]  # ✅ Now matches query alias
                marker_genes = record["marker_list"] or []  # ✅ Now matches query alias, handle null
                
                if subtype_name not in subtypes_data:
                    subtypes_data[subtype_name] = {"markers": []}
                
                # ✅ FIXED: Markers are already a list, don't iterate through them
                subtypes_data[subtype_name]["markers"] = marker_genes
                
    except Exception as e:
        print(f"Error accessing Neo4j: {e}")
        return {}
    finally:
        driver.close()
    return subtypes_data

def filter_existing_genes(adata, gene_list):
    """
    Filter genes to only those present in the dataset, handling Ensembl ID conversion
    without modifying the original AnnData object.
    
    Returns:
        existing_genes (list): List of genes that exist in the dataset
        adata (AnnData): The original adata object (unchanged)
    """
    # Determine which var_names to check (raw or main)
    if hasattr(adata, 'raw') and adata.raw is not None:
        var_names = adata.raw.var_names
        use_raw = True
    else:
        var_names = adata.var_names
        use_raw = False
    # Check if var_names are likely Ensembl IDs (e.g., start with 'ENSG')
    needs_conversion = len(var_names) > 0 and var_names[0].startswith('ENSG')
    
    if needs_conversion:
        print("Detected Ensembl IDs in .var_names. Converting to gene symbols for filtering (original data unchanged)...")
        
        try:
            import mygene
            mg = mygene.MyGeneInfo()
            
            # Clean Ensembl IDs by removing version numbers (e.g., .1)
            ensembl_ids = var_names.to_series().str.split('.').str[0]          
            # Query gene symbols
            gene_info = mg.querymany(
                ensembl_ids,
                scopes='ensembl.gene',
                fields='symbol',
                species='human',
                as_dataframe=True
            )
            
            # Handle cases where mygene returns multiple hits for a single ID
            if not gene_info.empty:
                gene_info = gene_info[~gene_info.index.duplicated(keep='first')]
                
                # Map symbols, falling back to the original ID if no symbol is found
                id_to_symbol_map = gene_info.get('symbol', pd.Series())
                converted_var_names = ensembl_ids.map(id_to_symbol_map).fillna(ensembl_ids)
            else:
                # If no results from mygene, keep original names
                converted_var_names = ensembl_ids
            
            # Create a mapping from original gene symbols to both original and converted names
            # This allows matching against both Ensembl IDs and gene symbols
            all_searchable_names = set(var_names.tolist() + converted_var_names.tolist())
            
            print("Conversion complete. Using converted names for filtering without modifying original data.")

            
        except Exception as e:
            print(f"Error during Ensembl ID conversion: {e}")
            print("Falling back to original var_names")
            all_searchable_names = set(var_names.tolist())
    else:
        # If not Ensembl IDs, use them as they are
        all_searchable_names = set(var_names.tolist())
    
    # Filter the input gene list against all searchable names
    existing_genes = [gene for gene in gene_list if gene in all_searchable_names]
    
    # Log filtering statistics
    original_count = len(gene_list)
    found_count = len(existing_genes)
    print(f"Gene filtering complete: {found_count}/{original_count} genes found in dataset")
    if found_count < original_count:
        missing_genes = [gene for gene in gene_list if gene not in all_searchable_names]
        print(f"Missing genes: {missing_genes[:10]}{'...' if len(missing_genes) > 10 else ''}")
    
    # IMPORTANT: Always return the original adata object unchanged
    return existing_genes, adata

def save_analysis_results(adata, prefix, leiden_key='leiden', save_umap=True, 
                         save_dendrogram=True, save_dotplot=False, markers=None):
    """Save analysis results to files with consistent naming."""
    out_dir = os.path.dirname(prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if save_umap:
        umap_cols = ['UMAP_1', 'UMAP_2', leiden_key]
        if 'Exp_sample_category' in adata.obs.columns:
            umap_cols.append('Exp_sample_category')
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
    from .annotation import unified_cell_type_handler
    cell_type = unified_cell_type_handler(cell_type)
    try:
        adata_modified = adata.copy()
        categories = adata_modified.obs["Exp_sample_category"].unique()
        result_list = []
        for cat in categories:
            mask = adata_modified.obs["Exp_sample_category"] == cat
            adata_cat = adata_modified[mask].copy()
            if len(adata_cat) == 0:
                print(f"Error: No cells for category {cat}.")
                continue
            adata_cat.obs["cell_type_group"] = "Other"
            cell_type_mask = adata_cat.obs["cell_type"] == str(cell_type)
            adata_cat.obs.loc[cell_type_mask, "cell_type_group"] = str(cell_type)
            if sum(cell_type_mask) == 0:
                print(f"Error: No {cell_type} found in {cat} condition.")
                continue
            key_name = f"{cell_type}_markers_{cat}"
            sc.tl.rank_genes_groups(adata_cat, groupby="cell_type_group", 
                                    groups=[str(cell_type)], reference="Other",
                                    method="wilcoxon", n_genes=n_genes, 
                                    key_added=key_name, use_raw_=False)
            data = sc.get.rank_genes_groups_df(adata_cat, group=str(cell_type), key=key_name)
            significant_genes = list(data.loc[(data['pvals_adj'] < pval_threshold) & 
                                              (abs(data['logfoldchanges']) > logfc_threshold), 'names'])
            if save_csv:
                file_name = f"{cell_type}_markers_{cat}.csv"
                os.makedirs('schatbot/deg_res', exist_ok=True)
                data.to_csv(f'schatbot/deg_res/{file_name}', index=False)
                print(f"{cat} condition {cell_type} marker results saved to {file_name}")
            # Get description directly from JSON file
            description = None
            sample_mapping = get_mapping("media")
            if sample_mapping:
                description = sample_mapping.get('Sample description', {}).get(cat)
            result_list.append({
                "category": cat,
                "significant_genes": significant_genes,
                "description": description
            })
        return result_list
    except Exception as e:
        print(f"Error in analysis: {e}")
        return {}

def compare_cell_count(adata, cell_type):
    from annotation import unified_cell_type_handler
    cell_type = unified_cell_type_handler(cell_type)
    
    # Get all unique sample categories
    categories = adata.obs["Exp_sample_category"].unique()
    result_list = []
    
    for cat in categories:
        # Create masks for this category and cell type
        mask_sample = adata.obs["Exp_sample_category"] == cat
        mask_cells = adata.obs["cell_type"] == cell_type
        combined_mask = mask_sample & mask_cells
        
        # Count cells
        cell_count = combined_mask.sum()
        
        # Get description from JSON file
        description = None
        sample_mapping = get_mapping("media")
        if sample_mapping:
            description = sample_mapping.get('Sample description', {}).get(cat)
        
        result_list.append({
            "category": cat,
            "cell_count": cell_count,
            "description": description
        })
    
    return result_list

def repeat():
    return "Hello"

if __name__ == "__main__":
    import pickle
    with open('schatbot/annotated_adata/Overall_annotated_adata.pkl', 'rb') as f:
        adata = pickle.load(f)
        
    result = compare_cell_count(adata, "Immune cell")
    print(result)