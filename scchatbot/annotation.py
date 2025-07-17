import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import pickle
import os
import ast
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from scchatbot.utils import get_rag, get_subtypes, save_analysis_results, explain_gene, get_mapping, filter_existing_genes, extract_genes, get_h5ad
import re
import scvi

# Cell type naming/standardization

def unified_cell_type_handler(cell_type):
    """
    Standardizes cell type names to match the new database format:
    - Singular form (ends with "cell" not "cells")
    - CD markers use "-positive" format
    - First word capitalized (except special abbreviations)
    - Handles multi-word cell types with proper capitalization
    """
    if not cell_type or not isinstance(cell_type, str):
        return "Unknown cell"
    # Clean and normalize input
    clean_type = cell_type.strip()
    # Handle CD markers: convert all formats to -positive/-negative
    # Convert + to -positive
    clean_type = re.sub(r'\bCD(\d+)\+', r'CD\1-positive', clean_type, flags=re.IGNORECASE)
    # Convert - to -negative
    clean_type = re.sub(r'\bCD(\d+)-(?!positive|negative)', r'CD\1-negative', clean_type, flags=re.IGNORECASE)
    # Convert space-separated format
    clean_type = re.sub(r'\bCD(\d+) positive\b', r'CD\1-positive', clean_type, flags=re.IGNORECASE)
    clean_type = re.sub(r'\bCD(\d+) negative\b', r'CD\1-negative', clean_type, flags=re.IGNORECASE)
    # Remove trailing "cell" or "cells" for processing
    flag = 0
    if clean_type.lower().endswith(' cells'):
        base_type = clean_type[:-6].strip()
        flag = 1
    elif clean_type.lower().endswith(' cell'):
        base_type = clean_type[:-5].strip()
        flag = 2
    else:
        base_type = clean_type
        flag = 3
    # Split into words for processing
    words = base_type.split()
    if not words:
        return "Unknown cell"
    # Process each word
    processed_words = []
    for i, word in enumerate(words):
        # Strip trailing punctuation for processing
        punct = ''
        m = re.match(r'^(.*?)([,.\.;:])$', word)
        if m:
            core_word, punct = m.groups()
        else:
            core_word = word
        word_to_process = core_word
        word_lower = word_to_process.lower()
        # Special handling for CD markers: uppercase CD and number, lowercase suffix
        cd_match = re.match(r'^(cd\d+)(?:[+\-]|(-positive|-negative))$', word_to_process, flags=re.IGNORECASE)
        if cd_match:
            cd_part = cd_match.group(1).upper()
            suffix = ''
            # handle explicit suffix (e.g., "-positive" or "-negative")
            suff_match = re.search(r'(-positive|-negative)', word_to_process, flags=re.IGNORECASE)
            if suff_match:
                suffix = suff_match.group(1).lower()
            processed_core = cd_part + suffix
            processed_words.append(processed_core + punct)
            continue
        elif word_lower in ['t', 'b', 'nk', 'th1', 'th2', 'th17', 'treg', 'dc']:
            processed_core = word_to_process.upper()
            processed_words.append(processed_core + punct)
        elif word_lower in ['alpha', 'beta', 'gamma', 'delta']:
            processed_core = word_lower
            processed_words.append(processed_core + punct)
        elif word_lower in ['and', 'or', 'of', 'in']:
            processed_core = word_lower
            processed_words.append(processed_core + punct)
        elif word_to_process in ['+', '-', '/', ',']:
            processed_core = word_to_process
            processed_words.append(processed_core + punct)
        else:
            # Capitalize first letter of regular words
            if i == 0 or words[i-1].lower() in ['and', 'or']:
                processed_core = word_to_process.capitalize()
            else:
                processed_core = word_to_process.lower()
            processed_words.append(processed_core + punct)
    # Join words and add "cell" at the end (singular)
    result = ' '.join(processed_words)
    # Special cases: immunocyte names should always be singular
    special = [
        'platelet', 'platelets',
        'erythrocyte', 'erythrocytes',
        'lymphocyte', 'lymphocytes',
        'monocyte', 'monocytes',
        'neutrophil', 'neutrophils',
        'eosinophil', 'eosinophils',
        'basophil', 'basophils'
    ]
    if result.lower() in special:
        # singularize by stripping trailing 's'
        singular = result.lower().rstrip('s')
        return singular.capitalize()
    # Ensure it ends with "cell" (singular)
    if flag == 1 or flag == 2:
        result += ' cell'
    return result

def standardize_cell_type(cell_type):
    """
    Standardize cell type strings for flexible matching with the new database.
    Converts to lowercase and handles CD markers consistently.
    """
    clean_type = cell_type.lower().strip()
    # Normalize CD markers to -positive/-negative format
    clean_type = re.sub(r'\bcd(\d+)\+', r'cd\1-positive', clean_type)
    clean_type = re.sub(r'\bcd(\d+)-(?!positive|negative)', r'cd\1-negative', clean_type)
    clean_type = re.sub(r'\bcd(\d+) positive\b', r'cd\1-positive', clean_type)
    clean_type = re.sub(r'\bcd(\d+) negative\b', r'cd\1-negative', clean_type)
    # Remove "cells" or "cell" suffix for base form
    if clean_type.endswith(' cells'):
        clean_type = clean_type[:-6].strip()
    elif clean_type.endswith(' cell'):
        clean_type = clean_type[:-5].strip()
    return clean_type

def get_possible_cell_types(cell_type):
    """
    Generate all possible forms of a cell type for flexible matching.
    """
    base_form = standardize_cell_type(cell_type)
    result = unified_cell_type_handler(cell_type)
    words = base_form.split()
    possible_types = [base_form]
    possible_types.append(result)
    if not result.lower().endswith("cells"):
        possible_types.append(f"{result} cells")
    if len(words) == 1:
        if words[0].endswith('s'):
            possible_types.append(words[0][:-1])
        else:
            possible_types.append(f"{words[0]}s")
    return list(dict.fromkeys(possible_types))

# Data processing for annotation

def preprocess_data(adata, sample_mapping=None):
    """
    Preprocess the AnnData object with consistent steps.
    """
    if not adata.var_names.is_unique:
        print("Warning: Gene names are not unique. Making them unique.")
        adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 20]
    non_mt_mask = ~adata.var['mt']
    adata = adata[:, non_mt_mask].copy()
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    if hasattr(adata, 'raw') and adata.raw is not None and not adata.raw.var_names.is_unique:
        print("Warning: Raw gene names are not unique. Making them unique.")
        adata.raw.var_names_make_unique()
    
    if sample_mapping:
        from scvi.model import SCVI
        sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True,
                                   flavor="seurat_v3", batch_key="Exp_sample_category")
        SCVI.setup_anndata(adata, layer="counts", categorical_covariate_keys=["Exp_sample_category"],
                          continuous_covariate_keys=['pct_counts_mt', 'total_counts'])
        
        if os.path.exists("scchatbot/scvi_model") and any(os.scandir("scchatbot/scvi_model")):
            model = SCVI.load(dir_path="scchatbot/scvi_model", adata=adata)
        else:
            model = scvi.model.SCVI(adata)
            model.train()
            model.save("scchatbot/scvi_model", overwrite=True)

        latent = model.get_latent_representation()
        adata.obsm['X_scVI'] = latent
        adata.layers['scvi_normalized'] = model.get_normalized_expression(library_size=1e4)
        sc.pp.neighbors(adata, use_rep='X_scVI')
    else:
        sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True, layer='counts', 
                                   flavor="seurat_v3")
        n_neighbors = min(15, int(0.5 * np.sqrt(adata.n_obs)))
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    return adata


def perform_clustering(adata, resolution=1, random_state=42):
    """
    Perform UMAP and Leiden clustering with consistent parameters.
    """
    sc.tl.umap(adata, random_state=random_state)
    sc.tl.leiden(adata, resolution=resolution, random_state=random_state)
    umap_df = adata.obsm['X_umap']
    adata.obs['UMAP_1'] = umap_df[:, 0]
    adata.obs['UMAP_2'] = umap_df[:, 1]
    return adata


def rank_genes(adata, groupby='leiden', method='wilcoxon', n_genes=25, key_added=None):
    """
    Rank genes by group with customizable key name.
    """
    if key_added is None:
        key_added = f'rank_genes_{groupby}'
    if len(adata.obs[groupby].unique()) <= 1:
        print(f"WARNING: Only one group found in {groupby}, cannot perform differential expression")
        adata.uns[key_added] = {
            'params': {'groupby': groupby},
            'names': np.zeros((1,), dtype=[('0', 'U50')])
        }
        return adata
    try:
        adata.obs = adata.obs.copy()
        sc.tl.rank_genes_groups(adata, groupby, method=method, n_genes=n_genes, key_added=key_added, use_raw=False)
    except ValueError as e:
        print(f"ERROR in rank_genes_groups: {e}")
        # Fallback to t-test if 'wilcoxon' fails due to small group size
        adata.obs = adata.obs.copy()
        sc.tl.rank_genes_groups(adata, groupby, method='t-test', n_genes=n_genes, key_added=key_added, use_raw=False)

    return adata


def create_marker_anndata(adata, markers, copy_uns=True, copy_obsm=True):
    """
    Create a copy of AnnData with only marker genes.
    """
    import scipy
    filtered_markers, updated_adata = filter_existing_genes(adata, markers)
    
    # Remove duplicates from markers
    filtered_markers = list(set(filtered_markers))
    
    if len(filtered_markers) == 0:
        print("WARNING: No marker genes found in the dataset!")
        return anndata.AnnData(
            X=scipy.sparse.csr_matrix((updated_adata.n_obs, 0)),
            obs=updated_adata.obs.copy()
        ), [], updated_adata
    
    # Determine which data source to use and get gene indices
    if hasattr(updated_adata, 'raw') and updated_adata.raw is not None:
        # Use raw data
        raw_var_names = updated_adata.raw.var_names
        raw_indices = [i for i, name in enumerate(raw_var_names) if name in filtered_markers]
        
        if len(raw_indices) == 0:
            print("WARNING: No marker genes found in raw data!")
            return anndata.AnnData(
                X=scipy.sparse.csr_matrix((updated_adata.n_obs, 0)),
                obs=updated_adata.obs.copy()
            ), [], updated_adata
        
        # Extract expression data
        if hasattr(updated_adata.raw, 'layers') and 'log1p' in updated_adata.raw.layers:
            X = updated_adata.raw.layers['log1p'][:, raw_indices].copy()
        else:
            X = updated_adata.raw.X[:, raw_indices].copy()
            # Check if data needs log transformation
            if scipy.sparse.issparse(X):
                max_val = X.max()
            else:
                max_val = np.max(X)
            if max_val > 100:
                print("Log-transforming marker data...")
                X = np.log1p(X)
        
        # Get variable metadata
        var = updated_adata.raw.var.iloc[raw_indices].copy()
        
    else:
        # Use main data
        main_var_names = updated_adata.var_names
        main_indices = [i for i, name in enumerate(main_var_names) if name in filtered_markers]
        
        if len(main_indices) == 0:
            print("WARNING: No marker genes found in main data!")
            return anndata.AnnData(
                X=scipy.sparse.csr_matrix((updated_adata.n_obs, 0)),
                obs=updated_adata.obs.copy()
            ), [], updated_adata
        
        X = updated_adata.X[:, main_indices].copy()
        var = updated_adata.var.iloc[main_indices].copy()
    
    
    # Create the marker AnnData object
    marker_adata = anndata.AnnData(
        X=X,
        obs=updated_adata.obs.copy(),
        var=var
    )   
    # Copy additional data if requested
    if copy_uns:
        marker_adata.uns = updated_adata.uns.copy()
    
    if copy_obsm:
        marker_adata.obsm = updated_adata.obsm.copy()
    
    if hasattr(updated_adata, 'obsp'):
        marker_adata.obsp = updated_adata.obsp.copy()
    
    return marker_adata, filtered_markers, updated_adata


def rank_ordering(adata_or_result, key=None, n_genes=25):
    """
    Extract top genes statistics from ranking results.
    """
    if isinstance(adata_or_result, anndata.AnnData):
        if key is None:
            rank_keys = [k for k in adata_or_result.uns.keys() if k.startswith('rank_genes_')]
            if not rank_keys:
                raise ValueError("No rank_genes results found in AnnData object")
            key = rank_keys[0]
        gene_names = adata_or_result.uns[key]['names']
    else:
        gene_names = adata_or_result['names']
    top_genes_stats = {group: {} for group in gene_names.dtype.names}
    for group in gene_names.dtype.names:
        top_genes_stats[group]['gene'] = gene_names[group][:n_genes]
    top_genes_stats_df = pd.concat({group: pd.DataFrame(top_genes_stats[group])
                                  for group in top_genes_stats}, axis=0)
    top_genes_stats_df = top_genes_stats_df.reset_index()
    top_genes_stats_df = top_genes_stats_df.rename(columns={'level_0': 'cluster', 'level_1': 'index'})
    return top_genes_stats_df

# Main annotation workflow

def initial_cell_annotation(resolution=1):
    """
    Generate initial UMAP clustering on the full dataset.
    """
    import matplotlib

    global sample_mapping
    matplotlib.use('Agg')
    path = get_h5ad("media", ".h5ad")
    if not path:
        # Return 5 None values to match expected unpacking
        return None, None, None, ".h5ad file isn't given, unable to generate UMAP.", None
    adata = sc.read_h5ad(path)
    sample_mapping = get_mapping("media")
    if sample_mapping:
        sample_column_name = sample_mapping.get("Sample name", "Sample")
        if sample_column_name in adata.obs.columns:
            adata.obs['Exp_sample_category'] = adata.obs[sample_column_name].map(sample_mapping["Sample categories"])
        else:
            print(f"Warning: '{sample_column_name}' column not found in adata.obs. Available columns:", list(adata.obs.columns))
            print("Proceeding without sample mapping.")
            sample_mapping = None
    adata = preprocess_data(adata, sample_mapping)
    adata = perform_clustering(adata, resolution=resolution)
    adata = rank_genes(adata, groupby='leiden', n_genes=25, key_added='rank_genes_all')
    markers = get_rag()
    marker_tree = markers.copy()
    markers = extract_genes(markers)
    adata_markers, filtered_markers, adata = create_marker_anndata(adata, markers)
    adata_markers = rank_genes(adata_markers, n_genes=25, key_added='rank_genes_markers')
    adata.uns['rank_genes_markers'] = adata_markers.uns['rank_genes_markers']
    use_rep = 'X_scVI' if sample_mapping else None
    if use_rep:
        sc.tl.dendrogram(adata, groupby='leiden', use_rep=use_rep)
    else:
        sc.tl.dendrogram(adata, groupby='leiden')
    adata.obs['cell_type'] = 'Unknown'
    save_analysis_results(
        adata, 
        prefix="scchatbot/runtime_data/basic_data/Overall cells", 
        save_dotplot=True, 
        markers=filtered_markers
    )
    top_genes_df = rank_ordering(adata, key='rank_genes_markers', n_genes=25)
    gene_dict = {}
    for cluster, group in top_genes_df.groupby("cluster"):
        gene_dict[cluster] = list(group["gene"])
    input_msg =(f"Top genes details: {gene_dict}. "
                f"Markers: {marker_tree}. ")

    messages = [
        SystemMessage(content="""
            You are a bioinformatics researcher that can do the cell annotation.
            The following are the data and its decription that you will receive.
            * 1. Gene list: top 25 cells arragned from the highest to lowest expression levels in each cluster.
            * 2. Marker tree: marker genes that can help annotating cell type.           
            
            Identify the cell type for each cluster using the following markers in the marker tree.
            This means you will have to use the markers to annotate the cell type in the gene list. 
            Provide your result as the most specific cell type that is possible to be determined.
            
            Provide your output in the following example format:
            Analysis: group_to_cell_type = {'0': 'Cell type','1': 'Cell type','2': 'Cell type','3': 'Cell type','4': 'Cell type', ...}.
            
            Strictly adhere to follwing rules:
            * 1. Adhere to the dictionary format and do not include any additional words or explanations.
            * 2. The cluster number in the result dictionary must be arranged with raising power.
            * 3. The annotated cell type must exist in the marker tree.
            """),
        HumanMessage(content=input_msg)
    ]
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )
    results = model.invoke(messages)
    annotation_result = results.content
    adata = label_clusters(adata=adata, cell_type="Overall", annotation_result=annotation_result)

    #adding
    explanation = explain_gene(gene_dict=gene_dict, marker_tree=marker_tree, annotation_result=annotation_result)
    print ("EXPLANATION", explanation)
    #done adding

    return gene_dict, marker_tree, adata, explanation, annotation_result


def process_cells(adata, cell_type, resolution=None):
    """
    Process specific cell type with full workflow - FIXED VERSION
    """
    print(f"🔍 Processing cell type: {cell_type}")
    resolution = 1 if resolution is None else resolution
    
    # Get subtypes from database
    markers_tree = get_subtypes(cell_type)
    # print(f"🔍 Retrieved markers_tree: {markers_tree}")
    
    # 🚨 NEW: Check if cell type has no subtypes (leaf node)
    if not markers_tree or len(markers_tree) == 0:
        print(f"✅ '{cell_type}' is a leaf node with no subtypes.")
        print(f"✅ No further refinement possible.")
        return {
            "status": "leaf_node",
            "message": f"'{cell_type}' has no subtypes in the database. This is the most specific level available.",
            "cell_type": cell_type,
            "subtypes_available": False
        }
    
    # Continue with original logic only if subtypes exist
    standardized = unified_cell_type_handler(cell_type)
    standardized_list = [standardized]
    
    mask = adata.obs["cell_type"].isin(standardized_list)
    if mask.sum() == 0:
        print(f"❌ No cells found with type '{standardized}'")
        return {
            "status": "no_cells_found", 
            "message": f"No cells found with type '{cell_type}' in the dataset.",
            "cell_type": cell_type
        }
    
    print(f"🔍 Found {mask.sum()} cells of type '{standardized}'")
    
    # Rest of the original function continues...
    filtered = adata[mask].copy()
    mask_idx = adata.obs.index[mask]
    filtered_idx = filtered.obs.index
    
    sc.tl.pca(filtered, svd_solver="arpack")
    sc.pp.neighbors(filtered)
    filtered = perform_clustering(filtered, resolution=resolution)
    sc.tl.dendrogram(filtered, groupby="leiden")
    
    base = standardize_cell_type(cell_type).replace(" ", "_").lower()
    rank_key = f"rank_genes_{base}"
    filtered = rank_genes(filtered, key_added=rank_key)
    
    markers_list = extract_genes(markers_tree)
    existing_markers = [
        g for g in markers_list
        if g in adata.raw.var_names or g in adata.obs.columns
    ]
    
    # 🚨 NEW: Additional check for sufficient markers
    if len(existing_markers) < 3:
        print(f"⚠️ Only {len(existing_markers)} markers available for '{cell_type}' subtypes.")
        print(f"⚠️ Insufficient markers for reliable subtype annotation.")
        return {
            "status": "insufficient_markers",
            "message": f"Only {len(existing_markers)} subtype markers found for '{cell_type}'. Need at least 3 for reliable annotation.",
            "cell_type": cell_type,
            "available_markers": existing_markers
        }
    
    # Continue with rest of the original process_cells logic...
    missing = set(markers_list) - set(existing_markers)
    if missing:
        print(f"⚠️ dropping {len(missing)} missing markers:", missing)
    markers_list = existing_markers
    if len(markers_list) >= 5:
        madata, _, _ = create_marker_anndata(filtered, markers_list)
        try:
            mk = f"rank_markers_{base}"
            madata = rank_genes(madata, key_added=mk)
            filtered.uns[mk] = madata.uns[mk]
            top_df = rank_ordering(madata, key=mk, n_genes=25)
        except Exception as e:
            print(f"  ▶ Marker ranking failed: {e}")
            top_df = rank_ordering(filtered, key=rank_key, n_genes=25)
    else:
        top_df = rank_ordering(filtered, key=rank_key, n_genes=25)
    gene_dict = {grp: list(gr["gene"]) for grp, gr in top_df.groupby("cluster")}
    save_analysis_results(
        filtered,
        prefix=f"process_cell_data/{standardized}",
        save_dotplot=bool(len(markers_list) >= 5),
        markers=markers_list
    )
    prompt = f"Top genes details: {gene_dict}. Markers: {markers_tree}."
    messages = [
        SystemMessage(content="""
            You are a bioinformatics researcher that can do the cell annotation.
            The following are the data and its decription that you will receive.
            * 1. Gene list: top 25 cells arragned from the highest to lowest expression levels in each cluster.
            * 2. Marker tree: marker genes that can help annotating cell type.           
            
            Identify the cell type for each cluster using the following markers in the marker tree.
            This means you will have to use the markers to annotate the cell type in the gene list. 
            Provide your result as the most specific cell type that is possible to be determined.
            
            Provide your output in the following example format:
            Analysis: group_to_cell_type = {'0': 'Cell type','1': 'Cell type','2': 'Cell type','3': 'Cell type','4': 'Cell type', ...}.
            
            Strictly adhere to follwing rules:
            * 1. Adhere to the dictionary format and do not include any additional words or explanations.
            * 2. The cluster number in the result dictionary must be arranged with raising power.
            * 3. The annotated cell type must exist in the marker tree.
        """),
        HumanMessage(content=prompt),
    ]
    results = ChatOpenAI(model="gpt-4o", temperature=0).invoke(messages)
    print("RESULTS", results.content)
    annotation_result = results.content
    annotated_filtered = label_clusters(
        annotation_result=annotation_result,
        cell_type=cell_type,
        adata=filtered
    ) 
    adata.obs.loc[mask, 'cell_type'] = annotated_filtered.obs['cell_type']
    for idx in filtered_idx[:3]:
        print(f"    • {idx}: filtered={annotated_filtered.obs.at[idx,'cell_type']} | adata={adata.obs.at[idx,'cell_type']}")
    umap_dir = "umaps/annotated"
    os.makedirs(umap_dir, exist_ok=True)
    overall_df = adata.obs[["UMAP_1", "UMAP_2", "cell_type"]].copy()
    if "patient_name" in adata.obs.columns:
        overall_df["patient_name"] = adata.obs["patient_name"]
    csv_path = os.path.join(umap_dir, "Overall cells_umap_data.csv")
    overall_df.to_csv(csv_path, index=False)
    df_check = pd.read_csv(csv_path)

    #adding
    explanation = explain_gene(gene_dict=gene_dict, marker_tree=markers_tree, annotation_result=annotation_result)
    final_result = (
        f"Annotation complete for {cell_type}.\n"
        f"• Annotation Result: {annotation_result}\n"
        f"• Top-genes per cluster: {gene_dict}\n"
        f"• Marker-tree: {markers_tree}\n"
        f"• Explanation: {explanation}"
    )    #done adding
    return final_result


# 🚨 ALSO ADD: Modified calling logic to handle the new return format
def handle_process_cells_result(adata, cell_type, resolution=None):
    """
    Wrapper function to handle the new process_cells return format
    """
    result = process_cells(adata, cell_type, resolution)
    
    # Handle special return cases
    if isinstance(result, dict) and "status" in result:
        if result["status"] == "leaf_node":
            print(f"✅ {result['message']}")
            return None  # or return some default response
        elif result["status"] == "no_cells_found":
            print(f"❌ {result['message']}")
            return None
        elif result["status"] == "insufficient_markers":
            print(f"⚠️ {result['message']}")
            return None
    
    # If we get here, it's the normal return with annotation results
    return result


def label_clusters(annotation_result, cell_type, adata):
    """
    Label clusters with consistent cell type handling.
    """
    standardized_name = unified_cell_type_handler(cell_type)
    base_form = standardize_cell_type(cell_type).lower()
    try:
        adata = adata.copy()
        start_idx = annotation_result.find("{")
        end_idx = annotation_result.rfind("}") + 1
        str_map = annotation_result[start_idx:end_idx]
        map2 = ast.literal_eval(str_map)
        map2 = {str(key): value for key, value in map2.items()}
        if base_form == "overall":
            adata.obs['cell_type'] = 'Unknown'
            for group, cell_type_value in map2.items():
                adata.obs.loc[adata.obs['leiden'] == group, 'cell_type'] = cell_type_value
            save_analysis_results(
                adata,
                prefix=f"umaps/annotated/{standardized_name}",
                save_dendrogram=False,
                save_dotplot=False
            )
            fname = f'scchatbot/annotated_adata/{standardized_name}_annotated_adata.pkl'
            with open(fname, "wb") as file:
                pickle.dump(adata, file)
        else:
            specific_cells = adata.copy()
            specific_cells.obs['cell_type'] = 'Unknown'
            for group, cell_type_value in map2.items():
                specific_cells.obs.loc[specific_cells.obs['leiden'] == group, 'cell_type'] = cell_type_value
            save_analysis_results(
                specific_cells,
                prefix=f"umaps/annotated/{standardized_name}",
                save_dendrogram=False,
                save_dotplot=False
            )
            fname = f'scchatbot/annotated_adata/{standardized_name}_annotated_adata.pkl'
            with open(fname, "wb") as file:
                pickle.dump(specific_cells, file)
            return specific_cells
    except (SyntaxError, ValueError) as e:
        print(f"Error in parsing the map: {e}")
    return adata


# if __name__ == "__main__":
#     gene_dict, marker_tree, adata, explanation, annotation_result = initial_cell_annotation()