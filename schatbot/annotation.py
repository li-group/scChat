import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import pickle
import os
import ast
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from .utils import get_rag, get_subtypes, save_analysis_results

# Cell type naming/standardization

def unified_cell_type_handler(cell_type):
    """
    Standardizes cell type names with proper handling for special cases.
    """
    known_cell_types = {
        "platelet": "Platelets",
        "platelets": "Platelets",
        "lymphocyte": "Lymphocytes",
        "lymphocytes": "Lymphocytes",
        "natural killer cell": "Natural killer cells",
        "natural killer cells": "Natural killer cells",
        "plasmacytoid dendritic cell": "Plasmacytoid dendritic cells",
        "plasmacytoid dendritic cells": "Plasmacytoid dendritic cells"
    }
    clean_type = cell_type.lower().strip()
    if clean_type.endswith(' cells'):
        clean_type = clean_type[:-6].strip()
    elif clean_type.endswith(' cell'):
        clean_type = clean_type[:-5].strip()
    if clean_type in known_cell_types:
        return known_cell_types[clean_type]
    words = clean_type.split()
    if len(words) == 1:
        if words[0].endswith('s') and not words[0].endswith('ss'):
            return words[0].capitalize()
        else:
            return f"{words[0].capitalize()} cells"
    elif len(words) == 2:
        special_first_words = ['t', 'b', 'nk', 'cd4', 'cd8']
        if words[0].lower() in special_first_words:
            return f"{words[0].upper()} cells"
        else:
            return f"{words[0].capitalize()} {words[1].capitalize()} cells"
    elif len(words) >= 3:
        return f"{words[0].capitalize()} {' '.join(words[1:])} cells"
    return f"{cell_type} cells"


def standardize_cell_type(cell_type):
    """
    Standardize cell type strings for flexible matching.
    Handles multi-word cell types, singular/plural forms, and capitalization.
    """
    clean_type = cell_type.lower().strip()
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
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    if hasattr(adata, 'raw') and adata.raw is not None and not adata.raw.var_names.is_unique:
        print("Warning: Raw gene names are not unique. Making them unique.")
        adata.raw.var_names_make_unique()
    if sample_mapping:
        from scvi.model import SCVI
        sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True, layer='counts', 
                                   flavor="seurat_v3", batch_key="Sample")
        SCVI.setup_anndata(adata, layer="counts", categorical_covariate_keys=["Sample"],
                          continuous_covariate_keys=['pct_counts_mt', 'total_counts'])
        model = SCVI.load(dir_path="schatbot/glioma_scvi_model", adata=adata)
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


def perform_clustering(adata, resolution=2, random_state=42):
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
        sc.tl.rank_genes_groups(adata, groupby, method=method, n_genes=n_genes, key_added=key_added, use_raw=False)
    except Exception as e:
        print(f"ERROR in rank_genes_groups: {e}")
        print("Falling back to t-test method")
        try:
            sc.tl.rank_genes_groups(adata, groupby, method='t-test', n_genes=n_genes, key_added=key_added, use_raw=False)
        except Exception as e2:
            print(f"ERROR with fallback method: {e2}")
            adata.uns[key_added] = {
                'params': {'groupby': groupby},
                'names': np.zeros((1,), dtype=[('0', 'U50')])
            }
    return adata


def create_marker_anndata(adata, markers, copy_uns=True, copy_obsm=True):
    """
    Create a copy of AnnData with only marker genes.
    """
    from .utils import filter_existing_genes
    import scipy
    markers = filter_existing_genes(adata, markers)
    markers = list(set(markers))
    if len(markers) == 0:
        print("WARNING: No marker genes found in the dataset!")
        return anndata.AnnData(
            X=scipy.sparse.csr_matrix((adata.n_obs, 0)),
            obs=adata.obs.copy()
        ), []
    if hasattr(adata, 'raw') and adata.raw is not None:
        raw_indices = [i for i, name in enumerate(adata.raw.var_names) if name in markers]
        if hasattr(adata.raw, 'layers') and 'log1p' in adata.raw.layers:
            X = adata.raw.layers['log1p'][:, raw_indices].copy()
        else:
            X = adata.raw.X[:, raw_indices].copy()
            if scipy.sparse.issparse(X):
                max_val = X.max()
            else:
                max_val = np.max(X)
            if max_val > 100:
                print("Log-transforming marker data...")
                X = np.log1p(X)
    else:
        main_indices = [i for i, name in enumerate(adata.var_names) if name in markers]
        X = adata.X[:, main_indices].copy()
    var = adata.var.iloc[main_indices].copy() if 'main_indices' in locals() else adata.raw.var.iloc[raw_indices].copy()
    marker_adata = anndata.AnnData(
        X=X,
        obs=adata.obs.copy(),
        var=var
    )
    if copy_uns:
        marker_adata.uns = adata.uns.copy()
    if copy_obsm:
        marker_adata.obsm = adata.obsm.copy()
    if hasattr(adata, 'obsp'):
        marker_adata.obsp = adata.obsp.copy()
    return marker_adata, markers


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

def initial_cell_annotation(resolution=2):
    """
    Generate initial UMAP clustering on the full dataset.
    """
    import matplotlib
    from .utils import get_h5ad, get_mapping, extract_genes
    global sample_mapping
    matplotlib.use('Agg')
    path = get_h5ad("media", ".h5ad")
    if not path:
        return ".h5ad file isn't given, unable to generate UMAP."
    adata = sc.read_h5ad(path)
    sample_mapping = get_mapping("media")
    if sample_mapping:
        adata.obs['patient_name'] = adata.obs['Sample'].map(sample_mapping)
    adata = preprocess_data(adata, sample_mapping)
    adata = perform_clustering(adata, resolution=resolution)
    adata = rank_genes(adata, groupby='leiden', n_genes=25, key_added='rank_genes_all')
    markers = get_rag()
    marker_tree = markers.copy()
    markers = extract_genes(markers)
    adata_markers, filtered_markers = create_marker_anndata(adata, markers)
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
        prefix="schatbot/runtime_data/basic_data/Overall cells", 
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
            """),
        HumanMessage(content=input_msg)
    ]
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )
    results = model.invoke(messages)
    annotation_result = results.content
    label_clusters(adata=adata, cell_type="Overall", annotation_result=annotation_result)
    return gene_dict, marker_tree, adata


def process_cells(adata, cell_type, resolution=None):
    """
    Process specific cell type with full workflow:
      1) Subset and recluster
      2) Rank genes (general + marker-based)
      3) GPT‐driven annotation → label_clusters
      4) Save UMAP/dendrogram/dot-plot CSVs + pickles
      5) Merge back into full adata + resave overall UMAP
      6) Return annotation dict as string
    """
    from .utils import extract_genes
    resolution = 1 if resolution is None else resolution
    possible_types = get_possible_cell_types(cell_type)
    standardized = unified_cell_type_handler(cell_type)
    mask = adata.obs["cell_type"].isin(possible_types)
    if mask.sum() == 0:
        return {}, None, None
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
    markers_tree = get_subtypes(cell_type)
    markers_list = extract_genes(markers_tree)
    existing_markers = [
        g for g in markers_list
        if g in adata.raw.var_names or g in adata.obs.columns
    ]
    missing = set(markers_list) - set(existing_markers)
    if missing:
        print(f"⚠️ dropping {len(missing)} missing markers:", missing)
    markers_list = existing_markers
    if len(markers_list) >= 5:
        madata, _ = create_marker_anndata(filtered, markers_list)
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
        """),
        HumanMessage(content=prompt),
    ]
    results = ChatOpenAI(model="gpt-4o", temperature=0).invoke(messages)
    annotation_result = results.content
    out_pkl = f"annotated_adata/{standardized}_annotated_adata.pkl"
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    pickle.dump(filtered, open(out_pkl, "wb"))
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
    return str(annotation_result)


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
            fname = f'annotated_adata/{standardized_name}_annotated_adata.pkl'
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
            fname = f'annotated_adata/{standardized_name}_annotated_adata.pkl'
            with open(fname, "wb") as file:
                pickle.dump(specific_cells, file)
            return specific_cells
    except (SyntaxError, ValueError) as e:
        print(f"Error in parsing the map: {e}")
    return adata
