import pandas as pd
import scanpy as sc
import numpy as np

def extract_top_genes_stats(adata, groupby='leiden', n_genes=25) -> pd.DataFrame:
    """
    Extracts statistics for the top genes from the rank_genes_groups result.
    """
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
    top_genes_stats_df = top_genes_stats_df.reset_index().rename(columns={'level_0': 'cluster', 'level_1': 'index'})
    return top_genes_stats_df

def calculate_cluster_statistics(adata, category, n_genes=25):
    """
    Calculates clustering statistics and returns top genes, mean expression, and expression proportion.
    """
    # Note: In the original code, markers are obtained from get_rag_and_markers(False)
    # Here we assume that function is imported from cluster_labeling or defined elsewhere.
    from .cluster_labeling import get_rag_and_markers  # Ensure this function is available
    from .file_utils import clear_directory  # if needed
    markers_data = get_rag_and_markers(False)
    markers = []
    for cell_type, cell_data in markers_data.items():
        markers += cell_data['genes']
    # Filter markers that exist in adata.raw.var_names
    def filter_existing_genes(adata, gene_list):
        return [gene for gene in gene_list if gene in adata.raw.var_names]
    markers = filter_existing_genes(adata, markers)
    markers = list(set(markers))

    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon', n_genes=n_genes)
    top_genes_df = extract_top_genes_stats(adata, groupby='leiden', n_genes=n_genes)
    if 'X_scVI' in adata.obsm:
        sc.tl.dendrogram(adata, groupby='leiden', use_rep='X_scVI')
    else:
        sc.tl.dendrogram(adata, groupby='leiden')
    marker_expression = sc.get.obs_df(adata, keys=['leiden'] + markers, use_raw=True)
    marker_expression.set_index('leiden', inplace=True)
    mean_expression = marker_expression.groupby('leiden').mean()
    expression_proportion = marker_expression.gt(0).groupby('leiden').mean()
    return top_genes_df, mean_expression, expression_proportion

def retreive_stats():
    """
    Loads pre-saved statistics and calculates cluster statistics, then returns a summary string.
    """
    with open("basic_data/mean_expression.json", 'r') as file:
        mean_expression = json.load(file)
    with open("basic_data/expression_proportion.json", 'r') as file:
        expression_proportion = json.load(file)
    from .sc_analysis import adata  # Use the global adata from sc_analysis
    top_genes_df, _, _ = calculate_cluster_statistics(adata, 'overall')
    from .cluster_labeling import get_rag_and_markers  # Ensure this function is available
    markers_data = get_rag_and_markers(False)
    markers = ', '.join(markers_data)
    explanation = ("Please analyze the clustering statistics and classify each cluster based on the following data: "
                   "Top Genes:Mean Expression: Expression Proportion:, based on statistical data: 1. top_genes_df: 25 top genes "
                   "expression within each cluster with its p_val, p_val_adj, and logfoldchange; 2. mean_expression of marker genes; "
                   "3. expression_proportion of marker genes; and give back the mapping dictionary in the format like: "
                   "group_to_cell_type = {'0': 'Myeloid cells','1': 'T cells', ...} without further explanation.")
    mean_expression_str = ", ".join([f"{k}: {v}" for k, v in mean_expression.items()])
    expression_proportion_str = ", ".join([f"{k}: {v}" for k, v in expression_proportion.items()])
    summary = (f"Explanation: {explanation}. Mean expression data: {mean_expression_str}. "
               f"Expression proportion data: {expression_proportion_str}. Top genes details: {top_genes_df}. "
               f"markers: {markers}.")
    return summary

def sample_differential_expression_genes_comparison(cell_type: str, sample_1: str, sample_2: str) -> str:
    """
    Performs differential gene expression analysis for the specified cell type between two patient conditions.
    Returns a summary string.
    """
    global adata, sample_mapping
    import pickle
    cell_type3 = cell_type.split()[0].capitalize()
    cell_type2 = cell_type.split()[0].capitalize() + " cell"
    cell_type_formatted = cell_type.split()[0].capitalize() + " cells"
    
    with open(f'annotated_adata/{cell_type_formatted}_annotated_adata.pkl', 'rb') as file:
        adata2 = pd.read_pickle(file)
    
    filtered_cells = adata2[adata2.obs['cell_type'].isin([cell_type_formatted])].copy()
    if filtered_cells.shape[0] == 0:
        filtered_cells = adata2[adata2.obs['cell_type'].isin([cell_type2])].copy()
    if filtered_cells.shape[0] == 0:
        filtered_cells = adata2[adata2.obs['cell_type'].isin([cell_type3])].copy()

    adata_filtered = filtered_cells[filtered_cells.obs['patient_name'].isin([sample_1, sample_2])].copy()
    unique_patients = adata_filtered.obs['patient_name'].astype(str).unique()
    if sample_1 not in unique_patients or sample_2 not in unique_patients:
        return f"Error: One or both patients ({sample_1}, {sample_2}) not found in dataset for cell type '{cell_type_formatted}'."
    
    sc.tl.rank_genes_groups(adata_filtered, groupby='patient_name', groups=[sample_2],
                             reference=sample_1, method='wilcoxon')
    results_post = {
        'genes': adata_filtered.uns['rank_genes_groups']['names'][sample_2],
        'logfoldchanges': adata_filtered.uns['rank_genes_groups']['logfoldchanges'][sample_2],
        'pvals': adata_filtered.uns['rank_genes_groups']['pvals'][sample_2],
        'pvals_adj': adata_filtered.uns['rank_genes_groups']['pvals_adj'][sample_2]
    }
    df_post = pd.DataFrame(results_post)
    significant_genes_post = df_post[(df_post['pvals_adj'] < 0.05) & (df_post['logfoldchanges'].abs() > 1)]
    significant_genes_post.to_csv('SGP.csv', index=False)
    summary = (
        f"Reference Sample: {sample_1}, Comparison Sample: {sample_2}\n"
        "Explanation: Differential gene expression analysis performed.\n"
        f"Significant Genes Data:\n{significant_genes_post.to_string(index=False)}\n"
        "Explanation of attributes:\n"
        "Genes: The names of the genes analyzed.\n"
        "Log Fold Changes: Positive values indicate upregulation in sample_2, negative indicate downregulation.\n"
        "P-values: Statistical significance values.\n"
        "Adjusted P-values: Significance after correction for multiple comparisons."
    )
    return summary