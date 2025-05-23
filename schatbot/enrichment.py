import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from gprofiler import GProfiler
from typing import List, Dict, Any
import gseapy as gp
import re

def dea(adata, cell_type):
    try:
        adata_modified = adata.copy()
        mask = adata_modified.obs["cell_type"] == str(cell_type)
        group_name = str(cell_type)
        key_name = str(group_name) + "_markers"
        adata_modified.obs[str(group_name)] = "Other"
        adata_modified.obs.loc[mask, str(group_name)] = str(group_name)
        sc.tl.rank_genes_groups(adata_modified, groupby=str(group_name), method="wilcoxon", n_genes=100, key_added=key_name, use_raw_=False)
        
        return adata_modified
    except (SyntaxError, ValueError) as e:
        print(f"Error in parsing the map: {e}")
    return adata


def get_significant_gene(adata, cell_type, logfc_threshold=1, pval_threshold=0.05):
    """Extract significant genes and their log‐fold changes for a cell type."""
    key = f"{cell_type}_markers"
    adata_mod = dea(adata, cell_type)
    df = sc.get.rank_genes_groups_df(adata_mod, group=cell_type, key=key)
    gene_to_logfc = dict(zip(df['names'], df['logfoldchanges']))
    sig = df.loc[
        (df['pvals_adj'] < pval_threshold) &
        (df['logfoldchanges'].abs() > logfc_threshold),
        'names'
    ].tolist()
    return sig, gene_to_logfc

def reactome_enrichment(cell_type, significant_genes, gene_to_logfc, p_value_threshold=0.05, top_n_terms=10, 
                        save_raw=True, save_summary=True, save_plots=True, output_prefix="reactome"):
    """
    Performs Reactome pathway enrichment analysis on differentially expressed genes for a specific cell type.
    
    Parameters:
    -----------
    cell_type : str
        Cell type to analyze.
    significant_genes : list
        A list of significant gene symbols (strings).
    gene_to_logfc : dict
        A dictionary mapping gene symbols (str) to their log2 fold change values (float).
    p_value_threshold : float, default=0.05
        P-value threshold for significant enrichment terms.
    top_n_terms : int, default=10
        Number of top terms to display in plots.
    save_raw : bool, default=True
        Whether to save raw enrichment results.
    save_summary : bool, default=True
        Whether to save summary enrichment results.
    save_plots : bool, default=True
        Whether to save plots.
    output_prefix : str, default="reactome"
        Prefix to use for output filenames.
        
    Returns:
    --------
    dict
        Dictionary containing DataFrames of raw and processed enrichment results.
    """   
    os.makedirs(output_prefix, exist_ok=True)

    results = {
        'raw_results': None,
        'summary_results': None
    }

    if not significant_genes:
        return results
    
    # Initialize gProfiler
    gp_reactome = GProfiler(return_dataframe=True)
    
    # Run Reactome enrichment analysis
    reactome_enrichment_results = gp_reactome.profile(
        organism='hsapiens',
        query=significant_genes,
        sources=['REAC'],
        user_threshold=0.05,
        significance_threshold_method='fdr',
        all_results=True,
        no_evidences=False,
        ordered=False
    )
    
    results['raw_results'] = reactome_enrichment_results
    
    if not reactome_enrichment_results.empty:
        # Save raw Reactome results
        if save_raw:
            # raw_output_filename = f"{output_prefix}_results_raw_{cell_type}.csv"
            raw_output_filename = os.path.join(
            output_prefix,
            f"results_raw_{cell_type}.csv"
            )
            reactome_enrichment_results.to_csv(raw_output_filename, index=False)

        
        # Post-processing
        reactome_enrichment_dict = {}
        for index, row in reactome_enrichment_results.iterrows():
            term_id = row['native']
            if not all(col in row for col in ['name', 'p_value', 'source', 'intersection_size', 'intersections']):
                continue
            
            # Process intersections
            intersecting_genes = []
            if isinstance(row['intersections'], list):
                intersecting_genes = row['intersections']
            elif isinstance(row['intersections'], str):
                intersecting_genes = [gene.strip() for gene in row['intersections'].split(',') if gene.strip()]
            else:
                continue
            
            # Calculate average log2FC
            avg_log2fc = 0
            gene_count_in_term = 0
            if intersecting_genes:
                sum_log2fc = 0
                for gene in intersecting_genes:
                    if gene in gene_to_logfc:
                        sum_log2fc += gene_to_logfc[gene]
                        gene_count_in_term += 1
                if gene_count_in_term > 0:
                    avg_log2fc = sum_log2fc / gene_count_in_term
            
            term_size = row['term_size'] if 'term_size' in row else 0
            intersection_size = row['intersection_size'] if 'intersection_size' in row else len(intersecting_genes)
            gene_ratio = intersection_size / term_size if term_size > 0 else 0
            
            reactome_enrichment_dict[term_id] = {
                'name': row['name'],
                'p_value': row['p_value'],
                'source': row['source'],
                'intersection_size': intersection_size,
                'term_size': term_size,
                'gene_ratio': gene_ratio,
                'calculated_intersection_count': gene_count_in_term,
                'avg_log2fc': avg_log2fc
            }
        
        if reactome_enrichment_dict:
            # Create output dataframe
            reactome_output_df = pd.DataFrame([
                {
                    'Term_ID': term_id,
                    'Term': info['name'],
                    'p_value': info['p_value'],
                    'intersection_size': info.get('intersection_size', 0),
                    'term_size': info.get('term_size', 0),
                    'gene_ratio': info.get('gene_ratio', 0),
                    'calculated_intersection_count': info.get('calculated_intersection_count', 0),
                    'avg_log2fc': info.get('avg_log2fc', 0)
                }
                for term_id, info in reactome_enrichment_dict.items()
            ])
            
            # Sort by p-value
            reactome_output_df = reactome_output_df.sort_values('p_value')
            results['summary_results'] = reactome_output_df
            
            # Save summarized results
            if save_summary:
                # summary_output_filename = f'{output_prefix}_results_summary_{cell_type}.csv'
                summary_output_filename = os.path.join(
                    output_prefix,
                    f"results_summary_{cell_type}.csv"
                )
                reactome_output_df.to_csv(summary_output_filename, index=False)
            
            # Visualization
            if save_plots:
                # Filter for significant terms
                significant_df = reactome_output_df[reactome_output_df['p_value'] < p_value_threshold].copy()
                
                if not significant_df.empty:
                    # Handle potential zero p-values
                    if (significant_df['p_value'] == 0).any():
                        min_non_zero_p = significant_df[significant_df['p_value'] > 0]['p_value'].min()
                        replacement_p = min_non_zero_p / 1000 if pd.notna(min_non_zero_p) and min_non_zero_p > 0 else 1e-300
                        significant_df['p_value'] = significant_df['p_value'].replace(0, replacement_p)
                    
                    # Calculate -log10(p-value)
                    significant_df['-log10(p_value)'] = -np.log10(significant_df['p_value'])
                    
                    # Select top N terms
                    significant_df = significant_df.sort_values('p_value', ascending=True)
                    top_n_actual = min(top_n_terms, len(significant_df))
                    
                    if top_n_actual > 0:
                        plot_df = significant_df.head(top_n_actual)
                        
                        # --- Bar chart ---
                        # Sort by gene_ratio for bar plot ordering
                        plot_df_bar = plot_df.sort_values('gene_ratio', ascending=False)
                        plt.figure(figsize=(10, max(6, top_n_actual * 0.5)))  # Adjust height based on N
                        sns.barplot(
                            x='gene_ratio',
                            y='Term',
                            data=plot_df_bar,
                            palette='viridis',
                            orient='h'
                        )
                        plt.title(f'Top {top_n_actual} Enriched Reactome Pathways ({cell_type}) by Gene Ratio')
                        plt.xlabel('Gene Ratio')
                        plt.ylabel('Reactome Pathway')
                        plt.tight_layout()
                        barplot_filename = f'{output_prefix}_barplot_{cell_type}.png'
                        plt.savefig(barplot_filename, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Saved bar plot to {barplot_filename}")
                        
                        # --- Dot plot ---
                        # Sort by gene_ratio for dot plot ordering
                        plot_df_dot = plot_df.sort_values('gene_ratio', ascending=False)
                        plt.figure(figsize=(12, max(6, top_n_actual * 0.5)))  # Adjust height
                        scatter = sns.scatterplot(
                            data=plot_df_dot,
                            y='Term',
                            x='gene_ratio',
                            size='intersection_size',
                            hue='-log10(p_value)',
                            palette='viridis_r',
                            sizes=(40, 400),
                            edgecolor='grey',
                            linewidth=0.5,
                            alpha=0.8
                        )
                        plt.title(f'Top {top_n_actual} Enriched Reactome Pathways ({cell_type})')
                        plt.ylabel('Reactome Pathway')
                        plt.xlabel('Gene Ratio')
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='Legend')
                        handles, labels = scatter.get_legend_handles_labels()
                        legend_title_map = {'size': 'Intersection Size', 'hue': '-log10(P-value)'}
                        new_handles = []
                        new_labels = []
                        for i, label in enumerate(labels):
                            if label in legend_title_map:
                                new_labels.append(legend_title_map[label])
                                new_handles.append(handles[i])
                            elif handles[i] is not None:
                                new_labels.append(label)
                                new_handles.append(handles[i])
                        scatter.legend(new_handles, new_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
                        plt.tight_layout(rect=[0, 0, 0.85, 1])
                        dotplot_filename = f'{output_prefix}_dotplot_{cell_type}.png'
                        plt.savefig(dotplot_filename, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Saved dot plot to {dotplot_filename}")
                    else:
                        print(f"No significant terms found for plotting after filtering (p < {p_value_threshold}) for {cell_type}.")
                else:
                    print(f"No significant terms found passing p-value threshold {p_value_threshold} for {cell_type}.")
    
    return results


def go_enrichment(cell_type, significant_genes, gene_to_logfc, p_value_threshold=0.05, top_n_terms=10, 
                  go_domains=['BP', 'MF', 'CC'], save_raw=True, save_summary=True, save_plots=True,
                  output_prefix="go"):
    """
    Performs Gene Ontology enrichment analysis on differentially expressed genes for a specific cell type.
    
    Parameters:
    -----------
    cell_type : str
        Cell type to analyze.
    significant_genes : list
        A list of significant gene symbols (strings).
    gene_to_logfc : dict
        A dictionary mapping gene symbols (str) to their log2 fold change values (float).
    p_value_threshold : float, default=0.05
        P-value threshold for significant enrichment terms.
    top_n_terms : int, default=10
        Number of top terms to display in plots.
    go_domains : list, default=['BP', 'MF', 'CC']
        GO domains to analyze (BP: Biological Process, MF: Molecular Function, CC: Cellular Component)
    save_raw : bool, default=True
        Whether to save raw enrichment results.
    save_summary : bool, default=True
        Whether to save summary enrichment results.
    save_plots : bool, default=True
        Whether to save plots.
    output_prefix : str, default="go"
        Prefix to use for output filenames.
        
    Returns:
    --------
    dict
        Dictionary containing DataFrames of raw and processed enrichment results for each GO domain.
    """   
    results = {
        'raw_results': {},
        'summary_results': {}
    }
    
    if not significant_genes:
        return results
    
    # Domain to source mapping
    domain_sources = {
        'BP': ['GO:BP'],
        'MF': ['GO:MF'],
        'CC': ['GO:CC']
    }
    
    # Initialize gProfiler
    gp = GProfiler(return_dataframe=True)
    
    for domain in go_domains:
        if domain not in domain_sources:
            continue
            
        sources = domain_sources[domain]
        domain_prefix = f"{output_prefix}_{domain.lower()}"
        os.makedirs(domain_prefix, exist_ok=True)
        # Run GO enrichment analysis
        go_enrichment_results = gp.profile(
            organism='hsapiens',
            query=significant_genes,
            sources=sources,
            user_threshold=0.05,
            significance_threshold_method='fdr',
            all_results=True,
            no_evidences=False,
            ordered=False
        )
        
        results['raw_results'][domain] = go_enrichment_results
        
        if not go_enrichment_results.empty:
            # Save raw GO results
            if save_raw:
                # raw_output_filename = f"{domain_prefix}_results_raw_{cell_type}.csv"
                raw_output_filename = os.path.join(
                    domain_prefix,
                    f"results_raw_{cell_type}.csv"
                )
                go_enrichment_results.to_csv(raw_output_filename, index=False)
            
            # Post-processing
            go_enrichment_dict = {}
            for index, row in go_enrichment_results.iterrows():
                term_id = row['native']
                if not all(col in row for col in ['name', 'p_value', 'source', 'intersection_size', 'intersections']):
                    continue
                
                # Process intersections
                intersecting_genes = []
                if isinstance(row['intersections'], list):
                    intersecting_genes = row['intersections']
                elif isinstance(row['intersections'], str):
                    intersecting_genes = [gene.strip() for gene in row['intersections'].split(',') if gene.strip()]
                else:
                    continue
                
                # Calculate average log2FC
                avg_log2fc = 0
                gene_count_in_term = 0
                if intersecting_genes:
                    sum_log2fc = 0
                    for gene in intersecting_genes:
                        if gene in gene_to_logfc:
                            sum_log2fc += gene_to_logfc[gene]
                            gene_count_in_term += 1
                    if gene_count_in_term > 0:
                        avg_log2fc = sum_log2fc / gene_count_in_term
                
                term_size = row['term_size'] if 'term_size' in row else 0
                intersection_size = row['intersection_size'] if 'intersection_size' in row else len(intersecting_genes)
                gene_ratio = intersection_size / term_size if term_size > 0 else 0
                
                go_enrichment_dict[term_id] = {
                    'name': row['name'],
                    'p_value': row['p_value'],
                    'source': row['source'],
                    'intersection_size': intersection_size,
                    'term_size': term_size,
                    'gene_ratio': gene_ratio,
                    'calculated_intersection_count': gene_count_in_term,
                    'avg_log2fc': avg_log2fc
                }
            
            if go_enrichment_dict:
                # Create output dataframe
                go_output_df = pd.DataFrame([
                    {
                        'Term_ID': term_id,
                        'Term': info['name'],
                        'p_value': info['p_value'],
                        'intersection_size': info.get('intersection_size', 0),
                        'term_size': info.get('term_size', 0),
                        'gene_ratio': info.get('gene_ratio', 0),
                        'calculated_intersection_count': info.get('calculated_intersection_count', 0),
                        'avg_log2fc': info.get('avg_log2fc', 0)
                    }
                    for term_id, info in go_enrichment_dict.items()
                ])
                
                # Sort by p-value
                go_output_df = go_output_df.sort_values('p_value')
                results['summary_results'][domain] = go_output_df
                
                # Save summarized results
                if save_summary:
                    # summary_output_filename = f'{domain_prefix}_results_summary_{cell_type}.csv'
                    summary_output_filename = os.path.join(
                        domain_prefix,
                        f"results_summary_{cell_type}.csv"
                    )
                    go_output_df.to_csv(summary_output_filename, index=False)
                
                # Visualization
                if save_plots:
                    # Filter for significant terms
                    significant_df = go_output_df[go_output_df['p_value'] < p_value_threshold].copy()
                    
                    if not significant_df.empty:
                        # Handle potential zero p-values
                        if (significant_df['p_value'] == 0).any():
                            min_non_zero_p = significant_df[significant_df['p_value'] > 0]['p_value'].min()
                            replacement_p = min_non_zero_p / 1000 if pd.notna(min_non_zero_p) and min_non_zero_p > 0 else 1e-300
                            significant_df['p_value'] = significant_df['p_value'].replace(0, replacement_p)
                        
                        # Calculate -log10(p-value)
                        significant_df['-log10(p_value)'] = -np.log10(significant_df['p_value'])
                        
                        # Select top N terms
                        significant_df = significant_df.sort_values('p_value', ascending=True)
                        top_n_actual = min(top_n_terms, len(significant_df))
                        
                        if top_n_actual > 0:
                            plot_df = significant_df.head(top_n_actual)
                            
                            # --- Bar chart ---
                            # Sort by gene_ratio for bar plot ordering
                            plot_df_bar = plot_df.sort_values('gene_ratio', ascending=False)
                            plt.figure(figsize=(10, max(6, top_n_actual * 0.5)))  # Adjust height based on N
                            sns.barplot(
                                x='gene_ratio',
                                y='Term',
                                data=plot_df_bar,
                                palette='viridis',
                                orient='h'
                            )
                            plt.title(f'Top {top_n_actual} Enriched GO {domain} Terms ({cell_type}) by Gene Ratio')
                            plt.xlabel('Gene Ratio')
                            plt.ylabel(f'GO {domain} Term')
                            plt.tight_layout()
                            barplot_filename = f'{domain_prefix}_barplot_{cell_type}.png'
                            plt.savefig(barplot_filename, dpi=300, bbox_inches='tight')
                            plt.close()
                            print(f"Saved bar plot to {barplot_filename}")
                            
                            # --- Dot plot ---
                            # Sort by gene_ratio for dot plot ordering
                            plot_df_dot = plot_df.sort_values('gene_ratio', ascending=False)
                            plt.figure(figsize=(12, max(6, top_n_actual * 0.5)))  # Adjust height
                            scatter = sns.scatterplot(
                                data=plot_df_dot,
                                y='Term',
                                x='gene_ratio',
                                size='intersection_size',
                                hue='-log10(p_value)',
                                palette='viridis_r',
                                sizes=(40, 400),
                                edgecolor='grey',
                                linewidth=0.5,
                                alpha=0.8
                            )
                            plt.title(f'Top {top_n_actual} Enriched GO {domain} Terms ({cell_type})')
                            plt.ylabel(f'GO {domain} Term')
                            plt.xlabel('Gene Ratio')
                            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='Legend')
                            handles, labels = scatter.get_legend_handles_labels()
                            legend_title_map = {'size': 'Intersection Size', 'hue': '-log10(P-value)'}
                            new_handles = []
                            new_labels = []
                            for i, label in enumerate(labels):
                                if label in legend_title_map:
                                    new_labels.append(legend_title_map[label])
                                    new_handles.append(handles[i])
                                elif handles[i] is not None:
                                    new_labels.append(label)
                                    new_handles.append(handles[i])
                            scatter.legend(new_handles, new_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                            plt.grid(True, axis='y', linestyle='--', alpha=0.6)
                            plt.tight_layout(rect=[0, 0, 0.85, 1])
                            dotplot_filename = f'{domain_prefix}_dotplot_{cell_type}.png'
                            plt.savefig(dotplot_filename, dpi=300, bbox_inches='tight')
                            plt.close()
                            print(f"Saved dot plot to {dotplot_filename}")
                        else:
                            print(f"No significant terms found for plotting after filtering (p < {p_value_threshold}) for {cell_type}, {domain}.")
                    else:
                        print(f"No significant terms found passing p-value threshold {p_value_threshold} for {cell_type}, {domain}.")
    
    return results


def kegg_enrichment(cell_type, significant_genes, gene_to_logfc, p_value_threshold=0.05, top_n_terms=10,
                    save_raw=True, save_summary=True, save_plots=True, output_prefix="kegg"):
    """
    Performs KEGG pathway enrichment analysis on differentially expressed genes for a specific cell type.
    
    Parameters:
    -----------
    cell_type : str
        Cell type to analyze.
    significant_genes : list
        A list of significant gene symbols (strings).
    gene_to_logfc : dict
        A dictionary mapping gene symbols (str) to their log2 fold change values (float).
    p_value_threshold : float, default=0.05
        P-value threshold for significant enrichment terms.
    top_n_terms : int, default=10
        Number of top terms to display in plots.
    save_raw : bool, default=True
        Whether to save raw enrichment results.
    save_summary : bool, default=True
        Whether to save summary enrichment results.
    save_plots : bool, default=True
        Whether to save plots.
    output_prefix : str, default="kegg"
        Prefix to use for output filenames.
        
    Returns:
    --------
    dict
        Dictionary containing DataFrames of raw and processed enrichment results.
    """   
    results = {
        'raw_results': None,
        'summary_results': None
    }
    
    if not significant_genes:
        return results
    
    # Initialize gProfiler
    gp_kegg = GProfiler(return_dataframe=True)
    
    # Run KEGG enrichment analysis
    kegg_enrichment_results = gp_kegg.profile(
        organism='hsapiens',
        query=significant_genes,
        sources=['KEGG'],
        user_threshold=0.05,
        significance_threshold_method='fdr',
        all_results=True,
        no_evidences=False,
        ordered=False
    )
    
    results['raw_results'] = kegg_enrichment_results
    
    if not kegg_enrichment_results.empty:
        # Save raw KEGG results
        if save_raw:
            # raw_output_filename = f"{output_prefix}_results_raw_{cell_type}.csv"
            raw_output_filename = os.path.join(
            output_prefix,
            f"results_raw_{cell_type}.csv"
            )
            kegg_enrichment_results.to_csv(raw_output_filename, index=False)
        
        # Post-processing
        kegg_enrichment_dict = {}
        for index, row in kegg_enrichment_results.iterrows():
            term_id = row['native']
            if not all(col in row for col in ['name', 'p_value', 'source', 'intersection_size', 'intersections']):
                continue
            
            # Process intersections
            intersecting_genes = []
            if isinstance(row['intersections'], list):
                intersecting_genes = row['intersections']
            elif isinstance(row['intersections'], str):
                intersecting_genes = [gene.strip() for gene in row['intersections'].split(',') if gene.strip()]
            else:
                continue
            
            # Calculate average log2FC
            avg_log2fc = 0
            gene_count_in_term = 0
            if intersecting_genes:
                sum_log2fc = 0
                for gene in intersecting_genes:
                    if gene in gene_to_logfc:
                        sum_log2fc += gene_to_logfc[gene]
                        gene_count_in_term += 1
                if gene_count_in_term > 0:
                    avg_log2fc = sum_log2fc / gene_count_in_term
            
            term_size = row['term_size'] if 'term_size' in row else 0
            intersection_size = row['intersection_size'] if 'intersection_size' in row else len(intersecting_genes)
            gene_ratio = intersection_size / term_size if term_size > 0 else 0
            
            kegg_enrichment_dict[term_id] = {
                'name': row['name'],
                'p_value': row['p_value'],
                'source': row['source'],
                'intersection_size': intersection_size,
                'term_size': term_size,
                'gene_ratio': gene_ratio,
                'calculated_intersection_count': gene_count_in_term,
                'avg_log2fc': avg_log2fc
            }
        
        if kegg_enrichment_dict:
            # Create output dataframe
            kegg_output_df = pd.DataFrame([
                {
                    'Term_ID': term_id,
                    'Term': info['name'],
                    'p_value': info['p_value'],
                    'intersection_size': info.get('intersection_size', 0),
                    'term_size': info.get('term_size', 0),
                    'gene_ratio': info.get('gene_ratio', 0),
                    'calculated_intersection_count': info.get('calculated_intersection_count', 0),
                    'avg_log2fc': info.get('avg_log2fc', 0)
                }
                for term_id, info in kegg_enrichment_dict.items()
            ])
            
            # Sort by p-value
            kegg_output_df = kegg_output_df.sort_values('p_value')
            results['summary_results'] = kegg_output_df
            
            # Save summarized results
            if save_summary:
                # summary_output_filename = f'{output_prefix}_results_summary_{cell_type}.csv'
                summary_output_filename = os.path.join(
                    output_prefix,
                    f"results_summary_{cell_type}.csv"
                )
                kegg_output_df.to_csv(summary_output_filename, index=False)
            
            # Visualization
            if save_plots:
                # Filter for significant terms
                significant_df = kegg_output_df[kegg_output_df['p_value'] < p_value_threshold].copy()
                
                if not significant_df.empty:
                    # Handle potential zero p-values
                    if (significant_df['p_value'] == 0).any():
                        min_non_zero_p = significant_df[significant_df['p_value'] > 0]['p_value'].min()
                        replacement_p = min_non_zero_p / 1000 if pd.notna(min_non_zero_p) and min_non_zero_p > 0 else 1e-300
                        significant_df['p_value'] = significant_df['p_value'].replace(0, replacement_p)
                    
                    # Calculate -log10(p-value)
                    significant_df['-log10(p_value)'] = -np.log10(significant_df['p_value'])
                    
                    # Select top N terms
                    significant_df = significant_df.sort_values('p_value', ascending=True)
                    top_n_actual = min(top_n_terms, len(significant_df))
                    
                    if top_n_actual > 0:
                        plot_df = significant_df.head(top_n_actual)
                        
                        # --- Bar chart ---
                        # Sort by gene_ratio for bar plot ordering
                        plot_df_bar = plot_df.sort_values('gene_ratio', ascending=False)
                        plt.figure(figsize=(10, max(6, top_n_actual * 0.5)))  # Adjust height based on N
                        sns.barplot(
                            x='gene_ratio',
                            y='Term',
                            data=plot_df_bar,
                            palette='viridis',
                            orient='h'
                        )
                        plt.title(f'Top {top_n_actual} Enriched KEGG Pathways ({cell_type}) by Gene Ratio')
                        plt.xlabel('Gene Ratio')
                        plt.ylabel('KEGG Pathway')
                        plt.tight_layout()
                        barplot_filename = f'{output_prefix}_barplot_{cell_type}.png'
                        plt.savefig(barplot_filename, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Saved bar plot to {barplot_filename}")
                        
                        # --- Dot plot ---
                        # Sort by gene_ratio for dot plot ordering
                        plot_df_dot = plot_df.sort_values('gene_ratio', ascending=False)
                        plt.figure(figsize=(12, max(6, top_n_actual * 0.5)))  # Adjust height
                        scatter = sns.scatterplot(
                            data=plot_df_dot,
                            y='Term',
                            x='gene_ratio',
                            size='intersection_size',
                            hue='-log10(p_value)',
                            palette='viridis_r',
                            sizes=(40, 400),
                            edgecolor='grey',
                            linewidth=0.5,
                            alpha=0.8
                        )
                        plt.title(f'Top {top_n_actual} Enriched KEGG Pathways ({cell_type})')
                        plt.ylabel('KEGG Pathway')
                        plt.xlabel('Gene Ratio')
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='Legend')
                        handles, labels = scatter.get_legend_handles_labels()
                        legend_title_map = {'size': 'Intersection Size', 'hue': '-log10(P-value)'}
                        new_handles = []
                        new_labels = []
                        for i, label in enumerate(labels):
                            if label in legend_title_map:
                                new_labels.append(legend_title_map[label])
                                new_handles.append(handles[i])
                            elif handles[i] is not None:
                                new_labels.append(label)
                                new_handles.append(handles[i])
                        scatter.legend(new_handles, new_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
                        plt.tight_layout(rect=[0, 0, 0.85, 1])
                        dotplot_filename = f'{output_prefix}_dotplot_{cell_type}.png'
                        plt.savefig(dotplot_filename, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Saved dot plot to {dotplot_filename}")
                    else:
                        print(f"No significant terms found for plotting after filtering (p < {p_value_threshold}) for {cell_type}.")
                else:
                    print(f"No significant terms found passing p-value threshold {p_value_threshold} for {cell_type}.")
    
    return results

def gsea_enrichment_analysis(
    cell_type,
    significant_genes,
    gene_to_logfc,
    gene_set_library="MSigDB_Hallmark_2020",
    organism='Human',
    p_value_threshold=0.05,
    top_n_terms=10,
    save_raw=True,
    save_summary=True,
    save_plots=True,
    output_prefix="enrichment",
    gsea_library_folder="gsea_library"
):
    """
    Performs GSEA enrichment analysis on significant genes for a cell type.

    - Auto-creates `output_prefix` folder.
    - If any .gmt found under `gsea_library_folder`, uses the first one.
    - Parses `Overlap` as "X/Y" into intersection_size, gene_set_size, gene_ratio.
    """
    os.makedirs(output_prefix, exist_ok=True)

    # 1) Auto-detect custom GMT
    custom_lib = None
    if os.path.isdir(gsea_library_folder):
        import glob
        gmt_files = glob.glob(os.path.join(gsea_library_folder, "*.gmt"))
        if gmt_files:
            custom_lib = gmt_files[0]
            gene_set_library = os.path.splitext(os.path.basename(custom_lib))[0]

    results = {"raw_results": None, "summary_results": None}

    if not significant_genes:
        return results

    # 2) Run enrichr
    enr = gp.enrichr(
        gene_list=significant_genes,
        gene_sets=custom_lib or gene_set_library,
        organism=organism,
        outdir=None,
        cutoff=1.0
    )
    if enr is None or enr.results.empty:
        return results

    df_raw = enr.results.copy()
    results['raw_results'] = df_raw

    # 3) Save raw
    if save_raw:
        path = os.path.join(output_prefix, f"results_raw_{cell_type}.csv")
        df_raw.to_csv(path, index=False)

    # 4) Post-process into list of dicts
    processed = []
    for _, row in df_raw.iterrows():
        # parse Overlap "X/Y"
        overlap = str(row.get('Overlap', ""))
        m = re.match(r'(\d+)/(\d+)', overlap)
        intersection_size = int(m.group(1)) if m else 0
        gene_set_size    = int(m.group(2)) if m else 0
        gene_ratio       = intersection_size / gene_set_size if gene_set_size else 0

        # average log₂FC
        genes = (row.get('Genes') or "").split(';')
        sum_fc = 0; cnt = 0
        for g in genes:
            if g in gene_to_logfc:
                sum_fc += gene_to_logfc[g]; cnt += 1
        avg_log2fc = sum_fc/cnt if cnt else 0

        processed.append({
            'Term': row['Term'],
            'p_value': row['P-value'],
            'adj_p_value': row['Adjusted P-value'],
            'intersection_size': intersection_size,
            'gene_set_size': gene_set_size,
            'gene_ratio': gene_ratio,
            'avg_log2fc': avg_log2fc,
            'intersecting_genes': ';'.join([g for g in genes if g in gene_to_logfc])
        })

    df_sum = pd.DataFrame(processed).sort_values('p_value')
    results['summary_results'] = df_sum

    # 5) Save summary
    if save_summary:
        path = os.path.join(output_prefix, f"results_summary_{cell_type}.csv")
        df_sum.to_csv(path, index=False)

    # 6) Plotting
    if save_plots and not df_sum.empty:
        # replace zero p-values
        if (df_sum['p_value']==0).any():
            nz = df_sum.loc[df_sum['p_value']>0,'p_value'].min() or 1e-300
            df_sum['p_value'] = df_sum['p_value'].replace(0, nz/1000)

        df_sum['-log10(p_value)'] = -np.log10(df_sum['p_value'])
        top = df_sum.head(top_n_terms)

        # Barplot
        bar_df = top.sort_values('-log10(p_value)')
        plt.figure(figsize=(10, max(6, len(bar_df)*0.5)))
        sns.barplot(
            x='-log10(p_value)',
            y='Term',
            data=bar_df,
            orient='h'
        )
        plt.title(f"Top {top_n_terms} {gene_set_library} Terms ({cell_type})")
        plt.tight_layout()
        barfile = os.path.join(output_prefix, f"barplot_{cell_type}.png")
        plt.savefig(barfile, dpi=300, bbox_inches='tight'); plt.close()

        # Dotplot
        xcol = 'gene_ratio'
        dot_df = top.sort_values(xcol, ascending=False)
        plt.figure(figsize=(12, max(6, len(dot_df)*0.5)))
        sns.scatterplot(
            data=dot_df,
            y='Term',
            x=xcol,
            size='intersection_size',
            hue='-log10(p_value)',
            sizes=(40,400),
            edgecolor='grey',
            alpha=0.8
        )
        plt.title(f"Top {top_n_terms} {gene_set_library} Terms ({cell_type})")
        plt.xlabel("Gene Ratio")
        plt.tight_layout()
        dotfile = os.path.join(output_prefix, f"dotplot_{cell_type}.png")
        plt.savefig(dotfile, dpi=300, bbox_inches='tight'); plt.close()

    return results

def perform_enrichment_analyses(
    adata,
    cell_type: str,
    analyses: List[str] = None,
    logfc_threshold: float = 1.0,
    pval_threshold: float = 0.05,
    top_n_terms: int = 10
) -> Dict[str, Any]:
    """
    Runs reactome, go, kegg, gsea on DE genes, then returns:
      - per_analysis: { analysis_name → { raw_results: [...], summary_results: [...] } }
      - top_terms:     [ {analysis, Term_ID, Term, p_value, intersection_size, avg_log2fc, intersecting_genes} … ]
    """
    if analyses is None:
        analyses = ["reactome", "go", "kegg", "gsea"]

    sig_genes, gene_to_logfc = get_significant_gene(
        adata, cell_type,
        logfc_threshold=logfc_threshold,
        pval_threshold=pval_threshold
    )

    per_analysis: Dict[str, Dict[str, List[Dict[str,Any]]]] = {}

    for name in analyses:
        key = name.lower()
        if key == "reactome":
            out = reactome_enrichment(
                cell_type, sig_genes, gene_to_logfc,
                p_value_threshold=pval_threshold,
                top_n_terms=top_n_terms,
                output_prefix="schatbot/enrichment/reactome"
            )
        elif key == "go":
            out = go_enrichment(
                cell_type, sig_genes, gene_to_logfc,
                p_value_threshold=pval_threshold,
                top_n_terms=top_n_terms,
                output_prefix="schatbot/enrichment/go"
            )
        elif key == "kegg":
            out = kegg_enrichment(
                cell_type, sig_genes, gene_to_logfc,
                p_value_threshold=pval_threshold,
                top_n_terms=top_n_terms,
                output_prefix="schatbot/enrichment/kegg"
            )
        else:  # gsea
            out = gsea_enrichment_analysis(
                cell_type, sig_genes, gene_to_logfc,
                top_n_terms=top_n_terms,
                output_prefix="schatbot/enrichment/gsea"
            )

        # Convert any DataFrame → list of dicts
        raw_df = out.get("raw_results")
        sum_df = out.get("summary_results")
        raw_list = raw_df.to_dict(orient="records")   if hasattr(raw_df, "to_dict")   else []
        sum_list = sum_df.to_dict(orient="records")   if hasattr(sum_df, "to_dict")   else []

        per_analysis[key] = {
            "raw_results":   raw_list,
            "summary_results": sum_list
        }

    # now flatten the top_n_terms from each summary
    top_terms: List[Dict[str,Any]] = []
    for analysis, tables in per_analysis.items():
        for entry in tables["summary_results"][:top_n_terms]:
            rec = {
                "analysis": analysis,
                "name": entry.get("Term") or entry.get("name"),
                "description": entry.get("description") or entry.get("Term") or entry.get("name"),
                "intersections": entry.get("intersecting_genes") or entry.get("intersections")
            }
            top_terms.append(rec)

    return {
        "Top terms": top_terms,
        "Top terms stattistics": rec
    }