import pickle 
import json
import os
import time
import re
import openai
import json
import pandas as pd
import scanpy as sc
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from scvi.model import SCVI
import plotly.express as px
import plotly.graph_objects as go
import ast
import matplotlib
import warnings
import numpy as np
import shutil
import gseapy as gp
import requests
from langsmith import utils
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict, Dict, List, Any
import ast
from collections import deque
import anndata
from neo4j import GraphDatabase
import scipy
import warnings
from pandas.errors import PerformanceWarning

from gprofiler import GProfiler
import seaborn as sns
import gseapy as gp


warnings.filterwarnings("ignore", category=PerformanceWarning)


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


import os

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
            print(f"Saved raw Reactome results to {raw_output_filename}")

        
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
            
            reactome_enrichment_dict[term_id] = {
                'name': row['name'],
                'p_value': row['p_value'],
                'source': row['source'],
                'intersection_size': row['intersection_size'],
                'intersecting_genes': intersecting_genes,
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
                    'avg_log2fc': info.get('avg_log2fc', 0),
                    'intersecting_genes': ','.join(info.get('intersecting_genes', [])),
                    'calculated_intersection_count': info.get('calculated_intersection_count', 0)
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
                print(f"Saved summary Reactome results to {summary_output_filename}")
            
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
                        print(f"Replaced 0 p-values with {replacement_p} for plotting.")
                    
                    # Calculate -log10(p-value)
                    significant_df['-log10(p_value)'] = -np.log10(significant_df['p_value'])
                    
                    # Select top N terms
                    significant_df = significant_df.sort_values('p_value', ascending=True)
                    top_n_actual = min(top_n_terms, len(significant_df))
                    
                    if top_n_actual > 0:
                        plot_df = significant_df.head(top_n_actual)
                        
                        # --- Bar chart ---
                        # Sort by -log10(p_value) for bar plot ordering
                        plot_df_bar = plot_df.sort_values('-log10(p_value)', ascending=True)
                        plt.figure(figsize=(10, max(6, top_n_actual * 0.5)))  # Adjust height based on N
                        sns.barplot(
                            x='-log10(p_value)',
                            y='Term',
                            data=plot_df_bar,
                            palette='viridis',  # Using consistent palette
                            orient='h'
                        )
                        plt.title(f'Top {top_n_actual} Enriched Reactome Pathways ({cell_type}) by P-value')
                        plt.xlabel('-log10(P-value)')
                        plt.ylabel('Reactome Pathway')
                        plt.tight_layout()
                        # barplot_filename = f'{output_prefix}_barplot_{cell_type}.png'
                        barplot_filename = os.path.join(
                            output_prefix,
                            f"barplot_{cell_type}.png"
                        )
                        plt.savefig(barplot_filename, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Saved Reactome bar plot to {barplot_filename}")
                        
                        # --- Dot plot ---
                        # Sort by average log2FC for dot plot ordering
                        plot_df_dot = plot_df.sort_values('avg_log2fc', ascending=False)
                        plt.figure(figsize=(12, max(6, top_n_actual * 0.5)))  # Adjust height
                        scatter = sns.scatterplot(
                            data=plot_df_dot,
                            y='Term',
                            x='avg_log2fc',
                            size='intersection_size',
                            hue='-log10(p_value)',
                            palette='plasma_r',
                            sizes=(40, 400),
                            edgecolor='grey',
                            linewidth=0.5,
                            alpha=0.8
                        )
                        plt.title(f'Top {top_n_actual} Enriched Reactome Pathways ({cell_type})')
                        plt.ylabel('Reactome Pathway')
                        plt.xlabel('Average log2 Fold Change')
                        plt.axvline(0, color='grey', linestyle='--', lw=1)  # Line at logFC=0
                        
                        # Improve legend positioning
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='Legend')
                        # Customize legend titles
                        handles, labels = scatter.get_legend_handles_labels()
                        legend_title_map = {'size': 'Intersection Size', 'hue': '-log10(P-value)'}
                        new_handles = []
                        new_labels = []
                        for i, label in enumerate(labels):
                            if label in legend_title_map:  # Identify title rows generated by seaborn
                                new_labels.append(legend_title_map[label])  # Replace with custom title
                                new_handles.append(handles[i])
                            elif handles[i] is not None:  # Keep actual data points
                                new_labels.append(label)
                                new_handles.append(handles[i])

                        scatter.legend(new_handles, new_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

                        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
                        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend
                        # dotplot_filename = f'{output_prefix}_dotplot_{cell_type}.png'
                        dotplot_filename = os.path.join(
                            output_prefix,
                            f"dotplot_{cell_type}.png"
                        )
                        plt.savefig(dotplot_filename, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Saved Reactome dot plot to {dotplot_filename}")
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
                print(f"Saved raw GO {domain} results to {raw_output_filename}")
            
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
                
                go_enrichment_dict[term_id] = {
                    'name': row['name'],
                    'p_value': row['p_value'],
                    'source': row['source'],
                    'intersection_size': row['intersection_size'],
                    'intersecting_genes': intersecting_genes,
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
                        'avg_log2fc': info.get('avg_log2fc', 0),
                        'intersecting_genes': ','.join(info.get('intersecting_genes', [])),
                        'calculated_intersection_count': info.get('calculated_intersection_count', 0)
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
                    print(f"Saved summary GO {domain} results to {summary_output_filename}")
                
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
                            print(f"Replaced 0 p-values with {replacement_p} for plotting.")
                        
                        # Calculate -log10(p-value)
                        significant_df['-log10(p_value)'] = -np.log10(significant_df['p_value'])
                        
                        # Select top N terms
                        significant_df = significant_df.sort_values('p_value', ascending=True)
                        top_n_actual = min(top_n_terms, len(significant_df))
                        
                        if top_n_actual > 0:
                            plot_df = significant_df.head(top_n_actual)
                            
                            # --- Bar chart ---
                            # Sort by -log10(p_value) for bar plot ordering
                            plot_df_bar = plot_df.sort_values('-log10(p_value)', ascending=True)
                            plt.figure(figsize=(10, max(6, top_n_actual * 0.5)))  # Adjust height based on N
                            sns.barplot(
                                x='-log10(p_value)',
                                y='Term',
                                data=plot_df_bar,
                                palette='viridis',  # Using consistent palette
                                orient='h'
                            )
                            plt.title(f'Top {top_n_actual} Enriched GO {domain} Terms ({cell_type}) by P-value')
                            plt.xlabel('-log10(P-value)')
                            plt.ylabel(f'GO {domain} Term')
                            plt.tight_layout()
                            barplot_filename = f'{domain_prefix}_barplot_{cell_type}.png'
                            plt.savefig(barplot_filename, dpi=300, bbox_inches='tight')
                            plt.close()
                            print(f"Saved GO {domain} bar plot to {barplot_filename}")
                            
                            # --- Dot plot ---
                            # Sort by average log2FC for dot plot ordering
                            plot_df_dot = plot_df.sort_values('avg_log2fc', ascending=False)
                            plt.figure(figsize=(12, max(6, top_n_actual * 0.5)))  # Adjust height
                            scatter = sns.scatterplot(
                                data=plot_df_dot,
                                y='Term',
                                x='avg_log2fc',
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
                            plt.xlabel('Average log2 Fold Change')
                            plt.axvline(0, color='grey', linestyle='--', lw=1)  # Line at logFC=0
                            
                            # Improve legend positioning
                            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='Legend')
                            # Customize legend titles
                            handles, labels = scatter.get_legend_handles_labels()
                            legend_title_map = {'size': 'Intersection Size', 'hue': '-log10(P-value)'}
                            new_handles = []
                            new_labels = []
                            for i, label in enumerate(labels):
                                if label in legend_title_map:  # Identify title rows generated by seaborn
                                    new_labels.append(legend_title_map[label])  # Replace with custom title
                                    new_handles.append(handles[i])
                                elif handles[i] is not None:  # Keep actual data points
                                    new_labels.append(label)
                                    new_handles.append(handles[i])

                            scatter.legend(new_handles, new_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

                            plt.grid(True, axis='y', linestyle='--', alpha=0.6)
                            plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend
                            dotplot_filename = f'{domain_prefix}_dotplot_{cell_type}.png'
                            plt.savefig(dotplot_filename, dpi=300, bbox_inches='tight')
                            plt.close()
                            print(f"Saved GO {domain} dot plot to {dotplot_filename}")
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
            print(f"Saved raw KEGG results to {raw_output_filename}")
        
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
            
            kegg_enrichment_dict[term_id] = {
                'name': row['name'],
                'p_value': row['p_value'],
                'source': row['source'],
                'intersection_size': row['intersection_size'],
                'intersecting_genes': intersecting_genes,
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
                    'avg_log2fc': info.get('avg_log2fc', 0),
                    'intersecting_genes': ','.join(info.get('intersecting_genes', [])),
                    'calculated_intersection_count': info.get('calculated_intersection_count', 0)
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
                print(f"Saved summary KEGG results to {summary_output_filename}")
            
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
                        print(f"Replaced 0 p-values with {replacement_p} for plotting.")
                    
                    # Calculate -log10(p-value)
                    significant_df['-log10(p_value)'] = -np.log10(significant_df['p_value'])
                    
                    # Select top N terms
                    significant_df = significant_df.sort_values('p_value', ascending=True)
                    top_n_actual = min(top_n_terms, len(significant_df))
                    
                    if top_n_actual > 0:
                        plot_df = significant_df.head(top_n_actual)
                        
                        # --- Bar chart ---
                        # Sort by -log10(p_value) for bar plot ordering
                        plot_df_bar = plot_df.sort_values('-log10(p_value)', ascending=True)
                        plt.figure(figsize=(10, max(6, top_n_actual * 0.5)))  # Adjust height based on N
                        sns.barplot(
                            x='-log10(p_value)',
                            y='Term',
                            data=plot_df_bar,
                            palette='viridis',  # Using consistent palette
                            orient='h'
                        )
                        plt.title(f'Top {top_n_actual} Enriched KEGG Pathways ({cell_type}) by P-value')
                        plt.xlabel('-log10(P-value)')
                        plt.ylabel('KEGG Pathway')
                        plt.tight_layout()
                        # barplot_filename = f'{output_prefix}_barplot_{cell_type}.png'
                        barplot_filename = os.path.join(
                            output_prefix,
                            f"barplot_{cell_type}.png"
                        )
                        plt.savefig(barplot_filename, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Saved KEGG bar plot to {barplot_filename}")
                        
                        # --- Dot plot ---
                        # Sort by average log2FC for dot plot ordering
                        plot_df_dot = plot_df.sort_values('avg_log2fc', ascending=False)
                        plt.figure(figsize=(12, max(6, top_n_actual * 0.5)))  # Adjust height
                        scatter = sns.scatterplot(
                            data=plot_df_dot,
                            y='Term',
                            x='avg_log2fc',
                            size='intersection_size',
                            hue='-log10(p_value)',
                            palette='plasma_r',
                            sizes=(40, 400),
                            edgecolor='grey',
                            linewidth=0.5,
                            alpha=0.8
                        )
                        plt.title(f'Top {top_n_actual} Enriched KEGG Pathways ({cell_type})')
                        plt.ylabel('KEGG Pathway')
                        plt.xlabel('Average log2 Fold Change')
                        plt.axvline(0, color='grey', linestyle='--', lw=1)  # Line at logFC=0
                        
                        # Improve legend positioning
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='Legend')
                        # Customize legend titles
                        handles, labels = scatter.get_legend_handles_labels()
                        legend_title_map = {'size': 'Intersection Size', 'hue': '-log10(P-value)'}
                        new_handles = []
                        new_labels = []
                        for i, label in enumerate(labels):
                            if label in legend_title_map:  # Identify title rows generated by seaborn
                                new_labels.append(legend_title_map[label])  # Replace with custom title
                                new_handles.append(handles[i])
                            elif handles[i] is not None:  # Keep actual data points
                                new_labels.append(label)
                                new_handles.append(handles[i])

                        scatter.legend(new_handles, new_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

                        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
                        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend
                        # dotplot_filename = f'{output_prefix}_dotplot_{cell_type}.png'
                        dotplot_filename = os.path.join(
                            output_prefix,
                            f"dotplot_{cell_type}.png"
                        )
                        plt.savefig(dotplot_filename, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Saved KEGG dot plot to {dotplot_filename}")
                    else:
                        print(f"No significant terms found for plotting after filtering (p < {p_value_threshold}) for {cell_type}.")
                else:
                    print(f"No significant terms found passing p-value threshold {p_value_threshold} for {cell_type}.")
    
    return results

# def gsea_enrichment_analysis(cell_type, significant_genes, gene_to_logfc,
#                             gene_set_library="MSigDB_Hallmark_2020", organism='Human',
#                             p_value_threshold=0.05, top_n_terms=10,
#                             save_raw=True, save_summary=True, save_plots=True,
#                             output_prefix="enrichment"):
#     """
#     Performs gene set enrichment analysis using gseapy on significant genes
#     for a specific cell type.

#     Parameters:
#     -----------
#     cell_type : str
#         Identifier for the cell type being analyzed (used for filenames).
#     significant_genes : list
#         A list of significant gene symbols (strings).
#     gene_to_logfc : dict
#         A dictionary mapping gene symbols (str) to their log2 fold change values (float).
#     gene_set_library : str, default="MSigDB_Hallmark_2020"
#         The gene set library to use for enrichment analysis.
#     organism : str, default='Human'
#         Organism to use for enrichment analysis.
#     p_value_threshold : float, default=0.05
#         P-value threshold for significant enrichment terms (used for plotting).
#     top_n_terms : int, default=10
#         Number of top terms to display in plots.
#     save_raw : bool, default=True
#         Whether to save raw gseapy enrichment results.
#     save_summary : bool, default=True
#         Whether to save summary enrichment results (with calculated avg log2FC).
#     save_plots : bool, default=True
#         Whether to save enrichment plots (barplot and dotplot).
#     output_prefix : str, default="enrichment"
#         Prefix to use for output filenames.

#     Returns:
#     --------
#     dict
#         Dictionary containing DataFrames of raw and processed enrichment results:
#         {'raw_results': DataFrame, 'summary_results': DataFrame}
#     """
#     results = {
#         'raw_results': None,
#         'summary_results': None
#     }

#     if not significant_genes:
#         print(f"No significant genes provided for {cell_type}. Skipping enrichment.")
#         return results

#     try:
#         # Run enrichr analysis using gseapy
#         enr = gp.enrichr(gene_list=significant_genes,
#                         gene_sets=gene_set_library,
#                         organism=organism,  # Can be specified based on dataset
#                         outdir=None,  # Prevent gseapy from saving files automatically
#                         cutoff=1.0  # Get all results and filter later based on p_value_threshold for plotting
#                         )

#         if enr is None or enr.results.empty:
#             print(f"No enrichment results found for {cell_type} using {gene_set_library}.")
#             return results

#         # The results DataFrame is typically stored in enr.results
#         enrichment_results_df = enr.results.copy()  # Make a copy to avoid modifying original
#         results['raw_results'] = enrichment_results_df

#         # Save raw gseapy results
#         if save_raw:
#             # raw_output_filename = f"{output_prefix}_results_raw_{cell_type}.csv"
#             raw_output_filename = os.path.join(
#             output_prefix,
#             f"results_raw_{cell_type}.csv"
#             )
#             enrichment_results_df.to_csv(raw_output_filename, index=False)
#             print(f"Saved raw {gene_set_library} results to {raw_output_filename}")

#         # Post-processing to add average log2FC and format
#         processed_results_list = []
#         for index, row in enrichment_results_df.iterrows():
#             term_name = row['Term']
#             p_value = row['P-value']
#             adj_p_value = row['Adjusted P-value']  # Keep adjusted p-value as well
#             # Genes are semicolon-separated in gseapy enrichr results
#             intersecting_genes_str = row['Genes']
#             intersecting_genes = intersecting_genes_str.split(';') if isinstance(intersecting_genes_str, str) else []
#             # Extract intersection size from 'Overlap' column (e.g., "15/50")
#             overlap_str = row['Overlap']
#             intersection_size = 0
#             if isinstance(overlap_str, str):
#                 match = re.match(r'(\d+)/\d+', overlap_str)
#                 if match:
#                     intersection_size = int(match.group(1))

#             # Calculate average log2FC for intersecting genes found in gene_to_logfc
#             sum_log2fc = 0
#             gene_count_in_term = 0
#             valid_intersecting_genes = []
#             if intersecting_genes:
#                 for gene in intersecting_genes:
#                     if gene in gene_to_logfc:
#                         sum_log2fc += gene_to_logfc[gene]
#                         gene_count_in_term += 1
#                         valid_intersecting_genes.append(gene)  # Store genes actually used

#             avg_log2fc = sum_log2fc / gene_count_in_term if gene_count_in_term > 0 else 0

#             processed_results_list.append({
#                 'Term': term_name,
#                 'p_value': p_value,
#                 'adj_p_value': adj_p_value,
#                 'intersection_size': intersection_size,  # Actual count based on overlap string
#                 'calculated_intersection_count': gene_count_in_term,  # Count of genes with logFC found
#                 'avg_log2fc': avg_log2fc,
#                 'intersecting_genes': ';'.join(valid_intersecting_genes)  # Save genes with logFC
#             })

#         if processed_results_list:
#             # Create summary output dataframe
#             summary_output_df = pd.DataFrame(processed_results_list)

#             # Sort by p-value
#             summary_output_df = summary_output_df.sort_values('p_value', ascending=True)
#             results['summary_results'] = summary_output_df

#             # Save summarized results
#             if save_summary:
#                 # summary_output_filename = f'{output_prefix}_results_summary_{cell_type}.csv'
#                 summary_output_filename = os.path.join(
#                     output_prefix,
#                     f"results_summary_{cell_type}.csv"
#                 )
#                 summary_output_df.to_csv(summary_output_filename, index=False)
#                 print(f"Saved summary {gene_set_library} results to {summary_output_filename}")

#             # Visualization
#             if save_plots:
#                 # Filter for significant terms based on the specified p_value_threshold
#                 significant_df = summary_output_df[summary_output_df['p_value'] < p_value_threshold].copy()

#                 if not significant_df.empty:
#                     # Handle potential zero p-values for plotting
#                     if (significant_df['p_value'] == 0).any():
#                         min_non_zero_p = significant_df[significant_df['p_value'] > 0]['p_value'].min()
#                         # Use a small epsilon or a fraction of the minimum non-zero p-value
#                         replacement_p = min_non_zero_p / 1000 if pd.notna(min_non_zero_p) and min_non_zero_p > 0 else 1e-300  # Fallback if all are zero
#                         significant_df['p_value'] = significant_df['p_value'].replace(0, replacement_p)
#                         print(f"Replaced 0 p-values with {replacement_p} for plotting.")

#                     # Calculate -log10(p-value)
#                     significant_df['-log10(p_value)'] = -np.log10(significant_df['p_value'])

#                     # Select top N terms (already sorted by p-value)
#                     top_n_actual = min(top_n_terms, len(significant_df))

#                     if top_n_actual > 0:
#                         plot_df = significant_df.head(top_n_actual)

#                         # --- Bar chart ---
#                         # Sort by -log10(p_value) for bar plot ordering
#                         plot_df_bar = plot_df.sort_values('-log10(p_value)', ascending=True)
#                         plt.figure(figsize=(10, max(6, top_n_actual * 0.5)))  # Adjust height based on N
#                         sns.barplot(
#                             x='-log10(p_value)',
#                             y='Term',
#                             data=plot_df_bar,
#                             palette='viridis',  # Changed palette for variety
#                             orient='h'
#                         )
#                         plt.title(f'Top {top_n_actual} Enriched Terms from {gene_set_library} ({cell_type}) by P-value')
#                         plt.xlabel('-log10(P-value)')
#                         plt.ylabel('Term')
#                         plt.tight_layout()
#                         # barplot_filename = f'{output_prefix}_barplot_{cell_type}.png'
#                         barplot_filename = os.path.join(
#                             output_prefix,
#                             f"barplot_{cell_type}.png"
#                         )
#                         plt.savefig(barplot_filename, dpi=300, bbox_inches='tight')
#                         plt.close()
#                         print(f"Saved {gene_set_library} bar plot to {barplot_filename}")

#                         # --- Dot plot ---
#                         # Sort by average log2FC for dot plot ordering, or keep p-value sort
#                         plot_df_dot = plot_df.sort_values('avg_log2fc', ascending=False)
#                         plt.figure(figsize=(12, max(6, top_n_actual * 0.5)))  # Adjust height
#                         scatter = sns.scatterplot(
#                             data=plot_df_dot,
#                             y='Term',
#                             x='avg_log2fc',
#                             size='intersection_size',  # Use intersection size from gseapy
#                             hue='-log10(p_value)',  # Color by significance
#                             palette='plasma_r',  # Keep palette consistent with original example
#                             sizes=(40, 400),  # Control dot size range
#                             edgecolor='grey',
#                             linewidth=0.5,
#                             alpha=0.8
#                         )
#                         plt.title(f'Top {top_n_actual} Enriched Terms from {gene_set_library} ({cell_type})')
#                         plt.ylabel('Term')
#                         plt.xlabel('Average log2 Fold Change')
#                         plt.axvline(0, color='grey', linestyle='--', lw=1)  # Line at logFC=0
#                         # Improve legend positioning
#                         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='Legend')
#                         # Customize legend titles if necessary (might need manual creation for clarity)
#                         handles, labels = scatter.get_legend_handles_labels()
#                         # Find legend sections and add titles (this can be complex, basic version shown)
#                         legend_title_map = {'size': 'Intersection Size', 'hue': '-log10(P-value)'}
#                         new_handles = []
#                         new_labels = []
#                         # Simple re-titling - may need refinement depending on seaborn version
#                         for i, label in enumerate(labels):
#                             if label in legend_title_map:  # Identify title rows generated by seaborn
#                                 new_labels.append(legend_title_map[label])  # Replace with custom title
#                                 new_handles.append(handles[i])
#                             elif handles[i] is not None:  # Keep actual data points
#                                 new_labels.append(label)
#                                 new_handles.append(handles[i])

#                         scatter.legend(new_handles, new_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

#                         plt.grid(True, axis='y', linestyle='--', alpha=0.6)
#                         plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend
#                         # dotplot_filename = f'{output_prefix}_dotplot_{cell_type}.png'
#                         dotplot_filename = os.path.join(
#                             output_prefix,
#                             f"dotplot_{cell_type}.png"
#                         )
#                         plt.savefig(dotplot_filename, dpi=300, bbox_inches='tight')
#                         plt.close()
#                         print(f"Saved {gene_set_library} dot plot to {dotplot_filename}")
#                     else:
#                         print(f"No significant terms found for plotting after filtering (p < {p_value_threshold}) for {cell_type}.")
#                 else:
#                     print(f"No significant terms found passing p-value threshold {p_value_threshold} for {cell_type}.")
#         else:
#             print(f"No enrichment results processed for {cell_type}.")

#     except Exception as e:
#         print(f"An error occurred during gseapy enrichment analysis for {cell_type}: {e}")

#     return results

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
    gsea_library_folder="gsea_library",
    plot_x_axis="gene_ratio"    # or "avg_log2fc"
):
    """
    Performs GSEA enrichment analysis on significant genes for a cell type.

    - Auto-creates `output_prefix` folder.
    - If any .gmt found under `gsea_library_folder`, uses the first one.
    - Parses `Overlap` as "X/Y" into intersection_size, gene_set_size, gene_ratio.
    - Dotplot x-axis can be gene_ratio or avg_log2fc.
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
        print(f"Saved raw GSEA results to {path}")

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
        print(f"Saved summary GSEA results to {path}")

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
        print(f"Saved barplot to {barfile}")

        # Dotplot
        xcol = 'gene_ratio' if plot_x_axis=='gene_ratio' else 'avg_log2fc'
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
        plt.xlabel("Gene Ratio" if xcol=='gene_ratio' else "Average log₂ FC")
        if xcol!='gene_ratio':
            plt.axvline(0, linestyle='--', color='grey')
        plt.tight_layout()
        dotfile = os.path.join(output_prefix, f"dotplot_{cell_type}.png")
        plt.savefig(dotfile, dpi=300, bbox_inches='tight'); plt.close()
        print(f"Saved dotplot to {dotfile}")

    return results


from typing import List, Dict, Any
import pandas as pd


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
            rec = entry.copy()
            rec["analysis"] = analysis
            top_terms.append(rec)

    return {
        "per_analysis": per_analysis,
        "top_terms": top_terms
    }

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

def display_processed_umap(cell_type):
    """Display annotated UMAP plot with robust column checking."""
    try:
        cell_type2 = cell_type.split()[0].capitalize() + " cell"
        cell_type = cell_type.split()[0].capitalize() + " cells"
        
        # Try both possible file names
        if os.path.exists(f'umaps/{cell_type}_umap_data.csv'):
            umap_data = pd.read_csv(f'umaps/{cell_type}_umap_data.csv')
        elif os.path.exists(f'umaps/{cell_type2}_umap_data.csv'):
            umap_data = pd.read_csv(f'umaps/{cell_type2}_umap_data.csv')
        else:
            print(f"Warning: Could not find UMAP data for {cell_type} or {cell_type2}")
            return None
        
        # Check for required columns
        required_cols = ["UMAP_1", "UMAP_2", "cell_type"]
        missing_cols = [col for col in required_cols if col not in umap_data.columns]
        if missing_cols:
            print(f"Warning: Missing required columns in UMAP data: {missing_cols}")
            return None
        
        # Create plot parameters based on available columns
        plot_params = {
            "x": "UMAP_1",
            "y": "UMAP_2",
            "color": "cell_type",
            "title": f'{cell_type} UMAP Plot',
            "labels": {"UMAP_1": "UMAP 1", "UMAP_2": "UMAP 2"}
        }
        
        # Add symbol parameter only if patient_name exists
        if "patient_name" in umap_data.columns:
            plot_params["symbol"] = "patient_name"
        
        # Create the plot
        fig = px.scatter(umap_data, **plot_params)
        fig.update_traces(marker=dict(size=5, opacity=0.8))
        fig.update_layout(
            width=1200,
            height=800,
            autosize=True,
            showlegend=True
        )
        fig.show()
        fig_json = fig.to_json()
        return fig_json
    
    except Exception as e:
        print(f"Error in display_processed_umap: {e}")
        return None

def display_umap(cell_type):
    """Display clustering UMAP plot with robust column checking."""
    try:
        display_flag = True
        cell_type = cell_type.split()[0].capitalize() + " cells"
        file_path = f'process_cell_data/{cell_type}_umap_data.csv'
        
        if not os.path.exists(file_path):
            print(f"Warning: Could not find UMAP data at {file_path}")
            return None
            
        umap_data = pd.read_csv(file_path)
        
        # Check for required columns
        required_cols = ["UMAP_1", "UMAP_2", "leiden"]
        missing_cols = [col for col in required_cols if col not in umap_data.columns]
        if missing_cols:
            print(f"Warning: Missing required columns in UMAP data: {missing_cols}")
            return None
        
        # Handle cell type for specific cell types
        if cell_type != "Overall cells" and "cell_type" in umap_data.columns:
            umap_data['original_cell_type'] = umap_data['cell_type']
            umap_data['cell_type'] = 'Unknown'
        
        # Create plot parameters based on available columns
        plot_params = {
            "x": "UMAP_1",
            "y": "UMAP_2",
            "color": "leiden",
            "title": f"{cell_type} UMAP Plot",
            "labels": {"UMAP_1": "UMAP 1", "UMAP_2": "UMAP 2"}
        }
        
        # Add symbol parameter only if patient_name exists
        if "patient_name" in umap_data.columns:
            plot_params["symbol"] = "patient_name"
        
        # Create the plot
        fig = px.scatter(umap_data, **plot_params)
        fig.update_traces(marker=dict(size=5, opacity=0.8))
        fig.update_layout(
            width=1200,
            height=800,
            autosize=True,
            showlegend=False
        )
        
        # Add custom legend
        custom_legend = go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='rgba(0,0,0,0)'),
            legendgroup="Unknown",
            showlegend=True,
            name="Unknown"
        )
        fig.add_trace(custom_legend)
        
        fig.show()
        fig_json = fig.to_json()
        return fig_json
        
    except Exception as e:
        print(f"Error in display_umap: {e}")
        return None

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
    # Initialize Neo4j connection
    uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "37754262"))
    
    # Load specification from JSON
    specification = None
    file_path = "media/specification_graph.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            specification = json.load(file)
    else:
        print("specification not found")
        return "-"
    
    # Extract parameters from specification
    database = specification['database']
    system = specification['system']
    organ = specification['organ']

    # Initialize result structure
    combined_data = {}
    
    try:
        with driver.session(database=database) as session:
            # Get cell types and markers from Neo4j
            query = """
            MATCH (s:System {name: $system})-[:HAS_ORGAN]->(o:Organ {name: $organ})-[:HAS_CELL]->(c:CellType)-[:HAS_MARKER]->(m:Marker)
            RETURN c.name as cell_name, m.markers as marker_list
            """
            result = session.run(query, system=system, organ=organ)
            
            # Process results into expected format
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
    uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "DBMSPassword"))
    
    # Load specification for database info
    specification = None
    file_path = "media/specification_graph.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            specification = json.load(file)
    
    database = specification['database']
    
    # Initialize result structure
    subtypes_data = {}
    
    try:
        with driver.session(database=database) as session:
            # Get subtypes and markers
            query = """
            MATCH (parent:CellType {name: $parent_cell})-[:DEVELOPS_TO]->(c:CellType)-[:HAS_MARKER]->(m:Marker)
            RETURN c.name as cell_name, m.markers as marker_list
            """
            result = session.run(query, parent_cell=cell_type)
            
            # Process results
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

def unified_cell_type_handler(cell_type):
    """
    Standardizes cell type names with proper handling for special cases.
    
    Parameters:
    -----------
    cell_type : str
        The cell type string to process
    
    Returns:
    --------
    str
        Standardized cell type name suitable for file paths and database matching
    """
    # Dictionary of known cell types with their standardized forms
    known_cell_types = {
        # One-word cell types (already plural, don't add "cells")
        "platelet": "Platelets",
        "platelets": "Platelets",
        "lymphocyte": "Lymphocytes",
        "lymphocytes": "Lymphocytes",
        
        # Three-word cell types (specific capitalization)
        "natural killer cell": "Natural killer cells",
        "natural killer cells": "Natural killer cells",
        "plasmacytoid dendritic cell": "Plasmacytoid dendritic cells",
        "plasmacytoid dendritic cells": "Plasmacytoid dendritic cells"
    }
    
    # Clean and normalize the input
    clean_type = cell_type.lower().strip()
    if clean_type.endswith(' cells'):
        clean_type = clean_type[:-6].strip()
    elif clean_type.endswith(' cell'):
        clean_type = clean_type[:-5].strip()
    
    # Check if it's a known cell type
    if clean_type in known_cell_types:
        return known_cell_types[clean_type]
    
    # Handle based on word count
    words = clean_type.split()
    if len(words) == 1:
        # Check if it's already plural
        if words[0].endswith('s') and not words[0].endswith('ss'):
            # Already plural, just return as is (e.g., "Platelets")
            return words[0].capitalize()
        else:
            # Add "cells" to singular forms (e.g., "Monocyte cells")
            return f"{words[0].capitalize()} cells"
    
    elif len(words) == 2:
        # Special case for cell types like "T cell", "B cell"
        special_first_words = ['t', 'b', 'nk', 'cd4', 'cd8']
        
        if words[0].lower() in special_first_words:
            # Preserve special capitalization
            return f"{words[0].upper()} cells"
        else:
            # Standard two-word handling
            return f"{words[0].capitalize()} {words[1].capitalize()} cells"
    
    elif len(words) >= 3:
        # For three or more words, only capitalize the first word
        return f"{words[0].capitalize()} {' '.join(words[1:])} cells"
    
    # Fallback
    return f"{cell_type} cells"

def standardize_cell_type(cell_type):
    """
    Standardize cell type strings for flexible matching.
    Handles multi-word cell types, singular/plural forms, and capitalization.
    """
    # Clean and normalize the input
    clean_type = cell_type.lower().strip()
    if clean_type.endswith(' cells'):
        clean_type = clean_type[:-6].strip()
    elif clean_type.endswith(' cell'):
        clean_type = clean_type[:-5].strip()
    
    # Return the standardized base form
    return clean_type

def get_possible_cell_types(cell_type):
    """
    Generate all possible forms of a cell type for flexible matching.
    """
    # Get standardized base form
    base_form = standardize_cell_type(cell_type)
    # print ("TEST 1")
    # Generate variations with proper capitalization
    result = unified_cell_type_handler(cell_type)
    # print ("TEST 2")
    # Create variations based on the correct standardized form
    words = base_form.split()
    possible_types = [base_form]  # Add the base form
    # print ("TEST 3")
    # Add the properly standardized result and its variants
    possible_types.append(result)
    # print ("TEST 4")
    # Add variations with and without "cell" or "cells"
    if not result.lower().endswith("cells"):
        possible_types.append(f"{result} cells")
        # print ("TEST 4.5")
    # print ("TEST 5")
    if len(words) == 1:
        # For single words, add with/without "s" variations
        # print ("TEST 5.5")
        if words[0].endswith('s'):
            possible_types.append(words[0][:-1])  # Without 's'
        else:
            possible_types.append(f"{words[0]}s")  # With 's'
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(possible_types))

def filter_existing_genes(adata, gene_list):
    """Filter genes to only those present in the dataset, handling non-unique indices."""
    if hasattr(adata, 'raw') and adata.raw is not None:
        # Use raw var_names if available
        var_names = adata.raw.var_names
    else:
        var_names = adata.var_names
        
    # Use isin() which handles non-unique indices properly
    existing_genes = [gene for gene in gene_list if gene in var_names]
    return existing_genes

def preprocess_data(adata, sample_mapping=None):
    """Preprocess the AnnData object with consistent steps."""
    # Ensure var_names are unique before processing
    if not adata.var_names.is_unique:
        print("Warning: Gene names are not unique. Making them unique.")
        adata.var_names_make_unique()
    
    # Data preprocessing
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 20]
    non_mt_mask = ~adata.var['mt']
    adata = adata[:, non_mt_mask].copy()
    adata.layers['counts'] = adata.X.copy()  # used by scVI-tools
    
    # Normalization
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    
    # Ensure raw var_names are unique too
    if hasattr(adata, 'raw') and adata.raw is not None and not adata.raw.var_names.is_unique:
        print("Warning: Raw gene names are not unique. Making them unique.")
        adata.raw.var_names_make_unique()
    
    # Variable genes and neighbors
    if sample_mapping:
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
    """Perform UMAP and Leiden clustering with consistent parameters."""
    sc.tl.umap(adata, random_state=random_state)
    sc.tl.leiden(adata, resolution=resolution, random_state=random_state)
    
    # Add UMAP coordinates to obs for easier access
    umap_df = adata.obsm['X_umap']
    adata.obs['UMAP_1'] = umap_df[:, 0]
    adata.obs['UMAP_2'] = umap_df[:, 1]
    
    return adata

def rank_genes(adata, groupby='leiden', method='wilcoxon', n_genes=25, key_added=None):
    """Rank genes by group with customizable key name."""
    if key_added is None:
        key_added = f'rank_genes_{groupby}'
    
    # Check if there are enough cells in each group
    if len(adata.obs[groupby].unique()) <= 1:
        print(f"WARNING: Only one group found in {groupby}, cannot perform differential expression")
        # Create empty results to avoid errors
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
            # Create empty results
            adata.uns[key_added] = {
                'params': {'groupby': groupby},
                'names': np.zeros((1,), dtype=[('0', 'U50')])
            }
    
    return adata

def create_marker_anndata(adata, markers, copy_uns=True, copy_obsm=True):
    """Create a copy of AnnData with only marker genes."""
    # Filter markers to those present in the dataset
    markers = filter_existing_genes(adata, markers)
    markers = list(set(markers))
    
    # Check if we have any markers at all
    if len(markers) == 0:
        print("WARNING: No marker genes found in the dataset!")
        # Return a minimal valid AnnData to avoid errors
        return anndata.AnnData(
            X=scipy.sparse.csr_matrix((adata.n_obs, 0)),
            obs=adata.obs.copy()
        ), []
    
    # Create a new AnnData object with log-transformed data
    if hasattr(adata, 'raw') and adata.raw is not None:
        # Get indices of marker genes in the raw data
        raw_indices = [i for i, name in enumerate(adata.raw.var_names) if name in markers]
        
        # IMPORTANT: Use log-transformed layer if available, otherwise log-transform the raw data
        if hasattr(adata.raw, 'layers') and 'log1p' in adata.raw.layers:
            X = adata.raw.layers['log1p'][:, raw_indices].copy()
        else:
            # Get raw counts
            X = adata.raw.X[:, raw_indices].copy()
            # Log-transform if needed (check if data appears to be counts)
            if scipy.sparse.issparse(X):
                max_val = X.max()
            else:
                max_val = np.max(X)
            if max_val > 100:  # Heuristic to detect if data is not log-transformed
                print("Log-transforming marker data...")
                X = np.log1p(X)
    else:
        # Using main data
        main_indices = [i for i, name in enumerate(adata.var_names) if name in markers]
        X = adata.X[:, main_indices].copy()
    
    # Create the new AnnData object
    var = adata.var.iloc[main_indices].copy() if 'main_indices' in locals() else adata.raw.var.iloc[raw_indices].copy()
    marker_adata = anndata.AnnData(
        X=X,
        obs=adata.obs.copy(),
        var=var
    )
    
    # Copy additional data
    if copy_uns:
        marker_adata.uns = adata.uns.copy()
    
    if copy_obsm:
        marker_adata.obsm = adata.obsm.copy()
    
    if hasattr(adata, 'obsp'):
        marker_adata.obsp = adata.obsp.copy()
    
    return marker_adata, markers

def rank_ordering(adata_or_result, key=None, n_genes=25):
    """Extract top genes statistics from ranking results.
    
    Works with either AnnData object and key or direct result dictionary.
    """
    # Handle either AnnData with key or direct result dictionary
    if isinstance(adata_or_result, anndata.AnnData):
        if key is None:
            # Try to find a suitable key
            rank_keys = [k for k in adata_or_result.uns.keys() if k.startswith('rank_genes_')]
            if not rank_keys:
                raise ValueError("No rank_genes results found in AnnData object")
            key = rank_keys[0]
        gene_names = adata_or_result.uns[key]['names']
    else:
        # Assume it's a direct result dictionary
        gene_names = adata_or_result['names']
    
    # Extract gene names for each group
    top_genes_stats = {group: {} for group in gene_names.dtype.names}
    for group in gene_names.dtype.names:
        top_genes_stats[group]['gene'] = gene_names[group][:n_genes]
    
    # Convert to DataFrame
    top_genes_stats_df = pd.concat({group: pd.DataFrame(top_genes_stats[group])
                                  for group in top_genes_stats}, axis=0)
    top_genes_stats_df = top_genes_stats_df.reset_index()
    top_genes_stats_df = top_genes_stats_df.rename(columns={'level_0': 'cluster', 'level_1': 'index'})
    
    return top_genes_stats_df

def save_analysis_results(adata, prefix, leiden_key='leiden', save_umap=True, 
                         save_dendrogram=True, save_dotplot=False, markers=None):
    """Save analysis results to files with consistent naming."""
    # Save UMAP data
    print ("X1")
    if save_umap:
        print ("X2")
        umap_cols = ['UMAP_1', 'UMAP_2', leiden_key]
        if 'patient_name' in adata.obs.columns:
            umap_cols.append('patient_name')
        if 'cell_type' in adata.obs.columns:
            umap_cols.append('cell_type')
            
        adata.obs[umap_cols].to_csv(f"{prefix}_umap_data.csv", index=False)
    print ("X3")
    # Save dendrogram data
    if save_dendrogram and f'dendrogram_{leiden_key}' in adata.uns:
        print ("X4")
        dendrogram_data = adata.uns[f'dendrogram_{leiden_key}']
        pd_dendrogram_linkage = pd.DataFrame(
            dendrogram_data['linkage'],
            columns=['source', 'target', 'distance', 'count']
        )
        pd_dendrogram_linkage.to_csv(f"{prefix}_dendrogram_data.csv", index=False)
    print ("X5")
    # Save dot plot data
    if save_dotplot and markers:
        print ("X6")
        statistic_data = sc.get.obs_df(adata, keys=[leiden_key] + markers, use_raw=True)
        statistic_data.set_index(leiden_key, inplace=True)
        dot_plot_data = statistic_data.reset_index().melt(
            id_vars=leiden_key, var_name='gene', value_name='expression'
        )
        dot_plot_data.to_csv(f"{prefix}_dot_plot_data.csv", index=False)

# Main pipeline functions
def generate_umap(resolution=2):
    """Generate initial UMAP clustering on the full dataset."""
    global sample_mapping
    
    # Setup
    matplotlib.use('Agg')
    path = get_h5ad("media", ".h5ad")
    if not path:
        return ".h5ad file isn't given, unable to generate UMAP."
    # Load data
    adata = sc.read_h5ad(path)
    # Get sample mapping
    sample_mapping = get_mapping("media")
    # Apply sample mapping if available
    if sample_mapping:
        adata.obs['patient_name'] = adata.obs['Sample'].map(sample_mapping)
    # Preprocess data
    adata = preprocess_data(adata, sample_mapping)
    # sc.tl.pca(adata, svd_solver='arpack')
    # Perform clustering
    adata = perform_clustering(adata, resolution=resolution)
    # Rank all genes
    adata = rank_genes(adata, groupby='leiden', n_genes=25, key_added='rank_genes_all')
    # Get markers and create marker-specific AnnData
    markers = get_rag()
    marker_tree = markers.copy()
    markers = extract_genes(markers)
    adata_markers, filtered_markers = create_marker_anndata(adata, markers)
    # Rank genes in marker dataset
    adata_markers = rank_genes(adata_markers, n_genes=25, key_added='rank_genes_markers')
    # Copy marker ranking to original dataset
    adata.uns['rank_genes_markers'] = adata_markers.uns['rank_genes_markers']
    # Create dendrogram
    use_rep = 'X_scVI' if sample_mapping else None
    if use_rep:
        sc.tl.dendrogram(adata, groupby='leiden', use_rep=use_rep)
    else:
        sc.tl.dendrogram(adata, groupby='leiden')
    
    # Create dot plot
    # with plt.rc_context({'figure.figsize': (10, 10)}):
    #     print ("TEST 14")
    #     sc.pl.dotplot(adata, filtered_markers, groupby='leiden', swap_axes=True, use_raw=True,
    #                 standard_scale='var', dendrogram=True, color_map="Blues", save="dotplot.png")
    #     plt.close()
    # Initialize cell type as unknown
    adata.obs['cell_type'] = 'Unknown'
    # Save data
    # prefix = "schatbot/runtime_data/basic_data/Overall cells"
    save_analysis_results(
        adata, 
        prefix="schatbot/runtime_data/basic_data/Overall cells", 
        save_dotplot=True, 
        markers=filtered_markers
    )
    # save_analysis_results(
    #     adata, 
    #     prefix="process_cell_data/Overall cells", 
    #     save_dotplot=False
    # )
    # Extract top genes from marker-specific ranking
    top_genes_df = rank_ordering(adata, key='rank_genes_markers', n_genes=25)
    # Create gene dictionary
    gene_dict = {}
    for cluster, group in top_genes_df.groupby("cluster"):
        
        gene_dict[cluster] = list(group["gene"])
    #adding
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
    pth = "annotated_adata/Overall cells_annotated_adata.pkl"
    os.makedirs(os.path.dirname(pth), exist_ok=True)
    with open(pth, "wb") as file:
        pickle.dump(adata, file)
    label_clusters(adata=adata, cell_type="Overall", annotation_result=annotation_result)
    print(annotation_result)
    #done adding
    return gene_dict, marker_tree, adata #new
    # return str(annotation_result)

def process_cells(adata, cell_type, resolution=None):
    """Process specific cell type with consistent workflow."""
    if resolution is None:
        resolution = 1  # Default higher resolution for subtype clustering
    
    # Get all possible variations of the cell type for matching
    possible_types = get_possible_cell_types(cell_type)
    standardized_name = unified_cell_type_handler(cell_type)
    
    # Filter cells based on cell type with flexible matching
    mask = adata.obs['cell_type'].isin(possible_types)
    if mask.sum() == 0:
        print(f"WARNING: No cells found with cell type matching any of these variations: {possible_types}")
        return None, None, None
    
    filtered_cells = adata[mask].copy()
    
    # Perform new clustering on filtered cells
    sc.tl.pca(filtered_cells, svd_solver='arpack')
    sc.pp.neighbors(filtered_cells)
    filtered_cells = perform_clustering(filtered_cells, resolution=resolution)
    
    # Create ranking key based on cell type
    # Use standardized base form for key naming
    base_form = standardize_cell_type(cell_type).replace(' ', '_').lower()
    rank_key = f"rank_genes_{base_form}"
    
    # Rank genes for the filtered cells
    filtered_cells = rank_genes(filtered_cells, key_added=rank_key)
    
    # Get markers and create marker-specific sub-AnnData if needed
    markers = get_subtypes(cell_type)
    markers_tree = markers.copy()
    markers_list = extract_genes(markers)
    
    # Check if we have enough markers
    if not markers_list:
        print(f"WARNING: No marker genes found for {standardized_name}. Using general ranking.")
        # Use general ranking instead
        top_genes_df = rank_ordering(filtered_cells, key=rank_key, n_genes=25)
        
        # Create gene dictionary
        gene_dict = {}
        for cluster, group in top_genes_df.groupby("cluster"):
            gene_dict[cluster] = list(group["gene"])
        
        # Create dendrogram
        sc.tl.dendrogram(filtered_cells, groupby='leiden')
        
        # Save data
        save_analysis_results(
            filtered_cells,
            prefix=f"process_cell_data/{standardized_name}",
            save_dotplot=False
        )
        
        # Save processed data
        umap_df = filtered_cells.obs[['UMAP_1', 'UMAP_2', 'leiden', 'cell_type']]
        if 'patient_name' in filtered_cells.obs.columns:
            umap_df['patient_name'] = filtered_cells.obs['patient_name']
            
        fname = f'{standardized_name}_adata_processed.pkl'
        with open(fname, 'wb') as file:
            pickle.dump(umap_df, file)
        

        return gene_dict, filtered_cells, markers_tree
    
    # Rest of the function remains the same, but using standardized_name instead of 
    # the previous f"{cell_type_cap} cells" format for file paths and variable names
    
    # Create marker-specific AnnData
    filtered_markers, marker_list = create_marker_anndata(filtered_cells, markers_list)
    
    # Skip marker-specific ranking if fewer than 5 markers were found
    if len(marker_list) < 5:
        print(f"WARNING: Only {len(marker_list)} marker genes found. Using general gene ranking.")
        # Use general ranking instead
        top_genes_df = rank_ordering(filtered_cells, key=rank_key, n_genes=25)
    else:
        try:
            # Rank marker genes
            marker_key = f"rank_markers_{base_form}"
            filtered_markers = rank_genes(filtered_markers, key_added=marker_key)
            
            # Copy marker ranking to filtered dataset
            filtered_cells.uns[marker_key] = filtered_markers.uns[marker_key]
            
            # Extract top genes with preference for marker-specific ranking
            top_genes_df = rank_ordering(filtered_markers, key=marker_key, n_genes=25)
        except Exception as e:
            print(f"ERROR in marker-specific ranking: {e}")
            print("Falling back to general ranking")
            # Use general ranking in case of any error
            top_genes_df = rank_ordering(filtered_cells, key=rank_key, n_genes=25)
    
    # Create gene dictionary
    gene_dict = {}
    for cluster, group in top_genes_df.groupby("cluster"):
        gene_dict[cluster] = list(group["gene"])
    
    # Create dendrogram
    sc.tl.dendrogram(filtered_cells, groupby='leiden')
    
    # Save data
    save_analysis_results(
        filtered_cells,
        prefix=f"process_cell_data/{standardized_name}",
        save_dotplot=len(marker_list) >= 5,
        markers=marker_list
    )
    
    # Save processed data
    umap_df = filtered_cells.obs[['UMAP_1', 'UMAP_2', 'leiden', 'cell_type']]
    if 'patient_name' in filtered_cells.obs.columns:
        umap_df['patient_name'] = filtered_cells.obs['patient_name']
        
    fname = f'{standardized_name}_adata_processed.pkl'
    with open(fname, 'wb') as file:
        pickle.dump(umap_df, file)
    
    return gene_dict, filtered_cells, markers_tree

def label_clusters(annotation_result, cell_type, adata):
    """Label clusters with consistent cell type handling."""
    # Standardize cell type names
    standardized_name = unified_cell_type_handler(cell_type)
    base_form = standardize_cell_type(cell_type).lower()
    print ("HERELC 1")
    try:
        adata = adata.copy()
        print ("HERELC 2")
        # Parse annotation mapping
        start_idx = annotation_result.find("{")
        end_idx = annotation_result.rfind("}") + 1
        str_map = annotation_result[start_idx:end_idx]
        map2 = ast.literal_eval(str_map)
        map2 = {str(key): value for key, value in map2.items()}
        print ("HERELC 3")
        # Apply annotations - different handling for overall cells vs specific cell types
        if base_form == "overall":
            # For overall cells, directly apply annotations to the main dataset
            adata.obs['cell_type'] = 'Unknown'
            for group, cell_type_value in map2.items():
                adata.obs.loc[adata.obs['leiden'] == group, 'cell_type'] = cell_type_value
            print ("HERELC 5")
            # Save annotated data
            save_analysis_results(
                adata,
                prefix=f"umaps/annotated/{standardized_name}",
                save_dendrogram=False,
                save_dotplot=False
            )
            print ("HERELC 6")
            # Save annotated adata
            fname = f'annotated_adata/{standardized_name}_annotated_adata.pkl'
            with open(fname, "wb") as file:
                pickle.dump(adata, file)
                
        else:
            # For specific cell types, we need to re-cluster first
            # This is typically called after process_cells()
            specific_cells = adata.copy()
            
            # Apply annotations
            specific_cells.obs['cell_type'] = 'Unknown'
            for group, cell_type_value in map2.items():
                specific_cells.obs.loc[specific_cells.obs['leiden'] == group, 'cell_type'] = cell_type_value
            print ("HERELC 7")
            # Save annotated data
            save_analysis_results(
                specific_cells,
                prefix=f"umaps/annotated/{standardized_name}",
                save_dendrogram=False,
                save_dotplot=False
            )
            print ("HERELC 8")
            # Save annotated adata
            fname = f'annotated_adata/{standardized_name}_annotated_adata.pkl'
            with open(fname, "wb") as file:
                pickle.dump(specific_cells, file)
            print ("HERELC 9")
            return specific_cells
            
    except (SyntaxError, ValueError) as e:
        print(f"Error in parsing the map: {e}")
        
    return adata

if __name__ == "__main__":
    clear_directory("annotated_adata")
    clear_directory("basic_data")
    clear_directory("umaps")
    clear_directory("process_cell_data")  
    clear_directory("figures") 
    gene_dict, marker_tree, adata = generate_umap()  
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
    print(annotation_result)

def repeat():
    return "Hello"