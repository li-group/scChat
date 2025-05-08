import os
import json
import gseapy as gp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def gsea_analysis() -> str:
    """
    Performs GSEA analysis using significant genes from SGP.csv.
    Saves output plots and CSV files, then returns a JSON summary.
    """
    try:
        SGP = pd.read_csv('SGP.csv')
        logger.info("SGP.csv loaded successfully")
    except Exception as e:
        logger.error(f"Error loading SGP.csv: {e}")
        return json.dumps({"status": "error", "message": f"Error loading SGP.csv: {e}"})
    
    try:
        significant_genes_post = SGP
        significant_genes_post['rank'] = -np.log10(significant_genes_post.pvals_adj) * significant_genes_post.logfoldchanges
        significant_genes_post = significant_genes_post.sort_values('rank', ascending=False).reset_index(drop=True)
        ranking = significant_genes_post[['genes', 'rank']]
        gene_list = ranking['genes'].str.strip().to_list()
        logger.info("Gene list prepared")
    except Exception as e:
        logger.error(f"Error preparing gene list: {e}")
        return json.dumps({"status": "error", "message": f"Error preparing gene list: {e}"})
    
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
        logger.info("GSEA analysis completed")
    except Exception as e:
        logger.error(f"Error in GSEA analysis: {e}")
        return json.dumps({"status": "error", "message": f"Error in GSEA analysis: {e}"})
    
    out = []
    for term in pre_res.results:
        fdr = pre_res.results[term]['fdr']
        es = pre_res.results[term]['es']
        nes = pre_res.results[term]['nes']
        if fdr <= 0.05:
            out.append([term, fdr, es, nes])
    
    try:
        out_df = pd.DataFrame(out, columns=['Term', 'fdr', 'es', 'nes']).sort_values('fdr').reset_index(drop=True)
        logger.info("Filtered significant terms")
    except Exception as e:
        logger.error(f"Error filtering significant terms: {e}")
        return json.dumps({"status": "error", "message": f"Error filtering significant terms: {e}"})
    
    try:
        os.makedirs('gsea_plots', exist_ok=True)
        logger.info("Directory for plots ensured")
    except Exception as e:
        logger.error(f"Error ensuring plot directory: {e}")
        return json.dumps({"status": "error", "message": f"Error ensuring plot directory: {e}"})
    
    try:
        axs = pre_res.plot(terms=out_df['Term'], show_ranking=False, legend_kws={'loc': (1.05, 0)})
        plt.title("GSEA Enrichment Scores for Significant Terms (FDR ≤ 0.05)")
        plt.xlabel("Rank in Ordered Dataset")
        plt.ylabel("Enrichment Score (ES)")
        plt.savefig('gsea_plots/enrichment_scores.png')
        plt.close()
        logger.info("Enrichment scores plot saved")
    except Exception as e:
        logger.error(f"Failed to plot all terms together: {e}")
    
    try:
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            x=out_df['nes'],
            y=out_df['Term'],
            s=(out_df['es'].abs() * 500),
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
        logger.info("GSEA dot plot saved")
    except Exception as e:
        logger.error(f"Error saving GSEA dot plot: {e}")
    
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
            logger.info(f"GSEA plot data saved to '{output_file}'")
        else:
            logger.info("No significant gene sets found with FDR <= 0.05.")
    except Exception as e:
        logger.error(f"Error saving GSEA plot data: {e}")
    
    try:
        ranking_gene_list = significant_genes_post[['genes', 'rank']]
        ranking_gene_list.to_csv('ranking_gene_list.csv', index=False)
        logger.info("Ranking gene list saved to 'ranking_gene_list.csv'")
    except Exception as e:
        logger.error(f"Error saving ranking gene list: {e}")
    
    response = {
        "status": "success",
        "message": "GSEA analysis completed successfully",
        "enrichment_scores_plot": 'gsea_plots/enrichment_scores.png',
        "dot_plot": 'gsea_plots/gsea_dot_plot.png',
        "gsea_plot_data": 'gsea_plot_data.csv',
        "ranking_gene_list": 'ranking_gene_list.csv'
    }
    return json.dumps(response)