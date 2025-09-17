"""
Unified Result Accessor - Execution-plan-driven result retrieval
No hardcoding, purely based on execution history function calls and their success status.
"""

import os
import pandas as pd
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import uuid
import logging
logger = logging.getLogger(__name__)


MAX_PLOT_HTML_BYTES = 8_388_608

def _normalize_viz_result(result):
    """
    Normalize visualization payloads into a small dict so the LLM can be told
    that visualizations exist, regardless of size/shape.
    """
    MAX_PLOT_HTML_BYTES = 8_388_608
    if isinstance(result, str):
        if len(result) <= MAX_PLOT_HTML_BYTES:
            return {"kind": "inline_html", "html_length": len(result)}
        else:
            return {"kind": "too_large_inline", "html_length": len(result)}
    if isinstance(result, dict):
        if result.get("type") == "file_ref":
            return {"kind": "file_ref", "path": result.get("path")}
        if result.get("multiple_plots"):
            return {"kind": "bundle", "count": len(result.get("plots", []))}
    # fallback
    return {"kind": "unknown"}


class ResultAccessorBase(ABC):
    """Base class for analysis-specific result accessors"""
    
    @abstractmethod
    def can_handle(self, function_name: str) -> bool:
        """Check if this accessor can handle the given function name"""
        pass
    
    @abstractmethod
    def get_results(self, execution_step: Dict[str, Any]) -> Dict[str, Any]:
        """Get results for this execution step"""
        pass
    
    @abstractmethod
    def format_for_synthesis(self, results: Dict[str, Any], cell_type: str) -> str:
        """Format results for LLM synthesis"""
        pass


class EnrichmentResultAccessor(ResultAccessorBase):
    """Accessor for enrichment analysis results"""
    
    def can_handle(self, function_name: str) -> bool:
        return function_name == "perform_enrichment_analyses"
    
    def get_results(self, execution_step: Dict[str, Any]) -> Dict[str, Any]:
        """Get enrichment results from CSV files (reliable) + vector DB availability"""
        try:
            # Extract parameters from execution step
            step_data = execution_step.get("step", {})
            parameters = step_data.get("parameters", {})
            cell_type = parameters.get("cell_type", "unknown")
            analyses = parameters.get("analyses", ["go", "kegg", "reactome", "gsea"])
            
            results = {
                "cell_type": cell_type,
                "analysis_types": {},
                "vector_search_available": True,
                "source": "enrichment_csv_files"
            }
            
            # Get conditions dynamically from sample mapping
            conditions = self._load_conditions_from_mapping()
            
            # Map analysis types to their directory names (both full dataset and condition-specific)
            analysis_dirs = {
                "go": ["go_bp", "go_cc", "go_mf"] + [f"go_{condition}" for condition in conditions],
                "kegg": ["kegg"] + [f"kegg_{condition}" for condition in conditions],
                "reactome": ["reactome"] + [f"reactome_{condition}" for condition in conditions],
                "gsea": ["gsea"] + [f"gsea_{condition}" for condition in conditions]
            }

            for analysis_type in analyses:
                if analysis_type == "gsea":
                    # Special handling for GSEA with library-specific folders
                    import glob

                    # Try library-specific folders first
                    search_patterns = [
                        f"scchatbot/enrichment/gsea_*/results_summary_{cell_type}_*.csv",
                        f"scchatbot/enrichment/gsea*/results_summary_{cell_type}_*.csv",
                        f"scchatbot/enrichment/gsea_*/results_summary_{cell_type}.csv"
                    ]

                    found_files = []
                    for pattern in search_patterns:
                        found_files.extend(glob.glob(pattern))

                    # Process found files
                    for csv_path in found_files:
                        if os.path.exists(csv_path):
                            try:
                                df = pd.read_csv(csv_path)
                                if len(df) > 0:
                                    # Handle different column name variations across analysis types
                                    term_column = None
                                    for col_name in ["term_name", "Term", "pathway", "term"]:
                                        if col_name in df.columns:
                                            term_column = col_name
                                            break

                                    # Handle p-value column variations
                                    p_value_column = None
                                    for col_name in ["adj_p_value", "p_value", "pvalue", "FDR"]:
                                        if col_name in df.columns:
                                            p_value_column = col_name
                                            break

                                    if term_column and p_value_column:
                                        # Get folder name for key (e.g., gsea_MSigDBHallmark2020)
                                        folder_name = os.path.basename(os.path.dirname(csv_path))

                                        # Get top terms with their statistics
                                        results["analysis_types"][folder_name] = {
                                            "top_terms": df[term_column].head(10).tolist(),
                                            "p_values": df[p_value_column].head(10).tolist(),
                                            "total_significant": len(df),
                                            "source_file": csv_path,
                                            "columns_used": {"term": term_column, "p_value": p_value_column}
                                        }
                                    else:
                                        logger.info(f"⚠️ Column mapping failed for {csv_path}: term_col={term_column}, p_val_col={p_value_column}")
                                        logger.info(f"   Available columns: {list(df.columns)}")
                            except Exception as e:
                                logger.info(f"⚠️ Error reading {csv_path}: {e}")

                    # Fallback to simple structure if no library-specific files found
                    if not results["analysis_types"] and analysis_type in analysis_dirs:
                        for subdir in analysis_dirs[analysis_type]:
                            csv_path = f"scchatbot/enrichment/{subdir}/results_summary_{cell_type}.csv"
                            if os.path.exists(csv_path):
                                # Process fallback files (same logic as other analyses)
                                try:
                                    df = pd.read_csv(csv_path)
                                    if len(df) > 0:
                                        # Handle different column name variations across analysis types
                                        term_column = None
                                        for col_name in ["term_name", "Term", "pathway", "term"]:
                                            if col_name in df.columns:
                                                term_column = col_name
                                                break

                                        # Handle p-value column variations
                                        p_value_column = None
                                        for col_name in ["adj_p_value", "p_value", "pvalue", "FDR"]:
                                            if col_name in df.columns:
                                                p_value_column = col_name
                                                break

                                        if term_column and p_value_column:
                                            # Get top terms with their statistics
                                            results["analysis_types"][subdir] = {
                                                "top_terms": df[term_column].head(10).tolist(),
                                                "p_values": df[p_value_column].head(10).tolist(),
                                                "total_significant": len(df),
                                                "source_file": csv_path,
                                                "columns_used": {"term": term_column, "p_value": p_value_column}
                                            }
                                        else:
                                            logger.info(f"⚠️ Column mapping failed for {csv_path}: term_col={term_column}, p_val_col={p_value_column}")
                                            logger.info(f"   Available columns: {list(df.columns)}")
                                except Exception as e:
                                    logger.info(f"⚠️ Error reading {csv_path}: {e}")

                elif analysis_type in analysis_dirs:
                    # Original logic for GO, KEGG, Reactome
                    for subdir in analysis_dirs[analysis_type]:
                        csv_path = f"scchatbot/enrichment/{subdir}/results_summary_{cell_type}.csv"

                        if os.path.exists(csv_path):
                            try:
                                df = pd.read_csv(csv_path)
                                if len(df) > 0:
                                    # Handle different column name variations across analysis types
                                    term_column = None
                                    for col_name in ["term_name", "Term", "pathway", "term"]:
                                        if col_name in df.columns:
                                            term_column = col_name
                                            break
                                    
                                    # Handle p-value column variations
                                    p_value_column = None
                                    for col_name in ["adj_p_value", "p_value", "pvalue", "FDR"]:
                                        if col_name in df.columns:
                                            p_value_column = col_name
                                            break
                                    
                                    if term_column and p_value_column:
                                        # Get top terms with their statistics
                                        results["analysis_types"][subdir] = {
                                            "top_terms": df[term_column].head(10).tolist(),
                                            "p_values": df[p_value_column].head(10).tolist(),
                                            "total_significant": len(df),
                                            "source_file": csv_path,
                                            "columns_used": {"term": term_column, "p_value": p_value_column}
                                        }
                                    else:
                                        logger.info(f"⚠️ Column mapping failed for {csv_path}: term_col={term_column}, p_val_col={p_value_column}")
                                        logger.info(f"   Available columns: {list(df.columns)}")
                            except Exception as e:
                                logger.info(f"⚠️ Error reading {csv_path}: {e}")
            
            return results
            
        except Exception as e:
            logger.info(f"❌ EnrichmentResultAccessor error: {e}")
            return {"error": str(e), "cell_type": "unknown"}
    
    def format_for_synthesis(self, results: Dict[str, Any], cell_type: str) -> str:
        """Format enrichment results with specific pathway names and p-values"""
        if "error" in results:
            return f"  Enrichment analysis error for {cell_type}: {results['error']}"
        
        lines = [f"ENRICHMENT ANALYSIS - {cell_type.upper()}:"]
        
        analysis_types = results.get("analysis_types", {})
        if not analysis_types:
            return f"  No enrichment results found for {cell_type}"
        
        for analysis_name, data in analysis_types.items():
            top_terms = data.get("top_terms", [])
            p_values = data.get("p_values", [])
            total = data.get("total_significant", 0)
            
            if top_terms:
                # Format top terms with p-values
                formatted_terms = []
                for i, term in enumerate(top_terms[:5]):  # Top 5 terms
                    pval = p_values[i] if i < len(p_values) else "N/A"
                    if isinstance(pval, (int, float)):
                        formatted_terms.append(f"{term} (p={pval:.2e})")
                    else:
                        formatted_terms.append(f"{term} (p={pval})")
                
                lines.append(f"  {analysis_name.upper()}: {'; '.join(formatted_terms)}")
                lines.append(f"    Total significant: {total} terms")
        
        return "\n".join(lines)
    
    def _load_conditions_from_mapping(self) -> List[str]:
        """Load condition names dynamically from sample mapping file"""
        try:
            import json
            import os
            
            mapping_file = "media/sample_mapping.json"
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r') as f:
                    mapping_data = json.load(f)
                
                # Get conditions from Sample categories or Sample description
                conditions = []
                if "Sample categories" in mapping_data:
                    conditions.extend(mapping_data["Sample categories"].keys())
                elif "Sample description" in mapping_data:
                    conditions.extend(mapping_data["Sample description"].keys())
                
                # Keep original format as enrichment function saves with spaces, not underscores
                conditions = list(conditions)
                
                logger.info(f"🔍 ENRICHMENT: Loaded {len(conditions)} conditions from mapping file: {conditions}")
                return conditions
            else:
                raise FileNotFoundError(f"Sample mapping file not found: {mapping_file}")
                
        except Exception as e:
            raise Exception(f"Error loading conditions from mapping file: {e}")


class DEAResultAccessor(ResultAccessorBase):
    """Accessor for differential expression analysis results"""
    
    def can_handle(self, function_name: str) -> bool:
        return function_name == "dea_split_by_condition"
    
    def get_results(self, execution_step: Dict[str, Any]) -> Dict[str, Any]:
        """Get DEA results from CSV files"""
        try:
            # Extract cell type from execution step
            step_data = execution_step.get("step", {})
            parameters = step_data.get("parameters", {})
            cell_type = parameters.get("cell_type", "unknown")
            
            results = {
                "cell_type": cell_type,
                "conditions": {},
                "source": "dea_csv_files"
            }
            
            # Look for condition-specific marker files
            # Load conditions dynamically from sample mapping file
            potential_conditions = self._load_conditions_from_mapping()
            
            for condition in potential_conditions:
                csv_path = f"scchatbot/deg_res/{cell_type}_markers_{condition}.csv"
                
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        if len(df) > 0:
                            # Debug: Print column names
                            logger.info(f"📊 DEA CSV columns for {condition}: {list(df.columns)}")
                            
                            # Determine the correct column names
                            logfc_col = "log_fc" if "log_fc" in df.columns else "logFC" if "logFC" in df.columns else "logfoldchanges"
                            pval_col = "p_adj" if "p_adj" in df.columns else "FDR" if "FDR" in df.columns else "pvals_adj"
                            
                            # Separate upregulated and downregulated genes
                            # Extract top 50 genes to ensure important genes like CFD are included
                            df_up = df[df[logfc_col] > 0].nlargest(100, logfc_col)
                            df_down = df[df[logfc_col] < 0].nsmallest(100, logfc_col)
                            
                            results["conditions"][condition] = {
                                "upregulated_genes": df_up.to_dict('records'),
                                "downregulated_genes": df_down.to_dict('records'),
                                "total_significant": len(df[df[pval_col] < 0.05]),
                                "source_file": csv_path
                            }
                    except Exception as e:
                        logger.info(f"⚠️ Error reading DEA file {csv_path}: {e}")
            
            return results
            
        except Exception as e:
            logger.info(f"❌ DEAResultAccessor error: {e}")
            return {"error": str(e), "cell_type": "unknown"}
    
    def format_for_synthesis(self, results: Dict[str, Any], cell_type: str) -> str:
        """Format DEA results with specific gene names and fold changes"""
        if "error" in results:
            return f"  DEA analysis error for {cell_type}: {results['error']}"
        
        lines = [f"DIFFERENTIAL EXPRESSION ANALYSIS - {cell_type.upper()}:"]
        
        conditions = results.get("conditions", {})
        if not conditions:
            return f"  No DEA results found for {cell_type}"
        
        for condition, data in conditions.items():
            # Special handling for bulk analysis
            if condition == "bulk":
                condition_name = "Bulk Analysis (All Conditions Combined)"
            else:
                condition_name = condition.replace("_", " ").title()
            lines.append(f"  {condition_name}:")
            
            # Format upregulated genes - provide ALL extracted genes for complete LLM analysis
            all_up_genes = data.get("upregulated_genes", [])
            if all_up_genes:
                # Show top 10 for summary
                top_genes = all_up_genes[:10]
                gene_strs = []
                for gene in top_genes:
                    gene_name = gene.get("names", gene.get("gene", gene.get("Gene", "Unknown")))
                    log_fc = gene.get("logfoldchanges", gene.get("log_fc", gene.get("logFC", 0)))
                    gene_strs.append(f"{gene_name} (FC={log_fc:.2f})")
                lines.append(f"    Top 10 upregulated: {', '.join(gene_strs)}")
                
                # Provide ALL upregulated genes for LLM analysis (not just a subset)
                total_up = len(all_up_genes)
                lines.append(f"    Total upregulated genes available: {total_up}")
                
                # Include ALL genes in formatted output for complete LLM access
                all_gene_strs = []
                for gene in all_up_genes:
                    gene_name = gene.get("names", gene.get("gene", gene.get("Gene", "Unknown")))
                    log_fc = gene.get("logfoldchanges", gene.get("log_fc", gene.get("logFC", 0)))
                    all_gene_strs.append(f"{gene_name} (FC={log_fc:.2f})")
                lines.append(f"    All upregulated genes: {', '.join(all_gene_strs)}")
            
            # Format downregulated genes - provide ALL extracted genes for complete LLM analysis
            all_down_genes = data.get("downregulated_genes", [])
            if all_down_genes:
                # Show top 10 for summary
                top_genes = all_down_genes[:10]
                gene_strs = []
                for gene in top_genes:
                    gene_name = gene.get("names", gene.get("gene", gene.get("Gene", "Unknown")))
                    log_fc = gene.get("logfoldchanges", gene.get("log_fc", gene.get("logFC", 0)))
                    gene_strs.append(f"{gene_name} (FC={log_fc:.2f})")
                lines.append(f"    Top 10 downregulated: {', '.join(gene_strs)}")
                
                # Provide ALL downregulated genes for LLM analysis
                total_down = len(all_down_genes)
                lines.append(f"    Total downregulated genes available: {total_down}")
                
                # Include ALL genes in formatted output for complete LLM access
                all_gene_strs = []
                for gene in all_down_genes:
                    gene_name = gene.get("names", gene.get("gene", gene.get("Gene", "Unknown")))
                    log_fc = gene.get("logfoldchanges", gene.get("log_fc", gene.get("logFC", 0)))
                    all_gene_strs.append(f"{gene_name} (FC={log_fc:.2f})")
                lines.append(f"    All downregulated genes: {', '.join(all_gene_strs)}")
            
            total = data.get("total_significant", 0)
            lines.append(f"    Total significant genes: {total}")
        
        return "\n".join(lines)
    
    def _load_conditions_from_mapping(self) -> List[str]:
        """Load condition names dynamically from sample mapping file"""
        try:
            import json
            import os
            
            mapping_file = "media/sample_mapping.json"
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r') as f:
                    mapping_data = json.load(f)
                
                # Get conditions from Sample categories or Sample description
                conditions = []
                if "Sample categories" in mapping_data:
                    conditions.extend(mapping_data["Sample categories"].values())
                elif "Sample description" in mapping_data:
                    conditions.extend(mapping_data["Sample description"].keys())
                
                # Add "bulk" to handle bulk analysis results
                conditions.append("bulk")
                
                # Keep original format as DEA function saves with spaces, not underscores
                conditions = list(set(conditions))  # Remove duplicates
                
                logger.info(f"🔍 DEA: Loaded {len(conditions)} conditions from mapping file: {conditions}")
                return conditions
            else:
                raise FileNotFoundError(f"Sample mapping file not found: {mapping_file}")
                
        except Exception as e:
            raise Exception(f"Error loading conditions from mapping file: {e}")


class CellCountResultAccessor(ResultAccessorBase):
    """Accessor for cell count comparison results"""
    
    def can_handle(self, function_name: str) -> bool:
        return function_name == "compare_cell_counts"
    
    def get_results(self, execution_step: Dict[str, Any]) -> Dict[str, Any]:
        """Get cell count results from execution step result"""
        try:
            # Cell count results are typically stored directly in execution result
            # since they're computed on-demand
            result = execution_step.get("result")
            step_data = execution_step.get("step", {})
            parameters = step_data.get("parameters", {})
            cell_type = parameters.get("cell_type", "unknown")
            
            if isinstance(result, dict):
                return {
                    "cell_type": cell_type,
                    "count_data": result,
                    "source": "execution_result"
                }
            elif isinstance(result, str):
                # Try to parse structured result from string
                # This is a fallback for when results are stored as text
                return {
                    "cell_type": cell_type,
                    "summary": result,
                    "source": "execution_result_text"
                }
            else:
                return {
                    "cell_type": cell_type,
                    "error": f"Unexpected result type: {type(result)}",
                    "source": "execution_result"
                }
                
        except Exception as e:
            logger.info(f"❌ CellCountResultAccessor error: {e}")
            return {"error": str(e), "cell_type": "unknown"}
    
    def format_for_synthesis(self, results: Dict[str, Any], cell_type: str) -> str:
        """Format cell count results with condition comparisons"""
        if "error" in results:
            return f"  Cell count analysis error for {cell_type}: {results['error']}"
        
        lines = [f"CELL COUNT COMPARISON - {cell_type.upper()}:"]
        
        count_data = results.get("count_data", {})
        if count_data:
            # Handle structured count data
            count_results = count_data.get("count_results", [])
            for result in count_results:
                category = result.get("category", "Unknown")
                count = result.get("cell_count", 0)
                description = result.get("description", "")
                lines.append(f"  {category}: {count} cells")
                if description:
                    lines.append(f"    {description}")
        
        summary_text = results.get("summary", "")
        if summary_text:
            lines.append(f"  Summary: {summary_text}")
        
        return "\n".join(lines)


class SemanticSearchResultAccessor(ResultAccessorBase):
    """Accessor for semantic search results"""
    
    def can_handle(self, function_name: str) -> bool:
        return function_name == "search_enrichment_semantic"
    
    def get_results(self, execution_step: Dict[str, Any]) -> Dict[str, Any]:
        """Get semantic search results from execution step"""
        try:
            result = execution_step.get("result")
            step_data = execution_step.get("step", {})
            parameters = step_data.get("parameters", {})
            cell_type = parameters.get("cell_type", "unknown")
            query = parameters.get("query", "unknown")
            
            # DEBUG: Log parameter extraction for cell type confusion debugging
            logger.info(f"🔍 SEMANTIC SEARCH DEBUG: Extracted cell_type='{cell_type}', query='{query}'")
            logger.info(f"🔍 SEMANTIC SEARCH DEBUG: Full parameters: {parameters}")
            
            # Also check if result contains different cell type info
            if isinstance(result, dict) and "cell_type" in result:
                result_cell_type = result.get("cell_type")
                if result_cell_type != cell_type:
                    logger.info(f"⚠️ CELL TYPE MISMATCH: Parameters say '{cell_type}', result says '{result_cell_type}'")
                    # Use the result's cell type as it's more reliable
                    cell_type = result_cell_type
            
            if isinstance(result, dict):
                return {
                    "cell_type": cell_type,
                    "query": query,
                    "search_data": result,
                    "source": "execution_result"
                }
            else:
                return {
                    "cell_type": cell_type,
                    "query": query,
                    "error": f"Unexpected result type: {type(result)}",
                    "source": "execution_result"
                }
                
        except Exception as e:
            logger.info(f"❌ SemanticSearchResultAccessor error: {e}")
            return {"error": str(e), "cell_type": "unknown", "query": "unknown"}
    
    def format_for_synthesis(self, results: Dict[str, Any], cell_type: str) -> str:
        """Format semantic search results"""
        if "error" in results:
            return f"  Semantic search error for {cell_type}: {results['error']}"
        
        query = results.get("query", "unknown")
        lines = [f"SEMANTIC SEARCH - {cell_type.upper()} for '{query}':"]
        
        search_data = results.get("search_data", {})
        if search_data:
            search_results = search_data.get("search_results", {})
            results_list = search_results.get("results", [])
            
            if results_list:
                lines.append(f"  Found {len(results_list)} related terms:")
                for i, result in enumerate(results_list[:5], 1):  # Top 5 results
                    term_name = result.get("term_name", "Unknown")
                    p_value = result.get("p_value", "N/A")
                    similarity = result.get("similarity_score", 0)
                    analysis_type = result.get("analysis_type", "unknown").upper()
                    lines.append(f"    {i}. [{analysis_type}] {term_name} (p={p_value}, sim={similarity:.3f})")
            else:
                lines.append(f"  No related terms found for '{query}'")
        
        return "\n".join(lines)


class ValidationResultAccessor(ResultAccessorBase):
    """Accessor for validation step results"""
    
    def can_handle(self, function_name: str) -> bool:
        return function_name == "validate_processing_results"
    
    def get_results(self, execution_step: Dict[str, Any]) -> Dict[str, Any]:
        """Get validation results from execution step"""
        try:
            result = execution_step.get("result")
            step_data = execution_step.get("step", {})
            parameters = step_data.get("parameters", {})
            
            return {
                "validation_result": result,
                "parameters": parameters,
                "source": "execution_result"
            }
                
        except Exception as e:
            logger.info(f"❌ ValidationResultAccessor error: {e}")
            return {"error": str(e)}
    
    def format_for_synthesis(self, results: Dict[str, Any], cell_type: str) -> str:
        """Format validation results"""
        if "error" in results:
            return f"  Validation error: {results['error']}"
        
        # Validation results are typically not important for synthesis
        return f"  Validation completed successfully"


class VisualizationResultAccessor(ResultAccessorBase):
    """Accessor for visualization/display results"""
    
    def can_handle(self, function_name: str) -> bool:
        # return function_name.startswith("display_")
        return isinstance(function_name, str) and function_name.startswith("display_")
    
    def get_results(self, execution_step: Dict[str, Any]) -> Dict[str, Any]:
        """Get visualization results from execution step"""
        try:
            # result = execution_step.get("result")
            raw_result = execution_step.get("result")
            meta = execution_step.get("result_metadata", {})

            step_data = execution_step.get("step", {})
            parameters = step_data.get("parameters", {})
            function_name = step_data.get("function_name", "unknown_display")
            cell_type = parameters.get("cell_type", "unknown")
            
            # # Check if result contains HTML plot data
            # has_html_plot = isinstance(result, str) and ('<div' in result or '<html' in result) and len(result) > 1000
            
            # return {
            #     "function_name": function_name,
            #     "cell_type": cell_type,
            #     "parameters": parameters,
            #     "has_plot": has_html_plot,
            #     "plot_size": len(str(result)) if result else 0,
            #     "plot_html": result if has_html_plot else None,
            #     "source": "execution_result"
            # }
            # Normalize to recognize inline HTML, file_ref, or bundles
            norm = _normalize_viz_result(raw_result)
            has_plot = norm.get("kind") in {"inline_html", "too_large_inline", "file_ref", "bundle"}
            plot_size = len(raw_result) if isinstance(raw_result, str) else 0

            return {
                "function_name": function_name,
                "cell_type": cell_type,
                "parameters": parameters,
                "result": raw_result,          # keep raw payload for downstream consumers
                "metadata": meta,              # optional metadata from executor
                "has_plot": has_plot,
                "plot_kind": norm.get("kind", "unknown"),
                "plot_size": plot_size,
                "source": "execution_result",
            }

                
        except Exception as e:
            logger.info(f"❌ VisualizationResultAccessor error: {e}")
            return {"error": str(e), "cell_type": "unknown", "function_name": "unknown"}
    
    def format_for_synthesis(self, results: Dict[str, Any], cell_type: str) -> str:
        """Format visualization results"""
        if "error" in results:
            return f"  Visualization error for {cell_type}: {results['error']}"
        
        function_name = results.get("function_name", "unknown_display")
        # has_plot = results.get("has_plot", False)
        # plot_size = results.get("plot_size", 0)
        kind = results.get("plot_kind", "unknown")
        has_plot = results.get("has_plot", False)
        plot_size = results.get("plot_size", 0)

        # Create user-friendly description
        viz_descriptions = {
            "display_enrichment_visualization": "enrichment analysis visualization",
            "display_dotplot": "gene expression dotplot",
            "display_processed_umap": "annotated UMAP with cell types"
        }
        
        viz_type = viz_descriptions.get(function_name, function_name.replace("display_", "").replace("_", " "))
        
        lines = [f"VISUALIZATION - {cell_type.upper()}:"]
        
        if has_plot:
            # lines.append(f"  ✅ Generated {viz_type} ({plot_size:,} chars)")
            # lines.append(f"  📊 Interactive plot available for display")
            if kind == "file_ref":
                lines.append(f"  ✅ {viz_type} saved to file (rendered in UI)")
            elif kind == "bundle":
                lines.append(f"  ✅ {viz_type} (multiple plots)")
            else:
                lines.append(f"  ✅ {viz_type} ready ({plot_size:,} chars)")
            lines.append("  📊 Interactive plot available in the UI")

        else:
            lines.append(f"  ⚠️ {viz_type} attempted but no plot data found")
        
        # Add parameter info if relevant
        parameters = results.get("parameters", {})
        if parameters.get("analysis"):
            lines.append(f"  Analysis type: {parameters['analysis']}")
        if parameters.get("plot_type"):
            lines.append(f"  Plot format: {parameters['plot_type']}")
        
        return "\n".join(lines)


class ProcessCellsResultAccessor(ResultAccessorBase):
    """Accessor for process_cells results"""
    

    def can_handle(self, function_name: str) -> bool:
        return function_name == "process_cells"
    
    def get_results(self, execution_step: Dict[str, Any]) -> Dict[str, Any]:
        """Get process_cells results from execution step"""
        try:
            result = execution_step.get("result", "")
            step_data = execution_step.get("step", {})
            parameters = step_data.get("parameters", {})
            cell_type = parameters.get("cell_type", "unknown")
            
            # Extract discovered cell types from result text
            discovered_types = []
            if isinstance(result, str):
                # Look for "✅ Discovered new cell type: X" patterns
                import re
                discoveries = re.findall(r"✅ Discovered new cell type: ([^\n]+)", result)
                discovered_types = discoveries
            
            return {
                "parent_cell_type": cell_type,
                "discovered_types": discovered_types,
                "raw_result": result,
                "source": "execution_result"
            }
            
        except Exception as e:
            logger.info(f"❌ ProcessCellsResultAccessor error: {e}")
            return {"error": str(e), "cell_type": "unknown"}
    
    def format_for_synthesis(self, results: Dict[str, Any], cell_type: str) -> str:
        """Format process_cells results"""
        if "error" in results:
            return f"  Cell processing error for {cell_type}: {results['error']}"
        
        parent_type = results.get("parent_cell_type", cell_type)
        discovered = results.get("discovered_types", [])
        
        lines = [f"CELL TYPE DISCOVERY - {parent_type.upper()}:"]
        
        if discovered:
            lines.append(f"  Discovered subtypes: {', '.join(discovered)}")
            lines.append(f"  Total new types found: {len(discovered)}")
        else:
            lines.append(f"  No new cell subtypes discovered from {parent_type}")
        
        return "\n".join(lines)


class UnifiedResultAccessor:
    """
    Unified accessor that routes analysis results to appropriate handlers
    based purely on execution history function names - NO HARDCODING
    """
    
    def __init__(self):
        # Register all available result accessors
        self.accessors = [
            EnrichmentResultAccessor(),
            DEAResultAccessor(), 
            CellCountResultAccessor(),
            ProcessCellsResultAccessor(),
            SemanticSearchResultAccessor(),
            ValidationResultAccessor(),
            VisualizationResultAccessor()
        ]
    
    def get_analysis_results(self, execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract all analysis results based purely on execution history
        No hardcoding - purely function-name driven
        """
        unified_results = {
            "enrichment_analyses": {},
            "dea_analyses": {},
            "cellcount_analyses": {},
            "process_cell_results": {},
            "semantic_search_results": {},
            "validation_results": {},
            "visualization_results": {},
            "plots": [],                 # NEW: lightweight plot summaries
            "has_plots": False,          # NEW: flag for response generator
            "total_successful_steps": 0,
            "total_failed_steps": 0
        }
        
        logger.info(f"🔍 UNIFIED ACCESSOR: Processing {len(execution_history)} execution steps")
        
        for i, step in enumerate(execution_history):
            try:
                # Only process successful steps
                if not step.get("success", False):
                    unified_results["total_failed_steps"] += 1
                    continue
                
                unified_results["total_successful_steps"] += 1
                
                # Extract function name from step structure
                step_data = step.get("step", {})
                function_name = step_data.get("function_name", "unknown")
                parameters = step_data.get("parameters", {})
                cell_type = parameters.get("cell_type", f"step_{i}")
                
                logger.info(f"🔍 Step {i+1}: {function_name} for {cell_type}")
                
                # DEBUG: Extra logging for semantic search steps to catch cell type issues
                if function_name == "search_enrichment_semantic":
                    logger.info(f"🔍 SEMANTIC DEBUG: Step {i+1} parameters: {parameters}")
                    if "query" in parameters:
                        logger.info(f"🔍 SEMANTIC DEBUG: Query parameter: '{parameters['query']}'")
                # elif function_name.startswith("display_"):
                elif isinstance(function_name, str) and function_name.startswith("display_"):
                    logger.info(f"🎨 VISUALIZATION DEBUG: Step {i+1} function: {function_name}")
                    logger.info(f"🎨 VISUALIZATION DEBUG: Parameters: {parameters}")
                
                # Route to appropriate accessor based on function name
                handled = False
                for accessor in self.accessors:
                    if accessor.can_handle(function_name):
                        logger.info(f"✅ Handling {function_name} with {accessor.__class__.__name__}")
                        
                        results = accessor.get_results(step)
                        
                        # Store results in appropriate category
                        if isinstance(accessor, EnrichmentResultAccessor):
                            unified_results["enrichment_analyses"][cell_type] = results
                        elif isinstance(accessor, DEAResultAccessor):
                            unified_results["dea_analyses"][cell_type] = results
                        elif isinstance(accessor, CellCountResultAccessor):
                            unified_results["cellcount_analyses"][cell_type] = results
                        elif isinstance(accessor, ProcessCellsResultAccessor):
                            unified_results["process_cell_results"][cell_type] = results
                        elif isinstance(accessor, SemanticSearchResultAccessor):
                            unified_results["semantic_search_results"][cell_type] = results
                        elif isinstance(accessor, ValidationResultAccessor):
                            unified_results["validation_results"][cell_type] = results
                        elif isinstance(accessor, VisualizationResultAccessor):
                            # Use function name + cell type for unique visualization keys
                            function_name = results.get("function_name", "unknown_display")
                            viz_key = f"{function_name}_{cell_type}"
                            unified_results["visualization_results"][viz_key] = results

                            # NEW: normalize a tiny summary for the LLM hint
                            unified_results["plots"].append({
                                "key": viz_key,
                                "cell_type": cell_type,
                                "function": function_name,
                                "summary": _normalize_viz_result(results.get("result"))
                            })

                        
                        handled = True
                        break
                
                if not handled:
                    logger.info(f"⚠️ No accessor found for function: {function_name}")
                    
            except Exception as e:
                logger.info(f"❌ Error processing step {i}: {e}")
                unified_results["total_failed_steps"] += 1
        
        logger.info(f"✅ UNIFIED ACCESSOR: Processed {unified_results['total_successful_steps']} successful steps")
        unified_results["has_plots"] = len(unified_results["plots"]) > 0
        return unified_results
    
    def format_for_synthesis(self, unified_results: Dict[str, Any]) -> str:
        """
        Format all analysis results for LLM synthesis
        """
        sections = []
        
        # Add execution summary
        total_success = unified_results.get("total_successful_steps", 0)
        total_failed = unified_results.get("total_failed_steps", 0)
        sections.append(f"EXECUTION SUMMARY:")
        sections.append(f"- Successful analyses: {total_success}")
        sections.append(f"- Failed analyses: {total_failed}")
        sections.append("")

        # NEW: concise hint to the LLM that visualizations exist
        if unified_results.get("has_plots"):
            kinds = sorted({ p["summary"]["kind"] for p in unified_results.get("plots", []) })
            sections.append("[NOTE] Visualizations are available in the UI "
                            f"({', '.join(kinds)}). Do NOT say plots failed.")
            sections.append("")
        
        # Format each analysis type using its specific accessor
        for accessor in self.accessors:
            
            if isinstance(accessor, EnrichmentResultAccessor):
                enrichment_data = unified_results.get("enrichment_analyses", {})
                for cell_type, results in enrichment_data.items():
                    formatted = accessor.format_for_synthesis(results, cell_type)
                    sections.append(formatted)
                    sections.append("")
            
            elif isinstance(accessor, DEAResultAccessor):
                dea_data = unified_results.get("dea_analyses", {})
                for cell_type, results in dea_data.items():
                    formatted = accessor.format_for_synthesis(results, cell_type)
                    sections.append(formatted)
                    sections.append("")
            
            elif isinstance(accessor, CellCountResultAccessor):
                count_data = unified_results.get("cellcount_analyses", {})
                for cell_type, results in count_data.items():
                    formatted = accessor.format_for_synthesis(results, cell_type)
                    sections.append(formatted)
                    sections.append("")
            
            elif isinstance(accessor, ProcessCellsResultAccessor):
                process_data = unified_results.get("process_cell_results", {})
                for cell_type, results in process_data.items():
                    formatted = accessor.format_for_synthesis(results, cell_type)
                    sections.append(formatted)
                    sections.append("")
            
            elif isinstance(accessor, SemanticSearchResultAccessor):
                search_data = unified_results.get("semantic_search_results", {})
                for cell_type, results in search_data.items():
                    formatted = accessor.format_for_synthesis(results, cell_type)
                    sections.append(formatted)
                    sections.append("")
            
            elif isinstance(accessor, ValidationResultAccessor):
                validation_data = unified_results.get("validation_results", {})
                for cell_type, results in validation_data.items():
                    formatted = accessor.format_for_synthesis(results, cell_type)
                    sections.append(formatted)
                    sections.append("")
            
            elif isinstance(accessor, VisualizationResultAccessor):
                viz_data = unified_results.get("visualization_results", {})
                for viz_key, results in viz_data.items():
                    # Extract cell type from the key (format: function_name_cell_type)
                    cell_type = results.get("cell_type", "unknown")
                    formatted = accessor.format_for_synthesis(results, cell_type)
                    sections.append(formatted)
                    sections.append("")
            
        
        result = "\n".join(sections).strip()
        logger.info(f"🎯 UNIFIED FORMATTER: Generated {len(result)} characters for synthesis")
        return result


# Usage example for integration with response generation
def get_unified_results_for_synthesis(execution_history: List[Dict[str, Any]]) -> str:
    """
    Main entry point for getting formatted analysis results
    """
    accessor = UnifiedResultAccessor()
    unified_results = accessor.get_analysis_results(execution_history)
    formatted_results = accessor.format_for_synthesis(unified_results)
    return formatted_results