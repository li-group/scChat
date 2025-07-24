"""
Unified Result Accessor - Execution-plan-driven result retrieval
No hardcoding, purely based on execution history function calls and their success status.
"""

import os
import pandas as pd
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod


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
            
            # Map analysis types to their directory names
            analysis_dirs = {
                "go": ["go_bp", "go_cc", "go_mf"],
                "kegg": ["kegg"],
                "reactome": ["reactome"], 
                "gsea": ["gsea"]
            }
            
            for analysis_type in analyses:
                if analysis_type in analysis_dirs:
                    for subdir in analysis_dirs[analysis_type]:
                        csv_path = f"scchatbot/enrichment/{subdir}/results_summary_{cell_type}.csv"
                        
                        if os.path.exists(csv_path):
                            try:
                                df = pd.read_csv(csv_path)
                                if len(df) > 0:
                                    # Get top terms with their statistics
                                    results["analysis_types"][subdir] = {
                                        "top_terms": df["term_name"].head(10).tolist(),
                                        "p_values": df.get("adj_p_value", df.get("p_value", [])).head(10).tolist(),
                                        "total_significant": len(df),
                                        "source_file": csv_path
                                    }
                            except Exception as e:
                                print(f"⚠️ Error reading {csv_path}: {e}")
            
            return results
            
        except Exception as e:
            print(f"❌ EnrichmentResultAccessor error: {e}")
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
            # These are generated during DEA execution
            potential_conditions = ["cataract", "basal_laminar_drusen", "age_related_macular_degeneration"]
            
            for condition in potential_conditions:
                csv_path = f"scchatbot/dea/{cell_type}_markers_{condition}.csv"
                
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        if len(df) > 0:
                            # Separate upregulated and downregulated genes
                            df_up = df[df.get("log_fc", df.get("logFC", 0)) > 0].nlargest(10, "log_fc" if "log_fc" in df.columns else "logFC")
                            df_down = df[df.get("log_fc", df.get("logFC", 0)) < 0].nsmallest(10, "log_fc" if "log_fc" in df.columns else "logFC")
                            
                            results["conditions"][condition] = {
                                "upregulated_genes": df_up.to_dict('records'),
                                "downregulated_genes": df_down.to_dict('records'),
                                "total_significant": len(df[df.get("p_adj", df.get("FDR", 1)) < 0.05]),
                                "source_file": csv_path
                            }
                    except Exception as e:
                        print(f"⚠️ Error reading DEA file {csv_path}: {e}")
            
            return results
            
        except Exception as e:
            print(f"❌ DEAResultAccessor error: {e}")
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
            condition_name = condition.replace("_", " ").title()
            lines.append(f"  {condition_name}:")
            
            # Format upregulated genes
            up_genes = data.get("upregulated_genes", [])[:3]
            if up_genes:
                gene_strs = []
                for gene in up_genes:
                    gene_name = gene.get("gene", gene.get("Gene", "Unknown"))
                    log_fc = gene.get("log_fc", gene.get("logFC", 0))
                    gene_strs.append(f"{gene_name} (FC={log_fc:.2f})")
                lines.append(f"    Upregulated: {', '.join(gene_strs)}")
            
            # Format downregulated genes  
            down_genes = data.get("downregulated_genes", [])[:3]
            if down_genes:
                gene_strs = []
                for gene in down_genes:
                    gene_name = gene.get("gene", gene.get("Gene", "Unknown"))
                    log_fc = gene.get("log_fc", gene.get("logFC", 0))
                    gene_strs.append(f"{gene_name} (FC={log_fc:.2f})")
                lines.append(f"    Downregulated: {', '.join(gene_strs)}")
            
            total = data.get("total_significant", 0)
            lines.append(f"    Total significant genes: {total}")
        
        return "\n".join(lines)


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
            print(f"❌ CellCountResultAccessor error: {e}")
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
            print(f"❌ ProcessCellsResultAccessor error: {e}")
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
            ProcessCellsResultAccessor()
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
            "total_successful_steps": 0,
            "total_failed_steps": 0
        }
        
        print(f"🔍 UNIFIED ACCESSOR: Processing {len(execution_history)} execution steps")
        
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
                
                print(f"🔍 Step {i+1}: {function_name} for {cell_type}")
                
                # Route to appropriate accessor based on function name
                handled = False
                for accessor in self.accessors:
                    if accessor.can_handle(function_name):
                        print(f"✅ Handling {function_name} with {accessor.__class__.__name__}")
                        
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
                        
                        handled = True
                        break
                
                if not handled:
                    print(f"⚠️ No accessor found for function: {function_name}")
                    
            except Exception as e:
                print(f"❌ Error processing step {i}: {e}")
                unified_results["total_failed_steps"] += 1
        
        print(f"✅ UNIFIED ACCESSOR: Processed {unified_results['total_successful_steps']} successful steps")
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
        
        result = "\n".join(sections).strip()
        print(f"🎯 UNIFIED FORMATTER: Generated {len(result)} characters for synthesis")
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