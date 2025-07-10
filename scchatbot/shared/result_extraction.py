"""
Shared result extraction utilities for jury system and response generation.

This module provides functions to extract key findings from analysis results,
optimizing for token usage while preserving essential information.
"""

from typing import Dict, Any, List


def extract_key_findings_from_execution(execution_history: List[Dict]) -> Dict[str, Any]:
    """
    Extract key findings from execution history - used by both jury and response generation.
    
    Args:
        execution_history: List of execution steps with results
        
    Returns:
        Dict containing organized key findings from all successful analyses
    """
    findings = {
        "successful_analyses": {},
        "failed_analyses": {},
        "total_steps": len(execution_history),
        "successful_steps": 0,
        "failed_steps": 0
    }
    
    for step in execution_history:
        function_name = step.get("step", {}).get("function_name", "")
        parameters = step.get("step", {}).get("parameters", {})
        cell_type = parameters.get("cell_type", "unknown")
        
        if step.get("success", False):
            findings["successful_steps"] += 1
            result = step.get("result")
            
            # Extract key findings based on function type
            if function_name == "perform_enrichment_analyses":
                findings["successful_analyses"][f"enrichment_{cell_type}"] = extract_enrichment_key_findings(result)
            
            elif function_name == "dea_split_by_condition":
                findings["successful_analyses"][f"dea_{cell_type}"] = extract_dea_key_findings(result)
            
            elif function_name == "process_cells":
                findings["successful_analyses"][f"process_cells_{cell_type}"] = extract_process_cells_findings(result)
            
            elif function_name == "compare_cell_counts":
                findings["successful_analyses"][f"comparison_{cell_type}"] = extract_comparison_findings(result)
            
            elif function_name.startswith("display_"):
                # Track visualizations that were generated - exclude HTML content
                findings["successful_analyses"][f"visualization_{function_name}"] = {
                    "type": function_name,
                    "parameters": parameters,
                    "description": f"Generated {function_name} for {cell_type}",
                    "html_excluded": True  # Flag to indicate HTML was excluded
                }
        else:
            findings["failed_steps"] += 1
            findings["failed_analyses"][f"{function_name}_{cell_type}"] = {
                "function": function_name,
                "parameters": parameters,
                "error": step.get("error", "Unknown error")
            }
    
    return findings


def extract_enrichment_key_findings(result: Any) -> Dict[str, Any]:
    """
    Extract top enriched pathways/terms from enrichment analysis result.
    
    Args:
        result: Result from perform_enrichment_analyses
        
    Returns:
        Dict with top findings from each analysis type
    """
    if not result or not isinstance(result, dict):
        return {"error": "Invalid enrichment result format"}
    
    key_findings = {}
    
    # Extract from each analysis type
    analysis_types = ["reactome", "go", "kegg", "gsea"]
    
    for analysis_type in analysis_types:
        if analysis_type in result:
            analysis_result = result[analysis_type]
            
            if isinstance(analysis_result, dict):
                # Extract top 5 terms with their significance
                top_terms = analysis_result.get("top_terms", [])[:5]
                top_pvalues = analysis_result.get("top_pvalues", [])[:5]
                
                key_findings[analysis_type] = {
                    "top_terms": top_terms,
                    "p_values": top_pvalues,
                    "total_significant": analysis_result.get("total_significant", 0)
                }
            elif isinstance(analysis_result, str):
                # Sometimes result is a summary string
                key_findings[analysis_type] = {
                    "summary": analysis_result[:200] + "..." if len(analysis_result) > 200 else analysis_result
                }
    
    return key_findings


def extract_dea_key_findings(result: Any) -> Dict[str, Any]:
    """
    Extract top differentially expressed genes from DEA result.
    
    Args:
        result: Result from dea_split_by_condition
        
    Returns:
        Dict with top up/down regulated genes
    """
    if not result:
        return {"error": "No DEA result available"}
    
    # Handle different result formats
    if isinstance(result, str):
        return {"summary": result[:300] + "..." if len(result) > 300 else result}
    
    if isinstance(result, dict):
        key_findings = {}
        
        # Extract condition-specific results
        for condition, condition_result in result.items():
            if isinstance(condition_result, dict):
                key_findings[condition] = {
                    "top_upregulated": condition_result.get("top_upregulated_genes", [])[:10],
                    "top_downregulated": condition_result.get("top_downregulated_genes", [])[:10],
                    "total_significant": condition_result.get("total_significant_genes", 0),
                    "upregulated_count": condition_result.get("upregulated_count", 0),
                    "downregulated_count": condition_result.get("downregulated_count", 0)
                }
        
        return key_findings
    
    return {"summary": str(result)[:200]}


def extract_process_cells_findings(result: Any) -> Dict[str, Any]:
    """
    Extract discovered cell subtypes from process_cells result.
    
    Args:
        result: Result from process_cells
        
    Returns:
        Dict with discovered subtypes and parent information
    """
    if not result:
        return {"error": "No process_cells result available"}
    
    if isinstance(result, str):
        # Extract information from text result
        findings = {"description": result}
        
        # Try to extract specific subtype information
        if "discovered" in result.lower():
            findings["status"] = "subtypes_discovered"
        elif "no new" in result.lower() or "no subtypes" in result.lower():
            findings["status"] = "no_subtypes_found"
        else:
            findings["status"] = "analysis_completed"
        
        return findings
    
    elif isinstance(result, dict):
        return {
            "subtypes_found": result.get("new_cell_types", []),
            "parent_cell_type": result.get("parent_cell_type", ""),
            "total_subtypes": len(result.get("new_cell_types", [])),
            "description": result.get("description", "")
        }
    
    return {"summary": str(result)[:200]}


def extract_comparison_findings(result: Any) -> Dict[str, Any]:
    """
    Extract cell count comparison results.
    
    Args:
        result: Result from compare_cell_counts
        
    Returns:
        Dict with comparison statistics and insights
    """
    if not result:
        return {"error": "No comparison result available"}
    
    if isinstance(result, str):
        # Extract key statistics from text result
        findings = {"description": result}
        
        # Try to extract specific numbers or trends
        if "significant" in result.lower():
            findings["significance"] = "significant_difference_found"
        elif "no significant" in result.lower():
            findings["significance"] = "no_significant_difference"
        else:
            findings["significance"] = "analysis_completed"
        
        return findings
    
    elif isinstance(result, dict):
        return {
            "cell_counts": result.get("cell_counts", {}),
            "statistics": result.get("statistics", {}),
            "significant_differences": result.get("significant_differences", []),
            "summary": result.get("summary", "")
        }
    
    return {"summary": str(result)[:200]}


def format_findings_for_synthesis(findings: Dict[str, Any]) -> str:
    """
    Format extracted findings into a structured text for LLM synthesis.
    
    Args:
        findings: Dict from extract_key_findings_from_execution
        
    Returns:
        Formatted string ready for LLM prompt
    """
    formatted_sections = []
    
    # Summary statistics
    formatted_sections.append(f"EXECUTION SUMMARY:")
    formatted_sections.append(f"- Total steps: {findings['total_steps']}")
    formatted_sections.append(f"- Successful: {findings['successful_steps']}")
    formatted_sections.append(f"- Failed: {findings['failed_steps']}")
    formatted_sections.append("")
    
    # Successful analyses
    if findings["successful_analyses"]:
        formatted_sections.append("SUCCESSFUL ANALYSES:")
        for analysis_key, analysis_data in findings["successful_analyses"].items():
            formatted_sections.append(f"\n{analysis_key.upper()}:")
            formatted_sections.append(_format_single_analysis(analysis_data))
    
    # Failed analyses
    if findings["failed_analyses"]:
        formatted_sections.append("\nFAILED ANALYSES:")
        for analysis_key, failure_data in findings["failed_analyses"].items():
            formatted_sections.append(f"- {analysis_key}: {failure_data['error']}")
    
    return "\n".join(formatted_sections)


def _format_single_analysis(analysis_data: Dict[str, Any]) -> str:
    """Helper function to format individual analysis results."""
    
    if "error" in analysis_data:
        return f"  ERROR: {analysis_data['error']}"
    
    if "top_terms" in analysis_data:
        # Enrichment analysis formatting
        formatted_lines = []
        for analysis_type, data in analysis_data.items():
            if isinstance(data, dict) and "top_terms" in data:
                top_terms = data["top_terms"][:3]  # Top 3 for synthesis
                formatted_lines.append(f"  {analysis_type.upper()}: {', '.join(top_terms)}")
        return "\n".join(formatted_lines)
    
    elif "top_upregulated" in analysis_data:
        # DEA analysis formatting
        formatted_lines = []
        for condition, data in analysis_data.items():
            if isinstance(data, dict):
                up_genes = data.get("top_upregulated", [])[:5]
                down_genes = data.get("top_downregulated", [])[:5]
                formatted_lines.append(f"  {condition}:")
                formatted_lines.append(f"    Upregulated: {', '.join(up_genes)}")
                formatted_lines.append(f"    Downregulated: {', '.join(down_genes)}")
        return "\n".join(formatted_lines)
    
    elif "type" in analysis_data and analysis_data["type"].startswith("display_"):
        # Visualization results - only include description, exclude HTML
        return f"  {analysis_data.get('description', 'Visualization generated')}"
    
    elif "description" in analysis_data:
        # Process cells or other descriptive results
        # Check if description contains HTML and exclude it
        description = analysis_data['description']
        if '<div' in description or '<html' in description or '<script' in description:
            # Skip HTML content, just indicate visualization was generated
            return f"  Visualization generated successfully"
        return f"  {description[:150]}..."
    
    elif "summary" in analysis_data:
        # Summary format - also check for HTML
        summary = analysis_data['summary']
        if '<div' in summary or '<html' in summary or '<script' in summary:
            return f"  Visualization generated successfully"
        return f"  {summary}"
    
    else:
        # Fallback formatting - check for HTML content
        data_str = str(analysis_data)
        if '<div' in data_str or '<html' in data_str or '<script' in data_str:
            return f"  Visualization generated successfully"
        return f"  {data_str[:100]}..."