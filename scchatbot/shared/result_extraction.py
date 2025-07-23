"""
Shared result extraction utilities for response generation.

This module provides functions to extract key findings from analysis results,
optimizing for token usage while preserving essential information.
"""

from typing import Dict, Any, List


def extract_key_findings_from_execution(execution_history: List[Dict]) -> Dict[str, Any]:
    """
    Extract key findings from execution history for response generation.
    
    Args:
        execution_history: List of execution steps with results
        
    Returns:
        Dict containing organized key findings from all successful analyses
    """
    try:
        # Process execution history
        findings = {
            "successful_analyses": {},
            "failed_analyses": {},
            "total_steps": len(execution_history),
            "successful_steps": 0,
            "failed_steps": 0
        }
    except Exception as e:
        raise
    
    
    
    for i, step in enumerate(execution_history):
        # FIX: execution history stores function data in 'step' sub-dict
        try:
            # Ensure step is a dictionary - skip strings or other types
            if not isinstance(step, dict):
                continue
                
            # Try to get step data from nested structure first
            step_data = step.get("step", {})
            if isinstance(step_data, dict) and "function_name" in step_data:
                # New nested structure
                function_name = step_data.get("function_name", "")
                parameters = step_data.get("parameters", {})
            elif "function_name" in step:
                # Flat structure (fallback)
                function_name = step.get("function_name", "")
                parameters = step.get("parameters", {})
                step_data = step  # Use step directly
            else:
                # No recognizable function structure
                continue
                
            if not isinstance(parameters, dict):
                parameters = {}
                
            cell_type = parameters.get("cell_type", "unknown")
        except Exception as e:
            continue
        
        # Double-check step is still a dictionary (extra safety)
        if not isinstance(step, dict):
            continue
            
        if step.get("success", False):
            findings["successful_steps"] += 1
            result = step.get("result")
            result_type = step.get("result_type", "text")  # Default to legacy
            
            # Extract key findings based on function type
            if function_name == "perform_enrichment_analyses":
                # Always use structured approach - database-first system
                findings["successful_analyses"][f"enrichment_{cell_type}"] = _extract_enrichment_structured(result)
            
            elif function_name == "search_enrichment_semantic":
                if result_type == "structured":
                    # Store semantic search results for response generation
                    if isinstance(result, dict):
                        # Use a unique key that includes step index to avoid overwriting
                        analysis_key = f"semantic_search_{cell_type}_{i}"  # Include step index
                        
                        # But also check if we already have good results - don't overwrite with bad ones
                        existing_key = f"semantic_search_{cell_type}"
                        if existing_key in findings["successful_analyses"] and not findings["successful_analyses"][existing_key].get("error"):
                            continue
                            
                        findings["successful_analyses"][existing_key] = {
                            "search_results": result.get("search_results", {}),
                            "query": result.get("query", "unknown query"),
                            "cell_type": result.get("cell_type", cell_type),
                            "total_matches": result.get("total_matches", 0),
                            "function": "search_enrichment_semantic"
                        }
                    else:
                        # Don't overwrite good results with error
                        existing_key = f"semantic_search_{cell_type}"
                        if existing_key in findings["successful_analyses"] and not findings["successful_analyses"][existing_key].get("error"):
                            continue
                            
                        findings["successful_analyses"][existing_key] = {
                            "error": f"Expected dict but got {type(result)}",
                            "function": "search_enrichment_semantic",
                            "raw_string_result": str(result)[:1000]  # Include the string data
                        }
            
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
            # Get error message safely
            error_msg = step.get("error", "Unknown error") if isinstance(step, dict) else "Unknown error"
            findings["failed_analyses"][f"{function_name}_{cell_type}"] = {
                "function": function_name,
                "parameters": parameters,
                "error": error_msg
            }
    
    return findings



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
    
    elif "search_results" in analysis_data and "function" in analysis_data:
        # Semantic search results formatting
        if analysis_data["function"] == "search_enrichment_semantic":
            search_data = analysis_data["search_results"]
            query = analysis_data.get("query", "unknown query")
            cell_type = analysis_data.get("cell_type", "unknown")
            total_matches = search_data.get("total_matches", 0)
            
            if total_matches == 0:
                return f"  No enrichment terms found for '{query}' in {cell_type}"
            
            # Format the actual search results
            formatted_lines = [f"  Found {total_matches} enrichment terms for '{query}' in {cell_type}:"]
            
            results = search_data.get("results", [])  # Use ALL found results for LLM filtering
            
            # Apply LLM-based relevance filtering and summarization
            llm_summary = _filter_and_summarize_semantic_results(results, query, cell_type)
            
            return f"  {llm_summary}"
    
    else:
        # Fallback formatting - check for HTML content
        data_str = str(analysis_data)
        if '<div' in data_str or '<html' in data_str or '<script' in data_str:
            return f"  Visualization generated successfully"
        return f"  {data_str[:100]}..."


def _filter_and_summarize_semantic_results(results: list, query: str, cell_type: str) -> str:
    """
    Use LLM to filter semantic search results for relevance and create a focused summary.
    
    Args:
        results: List of semantic search results
        query: Original search query (e.g., "Cell cycle regulation")
        cell_type: Cell type being analyzed (e.g., "Endothelial cell")
        
    Returns:
        LLM-generated summary of relevant findings
    """
    if not results:
        return f"No enrichment terms found for '{query}' in {cell_type}"
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        
        # Format all results for LLM evaluation
        results_text = []
        for i, result in enumerate(results, 1):
            term_name = result.get("term_name", "Unknown term")
            analysis_type = result.get("analysis_type", "unknown").upper()
            p_value = result.get("adj_p_value", result.get("p_value", "N/A"))
            similarity = result.get("similarity_score", 0)
            description = result.get("description", "")
            
            results_text.append(f"{i}. [{analysis_type}] {term_name}")
            results_text.append(f"   p-value: {p_value}, similarity: {similarity:.3f}")
            if description:
                results_text.append(f"   Description: {description}")
            results_text.append("")
        
        results_formatted = "\n".join(results_text)
        
        # Create LLM prompt for relevance filtering and summarization
        prompt = f"""You are a bioinformatics expert analyzing enrichment results. 

The user asked about: "{query}" in {cell_type}

Here are the semantic search results from the enrichment analysis:

{results_formatted}

Your task:
1. Identify which of these terms are ACTUALLY related to "{query}"
2. Focus on biological relevance, not just keyword similarity
3. Create a concise summary answering whether "{query}" is enriched in {cell_type}

Provide a focused response in this format:
- If relevant terms found: "Based on the enrichment analysis, [summary of findings with specific term names and p-values]"
- If no truly relevant terms: "The enrichment analysis did not find strong evidence for {query} in {cell_type}. The closest matches were [brief mention] but these are not directly related."

Be specific about p-values and term names for truly relevant findings."""

        # Call LLM for filtering and summarization
        model = ChatOpenAI(model="gpt-4o", temperature=0.3, max_tokens=300)
        
        messages = [
            SystemMessage(content="You are a bioinformatics expert who understands pathway enrichment analysis."),
            HumanMessage(content=prompt)
        ]
        
        response = model.invoke(messages)
        return response.content.strip()
        
    except Exception as e:
        # Fallback to original formatting
        formatted_lines = [f"Found {len(results)} enrichment terms for '{query}' in {cell_type}:"]
        for i, result in enumerate(results[:5], 1):  # Show top 5 as fallback
            term_name = result.get("term_name", "Unknown term")
            analysis_type = result.get("analysis_type", "unknown").upper()
            p_value = result.get("adj_p_value", result.get("p_value", "N/A"))
            formatted_lines.append(f"    {i}. [{analysis_type}] {term_name} (p={p_value})")
        
        return "\n".join(formatted_lines)


def _extract_enrichment_structured(structured_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract findings directly from structured enrichment result
    
    Args:
        structured_result: Full structured dictionary from perform_enrichment_analyses
        
    Returns:
        Dict with top findings from each analysis type
    """
    key_findings = {}
    
    # Direct access to structured data - no parsing needed!
    analysis_types = ["reactome", "go", "kegg", "gsea"]
    
    for analysis_type in analysis_types:
        if analysis_type in structured_result:
            analysis_data = structured_result[analysis_type]
            
            if isinstance(analysis_data, dict):
                key_findings[analysis_type] = {
                    "top_terms": analysis_data.get("top_terms", [])[:10],
                    "p_values": analysis_data.get("top_pvalues", [])[:10],
                    "total_significant": analysis_data.get("total_significant", 0)
                }
    
    # Note: Enrichment vector database indexing is now handled directly 
    # in the perform_enrichment_analyses function for immediate availability
    
    return key_findings