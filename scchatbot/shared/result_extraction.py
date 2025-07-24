"""
Shared result extraction utilities for response generation.

LEGACY FUNCTIONS REMOVED - Now using UnifiedResultAccessor
See scchatbot/workflow/unified_result_accessor.py for the new implementation

This file now only contains functions still needed for semantic search LLM filtering.
"""

from typing import Dict, Any, List


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