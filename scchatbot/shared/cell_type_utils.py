"""
Shared cell type utilities.

This module contains cell type related functions used by both
workflow and jury systems, eliminating code duplication.
"""

import json
import openai
from typing import List, Dict, Any


def extract_cell_types_from_question(question: str, hierarchy_manager=None) -> List[str]:
    """
    Extract cell types mentioned in the user's question using LLM + Neo4j validation.
    
    Strategy:
    1. Get all valid cell types from Neo4j database
    2. Ask LLM to identify which ones are mentioned in the question
    3. Handle natural language variations and synonyms
    
    Examples:
    - "How is 'Regulatory T cells' distinguished from 'Conventional memory CD4 T cells'?"
    - LLM maps to actual Neo4j cell types like "T regulatory cell", "CD4-positive T cell"
    
    Args:
        question: User's question string
        hierarchy_manager: HierarchicalCellTypeManager instance
        
    Returns:
        List of validated cell type names from the database
    """
    if not question or not hierarchy_manager:
        return []
    
    # Get all valid cell types from Neo4j database
    valid_cell_types = getattr(hierarchy_manager, 'valid_cell_types', [])
    if not valid_cell_types:
        print("⚠️ No valid cell types available from Neo4j database")
        return []
    
    print(f"🔍 Analyzing question against {len(valid_cell_types)} valid cell types from Neo4j...")
    
    # Use LLM to identify which valid cell types are mentioned in the question
    llm_prompt = f"""
    You are a cell biology expert. Analyze this user question and identify which specific cell types are mentioned.
    
    USER QUESTION: "{question}"
    
    VALID CELL TYPES FROM DATABASE:
    {', '.join(valid_cell_types[:50])}{'...' if len(valid_cell_types) > 50 else ''}
    
    Your task:
    1. Identify any cell types mentioned in the question (handle synonyms, abbreviations, variations)
    2. Match them to the exact names from the valid cell types list
    3. Only return cell types that exist in the valid list
    
    Examples of matching:
    - "Regulatory T cells" → "T regulatory cell" (if that's the database name)
    - "Tregs" → "T regulatory cell" 
    - "NK cells" → "Natural killer cell"
    - "CD4+ T cells" → "CD4-positive T cell"
    
    Respond in JSON format:
    {{
        "identified_cell_types": ["exact database name 1", "exact database name 2"],
        "reasoning": "Brief explanation of matches found"
    }}
    
    If no valid cell types are mentioned, return empty list.
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": llm_prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        identified_types = result.get("identified_cell_types", [])
        reasoning = result.get("reasoning", "")
        
        # Validate that all identified types are actually in the valid list
        validated_types = []
        for cell_type in identified_types:
            if cell_type in valid_cell_types:
                validated_types.append(cell_type)
                print(f"✅ LLM identified and validated: '{cell_type}'")
            else:
                print(f"⚠️ LLM identified '{cell_type}' but not in valid list - skipping")
        
        if validated_types:
            print(f"🎯 LLM reasoning: {reasoning}")
            return validated_types
        else:
            print(f"❌ No valid cell types identified in question")
            return []
            
    except Exception as e:
        print(f"❌ Error in LLM cell type extraction: {e}")
        return []


def needs_cell_discovery(needed_cell_types: List[str], available_cell_types: List[str]) -> bool:
    """
    Check if we need to discover cell types using process_cells.
    
    Returns True if any needed cell types are not in available cell types.
    
    Args:
        needed_cell_types: Cell types required for analysis
        available_cell_types: Currently available cell types
        
    Returns:
        True if discovery is needed, False otherwise
    """
    for needed_type in needed_cell_types:
        if needed_type not in available_cell_types:
            print(f"🧬 Cell discovery needed: '{needed_type}' not in available types {available_cell_types}")
            return True
    
    print(f"🔍 All needed cell types are already available")
    return False


def create_cell_discovery_steps(needed_cell_types: List[str], available_cell_types: List[str], 
                               query_type: str, hierarchy_manager=None) -> List[Dict[str, Any]]:
    """
    Create process_cells steps to discover needed cell types using Neo4j hierarchy.
    
    Strategy:
    1. For each needed cell type, find path from available types using hierarchy_manager
    2. Add process_cells steps for the path
    3. Add analysis steps for the target query type
    
    Args:
        needed_cell_types: Cell types that need to be discovered
        available_cell_types: Currently available cell types
        query_type: Type of query being performed ("comparison", "analysis", etc.)
        hierarchy_manager: HierarchicalCellTypeManager instance
        
    Returns:
        List of step dictionaries for cell discovery
    """
    if not hierarchy_manager:
        print("⚠️ No hierarchy manager available for cell discovery")
        return []
    
    discovery_steps = []
    
    for needed_type in needed_cell_types:
        if needed_type in available_cell_types:
            print(f"✅ '{needed_type}' already available, no discovery needed")
            continue
        
        # Find processing path using hierarchy manager
        processing_path = None
        best_parent = None
        
        for available_type in available_cell_types:
            path_result = hierarchy_manager.find_parent_path(needed_type, [available_type])
            if path_result:
                best_parent, processing_path = path_result
                print(f"🔄 Found path from '{best_parent}' to '{needed_type}': {' → '.join(processing_path)}")
                break
        
        if processing_path and len(processing_path) > 1:
            # Add process_cells steps for the path (skip the last element as it's the target)
            for i in range(len(processing_path) - 1):
                current_type = processing_path[i]
                target_type = processing_path[i + 1]
                
                # Only add step if we haven't already added it for this current_type
                existing_step = None
                for step in discovery_steps:
                    if (step.get("function_name") == "process_cells" and 
                        step.get("parameters", {}).get("cell_type") == current_type):
                        existing_step = step
                        break
                
                if not existing_step:
                    discovery_steps.append({
                        "step_type": "analysis",
                        "function_name": "process_cells",
                        "parameters": {"cell_type": current_type},
                        "description": f"Process {current_type} to discover {target_type}",
                        "expected_outcome": f"Discover {target_type} cell type",
                        "target_cell_type": current_type
                    })
                    print(f"🧬 Added process_cells({current_type}) to discover {target_type}")
                else:
                    print(f"🔄 process_cells({current_type}) already exists, skipping")
        else:
            print(f"⚠️ No processing path found for '{needed_type}' from available types")
    
    if discovery_steps:
        print(f"🧬 Created discovery plan: {len(discovery_steps)} steps")
        for i, step in enumerate(discovery_steps):
            func_name = step.get('function_name', 'unknown')
            cell_type = step.get('parameters', {}).get('cell_type', 'unknown')
            print(f"🧬   Step {i+1}: {func_name}({cell_type})")
    else:
        print(f"🧬 No discovery steps needed - all cell types available")
    
    return discovery_steps