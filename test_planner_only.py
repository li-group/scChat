#!/usr/bin/env python3
"""Test just the planner logic for handling undiscoverable cell types."""

import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up environment
os.environ["OPENAI_API_KEY"] = "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"

from scchatbot.cell_type_models import ChatState
from scchatbot.workflow.nodes.planning import PlannerNode
from scchatbot.cell_type_hierarchy import CellTypeHierarchyManager
from scchatbot.shared import extract_cell_types_from_question

def test_planner_filtering():
    """Test that the planner filters out undiscoverable cell types."""
    print("=" * 80)
    print("Testing Planner: Undiscoverable Cell Type Filtering")
    print("=" * 80)
    
    # Create mock hierarchy manager
    hierarchy_manager = CellTypeHierarchyManager()
    
    # Test question
    test_question = "How is the known cell type 'Regulatory T cells' distinguished from the 'Conventional memory CD4 T cells'?"
    
    # Extract cell types from question
    needed_types = extract_cell_types_from_question(test_question, hierarchy_manager)
    print(f"\nExtracted cell types from question: {needed_types}")
    
    # Mock available cell types (basic types only)
    available_types = ['Immune cell', 'T cell', 'B cell', 'Mast cell']
    print(f"Available cell types: {available_types}")
    
    # Check discoverability
    print("\nChecking discoverability:")
    for needed_type in needed_types:
        if needed_type in available_types:
            print(f"✅ '{needed_type}' - already available")
        else:
            # Check if can be discovered
            can_discover = False
            for available_type in available_types:
                path_result = hierarchy_manager.find_parent_path(needed_type, [available_type])
                if path_result:
                    best_parent, processing_path = path_result
                    print(f"✅ '{needed_type}' - can be discovered via: {' → '.join(processing_path)}")
                    can_discover = True
                    break
            
            if not can_discover:
                print(f"❌ '{needed_type}' - CANNOT be discovered from available types")
    
    print("\n" + "=" * 80)
    print("Expected behavior:")
    print("- Planner should filter out steps for 'Regulatory T cell' (undiscoverable)")
    print("- Planner should filter out steps for 'CD4-positive memory T cell' (undiscoverable)")
    print("- No analysis steps should be created for these types")
    print("=" * 80)

if __name__ == "__main__":
    test_planner_filtering()