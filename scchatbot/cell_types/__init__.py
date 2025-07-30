"""
Cell type management module for scChat.

This module provides unified cell type management functionality:
- standardization: Name standardization and normalization
- hierarchy_manager: Neo4j-based hierarchical management
- annotation_pipeline: Cell annotation workflows (to be added)  
- validation: Type validation and discovery (to be added)

The functions were moved from scattered locations to create a centralized,
well-organized cell type management system.
"""

# Import commonly used standardization functions for easy access
from .standardization import (
    unified_cell_type_handler,
    standardize_cell_type,
    get_possible_cell_types,
    get_subtypes
)

# Import hierarchy management classes
from .hierarchy_manager import (
    HierarchicalCellTypeManager,
    CellTypeExtractor
)

# Import annotation pipeline functions
from .annotation_pipeline import (
    initial_cell_annotation,
    process_cells,
    handle_process_cells_result,
    preprocess_data,
    perform_clustering,
    rank_genes,
    create_marker_anndata,
    rank_ordering,
    label_clusters
)

# Import validation utilities
from .validation import (
    extract_cell_types_from_question,
    needs_cell_discovery,
    create_cell_discovery_steps
)

__all__ = [
    # Standardization functions
    'unified_cell_type_handler',
    'standardize_cell_type', 
    'get_possible_cell_types',
    'get_subtypes',
    
    # Hierarchy management classes
    'HierarchicalCellTypeManager',
    'CellTypeExtractor',
    
    # Annotation pipeline functions
    'initial_cell_annotation',
    'process_cells',
    'handle_process_cells_result',
    'preprocess_data',
    'perform_clustering', 
    'rank_genes',
    'create_marker_anndata',
    'rank_ordering',
    'label_clusters',
    
    # Validation utilities
    'extract_cell_types_from_question',
    'needs_cell_discovery',
    'create_cell_discovery_steps'
]