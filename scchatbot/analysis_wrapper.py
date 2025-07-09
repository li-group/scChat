"""
Analysis function wrapper with hierarchy awareness.

This module provides a wrapper that makes existing analysis functions
hierarchy-aware, enabling automatic cell type resolution and aggregation.
"""

from typing import Dict, Any

from .enrichment import perform_enrichment_analyses
from .utils import dea_split_by_condition, compare_cell_count


class AnalysisFunctionWrapper:
    """
    Wrapper that makes existing analysis functions hierarchy-aware.
    
    This wrapper intercepts analysis function calls and uses the hierarchy
    manager to resolve cell types, aggregate descendants, or identify
    processing requirements before calling the original analysis functions.
    """
    
    def __init__(self, hierarchy_manager):
        self.hierarchy_manager = hierarchy_manager
    
    def perform_enrichment_analyses_hierarchical(self, cell_type: str, **kwargs):
        """Hierarchy-aware enrichment analysis"""
        try:
            # Get analysis-ready adata
            analysis_adata, metadata = self.hierarchy_manager.get_analysis_ready_adata(cell_type)
            
            print(f"🧬 Performing enrichment analysis on {metadata.get('cell_count', 'unknown')} cells")
            
            # Call original function with resolved adata
            result = perform_enrichment_analyses(analysis_adata, cell_type, **kwargs)
            
            # Add hierarchy metadata to result
            if isinstance(result, dict):
                result["hierarchy_metadata"] = metadata
                result["resolution_info"] = f"Resolved via {metadata['resolution_method']}"
            
            return result
            
        except ValueError as e:
            # Handle case where process_cells is needed
            if "requires process_cells" in str(e):
                return {
                    "status": "needs_processing", 
                    "message": str(e),
                    "required_steps": metadata.get("processing_path", []) if 'metadata' in locals() else []
                }
            raise e
    
    def dea_split_by_condition_hierarchical(self, cell_type: str, **kwargs):
        """Hierarchy-aware DEA"""
        try:
            analysis_adata, metadata = self.hierarchy_manager.get_analysis_ready_adata(cell_type)
            
            print(f"🧬 Performing DEA on {metadata.get('cell_count', 'unknown')} cells")
            
            # Call original function with resolved adata
            result = dea_split_by_condition(analysis_adata, cell_type, **kwargs)
            
            # Add metadata
            if isinstance(result, (list, dict)):
                return {
                    "dea_results": result,
                    "hierarchy_metadata": metadata,
                    "resolution_info": f"Resolved via {metadata['resolution_method']}"
                }
            
            return result
            
        except ValueError as e:
            if "requires process_cells" in str(e):
                return {
                    "status": "needs_processing",
                    "message": str(e),
                    "required_steps": metadata.get("processing_path", []) if 'metadata' in locals() else []
                }
            raise e
    
    def compare_cell_count_hierarchical(self, cell_type: str, **kwargs):
        """Hierarchy-aware cell count comparison"""
        try:
            analysis_adata, metadata = self.hierarchy_manager.get_analysis_ready_adata(cell_type)
            
            # Call original function
            result = compare_cell_count(analysis_adata, cell_type, **kwargs)
            
            # Add metadata
            if isinstance(result, list):
                return {
                    "count_results": result,
                    "hierarchy_metadata": metadata,
                    "resolution_info": f"Resolved via {metadata['resolution_method']}"
                }
            
            return result
            
        except ValueError as e:
            if "requires process_cells" in str(e):
                return {
                    "status": "needs_processing",
                    "message": str(e),
                    "required_steps": metadata.get("processing_path", []) if 'metadata' in locals() else []
                }
            raise e