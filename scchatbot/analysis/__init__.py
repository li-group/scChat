"""
Analysis package for scChat.

This package contains analysis-related functionality including:
- Enrichment analysis (GO, KEGG, Reactome, GSEA)
- Differential expression analysis (DEA)
- Statistical utilities for analysis
"""

# Import main analysis functions for easy access
from .enrichment_analysis import (
    dea,
    perform_enrichment_analyses,
    get_significant_gene,
    reactome_enrichment,
    go_enrichment,
    kegg_enrichment,
    gsea_enrichment_analysis
)

from .analysis_wrapper import (
    dea_split_by_condition_hierarchical
)

__all__ = [
    # Enrichment analysis functions
    'dea',
    'perform_enrichment_analyses',
    'get_significant_gene',
    'reactome_enrichment',
    'go_enrichment', 
    'kegg_enrichment',
    'gsea_enrichment_analysis',
    
    # DEA functions
    'dea_split_by_condition_hierarchical'
]