"""
Single-cell RNA-seq analysis chatbot package.

This package provides a modular multi-agent chatbot system for 
single-cell RNA-seq data analysis and visualization.
"""

# Import main chatbot class for backward compatibility
from .chatbot import ChatBot

# Import key classes that users might need
from .chatbot import (
    CellTypeRelation,
    CellTypeLineage,
    ChatState,
    ExecutionStep,
    ExecutionPlan,
    HierarchicalCellTypeManager,
    CellTypeExtractor,
    FunctionHistoryManager,
    SimpleIntelligentCache,
    AnalysisFunctionWrapper,
    WorkflowNodes
)

# Package metadata
__version__ = "2.0.0"
__author__ = "scChat Development Team"
__description__ = "Modular multi-agent chatbot for single-cell RNA-seq analysis"

# Export main interface
__all__ = [
    'ChatBot',
    'CellTypeRelation',
    'CellTypeLineage',
    'ChatState',
    'ExecutionStep',
    'ExecutionPlan',
    'HierarchicalCellTypeManager',
    'CellTypeExtractor',
    'FunctionHistoryManager',
    'SimpleIntelligentCache',
    'AnalysisFunctionWrapper',
    'WorkflowNodes'
]