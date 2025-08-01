"""
Single-cell RNA-seq analysis chatbot package.

This package provides a modular multi-agent chatbot system for 
single-cell RNA-seq data analysis and visualization.
"""

# Package metadata
__version__ = "2.0.0"
__author__ = "scChat Development Team"
__description__ = "Modular multi-agent chatbot for single-cell RNA-seq analysis"

def _get_chatbot():
    """Lazy import for ChatBot to avoid early initialization."""
    from .chatbot import ChatBot
    return ChatBot

def _get_models():
    """Lazy import for models to avoid early initialization."""
    from .chatbot import (
        CellTypeRelation,
        CellTypeLineage,
        ChatState,
        ExecutionStep,
        HierarchicalCellTypeManager,
        CellTypeExtractor,
        FunctionHistoryManager,
        AnalysisFunctionWrapper,
        WorkflowNodes
    )
    return {
        'CellTypeRelation': CellTypeRelation,
        'CellTypeLineage': CellTypeLineage,
        'ChatState': ChatState,
        'ExecutionStep': ExecutionStep,
        'HierarchicalCellTypeManager': HierarchicalCellTypeManager,
        'CellTypeExtractor': CellTypeExtractor,
        'FunctionHistoryManager': FunctionHistoryManager,
        'AnalysisFunctionWrapper': AnalysisFunctionWrapper,
        'WorkflowNodes': WorkflowNodes
    }

# Provide lazy access
def __getattr__(name):
    if name == 'ChatBot':
        return _get_chatbot()
    elif name in ['CellTypeRelation', 'CellTypeLineage', 'ChatState', 'ExecutionStep', 
                  'HierarchicalCellTypeManager', 'CellTypeExtractor',
                  'FunctionHistoryManager', 'AnalysisFunctionWrapper', 'WorkflowNodes']:
        return _get_models()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Export main interface (for IDE support)
__all__ = [
    'ChatBot',
    'CellTypeRelation',
    'CellTypeLineage',
    'ChatState',
    'ExecutionStep',
    'HierarchicalCellTypeManager',
    'CellTypeExtractor',
    'FunctionHistoryManager',
    'AnalysisFunctionWrapper',
    'WorkflowNodes'
]