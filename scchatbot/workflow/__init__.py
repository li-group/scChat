"""
Workflow module for the multi-agent chatbot.

This module contains the unified WorkflowNodes class assembled from modular components
for better maintainability and organization.
"""

from .core_nodes import CoreNodes


class WorkflowNodes(CoreNodes):
    """
    Unified WorkflowNodes class maintaining original interface.
    
    This class provides all workflow functionality through CoreNodes:
    - Individual workflow node orchestrator (input, planning, execution, evaluation, response)
    - Plan evaluation and processing (used by PlannerNode)
    - Response generation methods
    
    The class maintains the same API as the original WorkflowNodes for backward compatibility.
    """
    pass


__all__ = ['WorkflowNodes']