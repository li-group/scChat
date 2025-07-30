"""
Workflow module for the multi-agent chatbot.

This module contains the unified WorkflowNodes class assembled from modular components
for better maintainability and organization.
"""

from .core_nodes import CoreNodes
from .utils import UtilsMixin


class WorkflowNodes(CoreNodes, UtilsMixin):
    """
    Unified WorkflowNodes class maintaining original interface.
    
    This class combines all workflow functionality through multiple inheritance:
    - CoreNodes: Individual workflow node orchestrator (input, planning, execution, evaluation, response)
    - EvaluationMixin: Plan evaluation and processing (used by PlannerNode)
    - ResponseMixin: Response generation methods
    - UtilsMixin: Workflow-specific utilities
    
    The class maintains the same API as the original WorkflowNodes for backward compatibility.
    """
    pass


__all__ = ['WorkflowNodes']