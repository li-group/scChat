"""
Workflow module for the multi-agent chatbot.

This module contains the unified WorkflowNodes class assembled from modular components
for better maintainability and organization.
"""

from .core_nodes import CoreNodes
from .execution import ExecutionMixin
from .evaluation import EvaluationMixin
from .response import ResponseMixin
from .utils import UtilsMixin


class WorkflowNodes(CoreNodes, ExecutionMixin, EvaluationMixin, ResponseMixin, UtilsMixin):
    """
    Unified WorkflowNodes class maintaining original interface.
    
    This class combines all workflow functionality through multiple inheritance:
    - CoreNodes: Basic workflow node implementations (input_processor, planner, executor)
    - ExecutionMixin: Execution logic and step management
    - EvaluationMixin: Plan evaluation and processing
    - ResponseMixin: Response generation methods
    - UtilsMixin: Workflow-specific utilities
    
    The class maintains the same API as the original WorkflowNodes for backward compatibility.
    """
    pass


__all__ = ['WorkflowNodes']