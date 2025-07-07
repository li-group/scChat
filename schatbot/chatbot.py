"""
Main chatbot interface using modular architecture.

This module provides the main ChatBot class that inherits from the
multi-agent base and maintains API compatibility with the original
monolithic implementation.
"""

from .multi_agent_base import MultiAgentChatBot


class ChatBot(MultiAgentChatBot):
    """
    Main ChatBot class providing the user interface.
    
    This class inherits from MultiAgentChatBot and provides the same
    API as the original monolithic implementation while using the
    new modular architecture under the hood.
    """
    
    def __del__(self):
        """Cleanup resources when the ChatBot is destroyed"""
        try:
            self.cleanup()
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")


# Re-export key classes for backward compatibility
from .cell_type_models import (
    CellTypeRelation,
    CellTypeLineage,
    ChatState,
    ExecutionStep,
    ExecutionPlan,
    CriticEvaluation
)

from .cell_type_hierarchy import (
    HierarchicalCellTypeManager,
    CellTypeExtractor
)

from .function_history import FunctionHistoryManager
from .cache_manager import SimpleIntelligentCache
from .analysis_wrapper import AnalysisFunctionWrapper
from .critic_system import CriticLoopManager, CriticAgent
from .workflow_nodes import WorkflowNodes


# For backward compatibility, export the main class
__all__ = [
    'ChatBot',
    'CellTypeRelation',
    'CellTypeLineage', 
    'ChatState',
    'ExecutionStep',
    'ExecutionPlan',
    'CriticEvaluation',
    'HierarchicalCellTypeManager',
    'CellTypeExtractor',
    'FunctionHistoryManager',
    'SimpleIntelligentCache',
    'AnalysisFunctionWrapper',
    'CriticLoopManager',
    'CriticAgent',
    'WorkflowNodes'
]