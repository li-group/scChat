"""
Jury system module for parallel evaluation.

This module contains the unified JurySystem class that combines all jury functionality
through multiple inheritance of specialized mixins.
"""

from .core_system import CoreSystemMixin
from .evaluation import EvaluationCoordinatorMixin
from .verdict_processor import VerdictProcessorMixin
from .revision import PlanRevisionMixin
from .utils import JuryUtilsMixin


class JurySystem(
    CoreSystemMixin,
    EvaluationCoordinatorMixin,
    VerdictProcessorMixin,
    PlanRevisionMixin,
    JuryUtilsMixin
):
    """
    Unified jury system that combines all jury functionality.
    
    This class uses multiple inheritance to combine all jury mixins:
    - CoreSystemMixin: Initialization and main evaluation orchestration
    - EvaluationCoordinatorMixin: Judge evaluation coordination
    - VerdictProcessorMixin: Verdict processing and routing decisions
    - PlanRevisionMixin: Targeted plan revisions based on judge feedback
    - JuryUtilsMixin: Utility functions for jury operations
    
    The jury system replaces the single critic agent with a panel of specialized judges
    that evaluate different aspects of the chatbot's performance in parallel.
    """
    
    def __init__(self, simple_cache=None, hierarchy_manager=None, history_manager=None, 
                 function_descriptions=None, existing_critic_agent=None):
        """
        Initialize the unified jury system.
        
        Args:
            simple_cache: Cache manager for analysis results
            hierarchy_manager: Cell type hierarchy manager
            history_manager: Function execution history manager
            function_descriptions: Available function descriptions
            existing_critic_agent: DEPRECATED - for backward compatibility only
        """
        # Initialize all mixins by calling the CoreSystemMixin's __init__
        # which handles all the initialization logic
        super().__init__(
            simple_cache=simple_cache,
            hierarchy_manager=hierarchy_manager,
            history_manager=history_manager,
            function_descriptions=function_descriptions,
            existing_critic_agent=existing_critic_agent
        )


__all__ = ['JurySystem']