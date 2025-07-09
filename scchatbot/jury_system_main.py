"""
Main JurySystem facade for backward compatibility.

This module provides the original JurySystem API while delegating to the
new modular jury system implementation.
"""

from .jury import JurySystem as ModularJurySystem


class JurySystem(ModularJurySystem):
    """
    Facade for the modular jury system.
    
    This class maintains the original API for backward compatibility
    while delegating all functionality to the new modular implementation.
    """
    
    def __init__(self, simple_cache=None, hierarchy_manager=None, history_manager=None, 
                 function_descriptions=None, existing_critic_agent=None):
        """
        Initialize the jury system facade.
        
        Args:
            simple_cache: Cache manager for analysis results
            hierarchy_manager: Cell type hierarchy manager
            history_manager: Function execution history manager
            function_descriptions: Available function descriptions
            existing_critic_agent: DEPRECATED - for backward compatibility only
        """
        # Delegate initialization to the modular implementation
        super().__init__(
            simple_cache=simple_cache,
            hierarchy_manager=hierarchy_manager,
            history_manager=history_manager,
            function_descriptions=function_descriptions,
            existing_critic_agent=existing_critic_agent
        )


# Backward compatibility alias
JurySystemMain = JurySystem