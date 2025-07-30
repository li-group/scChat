"""
Core workflow nodes orchestrator.

This module now serves as an orchestrator that coordinates the individual
workflow nodes for better modularity and maintainability.
"""

import json
import re
from typing import Dict, Any, List
from datetime import datetime

from ..cell_type_models import ChatState, ExecutionStep
from langchain_core.messages import HumanMessage, AIMessage
from ..shared import extract_cell_types_from_question, needs_cell_discovery

# Import individual node implementations
from .nodes import (
    InputProcessorNode,
    PlannerNode,
    ExecutorNode,
    ValidationNode,
    ResponseGeneratorNode
)
from .evaluation import EvaluationMixin

# Import EnrichmentChecker with error handling
try:
    from ..enrichment_checker import EnrichmentChecker
    ENRICHMENT_CHECKER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ EnrichmentChecker not available: {e}")
    EnrichmentChecker = None
    ENRICHMENT_CHECKER_AVAILABLE = False


class CoreNodes:
    """
    Core workflow orchestrator that coordinates individual node implementations.
    
    This class now acts as a thin orchestration layer over the individual 
    workflow nodes, providing a unified interface while maintaining the
    modularity of separate node implementations.
    """
    
    def __init__(self, initial_annotation_content, initial_cell_types, adata, 
                 history_manager, hierarchy_manager, cell_type_extractor,
                 function_descriptions, function_mapping, visualization_functions):
        """Initialize the orchestrator and all node instances."""
        
        # Store common dependencies
        self.initial_annotation_content = initial_annotation_content
        self.initial_cell_types = initial_cell_types
        self.adata = adata
        self.history_manager = history_manager
        self.hierarchy_manager = hierarchy_manager
        self.cell_type_extractor = cell_type_extractor
        self.function_descriptions = function_descriptions
        self.function_mapping = function_mapping
        self.visualization_functions = visualization_functions
        
        # Initialize EnrichmentChecker with error handling
        self.enrichment_checker = None
        self.enrichment_checker_available = False
        
        if ENRICHMENT_CHECKER_AVAILABLE:
            try:
                self.enrichment_checker = EnrichmentChecker()
                self.enrichment_checker_available = (
                    self.enrichment_checker.connection_status == "connected"
                )
                print(f"✅ EnrichmentChecker initialized: {self.enrichment_checker.connection_status}")
            except Exception as e:
                print(f"⚠️ EnrichmentChecker initialization failed: {e}")
                self.enrichment_checker = None
                self.enrichment_checker_available = False
        else:
            print("⚠️ EnrichmentChecker module not available")
        
        # Initialize individual node instances
        self._initialize_nodes()
    
    def _initialize_nodes(self):
        """Initialize all workflow node instances with shared dependencies."""
        common_args = (
            self.initial_annotation_content,
            self.initial_cell_types,
            self.adata,
            self.history_manager,
            self.hierarchy_manager,
            self.cell_type_extractor,
            self.function_descriptions,
            self.function_mapping,
            self.visualization_functions,
            self.enrichment_checker,  # Add EnrichmentChecker to common args
            self.enrichment_checker_available  # Add availability flag
        )
        
        self.input_processor = InputProcessorNode(*common_args)
        self.planner = PlannerNode(*common_args)
        self.executor = ExecutorNode(*common_args)
        self.validator = ValidationNode(*common_args)
        self.response_generator = ResponseGeneratorNode(*common_args)
        
        print("✅ All workflow nodes initialized successfully")
    
    # Orchestrator methods that delegate to individual nodes
    
    def input_processor_node(self, state: ChatState) -> ChatState:
        """Process incoming user message - delegates to InputProcessorNode."""
        print("🔄 Orchestrator: Delegating to InputProcessorNode")
        return self.input_processor.execute(state)
    
    def planner_node(self, state: ChatState) -> ChatState:
        """Create execution plan - delegates to PlannerNode."""
        print("🔄 Orchestrator: Delegating to PlannerNode")
        return self.planner.execute(state)
    
    def executor_node(self, state: ChatState) -> ChatState:
        """Execute plan step - delegates to ExecutorNode."""
        print("🔄 Orchestrator: Delegating to ExecutorNode")
        return self.executor.execute(state)
    
    def unified_response_generator_node(self, state: ChatState) -> ChatState:
        """Generate response - delegates to ResponseGeneratorNode."""
        print("🔄 Orchestrator: Delegating to ResponseGeneratorNode")
        return self.response_generator.execute(state)
    
    # Validation methods that delegate to ValidationNode
    
    def validate_processing_results(self, processed_parent: str, expected_children: List[str]) -> Dict[str, Any]:
        """Validate processing results - delegates to ValidationNode."""
        return self.validator.validate_processing_results(processed_parent, expected_children)
    
    # Legacy compatibility methods - these ensure backward compatibility
    # with existing code that might call these methods directly
    
    def _call_llm(self, prompt: str, model_name: str = "gpt-4o") -> str:
        """Legacy LLM call method - uses base node implementation."""
        return self.input_processor._call_llm(prompt)
    
    def _classify_question_type(self, question: str) -> str:
        """Legacy question classification - simplified implementation."""
        # This could be enhanced or removed if no longer needed
        return "analysis"  # Default classification
    
    def _store_execution_result(self, step_data: Dict, result: Any, success: bool, original_function_name: str = None) -> Dict[str, Any]:
        """Legacy result storage - delegates to ExecutorNode."""
        return self.executor._store_execution_result(step_data, result, success, original_function_name)
    
    def _update_available_cell_types_from_result(self, state: ChatState, result: Any) -> None:
        """Legacy cell type update - delegates to ExecutorNode."""
        return self.executor._update_available_cell_types_from_result(state, result)
    
    def _update_remaining_steps_with_discovered_types(self, state: ChatState, validation_result: Dict[str, Any]) -> None:
        """Legacy step update - delegates to ValidationNode."""
        return self.validator.update_remaining_steps_with_discovered_types(state, validation_result)
    
    # Additional orchestrator utilities
    
    def get_node_status(self) -> Dict[str, str]:
        """Get status of all workflow nodes."""
        return {
            "input_processor": "ready",
            "planner": "ready", 
            "executor": "ready",
            "validator": "ready",
            "response_generator": "ready",
            "enrichment_checker": "available" if self.enrichment_checker_available else "unavailable"
        }
    
    def reset_nodes(self):
        """Reset all nodes to initial state if needed."""
        print("🔄 Orchestrator: Resetting all workflow nodes")
        self._initialize_nodes()
    
    # Utility methods for backward compatibility with existing workflows
    
    def get_unified_results_for_synthesis(self, execution_history: List[Dict]) -> str:
        """Get unified results for synthesis - delegates to response generator."""
        # This could be moved to a utility module later
        from .unified_result_accessor import get_unified_results_for_synthesis
        return get_unified_results_for_synthesis(execution_history)


# For backward compatibility, maintain the same interface
# This allows existing code to continue working without changes
def create_core_nodes(*args, **kwargs):
    """Factory function for creating CoreNodes instances."""
    return CoreNodes(*args, **kwargs)