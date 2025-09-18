"""
Base classes for workflow nodes.

This module contains the base classes and common functionality shared across
all workflow nodes in the system.
"""

import json
from typing import Dict, Any, List
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ..cell_types.models import ChatState
import logging
logger = logging.getLogger(__name__)


class BaseWorkflowNode(ABC):
    """
    Abstract base class for all workflow nodes.
    
    Provides common functionality including LLM calls, state validation,
    and shared utilities that all nodes can use.
    """
    
    def __init__(self, initial_annotation_content, initial_cell_types, adata, 
                 history_manager, hierarchy_manager, cell_type_extractor,
                 function_descriptions, function_mapping, visualization_functions,
                 enrichment_checker=None, enrichment_checker_available=False):
        """Initialize base node with common dependencies."""
        self.initial_annotation_content = initial_annotation_content
        self.initial_cell_types = initial_cell_types
        self.adata = adata
        self.history_manager = history_manager
        self.hierarchy_manager = hierarchy_manager
        self.cell_type_extractor = cell_type_extractor
        self.function_descriptions = function_descriptions
        self.function_mapping = function_mapping
        self.visualization_functions = visualization_functions
        self.enrichment_checker = enrichment_checker
        self.enrichment_checker_available = enrichment_checker_available
        
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0
        )
    
    @abstractmethod
    def execute(self, state: ChatState) -> ChatState:
        """
        Execute the node's main functionality.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        pass
    
    def _call_llm(self, prompt: str, system_message: str = None) -> str:
        """
        Make a call to the LLM with error handling.
        
        Args:
            prompt: The prompt to send to the LLM
            system_message: Optional system message for context
            
        Returns:
            LLM response as string
        """
        try:
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=prompt))
            
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.info(f"⚠️ LLM call failed: {e}")
            return ""
    
    def _validate_state(self, state: ChatState, required_keys: List[str]) -> bool:
        """
        Validate that state contains required keys.
        
        Args:
            state: State to validate
            required_keys: List of required keys
            
        Returns:
            True if all required keys present
        """
        missing_keys = [key for key in required_keys if key not in state]
        if missing_keys:
            logger.info(f"⚠️ State validation failed. Missing keys: {missing_keys}")
            return False
        return True
    
    def _safe_json_parse(self, json_string: str) -> Any:
        """
        Safely parse JSON with common cleanup operations.
        
        Args:
            json_string: JSON string to parse
            
        Returns:
            Parsed JSON object or None if parsing fails
        """
        if not json_string or json_string.strip() == "":
            return None
        
        try:
            clean_json = json_string.strip()
            
            if clean_json.startswith('```json'):
                clean_json = clean_json[7:]
            elif clean_json.startswith('```'):
                clean_json = clean_json[3:]
            
            if clean_json.endswith('```'):
                clean_json = clean_json[:-3]
            
            clean_json = clean_json.strip()
            
            if not clean_json:
                return None
                
            return json.loads(clean_json)
        except json.JSONDecodeError as e:
            logger.info(f"⚠️ JSON parsing failed: {e}")
            logger.info(f"⚠️ Raw input: '{json_string}'")
            return None
    
    def _log_node_start(self, node_name: str, state: ChatState):
        """Log node execution start with context."""
        logger.info(f"🔄 {node_name}: Starting execution")
        if "current_message" in state:
            logger.info(f"   Current message: {state['current_message'][:100]}...")
    
    def _log_node_complete(self, node_name: str, state: ChatState):
        """Log node execution completion with context."""
        logger.info(f"✅ {node_name}: Execution complete")


class ProcessingNodeMixin:
    """
    Mixin for nodes that process data and need common processing utilities.
    """
    
    def _build_session_state_context(self, state: ChatState) -> str:
        """Build context from current session state and recent execution results"""
        logger.info(f"🔍 CONTEXT BUILDER: Building session context...")
        context_parts = []
        
        available_cell_types = list(state.get("available_cell_types", []))
        logger.info(f"🔍 CONTEXT BUILDER: Found {len(available_cell_types)} cell types: {available_cell_types}")
        
        if available_cell_types:
            context_parts.append(f"Cell types currently available in the dataset:")
            for cell_type in sorted(available_cell_types):
                context_parts.append(f"  • {cell_type}")
            context_parts.append("")  # Add spacing
        
        execution_history = state.get("execution_history", [])
        if execution_history:
            recent_steps = execution_history[-3:]  # Last 3 steps for brevity
            successful_steps = [step for step in recent_steps if step.get("success")]
            
            if successful_steps:
                context_parts.append("Recent analysis activities:")
                for step in successful_steps:
                    function_name = step.get("function_name", "unknown")
                    params = step.get("parameters", {})
                    cell_type = params.get("cell_type", "")
                    
                    if function_name == "perform_enrichment_analyses":
                        desc = f"Enrichment analysis completed for {cell_type}"
                    elif function_name == "process_cells":
                        desc = f"Cell processing and annotation for {cell_type}"
                    elif function_name == "search_enrichment_semantic":
                        query = params.get("query", "")
                        desc = f"Searched for '{query}' in {cell_type} results"
                    else:
                        desc = f"{function_name.replace('_', ' ').title()}"
                        if cell_type:
                            desc += f" for {cell_type}"
                    
                    context_parts.append(f"  • {desc}")
                context_parts.append("")  # Add spacing
        
        result = "\n".join(context_parts) if context_parts else "No current session data available"
        logger.info(f"🔍 CONTEXT BUILDER: Generated {len(result)} chars, starts with: '{result[:50]}...'")
        return result