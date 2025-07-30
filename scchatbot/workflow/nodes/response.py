"""
Response generation node implementation.

This module contains the ResponseGeneratorNode which generates final responses
by synthesizing analysis results and conversation context.
"""

from typing import Dict, Any, List

from ...cell_type_models import ChatState
from ..node_base import BaseWorkflowNode
from ..response import ResponseMixin


class ResponseGeneratorNode(BaseWorkflowNode, ResponseMixin):
    """
    Response generator node that creates final responses.
    
    Responsibilities:
    - Synthesize analysis results into coherent responses
    - Integrate conversation context and history
    - Generate LLM-based responses with proper formatting
    - Handle visualization integration
    """
    
    def execute(self, state: ChatState) -> ChatState:
        """Main execution method for response generation."""
        return self.unified_response_generator_node(state)
    
    def generate_response(self, state: ChatState) -> ChatState:
        """
        Alternative entry point for response generation.
        Delegates to the main unified response generator.
        """
        self._log_node_start("ResponseGenerator", state)
        
        result = self.unified_response_generator_node(state)
        
        self._log_node_complete("ResponseGenerator", state)
        return result