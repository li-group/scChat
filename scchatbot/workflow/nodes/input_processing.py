"""
Input processing node implementation.

This module contains the InputProcessorNode which handles incoming user messages,
manages conversation context, and prepares the state for planning.
"""

import json
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from ...cell_types.models import ChatState
from ..node_base import BaseWorkflowNode, ProcessingNodeMixin

import logging
logger = logging.getLogger(__name__)


class InputProcessorNode(BaseWorkflowNode, ProcessingNodeMixin):
    """
    Input processor node that handles incoming user messages.
    
    Responsibilities:
    - Process incoming user messages
    - Manage conversation context and history
    - Initialize/preserve cell type discoveries
    - Prepare state for planning phase
    """
    
    def execute(self, state: ChatState) -> ChatState:
        """Main execution method for input processing."""
        return self.input_processor_node(state)
    
    def input_processor_node(self, state: ChatState) -> ChatState:
        """Process incoming user message with conversation-aware state preservation"""
        self._log_node_start("InputProcessor", state)
        
        if not state.get("messages"):
            state["messages"] = [AIMessage(content=self.initial_annotation_content)]
        
        current_message = state["current_message"]
        state["has_conversation_context"] = False
        
        if hasattr(self.history_manager, 'search_conversations'):
            self._process_conversation_context(state, current_message)
        
        state["messages"].append(HumanMessage(content=state["current_message"]))
        
        self._continue_input_processing(state)
        
        self._log_node_complete("InputProcessor", state)
        return state
    
    def _process_conversation_context(self, state: ChatState, current_message: str):
        """Process conversation context using LLM-driven context analysis."""
        available_cell_types = list(state.get("available_cell_types", []))
        recent_analyses = [h.get("function_name") for h in state.get("execution_history", [])[-3:]]
        
        context_analysis_prompt = f"""
                                    User asked: "{current_message}"
                                    
                                    Current session context:
                                    - Available cell types: {available_cell_types}
                                    - Recent analyses: {recent_analyses}
                                    
                                    To answer this question, determine what context to provide:
                                    
                                    1. If asking about current state (what cell types, current results): ["current_session_state"]
                                    2. If referencing previous work or asking follow-up questions: generate search queries for conversation history
                                    3. If unclear or asking "what else": provide both current state AND search previous conversations
                                    
                                    Examples:
                                    - "What cell types do we have?" → ["current_session_state"]
                                    - "What about the pathway analysis we did before?" → ["pathway analysis results", "enrichment analysis"]  
                                    - "What else?" → ["current_session_state", "recent analysis results", "what other analyses available"]
                                    
                                    Return a JSON list of search queries/context types, or an empty list if no context needed.
                                    Only return the JSON list, nothing else.
                                    """
        
        try:
            search_queries_json = self._call_llm(context_analysis_prompt)
            logger.info(f"🔍 LLM search query response: '{search_queries_json}'")
            
            search_queries = self._safe_json_parse(search_queries_json)
            if search_queries is None:
                logger.info("⚠️ LLM returned invalid response for context search")
                search_queries = []
            
            if search_queries:
                logger.info(f"🧠 LLM requested {len(search_queries)} context sources: {search_queries}")
                self._add_context_to_state(state, search_queries)
                
        except Exception as e:
            logger.info(f"⚠️ Context retrieval skipped: {e}")
    
    def _add_context_to_state(self, state: ChatState, search_queries: List[str]):
        """Add context information to state based on search queries."""
        messages_to_keep = []
        context_messages_removed = 0
        
        for msg in state.get("messages", []):
            if isinstance(msg, AIMessage) and any(prefix in msg.content for prefix in 
                ["CURRENT_SESSION_STATE:", "CONVERSATION_HISTORY:", "CONVERSATION_CONTEXT:"]):
                context_messages_removed += 1
                logger.info(f"🧹 CONTEXT CLEANUP: Removing old context message ({len(msg.content)} chars)")
            else:
                messages_to_keep.append(msg)
        
        if context_messages_removed > 0:
            state["messages"] = messages_to_keep
            logger.info(f"🧹 CONTEXT CLEANUP: Removed {context_messages_removed} old context messages")
        
        context_parts = []
        
        for query in search_queries:
            if query == "current_session_state":
                session_context = self._build_session_state_context(state)
                context_parts.append(f"CURRENT_SESSION_STATE: {session_context}")
                logger.info(f"✅ Added current session state context ({len(session_context)} chars)")
                logger.info(f"🔍 SESSION CONTEXT DEBUG: First 100 chars: '{session_context[:100]}...'")
            else:
                results = self.history_manager.search_conversations(query, k=2)
                if results:
                    formatted_results = self.history_manager.format_search_results(results)
                    context_parts.append(f"CONVERSATION_HISTORY: {formatted_results}")
                    logger.info(f"✅ Added conversation context for '{query}' ({len(formatted_results)} chars)")
        
        if context_parts:
            full_context = "\n\n".join(context_parts)
            state["messages"].append(AIMessage(content=full_context))
            state["has_conversation_context"] = True
            logger.info(f"✅ Added fresh unified context ({len(full_context)} chars)")
    
    def _continue_input_processing(self, state: ChatState):
        """Continue the input processing after context retrieval"""
        if not state.get("available_cell_types"):
            state["available_cell_types"] = self.initial_cell_types
            logger.info(f"🔄 Initialized with {len(self.initial_cell_types)} initial cell types")
        else:
            existing_count = len(state["available_cell_types"])
            logger.info(f"✅ Preserving {existing_count} discovered cell types from previous operations")
            
            logger.info(f"   Preserved types: {sorted(state['available_cell_types'])}")
        
        state["adata"] = self.adata
        
        state["execution_plan"] = None
        state["current_step_index"] = 0
        state["execution_history"] = []  # Clear for new question
        state["function_result"] = None
        state["function_name"] = None
        state["function_args"] = None
        state["conversation_complete"] = False
        state["errors"] = []
        
        state["function_history_summary"] = self.history_manager.get_available_results()
        state["missing_cell_types"] = []
        state["required_preprocessing"] = []
        
        state["unavailable_cell_types"] = []
        
        return state