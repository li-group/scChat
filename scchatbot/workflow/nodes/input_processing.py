"""
Input processing node implementation.

This module contains the InputProcessorNode which handles incoming user messages,
manages conversation context, and prepares the state for planning.
"""

import json
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from ...cell_type_models import ChatState
from ..node_base import BaseWorkflowNode, ProcessingNodeMixin


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
        
        # Initialize state if this is a new conversation
        if not state.get("messages"):
            state["messages"] = [AIMessage(content=self.initial_annotation_content)]
        
        current_message = state["current_message"]
        state["has_conversation_context"] = False
        
        # Use LLM to determine what context information is needed
        if hasattr(self.history_manager, 'search_conversations'):
            self._process_conversation_context(state, current_message)
        
        # Add user message
        state["messages"].append(HumanMessage(content=state["current_message"]))
        
        # Continue with the rest of input processing
        self._continue_input_processing(state)
        
        self._log_node_complete("InputProcessor", state)
        return state
    
    def _process_conversation_context(self, state: ChatState, current_message: str):
        """Process conversation context using LLM-driven context analysis."""
        # Provide current session state information to LLM
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
            # LLM decides if context is needed and what to search for
            search_queries_json = self._call_llm(context_analysis_prompt)
            print(f"🔍 LLM search query response: '{search_queries_json}'")
            
            search_queries = self._safe_json_parse(search_queries_json)
            if search_queries is None:
                print("⚠️ LLM returned invalid response for context search")
                search_queries = []
            
            if search_queries:
                print(f"🧠 LLM requested {len(search_queries)} context sources: {search_queries}")
                self._add_context_to_state(state, search_queries)
                
        except Exception as e:
            # Silent fail - continue without context
            print(f"⚠️ Context retrieval skipped: {e}")
    
    def _add_context_to_state(self, state: ChatState, search_queries: List[str]):
        """Add context information to state based on search queries."""
        # 🧹 CLEAR OLD CONTEXT: Remove any existing context messages to avoid accumulation
        messages_to_keep = []
        context_messages_removed = 0
        
        for msg in state.get("messages", []):
            if isinstance(msg, AIMessage) and any(prefix in msg.content for prefix in 
                ["CURRENT_SESSION_STATE:", "CONVERSATION_HISTORY:", "CONVERSATION_CONTEXT:"]):
                context_messages_removed += 1
                print(f"🧹 CONTEXT CLEANUP: Removing old context message ({len(msg.content)} chars)")
            else:
                messages_to_keep.append(msg)
        
        if context_messages_removed > 0:
            state["messages"] = messages_to_keep
            print(f"🧹 CONTEXT CLEANUP: Removed {context_messages_removed} old context messages")
        
        context_parts = []
        
        # Handle different types of context requests  
        for query in search_queries:
            if query == "current_session_state":
                # Add current session state information
                session_context = self._build_session_state_context(state)
                context_parts.append(f"CURRENT_SESSION_STATE: {session_context}")
                print(f"✅ Added current session state context ({len(session_context)} chars)")
                print(f"🔍 SESSION CONTEXT DEBUG: First 100 chars: '{session_context[:100]}...'")
            else:
                # Treat as conversation search query
                results = self.history_manager.search_conversations(query, k=2)
                if results:
                    formatted_results = self.history_manager.format_search_results(results)
                    context_parts.append(f"CONVERSATION_HISTORY: {formatted_results}")
                    print(f"✅ Added conversation context for '{query}' ({len(formatted_results)} chars)")
        
        if context_parts:
            # Add fresh context to state
            full_context = "\n\n".join(context_parts)
            state["messages"].append(AIMessage(content=full_context))
            state["has_conversation_context"] = True
            print(f"✅ Added fresh unified context ({len(full_context)} chars)")
    
    def _continue_input_processing(self, state: ChatState):
        """Continue the input processing after context retrieval"""
        # ✅ SMART INITIALIZATION: Preserve discovered cell types across questions
        if not state.get("available_cell_types"):
            # First question or state is empty - use initial types
            state["available_cell_types"] = self.initial_cell_types
            print(f"🔄 Initialized with {len(self.initial_cell_types)} initial cell types")
        else:
            # Subsequent questions - preserve existing discovered types
            existing_count = len(state["available_cell_types"])
            print(f"✅ Preserving {existing_count} discovered cell types from previous operations")
            
            # Optional: Debug logging to track what types are preserved
            print(f"   Preserved types: {sorted(state['available_cell_types'])}")
        
        state["adata"] = self.adata
        
        # Reset only transient state for new questions, preserve conversation context
        state["execution_plan"] = None
        state["current_step_index"] = 0
        state["execution_history"] = []  # Clear for new question
        state["function_result"] = None
        state["function_name"] = None
        state["function_args"] = None
        state["conversation_complete"] = False
        state["errors"] = []
        
        # Load function history and memory context
        state["function_history_summary"] = self.history_manager.get_available_results()
        state["missing_cell_types"] = []
        state["required_preprocessing"] = []
        
        # Initialize unavailable cell types tracking
        state["unavailable_cell_types"] = []
        
        return state