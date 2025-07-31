"""
Core workflow nodes implementation.

This module contains the basic node implementations:
- input_processor_node
- planner_node  
- executor_node
"""

import json
import re
from typing import Dict, Any, List
from datetime import datetime

from ..cell_type_models import ChatState, ExecutionStep
from langchain_core.messages import HumanMessage, AIMessage
from ..shared import extract_cell_types_from_question, needs_cell_discovery

# Import EnrichmentChecker with error handling
# ABLATION STUDY: EnrichmentChecker disabled for testing
try:
    from ..enrichment_checker import EnrichmentChecker
    ENRICHMENT_CHECKER_AVAILABLE = True  # 🔴 DISABLED for ablation study
except ImportError as e:
    print(f"⚠️ EnrichmentChecker not available: {e}")
    EnrichmentChecker = None
    ENRICHMENT_CHECKER_AVAILABLE = False


class CoreNodes:
    """Core workflow nodes for input processing, planning, and execution."""
    
    def __init__(self, initial_annotation_content, initial_cell_types, adata, 
                 history_manager, hierarchy_manager, cell_type_extractor,
                 function_descriptions, function_mapping, visualization_functions):
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
    
    def input_processor_node(self, state: ChatState) -> ChatState:
        """Process incoming user message with conversation-aware state preservation"""
        # Initialize state if this is a new conversation
        if not state.get("messages"):
            state["messages"] = [AIMessage(content=self.initial_annotation_content)]
        
        current_message = state["current_message"]
        state["has_conversation_context"] = False
        
        # Use LLM to determine what context information is needed
        if hasattr(self.history_manager, 'search_conversations'):
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
                
                # Handle empty or malformed JSON responses
                if not search_queries_json or search_queries_json.strip() == "":
                    print("⚠️ LLM returned empty response for context search")
                    search_queries = []
                else:
                    try:
                        # Strip markdown code blocks if present and handle truncated responses
                        clean_json = search_queries_json.strip()
                        
                        # Handle various markdown formats
                        if clean_json.startswith('```json'):
                            clean_json = clean_json[7:]
                        elif clean_json.startswith('```'):
                            clean_json = clean_json[3:]
                        
                        if clean_json.endswith('```'):
                            clean_json = clean_json[:-3]
                        
                        clean_json = clean_json.strip()
                        
                        # Handle empty or incomplete responses
                        if not clean_json or clean_json == '':
                            search_queries = []
                        else:
                            search_queries = json.loads(clean_json)
                    except json.JSONDecodeError as json_error:
                        print(f"⚠️ LLM returned malformed JSON: {json_error}")
                        print(f"⚠️ Raw response: '{search_queries_json}'")
                        search_queries = []
                
                if search_queries:
                    print(f"🧠 LLM requested {len(search_queries)} context sources: {search_queries}")
                    
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
                    
            except Exception as e:
                # Silent fail - continue without context
                print(f"⚠️ Context retrieval skipped: {e}")
        
        # Add user message
        state["messages"].append(HumanMessage(content=state["current_message"]))
        
        # Continue with the rest of input processing
        self._continue_input_processing(state)
        return state
    
    def _build_session_state_context(self, state: ChatState) -> str:
        """Build context from current session state and recent execution results"""
        print(f"🔍 CONTEXT BUILDER: Building session context...")
        context_parts = []
        
        # Available cell types with better formatting
        available_cell_types = list(state.get("available_cell_types", []))
        print(f"🔍 CONTEXT BUILDER: Found {len(available_cell_types)} cell types: {available_cell_types}")
        
        if available_cell_types:
            context_parts.append(f"Cell types currently available in the dataset:")
            for cell_type in sorted(available_cell_types):
                context_parts.append(f"  • {cell_type}")
            context_parts.append("")  # Add spacing
        
        # Recent execution history with better context
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
                    
                    # Create more natural descriptions
                    if function_name == "perform_enrichment_analyses":
                        desc = f"Enrichment analysis completed for {cell_type}"
                    elif function_name == "process_cells":
                        desc = f"Cell processing and annotation for {cell_type}"
                    elif function_name == "search_enrichment_semantic":
                        query = params.get("query", "")
                        desc = f"Searched for '{query}' in {cell_type} results"
                    elif function_name == "conversational_response":
                        desc = "Provided response based on current data"
                    else:
                        desc = f"{function_name.replace('_', ' ').title()}"
                        if cell_type:
                            desc += f" for {cell_type}"
                    
                    context_parts.append(f"  • {desc}")
                context_parts.append("")  # Add spacing
        
        result = "\n".join(context_parts) if context_parts else "No current session data available"
        print(f"🔍 CONTEXT BUILDER: Generated {len(result)} chars, starts with: '{result[:50]}...'")
        return result
    
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

    def planner_node(self, state: ChatState) -> ChatState:
        """Create initial execution plan with current cell type awareness and enhanced prompting"""
        message = state["current_message"]
        available_functions = self.function_descriptions
        available_cell_types = state["available_cell_types"]
        function_history = state["function_history_summary"]
        unavailable_cell_types = state.get("unavailable_cell_types", [])
        
        # 🧬 Enhanced cell type awareness logging
        initial_count = len(self.initial_cell_types)
        current_count = len(available_cell_types)
        discovered_count = current_count - initial_count
        
        print(f"🧬 PLANNER: Cell type status for planning:")
        print(f"   • Initial types: {initial_count}")
        print(f"   • Currently available: {current_count}")
        print(f"   • Discovered this session: {discovered_count}")
        if unavailable_cell_types:
            print(f"   • Failed discoveries: {len(unavailable_cell_types)} - {', '.join(unavailable_cell_types)}")
        
        # Show discovered types if any
        if discovered_count > 0:
            discovered_types = set(available_cell_types) - set(self.initial_cell_types)
            print(f"   • New types discovered: {', '.join(sorted(discovered_types))}")
        
        print(f"🧬 Planning for question: '{message}'")
        
        # Enhanced LLM-based planning without artificial query type constraints
        return self._create_enhanced_plan(state, message, available_functions, available_cell_types, function_history, unavailable_cell_types)
    
    def _create_enhanced_plan(self, state: ChatState, message: str, available_functions: List, available_cell_types: List[str], function_history: Dict, unavailable_cell_types: List[str]) -> ChatState:
        """Create enhanced plan using semantic LLM understanding without artificial query type constraints"""
        
        # Extract conversation context for semantic search awareness
        conversation_context = ""
        has_conversation_context = state.get("has_conversation_context", False)
        if has_conversation_context:
            # Extract conversation context from messages
            for msg in state.get("messages", []):
                if hasattr(msg, 'content') and msg.content.startswith("CONVERSATION_CONTEXT:"):
                    conversation_context = msg.content[len("CONVERSATION_CONTEXT: "):]
                    break
        
        planning_prompt = f"""
        You are an intelligent planner for single-cell RNA-seq analysis. 
        
        Create a step-by-step execution plan for the user query.
        
        CONTEXT:
        - Currently available cell types ({len(available_cell_types)}): {', '.join(sorted(available_cell_types))}
        {f"- Cell types that failed discovery ({len(unavailable_cell_types)}): {', '.join(sorted(unavailable_cell_types))}" if unavailable_cell_types else "- No failed cell type discoveries"}
        - Cell type status: {'Expanded from initial set' if len(available_cell_types) > len(self.initial_cell_types) else 'Using initial cell types only'}
        - Previous analyses: {json.dumps(function_history, indent=2)}
        {"- Conversation context: " + conversation_context if conversation_context else ""}
        
        Available functions:
        {self._summarize_functions(available_functions)}
        
        User question: "{message}"
        
        SEMANTIC DECISION FRAMEWORK:
        🔬 USE ANALYSIS FUNCTIONS when the user wants to DISCOVER or ANALYZE data:
        - Questions about relationships, abundance, pathways, cellular processes
        - Requests to compare, find differences, or understand biological mechanisms  
        - Examples: "What pathways are enriched?", "How do cell types differ?", "What is the relationship between X and Y?"
        - → Use functions like: perform_enrichment_analyses, compare_cell_counts, dea_split_by_condition
        
        💬 USE CONVERSATIONAL RESPONSE only when the user wants to INTERPRET existing results:
        - Questions asking for explanation of already-computed results
        - Requests to clarify meaning of specific terms or findings
        - Examples: "What does this pathway mean?", "Explain these results I'm seeing"
        - → Use: conversational_response
        
        🎯 DEFAULT: When in doubt, prefer analysis over conversation. It's better to provide data-driven insights.
        
        Create a plan in this JSON format:
        {{
            "plan_summary": "Brief description of how you'll answer this question",
            "visualization_only": true/false,
            "steps": [
                {{
                    "step_type": "analysis|visualization|conversation",
                    "function_name": "exact_function_name", 
                    "parameters": {{"param1": "value1"}},
                    "description": "What this step accomplishes",
                    "expected_outcome": "What we expect to produce",
                    "target_cell_type": "If applicable, which cell type"
                }}
            ]
        }}
        
        IMPORTANT GUIDELINES: 
        - When analyzing multiple cell types, create separate steps for each cell type
        - For example, if comparing "T cells" and "B cells", create separate steps:
          Step 1: analyze T cell, Step 2: analyze B cell, Step 3: compare results
        - Never put multiple cell types in a single parameter (e.g., don't use "T cells, B cells")
        - Use exact cell type names (e.g., "T cell", "B cell", not "T cells, B cells")
        - SKIP steps for unavailable cell types: {', '.join(unavailable_cell_types)}
        - CELL TYPE STRATEGY: If a requested cell type is not in the available list, consider if it needs discovery
        - AVAILABLE TYPES PRIORITY: Prefer using currently available cell types when possible
        - Focus on creating a logical flow to answer the user's question
        
        VISUALIZATION-ONLY DETECTION:
        - Set "visualization_only": true if the user ONLY wants to see plots/visualizations
        - Examples of visualization-only requests:
          * "Show the GSEA barplot for T cell"
          * "Display the GO dotplot"
          * "Plot the enrichment results"
          * "Generate UMAP visualization"
        - Set "visualization_only": false for analysis requests or questions needing interpretation:
          * "What pathways are enriched in T cells?"
          * "Run GSEA analysis and explain results"
          * "Compare T cell vs B cell enrichment"
          
        ENRICHMENT ANALYSIS GUIDELINES:
        - For enrichment analysis steps, use MINIMAL parameters - only specify "cell_type"
        - Do NOT specify "analyses", "gene_set_library", or "pathway_include" parameters
        - The EnrichmentChecker will automatically determine optimal analysis methods and parameters
        - Examples:
          * "Run GSEA analysis on T cells" → {{"cell_type": "T cell"}} (EnrichmentChecker adds analyses)
          * "Find pathways enriched in B cells" → {{"cell_type": "B cell"}} (EnrichmentChecker determines methods)
          * "Analyze immune pathways" → {{"cell_type": "Immune cell"}} (EnrichmentChecker handles targeting)
          
        VISUALIZATION GUIDELINES:
        - For enrichment visualization, ALWAYS prefer "display_enrichment_visualization":
          * Use "display_enrichment_visualization" for ALL enrichment plots (shows both bar + dot by default)
          * Only use "display_enrichment_barplot" or "display_enrichment_dotplot" if user specifically asks for ONLY one type
          * ALWAYS specify the "analysis" parameter to match what was performed
          * Examples:
            - "show GO plots" → use "display_enrichment_visualization" with "analysis": "go"
            - "visualize GSEA results" → use "display_enrichment_visualization" with "analysis": "gsea"  
            - "show ONLY barplot" → use "display_enrichment_barplot"
            - "display both plots" → use "display_enrichment_visualization" (default plot_type="both")
            
        SEMANTIC SEARCH GUIDELINES:
        - For questions seeking specific pathway/term information beyond the top-ranked results, consider using "search_enrichment_semantic"
        - Use semantic search when:
          * User asks about specific pathways that might not be in top results (e.g., "cell cycle regulation", "apoptosis pathways")
          * User references conversation context about previous analyses and wants to explore related pathways
          * User wants to find terms similar to those mentioned in conversation context
        - Parameters for search_enrichment_semantic:
          * "query": the pathway/term to search for (e.g., "cell cycle regulation")
          * "cell_type": target cell type (can be inferred from conversation context if not explicitly mentioned)
          * Optional: "analysis_type", "condition", "limit"
        - Examples:
          * "Show me cell cycle related terms from the T cell analysis" → search_enrichment_semantic with query="cell cycle" and cell_type="T cell"
          * "Are there any apoptosis pathways in our results?" → search_enrichment_semantic with query="apoptosis"
          * "Find pathways similar to what we discussed earlier" → use conversation context to determine relevant search terms
        """
        
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage
            
            # Create messages in LangChain format
            messages = [
                SystemMessage(content="You are a bioinformatics analysis planner. Generate execution plans in JSON format."),
                HumanMessage(content=planning_prompt)
            ]
            
            # Initialize model
            model = ChatOpenAI(
                model="gpt-4o",
                temperature=0.1,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
            
            # Get response
            response = model.invoke(messages)
            plan_data = json.loads(response.content)
            
            # 🧬 ENHANCED PLANNER: Add cell discovery if needed 
            enhanced_plan = self._add_cell_discovery_to_plan(plan_data, message, available_cell_types)
            
            # Apply enrichment enhancement to all enrichment steps
            print(f"🔍 ENRICHMENT DEBUG: Checking for enrichment steps in plan...")
            enrichment_steps = [s for s in enhanced_plan.get("steps", []) if s.get("function_name") == "perform_enrichment_analyses"]
            print(f"🔍 ENRICHMENT DEBUG: Found {len(enrichment_steps)} enrichment steps")
            
            enhanced_plan = self._enhance_all_enrichment_steps(enhanced_plan, message)
            
            # Skip steps for unavailable cell types
            if unavailable_cell_types:
                enhanced_plan = self._skip_unavailable_cell_steps(enhanced_plan, unavailable_cell_types)
            
            # Apply plan processing (moved from evaluator)
            # 1. Light consolidation - only remove exact consecutive duplicates
            enhanced_plan = self._light_consolidate_process_cells(enhanced_plan)
            
            # 2. Light validation - only log warnings for missing cell types
            self._log_missing_cell_type_warnings(enhanced_plan)
            
            # 3. Skip adding validation steps - they're already added by discovery pipeline
            # enhanced_plan = self._add_validation_steps_after_process_cells(enhanced_plan)
            
            # Store as execution plan directly (planner now outputs final plan)
            state["execution_plan"] = enhanced_plan
            state["execution_plan"]["original_question"] = message
            
            # Log plan statistics
            print(f"✅ Planner created execution plan with {len(enhanced_plan['steps'])} steps")
            print(f"   • {len([s for s in enhanced_plan['steps'] if s.get('function_name') == 'process_cells'])} process_cells steps")
            print(f"   • {len([s for s in enhanced_plan['steps'] if s.get('step_type') == 'validation'])} validation steps")
            print(f"   • {len([s for s in enhanced_plan['steps'] if s.get('step_type') == 'analysis'])} analysis steps")
            
        except Exception as e:
            print(f"Planning error: {e}")
            # Fallback: create a simple conversational response plan
            state["execution_plan"] = {
                "plan_summary": "Fallback conversational response",
                "steps": [{
                    "step_type": "conversation",
                    "function_name": "conversational_response",
                    "parameters": {"response_type": "general"},
                    "description": "Provide a helpful response",
                    "expected_outcome": "Address user query",
                    "target_cell_type": None
                }]
            }
            
        return state
    

    def _store_execution_result(self, step_data: Dict, result: Any, success: bool, original_function_name: str = None) -> Dict[str, Any]:
        """
        New intelligent result storage that preserves structure for critical functions
        """
        # Use provided original function name or fallback to step_data
        function_name = original_function_name or step_data.get("function_name", "")
        
        # Critical functions that need full structure preservation
        STRUCTURE_PRESERVED_FUNCTIONS = {
            "perform_enrichment_analyses",
            "dea_split_by_condition", 
            "process_cells",
            "compare_cell_counts",
            "search_enrichment_semantic"
        }
        
        if function_name in STRUCTURE_PRESERVED_FUNCTIONS and success:
            return {
                "result_type": "structured",
                "result": result,  # Full structured data
                "result_summary": self._create_result_summary(function_name, result)
            }
        
        elif function_name.startswith("display_") and success:
            # Visualization functions - keep HTML but add metadata
            return {
                "result_type": "visualization", 
                "result": result,  # Full HTML
                "result_metadata": self._extract_viz_metadata(function_name, result),
                "result_summary": f"Visualization generated: {function_name}"
            }
        
        else:
            # Other functions - use existing truncation
            return {
                "result_type": "text",
                "result": str(result)[:500] if result else "Success",
                "result_summary": str(result)[:100] if result else "Success"
            }

    def _create_result_summary(self, function_name: str, result: Any) -> str:
        """Create human-readable summaries for logging while preserving full data"""
        
        if function_name == "perform_enrichment_analyses" and isinstance(result, dict):
            summary_parts = []
            for analysis_type in ["go", "kegg", "reactome", "gsea"]:
                if analysis_type in result:
                    count = result[analysis_type].get("total_significant", 0)
                    summary_parts.append(f"{analysis_type.upper()}: {count} terms")
            return f"Enrichment: {', '.join(summary_parts)}"
        
        elif function_name == "process_cells":
            if isinstance(result, str) and "discovered" in result.lower():
                return f"Process cells: Discovery completed"
            return f"Process cells: {str(result)[:100]}"
        
        elif function_name == "dea_split_by_condition":
            return f"DEA: Analysis completed"
        
        elif function_name == "search_enrichment_semantic":
            if isinstance(result, str) and "enrichment terms" in result:
                lines = result.split('\n')
                term_count = len([line for line in lines if line.strip() and not line.startswith('##')])
                return f"Semantic search: {term_count} matching terms found"
            return f"Semantic search: Results found"
        
        return str(result)[:100]

    def _extract_viz_metadata(self, function_name: str, result: Any) -> Dict[str, Any]:
        """Extract metadata from visualization results"""
        return {
            "visualization_type": function_name,
            "html_length": len(result) if isinstance(result, str) else 0,
            "contains_html": bool(isinstance(result, str) and ('<div' in result or '<html' in result))
        }

    def executor_node(self, state: ChatState) -> ChatState:
        """Execute the current step in the plan with hierarchy awareness and validation"""
        if not state["execution_plan"] or state["current_step_index"] >= len(state["execution_plan"]["steps"]):
            state["conversation_complete"] = True
            return state
            
        step_data = state["execution_plan"]["steps"][state["current_step_index"]]
        
        # Check if this step should be skipped
        if step_data.get("skip_reason"):
            print(f"⏭️ Skipping step {state['current_step_index'] + 1}: {step_data.get('skip_reason')}")
            
            # Record skipped step in execution history for post-execution awareness
            state["execution_history"].append({
                "step_index": state["current_step_index"],
                "step": step_data.copy(),
                "success": False,
                "result": None,
                "result_type": "skipped",
                "result_summary": f"Skipped - {step_data.get('skip_reason')}",
                "error": step_data.get('skip_reason'),
                "skipped": True  # Flag to distinguish from failures
            })
            
            state["current_step_index"] += 1
            # Don't run intelligent evaluator for skipped steps - nothing to evaluate
            return state
        
        step = ExecutionStep(**step_data)
        
        print(f"🔄 Executing step {state['current_step_index'] + 1}: {step.description}")
        
        # DEBUG: Log the original function name to track mutations
        original_function_name = step_data.get("function_name", "unknown")
        print(f"🔍 STORAGE DEBUG: Original function_name from plan: '{original_function_name}'")
        
        success = False
        result = None
        error_msg = None
        
        try:
            # Handle validation steps specially
            if step.step_type == "validation":
                print("🔍 Executing validation step...")
                result = self.validate_processing_results(
                    step.parameters.get("processed_parent"),
                    step.parameters.get("expected_children", [])
                )
                
                # Check validation result
                if result["status"] == "success":
                    success = True
                    print(f"✅ Validation passed: {result['message']}")
                    # Update available cell types with discovered types
                    state["available_cell_types"] = result["available_types"]
                elif result["status"] == "partial_success":
                    success = True  # Continue but with warnings
                    print(f"⚠️ Validation partial: {result['message']}")
                    # Update available cell types with what we actually found
                    state["available_cell_types"] = result["available_types"]
                    
                    # Track unavailable cell types
                    expected_types = step.parameters.get("expected_children", [])
                    available_types = result.get("available_types", [])
                    missing_types = [ct for ct in expected_types if ct not in available_types]
                    if missing_types:
                        current_unavailable = state.get("unavailable_cell_types", [])
                        state["unavailable_cell_types"] = list(set(current_unavailable + missing_types))
                        print(f"📋 Added to unavailable cell types: {missing_types}")
                else:
                    success = False
                    error_msg = result["message"]
                    print(f"❌ Validation failed: {error_msg}")
                    
                    # Track all expected cell types as unavailable on complete failure
                    expected_types = step.parameters.get("expected_children", [])
                    if expected_types:
                        current_unavailable = state.get("unavailable_cell_types", [])
                        state["unavailable_cell_types"] = list(set(current_unavailable + expected_types))
                        print(f"📋 Added to unavailable cell types (validation failed): {expected_types}")
                
                # CRITICAL FIX: Update subsequent analysis steps to use actually discovered cell types
                if success and result.get("found_children"):
                    self._update_remaining_steps_with_discovered_types(state, result)
                
            # Handle final question step differently
            elif step.step_type == "final_question":
                print("🎯 Executing final comprehensive question...")
                result = self._execute_final_question(state)
                success = True
                
            else:
                # Handle regular analysis/visualization steps
                if step.function_name not in self.function_mapping:
                    raise Exception(f"Function '{step.function_name}' not found")
                
                # Debug visualization function parameters
                if step.function_name in self.visualization_functions:
                    print(f"🔍 STEP DEBUG: Calling visualization function '{step.function_name}' with step parameters: {step.parameters}")
                
                # For visualization functions, enhance parameters with cell_type from execution context if missing
                enhanced_params = step.parameters.copy()
                if step.function_name in self.visualization_functions and "cell_type" not in enhanced_params:
                    # Look for cell_type in recent execution history
                    execution_history = state.get("execution_history", [])
                    
                    for recent_execution in reversed(execution_history[-5:]):  # Check last 5 executions
                        step_data = recent_execution.get("step", {})
                        step_params = step_data.get("parameters", {})
                        
                        if recent_execution.get("success") and "cell_type" in step_params:
                            cell_type = step_params["cell_type"]
                            if cell_type != "overall":  # Skip generic cell types
                                enhanced_params["cell_type"] = cell_type
                                print(f"🔧 Enhanced visualization with cell_type from execution context: {cell_type}")
                                break
                    
                    # If still no cell_type found, use default
                    if "cell_type" not in enhanced_params:
                        enhanced_params["cell_type"] = "overall"
                        print(f"⚠️ WARNING: cell_type defaulted to 'overall' - this may indicate planner issue.")
                
                func = self.function_mapping[step.function_name]
                result = func(**enhanced_params)
                success = True
            
                if step.function_name == "process_cells" and self.hierarchy_manager:
                    new_cell_types = self.cell_type_extractor.extract_from_annotation_result(result)
                    if new_cell_types:
                        print(f"🧬 Updating hierarchy manager with new cell types: {new_cell_types}")
                        self.hierarchy_manager.update_after_process_cells(
                            step.parameters.get("cell_type", "unknown"),
                            new_cell_types
                        )
                        
                        # Update available cell types in state
                        state["available_cell_types"] = list(set(state["available_cell_types"] + new_cell_types))
                        for new_type in new_cell_types:
                            print(f"✅ Discovered new cell type: {new_type}")
                    else:
                        print("⚠️ No new cell types discovered from process_cells")
            
            # Store results
            state["function_result"] = result
            state["function_name"] = step.function_name
            state["function_args"] = step.parameters
            
            print(f"✅ Step {state['current_step_index'] + 1} completed successfully")
            
            # Update available cell types if this was a successful process_cells step
            if success and step.function_name == "process_cells" and result:
                self._update_available_cell_types_from_result(state, result)
            
        except Exception as e:
            error_msg = str(e)
            success = False
            print(f"❌ Step {state['current_step_index'] + 1} failed: {error_msg}")
            state["errors"].append(f"Step {state['current_step_index'] + 1} failed: {error_msg}")
        
        # Record function execution in history (skip validation steps for history)
        if step.step_type != "validation":
            self.history_manager.record_execution(
                function_name=step.function_name,
                parameters=step.parameters,
                result=result,
                success=success,
                error=error_msg
            )
        
        # Record execution in state using new structured storage approach
        # Pass original function name to prevent using mutated name
        result_storage = self._store_execution_result(step_data, result, success, original_function_name)
        
        # CRITICAL FIX: Create a deep copy of step_data to prevent mutation issues
        # This ensures each execution history entry has its own independent step data
        import copy
        step_data_copy = copy.deepcopy(step_data)
        
        # DEBUG: Check if function name was mutated
        current_function_name = step_data.get("function_name", "unknown")
        if current_function_name != original_function_name:
            print(f"⚠️ MUTATION DETECTED: function_name changed from '{original_function_name}' to '{current_function_name}'")
            # Fix the function name in the copy
            step_data_copy["function_name"] = original_function_name
            print(f"🔧 CORRECTED: Restored function_name to '{original_function_name}' in execution history")
        
        state["execution_history"].append({
            "step_index": state["current_step_index"],
            "step": step_data_copy,  # Use copy to prevent mutation
            "success": success,
            "result": result_storage["result"],  # Full structure preserved for critical functions
            "result_type": result_storage["result_type"],
            "result_summary": result_storage["result_summary"],  # For logging
            "error": error_msg
        })
        
        # Log storage decision for monitoring (use original function name)
        function_name = original_function_name  # Use original name, not potentially mutated one
        if result_storage["result_type"] == "structured":
            print(f"📊 Structured storage: {function_name} - Full data preserved")
        elif result_storage["result_type"] == "visualization":
            print(f"🎨 Visualization storage: {function_name} - HTML preserved")
        else:
            print(f"📄 Text storage: {function_name} - Truncated to 500 chars")
        
        # Increment step index if step was successful OR if it's a validation step
        # (validation steps should always advance to avoid infinite loops)
        should_advance = success or step.step_type == "validation"
        
        if should_advance:
            state["current_step_index"] += 1
            if success:
                print(f"🔄 Advanced to step {state['current_step_index'] + 1}")
            else:
                print(f"🔄 Advanced to step {state['current_step_index'] + 1} (validation failure, but continuing)")
        else:
            print(f"❌ Step failed, not advancing. Still on step {state['current_step_index'] + 1}")
            
        return state

    def _add_cell_discovery_to_plan(self, plan_data: Dict[str, Any], message: str, available_cell_types: List[str]) -> Dict[str, Any]:
        """
        Enhance the initial plan by adding cell discovery steps if needed.
        
        Uses V2 implementation that preserves LLM's analysis steps.
        """
        return self._add_cell_discovery_to_plan_v2(plan_data, message, available_cell_types)
    
    
    def _add_cell_discovery_to_plan_v2(self, plan_data: Dict[str, Any], message: str, available_cell_types: List[str]) -> Dict[str, Any]:
        """
        V2 (NEW): Preserves LLM's analysis steps while adding discovery.
        """
        if not plan_data or not self.hierarchy_manager:
            return plan_data
        
        # Extract cell types mentioned in the user's question
        needed_cell_types = extract_cell_types_from_question(message, self.hierarchy_manager)
        
        if not needed_cell_types:
            print("🔍 No specific cell types identified in question")
            return plan_data
        
        print(f"🧬 Planner identified needed cell types: {needed_cell_types}")
        print(f"🧬 Available cell types: {available_cell_types}")
        
        # Step 1: Extract and categorize original steps
        original_steps = plan_data.get("steps", [])
        llm_analysis_steps = []
        other_steps = []
        
        for step in original_steps:
            func_name = step.get("function_name", "")
            # Preserve cell-type-specific analysis steps
            if func_name in ["perform_enrichment_analyses", "dea_split_by_condition", 
                           "compare_cell_counts", "analyze_cell_interaction"]:
                llm_analysis_steps.append(step)
                print(f"📋 Preserving LLM analysis step: {func_name}({step.get('parameters', {}).get('cell_type', 'unknown')})")
            else:
                other_steps.append(step)
        
        # Step 2: Fix cell type names in preserved steps
        llm_analysis_steps = self._fix_cell_type_names_in_steps(llm_analysis_steps, needed_cell_types, message)
        other_steps = self._fix_cell_type_names_in_steps(other_steps, needed_cell_types, message)
        
        # Step 3: Create discovery steps ONLY (no analysis)
        discovery_steps = []
        if needs_cell_discovery(needed_cell_types, available_cell_types):
            print("🧬 Creating discovery steps only...")
            discovery_steps = self._create_discovery_steps_only(needed_cell_types, available_cell_types)
        
        # Step 4: Skip validation steps - intelligent evaluator handles validation
        validation_steps = []
        # if discovery_steps:
        #     validation_steps = self._create_validation_steps(discovery_steps)
        
        # Step 5: Update analysis steps to use discovered cell types instead of parent types
        updated_analysis_steps = self._update_analysis_steps_for_discovered_types(
            llm_analysis_steps, needed_cell_types, available_cell_types
        )
        
        # Step 6: Merge steps intelligently
        final_steps = []
        final_steps.extend(discovery_steps)
        final_steps.extend(validation_steps)
        final_steps.extend(updated_analysis_steps)  # Updated to use discovered types!
        final_steps.extend(other_steps)
        
        plan_data["steps"] = final_steps
        
        # Update plan summary
        if discovery_steps:
            original_summary = plan_data.get("plan_summary", "")
            plan_data["plan_summary"] = f"Discover needed cell types then {original_summary.lower()}"
        
        print(f"📋 Plan merge complete:")
        print(f"   - Discovery steps: {len(discovery_steps)}")
        print(f"   - Validation steps: {len(validation_steps)}")
        print(f"   - LLM analysis steps: {len(llm_analysis_steps)}")
        print(f"   - Other steps: {len(other_steps)}")
        
        return plan_data
    
    def _create_discovery_steps_only(self, needed_cell_types: List[str], available_cell_types: List[str]) -> List[Dict[str, Any]]:
        """
        Create ONLY discovery steps, no analysis steps.
        This is different from create_cell_discovery_steps which adds hardcoded analysis.
        """
        discovery_steps = []
        parent_to_children = {}  # Track what each parent should discover
        
        # Step 1: Build parent → children mapping from all paths
        for needed_type in needed_cell_types:
            if needed_type in available_cell_types:
                print(f"✅ '{needed_type}' already available, no discovery needed")
                continue
            
            # Find processing path using hierarchy manager
            processing_path = None
            best_parent = None
            
            for available_type in available_cell_types:
                path_result = self.hierarchy_manager.find_parent_path(needed_type, [available_type])
                if path_result:
                    best_parent, processing_path = path_result
                    print(f"🔄 Found path from '{best_parent}' to '{needed_type}': {' → '.join(processing_path)}")
                    break
            
            if processing_path and len(processing_path) > 1:
                # Build parent → children mapping for this path
                for i in range(len(processing_path) - 1):
                    parent_type = processing_path[i]
                    child_type = processing_path[i + 1]
                    
                    if parent_type not in parent_to_children:
                        parent_to_children[parent_type] = []
                    
                    if child_type not in parent_to_children[parent_type]:
                        parent_to_children[parent_type].append(child_type)
            else:
                print(f"⚠️ No processing path found for '{needed_type}'")
        
        # Step 2: Create process_cells steps with expected_children metadata
        for parent_type, expected_children in parent_to_children.items():
            # Check if we already have this step
            existing = any(
                s.get("function_name") == "process_cells" and 
                s.get("parameters", {}).get("cell_type") == parent_type
                for s in discovery_steps
            )
            
            if not existing:
                discovery_steps.append({
                    "step_type": "analysis",
                    "function_name": "process_cells",
                    "parameters": {"cell_type": parent_type},
                    "description": f"Process {parent_type} to discover {', '.join(expected_children)}",
                    "expected_outcome": f"Discover {', '.join(expected_children)} cell type(s)",
                    "expected_children": expected_children
                })
                print(f"🧬 Added process_cells({parent_type}) to discover {expected_children}")
        
        return discovery_steps
    
    def _create_validation_steps(self, discovery_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create validation steps for discovery steps using metadata.
        """
        validation_steps = []
        
        # Create individual validation steps for each process_cells step
        for step in discovery_steps:
            if step.get("function_name") == "process_cells":
                cell_type = step.get("parameters", {}).get("cell_type")
                expected_children = step.get("expected_children", [])
                
                if cell_type and expected_children:
                    validation_steps.append({
                        "step_type": "validation",
                        "function_name": "validate_processing_results",
                        "parameters": {
                            "processed_parent": cell_type,
                            "expected_children": expected_children
                        },
                        "description": f"Validate that {cell_type} processing discovered expected cell types: {', '.join(expected_children)}",
                        "expected_outcome": f"Confirm {', '.join(expected_children)} are available after processing {cell_type}",
                        "target_cell_type": None
                    })
                    print(f"🔍 Created validation step for process_cells({cell_type}) expecting: {expected_children}")
        
        return validation_steps
    
    
    def _fix_cell_type_names_in_steps(self, steps: List[Dict[str, Any]], correct_cell_types: List[str], original_question: str) -> List[Dict[str, Any]]:
        """
        Fix cell type names in plan steps by mapping original question names to correct Neo4j names.
        
        For example: "Conventional memory CD4 T cells" → "CD4-positive memory T cell"
        """
        if not correct_cell_types:
            return steps
        
        # Create mapping from question text to correct Neo4j names
        cell_type_mapping = {}
        
        # Simple approach: try to map based on key terms
        for correct_name in correct_cell_types:
            # Look for partial matches in the original question
            if "regulatory" in original_question.lower() and "regulatory" in correct_name.lower():
                # Map variants of "Regulatory T cells" to "Regulatory T cell"
                cell_type_mapping["Regulatory T cells"] = correct_name
                cell_type_mapping["Regulatory T cell"] = correct_name
                cell_type_mapping["Tregs"] = correct_name
                
            elif "cd4" in original_question.lower() and "cd4" in correct_name.lower():
                # Map variants of "Conventional memory CD4 T cells" to "CD4-positive memory T cell"
                cell_type_mapping["Conventional memory CD4 T cells"] = correct_name
                cell_type_mapping["CD4+ T cells"] = correct_name
                cell_type_mapping["CD4 T cells"] = correct_name
        
        print(f"🔄 Cell type mapping: {cell_type_mapping}")
        
        # Apply mapping to all steps
        corrected_steps = []
        for step in steps:
            corrected_step = step.copy()
            
            # Fix cell type in parameters
            if "parameters" in corrected_step and "cell_type" in corrected_step["parameters"]:
                old_cell_type = corrected_step["parameters"]["cell_type"]
                if old_cell_type in cell_type_mapping:
                    new_cell_type = cell_type_mapping[old_cell_type]
                    corrected_step["parameters"]["cell_type"] = new_cell_type
                    print(f"🔄 Fixed step cell type: '{old_cell_type}' → '{new_cell_type}'")
                    
                    # Also update description and expected_outcome if they contain the old name
                    if "description" in corrected_step:
                        corrected_step["description"] = corrected_step["description"].replace(old_cell_type, new_cell_type)
                    if "expected_outcome" in corrected_step:
                        corrected_step["expected_outcome"] = corrected_step["expected_outcome"].replace(old_cell_type, new_cell_type)
            
            corrected_steps.append(corrected_step)
        
        return corrected_steps
    
    def _update_analysis_steps_for_discovered_types(self, analysis_steps: List[Dict[str, Any]], 
                                                   needed_cell_types: List[str], 
                                                   available_cell_types: List[str]) -> List[Dict[str, Any]]:
        """
        Update analysis steps to use discovered specific cell types instead of parent types.
        
        For example, if we have an analysis step for "Immune cell" but we're discovering 
        "Regulatory T cell" and "CD4-positive memory T cell", create separate analysis 
        steps for each discovered type.
        """
        updated_steps = []
        
        for step in analysis_steps:
            step_cell_type = step.get("parameters", {}).get("cell_type")
            
            # Check if this step uses a parent cell type that we're discovering from
            if step_cell_type in available_cell_types:
                # This step uses a parent type - check if we're discovering specific types from it
                types_to_discover_from_parent = [
                    needed_type for needed_type in needed_cell_types 
                    if needed_type not in available_cell_types
                ]
                
                if types_to_discover_from_parent:
                    # Create analysis steps for each specific discovered type
                    for discovered_type in types_to_discover_from_parent:
                        # Check if this type will actually be discovered (not in unavailable list)
                        # We'll assume it might be discovered for now - validation will catch failures
                        
                        updated_step = step.copy()
                        updated_step["parameters"] = step["parameters"].copy()
                        updated_step["parameters"]["cell_type"] = discovered_type
                        
                        # Update description to reflect the specific cell type
                        original_desc = step.get("description", "")
                        updated_desc = original_desc.replace(step_cell_type, discovered_type)
                        updated_step["description"] = updated_desc
                        
                        updated_steps.append(updated_step)
                        print(f"🔄 Updated analysis step: {step.get('function_name')}({step_cell_type}) → {step.get('function_name')}({discovered_type})")
                else:
                    # No discovery needed, keep original step
                    updated_steps.append(step)
            else:
                # This step doesn't use a parent type, keep as-is
                updated_steps.append(step)
        
        return updated_steps
    
    def _update_remaining_steps_with_discovered_types(self, state: ChatState, validation_result: Dict[str, Any]) -> None:
        """
        Update remaining analysis steps to use actually discovered cell types instead of unavailable ones.
        
        This is called after validation to ensure subsequent analysis steps use real discovered types.
        """
        execution_plan = state.get("execution_plan", {})
        steps = execution_plan.get("steps", [])
        current_step_index = state.get("current_step_index", 0)
        
        # Get the mapping of what was expected vs what was actually found
        expected_children = validation_result.get("missing_children", []) + validation_result.get("found_children", [])
        found_children = validation_result.get("found_children", [])
        missing_children = validation_result.get("missing_children", [])
        
        print(f"🔄 Updating remaining steps: expected={expected_children}, found={found_children}, missing={missing_children}")
        
        # Build replacement mapping
        replacement_mapping = {}
        
        # For missing children, add them to unavailable (already done above)
        # For found children that are subtypes, we need to map the originals to the subtypes
        processed_parent = None
        for i in range(current_step_index):
            step = steps[i]
            if step.get("step_type") == "validation" and step.get("parameters", {}).get("processed_parent"):
                processed_parent = step["parameters"]["processed_parent"]
                break
        
        if processed_parent:
            # Map any analysis steps that reference missing expected children to use found subtypes instead
            for missing_child in missing_children:
                # If we expected "Regulatory T cell" but found subtypes like "Memory T cell", 
                # we need to create analysis steps for the found subtypes
                for found_child in found_children:
                    replacement_mapping[missing_child] = found_child
                    break  # Use first found subtype as replacement
        
        # Update remaining steps
        updated_count = 0
        for i in range(current_step_index + 1, len(steps)):
            step = steps[i]
            step_cell_type = step.get("parameters", {}).get("cell_type")
            
            if step_cell_type in missing_children:
                # This step references a missing cell type
                if step_cell_type in replacement_mapping:
                    # Replace with discovered subtype
                    new_cell_type = replacement_mapping[step_cell_type]
                    steps[i]["parameters"]["cell_type"] = new_cell_type
                    
                    # Update description
                    old_desc = step.get("description", "")
                    new_desc = old_desc.replace(step_cell_type, new_cell_type)
                    steps[i]["description"] = new_desc
                    
                    print(f"🔄 Updated step {i+1}: {step.get('function_name')}({step_cell_type}) → {step.get('function_name')}({new_cell_type})")
                    updated_count += 1
                else:
                    # Mark this step to be skipped (cell type unavailable)
                    steps[i]["skip_reason"] = f"Cell type '{step_cell_type}' not discovered"
                    print(f"⏭️ Marked step {i+1} for skipping: {step.get('function_name')}({step_cell_type})")
        
        if updated_count > 0:
            print(f"✅ Updated {updated_count} remaining analysis steps to use discovered cell types")

    def _skip_unavailable_cell_steps(self, plan: Dict[str, Any], unavailable_cell_types: List[str]) -> Dict[str, Any]:
        """Skip steps for unavailable cell types"""
        if not unavailable_cell_types:
            return plan
        
        # Filter out steps that target unavailable cell types
        filtered_steps = []
        for step in plan.get("steps", []):
            target_cell_type = step.get("target_cell_type")
            if target_cell_type and target_cell_type in unavailable_cell_types:
                print(f"⏭️ Skipping step for unavailable cell type: {target_cell_type}")
                continue
            filtered_steps.append(step)
        
        plan["steps"] = filtered_steps
        
        # Update plan summary if steps were skipped
        if len(filtered_steps) < len(plan.get("steps", [])):
            plan["plan_summary"] += f" (skipped {len(plan.get('steps', [])) - len(filtered_steps)} steps for unavailable cell types)"
        
        return plan

    # Helper methods that may be needed by the extracted methods
    def _summarize_functions(self, functions: List[Dict]) -> str:
        """Summarize available functions for planning context"""
        if not functions:
            return "No functions available"
        
        summary = []
        for func in functions:
            name = func.get("name", "unknown")
            description = func.get("description", "").split(".")[0]  # First sentence only
            summary.append(f"- {name}: {description}")
        
        return "\n".join(summary)


    def _update_available_cell_types_from_result(self, state: ChatState, result: Any) -> None:
        """
        Update available_cell_types with newly discovered cell types from process_cells result.
        """
        if not result:
            return
        
        # Extract discovered cell types from the result
        discovered_types = []
        
        try:
            # The process_cells result should contain information about discovered cell types
            # Check if result is a dict with discovered types
            if isinstance(result, dict):
                if "discovered_cell_types" in result:
                    discovered_types = result["discovered_cell_types"]
                elif "new_cell_types" in result:
                    discovered_types = result["new_cell_types"]
            
            # If no explicit discovered types, try to extract from string result
            elif isinstance(result, str):
                # Look for patterns like "✅ Discovered new cell type: T cell"
                import re
                discoveries = re.findall(r"✅ Discovered new cell type: ([^\\n]+)", result)
                discovered_types.extend(discoveries)
            
            # Also check the hierarchy manager for newly available types
            if self.hierarchy_manager and hasattr(self.hierarchy_manager, 'get_available_cell_types'):
                current_available = self.hierarchy_manager.get_available_cell_types()
                original_available = set(state.get("available_cell_types", []))
                newly_available = set(current_available) - original_available
                discovered_types.extend(list(newly_available))
            
            # Update state with newly discovered types
            if discovered_types:
                current_available = set(state.get("available_cell_types", []))
                for cell_type in discovered_types:
                    if cell_type and cell_type not in current_available:
                        current_available.add(cell_type)
                        print(f"🧬 Added newly discovered cell type to available list: '{cell_type}'")
                
                state["available_cell_types"] = list(current_available)
                print(f"✅ Updated available cell types: {len(current_available)} types now available")
        
        except Exception as e:
            print(f"⚠️ Error updating available cell types: {e}")
            # Continue without failing

    def validate_processing_results(self, processed_parent: str, expected_children: List[str]) -> Dict[str, Any]:
        """Validate that process_cells discovered the expected cell types"""
        if not self.adata:
            return {"status": "error", "message": "No adata available"}
        
        current_cell_types = set(self.adata.obs["cell_type"].unique())
        found_children = []
        missing_children = []
        
        for expected_child in expected_children:
            # Check exact match first
            if expected_child in current_cell_types:
                found_children.append(expected_child)
                print(f"✅ Exact match found: '{expected_child}'")
            else:
                # Check if any discovered types are subtypes of the expected type using hierarchy
                subtypes_found = []
                if self.hierarchy_manager:
                    for available_type in current_cell_types:
                        try:
                            relation = self.hierarchy_manager.get_cell_type_relation(available_type, expected_child)
                            if relation.name == "DESCENDANT":
                                subtypes_found.append(available_type)
                        except:
                            continue
                
                if subtypes_found:
                    found_children.extend(subtypes_found)
                    print(f"✅ Subtype validation: '{expected_child}' satisfied by subtypes: {subtypes_found}")
                else:
                    missing_children.append(expected_child)
                    print(f"❌ Missing expected cell type: '{expected_child}' (no exact match or valid subtypes)")
        
        if missing_children:
            print(f"⚠️ Validation Warning: Expected children not found: {missing_children}")
            print(f"   Available cell types: {sorted(current_cell_types)}")
            
            # Try to suggest alternatives
            suggestions = []
            for missing in missing_children:
                for available in current_cell_types:
                    if self.hierarchy_manager and self.hierarchy_manager.get_cell_type_relation(missing, available).name in ["ANCESTOR", "DESCENDANT", "SIBLING"]:
                        suggestions.append(f"'{missing}' → '{available}'")
            
            return {
                "status": "partial_success" if found_children else "warning",
                "message": f"Found {len(found_children)}/{len(expected_children)} expected cell types. Missing: {missing_children}",
                "found_children": found_children,
                "missing_children": missing_children,
                "suggestions": suggestions,
                "available_types": list(current_cell_types)
            }
        else:
            return {
                "status": "success",
                "message": f"All {len(expected_children)} expected cell types found successfully",
                "found_children": found_children,
                "missing_children": [],
                "available_types": list(current_cell_types)
            }


    def _enhance_enrichment_step(self, step: Dict[str, Any], message: str) -> Dict[str, Any]:
        """
        Enhance enrichment analysis step using EnrichmentChecker intelligence.
        
        Args:
            step: Enrichment step from planner with minimal parameters
            message: Original user message for context
            
        Returns:
            Enhanced step with optimal analyses and parameters
        """
        if not self.enrichment_checker_available:
            print("⚠️ EnrichmentChecker unavailable, using step as-is")
            return step
        
        try:
            # Extract pathway terms from user message for context
            pathway_terms = self._extract_pathway_terms_from_message(message)
            
            print(f"🧬 Enhancing enrichment step with user context: {pathway_terms or 'general analysis'}")
            
            # Create plan step for EnrichmentChecker
            enrichment_plan_step = {
                "function_name": "perform_enrichment_analyses",
                "parameters": {
                    "cell_type": step.get("parameters", {}).get("cell_type", "unknown"),
                    "pathway_include": pathway_terms  # Use extracted terms or None
                }
            }
            
            # Get pathway enhancement
            enhanced_plan = self.enrichment_checker.enhance_enrichment_plan(enrichment_plan_step)
            
            # Apply enhancements to the original step
            enhanced_step = step.copy()
            if "parameters" not in enhanced_step:
                enhanced_step["parameters"] = {}
                
            # Add EnrichmentChecker recommendations
            enhanced_step["parameters"].update({
                "analyses": enhanced_plan["parameters"].get("analyses", ["gsea"])
            })
            
            # Only add gene_set_library if it's provided (for GSEA)
            if enhanced_plan["parameters"].get("gene_set_library"):
                enhanced_step["parameters"]["gene_set_library"] = enhanced_plan["parameters"]["gene_set_library"]
            
            # Update description
            validation_details = enhanced_plan.get("validation_details", {})
            if validation_details:
                enhanced_step["description"] += f" (Enhanced with pathway intelligence: {validation_details.get('total_recommendations', 0)} recommendations)"
            
            print(f"✅ Enhanced enrichment step: {enhanced_step['parameters']}")
            
            # Log enhancement statistics
            enhancement_data = {
                "recommended_analyses": enhanced_step["parameters"]["analyses"],
                "gene_set_library": enhanced_step["parameters"].get("gene_set_library"),
                "pathway_terms": pathway_terms,
                "validation_details": validation_details
            }
            self._log_pathway_enhancement_stats(enhancement_data)
            
            return enhanced_step
            
        except Exception as e:
            print(f"⚠️ Enrichment step enhancement failed: {e}")
            return step  # Return original step on failure

    def _extract_pathway_terms_from_message(self, message: str) -> str:
        """Extract pathway terms from user message using simple keyword matching."""
        # Simple extraction - could be enhanced with LLM in future
        pathway_keywords = [
            "interferon", "ifn", "apoptosis", "cell cycle", "inflammation",
            "immune", "signaling", "metabolism", "proliferation", "differentiation",
            "pathway", "response", "activation", "inhibition", "regulation",
            "stimulated", "enrichment", "enriched", "analysis", "pathways"
        ]
        
        message_lower = message.lower()
        found_terms = []
        
        for keyword in pathway_keywords:
            if keyword in message_lower:
                found_terms.append(keyword)
        
        return " ".join(found_terms) if found_terms else None

    def _enhance_all_enrichment_steps(self, plan_data: Dict[str, Any], message: str) -> Dict[str, Any]:
        """Enhance all enrichment analysis steps in the plan using EnrichmentChecker."""
        print(f"🧬 ENRICHMENT ENHANCER: Processing {len(plan_data.get('steps', []))} steps")
        enhanced_steps = []
        
        for i, step in enumerate(plan_data.get("steps", [])):
            if step.get("function_name") == "perform_enrichment_analyses":
                print(f"🧬 ENRICHMENT ENHANCER: Enhancing step {i+1}: {step.get('parameters', {}).get('cell_type', 'unknown')}")
                # Enhance this enrichment step
                enhanced_step = self._enhance_enrichment_step(step, message)
                enhanced_steps.append(enhanced_step)
            else:
                # Keep non-enrichment steps as-is
                enhanced_steps.append(step)
        
        plan_data["steps"] = enhanced_steps
        return plan_data


    def _log_pathway_enhancement_stats(self, enhancement_data: Dict[str, Any]) -> None:
        """Log pathway enhancement statistics for monitoring."""
        if not enhancement_data:
            return
        
        validation_details = enhancement_data.get("validation_details", {})
        
        print("📊 PATHWAY ENHANCEMENT STATS:")
        print(f"   • Recommended analyses: {enhancement_data['recommended_analyses']}")
        print(f"   • Confidence: {enhancement_data.get('confidence', 0.0):.2f}")
        print(f"   • Total recommendations: {validation_details.get('total_recommendations', 0)}")
        print(f"   • Pathway matches: {len(validation_details.get('pathway_matches', []))}")

    def step_evaluator_node(self, state: ChatState) -> ChatState:
        """
        Version 2: Intelligent Step-by-Step Evaluator with Proactive Problem Solving
        
        Analyzes each completed step and takes context-aware actions:
        - For process_cells: Validates discovered types and updates remaining plan
        - For failed analyses: Records errors for final response
        - For successful steps: Logs progress and insights
        """
        
        print("🧠 INTELLIGENT EVALUATOR V2: Starting context-aware evaluation...")
        
        # Get the most recent execution
        execution_history = state.get("execution_history", [])
        if not execution_history:
            print("⚠️ No execution history found for evaluation")
            state["last_step_evaluation"] = {"status": "no_history", "critical_failure": False}
            return state
        
        last_execution = execution_history[-1]
        step_index = last_execution.get("step_index", -1)
        success = last_execution.get("success", False)
        
        # Get function name from step data (more reliable)
        step_data = last_execution.get("step", {})
        function_name = step_data.get('function_name', 'unknown')
        
        print(f"🧠 Evaluating step {step_index + 1}: {function_name} ({'✅ Success' if success else '❌ Failed'})")
        
        # Function-specific intelligent evaluation
        evaluation_result = self._intelligent_function_evaluation(last_execution, state)
        
        # Store evaluation results
        state["last_step_evaluation"] = evaluation_result
        
        # Build evaluation history
        step_eval_history = state.get("step_evaluation_history", [])
        step_eval_history.append(evaluation_result)
        state["step_evaluation_history"] = step_eval_history
        
        print(f"✅ INTELLIGENT EVALUATOR V2: Evaluation complete for step {step_index + 1}")
        return state
    
    def _intelligent_function_evaluation(self, execution: Dict[str, Any], state: ChatState) -> Dict[str, Any]:
        """
        Context-aware evaluation that takes different actions based on function type and result.
        """
        step_data = execution.get("step", {})
        function_name = step_data.get('function_name', 'unknown')
        success = execution.get("success", False)
        result = execution.get("result")
        step_index = execution.get("step_index", -1)
        
        evaluation = {
            "step_index": step_index,
            "function_name": function_name,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "evaluation_version": "v2_intelligent_proactive",
            "actions_taken": []
        }
        
        if function_name == "process_cells":
            return self._evaluate_process_cells_intelligently(execution, state, evaluation)
        elif function_name in ["dea_split_by_condition", "perform_enrichment_analyses"]:
            return self._evaluate_analysis_function_intelligently(execution, state, evaluation)
        elif function_name.startswith("display_"):
            return self._evaluate_visualization_intelligently(execution, state, evaluation)
        else:
            return self._evaluate_generic_function(execution, state, evaluation)
    
    def _evaluate_process_cells_intelligently(self, execution: Dict[str, Any], 
                                            state: ChatState, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligent evaluation of process_cells: Validates discovered types and updates plan.
        """
        step_data = execution.get("step", {})
        result = execution.get("result")
        success = execution.get("success", False)
        cell_type = step_data.get("parameters", {}).get("cell_type")
        
        print(f"🧬 PROCESS_CELLS ANALYSIS: Evaluating {cell_type} processing...")
        
        if not success:
            evaluation["process_cells_evaluation"] = {
                "status": "failed",
                "message": "Process cells execution failed",
                "cell_type": cell_type
            }
            # Record error for final response
            self._record_execution_error(state, f"Failed to process {cell_type}", execution)
            return evaluation
        
        # Extract what was actually discovered
        current_cell_types = set(self.adata.obs["cell_type"].unique()) if self.adata else set()
        
        # Check if we have expected children for this process_cells step
        expected_children = step_data.get("expected_children", [])
        
        if expected_children:
            # Validate discovered types
            validation_result = self._validate_discovered_types(expected_children, current_cell_types)
            
            evaluation["process_cells_evaluation"] = {
                "status": "validated",
                "cell_type": cell_type,
                "expected_children": expected_children,
                "found_children": validation_result["found_children"],
                "missing_children": validation_result["missing_children"],
                "validation_status": validation_result["status"]
            }
            
            # PROACTIVE ACTION: Update remaining steps based on what was actually discovered
            if validation_result["missing_children"]:
                actions = self._update_plan_for_missing_cell_types(
                    state, validation_result["missing_children"]
                )
                evaluation["actions_taken"].extend(actions)
                
                # Add missing types to unavailable list
                current_unavailable = state.get("unavailable_cell_types", [])
                state["unavailable_cell_types"] = list(set(current_unavailable + validation_result["missing_children"]))
                
                print(f"📋 Updated unavailable cell types: +{validation_result['missing_children']}")
            
            # Update available cell types
            state["available_cell_types"] = list(current_cell_types)
            
        else:
            # No expected children specified, just log what was discovered
            evaluation["process_cells_evaluation"] = {
                "status": "completed",
                "cell_type": cell_type,
                "discovered_types": list(current_cell_types)
            }
        
        return evaluation
    
    def _validate_discovered_types(self, expected_children: list, current_cell_types: set) -> Dict[str, Any]:
        """Validate if expected cell types were discovered (with hierarchy awareness)."""
        found_children = []
        missing_children = []
        
        for expected_child in expected_children:
            if expected_child in current_cell_types:
                found_children.append(expected_child)
                print(f"✅ Exact match found: '{expected_child}'")
            else:
                # Check if any discovered types are subtypes using hierarchy
                subtypes_found = []
                if self.hierarchy_manager:
                    for available_type in current_cell_types:
                        try:
                            relation = self.hierarchy_manager.get_cell_type_relation(available_type, expected_child)
                            if relation.name == "DESCENDANT":
                                subtypes_found.append(available_type)
                        except:
                            continue
                
                if subtypes_found:
                    found_children.extend(subtypes_found)
                    print(f"✅ Subtype validation: '{expected_child}' satisfied by subtypes: {subtypes_found}")
                else:
                    missing_children.append(expected_child)
                    print(f"❌ Missing expected cell type: '{expected_child}' (no exact match or valid subtypes)")
        
        if missing_children:
            status = "partial_success" if found_children else "failed"
        else:
            status = "success"
        
        return {
            "status": status,
            "found_children": found_children,
            "missing_children": missing_children,
            "available_types": list(current_cell_types)
        }
    
    def _update_plan_for_missing_cell_types(self, state: ChatState, 
                                          missing_types: list) -> list:
        """
        Skip all steps referencing missing cell types.
        No alternatives - just transparently skip what's unavailable.
        """
        actions_taken = []
        execution_plan = state.get("execution_plan", {})
        steps = execution_plan.get("steps", [])
        current_step_index = state.get("current_step_index", 0)
        
        print(f"🔧 PLAN UPDATE: Skipping steps for missing types {missing_types}")
        print(f"   Current step index: {current_step_index}, Total steps: {len(steps)}")
        
        steps_skipped = 0
        
        for i in range(len(steps)):
            step = steps[i]
            step_cell_type = step.get("parameters", {}).get("cell_type")
            step_description = step.get("description", "")
            
            # Check if step references missing cell types in parameters OR description
            references_missing_type = False
            if step_cell_type and step_cell_type in missing_types:
                references_missing_type = True
                missing_ref = step_cell_type
            else:
                # Check if any missing type is mentioned in the description
                for missing_type in missing_types:
                    if missing_type in step_description:
                        references_missing_type = True
                        missing_ref = missing_type
                        break
            
            # Skip if this step doesn't reference any missing cell types
            if not references_missing_type:
                continue
            
            # Skip if step is already marked for skipping
            if step.get("skip_reason"):
                continue
                
            print(f"🔍 Found step {i+1} referencing missing cell type '{missing_ref}': {step.get('function_name')}")
            
            # Skip ALL steps referencing missing cell types, including current step
            steps[i]["skip_reason"] = f"Cell type '{missing_ref}' was not discovered and is unavailable"
            actions_taken.append(f"Skipped step {i+1}: {step.get('function_name')} (references {missing_ref})")
            steps_skipped += 1
            print(f"⏭️ Skipped step {i+1}: {step.get('function_name')} (references {missing_ref})")
        
        if steps_skipped > 0:
            print(f"✅ Plan update complete: {steps_skipped} steps marked for skipping")
        else:
            print("✅ No steps needed skipping")
        
        return actions_taken
    
    
    def _evaluate_analysis_function_intelligently(self, execution: Dict[str, Any], 
                                                state: ChatState, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligent evaluation of analysis functions (DEA, enrichment)."""
        success = execution.get("success", False)
        step_data = execution.get("step", {})
        function_name = step_data.get('function_name', 'unknown')
        cell_type = step_data.get("parameters", {}).get("cell_type")
        
        if not success:
            error_msg = execution.get("error", "Unknown error")
            print(f"📊 ANALYSIS ERROR: {function_name}({cell_type}) failed: {error_msg}")
            
            # Record meaningful error for final response
            self._record_execution_error(state, f"{function_name} failed for {cell_type}: {error_msg}", execution)
            
            evaluation["analysis_evaluation"] = {
                "status": "failed",
                "function": function_name,
                "cell_type": cell_type,
                "error": error_msg
            }
        else:
            print(f"✅ ANALYSIS SUCCESS: {function_name}({cell_type}) completed")
            evaluation["analysis_evaluation"] = {
                "status": "success",
                "function": function_name,
                "cell_type": cell_type
            }
        
        return evaluation
    
    def _evaluate_visualization_intelligently(self, execution: Dict[str, Any], 
                                            state: ChatState, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligent evaluation of visualization functions."""
        success = execution.get("success", False)
        step_data = execution.get("step", {})
        function_name = step_data.get('function_name', 'unknown')
        
        evaluation["visualization_evaluation"] = {
            "status": "success" if success else "failed",
            "function": function_name
        }
        
        return evaluation
    
    def _evaluate_generic_function(self, execution: Dict[str, Any], 
                                 state: ChatState, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Generic evaluation for other function types."""
        success = execution.get("success", False)
        evaluation["generic_evaluation"] = {"status": "success" if success else "failed"}
        return evaluation
    
    def _record_execution_error(self, state: ChatState, error_message: str, execution: Dict[str, Any]) -> None:
        """Record meaningful execution errors for final LLM response."""
        if "execution_errors" not in state:
            state["execution_errors"] = []
        
        state["execution_errors"].append({
            "step_index": execution.get("step_index", -1),
            "function_name": execution.get("step", {}).get("function_name", "unknown"),
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"📝 Recorded execution error: {error_message}")

    def _evaluate_single_step_v1(self, execution: Dict[str, Any], state: ChatState) -> Dict[str, Any]:
        """Comprehensive step evaluation leveraging fixed storage format"""
        
        # FIX: The execution history structure stores data directly, not in 'step' sub-dict
        result = execution.get("result")
        result_summary = execution.get("result_summary")  # This might be the actual result
        result_type = execution.get("result_type", "text")
        success = execution.get("success", False)
        function_name = execution.get("function_name", "")
        step_index = execution.get("step_index", -1)
        
        # Use result_summary if result is None (common case)
        if result is None and result_summary is not None:
            result = result_summary
        
        
        # Base evaluation structure
        evaluation = {
            "step_index": step_index,
            "function_name": function_name,
            "success": success,
            "result_type": result_type,
            "timestamp": datetime.now().isoformat(),
            "evaluation_version": "v1_checking_only",
            "critical_failure": False
        }
        
        if not success:
            # Evaluate failure
            evaluation.update(self._evaluate_step_failure(execution, state))
        else:
            # Evaluate success - use appropriate method based on result type
            if result_type == "structured":
                evaluation.update(self._evaluate_structured_result(function_name, result, state))
            else:
                evaluation.update(self._evaluate_text_result(function_name, result, state))
            
            # Function-specific evaluations
            evaluation.update(self._evaluate_function_specific(function_name, result, result_type, state))
        
        return evaluation

    def _evaluate_step_failure(self, execution: Dict[str, Any], state: ChatState) -> Dict[str, Any]:
        """Evaluate failed execution steps"""
        error_msg = execution.get("error", "Unknown error")
        function_name = execution.get("function_name", "")
        
        # Determine if this is a critical failure
        critical_errors = [
            "ImportError", "ModuleNotFoundError", "AttributeError",
            "FileNotFoundError", "PermissionError"
        ]
        
        is_critical = any(critical_error in error_msg for critical_error in critical_errors)
        
        return {
            "failure_evaluation": {
                "error_message": error_msg,
                "error_category": "system_error" if is_critical else "analysis_error",
                "suggested_action": "abort_workflow" if is_critical else "continue_with_caution"
            },
            "critical_failure": is_critical
        }

    def _evaluate_structured_result(self, function_name: str, result: Any, state: ChatState) -> Dict[str, Any]:
        """Evaluate structured results from critical functions"""
        return {
            "data_quality": "high_structured_access",
            "data_completeness": "full",
            "parsing_issues": None
        }

    def _evaluate_text_result(self, function_name: str, result: Any, state: ChatState) -> Dict[str, Any]:
        """Evaluate legacy text results"""
        result_length = len(str(result)) if result else 0
        
        return {
            "data_quality": "legacy_text_parsing",
            "data_completeness": "limited" if result_length >= 500 else "full",
            "parsing_issues": "truncation_likely" if result_length >= 500 else None
        }

    def _evaluate_function_specific(self, function_name: str, result: Any, 
                                   result_type: str, state: ChatState) -> Dict[str, Any]:
        """Function-specific evaluation logic"""
        
        if function_name == "perform_enrichment_analyses":
            return self._evaluate_enrichment_analysis(result, result_type, state)
        
        elif function_name == "process_cells":
            return self._evaluate_process_cells(result, result_type, state)
        
        elif function_name == "dea_split_by_condition":
            return self._evaluate_dea_analysis(result, result_type, state)
        
        elif function_name.startswith("display_"):
            return self._evaluate_visualization(function_name, result, result_type, state)
        
        return {"function_evaluation": "generic_success"}

    def _evaluate_enrichment_analysis(self, result: Any, result_type: str, state: ChatState) -> Dict[str, Any]:
        """Detailed enrichment analysis evaluation"""
        
        if result_type == "structured":
            # Direct structured access - no parsing needed!
            pathway_counts = {}
            total_pathways = 0
            significant_methods = []
            
            # Add type check before dictionary access
            if not isinstance(result, dict):
                return {"error": f"Expected dict but got {type(result)}"}
            
            for analysis_type in ["go", "kegg", "reactome", "gsea"]:
                if analysis_type in result and isinstance(result[analysis_type], dict):
                    count = result[analysis_type].get("total_significant", 0)
                    pathway_counts[analysis_type] = count
                    total_pathways += count
                    
                    if count > 0:
                        significant_methods.append(analysis_type.upper())
            
            evaluation = {
                "enrichment_evaluation": {
                    "pathway_counts": pathway_counts,
                    "total_significant_pathways": total_pathways,
                    "successful_methods": significant_methods,
                    "method_count": len(significant_methods),
                    "data_quality": "high_structured_access",
                    "top_pathways": {
                        method.lower(): result.get(method.lower(), {}).get("top_terms", [])[:3]
                        for method in significant_methods
                    }
                }
            }
            
            # Quality assessment
            if total_pathways == 0:
                evaluation["enrichment_evaluation"]["concerns"] = [
                    "No significant pathways found in any method",
                    "Consider adjusting parameters or trying different approaches"
                ]
            elif total_pathways > 100:
                evaluation["enrichment_evaluation"]["highlights"] = [
                    f"Rich pathway enrichment found ({total_pathways} total)",
                    f"Successful methods: {', '.join(significant_methods)}"
                ]
            
            return evaluation
        
        else:
            # Legacy text parsing (will be deprecated after storage fix)
            return {"enrichment_evaluation": {"data_quality": "legacy_text_parsing"}}

    def _evaluate_process_cells(self, result: Any, result_type: str, state: ChatState) -> Dict[str, Any]:
        """Evaluate process_cells analysis"""
        if isinstance(result, str):
            discovered = "discovered" in result.lower()
            return {
                "process_cells_evaluation": {
                    "discovery_status": "successful" if discovered else "no_new_types",
                    "discovered_cell_types": []  # Would need parsing for actual types
                }
            }
        return {"process_cells_evaluation": {"status": "completed"}}

    def _evaluate_dea_analysis(self, result: Any, result_type: str, state: ChatState) -> Dict[str, Any]:
        """Evaluate DEA analysis"""
        return {"dea_evaluation": {"status": "completed"}}

    def _evaluate_visualization(self, function_name: str, result: Any, result_type: str, state: ChatState) -> Dict[str, Any]:
        """Evaluate visualization generation"""
        html_length = len(result) if isinstance(result, str) else 0
        contains_html = isinstance(result, str) and ('<div' in result or '<html' in result)
        
        return {
            "visualization_evaluation": {
                "type": function_name,
                "html_generated": contains_html,
                "html_length": html_length,
                "status": "success" if contains_html else "no_html_content"
            }
        }

    def _log_step_evaluation(self, evaluation: Dict[str, Any]) -> None:
        """Comprehensive evaluation logging for monitoring and debugging"""
        
        step_index = evaluation.get("step_index", -1)
        function_name = evaluation.get("function_name", "unknown")
        success = evaluation.get("success", False)
        result_type = evaluation.get("result_type", "unknown")
        
        print(f"\n{'='*60}")
        print(f"📊 STEP EVALUATION REPORT - Step {step_index + 1}")
        print(f"{'='*60}")
        print(f"Function: {function_name}")
        print(f"Success: {'✅' if success else '❌'} ({success})")
        print(f"Result Type: {result_type}")
        print(f"Timestamp: {evaluation.get('timestamp', 'unknown')}")
        
        if not success:
            print(f"❌ FAILURE ANALYSIS:")
            failure_eval = evaluation.get("failure_evaluation", {})
            print(f"   Error Type: {failure_eval.get('error_category', 'Unknown')}")
            print(f"   Critical: {'🚨 YES' if evaluation.get('critical_failure') else '⚠️ No'}")
            
        else:
            print(f"✅ SUCCESS ANALYSIS:")
            
            # Function-specific reporting
            if "enrichment_evaluation" in evaluation:
                enrich_eval = evaluation["enrichment_evaluation"]
                pathway_counts = enrich_eval.get("pathway_counts", {})
                
                print(f"   🧬 ENRICHMENT RESULTS:")
                for method, count in pathway_counts.items():
                    status = "✅" if count > 0 else "⭕"
                    print(f"      {status} {method.upper()}: {count} pathways")
                
                total = enrich_eval.get("total_significant_pathways", 0)
                print(f"   📊 Total Significant: {total} pathways")
                print(f"   🎯 Data Quality: {enrich_eval.get('data_quality', 'unknown')}")
                
                if enrich_eval.get("top_pathways"):
                    print(f"   🔝 Sample Pathways:")
                    for method, pathways in enrich_eval.get("top_pathways", {}).items():
                        if pathways:
                            print(f"      {method}: {', '.join(pathways[:2])}")
            
            elif "process_cells_evaluation" in evaluation:
                process_eval = evaluation["process_cells_evaluation"]
                discovered = process_eval.get("discovered_cell_types", [])
                print(f"   🧬 PROCESS CELLS RESULTS:")
                print(f"      Discovered: {len(discovered)} new cell types")
                for cell_type in discovered[:3]:  # Show first 3
                    print(f"         - {cell_type}")
        
        print(f"{'='*60}\n")

    def _call_llm(self, prompt: str, model_name: str = "gpt-4o") -> str:
        """Simple LLM call for analysis tasks using LangChain"""
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage
            
            # Create messages in LangChain format
            messages = [
                SystemMessage(content="You are a helpful assistant that analyzes questions and generates search queries."),
                HumanMessage(content=prompt)
            ]
            
            # Initialize model
            model = ChatOpenAI(
                model=model_name,
                temperature=0.1,  # Low temperature for consistency
                max_tokens=500   # Reasonable limit for search queries
            )
            
            # Get response
            response = model.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            print(f"⚠️ LLM call failed: {e}")
            return "[]"  # Safe default for JSON parsing
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question type using a smaller LLM for efficiency"""
        
        classification_prompt = f"""Classify the following biology question into ONE category:

Question: "{question}"

Categories:
1. canonical_markers - Questions about well-known, established cell type markers (e.g., "What are canonical markers for...", "differentiate X from Y using markers")
2. pathway_analysis - Questions about biological pathways, processes, or functional enrichment (e.g., "What pathways are enriched...", "biological processes in...")
3. gene_expression - Questions about specific gene expression changes (e.g., "Is gene X upregulated...", "expression of Y in condition Z")
4. cell_abundance - Questions about cell type counts or proportions (e.g., "How many X cells...", "proportion of Y cells")
5. general_comparison - General comparison questions not fitting above categories

Return ONLY the category name, nothing else.
Category:"""
        
        try:
            # Use smaller model for efficiency
            response = self._call_llm(classification_prompt, model_name="gpt-4o-mini")
            category = response.strip().lower()
            
            # Validate category
            valid_categories = ["canonical_markers", "pathway_analysis", "gene_expression", 
                              "cell_abundance", "general_comparison"]
            
            if category not in valid_categories:
                print(f"⚠️ Invalid category '{category}', defaulting to general_comparison")
                return "general_comparison"
            
            print(f"📊 Question classified as: {category}")
            return category
            
        except Exception as e:
            print(f"⚠️ Question classification failed: {e}")
            return "general_comparison"
    
    def _post_execution_evaluation(self, state: ChatState) -> Dict[str, Any]:
        """
        Cell-type specific LLM-powered gap analysis - FINAL APPROACH
        """
        print("🔍 Starting post-execution evaluation...")
        
        original_question = state["execution_plan"]["original_question"]
        
        # Step 1: Extract mentioned cell types using existing function
        mentioned_types = extract_cell_types_from_question(original_question, self.hierarchy_manager)
        
        # Classify question type for better result filtering
        question_type = self._classify_question_type(original_question)
        
        if not mentioned_types:
            print("📋 No specific cell types mentioned, skipping post-execution evaluation")
            return {"mentioned_cell_types": [], "supplementary_steps": [], "evaluation_complete": True}
        
        supplementary_steps = []
        evaluation_details = {}
        
        # Get unavailable cell types to skip them
        unavailable_cell_types = state.get("unavailable_cell_types", [])
        
        # Step 2: For each mentioned cell type, ask LLM what analyses are needed
        for cell_type in mentioned_types:
            # Skip cell types that were marked as unavailable during validation
            if cell_type in unavailable_cell_types:
                print(f"⏭️ Skipping post-execution evaluation for unavailable cell type: {cell_type}")
                continue
                
            print(f"🔍 Evaluating coverage for cell type: {cell_type}")
            
            # Step 2a: Get LLM recommendations for this specific cell type
            required_analyses = self._get_llm_analysis_requirements(original_question, cell_type)
            
            # Step 2b: Check what was actually performed for this cell type
            performed_analyses = self._get_performed_analyses_for_cell_type(state, cell_type)
            
            # Step 2c: Find gaps and generate steps
            missing_steps = self._generate_missing_steps_for_cell_type(
                cell_type, required_analyses, performed_analyses
            )
            
            supplementary_steps.extend(missing_steps)
            evaluation_details[cell_type] = {
                "required_analyses": required_analyses,
                "performed_analyses": performed_analyses,
                "missing_steps_count": len(missing_steps)
            }
            
            if missing_steps:
                print(f"📋 Found {len(missing_steps)} missing steps for {cell_type}")
            else:
                print(f"✅ Complete coverage for {cell_type}")
        
        print(f"🔍 Post-execution evaluation complete: {len(supplementary_steps)} total supplementary steps")
        
        # Add analysis relevance hints based on question type
        all_performed_analyses = {}
        for cell_type, details in evaluation_details.items():
            all_performed_analyses[cell_type] = details["performed_analyses"]
        
        analysis_relevance = self._get_analysis_relevance_hints(question_type, all_performed_analyses)
        
        return {
            "mentioned_cell_types": mentioned_types,
            "evaluation_details": evaluation_details,
            "supplementary_steps": supplementary_steps,
            "evaluation_complete": True,
            "question_type": question_type,
            "analysis_relevance": analysis_relevance
        }

    def _get_llm_analysis_requirements(self, original_question: str, cell_type: str) -> List[str]:
        """Ask LLM what analyses this specific cell type needs"""
        
        # Get question type for context
        question_type = self._classify_question_type(original_question)
        
        analysis_prompt = f"""You are analyzing what bioinformatics analyses are needed for a specific cell type.

                                User Question: "{original_question}"
                                Cell Type: "{cell_type}"
                                Question Type: {question_type}

                                Available analysis functions with detailed descriptions:

                                CORE ANALYSIS FUNCTIONS:
                                - perform_enrichment_analyses: Run enrichment analyses on DE genes for a cell type. Supports REACTOME (pathways), GO (gene ontology), KEGG (pathways), GSEA (gene set enrichment). Use for pathway analysis when user asks about biological processes, pathways, or gene function.

                                - dea_split_by_condition: Perform differential expression analysis (DEA) split by condition. Use when comparing conditions or when user asks about gene expression differences between experimental groups.

                                - compare_cell_counts: Compare cell counts between experimental conditions for specific cell types. Use when analyzing how cell type abundance differs across conditions (e.g., pre vs post treatment, healthy vs disease).

                                VISUALIZATION FUNCTIONS:
                                - display_enrichment_visualization: PREFERRED function for showing comprehensive enrichment visualization with both barplot and dotplot. Use after running enrichment analyses to visualize results.

                                - display_dotplot: Display dotplot for annotated results. Use when user wants to see gene expression patterns across cell types.

                                - display_cell_type_composition: Display cell type composition graph. Use when user wants to see the proportion of different cell types.

                                - display_umap: Display basic UMAP without cell type annotations. Use for basic dimensionality reduction visualization.

                                - display_processed_umap: Display UMAP with cell type annotations. Use when user wants to see cell type annotations on UMAP.

                                SEARCH FUNCTIONS:
                                - search_enrichment_semantic: Search all enrichment terms semantically to find specific pathways or biological processes. Use when user asks about specific pathways, terms, or biological processes that might not appear in standard top results.

                                - conversational_response: Provide conversational response without function calls. Use for greetings, clarifications, explanations, or when no analysis is needed.

                                Task: Determine which analyses are needed for {cell_type} to answer the user's question.

                                IMPORTANT: The cell type "{cell_type}" already exists in the dataset. DO NOT suggest process_cells for this cell type.

                                Consider based on question type:
                                1. Canonical markers questions (differentiate X from Y, markers of X) → Use dea_split_by_condition ONLY
                                2. Gene expression questions → Use dea_split_by_condition
                                3. Pathway/biological process questions → Use perform_enrichment_analyses + search_enrichment_semantic
                                4. Cell abundance questions → Use compare_cell_counts
                                5. Specific pathway search → Use search_enrichment_semantic

                                CRITICAL GUIDELINES FOR CANONICAL MARKERS:
                                - For canonical markers questions, ONLY use dea_split_by_condition
                                - DO NOT include enrichment analyses for marker identification
                                - DO NOT suggest process_cells for already-available cell types
                                - Visualization is optional for canonical markers

                                GENERAL GUIDELINES:
                                - Only include display_enrichment_visualization ONCE per cell type
                                - Return ONLY a valid JSON array of function names, nothing else

                                Example response for pathway question: ["perform_enrichment_analyses", "search_enrichment_semantic", "display_enrichment_visualization"]
                                Example response for all kinds of markers: ["dea_split_by_condition"]

                                Required analyses for {cell_type}:"""
        
        try:
            response = self._call_llm(analysis_prompt)
            print(f"🔍 LLM raw response for {cell_type}: '{response}' (length: {len(response)})")
            
            if not response or response.strip() == "":
                print(f"⚠️ Empty LLM response for {cell_type}, using fallback")
                return ["perform_enrichment_analyses"]
            
            # Try to extract JSON from response (handle cases with markdown code blocks)
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith("```json"):
                response = response.replace("```json", "").replace("```", "").strip()
            elif response.startswith("```"):
                response = response.replace("```", "").strip()
            
            # Look for JSON array pattern if response contains other text
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            
            required_analyses = json.loads(response)
            
            # Ensure we got a list
            if not isinstance(required_analyses, list):
                print(f"⚠️ LLM returned {type(required_analyses)} instead of list for {cell_type}")
                return ["perform_enrichment_analyses"]
            
            print(f"🧠 LLM recommends for {cell_type}: {required_analyses}")
            return required_analyses
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing failed for {cell_type}: {e}")
            print(f"⚠️ Raw response was: '{response}'")
            return ["perform_enrichment_analyses"]  # Safe fallback
        except Exception as e:
            print(f"⚠️ LLM analysis requirement failed for {cell_type}: {e}")
            return ["perform_enrichment_analyses"]  # Safe fallback

    def _get_performed_analyses_for_cell_type(self, state: ChatState, cell_type: str) -> List[str]:
        """Check what analyses were actually performed for a specific cell type"""
        
        performed_analyses = []
        unavailable_cell_types = state.get("unavailable_cell_types", [])
        
        # If this cell type is known to be unavailable, don't try to generate steps for it
        if cell_type in unavailable_cell_types:
            print(f"🚫 Cell type '{cell_type}' is unavailable - not generating missing steps")
            # Return dummy analysis to prevent post-execution from trying to add steps
            return ["analysis_skipped_cell_type_unavailable"]
        
        for ex in state["execution_history"]:
            # Check both successful executions AND skipped steps 
            if ex.get("success") or ex.get("skipped"):
                function_name = ex.get("step", {}).get("function_name")
                params = ex.get("step", {}).get("parameters", {})
                ex_cell_type = params.get("cell_type")
                
                if ex_cell_type == cell_type and function_name:
                    if ex.get("skipped"):
                        print(f"📋 Found skipped analysis for {cell_type}: {function_name}")
                    performed_analyses.append(function_name)
        
        # Remove duplicates while preserving order
        unique_performed = []
        for analysis in performed_analyses:
            if analysis not in unique_performed:
                unique_performed.append(analysis)
        
        print(f"📊 Actually performed for {cell_type}: {unique_performed}")
        return unique_performed

    def _generate_missing_steps_for_cell_type(self, cell_type: str, required_analyses: List[str], 
                                            performed_analyses: List[str]) -> List[Dict[str, Any]]:
        """Generate supplementary steps for missing analyses for a specific cell type"""
        
        missing_steps = []
        
        for required_function in required_analyses:
            if required_function not in performed_analyses:
                # Generate step using existing step format
                step = {
                    "step_type": "analysis" if not required_function.startswith("display_") else "visualization",
                    "function_name": required_function,
                    "parameters": {"cell_type": cell_type},
                    "description": f"Post-evaluation: {required_function} for {cell_type}",
                    "expected_outcome": f"Complete analysis coverage for {cell_type}",
                    "target_cell_type": cell_type
                }
                
                # Add specific parameters for visualization functions
                if required_function == "display_enrichment_visualization":
                    step["parameters"]["analysis"] = "gsea"  # Default analysis type
                
                missing_steps.append(step)
                print(f"🔧 Generated missing step: {required_function}({cell_type})")
        
        return missing_steps
    
    def _get_analysis_relevance_hints(self, question_type: str, performed_analyses: Dict[str, List[str]]) -> Dict[str, Any]:
        """Generate hints about which analyses are most relevant for the question type"""
        
        relevance_map = {
            "canonical_markers": {
                "primary": ["dea_split_by_condition"],
                "secondary": ["display_dotplot"]
            },
            "pathway_analysis": {
                "primary": ["perform_enrichment_analyses", "search_enrichment_semantic"],
                "secondary": ["display_enrichment_visualization"]
            },
            "gene_expression": {
                "primary": ["dea_split_by_condition"],
                "secondary": ["display_dotplot"]
            },
            "cell_abundance": {
                "primary": ["compare_cell_counts"],
                "secondary": ["display_processed_umap"]
            },
            "general_comparison": {
                "primary": ["dea_split_by_condition", "perform_enrichment_analyses"],
                "secondary": ["search_enrichment_semantic", "display_enrichment_visualization"]
            }
        }
        
        relevance = relevance_map.get(question_type, relevance_map["general_comparison"])
        
        # Create hints for response generator
        hints = {
            "question_type": question_type,
            "relevance_categories": relevance,
            "guidance": self._get_response_guidance(question_type)
        }
        
        print(f"📝 Generated relevance hints for {question_type} question")
        return hints
    
    def _get_response_guidance(self, question_type: str) -> str:
        """Get specific guidance for response generation based on question type"""
        
        guidance_map = {
            "canonical_markers": "Focus on well-established markers from literature. Prioritize DEA results showing marker genes. Avoid emphasizing enrichment analysis unless directly relevant to marker function.",
            "pathway_analysis": "Emphasize enrichment analysis results, biological processes, and pathway information. DEA results support pathway findings.",
            "gene_expression": "Focus on specific gene expression values and fold changes from DEA. Show specific gene names and statistical significance.",  
            "cell_abundance": "Prioritize cell count comparisons and composition visualizations. Focus on quantitative differences between conditions.",
            "general_comparison": "Balance all available analyses based on what best answers the specific question."
        }
        
        return guidance_map.get(question_type, guidance_map["general_comparison"])
    
    def __del__(self):
        """Cleanup EnrichmentChecker connection on destruction."""
        if hasattr(self, 'enrichment_checker') and self.enrichment_checker:
            try:
                self.enrichment_checker.close()
            except:
                pass  # Ignore cleanup errors