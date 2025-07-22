"""
Core workflow nodes implementation.

This module contains the basic node implementations:
- input_processor_node
- planner_node  
- executor_node
"""

import json
import openai
from typing import Dict, Any, List
from datetime import datetime

from ..cell_type_models import ChatState, ExecutionStep
from langchain_core.messages import HumanMessage, AIMessage
from ..shared import extract_cell_types_from_question, needs_cell_discovery, create_cell_discovery_steps

# Import EnrichmentChecker with error handling
try:
    from ..enrichment_checker import EnrichmentChecker
    ENRICHMENT_CHECKER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ EnrichmentChecker not available: {e}")
    EnrichmentChecker = None
    ENRICHMENT_CHECKER_AVAILABLE = False


class CoreNodes:
    """Core workflow nodes for input processing, planning, and execution."""
    
    def __init__(self, initial_annotation_content, initial_cell_types, adata, 
                 history_manager, hierarchy_manager, cell_type_extractor,
                 function_descriptions, function_mapping, visualization_functions, simple_cache):
        self.initial_annotation_content = initial_annotation_content
        self.initial_cell_types = initial_cell_types
        self.adata = adata
        self.history_manager = history_manager
        self.hierarchy_manager = hierarchy_manager
        self.cell_type_extractor = cell_type_extractor
        self.function_descriptions = function_descriptions
        self.function_mapping = function_mapping
        self.visualization_functions = visualization_functions
        self.simple_cache = simple_cache
        
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
        """Process incoming user message and initialize state"""
        # Initialize state if this is a new conversation
        if not state.get("messages"):
            state["messages"] = [AIMessage(content=self.initial_annotation_content)]
        
        # Add user message
        state["messages"].append(HumanMessage(content=state["current_message"]))
        
        # Initialize state variables
        state["available_cell_types"] = self.initial_cell_types
        state["adata"] = self.adata
        state["execution_plan"] = None
        state["current_step_index"] = 0
        state["execution_history"] = []
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
        """Create initial execution plan with query type detection and enhanced prompting"""
        message = state["current_message"]
        available_functions = self.function_descriptions
        available_cell_types = state["available_cell_types"]
        function_history = state["function_history_summary"]
        unavailable_cell_types = state.get("unavailable_cell_types", [])
        
        # Step 1: Query type detection for prompt enhancement
        query_type = self._detect_query_type(message)
        state["query_type"] = query_type
        
        # Step 2: Enhanced LLM-based planning with query type context
        return self._create_enhanced_plan(state, message, available_functions, available_cell_types, function_history, unavailable_cell_types, query_type)
    
    def _create_enhanced_plan(self, state: ChatState, message: str, available_functions: List, available_cell_types: List[str], function_history: Dict, unavailable_cell_types: List[str], query_type: str) -> ChatState:
        """Create enhanced plan using LLM with query type-specific guidance"""
        
        # Get query type-specific instructions 
        query_guidance = self._get_query_type_guidance(query_type)
        
        planning_prompt = f"""
        You are an intelligent planner for single-cell RNA-seq analysis. 
        
        Create a step-by-step execution plan for the user query.
        
        CONTEXT:
        - Available cell types: {', '.join(available_cell_types)}
        - Unavailable cell types: {', '.join(unavailable_cell_types)}
        - Previous analyses: {json.dumps(function_history, indent=2)}
        - Query type detected: {query_type}
        
        Available functions:
        {self._summarize_functions(available_functions)}
        
        User question: "{message}"
        
        {query_guidance}
        
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
          
        CONVERSATIONAL QUESTION DETECTION:
        - For interpretive questions about existing results, use "conversational_response" function
        - Examples of conversational questions:
          * "What can we tell from this GSEA analysis?"
          * "What does allograft rejection behavior mean?"
          * "How do you interpret these results?"
          * "What are the biological implications?"
        - Create simple plan with single conversational_response step for these questions
        - Do NOT create complex analysis plans for interpretive questions
        
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
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": planning_prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            plan_data = json.loads(response.choices[0].message.content)
            
            # 🧬 ENHANCED PLANNER: Add cell discovery if needed 
            enhanced_plan = self._add_cell_discovery_to_plan(plan_data, message, available_cell_types)
            
            # Apply enrichment enhancement to all enrichment steps
            enhanced_plan = self._enhance_all_enrichment_steps(enhanced_plan, message)
            
            # Skip steps for unavailable cell types
            if unavailable_cell_types:
                enhanced_plan = self._skip_unavailable_cell_steps(enhanced_plan, unavailable_cell_types)
            
            # Apply plan processing (moved from evaluator)
            # 1. Light consolidation - only remove exact consecutive duplicates
            enhanced_plan = self._light_consolidate_process_cells(enhanced_plan)
            
            # 2. Light validation - only log warnings for missing cell types
            self._log_missing_cell_type_warnings(enhanced_plan)
            
            # 3. Add validation steps after process_cells operations
            enhanced_plan = self._add_validation_steps_after_process_cells(enhanced_plan)
            
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
    
    def _detect_query_type(self, message: str) -> str:
        """Detect the type of query to enable analysis-specific planning"""
        message_lower = message.lower()
        
        # Specific pathway analysis
        if any(keyword in message_lower for keyword in ["gsea", "gene set enrichment", "kegg", "reactome", "go", "gene ontology"]):
            return "pathway_specific"
        
        # Marker analysis  
        if any(keyword in message_lower for keyword in ["markers", "marker genes", "differentially expressed", "dea", "differential expression"]):
            return "markers"
        
        # Visualization requests
        if any(keyword in message_lower for keyword in ["show", "display", "plot", "visualize", "visualization"]):
            return "visualization"
        
        # Comparison requests
        if any(keyword in message_lower for keyword in ["compare", "comparison", "vs", "versus", "difference", "different"]):
            return "comparison"
        
        # Conversational/interpretation
        if any(keyword in message_lower for keyword in ["what", "how", "why", "explain", "interpret", "mean", "significance"]):
            return "conversational"
        
        # Default to general analysis
        return "general"
    
    def _get_query_type_guidance(self, query_type: str) -> str:
        """Get query type-specific guidance for the planning prompt"""
        
        if query_type == "pathway_specific":
            return """
        🎯 PATHWAY-SPECIFIC QUERY DETECTED:
        - This query asks about specific pathway analysis (GSEA, GO, KEGG, REACTOME)
        - FOCUS: Create a streamlined plan targeting enrichment analysis
        - PRIORITY: Use minimal enrichment steps - EnrichmentChecker will optimize them
        - EFFICIENCY: Avoid unnecessary broader analyses unless specifically requested
        - EXAMPLE: For "GSEA analysis of T cells" → {{"cell_type": "T cell"}} (EnrichmentChecker handles method selection)
        """
        
        elif query_type == "markers":
            return """
        🎯 MARKER ANALYSIS QUERY DETECTED:
        - This query asks about marker genes or differential expression
        - FOCUS: Create a streamlined plan targeting differential expression analysis
        - PRIORITY: Use run_dea function as the primary analysis method
        - EFFICIENCY: Include visualization (dotplot) if requested
        - EXAMPLE: For "marker genes of T cells" → focus on run_dea function
        """
        
        elif query_type == "visualization":
            return """
        🎯 VISUALIZATION QUERY DETECTED:
        - This query primarily asks for displaying/showing plots
        - FOCUS: Create a visualization-focused plan
        - PRIORITY: Set "visualization_only": true if no analysis is needed
        - EFFICIENCY: Minimal analysis steps, maximum visualization focus
        - EXAMPLE: For "show GSEA plot" → focus on display_enrichment_visualization
        """
        
        elif query_type == "comparison":
            return """
        🎯 COMPARISON QUERY DETECTED:
        - This query asks to compare different cell types or conditions
        - FOCUS: Create separate analysis steps for each entity being compared
        - PRIORITY: Ensure both/all entities get analyzed before comparison
        - EFFICIENCY: Structure as: analyze A, analyze B, compare results
        - EXAMPLE: For "T cells vs B cells" → separate steps for each cell type
        """
        
        elif query_type == "conversational":
            return """
        🎯 CONVERSATIONAL/INTERPRETIVE QUERY DETECTED:
        - This query asks for explanation or interpretation of results
        - FOCUS: Use conversational_response function for interpretive questions
        - PRIORITY: Avoid complex analysis plans for interpretation requests
        - EFFICIENCY: Single conversational step unless new analysis is specifically requested
        - EXAMPLE: For "what does this mean?" → use conversational_response
        """
        
        else:  # general
            return """
        🎯 GENERAL ANALYSIS QUERY DETECTED:
        - This query requires comprehensive analysis approach
        - FOCUS: Create a balanced plan covering the user's analytical needs
        - PRIORITY: Follow standard analysis workflow patterns
        - EFFICIENCY: Include appropriate analysis and visualization steps
        """

    def _store_execution_result(self, step_data: Dict, result: Any, success: bool) -> Dict[str, Any]:
        """
        New intelligent result storage that preserves structure for critical functions
        """
        function_name = step_data.get("function_name", "")
        
        # Critical functions that need full structure preservation
        STRUCTURE_PRESERVED_FUNCTIONS = {
            "perform_enrichment_analyses",
            "dea_split_by_condition", 
            "process_cells",
            "compare_cell_counts"
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
        step = ExecutionStep(**step_data)
        
        print(f"🔄 Executing step {state['current_step_index'] + 1}: {step.description}")
        
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
                    new_cell_types = self._extract_cell_types_from_result(result)
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
        result_storage = self._store_execution_result(step_data, result, success)
        
        state["execution_history"].append({
            "step_index": state["current_step_index"],
            "step": step_data,
            "success": success,
            "result": result_storage["result"],  # Full structure preserved for critical functions
            "result_type": result_storage["result_type"],
            "result_summary": result_storage["result_summary"],  # For logging
            "error": error_msg
        })
        
        # Log storage decision for monitoring
        function_name = step_data.get("function_name", "")
        if result_storage["result_type"] == "structured":
            print(f"📊 Structured storage: {function_name} - Full data preserved")
        elif result_storage["result_type"] == "visualization":
            print(f"🎨 Visualization storage: {function_name} - HTML preserved")
        else:
            print(f"📄 Text storage: {function_name} - Truncated to 500 chars")
        
        # Increment step index if step was successful
        if success:
            state["current_step_index"] += 1
            print(f"🔄 Advanced to step {state['current_step_index'] + 1}")
        else:
            print(f"❌ Step failed, not advancing. Still on step {state['current_step_index'] + 1}")
            
        return state

    def _add_cell_discovery_to_plan(self, plan_data: Dict[str, Any], message: str, available_cell_types: List[str]) -> Dict[str, Any]:
        """
        Enhance the initial plan by adding cell discovery steps if needed.
        
        Uses proven cell type extraction and discovery logic.
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
        
        # Fix cell type names in original plan steps first
        original_steps = plan_data.get("steps", [])
        corrected_steps = self._fix_cell_type_names_in_steps(original_steps, needed_cell_types, message)
        plan_data["steps"] = corrected_steps
        
        # Check if discovery is needed
        if needs_cell_discovery(needed_cell_types, available_cell_types):
            print("🧬 Adding cell discovery steps to plan...")
            
            # Create discovery steps
            discovery_steps = create_cell_discovery_steps(needed_cell_types, available_cell_types, "analysis", self.hierarchy_manager)
            
            if discovery_steps:
                # Insert discovery steps at the beginning of the plan
                plan_data["steps"] = discovery_steps + corrected_steps
                
                # Update plan summary
                original_summary = plan_data.get("plan_summary", "")
                plan_data["plan_summary"] = f"Discover needed cell types then {original_summary.lower()}"
                
                print(f"🧬 Enhanced plan with {len(discovery_steps)} discovery steps")
            else:
                print("🧬 No discovery steps created")
        else:
            print("🧬 All needed cell types already available")
        
        return plan_data
    
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

    def _extract_cell_types_from_result(self, result: Any) -> List[str]:
        """Extract cell types from analysis result"""
        if self.cell_type_extractor:
            return self.cell_type_extractor.extract_from_annotation_result(result)
        else:
            # Simple fallback extraction
            if isinstance(result, str) and "cell_type" in result:
                return ["T cell", "B cell"]  # Placeholder
            return []

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
            # Check exact match or fuzzy match
            if expected_child in current_cell_types:
                found_children.append(expected_child)
            else:
                # Try fuzzy matching
                fuzzy_matches = [ct for ct in current_cell_types 
                               if expected_child.lower() in ct.lower() or ct.lower() in expected_child.lower()]
                if fuzzy_matches:
                    found_children.extend(fuzzy_matches)
                    print(f"🔄 Fuzzy match: '{expected_child}' → {fuzzy_matches}")
                else:
                    missing_children.append(expected_child)
        
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

    def _execute_final_question(self, state: ChatState) -> str:
        """Execute a comprehensive final question using all available context"""
        original_question = state["execution_plan"]["original_question"]
        
        # This is a simplified version - in a real implementation this would
        # use the cache manager and provide comprehensive analysis
        return f"Based on the analysis performed, here's a comprehensive answer to your question: {original_question}"

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
        enhanced_steps = []
        
        for step in plan_data.get("steps", []):
            if step.get("function_name") == "perform_enrichment_analyses":
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
        Version 1: Step-by-step evaluation with checking only (NO plan modifications)
        
        Evaluates the most recent execution step and logs detailed findings.
        Builds evaluation history for future adaptive capabilities.
        """
        
        print("🔍 STEP EVALUATOR V1: Starting step-by-step evaluation...")
        
        # Get the most recent execution
        execution_history = state.get("execution_history", [])
        if not execution_history:
            print("⚠️ No execution history found for evaluation")
            state["last_step_evaluation"] = {"status": "no_history", "critical_failure": False}
            return state
        
        last_execution = execution_history[-1]
        step_index = last_execution.get("step_index", -1)
        
        # FIX: The execution history stores function_name directly, not in a 'step' sub-dict
        function_name = last_execution.get('function_name', 'unknown')
        
        print(f"🔍 Evaluating step {step_index + 1}: {function_name}")
        
        # Perform comprehensive evaluation
        evaluation = self._evaluate_single_step_v1(last_execution, state)
        
        # Store evaluation results
        state["last_step_evaluation"] = evaluation
        
        # Build evaluation history
        step_eval_history = state.get("step_evaluation_history", [])
        step_eval_history.append(evaluation)
        state["step_evaluation_history"] = step_eval_history
        
        # Log results (for monitoring and debugging)
        self._log_step_evaluation(evaluation)
        
        print(f"✅ STEP EVALUATOR V1: Evaluation complete for step {step_index + 1}")
        return state

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

    def __del__(self):
        """Cleanup EnrichmentChecker connection on destruction."""
        if hasattr(self, 'enrichment_checker') and self.enrichment_checker:
            try:
                self.enrichment_checker.close()
            except:
                pass  # Ignore cleanup errors