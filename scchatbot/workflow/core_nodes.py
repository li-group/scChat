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

from ..cell_type_models import ChatState, ExecutionStep
from langchain_core.messages import HumanMessage, AIMessage
from ..shared import extract_cell_types_from_question, needs_cell_discovery, create_cell_discovery_steps


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
        state["initial_plan"] = None
        state["execution_plan"] = None
        state["current_step_index"] = 0
        state["execution_history"] = []
        state["function_result"] = None
        state["function_name"] = None
        state["function_args"] = None
        state["conversation_complete"] = False
        state["errors"] = []
        
        # Initialize jury system fields
        state["jury_verdicts"] = None
        state["jury_decision"] = None
        state["revision_type"] = None
        state["jury_iteration"] = 0
        state["conflict_resolution_applied"] = False
        
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
        - If user mentions specific analysis types, use them in the "analyses" parameter:
          * "GSEA" or "gene set enrichment" → "analyses": ["gsea"]
          * "GO" or "gene ontology" → "analyses": ["go"] 
          * "KEGG" or "pathway" → "analyses": ["kegg"]
          * "REACTOME" → "analyses": ["reactome"]
        - If no specific type mentioned, omit "analyses" parameter (defaults to all types)
        - Examples:
          * "Run GSEA analysis" → "analyses": ["gsea"]
          * "Run enrichment analysis" → no analyses parameter (uses all)
          
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
            
            # Skip steps for unavailable cell types
            if unavailable_cell_types:
                enhanced_plan = self._skip_unavailable_cell_steps(enhanced_plan, unavailable_cell_types)
            
            # Store as initial plan (will be validated by enhanced evaluator)
            state["initial_plan"] = enhanced_plan
            
        except Exception as e:
            print(f"Planning error: {e}")
            # Fallback: create a simple conversational response plan
            state["initial_plan"] = {
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
        - FOCUS: Create a streamlined plan targeting the specific pathway analysis mentioned
        - PRIORITY: Use the exact analysis type mentioned in the query
        - EFFICIENCY: Avoid unnecessary broader analyses unless specifically requested
        - EXAMPLE: For "GSEA analysis of T cells" → focus on enrichment_analysis with "analyses": ["gsea"]
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
                
                func = self.function_mapping[step.function_name]
                result = func(**step.parameters)
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
        
        # Record execution in state with special handling for visualization functions
        is_visualization = step_data.get("function_name", "") in self.visualization_functions
        
        # Don't truncate visualization results as they contain HTML plots
        if is_visualization and result and isinstance(result, str):
            stored_result = str(result)  # Keep full HTML content
            print(f"🎨 Visualization result stored: {len(stored_result)} chars, contains HTML: {bool('<div' in stored_result or '<html' in stored_result)}")
        else:
            stored_result = str(result)[:500] if result else "Success"  # Truncate other results
        
        state["execution_history"].append({
            "step_index": state["current_step_index"],
            "step": step_data,
            "success": success,
            "result": stored_result,
            "error": error_msg
        })
        
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
        
        Uses the jury system's proven cell type extraction and discovery logic.
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