"""
Workflow node implementations for the multi-agent chatbot.

This module contains all the individual workflow nodes that make up
the LangGraph workflow for processing user requests and generating responses.
"""

import json
import openai
from typing import Dict, Any, List

from .cell_type_models import ChatState, ExecutionStep
from langchain_core.messages import HumanMessage, AIMessage


class WorkflowNodes:
    """
    Container for all workflow node implementations.
    
    These nodes form the core of the LangGraph workflow, handling input processing,
    planning, status checking, execution, evaluation, and response generation.
    """
    
    def __init__(self, initial_annotation_content, initial_cell_types, adata, 
                 history_manager, hierarchy_manager, cell_type_extractor,
                 function_descriptions, function_mapping, visualization_functions,
                 simple_cache):
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
        
        # Load function history and memory context
        state["function_history_summary"] = self.history_manager.get_available_results()
        state["missing_cell_types"] = []
        state["required_preprocessing"] = []
        
        return state

    def planner_node(self, state: ChatState) -> ChatState:
        """Create initial execution plan (before validation)"""
        message = state["current_message"]
        available_functions = self.function_descriptions
        available_cell_types = state["available_cell_types"]
        function_history = state["function_history_summary"]
        
        planning_prompt = f"""
        You are an intelligent planner for single-cell RNA-seq analysis. 
        
        Create a step-by-step execution plan for the user query.
        
        CONTEXT:
        - Available cell types: {', '.join(available_cell_types)}
        - Previous analyses: {json.dumps(function_history, indent=2)}
        
        Available functions:
        {self._summarize_functions(available_functions)}
        
        User question: "{message}"
        
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
        - The status checker will validate and modify your plan if needed
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
            
            # Store as initial plan (will be validated by status checker)
            state["initial_plan"] = plan_data
            
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

    def status_checker_node(self, state: ChatState) -> ChatState:
        """Validate and enhance the execution plan with hierarchy awareness"""
        if not state["initial_plan"]:
            state["execution_plan"] = state["initial_plan"]
            return state
        
        initial_steps = state["initial_plan"]["steps"]
        validated_steps = []
        missing_cell_types = []
        
        processing_paths_needed = {}  # parent_type -> set(target_children)
        processing_steps_created = {}  # parent_type -> step_object
        
        print(f"🔍 Hierarchical Status Checker: Validating {len(initial_steps)} steps...")
        
        for step in initial_steps:
            step_valid = True
            target_cell_type = step.get("target_cell_type") or step.get("parameters", {}).get("cell_type")
            
            # Check if step requires a specific cell type
            if target_cell_type and target_cell_type != "overall":
                # Parse multiple cell types if they exist
                cell_types_to_check = self._parse_cell_types(target_cell_type)
                
                for single_cell_type in cell_types_to_check:
                    single_cell_type = single_cell_type.strip()
                    
                    # 🧬 Use hierarchical manager to resolve cell type
                    if self.hierarchy_manager:
                        try:
                            resolved_series, metadata = self.hierarchy_manager.resolve_cell_type_for_analysis(single_cell_type)
                            
                            if metadata["resolution_method"] == "needs_processing":
                                processing_path = metadata["processing_path"]
                                print(f"🛤️ Processing path needed for '{single_cell_type}': {' → '.join(processing_path)}")
                                
                                # Track each parent->child relationship in the path
                                for i in range(len(processing_path) - 1):
                                    parent_type = processing_path[i]
                                    child_type = processing_path[i + 1]
                                    
                                    # Track that this parent needs to produce this child
                                    if parent_type not in processing_paths_needed:
                                        processing_paths_needed[parent_type] = set()
                                    processing_paths_needed[parent_type].add(child_type)
                                    
                                    print(f"📋 Tracking: {parent_type} → {child_type}")
                                
                                step_valid = True
                            
                            elif metadata["resolution_method"] == "not_found":
                                print(f"❌ Cell type '{single_cell_type}' not found. Suggestions: {metadata.get('suggestions', [])}")
                                step_valid = False
                            
                            elif metadata["resolution_method"] in ["direct", "ancestor_aggregation"]:
                                print(f"✅ Cell type '{single_cell_type}' can be resolved via {metadata['resolution_method']}")
                                step_valid = True
                                
                        except Exception as e:
                            print(f"⚠️ Error resolving cell type '{single_cell_type}': {e}")
                            step_valid = False
                
                # Handle multiple cell types (split into separate steps)
                if len(cell_types_to_check) > 1:
                    valid_cell_types = [ct.strip() for ct in cell_types_to_check 
                                      if self.hierarchy_manager and self.hierarchy_manager.is_valid_cell_type(ct.strip())]
                    
                    if valid_cell_types:
                        # Create a step for each valid cell type
                        for i, valid_ct in enumerate(valid_cell_types):
                            if i == 0:
                                # Modify the current step for the first cell type
                                step["parameters"]["cell_type"] = valid_ct
                                step["target_cell_type"] = valid_ct
                                step["description"] = step["description"].replace(target_cell_type, valid_ct)
                            else:
                                # Create new steps for additional cell types
                                new_step = step.copy()
                                new_step["parameters"] = step["parameters"].copy()
                                new_step["parameters"]["cell_type"] = valid_ct
                                new_step["target_cell_type"] = valid_ct
                                new_step["description"] = step["description"].replace(target_cell_type, valid_ct)
                                validated_steps.append(new_step)
                        step_valid = True
                    else:
                        step_valid = False
                
                # Handle single cell type (ensure parameter is set properly)
                elif len(cell_types_to_check) == 1 and step_valid:
                    single_cell_type = cell_types_to_check[0].strip()
                    # CRITICAL FIX: Ensure cell_type parameter is set for single cell types
                    if "parameters" not in step:
                        step["parameters"] = {}
                    step["parameters"]["cell_type"] = single_cell_type
                    step["target_cell_type"] = single_cell_type
                    print(f"🎯 FIXED: Set cell_type parameter to '{single_cell_type}' for step: {step['function_name']}")
            
            # Check for redundant function calls
            if step_valid and step["function_name"] != "conversational_response":
                if self.history_manager.has_been_executed(step["function_name"], step.get("parameters", {})):
                    print(f"🔄 Function already executed recently: {step['function_name']}")
            
            if step_valid:
                validated_steps.append(step)
        
        consolidated_preprocessing = []
        
        for parent_type, target_children in processing_paths_needed.items():
            target_children_list = sorted(list(target_children))
            
            # Create a single processing step that will discover multiple children
            consolidated_step = {
                "step_type": "analysis",
                "function_name": "process_cells",
                "parameters": {"cell_type": parent_type},
                "description": f"Process {parent_type} to discover {', '.join(target_children_list)}",
                "expected_outcome": f"Discover cell types: {', '.join(target_children_list)}",
                "target_cell_type": parent_type,
                "expected_children": target_children_list
            }
            
            consolidated_preprocessing.append(consolidated_step)
            
            # Update available cell types for subsequent validation
            state["available_cell_types"].extend(target_children_list)
            
            print(f"🎯 Consolidated processing: {parent_type} → {', '.join(target_children_list)}")
        
        # Combine consolidated preprocessing with validated steps
        final_steps = consolidated_preprocessing + validated_steps
        
        final_steps_with_validation = []
        for step in final_steps:
            final_steps_with_validation.append(step)
            
            # Add validation step after process_cells operations
            if step["function_name"] == "process_cells" and "expected_children" in step:
                validation_step = {
                    "step_type": "validation",
                    "function_name": "validate_processing_results",
                    "parameters": {
                        "processed_parent": step["parameters"]["cell_type"],
                        "expected_children": step["expected_children"]
                    },
                    "description": f"Validate that {step['parameters']['cell_type']} processing discovered expected cell types",
                    "expected_outcome": "Confirm all expected cell types are available",
                    "target_cell_type": None
                }
                final_steps_with_validation.append(validation_step)
        
        # Create validated execution plan
        validated_plan = {
            "steps": final_steps_with_validation,
            "original_question": state["current_message"],
            "plan_summary": state["initial_plan"]["plan_summary"],
            "estimated_steps": len(final_steps_with_validation),
            "consolidation_summary": f"Consolidated {len(processing_paths_needed)} unique processing operations",
            "validation_notes": f"Added {len(consolidated_preprocessing)} consolidated processing steps with validation"
        }
        
        state["execution_plan"] = validated_plan
        state["missing_cell_types"] = missing_cell_types
        state["required_preprocessing"] = consolidated_preprocessing
        
        print(f"Final plan has {len(final_steps_with_validation)} steps")
        print(f"   • {len(consolidated_preprocessing)} consolidated processing operations")
        print(f"   • {len(validated_steps)} analysis/visualization steps")
        print(f"   • {len([s for s in final_steps_with_validation if s.get('step_type') == 'validation'])} validation steps")
        
        return state

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
                else:
                    success = False
                    error_msg = result["message"]
                    print(f"❌ Validation failed: {error_msg}")
                
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
            
        return state

    def evaluator_node(self, state: ChatState) -> ChatState:
        """Evaluate execution results and determine next steps"""
        if not state["execution_history"]:
            state["conversation_complete"] = True
            return state
            
        last_execution = state["execution_history"][-1]
        
        if not last_execution["success"]:
            # If step failed, complete with error
            state["conversation_complete"] = True
            return state
        
        # Move to next step
        state["current_step_index"] += 1
        
        # Check if original plan steps are complete
        original_steps_complete = state["current_step_index"] >= len(state["execution_plan"]["steps"])
        
        # If all original steps are done, check if we need to add final question step
        if original_steps_complete:
            # Check if the last step was already a final question
            if state["execution_plan"]["steps"] and state["execution_plan"]["steps"][-1].get("step_type") == "final_question":
                # Final question already executed, we're truly done
                state["conversation_complete"] = True
                print("🏁 All steps including final question completed!")
            else:
                # Check if this is a visualization-only request
                is_visualization_only = state["execution_plan"].get("visualization_only", False)
                
                if is_visualization_only:
                    # For visualization-only requests, don't add comprehensive question
                    state["conversation_complete"] = True
                    print("🎨 Visualization-only request complete - skipping comprehensive analysis")
                else:
                    # Add final question step to the plan
                    print("📝 Adding final comprehensive question step...")
                    final_question_step = {
                        "step_type": "final_question",
                        "function_name": "final_question",
                        "parameters": {"original_question": state["execution_plan"]["original_question"]},
                        "description": "Ask comprehensive final question based on all analysis",
                        "expected_outcome": "Comprehensive answer to original question",
                        "target_cell_type": None
                    }
                    
                    # Add the final step to the plan
                    state["execution_plan"]["steps"].append(final_question_step)
                    print(f"✅ Added final question step. Total steps now: {len(state['execution_plan']['steps'])}")
        
        return state

    def response_generator_node(self, state: ChatState) -> ChatState:
        """Generate final response based on execution results"""
        if not state["execution_plan"] or not state["execution_history"]:
            state["response"] = json.dumps({"response": "I encountered an issue processing your request."})
        else:
            # Check if this is a visualization-only request
            is_visualization_only = state["execution_plan"].get("visualization_only", False)
            
            if is_visualization_only:
                # Handle visualization-only requests with simple response
                print("🎨 Generating simple response for visualization-only request...")
                return self._generate_visualization_only_response(state)
            
            # Check if the last executed step was a final question
            last_execution = state["execution_history"][-1] if state["execution_history"] else None
            
            if last_execution and last_execution["step"].get("step_type") == "final_question":
                # This was a final comprehensive question - use the result directly BUT include plots
                if last_execution["success"]:
                    comprehensive_answer = state["function_result"]
                    
                    # Collect plots from execution history for comprehensive answer
                    print("🔍 Starting plot collection for comprehensive answer...")
                    summary, collected_plots = self._generate_execution_summary_with_plots(state)
                    print(f"🔍 Plot collection complete. Summary length: {len(summary)}, Plots length: {len(collected_plots)}")
                    
                    response_data = {
                        "response": comprehensive_answer,
                        "response_type": "comprehensive_final_answer"
                    }
                    
                    # Include plots if any visualization functions were executed
                    if collected_plots:
                        response_data["graph_html"] = collected_plots
                        plot_count = len(collected_plots.split('<div class=')) - 1
                        print(f"🎨 Including {plot_count} plots in comprehensive answer")
                        print(f"🎨 Plot HTML preview: {collected_plots[:200]}...")
                    else:
                        print("❌ No plots collected for comprehensive answer")
                    
                    state["response"] = json.dumps(response_data)
                    print("🎯 Using comprehensive final answer as response")
                else:
                    # Final question failed
                    error_msg = last_execution["error"] if last_execution else "Unknown error"
                    state["response"] = json.dumps({
                        "response": f"I encountered an error generating the final answer: {error_msg}"
                    })
            else:
                # Handle regular single-step or multi-step responses (existing logic)
                if len(state["execution_plan"]["steps"]) == 1:
                    step = state["execution_plan"]["steps"][0]
                    execution = state["execution_history"][0] if state["execution_history"] else None
                    
                    if execution and execution["success"]:
                        # Handle single-step responses
                        if step["function_name"] in self.visualization_functions:
                            # Visualization response with enhanced description
                            viz_summary = self._generate_visualization_description(step, state["function_result"])
                            state["response"] = json.dumps({
                                "response": viz_summary, 
                                "graph_html": state["function_result"]
                            })
                        elif step["function_name"] == "conversational_response":
                            # Conversational response
                            state["response"] = json.dumps({
                                "response": state["function_result"]
                            })
                        else:
                            # Analysis response with AI interpretation
                            interpretation = self._get_ai_interpretation_for_result(
                                state["current_message"], 
                                state["function_result"],
                                state["messages"]
                            )
                            state["response"] = json.dumps({"response": interpretation})
                    else:
                        # Single step failed
                        error_msg = execution["error"] if execution else "Unknown error"
                        state["response"] = json.dumps({
                            "response": f"I encountered an error: {error_msg}"
                        })
                else:
                    # Multi-step plan - generate comprehensive summary with plots
                    summary, collected_plots = self._generate_execution_summary_with_plots(state)
                    response_data = {"response": summary}
                    
                    # Include plots if any visualization functions were executed
                    if collected_plots:
                        response_data["graph_html"] = collected_plots
                    
                    state["response"] = json.dumps(response_data)
        
        # Add response to message history
        try:
            response_content = json.loads(state["response"])["response"]
            state["messages"].append(AIMessage(content=response_content))
        except:
            state["messages"].append(AIMessage(content="Analysis completed."))
        
        return state

    # ========== Helper Methods ==========
    
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

    def _parse_cell_types(self, cell_type_string: str) -> List[str]:
        """Parse a string that might contain multiple cell types"""
        if self.cell_type_extractor:
            return self.cell_type_extractor.parse_multi_cell_type_string(cell_type_string)
        else:
            # Simple fallback parsing
            separators = [',', ' and ', ' & ', ';', ' vs ', ' versus ', ' or ']
            cell_types = [cell_type_string]
            
            for separator in separators:
                new_cell_types = []
                for ct in cell_types:
                    if separator in ct:
                        new_cell_types.extend([part.strip() for part in ct.split(separator)])
                    else:
                        new_cell_types.append(ct)
                cell_types = new_cell_types
            
            return [ct.strip() for ct in cell_types if ct.strip()]

    def _extract_cell_types_from_result(self, result: Any) -> List[str]:
        """Extract cell types from analysis result"""
        if self.cell_type_extractor:
            return self.cell_type_extractor.extract_from_annotation_result(result)
        else:
            # Simple fallback extraction
            if isinstance(result, str) and "cell_type" in result:
                return ["T cell", "B cell"]  # Placeholder
            return []

    def _get_relevant_cell_types_from_context(self, state: ChatState) -> List[str]:
        """Extract relevant cell types from execution context"""
        if self.cell_type_extractor:
            # First try without history to get current context
            current_cell_types = self.cell_type_extractor.extract_from_execution_context(state, include_history=False)
            
            # Only include history if current context is empty or too generic
            if not current_cell_types or (len(current_cell_types) == 1 and current_cell_types[0] in ["overall", "all"]):
                print("🔍 No specific cell types in current context - including historical data")
                return self.cell_type_extractor.extract_from_execution_context(state, include_history=True)
            else:
                print(f"🎯 Found specific cell types in current context: {current_cell_types} - focusing on these")
                return current_cell_types
        else:
            # Fallback if extractor not initialized
            print("⚠️ Cell type extractor not initialized, using state fallback")
            return state.get("available_cell_types", [])

    def _build_cached_analysis_context(self, cell_types: List[str]) -> str:
        """Build analysis context from cached results for relevant cell types"""
        analysis_context = ""
        
        for cell_type in cell_types:
            print(f"🔍 Retrieving cached insights for {cell_type}...")
            insights = self.simple_cache.get_analysis_insights(cell_type)
            
            if insights and insights.get("summary"):
                analysis_context += f"\n🧬 **CACHED ANALYSIS RESULTS FOR {cell_type.upper()}**:\n"
                
                # Add enrichment insights with specific pathway names
                for analysis_name, data in insights.get("enrichment_insights", {}).items():
                    if data.get("top_terms"):
                        top_terms = data["top_terms"][:3]  # Top 3 terms
                        p_values = data.get("p_values", [])[:3]
                        
                        analysis_context += f"• **{analysis_name}**: "
                        term_details = []
                        for i, term in enumerate(top_terms):
                            p_val = f" (p={p_values[i]:.2e})" if i < len(p_values) else ""
                            term_details.append(f"{term}{p_val}")
                        analysis_context += ", ".join(term_details)
                        analysis_context += f" [{data.get('total_significant', 0)} total significant]\n"
                
                # Add DEA insights with specific gene information
                for condition, data in insights.get("dea_insights", {}).items():
                    analysis_context += f"• **DEA ({condition})**: {data.get('significant_genes', 0)} significant genes "
                    analysis_context += f"({data.get('upregulated', 0)} ↑, {data.get('downregulated', 0)} ↓)\n"
                    
                    top_genes = data.get("top_genes", [])[:3]
                    if top_genes:
                        analysis_context += f"  Top upregulated: {', '.join(top_genes)}\n"
                
                analysis_context += "\n"
        
        return analysis_context if analysis_context else "No cached analysis results found.\n"

    def _execute_final_question(self, state: ChatState) -> str:
        """Execute a comprehensive final question using all available context"""
        original_question = state["execution_plan"]["original_question"]
        
        relevant_cell_types = self._get_relevant_cell_types_from_context(state)
        cached_context = self._build_cached_analysis_context(relevant_cell_types)
        
        # Build conversation context - only include current session, not previous unrelated conversations
        conversation_context = f"CURRENT QUESTION: {original_question}\n"
        
        # Only include the most recent user question for context, ignore previous assistant responses
        # to avoid polluting the context with irrelevant previous analyses
        
        # Build analysis summary from execution history
        analysis_summary = ""
        successful_analyses = [h for h in state["execution_history"] if h["success"] and h["step"].get("step_type") != "final_question"]
        
        if successful_analyses:
            analysis_summary = "ANALYSES PERFORMED IN THIS SESSION:\n"
            for h in successful_analyses:
                step_desc = h["step"]["description"]
                analysis_summary += f"✅ {step_desc}\n"
            analysis_summary += "\n"
        
        # Add hierarchical context if available
        hierarchy_context = ""
        if self.hierarchy_manager:
            lineage_summary = self.hierarchy_manager.get_lineage_summary()
            hierarchy_context = f"HIERARCHICAL CONTEXT:\n"
            hierarchy_context += f"• Total cells analyzed: {lineage_summary['total_cells']}\n"
            hierarchy_context += f"• Current cell types: {lineage_summary['unique_current_types']}\n"
            hierarchy_context += f"• Processing operations: {lineage_summary['processing_snapshots']}\n\n"
        
        final_prompt = f"""Based on the specific analysis results shown below, provide a comprehensive answer to the user's question.

                            ORIGINAL QUESTION:
                            {original_question}

                            SPECIFIC ANALYSIS RESULTS FROM CACHE:
                            {cached_context}

                            {hierarchy_context}{analysis_summary}

                            INSTRUCTIONS:
                            1. Reference the SPECIFIC pathways, genes, and statistics shown above
                            2. Use exact names and numbers from the cached results
                            3. Explain the biological significance of these specific findings
                            4. Connect the results directly to the user's question
                            5. Be quantitative and specific, not generic
                            6. Focus ONLY on the current question and relevant analysis results

                            Your response should cite the actual analysis results, not general knowledge or previous conversations."""

        try:
            # Use OpenAI to generate comprehensive response
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.2,  # Lower temperature for more consistent responses
                max_tokens=1500
            )
            
            final_answer = response.choices[0].message.content
            
            # 📊 Log cache usage
            if cached_context and "No cached analysis results found" not in cached_context:
                cache_cell_types = [ct for ct in relevant_cell_types if ct in cached_context]
                print(f"✅ Used cached insights from {len(cache_cell_types)} cell types: {cache_cell_types}")
            else:
                print("⚠️ No cached insights found - using execution history only")
            
            return final_answer
            
        except Exception as e:
            error_msg = f"Error generating final comprehensive answer: {e}"
            print(f"❌ {error_msg}")
            return error_msg

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

    def _get_ai_interpretation_for_result(self, original_question: str, result: Any, message_history: List) -> str:
        """Get AI interpretation for analysis results"""
        try:
            interpretation_prompt = f"""
            Provide a brief interpretation of this analysis result for the user's question: "{original_question}"
            
            Result: {str(result)[:1000]}
            
            Provide a concise, informative response that explains what the analysis shows.
            """
            
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": interpretation_prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ Error generating AI interpretation: {e}")
            return f"Analysis completed. Result: {str(result)[:200]}..."

    def _generate_execution_summary(self, state: ChatState) -> str:
        """Generate a summary of multi-step execution (legacy method)"""
        summary, _ = self._generate_execution_summary_with_plots(state)
        return summary
    
    def _generate_execution_summary_with_plots(self, state: ChatState):
        """Generate a summary of multi-step execution with collected plots"""
        print(f"🔍 Plot collection: Total execution history entries: {len(state.get('execution_history', []))}")
        
        successful_steps = [h for h in state["execution_history"] if h["success"]]
        
        # Debug: Show all function names in execution history
        all_function_names = [h["step"]["function_name"] for h in successful_steps]
        print(f"🔍 Plot collection: Function names in execution history: {all_function_names}")
        print(f"🔍 Plot collection: Visualization functions set: {self.visualization_functions}")
        
        visualization_steps = [h for h in successful_steps if h["step"]["function_name"] in self.visualization_functions]
        
        print(f"🔍 Plot collection: {len(successful_steps)} successful steps, {len(visualization_steps)} visualization steps")
        
        if not successful_steps:
            return "I encountered issues executing your request.", ""
        
        summary = f"I completed {len(successful_steps)} analysis steps:\n\n"
        collected_plots = []
        plot_descriptions = []
        
        for i, step in enumerate(successful_steps, 1):
            step_desc = step["step"]["description"]
            function_name = step["step"]["function_name"]
            result = step.get("result", "")
            
            # Check if this was a visualization step
            if function_name in self.visualization_functions:
                # Add plot description to summary
                plot_info = f"📊 {step_desc}"
                if step["step"].get("parameters", {}).get("cell_type"):
                    plot_info += f" (for {step['step']['parameters']['cell_type']})"
                summary += f"{i}. {plot_info}\n"
                
                # Debug: Check what we have in result
                print(f"🔍 Plot collection debug - Function: {function_name}")
                print(f"🔍 Result type: {type(result)}, length: {len(str(result)) if result else 0}")
                print(f"🔍 Has HTML markers: <div={bool('<div' in str(result))}, <html={bool('<html' in str(result))}")
                print(f"🔍 Result preview: {str(result)[:200]}...")
                
                # Collect the HTML plot if it's valid HTML
                if result and isinstance(result, str) and ("<div" in result or "<html" in result):
                    # Check if this is a duplicate plot
                    if step_desc not in plot_descriptions:
                        collected_plots.append(f"<div class='plot-container'><h4>{step_desc}</h4>{result}</div>")
                        plot_descriptions.append(step_desc)
                        print(f"✅ Plot collected: {step_desc}")
                    else:
                        print(f"⚠️ Duplicate plot detected, skipping: {step_desc}")
                else:
                    print(f"❌ Plot NOT collected for {step_desc} - invalid HTML or empty result")
            else:
                # Regular analysis step
                summary += f"{i}. {step_desc}\n"
        
        summary += "\nAll analyses have been completed successfully."
        
        # Add plot information to summary if we have plots
        if plot_descriptions:
            summary += f"\n\n📊 Generated {len(plot_descriptions)} visualization(s):"
            for desc in plot_descriptions:
                summary += f"\n• {desc}"
        
        # Combine all plots into a single HTML string
        combined_plots = "\n".join(collected_plots) if collected_plots else ""
        
        print(f"🔍 Plot collection summary: Collected {len(collected_plots)} plots, total HTML length: {len(combined_plots)}")
        if collected_plots:
            print(f"🔍 First plot preview: {collected_plots[0][:100]}...")
        
        return summary, combined_plots
    
    def _generate_visualization_only_response(self, state: ChatState) -> ChatState:
        """Generate simple response for visualization-only requests"""
        # Collect plots from execution history
        summary, collected_plots = self._generate_execution_summary_with_plots(state)
        
        # Find the visualization step(s) to create a simple description
        viz_steps = []
        for execution in state["execution_history"]:
            if (execution.get("success") and 
                execution["step"]["function_name"] in self.visualization_functions):
                viz_steps.append(execution["step"]["description"])
        
        # Create simple response
        if viz_steps:
            if len(viz_steps) == 1:
                simple_response = f"Here is the {viz_steps[0].lower()}:"
            else:
                simple_response = f"Here are the requested visualizations:"
        else:
            simple_response = "Here is the requested visualization:"
        
        response_data = {
            "response": simple_response,
            "response_type": "visualization_only"
        }
        
        # Include plots if available
        if collected_plots:
            response_data["graph_html"] = collected_plots
            plot_count = len(collected_plots.split('<div class=')) - 1
            print(f"🎨 Including {plot_count} plots in visualization-only response")
        else:
            print("❌ No plots found for visualization-only request")
            response_data["response"] = "I couldn't generate the requested visualization. Please check if the analysis data is available."
        
        state["response"] = json.dumps(response_data)
        return state
    
    def _generate_visualization_description(self, step: Dict, result: str) -> str:
        """Generate a descriptive summary for visualization functions"""
        function_name = step["function_name"]
        parameters = step.get("parameters", {})
        
        # Create user-friendly descriptions for different visualization types
        descriptions = {
            "display_dotplot": "gene expression dotplot",
            "display_cell_type_composition": "cell type composition dendrogram", 
            "display_gsea_dotplot": "GSEA enrichment dotplot",
            "display_umap": "UMAP dimensionality reduction plot",
            "display_processed_umap": "annotated UMAP plot with cell types",
            "display_enrichment_barplot": "enrichment analysis barplot",
            "display_enrichment_dotplot": "enrichment analysis dotplot",
            "display_enrichment_visualization": "comprehensive enrichment visualization"
        }
        
        base_desc = descriptions.get(function_name, function_name.replace("_", " "))
        
        # Add context based on parameters
        context_parts = []
        if parameters.get("cell_type"):
            context_parts.append(f"for {parameters['cell_type']}")
        if parameters.get("analysis"):
            context_parts.append(f"using {parameters['analysis'].upper()} analysis")
        if parameters.get("plot_type") == "both":
            context_parts.append("(both bar and dot plots)")
        
        context = " " + " ".join(context_parts) if context_parts else ""
        
        # Check if the plot was generated successfully
        if result and isinstance(result, str) and ("<div" in result or "<html" in result):
            status = "✅ Successfully generated"
        else:
            status = "⚠️ Generated"
        
        return f"{status} {base_desc}{context}. The interactive plot is displayed below."