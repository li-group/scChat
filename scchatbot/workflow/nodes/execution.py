"""
Execution node implementation.

This module contains the ExecutorNode which executes the steps in the execution plan,
manages function calls, and handles validation steps.
"""

import copy
from typing import Dict, Any, List

from ...cell_types.models import ChatState, ExecutionStep
from ..node_base import BaseWorkflowNode
from ..progress_manager import ProgressManager
import os
import uuid

import logging
logger = logging.getLogger(__name__)



MAX_INLINE_HTML_BYTES = 8_388_608

class ExecutorNode(BaseWorkflowNode):
    """
    Executor node that executes planned steps.
    
    Responsibilities:
    - Execute individual steps in the execution plan
    - Handle function calls with proper parameter management
    - Track execution history and results
    - Update available cell types based on processing results
    """
    
    def execute(self, state: ChatState) -> ChatState:
        """Main execution method."""
        return self.executor_node(state)
    
    def executor_node(self, state: ChatState) -> ChatState:
        """Execute the current step in the plan with hierarchy awareness and validation"""
        self._log_node_start("Executor", state)
        session_id = state.get("session_id", "default")
        progress_manager = ProgressManager(session_id)
        
        total_steps = len(state.get("execution_plan", {}).get("steps", []))
        progress_manager.set_total_steps(total_steps)
        progress_manager.update_stage("execution", "Starting analysis execution...")
        
        # === Fast-path: no actionable steps -> direct LLM answer ===

        plan = state.get("execution_plan") or {}
        steps = plan.get("steps") or []
        if not steps:
            state["no_tool_answer"] = True
            state["conversation_complete"] = True  # ensures final evaluator won't loop
            self._log_node_complete("Executor", state)
            return state
        # Continue executing while there are steps, with special handling for supplementary steps
        steps_executed = 0
        continue_execution = True
        
        while continue_execution and state.get("current_step_index", 0) < len(state.get("execution_plan", {}).get("steps", [])):
            if not state["execution_plan"] or state["current_step_index"] >= len(state["execution_plan"]["steps"]):
                state["conversation_complete"] = True
                break
                
            step_data = state["execution_plan"]["steps"][state["current_step_index"]]
            
            if hasattr(step_data, 'skip_reason') and step_data.skip_reason:
                self._handle_skipped_step(state, step_data)
                steps_executed += 1
                state["current_step_index"] += 1
                continue

            # Check if this step should be skipped due to aggregation
            steps_to_skip = state.get("_steps_to_skip", set())
            if state["current_step_index"] in steps_to_skip:
                logger.info(f"⏭️ Skipping step {state['current_step_index'] + 1}: Already included in aggregated visualization")
                # Create a skip reason and handle as skipped step
                step_data_copy = step_data.copy() if isinstance(step_data, dict) else step_data.__dict__.copy()
                step_data_copy["skip_reason"] = "Already included in aggregated display_cell_count_stacked_plot"
                self._handle_skipped_step(state, step_data_copy)
                steps_executed += 1
                state["current_step_index"] += 1
                continue

            if isinstance(step_data, ExecutionStep):
                step = step_data
            else:
                step = ExecutionStep(**step_data)
            
            logger.info(f"🔄 Executing step {state['current_step_index'] + 1}: {step.description}")

            if not getattr(step, "function_name", None):
                logger.info("ℹ️ No function for this step; treating as direct-answer/no-op.")
                result_text = step.description or ""
                if hasattr(step, "__dict__"):
                    safe_step = dict(step.__dict__)
                elif isinstance(step_data, dict):
                    safe_step = copy.deepcopy(step_data)
                else:
                    safe_step = {"description": step.description or ""}
                safe_step["function_name"] = safe_step.get("function_name") or "direct_answer"
                safe_step["parameters"] = safe_step.get("parameters", {}) or {}
                self._record_execution(state, safe_step, result_text, True, None, "direct_answer")
                state["current_step_index"] += 1
                steps_executed += 1
                logger.info(f"🔄 Advanced to step {state['current_step_index'] + 1}")
                continue
            #ADDEDAUG

            
            original_function_name = getattr(step_data, 'function_name', 'unknown') if hasattr(step_data, 'function_name') else step_data.get("function_name", "unknown")
            logger.info(f"🔍 STORAGE DEBUG: Original function_name from plan: '{original_function_name}'")
            
            success = False
            result = None
            error_msg = None
            
            try:
                step_num = state['current_step_index'] + 1
                progress_manager.send_custom_update(
                    f"Executing step {step_num}: {step.description}"
                )
                
                success, result, error_msg = self._execute_step(state, step)
                
                if success:
                    logger.info(f"✅ Step {state['current_step_index'] + 1} completed successfully")
                    self._handle_successful_step(state, step, result)
                    progress_manager.increment_step(f"Completed: {step.description}")
                
            except Exception as e:
                error_msg = str(e)
                success = False
                logger.info(f"❌ Step {state['current_step_index'] + 1} failed: {error_msg}")
                state["errors"].append(f"Step {state['current_step_index'] + 1} failed: {error_msg}")
            
            self._record_execution(state, step_data, result, success, error_msg, original_function_name)
            
            state["current_step_index"] += 1
            steps_executed += 1
            
            if success:
                logger.info(f"🔄 Advanced to step {state['current_step_index'] + 1}")
            else:
                logger.info(f"❌ Step failed: {error_msg}")
                logger.info(f"📝 Recording error and advancing to step {state['current_step_index'] + 1} (no retries for deterministic operations)")
                
                # Track failed supplementary steps to prevent infinite retry loops
                step_description = step.description if hasattr(step, 'description') else step_data.get("description", "")
                if "Post-evaluation:" in step_description:
                    step_signature = f"{step.function_name}_{step.parameters.get('cell_type', '') if hasattr(step, 'parameters') else step_data.get('parameters', {}).get('cell_type', '')}"
                    previously_failed = state.get("previously_failed_supplementary_steps", [])
                    if step_signature not in previously_failed:
                        previously_failed.append(step_signature)
                        state["previously_failed_supplementary_steps"] = previously_failed
                        logger.info(f"🚫 Recorded failed supplementary step: {step_signature}")
            
            # Check if we should continue executing (batch supplementary steps)
            continue_execution = self._should_continue_execution(state, steps_executed)
            
            if not continue_execution:
                logger.info(f"🛑 Stopping execution after {steps_executed} steps")
                break
        
        self._log_node_complete("Executor", state)
        return state
    
    def _handle_skipped_step(self, state: ChatState, step_data: Any):
        """Handle a step that should be skipped."""
        skip_reason = getattr(step_data, 'skip_reason', None) if hasattr(step_data, 'skip_reason') else step_data.get('skip_reason')
        logger.info(f"⏭️ Skipping step {state['current_step_index'] + 1}: {skip_reason}")
        
        # Record skipped step in execution history for critic awareness
        # Convert step_data to dict if it's an ExecutionStep object
        if hasattr(step_data, '__dict__'):
            step_dict = {k: v for k, v in step_data.__dict__.items()}
        elif isinstance(step_data, dict):
            step_dict = step_data.copy()
        else:
            step_dict = step_data
            
        state["execution_history"].append({
            "step_index": state["current_step_index"],
            "step": step_dict,
            "success": False,
            "result": None,
            "result_type": "skipped",
            "result_summary": f"Skipped - {skip_reason}",
            "error": skip_reason,
            "skipped": True  # Flag to distinguish from failures
        })
        
        # Note: state["current_step_index"] is now incremented in the main loop
    
    def _execute_step(self, state: ChatState, step: ExecutionStep) -> tuple[bool, Any, str]:
        """Execute a single step and return success, result, error_msg."""
        # Handle final question step
        if step.step_type == "final_question":
            logger.info("🎯 Executing final comprehensive question...")
            result = self._execute_final_question(state)
            return True, result, None
        
        else:
            # Handle regular analysis/visualization steps
            return self._execute_regular_step(state, step)
    
    
    def _execute_regular_step(self, state: ChatState, step: ExecutionStep) -> tuple[bool, Any, str]:
        """Execute a regular analysis or visualization step."""
        if step.function_name not in self.function_mapping:
            raise Exception(f"Function '{step.function_name}' not found")
        
        # Debug visualization function parameters
        if step.function_name in self.visualization_functions:
            logger.info(f"🔍 STEP DEBUG: Calling visualization function '{step.function_name}' with step parameters: {step.parameters}")
        
        # For visualization functions, enhance parameters with cell_type from execution context if missing
        enhanced_params = self._enhance_visualization_params(state, step)
        
        # Check if enhancement determined this step should fail
        if enhanced_params.get("_should_fail"):
            error_msg = enhanced_params.get("_fail_reason", "Step cannot be executed")
            logger.info(f"❌ Step execution blocked: {error_msg}")
            return False, None, error_msg
        
        # Remove internal flags before calling function
        enhanced_params.pop("_should_fail", None)
        enhanced_params.pop("_fail_reason", None)
        
        # CRITICAL FIX: Update step parameters with enhanced params for visualization functions
        # This ensures the execution history records the actual parameters used
        if step.function_name in self.visualization_functions and enhanced_params != step.parameters:
            logger.info(f"📝 Updating step parameters with enhanced params for execution history tracking")
            step.parameters = enhanced_params.copy()
        
        func = self.function_mapping[step.function_name]
        result = func(**enhanced_params)
        
        return True, result, None
    
    def _enhance_visualization_params(self, state: ChatState, step: ExecutionStep) -> Dict[str, Any]:
        """Enhance visualization parameters with context if needed."""
        enhanced_params = step.parameters.copy()
        
        if step.function_name in self.visualization_functions and "cell_type" not in enhanced_params:
            # Handle different types of visualizations
            if step.function_name in ["display_processed_umap", "display_leiden_umap", "display_overall_umap", "display_dotplot", "display_feature_plot", "display_violin_plot", "display_dea_heatmap"]:
                # These visualizations don't need enrichment analysis - they work with processed cell data

                # Special cases: these functions don't need cell_type parameter
                if step.function_name in ["display_feature_plot", "display_overall_umap"]:
                    logger.info(f"✅ {step.function_name} doesn't require cell_type parameter")
                    # Don't add cell_type parameter for these functions
                else:
                    # Other visualizations need cell_type parameter
                    execution_history = state.get("execution_history", [])

                    found_cell_type = False
                    for recent_execution in reversed(execution_history[-10:]):
                        step_data = recent_execution.get("step", {})

                        # Look for any successful cell processing or analysis
                        if (recent_execution.get("success") and
                            step_data.get("function_name") in ["process_cells", "perform_enrichment_analyses", "dea_split_by_condition"]):

                            step_params = step_data.get("parameters", {})
                            cell_type = step_params.get("cell_type")

                            if cell_type and cell_type != "unknown":
                                enhanced_params["cell_type"] = cell_type
                                logger.info(f"✅ Found cell type for {step.function_name}: {cell_type}")
                                found_cell_type = True
                                break

                    if not found_cell_type:
                        # For UMAP and other basic visualizations, try default cell types
                        available_cell_types = state.get("available_cell_types", [])
                        if available_cell_types and "Overall cells" in available_cell_types:
                            enhanced_params["cell_type"] = "Overall cells"
                            logger.info(f"✅ Using default cell type 'Overall cells' for {step.function_name}")
                        else:
                            logger.info(f"❌ No suitable cell type found for {step.function_name}")
                            enhanced_params["_should_fail"] = True
                            enhanced_params["_fail_reason"] = f"No processed cell data available for {step.function_name}"
            
            else:
                # Enrichment-based visualizations (display_enrichment_*)
                execution_history = state.get("execution_history", [])
                viz_analysis_type = enhanced_params.get("analysis", "gsea")  # What type of viz is requested
                
                found_matching_analysis = False
                for recent_execution in reversed(execution_history[-10:]):
                    step_data = recent_execution.get("step", {})
                    
                    # Only consider successful enrichment analyses
                    if (recent_execution.get("success") and 
                        step_data.get("function_name") == "perform_enrichment_analyses"):
                        
                        step_params = step_data.get("parameters", {})
                        analyses_performed = step_params.get("analyses", [])
                        cell_type = step_params.get("cell_type")
                        
                        # Check if this analysis matches what the viz needs
                        if viz_analysis_type in analyses_performed and cell_type:
                            enhanced_params["cell_type"] = cell_type
                            logger.info(f"✅ Found matching {viz_analysis_type} analysis for visualization: {cell_type}")
                            found_matching_analysis = True
                            break
                
                if not found_matching_analysis:
                    logger.info(f"❌ No successful {viz_analysis_type} enrichment analysis found to visualize")
                    enhanced_params["_should_fail"] = True
                    enhanced_params["_fail_reason"] = f"No {viz_analysis_type} enrichment analysis results available to visualize"
        
        # Special handling for display_cell_count_comparison - aggregate results from compare_cell_counts steps
        if step.function_name == "display_cell_count_comparison":
            logger.info(f"🔍 AGGREGATION: Handling display_cell_count_comparison - aggregating compare_cell_counts results")

            execution_history = state.get("execution_history", [])
            cell_types_data = {}

            # Find all successful compare_cell_counts executions
            for execution in execution_history:
                step_data = execution.get("step", {})
                if (execution.get("success") and
                    step_data.get("function_name") == "compare_cell_counts"):

                    cell_type = step_data.get("parameters", {}).get("cell_type")
                    result = execution.get("result")

                    if cell_type and result:
                        # Handle different result formats
                        if isinstance(result, dict) and "count_results" in result:
                            # Enhanced format from hierarchy wrapper
                            cell_types_data[cell_type] = result["count_results"]
                        elif isinstance(result, list):
                            # Direct format from compare_cell_count function
                            cell_types_data[cell_type] = result
                        logger.info(f"🔍 AGGREGATION: Added data for {cell_type}: {len(result) if isinstance(result, list) else 'dict'} entries")

            if cell_types_data:
                enhanced_params["cell_types_data"] = cell_types_data
                logger.info(f"✅ AGGREGATION: Successfully aggregated data for {len(cell_types_data)} cell types")
            else:
                logger.info(f"❌ AGGREGATION: No compare_cell_counts results found to aggregate")
                enhanced_params["_should_fail"] = True
                enhanced_params["_fail_reason"] = "No compare_cell_counts results available to visualize"

        # Special handling for display_cell_count_stacked_plot - aggregate all related cell types
        if step.function_name == "display_cell_count_stacked_plot":
            logger.info(f"🔍 STACKED AGGREGATION: Handling display_cell_count_stacked_plot - aggregating cell types for single plot")

            # Check if this step already has multiple cell types specified
            existing_cell_types = enhanced_params.get("cell_types", [])
            if not existing_cell_types:
                # Convert single cell_type to list if needed
                single_cell_type = enhanced_params.get("cell_type")
                if single_cell_type:
                    existing_cell_types = [single_cell_type]

            # Look for other pending display_cell_count_stacked_plot steps in the execution plan
            execution_plan = state.get("execution_plan", {})
            current_step_index = state.get("current_step_index", 0)

            aggregated_cell_types = set(existing_cell_types)
            steps_to_skip = []

            # Check upcoming steps for more cell types to include in the same plot
            if "steps" in execution_plan:
                for i, future_step_data in enumerate(execution_plan["steps"][current_step_index + 1:], start=current_step_index + 1):
                    future_step = future_step_data if hasattr(future_step_data, 'function_name') else type('obj', (object,), future_step_data)()

                    if (hasattr(future_step, 'function_name') and
                        future_step.function_name == "display_cell_count_stacked_plot"):

                        # Get cell types from future step
                        future_params = getattr(future_step, 'parameters', {}) or future_step_data.get('parameters', {})
                        future_cell_types = future_params.get("cell_types", [])
                        if not future_cell_types:
                            future_cell_type = future_params.get("cell_type")
                            if future_cell_type:
                                future_cell_types = [future_cell_type]

                        # Add to aggregated list
                        aggregated_cell_types.update(future_cell_types)
                        steps_to_skip.append(i)
                        logger.info(f"🔍 STACKED AGGREGATION: Found future step {i} with cell types: {future_cell_types}")

            # Update parameters with aggregated cell types
            if aggregated_cell_types:
                enhanced_params["cell_types"] = list(aggregated_cell_types)
                # Remove cell_type parameter if it exists (use cell_types instead)
                enhanced_params.pop("cell_type", None)
                logger.info(f"✅ STACKED AGGREGATION: Aggregated {len(aggregated_cell_types)} cell types for single plot: {list(aggregated_cell_types)}")

                # Mark future steps to be skipped
                if steps_to_skip:
                    state.setdefault("_steps_to_skip", set()).update(steps_to_skip)
                    logger.info(f"🚫 STACKED AGGREGATION: Marked steps {steps_to_skip} for skipping (already included in aggregated plot)")
            else:
                logger.info(f"⚠️ STACKED AGGREGATION: No cell types found for stacked plot")
                enhanced_params["_should_fail"] = True
                enhanced_params["_fail_reason"] = "No cell types specified for stacked plot"
        
        return enhanced_params
    
    def _handle_successful_step(self, state: ChatState, step: ExecutionStep, result: Any):
        """Handle a successful step execution."""
        # Store results
        state["function_result"] = result
        state["function_name"] = step.function_name
        state["function_args"] = step.parameters
        
        # Update hierarchy manager for process_cells steps
        if step.function_name == "process_cells" and self.hierarchy_manager:
            logger.info(f"🔍 EXECUTION NODE: About to call CellTypeExtractor.extract_from_annotation_result() for process_cells step")
            new_cell_types = self.cell_type_extractor.extract_from_annotation_result(result)
            if new_cell_types:
                logger.info(f"🧬 Updating hierarchy manager with new cell types: {new_cell_types}")
                self.hierarchy_manager.update_after_process_cells(
                    step.parameters.get("cell_type", "unknown"),
                    new_cell_types
                )
                
                # Update available cell types in state
                state["available_cell_types"] = list(set(state["available_cell_types"] + new_cell_types))
                for new_type in new_cell_types:
                    logger.info(f"✅ Discovered new cell type: {new_type}")
            else:
                logger.info("⚠️ No new cell types discovered from process_cells")
        
        # Update available cell types if this was a successful process_cells step
        if step.function_name == "process_cells" and result:
            self._update_available_cell_types_from_result(state, result)
    
    
    def _record_execution(self, state: ChatState, step_data: Dict[str, Any], result: Any, 
                         success: bool, error_msg: str, original_function_name: str):
        """Record execution in history and function tracking."""
        # Record function execution in history
        self.history_manager.record_execution(
            function_name=step_data.get("function_name"),
            parameters=step_data.get("parameters", {}),
            result=result,
            success=success,
            error=error_msg
        )
        
        # Record execution in state using structured storage approach
        result_storage = self._store_execution_result(step_data, result, success, original_function_name)
        
        # CRITICAL FIX: Create a deep copy of step_data to prevent mutation issues
        step_data_copy = copy.deepcopy(step_data)
        
        # DEBUG: Check if function name was mutated
        # current_function_name = step_data.get("function_name", "unknown")
        current_function_name = step_data.get("function_name") or (original_function_name or "direct_answer")
        if current_function_name != original_function_name:
            logger.info(f"⚠️ MUTATION DETECTED: function_name changed from '{original_function_name}' to '{current_function_name}'")
            # Fix the function name in the copy
            # step_data_copy["function_name"] = original_function_name
            # logger.info(f"🔧 CORRECTED: Restored function_name to '{original_function_name}' in execution history")
            step_data_copy["function_name"] = current_function_name #ADDEDAUG
            logger.info(f"🔧 NORMALIZED: Using safe function_name '{current_function_name}' in execution history") #ADDEDAUG

        
        state["execution_history"].append({
            "step_index": state["current_step_index"],
            "step": step_data_copy,  # Use copy to prevent mutation
            "success": success,
            "result": result_storage["result"],  # Full structure preserved for critical functions
            "result_type": result_storage["result_type"],
            "result_summary": result_storage["result_summary"],  # For logging
            "error": error_msg
        })
        
        # Log storage decision for monitoring
        self._log_storage_decision(result_storage, original_function_name)
    
    def _store_execution_result(self, step_data: Dict, result: Any, success: bool, original_function_name: str = None) -> Dict[str, Any]:
        """
        Intelligent result storage that preserves structure for critical functions
        """
        # Use provided original function name or fallback to step_data
        # function_name = original_function_name or step_data.get("function_name", "")
        function_name = (original_function_name or step_data.get("function_name") or "direct_answer")         #ADDEDAUG

        #ADDEDAUG
        if function_name is None:
            function_name = ""
        #ADDEDAUG

        
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
        


        elif isinstance(function_name, str) and function_name.startswith("display_") and success:
            viz_payload = result

            # If the plot is a huge inline HTML string, persist it as a file_ref
            if isinstance(result, str) and len(result) > MAX_INLINE_HTML_BYTES:
                os.makedirs("figures", exist_ok=True)
                ref = f"figures/plot_{uuid.uuid4().hex}.html"
                with open(ref, "w", encoding="utf-8") as f:
                    f.write(result)
                viz_payload = {"type": "file_ref", "path": ref}

            return {
                "result_type": "visualization",
                "result": viz_payload,
                "result_metadata": self._extract_viz_metadata(function_name, viz_payload),
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
        if isinstance(result, str):
            return {
                "visualization_type": function_name,
                "html_length": len(result),
                "contains_html": ("<div" in result or "<html" in result)
            }
        if isinstance(result, dict):
            if result.get("type") == "file_ref":
                return {"visualization_type": function_name, "file_ref": result.get("path")}
            if result.get("multiple_plots"):
                return {
                    "visualization_type": function_name,
                    "bundle": True,
                    "num_plots": len(result.get("plots", []))
                }
        return {"visualization_type": function_name}
    
    def _log_storage_decision(self, result_storage: Dict[str, Any], function_name: str):
        """Log storage decision for monitoring."""
        if result_storage["result_type"] == "structured":
            logger.info(f"📊 Structured storage: {function_name} - Full data preserved")
        elif result_storage["result_type"] == "visualization":
            logger.info(f"🎨 Visualization storage: {function_name} - HTML preserved")
        else:
            logger.info(f"📄 Text storage: {function_name} - Truncated to 500 chars")
    
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
                new_available = current_available.union(set(discovered_types))
                state["available_cell_types"] = list(new_available)
                
                logger.info(f"✅ Updated available cell types with discoveries: {discovered_types}")
                logger.info(f"🧬 Total available cell types: {len(state['available_cell_types'])}")
        
        except Exception as e:
            logger.info(f"⚠️ Error updating available cell types: {e}")
    
    
    def _should_continue_execution(self, state: ChatState, steps_executed: int) -> bool:
        """Determine if execution should continue in the same executor run.
        
        This method implements intelligent batching to reduce evaluation rounds:
        - Supplementary steps (added by evaluator) are executed together
        - Visualization steps following analyses are executed together
        - Stops after certain milestones to allow evaluation
        """
        # Check if we've reached the end of the plan
        current_index = state.get("current_step_index", 0)
        total_steps = len(state.get("execution_plan", {}).get("steps", []))
        
        if current_index >= total_steps:
            return False
        
        # Get the next step if available
        next_step = state["execution_plan"]["steps"][current_index]
        
        # Check if this is a supplementary step sequence
        if "Post-evaluation:" in next_step.get("description", ""):
            logger.info(f"🚀 Continuing execution - next step is supplementary: {next_step.get('function_name')}")
            return True
        
        # Check if next step is a visualization for the same cell type
        if steps_executed > 0:
            last_executed = state["execution_history"][-1]
            last_step = last_executed.get("step", {})
            last_cell_type = last_step.get("parameters", {}).get("cell_type")
            next_cell_type = next_step.get("parameters", {}).get("cell_type")
            
            # Continue if it's a visualization for the same cell type
            if (last_cell_type == next_cell_type and 
                next_step.get("function_name", "").startswith("display_") and
                not last_step.get("function_name", "").startswith("display_")):
                logger.info(f"🎨 Continuing execution - visualization for same cell type: {next_cell_type}")
                return True
        
        # Check if we're in a search + visualization sequence
        if (next_step.get("function_name") == "display_enrichment_visualization" and
            steps_executed > 0 and 
            state["execution_history"][-1].get("step", {}).get("function_name") == "search_enrichment_semantic"):
            logger.info(f"🔍➡️🎨 Continuing execution - visualization after search")
            return True
        
        # Default: stop after each step for regular workflow steps
        return False
    
    def _execute_final_question(self, state: ChatState) -> str:
        """Execute final question step - placeholder implementation."""
        return "Final question processing completed"