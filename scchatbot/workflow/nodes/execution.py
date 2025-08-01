"""
Execution node implementation.

This module contains the ExecutorNode which executes the steps in the execution plan,
manages function calls, and handles validation steps.
"""

import copy
from typing import Dict, Any, List

from ...cell_type_models import ChatState, ExecutionStep
from ..node_base import BaseWorkflowNode


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
        
        # Continue executing while there are steps, with special handling for supplementary steps
        steps_executed = 0
        continue_execution = True
        
        while continue_execution and state.get("current_step_index", 0) < len(state.get("execution_plan", {}).get("steps", [])):
            if not state["execution_plan"] or state["current_step_index"] >= len(state["execution_plan"]["steps"]):
                state["conversation_complete"] = True
                break
                
            step_data = state["execution_plan"]["steps"][state["current_step_index"]]
            
            # Check if this step should be skipped
            if step_data.get("skip_reason"):
                self._handle_skipped_step(state, step_data)
                steps_executed += 1
                state["current_step_index"] += 1
                continue
            
            step = ExecutionStep(**step_data)
            
            print(f"🔄 Executing step {state['current_step_index'] + 1}: {step.description}")
            
            # DEBUG: Log the original function name to track mutations
            original_function_name = step_data.get("function_name", "unknown")
            print(f"🔍 STORAGE DEBUG: Original function_name from plan: '{original_function_name}'")
            
            success = False
            result = None
            error_msg = None
            
            try:
                success, result, error_msg = self._execute_step(state, step)
                
                if success:
                    print(f"✅ Step {state['current_step_index'] + 1} completed successfully")
                    self._handle_successful_step(state, step, result)
                
            except Exception as e:
                error_msg = str(e)
                success = False
                print(f"❌ Step {state['current_step_index'] + 1} failed: {error_msg}")
                state["errors"].append(f"Step {state['current_step_index'] + 1} failed: {error_msg}")
            
            # Record execution
            self._record_execution(state, step_data, result, success, error_msg, original_function_name)
            
            # Simple error handling - always advance to prevent infinite retry loop
            state["current_step_index"] += 1
            steps_executed += 1
            
            if success:
                print(f"🔄 Advanced to step {state['current_step_index'] + 1}")
            else:
                print(f"❌ Step failed: {error_msg}")
                print(f"📝 Recording error and advancing to step {state['current_step_index'] + 1} (no retries for deterministic operations)")
            
            # Check if we should continue executing (batch supplementary steps)
            continue_execution = self._should_continue_execution(state, steps_executed)
            
            if not continue_execution:
                print(f"🛑 Stopping execution after {steps_executed} steps")
                break
        
        self._log_node_complete("Executor", state)
        return state
    
    def _handle_skipped_step(self, state: ChatState, step_data: Dict[str, Any]):
        """Handle a step that should be skipped."""
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
        
        # Note: state["current_step_index"] is now incremented in the main loop
    
    def _execute_step(self, state: ChatState, step: ExecutionStep) -> tuple[bool, Any, str]:
        """Execute a single step and return success, result, error_msg."""
        # Handle final question step
        if step.step_type == "final_question":
            print("🎯 Executing final comprehensive question...")
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
            print(f"🔍 STEP DEBUG: Calling visualization function '{step.function_name}' with step parameters: {step.parameters}")
        
        # For visualization functions, enhance parameters with cell_type from execution context if missing
        enhanced_params = self._enhance_visualization_params(state, step)
        
        # Check if enhancement determined this step should fail
        if enhanced_params.get("_should_fail"):
            error_msg = enhanced_params.get("_fail_reason", "Step cannot be executed")
            print(f"❌ Step execution blocked: {error_msg}")
            return False, None, error_msg
        
        # Remove internal flags before calling function
        enhanced_params.pop("_should_fail", None)
        enhanced_params.pop("_fail_reason", None)
        
        func = self.function_mapping[step.function_name]
        result = func(**enhanced_params)
        
        return True, result, None
    
    def _enhance_visualization_params(self, state: ChatState, step: ExecutionStep) -> Dict[str, Any]:
        """Enhance visualization parameters with context if needed."""
        enhanced_params = step.parameters.copy()
        
        if step.function_name in self.visualization_functions and "cell_type" not in enhanced_params:
            # CRITICAL: Don't fallback to unrelated cell types for visualization
            # Check if we have ANY successful enrichment analysis for this visualization type
            execution_history = state.get("execution_history", [])
            viz_analysis_type = enhanced_params.get("analysis", "gsea")  # What type of viz is requested
            
            found_matching_analysis = False
            for recent_execution in reversed(execution_history[-10:]):  # Check more history
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
                        print(f"✅ Found matching {viz_analysis_type} analysis for visualization: {cell_type}")
                        found_matching_analysis = True
                        break
            
            if not found_matching_analysis:
                # NO FALLBACK - Return error instead
                print(f"❌ No successful {viz_analysis_type} enrichment analysis found to visualize")
                # Mark this step to fail with clear error
                enhanced_params["_should_fail"] = True
                enhanced_params["_fail_reason"] = f"No {viz_analysis_type} enrichment analysis results available to visualize"
        
        return enhanced_params
    
    def _handle_successful_step(self, state: ChatState, step: ExecutionStep, result: Any):
        """Handle a successful step execution."""
        # Store results
        state["function_result"] = result
        state["function_name"] = step.function_name
        state["function_args"] = step.parameters
        
        # Update hierarchy manager for process_cells steps
        if step.function_name == "process_cells" and self.hierarchy_manager:
            print(f"🔍 EXECUTION NODE: About to call CellTypeExtractor.extract_from_annotation_result() for process_cells step")
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
        
        # Log storage decision for monitoring
        self._log_storage_decision(result_storage, original_function_name)
    
    def _store_execution_result(self, step_data: Dict, result: Any, success: bool, original_function_name: str = None) -> Dict[str, Any]:
        """
        Intelligent result storage that preserves structure for critical functions
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
    
    def _log_storage_decision(self, result_storage: Dict[str, Any], function_name: str):
        """Log storage decision for monitoring."""
        if result_storage["result_type"] == "structured":
            print(f"📊 Structured storage: {function_name} - Full data preserved")
        elif result_storage["result_type"] == "visualization":
            print(f"🎨 Visualization storage: {function_name} - HTML preserved")
        else:
            print(f"📄 Text storage: {function_name} - Truncated to 500 chars")
    
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
                
                print(f"✅ Updated available cell types with discoveries: {discovered_types}")
                print(f"🧬 Total available cell types: {len(state['available_cell_types'])}")
        
        except Exception as e:
            print(f"⚠️ Error updating available cell types: {e}")
    
    
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
            print(f"🚀 Continuing execution - next step is supplementary: {next_step.get('function_name')}")
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
                print(f"🎨 Continuing execution - visualization for same cell type: {next_cell_type}")
                return True
        
        # Check if we're in a search + visualization sequence
        if (next_step.get("function_name") == "display_enrichment_visualization" and
            steps_executed > 0 and 
            state["execution_history"][-1].get("step", {}).get("function_name") == "search_enrichment_semantic"):
            print(f"🔍➡️🎨 Continuing execution - visualization after search")
            return True
        
        # Default: stop after each step for regular workflow steps
        return False
    
    def _execute_final_question(self, state: ChatState) -> str:
        """Execute final question step - placeholder implementation."""
        return "Final question processing completed"