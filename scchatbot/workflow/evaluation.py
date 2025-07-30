"""
Evaluation logic for workflow nodes.

This module contains evaluation-related functionality extracted from workflow_nodes.py:
- evaluator_node implementation with light consolidation, validation, and routing
- Plan consolidation logic to remove consecutive duplicates
- Missing cell type validation and warning system
- Validation step insertion after process_cells operations
"""

from typing import Dict, Any, List
from ..cell_type_models import ChatState


class EvaluationMixin:
    """
    Evaluation logic mixin containing methods for plan evaluation and validation.
    
    This mixin provides functionality for:
    - Evaluating execution results and determining next steps
    - Light consolidation of execution plans
    - Validation of cell types and plan structure
    - Adding validation steps after process_cells operations
    """
    
    def evaluator_node(self, state: ChatState) -> ChatState:
        """
        Evaluator reviews execution results and prepares for response generation.
        
        NEW ROLE: Post-execution review (called AFTER all execution steps complete)
        - Review execution history for completeness
        - Identify any failed steps or missing results
        - Prepare context for response generation
        - Mark conversation as complete
        
        Args:
            state: Current workflow state with completed execution
            
        Returns:
            Updated state ready for response generation
        """
        
        print("🏁 Evaluator: Reviewing execution results...")
        
        # Defensive check: ensure execution_plan exists
        if not state.get("execution_plan"):
            print("⚠️ Evaluator: No execution plan found")
            state["conversation_complete"] = True
            return state
        
        # Count successful and failed steps
        execution_history = state.get("execution_history", [])
        successful_steps = [h for h in execution_history if h.get("success", False)]
        failed_steps = [h for h in execution_history if not h.get("success", False)]
        
        print(f"✅ Execution Summary:")
        print(f"   • Total steps executed: {len(execution_history)}")
        print(f"   • Successful: {len(successful_steps)}")
        print(f"   • Failed: {len(failed_steps)}")
        
        # Log any failures for response generation awareness
        if failed_steps:
            print(f"⚠️ {len(failed_steps)} steps failed:")
            for failed in failed_steps:
                step_desc = failed.get("step", {}).get("description", "Unknown step")
                error = failed.get("error", "Unknown error")
                print(f"   • {step_desc}: {error}")
        
        # Review available results
        if self.history_manager:
            available_results = self.history_manager.get_available_results()
            if available_results:
                print("📊 Available analysis results:")
                if "enrichment_analyses" in available_results:
                    print(f"   • Enrichment analyses: {list(available_results['enrichment_analyses'].keys())}")
                if "dea_analyses" in available_results:
                    print(f"   • DEA analyses: {list(available_results['dea_analyses'])}")
                if "processed_cell_types" in available_results:
                    print(f"   • Processed cell types: {available_results['processed_cell_types']}")
        
        # NEW: Add post-execution evaluation to check for missing analyses
        print("🔍 Evaluator: Starting post-execution evaluation...")
        
        # Check if we have the post-execution evaluation methods available (from CoreNodes)
        if hasattr(self, '_post_execution_evaluation'):
            evaluation_result = self._post_execution_evaluation(state)
            
            # Store evaluation result in state for response generator
            state["post_execution_evaluation"] = evaluation_result
            
            if evaluation_result.get("supplementary_steps"):
                print(f"🔍 Post-execution evaluation found {len(evaluation_result['supplementary_steps'])} additional steps needed")
                
                # Add supplementary steps to execution plan
                supplementary_steps = evaluation_result["supplementary_steps"]
                current_steps = state["execution_plan"].get("steps", [])
                state["execution_plan"]["steps"] = current_steps + supplementary_steps
                
                # Reset state to continue execution with supplementary steps
                state["current_step_index"] = len(current_steps)  # Start from first supplementary step
                state["conversation_complete"] = False  # Continue execution
                
                print(f"📋 Added {len(supplementary_steps)} supplementary steps to execution plan")
                print("🔄 Returning to executor to process supplementary steps...")
                
                # Return state for continued execution
                return state
            else:
                print("✅ Post-execution evaluation: No additional steps needed")
                if evaluation_result.get("question_type"):
                    print(f"📊 Question type: {evaluation_result['question_type']}")
                if evaluation_result.get("analysis_relevance"):
                    print(f"📝 Generated analysis relevance hints for response generation")
        else:
            print("⚠️ Post-execution evaluation methods not available")
        
        # Mark conversation as ready for response generation
        state["conversation_complete"] = True
        
        print("✅ Evaluator: Execution review complete, proceeding to response generation")
        
        return state

    def _light_consolidate_process_cells(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Very light consolidation - only remove exact consecutive duplicates.
        
        Preserves discovery paths like: Immune cell -> T cell -> Regulatory T cell
        
        Args:
            execution_plan: Plan with potentially duplicate consecutive steps
            
        Returns:
            Plan with consecutive duplicates removed
        """
        steps = execution_plan.get("steps", [])
        print(f"🔄 Light consolidation of {len(steps)} steps...")
        
        if len(steps) <= 1:
            return execution_plan
        
        consolidated_steps = [steps[0]]  # Always keep the first step
        
        for i in range(1, len(steps)):
            current_step = steps[i]
            previous_step = steps[i-1]
            
            # Only remove if it's an exact duplicate of the previous step
            if (current_step.get("function_name") == "process_cells" and
                previous_step.get("function_name") == "process_cells" and
                current_step.get("parameters", {}).get("cell_type") == 
                previous_step.get("parameters", {}).get("cell_type")):
                
                cell_type = current_step.get("parameters", {}).get("cell_type", "unknown")
                print(f"   🗑️ Removing consecutive duplicate process_cells({cell_type})")
            else:
                consolidated_steps.append(current_step)
        
        execution_plan["steps"] = consolidated_steps
        print(f"✅ Light consolidation: {len(steps)} → {len(consolidated_steps)} steps")
        
        return execution_plan

    def _log_missing_cell_type_warnings(self, execution_plan: Dict[str, Any]) -> None:
        """
        Light validation without duplicate hierarchy resolution.
        
        The planner already handled cell discovery and path finding, so this method
        just provides a summary without re-doing the expensive hierarchy lookups.
        
        Args:
            execution_plan: Plan to validate cell types for
        """
        steps = execution_plan.get("steps", [])
        discovery_steps = [s for s in steps if s.get("function_name") == "process_cells"]
        analysis_steps = [s for s in steps if s.get("function_name") in ["dea_split_by_condition", "perform_enrichment_analyses"]]
        
        print(f"📋 Plan validation summary:")
        print(f"   • {len(discovery_steps)} discovery steps")
        print(f"   • {len(analysis_steps)} analysis steps")
        
        if discovery_steps:
            print(f"🔄 Discovery sequence:")
            for step in discovery_steps:
                parent = step.get("parameters", {}).get("cell_type")
                target = step.get("target_cell_type", "unknown targets")
                print(f"   → process_cells({parent}) → {target}")
        
        # Skip expensive hierarchy resolution - planner already handled this

    def _add_validation_steps_after_process_cells(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add validation steps after each process_cells operation.
        
        Only adds validation steps if they don't already exist to prevent infinite loops.
        
        Args:
            execution_plan: Plan to add validation steps to
            
        Returns:
            Plan with validation steps added after process_cells operations
        """
        steps = execution_plan.get("steps", [])
        steps_with_validation = []
        
        print(f"🔍 Adding validation steps for process_cells operations...")
        
        for i, step in enumerate(steps):
            steps_with_validation.append(step)
            
            # Add validation step after process_cells operations
            if step.get("function_name") == "process_cells":
                cell_type = step.get("parameters", {}).get("cell_type")
                if cell_type:
                    # Check if the next step is already a validation step for this cell type
                    next_step = steps[i + 1] if i + 1 < len(steps) else None
                    if (next_step and 
                        next_step.get("step_type") == "validation" and 
                        next_step.get("parameters", {}).get("processed_parent") == cell_type):
                        print(f"   ⏭️ Validation step already exists after process_cells({cell_type})")
                        continue
                    
                    # Extract expected children from step if available
                    expected_children = step.get("expected_children", [])
                    
                    validation_step = {
                        "step_type": "validation",
                        "function_name": "validate_processing_results",
                        "parameters": {
                            "processed_parent": cell_type,
                            "expected_children": expected_children
                        },
                        "description": f"Validate that {cell_type} processing discovered expected cell types",
                        "expected_outcome": "Confirm expected cell types are available or handle gracefully",
                        "target_cell_type": None
                    }
                    steps_with_validation.append(validation_step)
                    
                    if expected_children:
                        print(f"   ✅ Added validation step after process_cells({cell_type}) expecting: {expected_children}")
                    else:
                        print(f"   ✅ Added validation step after process_cells({cell_type}) - no specific children expected")
        
        # Update execution plan
        execution_plan["steps"] = steps_with_validation
        
        validation_count = len([s for s in steps_with_validation if s.get("step_type") == "validation"])
        print(f"✅ Added {validation_count} validation steps")
        
        return execution_plan
    
    def step_evaluator_node(self, state: ChatState) -> ChatState:
        """
        Intelligent step-by-step evaluator based on original system logic.
        
        Analyzes each completed step and takes context-aware actions:
        - For process_cells: Validates discovered types and updates remaining plan
        - For failed analyses: Records errors and skips related steps
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
        
        # CRITICAL FIX: Don't evaluate skipped steps - nothing to evaluate
        if last_execution.get("skipped", False):
            print(f"⏭️ Step {step_index + 1} was skipped - no evaluation needed")
            state["last_step_evaluation"] = {
                "status": "skipped", 
                "step_index": step_index,
                "critical_failure": False,
                "skip_reason": last_execution.get("error", "Unknown skip reason")
            }
            return state
        
        # Get function name from step data
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
        
        from datetime import datetime
        
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
        current_cell_types = set(state.get("available_cell_types", []))
        
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
                if hasattr(self, 'hierarchy_manager') and self.hierarchy_manager:
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
    
    def _update_plan_for_missing_cell_types(self, state: ChatState, missing_types: list) -> list:
        """
        Skip FUTURE steps referencing missing cell types.
        Only skip steps that haven't been executed yet - don't retroactively mark past steps.
        """
        actions_taken = []
        execution_plan = state.get("execution_plan", {})
        steps = execution_plan.get("steps", [])
        current_step_index = state.get("current_step_index", 0)
        
        print(f"🔧 PLAN UPDATE: Skipping FUTURE steps for missing types {missing_types}")
        print(f"   Current step index: {current_step_index}, Total steps: {len(steps)}")
        
        steps_skipped = 0
        
        # CRITICAL FIX: Only check steps AFTER current step index (don't retroactively skip executed steps)
        for i in range(current_step_index, len(steps)):
            step = steps[i]
            step_cell_type = step.get("parameters", {}).get("cell_type")
            step_description = step.get("description", "")
            
            # Check if step references missing cell types in parameters OR description
            references_missing_type = False
            missing_ref = None
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
            
            # ALSO check if this is a visualization step that depends on skipped analysis
            if not references_missing_type and step.get("function_name", "").startswith("display_"):
                # Check if this visualization depends on any skipped analysis steps
                for j in range(current_step_index, i):
                    prev_step = steps[j]
                    if prev_step.get("skip_reason") and prev_step.get("function_name") in ["perform_enrichment_analyses", "dea_split_by_condition"]:
                        # This visualization depends on skipped analysis
                        references_missing_type = True
                        missing_ref = f"depends on skipped {prev_step.get('function_name')}"
                        print(f"🔗 Visualization step {i+1} depends on skipped analysis step {j+1}")
                        break
            
            # Skip if this step doesn't reference any missing cell types
            if not references_missing_type:
                continue
            
            # Skip if step is already marked for skipping
            if step.get("skip_reason"):
                continue
                
            print(f"🔍 Found FUTURE step {i+1} referencing missing cell type '{missing_ref}': {step.get('function_name')}")
            
            # Skip only future steps referencing missing cell types
            steps[i]["skip_reason"] = f"Cell type '{missing_ref}' was not discovered and is unavailable"
            actions_taken.append(f"Skipped future step {i+1}: {step.get('function_name')} (references {missing_ref})")
            steps_skipped += 1
            print(f"⏭️ Skipped FUTURE step {i+1}: {step.get('function_name')} (references {missing_ref})")
        
        if steps_skipped > 0:
            print(f"✅ Plan update complete: {steps_skipped} future steps marked for skipping")
        else:
            print("✅ No future steps needed skipping")
        
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
            
            # Check if this is a missing cell type error
            if ("not found" in error_msg.lower() or "None of" in error_msg or 
                "are in the [columns]" in error_msg):
                print(f"🔧 Detected missing cell type error for '{cell_type}' - updating plan...")
                
                # Mark this cell type as unavailable
                current_unavailable = state.get("unavailable_cell_types", [])
                if cell_type and cell_type not in current_unavailable:
                    state["unavailable_cell_types"] = current_unavailable + [cell_type]
                    print(f"📋 Added '{cell_type}' to unavailable cell types")
                
                # Skip remaining steps for this cell type
                actions = self._update_plan_for_missing_cell_types(state, [cell_type])
                evaluation["actions_taken"].extend(actions)
            
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
        
        from datetime import datetime
        
        state["execution_errors"].append({
            "step_index": execution.get("step_index", -1),
            "function_name": execution.get("step", {}).get("function_name", "unknown"),
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"📝 Recorded execution error: {error_message}")