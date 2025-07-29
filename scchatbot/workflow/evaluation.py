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
        Log warnings for missing cell types without removing steps.
        
        The planner should have already handled cell discovery, so this method
        only logs warnings for missing cell types but preserves the discovery paths.
        
        Args:
            execution_plan: Plan to validate cell types for
        """
        steps = execution_plan.get("steps", [])
        
        for step in steps:
            # Get cell type from step
            cell_type = step.get("parameters", {}).get("cell_type")
            
            if cell_type and cell_type != "overall":
                # Check if cell type is valid using hierarchy manager (light validation)
                if self.hierarchy_manager:
                    try:
                        resolved_series, metadata = self.hierarchy_manager.resolve_cell_type_for_analysis(cell_type)
                        
                        if metadata["resolution_method"] == "not_found":
                            print(f"⚠️ Evaluator: Step references missing cell type '{cell_type}' - {step.get('function_name', 'unknown')}")
                            print(f"   Note: This may be a target cell type that will be discovered by earlier process_cells steps.")
                        elif metadata["resolution_method"] == "needs_processing":
                            print(f"🔍 Evaluator: Step requires processing for '{cell_type}' - {step.get('function_name', 'unknown')}")
                        else:
                            print(f"✅ Evaluator: Valid cell type '{cell_type}' for {step.get('function_name', 'unknown')}")
                            
                    except Exception as e:
                        print(f"⚠️ Evaluator: Error validating cell type '{cell_type}': {e}")

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