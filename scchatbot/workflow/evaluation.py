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
        Evaluator with light consolidation, validation, and routing.
        
        Combines functionality from evaluator + status_checker:
        - Step management and completion detection
        - Light plan consolidation (remove consecutive duplicates only)
        - Plan validation (add validation steps)
        - Missing cell type warnings (but preserve discovery paths)
        - Routing to executor or jury
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with processed execution plan
        """
        
        # Handle plan conversion: initial_plan → execution_plan
        if not state.get("execution_plan") and state.get("initial_plan"):
            print("🔄 Evaluator: Converting initial_plan to execution_plan")
            state["execution_plan"] = state["initial_plan"].copy()
            state["execution_plan"]["original_question"] = state.get("current_message", "")
        
        # Defensive check: ensure execution_plan exists
        if not state.get("execution_plan"):
            print("⚠️ Evaluator: No execution plan or initial plan found, cannot proceed")
            state["conversation_complete"] = True
            return state
        
        # Defensive check: ensure execution_plan has steps
        if not state["execution_plan"].get("steps"):
            print("⚠️ Evaluator: Execution plan has no steps, cannot proceed")
            state["conversation_complete"] = True
            return state
        
        # Check if execution is complete
        current_step_index = state.get("current_step_index", 0)
        total_steps = len(state["execution_plan"]["steps"])
        
        if current_step_index >= total_steps:
            print("🏁 All execution steps complete - routing to jury for evaluation")
            return state  # Will be routed to jury by conditional routing
        
        # Check if plan has already been processed to avoid duplicate work
        # IMPORTANT: Also check if this plan has already been enhanced by counting validation steps
        validation_steps = len([s for s in state["execution_plan"]["steps"] if s.get("step_type") == "validation"])
        process_cells_steps = len([s for s in state["execution_plan"]["steps"] if s.get("function_name") == "process_cells"])
        
        if state.get("plan_processed") or (validation_steps > 0 and validation_steps >= process_cells_steps):
            print(f"✅ Plan already processed (validation_steps={validation_steps}, process_cells_steps={process_cells_steps}), continuing with execution")
            return state  # Will be routed to executor
            
        print(f"🔧 Evaluator processing plan with {total_steps} steps...")
        
        # Apply light processing to the plan
        original_step_count = len(state["execution_plan"]["steps"])
        
        # 1. Light consolidation - only remove exact consecutive duplicates but preserve discovery paths
        state["execution_plan"] = self._light_consolidate_process_cells(state["execution_plan"])
        
        # 2. Light validation - only log warnings for missing cell types (don't remove steps)
        self._log_missing_cell_type_warnings(state["execution_plan"])
        
        # 3. Add validation steps after process_cells operations
        state["execution_plan"] = self._add_validation_steps_after_process_cells(state["execution_plan"])
        
        final_step_count = len(state["execution_plan"]["steps"])
        print(f"✅ Evaluator: {original_step_count} → {final_step_count} steps")
        print(f"   • {len([s for s in state['execution_plan']['steps'] if s.get('function_name') == 'process_cells'])} process_cells steps")
        print(f"   • {len([s for s in state['execution_plan']['steps'] if s.get('step_type') == 'validation'])} validation steps")
        print(f"   • {len([s for s in state['execution_plan']['steps'] if s.get('step_type') == 'analysis'])} analysis steps")
        
        # Mark plan as processed to prevent duplicate processing
        state["plan_processed"] = True
        
        return state  # Will be routed to executor

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
                    
                    validation_step = {
                        "step_type": "validation",
                        "function_name": "validate_processing_results",
                        "parameters": {
                            "processed_parent": cell_type,
                            "expected_children": []  # Will be populated during execution
                        },
                        "description": f"Validate that {cell_type} processing discovered expected cell types",
                        "expected_outcome": "Confirm expected cell types are available or handle gracefully",
                        "target_cell_type": None
                    }
                    steps_with_validation.append(validation_step)
                    print(f"   ✅ Added validation step after process_cells({cell_type})")
        
        # Update execution plan
        execution_plan["steps"] = steps_with_validation
        
        validation_count = len([s for s in steps_with_validation if s.get("step_type") == "validation"])
        print(f"✅ Added {validation_count} validation steps")
        
        return execution_plan