"""
Validation node implementation.

This module contains validation-related functionality for workflow steps,
particularly for validating cell type discovery and processing results.
"""

from typing import Dict, Any, List

from ...cell_type_models import ChatState
from ..node_base import BaseWorkflowNode


class ValidationNode(BaseWorkflowNode):
    """
    Validation node that handles validation of processing results.
    
    Responsibilities:
    - Validate cell type discovery results
    - Check that expected cell types were found
    - Provide suggestions for missing cell types
    - Update available cell types based on validation results
    """
    
    def execute(self, state: ChatState) -> ChatState:
        """Main execution method for validation."""
        # This is typically called from ExecutorNode rather than standalone
        return state
    
    def validate_processing_results(self, processed_parent: str, expected_children: List[str]) -> Dict[str, Any]:
        """Validate that process_cells discovered the expected cell types"""
        self._log_node_start("Validation", {"current_message": f"Validating {processed_parent} -> {expected_children}"})
        
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
                subtypes_found = self._find_subtypes_in_available(expected_child, current_cell_types)
                
                if subtypes_found:
                    found_children.extend(subtypes_found)
                    print(f"✅ Subtype validation: '{expected_child}' satisfied by subtypes: {subtypes_found}")
                else:
                    missing_children.append(expected_child)
                    print(f"❌ Missing expected cell type: '{expected_child}' (no exact match or valid subtypes)")
        
        # Generate result based on findings
        result = self._generate_validation_result(
            expected_children, found_children, missing_children, current_cell_types
        )
        
        self._log_node_complete("Validation", {"result": result["status"]})
        return result
    
    def _find_subtypes_in_available(self, expected_child: str, current_cell_types: set) -> List[str]:
        """Find subtypes of expected child in available cell types."""
        subtypes_found = []
        if self.hierarchy_manager:
            for available_type in current_cell_types:
                try:
                    relation = self.hierarchy_manager.get_cell_type_relation(available_type, expected_child)
                    if relation.name == "DESCENDANT":
                        subtypes_found.append(available_type)
                except:
                    continue
        return subtypes_found
    
    def _generate_validation_result(self, expected_children: List[str], found_children: List[str], 
                                   missing_children: List[str], current_cell_types: set) -> Dict[str, Any]:
        """Generate comprehensive validation result."""
        if missing_children:
            print(f"⚠️ Validation Warning: Expected children not found: {missing_children}")
            print(f"   Available cell types: {sorted(current_cell_types)}")
            
            # Try to suggest alternatives
            suggestions = self._generate_suggestions(missing_children, current_cell_types)
            
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
    
    def _generate_suggestions(self, missing_children: List[str], current_cell_types: set) -> List[str]:
        """Generate suggestions for missing cell types."""
        suggestions = []
        for missing in missing_children:
            for available in current_cell_types:
                if self.hierarchy_manager:
                    try:
                        relation = self.hierarchy_manager.get_cell_type_relation(missing, available)
                        if relation.name in ["ANCESTOR", "DESCENDANT", "SIBLING"]:
                            suggestions.append(f"'{missing}' → '{available}'")
                    except:
                        continue
        return suggestions
    
    def create_validation_steps(self, discovery_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create validation steps for discovery operations."""
        validation_steps = []
        
        for step in discovery_steps:
            if step.get("function_name") == "process_cells":
                cell_type = step.get("parameters", {}).get("cell_type")
                expected_children = step.get("expected_children", [])
                
                if cell_type and expected_children:
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
                    validation_steps.append(validation_step)
        
        return validation_steps
    
    def update_remaining_steps_with_discovered_types(self, state: ChatState, validation_result: Dict[str, Any]) -> None:
        """
        Update remaining analysis steps to use actually discovered cell types instead of expected ones.
        
        This prevents analysis steps from failing when expected cell types weren't discovered
        but valid subtypes were found instead.
        """
        if not validation_result.get("found_children"):
            return
        
        # Get the mapping of expected to found cell types
        found_children = validation_result["found_children"]
        missing_children = validation_result.get("missing_children", [])
        
        # Update remaining steps in the execution plan
        execution_plan = state.get("execution_plan", {})
        steps = execution_plan.get("steps", [])
        current_index = state.get("current_step_index", 0)
        
        print(f"🔧 Updating remaining {len(steps) - current_index - 1} steps with discovered types...")
        
        updated_count = 0
        for i in range(current_index + 1, len(steps)):
            step = steps[i]
            
            # Skip if this is another validation step
            if step.get("step_type") == "validation":
                continue
            
            # Check if step references any missing cell types
            step_cell_type = step.get("parameters", {}).get("cell_type")
            target_cell_type = step.get("target_cell_type")
            
            # Update step parameters if they reference missing cell types
            updated = False
            
            if step_cell_type in missing_children:
                # Find suitable replacement from found children
                replacement = self._find_suitable_replacement(step_cell_type, found_children)
                if replacement:
                    step["parameters"]["cell_type"] = replacement
                    print(f"   • Updated step {i+1}: '{step_cell_type}' → '{replacement}'")
                    updated = True
                else:
                    # Mark step for skipping
                    step["skip_reason"] = f"Cell type '{step_cell_type}' not discovered"
                    print(f"   • Marked step {i+1} for skipping: {step_cell_type} not found")
                    updated = True
            
            if target_cell_type in missing_children:
                replacement = self._find_suitable_replacement(target_cell_type, found_children)
                if replacement:
                    step["target_cell_type"] = replacement
                    updated = True
            
            if updated:
                updated_count += 1
        
        print(f"✅ Updated {updated_count} remaining steps with discovered cell types")
    
    def _find_suitable_replacement(self, missing_type: str, found_children: List[str]) -> str:
        """Find a suitable replacement cell type from found children."""
        # For now, use simple heuristics. Could be enhanced with hierarchy logic.
        # Return the first found child that might be related
        if found_children:
            # Prefer exact matches first (shouldn't happen but safety check)
            if missing_type in found_children:
                return missing_type
            
            # Otherwise return first available (could be enhanced with similarity matching)
            return found_children[0]
        
        return None