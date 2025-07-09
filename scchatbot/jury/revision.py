"""
Jury revision and plan enhancement logic.

This module contains revision methods extracted from jury_system_main.py:
- targeted_revision_node(): Handle analysis revisions based on specific judge failures
- _create_targeted_revision_plan(): Create specific revision actions based on failed judge feedback
- _apply_targeted_revisions(): Apply targeted revisions to the execution plan
"""

from typing import Dict, Any, List
from ..shared import extract_cell_types_from_question, needs_cell_discovery, create_cell_discovery_steps


class PlanRevisionMixin:
    """
    Plan revision mixin for targeted analysis improvements.
    """

    def targeted_revision_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle analysis revisions based on specific judge failures.
        
        Args:
            state: Current chat state
            
        Returns:
            Updated state with revised execution plan
        """
        
        jury_verdicts = state.get("jury_verdicts", {})
        failed_judges = [name for name, verdict in jury_verdicts.items() 
                        if not verdict.get("pass", True)]
        
        print(f"🔬 Applying targeted revisions for failed judges: {failed_judges}")
        
        try:
            # Create targeted revision plan
            revision_plan = self._create_targeted_revision_plan(failed_judges, jury_verdicts, state)
            
            # Apply revisions to execution plan
            if revision_plan:
                updated_plan = self._apply_targeted_revisions(
                    state.get("execution_plan", {}), revision_plan, state
                )
                
                state["execution_plan"] = updated_plan
                state["current_step_index"] = 0  # Reset execution
                state["revision_applied"] = True
                state["revision_type"] = "targeted_analysis"
                
                print(f"✅ Applied {len(revision_plan)} targeted revisions")
            else:
                print("⚠️ No specific revisions identified, keeping original plan")
        
        except Exception as e:
            print(f"❌ Error in targeted revision: {e}")
            state["revision_error"] = str(e)
        
        return state
    
    def _create_targeted_revision_plan(self, failed_judges: List[str], jury_verdicts: Dict[str, Any], state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create specific revision actions based on failed judge feedback.
        """
        
        revisions = []
        
        print(f"🔍 Creating targeted revisions for {len(failed_judges)} failed judges: {failed_judges}")
        
        for judge_name in failed_judges:
            verdict = jury_verdicts.get(judge_name, {})
            issue = verdict.get("primary_issue", "")
            guidance = verdict.get("improvement_direction", "")
            
            print(f"🔍 Processing judge: {judge_name}")
            
            if judge_name == "workflow_judge":
                revisions.append({
                    "type": "workflow_improvement",
                    "issue": issue,
                    "action": "reorder_steps",
                    "guidance": guidance
                })
            
            elif judge_name == "efficiency_judge":
                redundant_steps = verdict.get("redundant_steps", [])
                if redundant_steps:
                    revisions.append({
                        "type": "efficiency_improvement",
                        "action": "remove_redundancy",
                        "redundant_steps": redundant_steps,
                        "guidance": guidance
                    })
            
            elif judge_name == "completeness_judge":
                missing_components = verdict.get("missing_components", [])
                if missing_components:
                    revisions.append({
                        "type": "completeness_improvement",
                        "action": "add_missing_steps",
                        "missing_components": missing_components,
                        "guidance": guidance
                    })
            
            elif judge_name == "user_intent_judge":
                # Handle cases where planner made wrong plan type (e.g., conversational vs analytical)
                query_type_detected = verdict.get("query_type_detected", "")
                print(f"🔍 User intent judge: query_type_detected = '{query_type_detected}'")
                
                is_conversational = self._is_conversational_plan(state)
                print(f"🔍 Is current plan conversational? {is_conversational}")
                
                if query_type_detected in ["comparison", "analysis", "discovery"] and is_conversational:
                    print(f"🔄 CREATING PLAN TYPE CORRECTION: {query_type_detected}")
                    revisions.append({
                        "type": "plan_type_correction",
                        "action": "convert_to_analysis",
                        "target_query_type": query_type_detected,
                        "guidance": guidance,
                        "issue": issue,
                        "original_query": state.get("execution_plan", {}).get("original_question", ""),
                        "available_cell_types": state.get("available_cell_types", [])
                    })
                elif query_type_detected in ["comparison", "analysis", "discovery"]:
                    # Even if not purely conversational, might need cell discovery
                    print(f"🧬 CHECKING CELL DISCOVERY NEEDS for {query_type_detected}")
                    original_query = state.get("execution_plan", {}).get("original_question", "")
                    needed_cell_types = extract_cell_types_from_question(original_query, self.hierarchy_manager)
                    available_cell_types = state.get("available_cell_types", [])
                    
                    if needed_cell_types and needs_cell_discovery(needed_cell_types, available_cell_types):
                        print(f"🧬 CREATING CELL DISCOVERY PLAN")
                        revisions.append({
                            "type": "cell_discovery",
                            "action": "add_discovery_steps",
                            "needed_cell_types": needed_cell_types,
                            "available_cell_types": available_cell_types,
                            "target_query_type": query_type_detected,
                            "guidance": guidance
                        })
                    else:
                        print(f"🔍 No cell discovery needed: needed={needed_cell_types}, available={available_cell_types}")
                else:
                    print(f"🔍 No plan correction needed: query_type={query_type_detected}, conversational={is_conversational}")
        
        print(f"🔍 Created {len(revisions)} revisions total")
        for i, revision in enumerate(revisions):
            print(f"🔍 Revision {i+1}: {revision['type']} - {revision.get('action', 'no action')}")
        
        return revisions
    
    def _apply_targeted_revisions(self, execution_plan: Dict[str, Any], revision_plan: List[Dict[str, Any]], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply targeted revisions to the execution plan.
        """
        
        print(f"🔍 Applying {len(revision_plan)} revisions to execution plan")
        print(f"🔍 Original plan has {len(execution_plan.get('steps', []))} steps")
        
        # Start with current plan
        revised_plan = execution_plan.copy()
        steps = revised_plan.get("steps", []).copy()
        
        for revision in revision_plan:
            if revision["type"] == "efficiency_improvement" and revision["action"] == "remove_redundancy":
                # Remove redundant steps
                redundant_descriptions = revision.get("redundant_steps", [])
                steps = [step for step in steps 
                        if step.get("description", "") not in redundant_descriptions]
                print(f"🔍 Removed {len(redundant_descriptions)} redundant steps")
            
            elif revision["type"] == "completeness_improvement" and revision["action"] == "add_missing_steps":
                # Add missing components
                missing_components = revision.get("missing_components", [])
                for component in missing_components:
                    if component == "visualization":
                        # Add visualization step
                        viz_step = {
                            "step_type": "visualization",
                            "function_name": "display_enrichment_visualization",
                            "parameters": {"cell_type": "T cell", "analysis": "gsea"},
                            "description": "Display enrichment analysis visualization",
                            "expected_outcome": "Interactive visualization of enrichment results"
                        }
                        steps.append(viz_step)
                        print(f"🔍 Added visualization step")
                
            elif revision["type"] == "plan_type_correction" and revision["action"] == "convert_to_analysis":
                # Replace conversational plan with analytical plan
                query_type = revision.get("target_query_type", "analysis")
                original_query = revision.get("original_query", "")
                available_cell_types = revision.get("available_cell_types", [])
                
                print(f"🔄 Converting conversational plan to {query_type} plan")
                
                # Create new analytical steps based on query type
                if query_type == "comparison":
                    # Create comparison analysis steps
                    needed_cell_types = extract_cell_types_from_question(original_query, self.hierarchy_manager)
                    if len(needed_cell_types) >= 2:
                        new_steps = []
                        for cell_type in needed_cell_types[:2]:  # Compare first two
                            new_steps.append({
                                "step_type": "analysis",
                                "function_name": "perform_enrichment_analyses",
                                "parameters": {"cell_type": cell_type},
                                "description": f"Perform enrichment analysis for {cell_type}",
                                "expected_outcome": f"Enrichment results for {cell_type}"
                            })
                        steps = new_steps
                        print(f"🔍 Created comparison analysis for {needed_cell_types[:2]}")
                
                elif query_type == "analysis":
                    # Create general analysis steps
                    needed_cell_types = extract_cell_types_from_question(original_query, self.hierarchy_manager)
                    if needed_cell_types:
                        new_steps = []
                        for cell_type in needed_cell_types:
                            new_steps.append({
                                "step_type": "analysis",
                                "function_name": "perform_enrichment_analyses",
                                "parameters": {"cell_type": cell_type},
                                "description": f"Perform enrichment analysis for {cell_type}",
                                "expected_outcome": f"Enrichment results for {cell_type}"
                            })
                        steps = new_steps
                        print(f"🔍 Created analysis steps for {needed_cell_types}")
            
            elif revision["type"] == "cell_discovery" and revision["action"] == "add_discovery_steps":
                # Add cell discovery steps
                needed_cell_types = revision.get("needed_cell_types", [])
                available_cell_types = revision.get("available_cell_types", [])
                
                discovery_steps = create_cell_discovery_steps(needed_cell_types, available_cell_types, "analysis", self.hierarchy_manager)
                if discovery_steps:
                    steps = discovery_steps + steps
                    print(f"🧬 Added {len(discovery_steps)} cell discovery steps")
        
        # Update the plan with revised steps
        revised_plan["steps"] = steps
        
        print(f"🔍 Final revised plan has {len(steps)} steps")
        
        return revised_plan