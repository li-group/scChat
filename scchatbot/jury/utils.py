"""
Jury system utility functions.

This module contains utility methods extracted from jury_system_main.py:
- _create_smart_execution_summary(): Create intelligent execution summaries
- _prepare_cache_aware_evaluation_inputs(): Prepare evaluation inputs with cache awareness
- _truncate_execution_history_for_judges(): Truncate execution history for efficiency
- _get_relevant_cell_types_from_context(): Extract relevant cell types from context
- _get_cache_analysis_summary(): Get cache analysis summary
- _build_comprehensive_analysis_context(): Build comprehensive analysis context
- _is_conversational_plan(): Check if plan is conversational
"""

from typing import Dict, Any, List


class JuryUtilsMixin:
    """
    Jury system utility methods mixin.
    """

    def _create_smart_execution_summary(self, execution_history: List[Dict[str, Any]]) -> str:
        """
        Create intelligent execution summary that highlights key information for judges.
        
        Args:
            execution_history: List of execution history entries
            
        Returns:
            Smart summary focusing on key execution patterns
        """
        if not execution_history:
            return "No execution history available"
        
        # Categorize steps by type
        analysis_steps = []
        visualization_steps = []
        process_cells_steps = []
        failed_steps = []
        
        for entry in execution_history:
            if not entry.get("success", False):
                failed_steps.append(entry)
                continue
                
            function_name = entry.get("step", {}).get("function_name", "")
            
            if function_name == "process_cells":
                process_cells_steps.append(entry)
            elif function_name.startswith("display_"):
                visualization_steps.append(entry)
            elif function_name in ["perform_enrichment_analyses", "dea_split_by_condition"]:
                analysis_steps.append(entry)
        
        # Build smart summary
        summary_parts = []
        
        # Cell processing summary
        if process_cells_steps:
            cell_types = [step.get("step", {}).get("parameters", {}).get("cell_type", "unknown") 
                         for step in process_cells_steps]
            summary_parts.append(f"Cell processing: {len(cell_types)} cell types ({', '.join(cell_types)})")
        
        # Analysis summary
        if analysis_steps:
            analysis_cell_types = [step.get("step", {}).get("parameters", {}).get("cell_type", "unknown") 
                                 for step in analysis_steps]
            summary_parts.append(f"Analysis: {len(analysis_steps)} analyses on {len(set(analysis_cell_types))} cell types")
        
        # Visualization summary
        if visualization_steps:
            viz_functions = [step.get("step", {}).get("function_name", "unknown") for step in visualization_steps]
            summary_parts.append(f"Visualizations: {len(viz_functions)} plots ({', '.join(set(viz_functions))})")
        
        # Failures summary
        if failed_steps:
            failed_functions = [step.get("step", {}).get("function_name", "unknown") for step in failed_steps]
            summary_parts.append(f"Failures: {len(failed_steps)} steps ({', '.join(failed_functions)})")
        
        return " | ".join(summary_parts) if summary_parts else "No significant execution patterns"

    def _prepare_cache_aware_evaluation_inputs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare evaluation inputs with cache awareness to provide rich context.
        
        Args:
            state: Current workflow state
            
        Returns:
            Enhanced evaluation inputs with cache context
        """
        execution_plan = state.get("execution_plan", {})
        execution_history = state.get("execution_history", [])
        
        # Get relevant cell types from context
        relevant_cell_types = self._get_relevant_cell_types_from_context(state)
        
        # Get cache analysis summary
        cache_summary = self._get_cache_analysis_summary(relevant_cell_types)
        
        # Build comprehensive analysis context
        analysis_context = self._build_comprehensive_analysis_context(state, execution_history)
        
        return {
            "execution_plan": execution_plan,
            "execution_history": self._truncate_execution_history_for_judges(execution_history),
            "cache_summary": cache_summary,
            "analysis_context": analysis_context,
            "relevant_cell_types": relevant_cell_types,
            "original_query": state.get("current_message", ""),
            "available_cell_types": state.get("available_cell_types", []),
            "unavailable_cell_types": state.get("unavailable_cell_types", [])
        }

    def _truncate_execution_history_for_judges(self, execution_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Truncate execution history to essential information for judges.
        
        Args:
            execution_history: Full execution history
            
        Returns:
            Truncated history with essential information
        """
        if not execution_history:
            return []
        
        # Keep only essential fields and truncate results
        truncated_history = []
        
        for entry in execution_history:
            truncated_entry = {
                "step_index": entry.get("step_index", 0),
                "success": entry.get("success", False),
                "step": {
                    "function_name": entry.get("step", {}).get("function_name", ""),
                    "parameters": entry.get("step", {}).get("parameters", {}),
                    "description": entry.get("step", {}).get("description", "")
                }
            }
            
            # Include truncated result (first 200 characters)
            result = entry.get("result", "")
            if result:
                truncated_entry["result_summary"] = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
            
            # Include error if present
            if "error" in entry:
                truncated_entry["error"] = entry["error"]
            
            truncated_history.append(truncated_entry)
        
        return truncated_history

    def _get_relevant_cell_types_from_context(self, state: Dict[str, Any]) -> List[str]:
        """
        Extract relevant cell types from execution context.
        
        Args:
            state: Current workflow state
            
        Returns:
            List of relevant cell types
        """
        relevant_cell_types = []
        
        # Get cell types from execution history
        execution_history = state.get("execution_history", [])
        for entry in execution_history:
            if entry.get("success", False):
                cell_type = entry.get("step", {}).get("parameters", {}).get("cell_type")
                if cell_type and cell_type not in relevant_cell_types:
                    relevant_cell_types.append(cell_type)
        
        # Get cell types from execution plan
        execution_plan = state.get("execution_plan", {})
        for step in execution_plan.get("steps", []):
            cell_type = step.get("parameters", {}).get("cell_type")
            if cell_type and cell_type not in relevant_cell_types:
                relevant_cell_types.append(cell_type)
        
        # Fallback to available cell types
        if not relevant_cell_types:
            relevant_cell_types = state.get("available_cell_types", [])
        
        return relevant_cell_types

    def _get_cache_analysis_summary(self, relevant_cell_types: List[str]) -> str:
        """
        Get cache analysis summary for relevant cell types.
        
        Args:
            relevant_cell_types: List of relevant cell types
            
        Returns:
            Cache analysis summary
        """
        if not self.simple_cache or not relevant_cell_types:
            return "No cache analysis available"
        
        cache_summaries = []
        
        for cell_type in relevant_cell_types:
            insights = self.simple_cache.get_analysis_insights(cell_type)
            if insights and insights.get("summary"):
                cache_summaries.append(f"{cell_type}: {insights['summary'][:100]}...")
        
        return " | ".join(cache_summaries) if cache_summaries else "No cached insights for relevant cell types"

    def _get_actually_available_analyses(self, relevant_cell_types: List[str]) -> str:
        """
        Get actually available analyses for relevant cell types.
        
        Args:
            relevant_cell_types: List of relevant cell types
            
        Returns:
            Summary of available analyses
        """
        if not self.simple_cache or not relevant_cell_types:
            return "No analysis information available"
        
        available_analyses = []
        
        for cell_type in relevant_cell_types:
            insights = self.simple_cache.get_analysis_insights(cell_type)
            if insights:
                analyses = []
                if insights.get("dea_insights"):
                    analyses.append("DEA")
                if insights.get("enrichment_insights"):
                    analyses.append("Enrichment")
                if analyses:
                    available_analyses.append(f"{cell_type}: {', '.join(analyses)}")
        
        return " | ".join(available_analyses) if available_analyses else "No specific analyses available"

    def _build_comprehensive_analysis_context(self, state: Dict[str, Any], execution_history: List) -> str:
        """
        Build comprehensive analysis context for judges.
        
        Args:
            state: Current workflow state
            execution_history: Execution history
            
        Returns:
            Comprehensive analysis context
        """
        context_parts = []
        
        # Add execution summary
        execution_summary = self._create_smart_execution_summary(execution_history)
        context_parts.append(f"Execution: {execution_summary}")
        
        # Add cell type information
        available_cell_types = state.get("available_cell_types", [])
        if available_cell_types:
            context_parts.append(f"Available cell types: {', '.join(available_cell_types[:5])}{'...' if len(available_cell_types) > 5 else ''}")
        
        # Add unavailable cell types if any
        unavailable_cell_types = state.get("unavailable_cell_types", [])
        if unavailable_cell_types:
            context_parts.append(f"Unavailable cell types: {', '.join(unavailable_cell_types)}")
        
        # Add relevant cell types analysis
        relevant_cell_types = self._get_relevant_cell_types_from_context(state)
        if relevant_cell_types:
            available_analyses = self._get_actually_available_analyses(relevant_cell_types)
            context_parts.append(f"Relevant analyses: {available_analyses}")
        
        return " | ".join(context_parts) if context_parts else "No comprehensive context available"

    def _is_conversational_plan(self, state: Dict[str, Any]) -> bool:
        """
        Check if the current plan is conversational (not analytical).
        
        Args:
            state: Current workflow state
            
        Returns:
            True if plan is conversational, False otherwise
        """
        execution_plan = state.get("execution_plan", {})
        steps = execution_plan.get("steps", [])
        
        if not steps:
            return False
        
        # Check if all steps are conversational
        conversational_functions = ["conversational_response", "final_question"]
        
        for step in steps:
            function_name = step.get("function_name", "")
            if function_name in conversational_functions:
                return True
        
        # Check if plan summary indicates conversational intent
        plan_summary = execution_plan.get("plan_summary", "").lower()
        if any(keyword in plan_summary for keyword in ["conversational", "response", "answer", "explain"]):
            return True
        
        return False
