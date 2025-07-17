"""
Workflow utilities for workflow nodes.

This module contains utility methods extracted from workflow_nodes.py:
- _summarize_functions(): Summarize available functions for planning context (lines ~1097-1108)
- _parse_cell_types(): Parse strings containing multiple cell types (lines ~1110-1128) 
- _get_relevant_cell_types_from_context(): Extract relevant cell types from execution context (lines ~1140-1156)
- _build_cached_analysis_context(): Build analysis context from cached results (lines ~1158-1194)
- _generate_execution_summary*(): Methods for execution summary generation (lines ~1349-1501)
- Cell discovery enhancement methods (lines ~1505-1603)
"""

import json
import openai
from typing import Dict, Any, List

from ..cell_type_models import ChatState
from ..shared import extract_cell_types_from_question, needs_cell_discovery, create_cell_discovery_steps


class UtilsMixin:
    """Workflow utilities mixin for WorkflowNodes class."""
    
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

    # ========== Execution Summary Generation ==========
    
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
            print(f"📊 Including {len(viz_steps)} visualization(s) in response")
        
        state["response"] = json.dumps(response_data)
        return state

    # ========== Cell Discovery Enhancement Methods ==========
    
    def _add_cell_discovery_to_plan(self, plan_data: Dict[str, Any], message: str, available_cell_types: List[str]) -> Dict[str, Any]:
        """
        Enhance the initial plan by adding cell discovery steps if needed.
        
        Uses proven cell type extraction and discovery logic.
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