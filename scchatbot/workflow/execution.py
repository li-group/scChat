"""
Execution logic for workflow nodes.

This module contains execution-related methods extracted from workflow_nodes.py:
- _execute_final_question(): Executes comprehensive final question using all available context
- validate_processing_results(): Validates that process_cells discovered expected cell types
- _update_available_cell_types_from_result(): Updates available cell types from processing results
- _extract_cell_types_from_result(): Extracts cell types from analysis results
"""

import json
import openai
import re
from typing import Dict, Any, List

from ..cell_type_models import ChatState


class ExecutionMixin:
    """Execution logic mixin for WorkflowNodes class."""
    
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
                for cell_type in discovered_types:
                    if cell_type and cell_type not in current_available:
                        current_available.add(cell_type)
                        print(f"🧬 Added newly discovered cell type to available list: '{cell_type}'")
                
                state["available_cell_types"] = list(current_available)
                print(f"✅ Updated available cell types: {len(current_available)} types now available")
        
        except Exception as e:
            print(f"⚠️ Error updating available cell types: {e}")
            # Continue without failing

    def _extract_cell_types_from_result(self, result: Any) -> List[str]:
        """Extract cell types from analysis result"""
        if self.cell_type_extractor:
            return self.cell_type_extractor.extract_from_annotation_result(result)
        else:
            # Simple fallback extraction
            if isinstance(result, str) and "cell_type" in result:
                return ["T cell", "B cell"]  # Placeholder
            return []