"""
Evaluation node implementation.

This module contains the EvaluatorNode which handles post-execution evaluation,
plan consolidation, and validation logic.
"""

from typing import Dict, Any, List
import json
import re
from ...cell_type_models import ChatState
from ..node_base import BaseWorkflowNode


class EvaluatorNode(BaseWorkflowNode):
    """
    Evaluator node that handles post-execution review and validation.
    
    Responsibilities:
    - Review execution history for completeness
    - Identify failed steps and missing results
    - Prepare context for response generation
    - Handle plan consolidation and validation warnings
    - Manage missing cell type updates
    """
    
    def execute(self, state: ChatState) -> ChatState:
        """Handle both validation steps AND post-execution evaluation."""
        # Check if current step is a validation step  
        execution_plan = state.get("execution_plan", {})
        steps = execution_plan.get("steps", [])
        current_index = state.get("current_step_index", 0)
        
        if current_index < len(steps):
            current_step = steps[current_index]
            if current_step.get("step_type") == "validation":
                # Handle validation step (moved from ExecutorNode)
                return self._execute_validation_step(state, current_step)
        
        # Otherwise handle post-execution evaluation (current logic)
        return self.evaluator_node(state)
    
    def evaluator_node(self, state: ChatState) -> ChatState:
        """
        Sophisticated LLM-powered post-execution evaluation with gap analysis.
        
        This method performs comprehensive post-execution evaluation by:
        - Extracting mentioned cell types from the original question
        - Using LLM to determine required analyses per cell type
        - Checking what was actually performed
        - Generating supplementary steps for missing analyses
        - Skipping unavailable cell types
        
        Args:
            state: Current workflow state with completed execution
            
        Returns:
            Updated state with post-execution evaluation results
        """
        
        print("🏁 Evaluator: Starting sophisticated post-execution evaluation...")
        
        # Defensive check: ensure execution_plan exists
        if not state.get("execution_plan"):
            print("⚠️ Evaluator: No execution plan found")
            state["conversation_complete"] = True
            return state
        
        # Run the sophisticated LLM-powered post-execution evaluation
        evaluation_result = self._post_execution_evaluation(state)
        
        # Store evaluation results in state for response generation
        state["post_execution_evaluation"] = evaluation_result
        
        # Count successful and failed steps for logging
        execution_history = state.get("execution_history", [])
        successful_steps = [h for h in execution_history if h.get("success", False)]
        failed_steps = [h for h in execution_history if not h.get("success", False)]
        
        print(f"✅ Execution Summary:")
        print(f"   • Total steps executed: {len(execution_history)}")
        print(f"   • Successful: {len(successful_steps)}")
        print(f"   • Failed: {len(failed_steps)}")
        print(f"   • Supplementary steps generated: {len(evaluation_result['supplementary_steps'])}")
        
        # Review available results if history manager exists
        if self.history_manager:
            available_results = self.history_manager.get_available_results()
            if available_results:
                print("📊 Available analysis results:")
                if "enrichment_analyses" in available_results:
                    print(f"   • Enrichment analyses: {list(available_results['enrichment_analyses'].keys())}")
                if "deg_analyses" in available_results:
                    print(f"   • DEG analyses: {list(available_results['deg_analyses'].keys())}")
                if "cell_annotations" in available_results:
                    print(f"   • Cell annotations: {list(available_results['cell_annotations'].keys())}")
        
        # Mark conversation as complete for response generation
        state["conversation_complete"] = True
        print("🎯 Evaluator: Post-execution evaluation complete, ready for response generation")
        
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
    
    def update_remaining_steps_with_discovered_types(self, state: ChatState, validation_result: Dict[str, Any]) -> None:
        """
        Update remaining analysis steps to use actually discovered cell types instead of unavailable ones.
        
        This is called after validation to ensure subsequent analysis steps use real discovered types.
        Based on backup implementation lines 1070-1134.
        """
        execution_plan = state.get("execution_plan", {})
        steps = execution_plan.get("steps", [])
        current_step_index = state.get("current_step_index", 0)
        
        # Get the mapping of what was expected vs what was actually found
        expected_children = validation_result.get("missing_children", []) + validation_result.get("found_children", [])
        found_children = validation_result.get("found_children", [])
        missing_children = validation_result.get("missing_children", [])
        
        print(f"🔄 Updating remaining steps: expected={expected_children}, found={found_children}, missing={missing_children}")
        print(f"🔧 DEBUG: Validation result full structure: {validation_result}")
        
        # No replacement mapping - we should skip steps for missing cell types entirely
        # This avoids creating duplicate steps when some expected children are found
        print(f"🔧 DEBUG: Will skip steps for missing cell types: {missing_children}")
        
        # Update remaining steps
        skipped_count = 0
        print(f"🔍 DEBUG: Scanning {len(steps) - current_step_index - 1} remaining steps...")
        
        for i in range(current_step_index + 1, len(steps)):
            step = steps[i]
            step_cell_type = step.get("parameters", {}).get("cell_type")
            function_name = step.get("function_name", "unknown")
            
            # Better parameter display for debugging
            if step_cell_type is None:
                # Check for other parameter types
                params = step.get("parameters", {})
                if "processed_parent" in params:
                    param_str = f"processed_parent='{params.get('processed_parent')}', expected_children={params.get('expected_children', [])}"
                elif "analysis" in params:
                    param_str = f"analysis='{params.get('analysis')}'"
                else:
                    param_str = f"params={params}"
                print(f"🔍 DEBUG: Step {i+1}: {function_name}({param_str})")
            else:
                print(f"🔍 DEBUG: Step {i+1}: {function_name}(cell_type='{step_cell_type}')")
            
            if step_cell_type in missing_children:
                print(f"🔍 DEBUG: Step {i+1} references missing cell type '{step_cell_type}'")
                # Mark this step to be skipped (cell type unavailable)
                steps[i]["skip_reason"] = f"Cell type '{step_cell_type}' not discovered"
                print(f"⏭️ Marked step {i+1} for skipping: {function_name}({step_cell_type})")
                print(f"🔧 DEBUG: Step {i+1} now has skip_reason: {steps[i].get('skip_reason')}")
                skipped_count += 1
            else:
                print(f"🔍 DEBUG: Step {i+1} cell type '{step_cell_type}' not in missing list {missing_children}")
        
        if skipped_count > 0:
            print(f"✅ Step update complete: {skipped_count} marked for skipping")
    
    
    def _execute_validation_step(self, state: ChatState, step_data) -> ChatState:
        """Handle validation step execution (moved from ExecutorNode, matches backup lines 617-656)"""
        print("🔍 Executing validation step...")
        
        # Extract parameters from step_data
        processed_parent = step_data.get("parameters", {}).get("processed_parent")
        expected_children = step_data.get("parameters", {}).get("expected_children", [])
        
        result = self.validate_processing_results(processed_parent, expected_children)
        
        # Initialize tracking variables
        success = False
        error_msg = None
        
        # Process validation result exactly like backup
        if result["status"] == "success":
            success = True
            print(f"✅ Validation passed: {result['message']}")
            # Update available cell types with discovered types
            state["available_cell_types"] = result["available_types"]
            
        elif result["status"] == "partial_success":
            success = True  # Continue but with warnings
            print(f"⚠️ Validation partial: {result['message']}")
            # Update available cell types with what we actually found
            state["available_cell_types"] = result["available_types"]
            
            # Track unavailable cell types exactly like backup
            expected_types = step_data.get("parameters", {}).get("expected_children", [])
            available_types = result.get("available_types", [])
            missing_types = [ct for ct in expected_types if ct not in available_types]
            if missing_types:
                current_unavailable = state.get("unavailable_cell_types", [])
                state["unavailable_cell_types"] = list(set(current_unavailable + missing_types))
                print(f"📋 Added to unavailable cell types: {missing_types}")
            
        else:
            success = False
            error_msg = result["message"]
            print(f"❌ Validation failed: {error_msg}")
            
            # Track all expected cell types as unavailable on complete failure
            expected_types = step_data.get("parameters", {}).get("expected_children", [])
            if expected_types:
                current_unavailable = state.get("unavailable_cell_types", [])
                state["unavailable_cell_types"] = list(set(current_unavailable + expected_types))
                print(f"📋 Added to unavailable cell types (validation failed): {expected_types}")
                
        # CRITICAL FIX: Update subsequent analysis steps based on validation results
        # This MUST run for ANY validation (success, partial, or complete failure)
        # to skip steps for missing cell types or update steps for discovered types
        print(f"🔧 CALLING update_remaining_steps_with_discovered_types with result status: {result.get('status')}")
        self.update_remaining_steps_with_discovered_types(state, result)
        
        # Record execution in state history with proper structure (matches backup lines 762-770)
        current_step_index = state.get("current_step_index", 0)
        
        # Create execution history entry with proper structure
        import copy
        step_data_copy = copy.deepcopy(step_data)  # Prevent mutation
        
        state["execution_history"].append({
            "step_index": current_step_index,
            "step": step_data_copy,
            "success": success,
            "result": result,  # Full validation result
            "result_type": "validation",
            "result_summary": result.get("message", "Validation completed"),
            "error": error_msg
        })
        
        # CRITICAL: Validation steps always advance to avoid infinite loops (matches backup line 783)
        state["current_step_index"] = current_step_index + 1
        
        if success:
            print(f"🔄 Advanced to step {current_step_index + 2}")
        else:
            print(f"🔄 Advanced to step {current_step_index + 2} (validation failure, but continuing)")
        
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
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question type using a smaller LLM for efficiency"""
        
        classification_prompt = f"""Classify the following biology question into ONE category:
Question: "{question}"
Categories:
1. canonical_markers - Questions about well-known, established cell type markers (e.g., "What are canonical markers for...", "differentiate X from Y using markers")
2. pathway_analysis - Questions about biological pathways, processes, or functional enrichment (e.g., "What pathways are enriched...", "biological processes in...")
3. gene_expression - Questions about specific gene expression changes (e.g., "Is gene X upregulated...", "expression of Y in condition Z")
4. cell_abundance - Questions about cell type counts or proportions (e.g., "How many X cells...", "proportion of Y cells")
5. general_comparison - General comparison questions not fitting above categories
Return ONLY the category name, nothing else.
Category:"""
        
        try:
            # Use the LLM call method from base class
            response = self._call_llm(classification_prompt)
            category = response.strip().lower()
            
            # Validate category
            valid_categories = ["canonical_markers", "pathway_analysis", "gene_expression", 
                              "cell_abundance", "general_comparison"]
            
            if category not in valid_categories:
                print(f"⚠️ Invalid category '{category}', defaulting to general_comparison")
                return "general_comparison"
            
            print(f"📊 Question classified as: {category}")
            return category
            
        except Exception as e:
            print(f"⚠️ Question classification failed: {e}")
            return "general_comparison"
    
    def _post_execution_evaluation(self, state: ChatState) -> Dict[str, Any]:
        """
        Cell-type specific LLM-powered gap analysis - FINAL APPROACH
        """
        print("🔍 Starting post-execution evaluation...")
        
        original_question = state["execution_plan"]["original_question"]
        
        # Import the extraction function
        from ...cell_types.validation import extract_cell_types_from_question
        
        # Step 1: Extract mentioned cell types using existing function
        mentioned_types = extract_cell_types_from_question(original_question, self.hierarchy_manager)
        
        # Classify question type for better result filtering
        question_type = self._classify_question_type(original_question)
        
        if not mentioned_types:
            print("📋 No specific cell types mentioned, skipping post-execution evaluation")
            return {"mentioned_cell_types": [], "supplementary_steps": [], "evaluation_complete": True}
        
        supplementary_steps = []
        evaluation_details = {}
        
        # Get unavailable cell types to skip them
        unavailable_cell_types = state.get("unavailable_cell_types", [])
        
        # Step 2: For each mentioned cell type, ask LLM what analyses are needed
        for cell_type in mentioned_types:
            # Skip cell types that were marked as unavailable during validation
            if cell_type in unavailable_cell_types:
                print(f"⏭️ Skipping post-execution evaluation for unavailable cell type: {cell_type}")
                continue
                
            print(f"🔍 Evaluating coverage for cell type: {cell_type}")
            
            # Step 2a: Get LLM recommendations for this specific cell type
            required_analyses = self._get_llm_analysis_requirements(original_question, cell_type)
            
            # Step 2b: Check what was actually performed for this cell type
            performed_analyses = self._get_performed_analyses_for_cell_type(state, cell_type)
            
            # Step 2c: Find gaps and generate steps
            missing_steps = self._generate_missing_steps_for_cell_type(
                cell_type, required_analyses, performed_analyses
            )
            
            supplementary_steps.extend(missing_steps)
            evaluation_details[cell_type] = {
                "required_analyses": required_analyses,
                "performed_analyses": performed_analyses,
                "missing_steps_count": len(missing_steps)
            }
            
            if missing_steps:
                print(f"📋 Found {len(missing_steps)} missing steps for {cell_type}")
            else:
                print(f"✅ Complete coverage for {cell_type}")
        
        print(f"🔍 Post-execution evaluation complete: {len(supplementary_steps)} total supplementary steps")
        
        # Add analysis relevance hints based on question type
        all_performed_analyses = {}
        for cell_type, details in evaluation_details.items():
            all_performed_analyses[cell_type] = details["performed_analyses"]
        
        analysis_relevance = self._get_analysis_relevance_hints(question_type, all_performed_analyses)
        
        return {
            "mentioned_cell_types": mentioned_types,
            "evaluation_details": evaluation_details,
            "supplementary_steps": supplementary_steps,
            "evaluation_complete": True,
            "question_type": question_type,
            "analysis_relevance": analysis_relevance
        }

    def _get_llm_analysis_requirements(self, original_question: str, cell_type: str) -> List[str]:
        """Ask LLM what analyses this specific cell type needs"""
        
        # Get question type for context
        question_type = self._classify_question_type(original_question)
        
        analysis_prompt = f"""You are analyzing what bioinformatics analyses are needed for a specific cell type.

User Question: "{original_question}"
Cell Type: "{cell_type}"
Question Type: {question_type}

Available analysis functions with detailed descriptions:

CORE ANALYSIS FUNCTIONS:
- perform_enrichment_analyses: Run enrichment analyses on DE genes for a cell type. Supports REACTOME (pathways), GO (gene ontology), KEGG (pathways), GSEA (gene set enrichment). Use for pathway analysis when user asks about biological processes, pathways, or gene function.

- dea_split_by_condition: Perform differential expression analysis (DEA) split by condition. Use when comparing conditions or when user asks about gene expression differences between experimental groups.

- compare_cell_counts: Compare cell counts between experimental conditions for specific cell types. Use when analyzing how cell type abundance differs across conditions (e.g., pre vs post treatment, healthy vs disease).

VISUALIZATION FUNCTIONS:
- display_enrichment_visualization: PREFERRED function for showing comprehensive enrichment visualization with both barplot and dotplot. Use after running enrichment analyses to visualize results.

- display_dotplot: Display dotplot for annotated results. Use when user wants to see gene expression patterns across cell types.

- display_cell_type_composition: Display cell type composition graph. Use when user wants to see the proportion of different cell types.

- display_umap: Display basic UMAP without cell type annotations. Use for basic dimensionality reduction visualization.

- display_processed_umap: Display UMAP with cell type annotations. Use when user wants to see cell type annotations on UMAP.

SEARCH FUNCTIONS:
- search_enrichment_semantic: Search all enrichment terms semantically to find specific pathways or biological processes. Use when user asks about specific pathways, terms, or biological processes that might not appear in standard top results.

- conversational_response: Provide conversational response without function calls. Use for greetings, clarifications, explanations, or when no analysis is needed.

Task: Determine which analyses are needed for {cell_type} to answer the user's question.

IMPORTANT: The cell type "{cell_type}" already exists in the dataset. DO NOT suggest process_cells for this cell type.

Consider based on question type:
1. Canonical markers questions (differentiate X from Y, markers of X) → Use dea_split_by_condition ONLY
2. Gene expression questions → Use dea_split_by_condition
3. Pathway/biological process questions → Use perform_enrichment_analyses + search_enrichment_semantic
4. Cell abundance questions → Use compare_cell_counts
5. Specific pathway search → Use search_enrichment_semantic

CRITICAL GUIDELINES FOR CANONICAL MARKERS:
- For canonical markers questions, ONLY use dea_split_by_condition
- DO NOT include enrichment analyses for marker identification
- DO NOT suggest process_cells for already-available cell types
- Visualization is optional for canonical markers

GENERAL GUIDELINES:
- Only include display_enrichment_visualization ONCE per cell type
- Return ONLY a valid JSON array of function names, nothing else

Example response for pathway question: ["perform_enrichment_analyses", "search_enrichment_semantic", "display_enrichment_visualization"]
Example response for all kinds of markers: ["dea_split_by_condition"]

Required analyses for {cell_type}:"""
        
        try:
            response = self._call_llm(analysis_prompt)
            print(f"🔍 LLM raw response for {cell_type}: '{response}' (length: {len(response)})")
            
            if not response or response.strip() == "":
                print(f"⚠️ Empty LLM response for {cell_type}, using fallback")
                return ["perform_enrichment_analyses"]
            
            # Try to extract JSON from response (handle cases with markdown code blocks)
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith("```json"):
                response = response.replace("```json", "").replace("```", "").strip()
            elif response.startswith("```"):
                response = response.replace("```", "").strip()
            
            # Look for JSON array pattern if response contains other text
            import re
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            
            import json
            required_analyses = json.loads(response)
            
            # Ensure we got a list
            if not isinstance(required_analyses, list):
                print(f"⚠️ LLM returned {type(required_analyses)} instead of list for {cell_type}")
                return ["perform_enrichment_analyses"]
            
            print(f"🧠 LLM recommends for {cell_type}: {required_analyses}")
            return required_analyses
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing failed for {cell_type}: {e}")
            print(f"⚠️ Raw response was: '{response}'")
            return ["perform_enrichment_analyses"]  # Safe fallback
        except Exception as e:
            print(f"⚠️ LLM analysis requirement failed for {cell_type}: {e}")
            return ["perform_enrichment_analyses"]  # Safe fallback

    def _get_performed_analyses_for_cell_type(self, state: ChatState, cell_type: str) -> List[str]:
        """Check what analyses were actually performed for a specific cell type"""
        
        performed_analyses = []
        unavailable_cell_types = state.get("unavailable_cell_types", [])
        
        print(f"🔧 DEBUG: Checking if '{cell_type}' is in unavailable list: {unavailable_cell_types}")
        
        # If this cell type is known to be unavailable, don't try to generate steps for it
        if cell_type in unavailable_cell_types:
            print(f"🚫 Cell type '{cell_type}' is unavailable - not generating missing steps")
            # Return dummy analysis to prevent post-execution from trying to add steps
            return ["analysis_skipped_cell_type_unavailable"]
        
        for ex in state["execution_history"]:
            # Check both successful executions AND skipped steps 
            if ex.get("success") or ex.get("skipped"):
                function_name = ex.get("step", {}).get("function_name")
                params = ex.get("step", {}).get("parameters", {})
                ex_cell_type = params.get("cell_type")
                
                if ex_cell_type == cell_type and function_name:
                    if ex.get("skipped"):
                        print(f"📋 Found skipped analysis for {cell_type}: {function_name}")
                    performed_analyses.append(function_name)
        
        # Remove duplicates while preserving order
        unique_performed = []
        for analysis in performed_analyses:
            if analysis not in unique_performed:
                unique_performed.append(analysis)
        
        print(f"📊 Actually performed for {cell_type}: {unique_performed}")
        return unique_performed

    def _generate_missing_steps_for_cell_type(self, cell_type: str, required_analyses: List[str], 
                                            performed_analyses: List[str]) -> List[Dict[str, Any]]:
        """Generate supplementary steps for missing analyses for a specific cell type"""
        
        missing_steps = []
        
        for required_function in required_analyses:
            if required_function not in performed_analyses:
                # Generate step using existing step format
                step = {
                    "step_type": "analysis" if not required_function.startswith("display_") else "visualization",
                    "function_name": required_function,
                    "parameters": {"cell_type": cell_type},
                    "description": f"Post-evaluation: {required_function} for {cell_type}",
                    "expected_outcome": f"Complete analysis coverage for {cell_type}",
                    "target_cell_type": cell_type
                }
                
                # Add specific parameters for visualization functions
                if required_function == "display_enrichment_visualization":
                    step["parameters"]["analysis"] = "gsea"  # Default analysis type
                
                missing_steps.append(step)
                print(f"🔧 Generated missing step: {required_function}({cell_type})")
        
        return missing_steps
    
    def _get_analysis_relevance_hints(self, question_type: str, performed_analyses: Dict[str, List[str]]) -> Dict[str, Any]:
        """Generate hints about which analyses are most relevant for the question type"""
        
        relevance_map = {
            "canonical_markers": {
                "primary": ["dea_split_by_condition"],
                "secondary": ["display_dotplot"]
            },
            "pathway_analysis": {
                "primary": ["perform_enrichment_analyses", "search_enrichment_semantic"],
                "secondary": ["display_enrichment_visualization"]
            },
            "gene_expression": {
                "primary": ["dea_split_by_condition"],
                "secondary": ["display_dotplot"]
            },
            "cell_abundance": {
                "primary": ["compare_cell_counts"],
                "secondary": ["display_processed_umap"]
            },
            "general_comparison": {
                "primary": ["dea_split_by_condition", "perform_enrichment_analyses"],
                "secondary": ["search_enrichment_semantic", "display_enrichment_visualization"]
            }
        }
        
        relevance = relevance_map.get(question_type, relevance_map["general_comparison"])
        
        # Create hints for response generator
        hints = {
            "question_type": question_type,
            "relevance_categories": relevance,
            "guidance": self._get_response_guidance(question_type)
        }
        
        print(f"📝 Generated relevance hints for {question_type} question")
        return hints
    
    def _get_response_guidance(self, question_type: str) -> str:
        """Get specific guidance for response generation based on question type"""
        
        guidance_map = {
            "canonical_markers": "Focus on well-established markers from literature. Prioritize DEA results showing marker genes. Avoid emphasizing enrichment analysis unless directly relevant to marker function.",
            "pathway_analysis": "Emphasize enrichment analysis results, biological processes, and pathway information. DEA results support pathway findings.",
            "gene_expression": "Focus on specific gene expression values and fold changes from DEA. Show specific gene names and statistical significance.",  
            "cell_abundance": "Prioritize cell count comparisons and composition visualizations. Focus on quantitative differences between conditions.",
            "general_comparison": "Balance all available analyses based on what best answers the specific question."
        }
        
        return guidance_map.get(question_type, guidance_map["general_comparison"])
        
    def _get_timestamp(self) -> str:
        """Get current timestamp for execution tracking."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
