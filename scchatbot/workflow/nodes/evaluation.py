"""
Evaluation node implementation.

This module contains the EvaluatorNode which handles evaluation and critic agent,
plan consolidation, and validation logic.
"""

from typing import Dict, Any, List
import json
import re
from ...cell_types.models import ChatState
from ..node_base import BaseWorkflowNode
import logging
logger = logging.getLogger(__name__)



class EvaluatorNode(BaseWorkflowNode):
    """
    Evaluator node that handles evaluation and critic agent functionality.
    
    Responsibilities:
    - Review execution history for completeness
    - Identify failed steps and missing results
    - Prepare context for response generation
    - Handle plan consolidation and validation warnings
    - Manage missing cell type updates
    """
    
    def execute(self, state: ChatState) -> ChatState:
        """Handle both validation steps AND critic analysis."""
        # Check if current step is a validation step  
        execution_plan = state.get("execution_plan", {})
        # steps = execution_plan.get("steps", [])
        steps = execution_plan.get("steps", []) or []

        current_index = state.get("current_step_index", 0)
        
        # --- NEW: fast-path for "no-tool / direct-answer" plans ---
        # (a) truly empty plan, or (b) all steps lack a function_name
        has_callable = any((s.get("function_name") or "").strip() for s in steps)
        if not steps or not has_callable:
            logger.info("ℹ️ Evaluator: no callable steps in plan; skipping critic analysis.")
            # Provide a stub so downstream never sees None
            state["critic"] = {
                "mentioned_cell_types": [],
                "supplementary_steps": [],
                "evaluation_complete": True,
                "question_type": "general_comparison",
                "analysis_relevance": {}
            }
            state["conversation_complete"] = True
            return state


        if current_index < len(steps):
            current_step = steps[current_index]
            if current_step.get("step_type") == "validation":
                # Handle validation step (moved from ExecutorNode)
                return self._execute_validation_step(state, current_step)
        
        # CRITICAL FIX: Only run critic analysis when ALL steps are complete
        # Check if we've reached the end of ALL steps (including any supplementary ones)
        if current_index >= len(steps):
            logger.info(f"🎯 All steps complete ({current_index}/{len(steps)}), running critic analysis")
            return self.evaluator_node(state)
        else:
            logger.info(f"🔄 Steps still remaining ({current_index}/{len(steps)}), skipping critic analysis")
            return state
    
    def evaluator_node(self, state: ChatState) -> ChatState:
        """
        Sophisticated LLM-powered critic analysis with gap analysis.

        This method performs comprehensive critic analysis by:
        - Extracting mentioned cell types from the original question
        - Using LLM to determine required analyses per cell type
        - Checking what was actually performed
        - Generating supplementary steps for missing analyses
        - Skipping unavailable cell types
        
        Args:
            state: Current workflow state with completed execution
            
        Returns:
            Updated state with critic analysis results
        """
        
        logger.info("🏁 Evaluator: Starting sophisticated critic analysis...")
        
        # Defensive check: ensure execution_plan exists
        if not state.get("execution_plan"):
            logger.info("⚠️ Evaluator: No execution plan found")
            state["conversation_complete"] = True
            return state
        
        # Run the sophisticated LLM-powered critic analysis
        evaluation_result = self._critic(state)
        
        # Store critic results in state for response generation
        state["critic"] = evaluation_result
        
        # Count successful and failed steps for logging
        execution_history = state.get("execution_history", [])
        successful_steps = [h for h in execution_history if h.get("success", False)]
        failed_steps = [h for h in execution_history if not h.get("success", False)]
        
        logger.info(f"✅ Execution Summary:")
        logger.info(f"   • Total steps executed: {len(execution_history)}")
        logger.info(f"   • Successful: {len(successful_steps)}")
        logger.info(f"   • Failed: {len(failed_steps)}")
        logger.info(f"   • Supplementary steps generated: {len(evaluation_result['supplementary_steps'])}")
        
        # Review available results if history manager exists
        if self.history_manager:
            available_results = self.history_manager.get_available_results()
            if available_results:
                logger.info("📊 Available analysis results:")
                if "enrichment_analyses" in available_results:
                    logger.info(f"   • Enrichment analyses: {list(available_results['enrichment_analyses'].keys())}")
                if "deg_analyses" in available_results:
                    logger.info(f"   • DEG analyses: {list(available_results['deg_analyses'].keys())}")
                if "cell_annotations" in available_results:
                    logger.info(f"   • Cell annotations: {list(available_results['cell_annotations'].keys())}")
        
        # Mark conversation as complete for response generation
        state["conversation_complete"] = True
        logger.info("🎯 Evaluator: Critic analysis complete, ready for response generation")
        
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
                logger.info(f"✅ Exact match found: '{expected_child}'")
            else:
                # Check if any discovered types are subtypes of the expected type using hierarchy
                subtypes_found = self._find_subtypes_in_available(expected_child, current_cell_types)
                
                if subtypes_found:
                    found_children.extend(subtypes_found)
                    logger.info(f"✅ Subtype validation: '{expected_child}' satisfied by subtypes: {subtypes_found}")
                else:
                    missing_children.append(expected_child)
                    logger.info(f"❌ Missing expected cell type: '{expected_child}' (no exact match or valid subtypes)")
        
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
            logger.info(f"⚠️ Validation Warning: Expected children not found: {missing_children}")
            logger.info(f"   Available cell types: {sorted(current_cell_types)}")
            
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
        
        logger.info(f"🔄 Updating remaining steps: expected={expected_children}, found={found_children}, missing={missing_children}")
        logger.info(f"🔧 DEBUG: Validation result full structure: {validation_result}")
        
        # No replacement mapping - we should skip steps for missing cell types entirely
        # This avoids creating duplicate steps when some expected children are found
        logger.info(f"🔧 DEBUG: Will skip steps for missing cell types: {missing_children}")
        
        # Update remaining steps
        skipped_count = 0
        logger.info(f"🔍 DEBUG: Scanning {len(steps) - current_step_index - 1} remaining steps...")
        
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
                logger.info(f"🔍 DEBUG: Step {i+1}: {function_name}({param_str})")
            else:
                logger.info(f"🔍 DEBUG: Step {i+1}: {function_name}(cell_type='{step_cell_type}')")
            
            if step_cell_type in missing_children:
                logger.info(f"🔍 DEBUG: Step {i+1} references missing cell type '{step_cell_type}'")
                # Mark this step to be skipped (cell type unavailable)
                steps[i]["skip_reason"] = f"Cell type '{step_cell_type}' not discovered"
                logger.info(f"⏭️ Marked step {i+1} for skipping: {function_name}({step_cell_type})")
                logger.info(f"🔧 DEBUG: Step {i+1} now has skip_reason: {steps[i].get('skip_reason')}")
                skipped_count += 1
            else:
                logger.info(f"🔍 DEBUG: Step {i+1} cell type '{step_cell_type}' not in missing list {missing_children}")
        
        if skipped_count > 0:
            logger.info(f"✅ Step update complete: {skipped_count} marked for skipping")
    
    
    def _execute_validation_step(self, state: ChatState, step_data) -> ChatState:
        """Handle validation step execution (moved from ExecutorNode, matches backup lines 617-656)"""
        logger.info("🔍 Executing validation step...")
        
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
            logger.info(f"✅ Validation passed: {result['message']}")
            # Update available cell types with discovered types
            state["available_cell_types"] = result["available_types"]
            
        elif result["status"] == "partial_success":
            success = True  # Continue but with warnings
            logger.info(f"⚠️ Validation partial: {result['message']}")
            # Update available cell types with what we actually found
            state["available_cell_types"] = result["available_types"]
            
            # Track unavailable cell types exactly like backup
            expected_types = step_data.get("parameters", {}).get("expected_children", [])
            available_types = result.get("available_types", [])
            missing_types = [ct for ct in expected_types if ct not in available_types]
            if missing_types:
                current_unavailable = state.get("unavailable_cell_types", [])
                state["unavailable_cell_types"] = list(set(current_unavailable + missing_types))
                logger.info(f"📋 Added to unavailable cell types: {missing_types}")
            
        else:
            success = False
            error_msg = result["message"]
            logger.info(f"❌ Validation failed: {error_msg}")
            
            # Track all expected cell types as unavailable on complete failure
            expected_types = step_data.get("parameters", {}).get("expected_children", [])
            if expected_types:
                current_unavailable = state.get("unavailable_cell_types", [])
                state["unavailable_cell_types"] = list(set(current_unavailable + expected_types))
                logger.info(f"📋 Added to unavailable cell types (validation failed): {expected_types}")
                
        # CRITICAL FIX: Update subsequent analysis steps based on validation results
        # This MUST run for ANY validation (success, partial, or complete failure)
        # to skip steps for missing cell types or update steps for discovered types
        logger.info(f"🔧 CALLING update_remaining_steps_with_discovered_types with result status: {result.get('status')}")
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
            logger.info(f"🔄 Advanced to step {current_step_index + 2}")
        else:
            logger.info(f"🔄 Advanced to step {current_step_index + 2} (validation failure, but continuing)")
        
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
        logger.info(f"🔄 Light consolidation of {len(steps)} steps...")
        
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
                logger.info(f"   🗑️ Removing consecutive duplicate process_cells({cell_type})")
            else:
                consolidated_steps.append(current_step)
        
        execution_plan["steps"] = consolidated_steps
        logger.info(f"✅ Light consolidation: {len(steps)} → {len(consolidated_steps)} steps")
        
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
        
        logger.info(f"📋 Plan validation summary:")
        logger.info(f"   • {len(discovery_steps)} discovery steps")
        logger.info(f"   • {len(analysis_steps)} analysis steps")
        
        if discovery_steps:
            logger.info(f"🔄 Discovery sequence:")
            for step in discovery_steps:
                parent = step.get("parameters", {}).get("cell_type")
                target = step.get("target_cell_type", "unknown targets")
                logger.info(f"   → process_cells({parent}) → {target}")
        
        # Skip expensive hierarchy resolution - planner already handled this
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question type using a smaller LLM for efficiency"""
        
        classification_prompt = f"""Classify the following biology question into ONE category:
                            Question: "{question}"
                            Categories:
                            1. gene_markers - Questions about genes, markers, or gene expression (e.g., "What are canonical markers for...", "differentiate X from Y using markers", "Is gene X upregulated...", "expression of Y in condition Z")
                            2. pathway_analysis - Questions about biological pathways, processes, or functional enrichment (e.g., "What pathways are enriched...", "biological processes in...")
                            3. cell_abundance - Questions about cell type counts or proportions (e.g., "How many X cells...", "proportion of Y cells")
                            4. general_comparison - General comparison questions not fitting above categories
                            Return ONLY the category name, nothing else.
                            Category:"""
        
        try:
            # Use the LLM call method from base class
            response = self._call_llm(classification_prompt)
            category = response.strip().lower()
            
            # Validate category
            valid_categories = ["gene_markers", "pathway_analysis", 
                              "cell_abundance", "general_comparison"]
            
            if category not in valid_categories:
                logger.info(f"⚠️ Invalid category '{category}', defaulting to general_comparison")
                return "general_comparison"
            
            logger.info(f"📊 Question classified as: {category}")
            return category
            
        except Exception as e:
            logger.info(f"⚠️ Question classification failed: {e}")
            return "general_comparison"
    
    def _critic(self, state: ChatState) -> Dict[str, Any]:
        """
        Cell-type specific LLM-powered gap analysis - Critic Agent
        """
        logger.info("🔍 Starting critic analysis...")
        
        original_question = state["execution_plan"]["original_question"]
        
        # Import the extraction function
        from ...cell_types.validation import extract_cell_types_from_question
        
        # Step 1: Extract mentioned cell types using existing function
        mentioned_types = extract_cell_types_from_question(original_question, self.hierarchy_manager)
        
        # Classify question type for better result filtering
        question_type = self._classify_question_type(original_question)
        
        if not mentioned_types:
            logger.info("📋 No specific cell types mentioned, skipping critic analysis")
            return {"mentioned_cell_types": [], "supplementary_steps": [], "evaluation_complete": True}
        
        supplementary_steps = []
        evaluation_details = {}
        
        # Get unavailable cell types to skip them
        unavailable_cell_types = state.get("unavailable_cell_types", [])
        
        # Step 2: For each mentioned cell type, ask LLM what analyses are needed
        for cell_type in mentioned_types:
            # Skip cell types that were marked as unavailable during validation
            if cell_type in unavailable_cell_types:
                logger.info(f"⏭️ Skipping critic analysis for unavailable cell type: {cell_type}")
                continue
                
            logger.info(f"🔍 Evaluating coverage for cell type: {cell_type}")
            
            # Step 2a: Get LLM recommendations for this specific cell type
            required_analyses = self._get_llm_analysis_requirements(original_question, cell_type)
            
            # Step 2b: Check what was actually performed for this cell type
            performed_analyses = self._get_performed_analyses_for_cell_type(state, cell_type)
            
            # Step 2c: Find gaps and generate steps
            missing_steps = self._generate_missing_steps_for_cell_type(
                cell_type, required_analyses, performed_analyses, state
            )
            
            supplementary_steps.extend(missing_steps)
            evaluation_details[cell_type] = {
                "required_analyses": required_analyses,
                "performed_analyses": performed_analyses,
                "missing_steps_count": len(missing_steps)
            }
            
            if missing_steps:
                logger.info(f"📋 Found {len(missing_steps)} missing steps for {cell_type}")
            else:
                logger.info(f"✅ Complete coverage for {cell_type}")
        
        logger.info(f"🔍 Critic analysis complete: {len(supplementary_steps)} total supplementary steps")
        
        # Add analysis relevance hints based on question type
        all_performed_analyses = {}
        for cell_type, details in evaluation_details.items():
            all_performed_analyses[cell_type] = details["performed_analyses"]
        
        analysis_relevance = self._get_analysis_relevance_hints(question_type, all_performed_analyses)
        
        # Track critic attempts to prevent infinite loops
        critic_attempts = state.get("critic_attempts", 0)
        previously_failed_steps = state.get("previously_failed_supplementary_steps", [])
        
        # Check if we're suggesting the same failed steps again
        new_unique_steps = []
        for step in supplementary_steps:
            step_signature = f"{step.get('function_name')}_{step.get('parameters', {}).get('cell_type', '')}"
            if step_signature not in previously_failed_steps:
                new_unique_steps.append(step)
            else:
                logger.info(f"⚠️ Skipping previously failed supplementary step: {step_signature}")
        
        # CRITICAL FIX: Add retry limit for critic supplementary steps
        MAX_CRITIC_ATTEMPTS = 2
        
        if new_unique_steps and critic_attempts < MAX_CRITIC_ATTEMPTS:
            execution_plan = state.get("execution_plan", {})
            if "steps" in execution_plan:
                # Add supplementary steps to the execution plan
                execution_plan["steps"].extend(new_unique_steps)
                logger.info(f"✅ Added {len(new_unique_steps)} supplementary steps to execution plan (attempt {critic_attempts + 1}/{MAX_CRITIC_ATTEMPTS})")
                
                # Increment attempt counter
                state["critic_attempts"] = critic_attempts + 1
                
                # Mark conversation as incomplete so execution continues
                state["conversation_complete"] = False
                logger.info(f"🔄 Marked conversation as incomplete to continue execution")
            else:
                logger.info("⚠️ No execution plan found to add supplementary steps")
        elif critic_attempts >= MAX_CRITIC_ATTEMPTS:
            logger.info(f"⚠️ Reached maximum critic attempts ({MAX_CRITIC_ATTEMPTS}), stopping supplementary step generation")
            state["conversation_complete"] = True
        elif not new_unique_steps and supplementary_steps:
            logger.info(f"⚠️ All {len(supplementary_steps)} supplementary steps have been tried before and failed, stopping")
            state["conversation_complete"] = True
        
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

                                - display_processed_umap: Display UMAP with cell type annotations. Use when user wants to see cell type annotations on UMAP.

                                SEARCH FUNCTIONS:
                                - search_enrichment_semantic: Search all enrichment terms semantically to find specific pathways or biological processes. Use when user asks about specific pathways, terms, or biological processes that might not appear in standard top results.

                                Task: Determine which analyses are needed for {cell_type} to answer the user's question.

                                IMPORTANT: The cell type "{cell_type}" already exists in the dataset. DO NOT suggest process_cells for this cell type.

                                Consider based on question type:
                                1. Gene/marker questions (differentiate X from Y, markers of X, gene expression) → Use dea_split_by_condition ONLY
                                2. Pathway/biological process questions → Use perform_enrichment_analyses + search_enrichment_semantic
                                3. Cell abundance questions → Use compare_cell_counts
                                4. Specific pathway search → Use search_enrichment_semantic

                                CRITICAL GUIDELINES FOR GENE/MARKER QUESTIONS:
                                - For gene/marker questions, ONLY use dea_split_by_condition
                                - DO NOT include enrichment analyses for marker identification
                                - DO NOT suggest process_cells for already-available cell types
                                - Visualization is optional for gene/marker questions

                                GENERAL GUIDELINES:
                                - Only include display_enrichment_visualization ONCE per cell type
                                - Return ONLY a valid JSON array of function names, nothing else

                                Example response for pathway question: ["perform_enrichment_analyses", "search_enrichment_semantic", "display_enrichment_visualization"]
                                Example response for all kinds of markers: ["dea_split_by_condition"]

                                Required analyses for {cell_type}:"""
        
        try:
            response = self._call_llm(analysis_prompt)
            logger.info(f"🔍 LLM raw response for {cell_type}: '{response}' (length: {len(response)})")
            
            if not response or response.strip() == "":
                logger.info(f"⚠️ Empty LLM response for {cell_type}, using fallback")
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
                logger.info(f"⚠️ LLM returned {type(required_analyses)} instead of list for {cell_type}")
                return ["perform_enrichment_analyses"]
            
            logger.info(f"🧠 LLM recommends for {cell_type}: {required_analyses}")
            return required_analyses
        except json.JSONDecodeError as e:
            logger.info(f"⚠️ JSON parsing failed for {cell_type}: {e}")
            logger.info(f"⚠️ Raw response was: '{response}'")
            return ["perform_enrichment_analyses"]  # Safe fallback
        except Exception as e:
            logger.info(f"⚠️ LLM analysis requirement failed for {cell_type}: {e}")
            return ["perform_enrichment_analyses"]  # Safe fallback

    def _get_performed_analyses_for_cell_type(self, state: ChatState, cell_type: str) -> List[str]:
        """Check what analyses were actually performed for a specific cell type"""
        
        performed_analyses = []
        unavailable_cell_types = state.get("unavailable_cell_types", [])
        execution_history = state.get("execution_history", [])
        
        logger.info(f"🔧 DEBUG: Checking if '{cell_type}' is in unavailable list: {unavailable_cell_types}")
        logger.info(f"🔧 DEBUG: Checking execution history with {len(execution_history)} entries")
        
        # If this cell type is known to be unavailable, don't try to generate steps for it
        if cell_type in unavailable_cell_types:
            logger.info(f"🚫 Cell type '{cell_type}' is unavailable - not generating missing steps")
            # Return dummy analysis to prevent critic from trying to add steps
            return ["analysis_skipped_cell_type_unavailable"]
        
        for i, ex in enumerate(execution_history):
            # CRITICAL FIX: Check successful executions more thoroughly
            success = ex.get("success", False)
            skipped = ex.get("skipped", False)
            
            if success or skipped:
                step_data = ex.get("step", {})
                function_name = step_data.get("function_name")
                params = step_data.get("parameters", {})
                ex_cell_type = params.get("cell_type")
                
                # Debug logging for each matching execution
                if ex_cell_type == cell_type and function_name:
                    status = "skipped" if skipped else "successful"
                    logger.info(f"📋 Found {status} analysis for {cell_type}: {function_name} (execution {i+1})")
                    performed_analyses.append(function_name)
                    
                    # Additional debug for successful enrichment analyses
                    if function_name == "perform_enrichment_analyses" and success:
                        result = ex.get("result", {})
                        if isinstance(result, dict):
                            total_results = sum(result.get(key, {}).get("total_significant", 0) 
                                              for key in ["go", "kegg", "reactome", "gsea"] 
                                              if isinstance(result.get(key), dict))
                            logger.info(f"   → Enrichment analysis had {total_results} total significant results")
        
        # Remove duplicates while preserving order
        unique_performed = []
        for analysis in performed_analyses:
            if analysis not in unique_performed:
                unique_performed.append(analysis)
        
        logger.info(f"📊 Actually performed for {cell_type}: {unique_performed}")
        
        # CRITICAL FIX: If no analyses found but cell type is available, 
        # check if there are any related executions with different parameter names
        if not unique_performed and cell_type not in unavailable_cell_types:
            logger.info(f"🔍 No direct matches found for {cell_type}, checking for related executions...")
            # Look for any executions that might be related to this cell type
            for i, ex in enumerate(execution_history):
                if ex.get("success", False):
                    step_data = ex.get("step", {})
                    function_name = step_data.get("function_name")
                    params = step_data.get("parameters", {})
                    
                    # Check if cell type appears in any parameter values
                    for param_key, param_value in params.items():
                        if isinstance(param_value, str) and cell_type.lower() in param_value.lower():
                            logger.info(f"🔍 Found related execution: {function_name} with {param_key}='{param_value}'")
        
        return unique_performed

    def _generate_missing_steps_for_cell_type(self, cell_type: str, required_analyses: List[str], 
                                            performed_analyses: List[str], state: ChatState) -> List[Dict[str, Any]]:
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
                
                # CRITICAL FIX: Add enrichment_checker enhancement for enrichment analyses
                if required_function == "perform_enrichment_analyses":
                    # Simple approach: add pathway_include with basic keywords to trigger enrichment_checker
                    original_question = state.get("execution_plan", {}).get("original_question", "")
                    
                    # Extract simple pathway-related keywords from the question
                    import re
                    pathway_terms = []
                    for term in ["pathway", "signaling", "process", "function", "regulation", "cycle", "response"]:
                        if term in original_question.lower():
                            # Get surrounding context for this term
                            pattern = rf'\b\w*{term}\w*\b'
                            matches = re.findall(pattern, original_question, re.IGNORECASE)
                            pathway_terms.extend(matches)
                    
                    if pathway_terms:
                        pathway_keywords = ", ".join(pathway_terms[:3])  # Top 3 terms
                        step["parameters"]["pathway_include"] = pathway_keywords
                        step["description"] += f" (with enrichment_checker)"
                        logger.info(f"🔧 Added enrichment_checker trigger with keywords: {pathway_keywords}")
                    else:
                        # Still add pathway_include to trigger enrichment_checker with general enhancement
                        step["parameters"]["pathway_include"] = f"{cell_type} related pathways"
                        step["description"] += " (with enrichment_checker)"
                        logger.info(f"🔧 Added enrichment_checker trigger for {cell_type}")
                
                # Add specific parameters for visualization functions
                if required_function == "display_enrichment_visualization":
                    # Detect what enrichment analysis was actually performed
                    enrichment_type = self._detect_enrichment_type_for_cell(state, cell_type)
                    step["parameters"]["analysis"] = enrichment_type
                
                missing_steps.append(step)
                logger.info(f"🔧 Generated missing step: {required_function}({cell_type})")
        
        return missing_steps
    
    def _detect_enrichment_type_for_cell(self, state: ChatState, cell_type: str) -> str:
        """Detect what type of enrichment analysis was performed for a cell type"""
        execution_history = state.get("execution_history", [])
        
        # Look for the most recent enrichment analysis for this cell type
        for execution in reversed(execution_history):
            step_data = execution.get("step", {})
            if (step_data.get("function_name") == "perform_enrichment_analyses" and
                step_data.get("parameters", {}).get("cell_type") == cell_type and
                execution.get("success")):
                
                # Check what analyses were performed
                result = execution.get("result", {})
                if isinstance(result, dict):
                    # Check which analysis types have results
                    if "go" in result and result["go"].get("total_significant", 0) > 0:
                        return "go"
                    elif "kegg" in result and result["kegg"].get("total_significant", 0) > 0:
                        return "kegg"
                    elif "reactome" in result and result["reactome"].get("total_significant", 0) > 0:
                        return "reactome"
                    elif "gsea" in result and result["gsea"].get("total_significant", 0) > 0:
                        return "gsea"
        
        # Default to go if nothing found
        return "go"
    
    def _get_analysis_relevance_hints(self, question_type: str, performed_analyses: Dict[str, List[str]]) -> Dict[str, Any]:
        """Generate hints about which analyses are most relevant for the question type"""
        
        relevance_map = {
            "gene_markers": {
                "primary": ["dea_split_by_condition"],
                "secondary": ["display_dotplot"]
            },
            "pathway_analysis": {
                "primary": ["perform_enrichment_analyses", "search_enrichment_semantic"],
                "secondary": ["display_enrichment_visualization"]
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
        
        logger.info(f"📝 Generated relevance hints for {question_type} question")
        return hints
    
    def _get_response_guidance(self, question_type: str) -> str:
        """Get specific guidance for response generation based on question type"""
        
        guidance_map = {
            "gene_markers": "Focus on gene expression, markers, and differential expression analysis. Prioritize DEA results showing specific genes, fold changes, and statistical significance. For marker questions, emphasize well-established markers from literature.",
            "pathway_analysis": "Emphasize enrichment analysis results, biological processes, and pathway information. DEA results support pathway findings.",
            "cell_abundance": "Prioritize cell count comparisons and composition visualizations. Focus on quantitative differences between conditions.",
            "general_comparison": "Balance all available analyses based on what best answers the specific question."
        }
        
        return guidance_map.get(question_type, guidance_map["general_comparison"])
        
    def _get_timestamp(self) -> str:
        """Get current timestamp for execution tracking."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
