"""
Critic system for response evaluation and revision.

This module provides the critic agent functionality for evaluating
response quality, detecting impossible requests, and managing
revision cycles to improve analysis results.
"""

import json
import glob
from typing import Dict, Any, List, Literal
import openai

from .cell_type_models import ChatState


class CriticLoopManager:
    """Manages critic agent iteration cycles and prevents infinite loops"""
    MAX_ITERATIONS = 3
    
    @staticmethod
    def should_continue_iteration(state: ChatState) -> bool:
        return (
            state.get("critic_iterations", 0) < CriticLoopManager.MAX_ITERATIONS and
            not state.get("impossible_request_detected", False) and
            state.get("critic_feedback_history", []) and
            state["critic_feedback_history"][-1]["needs_revision"]
        )
    
    @staticmethod
    def initialize_critic_state(state: ChatState) -> ChatState:
        state.setdefault("critic_iterations", 0)
        state.setdefault("critic_feedback_history", [])
        state.setdefault("plan_revision_history", [])
        state.setdefault("original_execution_complete", False)
        state.setdefault("cumulative_analysis_results", {})
        state.setdefault("impossible_request_detected", False)
        state.setdefault("degradation_strategy", None)
        state.setdefault("error_recovery_strategy", None)
        state.setdefault("revision_applied", False)
        return state


class CriticAgent:
    """
    Critic agent for evaluating response quality and suggesting improvements.
    
    The critic agent evaluates the quality of generated responses, identifies
    missing analyses, detects impossible requests, and manages revision cycles
    to improve the overall quality of analysis results.
    """
    
    def __init__(self, simple_cache, hierarchy_manager, history_manager, function_descriptions):
        self.simple_cache = simple_cache
        self.hierarchy_manager = hierarchy_manager
        self.history_manager = history_manager
        self.function_descriptions = function_descriptions
    
    def critic_agent_node(self, state: ChatState) -> ChatState:
        """Main critic agent node for evaluating response quality"""
        
        # Initialize critic state
        state = CriticLoopManager.initialize_critic_state(state)
        
        # Extract information for evaluation
        original_question = state["execution_plan"]["original_question"]
        final_response = state["response"]
        execution_history = state["execution_history"]
        
        # Extract response content
        try:
            response_data = json.loads(final_response)
            response_content = response_data.get("response", "")
        except:
            response_content = final_response
        
        print(f"🔍 Critic Agent - Iteration {state['critic_iterations'] + 1}: Evaluating response quality...")
        
        # Perform comprehensive evaluation
        evaluation = self._evaluate_response_quality(
            original_question, 
            response_content, 
            execution_history,
            state
        )
        
        # Update state
        state["critic_iterations"] += 1
        state["critic_feedback_history"].append(evaluation)
        state["original_execution_complete"] = True
        
        # Accumulate analysis results for context
        self._accumulate_analysis_results(state, evaluation)
        
        print(f"✅ Critic evaluation complete:")
        print(f"   • Relevance: {evaluation['relevance_score']:.2f}")
        print(f"   • Completeness: {evaluation['completeness_score']:.2f}")
        print(f"   • Needs revision: {evaluation['needs_revision']}")
        print(f"   • Missing analyses: {evaluation['missing_analyses']}")
        
        return state

    def planner_reviser_node(self, state: ChatState) -> ChatState:
        """Plan revision node with comprehensive error handling"""
        
        iteration = state.get("critic_iterations", 0)
        latest_feedback = state["critic_feedback_history"][-1]
        
        print(f"🔄 Plan Revision - Iteration {iteration}")
        
        # Step 1: Detect impossible requests (CRITICAL for loop prevention)
        impossible_patterns = self._detect_impossible_requests(state)
        
        if self._has_impossible_patterns(impossible_patterns):
            print("🚫 Impossible requests detected - flagging for graceful degradation")
            state["impossible_request_detected"] = True
            state["degradation_strategy"] = self._handle_impossible_requests(impossible_patterns, state)
            return state
        
        # Step 2: Handle execution errors
        error_recovery = {}
        if state.get("execution_history"):
            error_recovery = self._handle_execution_errors(state["execution_history"], latest_feedback)
            state["error_recovery_strategy"] = error_recovery
        
        # Step 3: Handle content gaps
        content_revision = self._handle_content_gaps(latest_feedback, state["execution_plan"])
        
        # Step 4: Generate revised plan
        revised_plan = self._generate_revised_plan(
            state["execution_plan"],
            content_revision,
            error_recovery,
            iteration
        )
        
        # Step 5: Update state for status checker
        state["execution_plan"] = revised_plan
        state["current_step_index"] = 0  # Reset execution index
        state["revision_applied"] = True
        
        # Record revision history
        state["plan_revision_history"].append({
            "iteration": iteration,
            "reason": latest_feedback["reasoning"],
            "content_changes": content_revision,
            "error_recovery": error_recovery
        })
        
        print(f"✅ Plan revised - Added {len(content_revision.get('add_steps', []))} steps")
        
        return state

    def impossible_handler_node(self, state: ChatState) -> ChatState:
        """Handle impossible requests with graceful degradation"""
        
        degradation_strategy = state.get("degradation_strategy", {})
        
        print("🚫 Handling impossible request with graceful degradation...")
        
        # Generate explanation response
        explanation = self._generate_impossible_request_explanation(
            state["execution_plan"]["original_question"],
            degradation_strategy,
            state
        )
        
        # Update response
        state["response"] = json.dumps({
            "response": explanation,
            "response_type": "impossible_request_handled"
        })
        
        state["conversation_complete"] = True
        
        return state

    def route_from_critic(self, state: ChatState) -> Literal["revise", "complete", "impossible"]:
        """Enhanced routing from critic agent with impossible request detection"""
        
        # Check for impossible requests first
        if state.get("impossible_request_detected"):
            print("🚫 Routing to impossible request handler")
            return "impossible"
        
        # Check iteration limit
        if state["critic_iterations"] >= CriticLoopManager.MAX_ITERATIONS:
            print(f"🔄 Maximum iterations ({CriticLoopManager.MAX_ITERATIONS}) reached - completing")
            return "complete"
        
        # Check if revision is needed
        if CriticLoopManager.should_continue_iteration(state):
            print(f"🔄 Critic recommends revision - starting iteration {state['critic_iterations'] + 1}")
            return "revise"
        
        print("✅ Critic satisfied with response - completing")
        return "complete"

    # ========== CRITIC EVALUATION METHODS ==========
    def _evaluate_response_quality(self, question: str, response: str, 
                                execution_history: List, state: ChatState) -> Dict:
        """Enhanced critic evaluation that is cache-aware"""
        
        # Prepare execution summary
        execution_summary = self._summarize_execution_history(execution_history)
        
        # Get available cell types for context
        available_cell_types = ", ".join(state.get("available_cell_types", []))
        
        # Check for repeated failures
        repeated_failures = self._detect_repeated_failures(state)
        
        relevant_cell_types = self._get_relevant_cell_types_from_context(state)
        cache_analysis_summary = self._get_cache_analysis_summary(relevant_cell_types)
        
        comprehensive_analysis_context = self._build_comprehensive_analysis_context(state, execution_history)
        
        # NEW: Check what analyses are actually available vs. what's being requested
        available_analyses = self._get_actually_available_analyses(relevant_cell_types)
        
        critic_prompt = f"""
                        You are a scientific analysis critic evaluating single-cell RNA-seq analysis results.

                        ORIGINAL QUESTION:
                        {question}

                        GENERATED RESPONSE:
                        {response}

                        ANALYSES PERFORMED IN CURRENT SESSION:
                        {execution_summary}

                        🆕 CACHED ANALYSIS RESULTS AVAILABLE:
                        {cache_analysis_summary}
                        
                        🎯 CONFIRMED AVAILABLE ANALYSES:
                        {available_analyses}

                        🆕 COMPREHENSIVE ANALYSIS CONTEXT (Current + Cache + History):
                        {comprehensive_analysis_context}

                        AVAILABLE CELL TYPES IN DATASET:
                        {available_cell_types}

                        Available functions:
                        {self._summarize_functions(self.function_descriptions)}

                        CONTEXT:
                        - Iteration: {state.get('critic_iterations', 0) + 1}/3
                        - Previous failures: {repeated_failures}

                        CRITICAL CACHE AWARENESS RULES:
                        1. If an analysis is listed in "CONFIRMED AVAILABLE ANALYSES", it has been completed and should NOT be marked as missing
                        2. Only mark analyses as missing if they are truly unavailable AND needed to answer the question
                        3. If cached results exist but aren't being utilized in the response, mark it as a presentation issue, not missing analysis
                        4. Focus on whether the RESPONSE adequately answers the question using available data

                        Evaluate the response on these criteria:

                        1. RELEVANCE (0.0-1.0): Does the response directly address what was asked?

                        2. COMPLETENESS (0.0-1.0): Are all parts of the question covered using available analyses?

                        3. PRESENTATION QUALITY: Is available analysis data properly presented and explained?

                        4. MISSING ELEMENTS: What specific analyses are TRULY missing (not cached anywhere)?

                        IMPORTANT CONSTRAINTS:
                        - DO NOT suggest analyses that are already available (check CONFIRMED AVAILABLE ANALYSES)
                        - Only suggest new analyses if they are genuinely needed and unavailable
                        - If data exists but presentation is poor, recommend response improvement, not new analyses

                        Respond in JSON format:
                        {{
                            "relevance_score": 0.0-1.0,
                            "completeness_score": 0.0-1.0,
                            "needs_revision": true/false,
                            "missing_analyses": ["only genuinely missing analyses with function(cell_type) format (The function name should be exactly same as it is in the available function)"],
                            "recommendations": ["actionable advice for improvement"],
                            "reasoning": "Detailed explanation focusing on cache utilization",
                            "evaluation_type": "content_gap|presentation_issue|truly_missing|satisfied",
                            "impossible_requests": ["requests that cannot be fulfilled"],
                            "cache_utilization": "Assessment of how well available cached results were used"
                        }}
                        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                temperature=0.1,
                messages=[{"role": "user", "content": critic_prompt}],
                response_format={"type": "json_object"}
            )
            
            evaluation = json.loads(response.choices[0].message.content)
            evaluation = self._sanitize_critic_evaluation_with_cache_awareness(evaluation, state)
            
            return evaluation
            
        except Exception as e:
            print(f"❌ Critic evaluation failed: {e}")
            # Fallback evaluation
            return {
                "relevance_score": 0.8,
                "completeness_score": 0.8,
                "needs_revision": False,
                "missing_analyses": [],
                "recommendations": [],
                "reasoning": f"Critic evaluation failed: {e}",
                "evaluation_type": "error",
                "impossible_requests": [],
                "cache_utilization": "Error in evaluation"
            }

    def _get_cache_analysis_summary(self, relevant_cell_types: List[str]) -> str:
        """Get summary of what analyses are available in cache"""
        
        cache_summary = "CACHED ANALYSIS AVAILABILITY:\n"
        
        for cell_type in relevant_cell_types:
            cache_summary += f"\n🧬 {cell_type.upper()}:\n"
            
            # Check for enrichment analyses in cache
            enrichment_available = []
            enrichment_patterns = self.simple_cache._get_cache_file_patterns("enrichment", cell_type, 
                                                                            {"analyses": ["reactome", "go", "kegg", "gsea"]})
            for pattern in enrichment_patterns:
                matching_files = glob.glob(pattern)
                for file_path in matching_files:
                    if self.simple_cache._is_file_recent(file_path):
                        analysis_name = self.simple_cache._extract_analysis_name_from_path(file_path)
                        enrichment_available.append(analysis_name)
            
            if enrichment_available:
                cache_summary += f"  ✅ ENRICHMENT ANALYSES: {', '.join(set(enrichment_available))}\n"
            else:
                cache_summary += f"  ❌ ENRICHMENT ANALYSES: None available\n"
            
            # Check for DEA analyses in cache
            dea_available = []
            dea_patterns = self.simple_cache._get_cache_file_patterns("dea", cell_type)
            for pattern in dea_patterns:
                matching_files = glob.glob(pattern)
                for file_path in matching_files:
                    if self.simple_cache._is_file_recent(file_path):
                        condition = self.simple_cache._extract_condition_from_path(file_path)
                        dea_available.append(condition)
            
            if dea_available:
                cache_summary += f"  ✅ DEA ANALYSES: {', '.join(dea_available)}\n"
            else:
                cache_summary += f"  ❌ DEA ANALYSES: None available\n"
        
        return cache_summary

    def _build_comprehensive_analysis_context(self, state: ChatState, execution_history: List) -> str:
        """Build comprehensive context from current session + cache + function history"""
        
        context = "COMPREHENSIVE ANALYSIS STATUS:\n\n"
        
        # 1. Current session analyses
        current_session_analyses = []
        for execution in execution_history:
            if execution["success"]:
                func_name = execution["step"]["function_name"]
                if func_name in ["perform_enrichment_analyses", "dea_split_by_condition", "process_cells"]:
                    cell_type = execution["step"]["parameters"].get("cell_type", "unknown")
                    current_session_analyses.append(f"{func_name}({cell_type})")
        
        if current_session_analyses:
            context += "📋 CURRENT SESSION:\n"
            for analysis in current_session_analyses:
                context += f"  ✅ {analysis}\n"
        else:
            context += "📋 CURRENT SESSION: No analyses performed\n"
        
        # 2. Function history (previous sessions)
        recent_analyses = self.history_manager.get_recent_executions("perform_enrichment_analyses", limit=5)
        recent_analyses.extend(self.history_manager.get_recent_executions("dea_split_by_condition", limit=5))
        recent_analyses.extend(self.history_manager.get_recent_executions("process_cells", limit=5))
        
        if recent_analyses:
            context += "\n📜 PREVIOUS SESSIONS (Last 5 of each type):\n"
            for execution in recent_analyses:
                if execution.get("success"):
                    func_name = execution["function_name"]
                    cell_type = execution["parameters"].get("cell_type", "unknown")
                    timestamp = execution.get("timestamp", "unknown")
                    context += f"  ✅ {func_name}({cell_type}) - {timestamp}\n"
        else:
            context += "\n📜 PREVIOUS SESSIONS: No recent analyses found\n"
        
        # 3. Cache status summary
        relevant_cell_types = self._get_relevant_cell_types_from_context(state)
        cache_stats = {}
        for cell_type in relevant_cell_types:
            insights = self.simple_cache.get_analysis_insights(cell_type)
            cache_stats[cell_type] = {
                "enrichment_analyses": len(insights.get("enrichment_insights", {})),
                "dea_analyses": len(insights.get("dea_insights", {}))
            }
        
        if cache_stats:
            context += "\n💾 CACHE STATUS:\n"
            for cell_type, stats in cache_stats.items():
                context += f"  {cell_type}: {stats['enrichment_analyses']} enrichment, {stats['dea_analyses']} DEA analyses\n"
        
        return context

    def _get_actually_available_analyses(self, relevant_cell_types: List[str]) -> str:
        """Get a definitive list of what analyses are actually available"""
        available = []
        
        for cell_type in relevant_cell_types:
            # Check DEA availability
            if self.simple_cache._check_dea_cache_exists(cell_type):
                available.append(f"✅ dea_split_by_condition({cell_type}) - CACHED")
            
            # Check enrichment availability  
            if self.simple_cache._check_enrichment_cache_exists(cell_type):
                available.append(f"✅ perform_enrichment_analyses({cell_type}) - CACHED")
        
        return "\n".join(available) if available else "No cached analyses found"

    # ========== HELPER METHODS ==========
    
    def _summarize_functions(self, functions: List[Dict]) -> str:
        """Summarize available functions for critic context"""
        if not functions:
            return "No functions available"
        
        summary = []
        for func in functions:
            name = func.get("name", "unknown")
            description = func.get("description", "").split(".")[0]  # First sentence only
            summary.append(f"- {name}: {description}")
        
        return "\n".join(summary)

    def _summarize_execution_history(self, execution_history: List) -> str:
        """Summarize execution history for critic evaluation"""
        if not execution_history:
            return "No analyses performed in current session"
        
        summary = []
        for execution in execution_history:
            step = execution.get("step", {})
            func_name = step.get("function_name", "unknown")
            success = execution.get("success", False)
            cell_type = step.get("parameters", {}).get("cell_type", "unknown")
            
            status = "✅" if success else "❌"
            summary.append(f"{status} {func_name}({cell_type})")
        
        return "\n".join(summary)

    def _get_relevant_cell_types_from_context(self, state: ChatState) -> List[str]:
        """Extract relevant cell types from the current context"""
        cell_types = set()
        
        # From execution plan
        if state.get("execution_plan", {}).get("steps"):
            for step in state["execution_plan"]["steps"]:
                cell_type = step.get("parameters", {}).get("cell_type")
                if cell_type and cell_type != "overall":
                    cell_types.add(cell_type)
        
        # From execution history
        for execution in state.get("execution_history", []):
            if execution.get("step", {}).get("parameters", {}).get("cell_type"):
                cell_type = execution["step"]["parameters"]["cell_type"]
                if cell_type != "overall":
                    cell_types.add(cell_type)
        
        return list(cell_types)[:5]  # Limit to 5 most relevant

    def _detect_repeated_failures(self, state: ChatState) -> List[str]:
        """Detect repeated failures in critic feedback"""
        failures = []
        
        if state.get("critic_feedback_history"):
            failure_counts = {}
            for feedback in state["critic_feedback_history"]:
                for missing in feedback.get("missing_analyses", []):
                    failure_counts[missing] = failure_counts.get(missing, 0) + 1
            
            failures = [analysis for analysis, count in failure_counts.items() if count >= 2]
        
        return failures

    def _is_analysis_cached(self, analysis_description: str, cell_type: str) -> bool:
        """Check if a specific analysis is available in cache"""
        
        analysis_lower = analysis_description.lower()
        
        if ("enrichment" in analysis_lower or "gsea" in analysis_lower or 
            "pathway" in analysis_lower or "perform_enrichment_analyses" in analysis_lower):
            # Check enrichment cache directly
            return self.simple_cache._check_enrichment_cache_exists(cell_type)
        
        elif ("dea" in analysis_lower or "differential" in analysis_lower or 
              "dea_split_by_condition" in analysis_lower):
            # Check DEA cache directly
            return self.simple_cache._check_dea_cache_exists(cell_type)
        
        elif "process" in analysis_lower:
            # Check if processed results exist
            if self.hierarchy_manager:
                return self.hierarchy_manager.is_valid_cell_type(cell_type)
        
        return False

    def _extract_cell_type_from_analysis(self, analysis_description: str) -> str:
        """Extract cell type from analysis description"""
        # Simple extraction - can be enhanced
        words = analysis_description.split()
        for i, word in enumerate(words):
            if word.lower() == "cell" and i > 0:
                return f"{words[i-1]} {word}"
        return "unknown"

    def _detect_impossible_requests(self, state: ChatState) -> Dict:
        """Detect impossible requests that cannot be fulfilled"""
        return {
            "repeated_cell_type_failures": [],
            "hierarchical_dead_ends": [],
            "cache_misunderstanding": []
        }

    def _has_impossible_patterns(self, impossible_patterns: Dict) -> bool:
        """Check if impossible patterns exist"""
        return any(patterns for patterns in impossible_patterns.values())

    def _handle_impossible_requests(self, impossible_patterns: Dict, state: ChatState) -> Dict:
        """Handle impossible requests and create degradation strategy"""
        return {
            "type": "impossible_request",
            "patterns": impossible_patterns,
            "message": "Request cannot be fulfilled with available data"
        }

    def _handle_content_gaps(self, critic_feedback: Dict, current_plan: Dict) -> Dict:
        """Handle content gaps identified by critic"""
        missing_analyses = critic_feedback.get("missing_analyses", [])
        add_steps = []
        
        print(f"🔍 Content gaps: Processing {len(missing_analyses)} missing analyses: {missing_analyses}")
        
        for analysis in missing_analyses:
            # Parse the analysis string (e.g., "perform_enrichment_analyses(T cell)")
            print(f"🔍 Parsing missing analysis: {analysis}")
            step = self._parse_missing_analysis_to_step(analysis)
            if step:
                add_steps.append(step)
                print(f"✅ Parsed step: {step}")
            else:
                print(f"❌ Failed to parse: {analysis}")
        
        print(f"🔍 Content gaps result: {len(add_steps)} steps to add")
        return {
            "add_steps": add_steps,
            "missing_analyses": missing_analyses
        }

    def _handle_execution_errors(self, execution_history: List, critic_feedback: Dict) -> Dict:
        """Handle execution errors from history"""
        return {
            "retry_failed": [],
            "parameter_fixes": []
        }

    def _generate_revised_plan(self, original_plan: Dict, content_revision: Dict, 
                             error_recovery: Dict, iteration: int) -> Dict:
        """Generate revised execution plan"""
        revised_plan = original_plan.copy()
        
        # Add missing analysis steps
        if content_revision.get("add_steps"):
            current_steps = revised_plan.get("steps", [])
            new_steps = content_revision["add_steps"]
            
            # Add new steps to the plan
            revised_plan["steps"] = current_steps + new_steps
            print(f"🔄 Plan Revision - Added {len(new_steps)} missing analysis steps")
        
        # Handle retry logic for failed steps
        if error_recovery.get("retry_failed"):
            # Implementation for retrying failed steps would go here
            pass
        
        return revised_plan

    def _parse_missing_analysis_to_step(self, analysis_str: str) -> Dict:
        """Parse missing analysis string into executable step"""
        import re
        
        # Pattern to match function_name(parameter)
        pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]*)\)'
        match = re.match(pattern, analysis_str)
        
        if match:
            function_name = match.group(1)
            parameter_str = match.group(2).strip()
            
            # Create step based on function name
            if function_name == "perform_enrichment_analyses":
                return {
                    "description": f"Perform enrichment analyses for {parameter_str}",
                    "function_name": "perform_enrichment_analyses",
                    "parameters": {"cell_type": parameter_str}
                }
            elif function_name == "dea_split_by_condition":
                return {
                    "description": f"Perform differential expression analysis for {parameter_str}",
                    "function_name": "dea_split_by_condition", 
                    "parameters": {"cell_type": parameter_str}
                }
            elif function_name == "process_cells":
                return {
                    "description": f"Process cells for {parameter_str}",
                    "function_name": "process_cells",
                    "parameters": {"cell_type": parameter_str}
                }
            elif function_name == "display_enrichment_visualization":
                return {
                    "step_type": "visualization",
                    "description": f"Display enrichment visualization for {parameter_str}",
                    "function_name": "display_enrichment_visualization",
                    "parameters": {"cell_type": parameter_str, "analysis": "gsea", "plot_type": "both"}
                }
            elif function_name == "display_enrichment_barplot":
                return {
                    "step_type": "visualization", 
                    "description": f"Display enrichment barplot for {parameter_str}",
                    "function_name": "display_enrichment_barplot",
                    "parameters": {"cell_type": parameter_str, "analysis": "gsea"}
                }
            elif function_name == "display_enrichment_dotplot":
                return {
                    "step_type": "visualization",
                    "description": f"Display enrichment dotplot for {parameter_str}",
                    "function_name": "display_enrichment_dotplot", 
                    "parameters": {"cell_type": parameter_str, "analysis": "gsea"}
                }
            elif function_name == "display_gsea_dotplot":
                return {
                    "step_type": "visualization",
                    "description": f"Display GSEA dotplot for {parameter_str}",
                    "function_name": "display_gsea_dotplot",
                    "parameters": {"cell_type": parameter_str, "condition": None, "top_n": 20}
                }
        
        print(f"⚠️ Unhandled missing analysis format: {analysis_str}")
        return None

    def _generate_impossible_request_explanation(self, original_question: str, 
                                               degradation_strategy: Dict, state: ChatState) -> str:
        """Generate explanation for impossible requests"""
        return f"I understand you're asking about: {original_question}\n\nHowever, this analysis cannot be completed with the available data and resources. The system has attempted multiple approaches but encountered fundamental limitations."

    def _sanitize_critic_evaluation_with_cache_awareness(self, evaluation: Dict, state: ChatState) -> Dict:
        """Sanitize and validate critic evaluation with enhanced cache awareness"""
        # Ensure required fields exist with defaults
        evaluation.setdefault("relevance_score", 0.8)
        evaluation.setdefault("completeness_score", 0.8)
        evaluation.setdefault("needs_revision", False)
        evaluation.setdefault("missing_analyses", [])
        evaluation.setdefault("recommendations", [])
        evaluation.setdefault("reasoning", "No reasoning provided")
        evaluation.setdefault("evaluation_type", "unknown")
        evaluation.setdefault("impossible_requests", [])
        evaluation.setdefault("cache_utilization", "Unknown")
        
        # Get relevant cell types from context
        relevant_cell_types = self._get_relevant_cell_types_from_context(state)
        
        # ENHANCED CACHE CHECK: Remove analyses that are actually cached
        missing_analyses = evaluation.get("missing_analyses", [])
        actually_missing = []
        cache_satisfied = []
        
        for analysis in missing_analyses:
            # Extract cell type from analysis description
            cell_type = self._extract_cell_type_from_analysis(analysis)
            
            if cell_type and cell_type in relevant_cell_types:
                # Check if this analysis is actually available in cache
                is_cached = self._is_analysis_cached(analysis, cell_type)
                
                if is_cached:
                    cache_satisfied.append(analysis)
                    print(f"🎯 Override: {analysis} for {cell_type} is cached - removing from missing")
                else:
                    actually_missing.append(analysis)
            else:
                # Keep analyses for unknown cell types as potentially missing
                actually_missing.append(analysis)
        
        # Update evaluation with filtered missing analyses
        evaluation["missing_analyses"] = actually_missing
        evaluation["cache_satisfied_analyses"] = cache_satisfied
        
        # AGGRESSIVE REVISION PREVENTION: If most/all analyses are cached, don't revise
        if cache_satisfied and len(actually_missing) <= 1:
            evaluation["needs_revision"] = False
            evaluation["reasoning"] += f" [CACHE OVERRIDE: {len(cache_satisfied)} analyses available in cache: {cache_satisfied}]"
            evaluation["evaluation_type"] = "cache_satisfied"
            print(f"🎯 AGGRESSIVE OVERRIDE: {len(cache_satisfied)} analyses cached, only {len(actually_missing)} missing - NO REVISION")
        
        # FINAL CHECK: High scores + no missing analyses = no revision
        relevance = evaluation.get("relevance_score", 0.0)
        completeness = evaluation.get("completeness_score", 0.0)
        
        if not actually_missing and relevance >= 0.7 and completeness >= 0.7:
            evaluation["needs_revision"] = False
            print(f"🎯 FINAL OVERRIDE: No missing analyses and good scores (R:{relevance:.2f}, C:{completeness:.2f}) - NO REVISION")
        
        return evaluation

    def _accumulate_analysis_results(self, state: ChatState, evaluation: Dict):
        """Accumulate analysis results for context building"""
        if "cumulative_analysis_results" not in state:
            state["cumulative_analysis_results"] = {}
        
        # Add current evaluation to cumulative results
        iteration = state.get("critic_iterations", 0)
        state["cumulative_analysis_results"][f"iteration_{iteration}"] = {
            "evaluation": evaluation,
            "timestamp": "current"
        }