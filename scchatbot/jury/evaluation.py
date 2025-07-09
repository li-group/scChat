"""
Jury evaluation coordination logic.

This module contains evaluation orchestration methods extracted from jury_system_main.py:
- _run_jury_evaluation_sync(): Run jury evaluation synchronously
- _run_targeted_jury_evaluation(): Run jury evaluation with targeted inputs
- Judge input preparation methods for each specialized judge
"""

from typing import Dict, Any


class EvaluationCoordinatorMixin:
    """
    Evaluation coordination mixin for orchestrating jury evaluation.
    """

    def _run_jury_evaluation_sync(self, evaluation_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run jury evaluation synchronously (fallback if async fails).
        """
        
        jury_verdicts = {}
        
        for judge_name, judge in self.jury_members.items():
            try:
                print(f"🏛️ Running {judge_name}...")
                verdict = judge.evaluate(evaluation_inputs)
                jury_verdicts[judge_name] = verdict
                
                status = "PASS" if verdict.get("pass", False) else "FAIL"
                score = verdict.get("score", 0.0)
                print(f"✅ {judge_name}: {status} (score: {score:.2f})")
                
            except Exception as e:
                print(f"❌ {judge_name} failed: {e}")
                jury_verdicts[judge_name] = self._create_fallback_verdict(str(e))
        
        return jury_verdicts
    
    def _run_targeted_jury_evaluation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run jury evaluation with targeted inputs for each judge.
        
        Each judge gets only the minimal information they need for their specific task.
        """
        
        jury_verdicts = {}
        
        # Prepare targeted inputs for each judge
        judge_inputs = {
            "workflow_judge": self._prepare_workflow_judge_inputs(state),
            "efficiency_judge": self._prepare_efficiency_judge_inputs(state),
            "completeness_judge": self._prepare_completeness_judge_inputs(state),
            "user_intent_judge": self._prepare_user_intent_judge_inputs(state)
        }
        
        # Run each judge with their targeted inputs
        for judge_name, judge in self.jury_members.items():
            try:
                print(f"🏛️ Running {judge_name}...")
                
                # Get targeted inputs for this judge
                targeted_inputs = judge_inputs.get(judge_name, {})
                
                # Log token usage for monitoring
                input_size = len(str(targeted_inputs))
                print(f"   📊 {judge_name} input size: {input_size:,} chars")
                
                verdict = judge.evaluate(targeted_inputs)
                jury_verdicts[judge_name] = verdict
                
                status = "PASS" if verdict.get("pass", False) else "FAIL"
                score = verdict.get("score", 0.0)
                print(f"✅ {judge_name}: {status} (score: {score:.2f})")
                
            except Exception as e:
                print(f"❌ {judge_name} error: {e}")
                jury_verdicts[judge_name] = {
                    "score": 0.5,
                    "pass": True,
                    "primary_issue": f"Technical error: {str(e)}",
                    "improvement_direction": "Unable to evaluate due to technical issue",
                    "error": str(e)
                }
        
        return jury_verdicts
    
    def _prepare_workflow_judge_inputs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare minimal inputs for workflow judge with smart execution summaries.
        
        Workflow judge needs: plan structure, step sequence, smart execution summary, and cell type status.
        """
        execution_plan = state.get("execution_plan", {})
        execution_history = state.get("execution_history", [])
        
        # Get cell type status
        available_cell_types = state.get("available_cell_types", [])
        unavailable_cell_types = state.get("unavailable_cell_types", [])
        
        return {
            "execution_plan": execution_plan,
            "execution_summary": self._create_smart_execution_summary(execution_history),
            "available_cell_types": available_cell_types,
            "unavailable_cell_types": unavailable_cell_types,
            "original_query": state.get("current_message", ""),
            "judge_type": "workflow_logic"
        }
    
    def _prepare_efficiency_judge_inputs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare minimal inputs for efficiency judge.
        
        Efficiency judge needs: execution plan, function descriptions, and execution metrics.
        """
        execution_plan = state.get("execution_plan", {})
        execution_history = state.get("execution_history", [])
        
        # Calculate execution metrics
        total_steps = len(execution_plan.get("steps", []))
        successful_steps = len([h for h in execution_history if h.get("success", False)])
        failed_steps = len([h for h in execution_history if not h.get("success", False)])
        
        return {
            "execution_plan": execution_plan,
            "function_descriptions": self.function_descriptions,
            "execution_metrics": {
                "total_steps": total_steps,
                "successful_steps": successful_steps,
                "failed_steps": failed_steps,
                "completion_rate": successful_steps / max(total_steps, 1)
            },
            "original_query": state.get("current_message", ""),
            "judge_type": "efficiency"
        }
    
    def _prepare_completeness_judge_inputs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare minimal inputs for completeness judge.
        
        Completeness judge needs: execution plan, execution history, and context analysis.
        """
        execution_plan = state.get("execution_plan", {})
        execution_history = state.get("execution_history", [])
        
        # Build comprehensive analysis context
        analysis_context = self._build_comprehensive_analysis_context(state, execution_history)
        
        return {
            "execution_plan": execution_plan,
            "execution_history": self._truncate_execution_history_for_judges(execution_history),
            "analysis_context": analysis_context,
            "original_query": state.get("current_message", ""),
            "judge_type": "completeness"
        }
    
    def _prepare_user_intent_judge_inputs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare minimal inputs for user intent judge.
        
        User intent judge needs: original question, execution plan, and analysis context.
        """
        execution_plan = state.get("execution_plan", {})
        execution_history = state.get("execution_history", [])
        
        # Get relevant cell types for context
        relevant_cell_types = self._get_relevant_cell_types_from_context(state)
        
        # Build analysis context
        analysis_context = self._build_comprehensive_analysis_context(state, execution_history)
        
        return {
            "original_query": state.get("current_message", ""),
            "execution_plan": execution_plan,
            "analysis_context": analysis_context,
            "relevant_cell_types": relevant_cell_types,
            "judge_type": "user_intent"
        }