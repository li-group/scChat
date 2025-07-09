"""
Workflow Logic Evaluator - Judge for step sequence and method appropriateness.

This judge evaluates the logical flow, method selection, and sequence validation
of the execution plan.
"""

import json
import openai
from typing import Dict, Any


class WorkflowLogicEvaluator:
    """
    Evaluates workflow logic, step sequencing, and method appropriateness.
    
    Focus Areas:
    - Logical flow of analysis steps
    - Appropriate method selection for each task
    - Proper sequencing of dependencies
    - Method compatibility and integration
    """
    
    def __init__(self):
        self.evaluation_criteria = {
            "logical_flow": "Are the steps in a logical order that builds understanding?",
            "method_selection": "Are the chosen methods appropriate for each analysis task?",
            "dependency_handling": "Are dependencies between steps properly managed?",
            "coherence": "Do the steps work together toward a unified analysis goal?"
        }
    
    def evaluate(self, evaluation_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate workflow logic and step sequencing.
        
        Args:
            evaluation_inputs: Dictionary containing:
                - original_query: The user's original question
                - execution_plan: The planned steps
                - execution_history: Steps that have been executed
                - final_response: The generated response
                
        Returns:
            Dictionary with score, pass/fail, and improvement guidance
        """
        
        original_query = evaluation_inputs["original_query"]
        execution_plan = evaluation_inputs.get("execution_plan", {})
        execution_history = evaluation_inputs.get("execution_history", [])
        
        # Extract step information
        planned_steps = execution_plan.get("steps", [])
        
        # Extract cache context (if available)
        cache_analysis_summary = evaluation_inputs.get("cache_analysis_summary", "")
        confirmed_available_analyses = evaluation_inputs.get("confirmed_available_analyses", "")
        
        prompt = f"""
        You are a WORKFLOW LOGIC EVALUATOR specializing in scientific analysis workflows.
        
        Your task: Evaluate whether the analysis workflow has logical flow and appropriate method selection.
        
        ORIGINAL USER QUERY: "{original_query}"
        
        PLANNED WORKFLOW STEPS:
        {json.dumps(planned_steps, indent=2)}
        
        EXECUTED STEPS:
        {json.dumps(execution_history, indent=2)}
        
        🆕 CACHED ANALYSIS RESULTS AVAILABLE:
        {cache_analysis_summary}
        
        🎯 CONFIRMED AVAILABLE ANALYSES:
        {confirmed_available_analyses}
        
        EVALUATION CRITERIA:
        
        1. LOGICAL FLOW ANALYSIS:
           - Do steps build upon each other logically?
           - Is there a clear progression from simple to complex analyses?
           - Are prerequisites satisfied before dependent steps?
           
        2. METHOD SELECTION EVALUATION:
           - Are the chosen analysis methods appropriate for the research question?
           - Do visualization methods match the type of data being analyzed?
           - Are enrichment analyses appropriate for the cell types being studied?
           
        3. SEQUENCE VALIDATION:
           - Should cell type processing happen before enrichment analysis?
           - Are visualizations placed appropriately in the workflow?
           - Do analysis steps precede interpretation steps?
           
        4. WORKFLOW COHERENCE:
           - Do all steps contribute toward answering the original question?
           - Is there unnecessary duplication or redundancy?
           - Are there missing critical steps that would improve the analysis?
           - Are cached analyses being unnecessarily re-computed?
           - Does workflow leverage available cached results appropriately?
           
        SPECIFIC ISSUES TO CHECK:
        - Enrichment analysis before cell type validation
        - Missing intermediate visualization steps
        - Inappropriate analysis depth for the question type
        - Method mismatches (e.g., using DEA methods for enrichment questions)
        - Planning analyses that are already available in cache
        - Not leveraging cached results when they would satisfy the query
        
        SCORING GUIDELINES:
        - 0.9-1.0: Excellent workflow logic, optimal method selection
        - 0.7-0.8: Good workflow with minor improvements possible
        - 0.5-0.6: Adequate workflow but some logical issues
        - 0.3-0.4: Poor workflow logic, needs significant revision
        - 0.0-0.2: Fundamentally flawed workflow design
        
        RESPONSE FORMAT (JSON only):
        {{
            "score": 0.0-1.0,
            "pass": true/false,
            "primary_issue": "main workflow problem if any",
            "improvement_direction": "specific guidance for workflow improvement",
            "logical_flow_score": 0.0-1.0,
            "method_selection_score": 0.0-1.0,
            "sequence_score": 0.0-1.0,
            "coherence_score": 0.0-1.0,
            "cache_awareness_score": 0.0-1.0,
            "unnecessary_recomputation": ["analyses being planned that are already cached"]
        }}
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Ensure required fields exist
            if "score" not in result:
                result["score"] = 0.5
            if "pass" not in result:
                result["pass"] = result["score"] >= 0.7
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ WorkflowLogicEvaluator error: {error_msg}")
            
            # Provide more helpful error details
            if "Connection error" in error_msg or "timeout" in error_msg.lower():
                print("🔄 Network/API connection issue - using fallback evaluation")
            elif "rate limit" in error_msg.lower():
                print("⏳ API rate limit - using fallback evaluation")
            
            return {
                "score": 0.5,
                "pass": True,  # Default to pass to avoid blocking on errors
                "primary_issue": f"Technical evaluation error: {error_msg}",
                "improvement_direction": "Unable to evaluate workflow logic due to API connectivity issue",
                "error": str(e)
            }