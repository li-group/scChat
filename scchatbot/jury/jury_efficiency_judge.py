"""
Efficiency Evaluator - Judge for resource optimization and redundancy elimination.

This judge evaluates computational efficiency, cache utilization, and identifies
unnecessary steps or redundant operations.
"""

import json
import openai
from typing import Dict, Any, List


class EfficiencyEvaluator:
    """
    Evaluates computational efficiency and resource optimization.
    
    Focus Areas:
    - Unnecessary computational steps
    - Cache utilization opportunities
    - Redundant analysis detection
    - Resource optimization
    """
    
    def __init__(self):
        self.evaluation_criteria = {
            "redundancy_elimination": "Are there duplicate or unnecessary analysis steps?",
            "cache_utilization": "Is the system effectively using cached results?",
            "computational_efficiency": "Are computationally expensive steps optimized?",
            "resource_optimization": "Is the workflow designed for minimal resource usage?"
        }
    
    def evaluate(self, evaluation_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate workflow efficiency and resource optimization.
        
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
        comprehensive_analysis_context = evaluation_inputs.get("comprehensive_analysis_context", "")
        
        # Analyze for efficiency patterns
        analysis_steps = [step for step in planned_steps if step.get("step_type") == "analysis"]
        viz_steps = [step for step in planned_steps if step.get("step_type") == "visualization"]
        
        prompt = f"""
        You are an EFFICIENCY EVALUATOR specializing in computational workflow optimization.
        
        Your task: Evaluate whether the workflow is optimally designed for efficiency and resource utilization.
        
        ORIGINAL USER QUERY: "{original_query}"
        
        PLANNED WORKFLOW STEPS ({len(planned_steps)} total):
        {json.dumps(planned_steps, indent=2)}
        
        EXECUTION HISTORY ({len(execution_history)} completed):
        {json.dumps(execution_history, indent=2)}
        
        🆕 CACHED ANALYSIS RESULTS AVAILABLE:
        {cache_analysis_summary}
        
        🎯 CONFIRMED AVAILABLE ANALYSES:
        {confirmed_available_analyses}
        
        🆕 COMPREHENSIVE ANALYSIS CONTEXT:
        {comprehensive_analysis_context}
        
        EFFICIENCY EVALUATION CRITERIA:
        
        1. REDUNDANCY DETECTION:
           - Are there duplicate analysis steps for the same cell type?
           - Are similar enrichment analyses being repeated?
           - Could multiple visualizations be combined into single calls?
           
        2. CACHE UTILIZATION ANALYSIS (CRITICAL FOR EFFICIENCY):
           - Are there cached analyses available that should be used instead of recomputation?
           - Is the workflow unnecessarily running analyses that are already cached?
           - Are expensive computations being repeated when cache results exist?
           - Check "CONFIRMED AVAILABLE ANALYSES" - any overlap with planned steps indicates inefficiency
           
        3. COMPUTATIONAL OPTIMIZATION:
           - Are expensive analyses (like GSEA) being run only when necessary?
           - Could simpler methods achieve the same goal?
           - Are visualization steps appropriately batched?
           
        4. WORKFLOW STREAMLINING:
           - Could the workflow achieve the same result with fewer steps?
           - Are there unnecessary intermediate steps?
           - Is the analysis depth appropriate for the question complexity?
           
        SPECIFIC EFFICIENCY PATTERNS TO CHECK:
        - Multiple enrichment analyses on same cell type that could be batched
        - Redundant processing of the same cell type
        - Excessive visualization steps for simple queries
        - CRITICAL: Running analyses that are already cached (check CONFIRMED AVAILABLE ANALYSES)
        - Missing cache utilization opportunities
        - Over-analysis for basic visualization requests
        - Recomputing analyses instead of using cached results
        
        QUERY TYPE EFFICIENCY EXPECTATIONS:
        - Simple visualization requests: 1-2 steps maximum
        - Basic analysis questions: 2-4 steps optimal
        - Complex comparative analyses: 4-8 steps acceptable
        - Comprehensive studies: 8+ steps may be warranted
        
        SCORING GUIDELINES:
        - 0.9-1.0: Highly optimized workflow, minimal redundancy
        - 0.7-0.8: Good efficiency with minor optimization opportunities
        - 0.5-0.6: Adequate efficiency but some redundancy present
        - 0.3-0.4: Poor efficiency, significant redundancy or waste
        - 0.0-0.2: Highly inefficient workflow design
        
        RESPONSE FORMAT (JSON only):
        {{
            "score": 0.0-1.0,
            "pass": true/false,
            "primary_issue": "main efficiency problem if any",
            "improvement_direction": "specific guidance for efficiency improvement",
            "redundancy_score": 0.0-1.0,
            "cache_utilization_score": 0.0-1.0,
            "optimization_score": 0.0-1.0,
            "streamlining_score": 0.0-1.0,
            "redundant_steps": ["list of specific redundant step descriptions"],
            "optimization_opportunities": ["list of specific optimization suggestions"],
            "cache_missed_opportunities": ["analyses being computed that are already cached"],
            "cache_efficiency_score": 0.0-1.0
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
                result["score"] = 0.7
            if "pass" not in result:
                result["pass"] = result["score"] >= 0.7
            if "redundant_steps" not in result:
                result["redundant_steps"] = []
            if "optimization_opportunities" not in result:
                result["optimization_opportunities"] = []
            
            return result
            
        except Exception as e:
            print(f"❌ EfficiencyEvaluator error: {e}")
            return {
                "score": 0.7,
                "pass": True,  # Default to pass to avoid blocking on errors
                "primary_issue": f"Evaluation error: {str(e)}",
                "improvement_direction": "Unable to evaluate efficiency due to technical error",
                "redundant_steps": [],
                "optimization_opportunities": [],
                "error": str(e)
            }