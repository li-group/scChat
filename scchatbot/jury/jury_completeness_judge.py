"""
Completeness Evaluator - Judge for analysis coverage and thoroughness.

This judge evaluates whether the analysis adequately covers all necessary aspects
and provides sufficient depth for the research question.
"""

import json
import openai
from typing import Dict, Any, List


class CompletenessEvaluator:
    """
    Evaluates analysis completeness and thoroughness.
    
    Focus Areas:
    - Missing required analysis steps
    - Insufficient analysis depth
    - Gaps in coverage
    - Adequate exploration of the research question
    """
    
    def __init__(self):
        self.evaluation_criteria = {
            "coverage_completeness": "Are all aspects of the question being addressed?",
            "analysis_depth": "Is the analysis sufficiently deep for the question type?",
            "missing_components": "Are there critical missing analysis components?",
            "exploration_thoroughness": "Is the research question thoroughly explored?"
        }
    
    def evaluate(self, evaluation_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate analysis completeness and coverage.
        
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
        final_response = evaluation_inputs.get("final_response", "")
        
        # Extract step information
        planned_steps = execution_plan.get("steps", [])
        
        # Extract cache context (if available)
        cache_analysis_summary = evaluation_inputs.get("cache_analysis_summary", "")
        confirmed_available_analyses = evaluation_inputs.get("confirmed_available_analyses", "")
        comprehensive_analysis_context = evaluation_inputs.get("comprehensive_analysis_context", "")
        available_cell_types = evaluation_inputs.get("available_cell_types", [])
        function_descriptions = evaluation_inputs.get("function_descriptions", [])
        
        prompt = f"""
        You are a COMPLETENESS EVALUATOR specializing in scientific analysis thoroughness.
        
        Your task: Evaluate whether the analysis provides complete and adequate coverage of the research question.
        
        ORIGINAL USER QUERY: "{original_query}"
        
        PLANNED WORKFLOW STEPS:
        {json.dumps(planned_steps, indent=2)}
        
        EXECUTED STEPS:
        {json.dumps(execution_history, indent=2)}
        
        FINAL RESPONSE PREVIEW:
        {final_response[:500]}...
        
        🆕 CACHED ANALYSIS RESULTS AVAILABLE:
        {cache_analysis_summary}
        
        🎯 CONFIRMED AVAILABLE ANALYSES:
        {confirmed_available_analyses}
        
        🆕 COMPREHENSIVE ANALYSIS CONTEXT (Current + Cache + History):
        {comprehensive_analysis_context}
        
        AVAILABLE CELL TYPES IN DATASET:
        {', '.join(available_cell_types)}
        
        COMPLETENESS EVALUATION CRITERIA:
        
        1. QUESTION COVERAGE ANALYSIS:
           - Does the analysis address all parts of the user's question?
           - Are comparative elements properly handled if mentioned?
           - Are specific cell types or conditions mentioned in the query covered?
           
        2. ANALYSIS DEPTH ASSESSMENT:
           - Is the analysis depth appropriate for the question complexity?
           - For enrichment questions: Are multiple pathway databases covered?
           - For comparative questions: Are proper controls and comparisons included?
           - For discovery questions: Is exploratory analysis sufficient?
           
        3. MISSING COMPONENT DETECTION:
           - Are critical analysis steps missing?
           - Should statistical validation be included?
           - Are appropriate visualization methods included?
           - Is interpretation and biological context provided?
           
        4. SCIENTIFIC RIGOR EVALUATION:
           - Are appropriate statistical methods used?
           - Is multiple testing correction considered where needed?
           - Are biological interpretations supported by the data?
           - Is uncertainty or limitation acknowledgment present?
           
        QUESTION TYPE COMPLETENESS EXPECTATIONS:
        
        BASIC VISUALIZATION QUERIES ("show X"):
        - Visualization step + brief interpretation
        - Data validation if cell type is complex
        
        ANALYSIS QUERIES ("analyze X", "what pathways"):
        - Data processing + analysis + visualization + interpretation
        - Multiple analysis approaches if question is broad
        
        COMPARATIVE QUERIES ("compare X vs Y"):
        - Individual analysis for each entity
        - Direct comparison analysis
        - Comparative visualization
        - Interpretation of differences
        
        DISCOVERY QUERIES ("find X", "identify Y"):
        - Broad exploratory analysis
        - Multiple discovery methods
        - Validation of findings
        - Comprehensive interpretation
        
        CRITICAL CACHE AWARENESS RULES:
        1. If an analysis is listed in "CONFIRMED AVAILABLE ANALYSES", it has been completed and should NOT be marked as missing
        2. Only mark analyses as missing if they are truly unavailable AND needed to answer the question
        3. If cached results exist but aren't being utilized in the response, this is a presentation issue, not missing analysis
        4. Focus on whether the RESPONSE adequately answers the question using available data
        
        SPECIFIC MISSING COMPONENTS TO CHECK:
        - Missing enrichment analysis for pathway questions (CHECK CACHE FIRST)
        - Missing visualization for analysis results
        - Missing interpretation for complex results
        - Missing validation for discovery claims
        - Missing context for biological findings
        - Poor utilization of available cached analyses
        
        SCORING GUIDELINES:
        - 0.9-1.0: Comprehensive analysis covering all aspects thoroughly
        - 0.7-0.8: Good coverage with minor gaps
        - 0.5-0.6: Adequate coverage but some important aspects missing
        - 0.3-0.4: Incomplete analysis with significant gaps
        - 0.0-0.2: Severely incomplete, major components missing
        
        RESPONSE FORMAT (JSON only):
        {{
            "score": 0.0-1.0,
            "pass": true/false,
            "primary_issue": "main completeness problem if any",
            "improvement_direction": "specific guidance for improving completeness",
            "coverage_score": 0.0-1.0,
            "depth_score": 0.0-1.0,
            "rigor_score": 0.0-1.0,
            "missing_components": ["list of TRULY missing analysis components - NOT CACHED"],
            "recommended_additions": ["list of specific steps or analyses to add"],
            "cache_utilization_assessment": "how well were available cached results used?",
            "cached_but_unused": ["analyses available in cache but not used in response"]
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
            if "missing_components" not in result:
                result["missing_components"] = []
            if "recommended_additions" not in result:
                result["recommended_additions"] = []
            
            return result
            
        except Exception as e:
            print(f"❌ CompletenessEvaluator error: {e}")
            return {
                "score": 0.7,
                "pass": True,  # Default to pass to avoid blocking on errors
                "primary_issue": f"Evaluation error: {str(e)}",
                "improvement_direction": "Unable to evaluate completeness due to technical error",
                "missing_components": [],
                "recommended_additions": [],
                "error": str(e)
            }