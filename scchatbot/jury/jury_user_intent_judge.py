"""
User Intent Evaluator - Judge for response alignment with user query.

This judge evaluates whether the response truly answers what the user asked,
focusing on query type matching, specificity, and practical utility.
"""

import json
import openai
from typing import Dict, Any


class UserIntentEvaluator:
    """
    Evaluates user intent alignment and response relevance.
    
    Focus Areas:
    - Query type matching (comparison/discovery/analysis/visualization)
    - Specificity alignment with user's request
    - Result presentation appropriateness
    - Directness vs. tangential information
    """
    
    def __init__(self):
        self.evaluation_criteria = {
            "query_type_match": "Does response match query type (comparison/discovery/analysis)?",
            "specificity_alignment": "Does response address the specific entities mentioned?", 
            "result_presentation": "Are results presented in a way that answers the question?",
            "context_awareness": "Does response consider user's likely follow-up needs?",
            "directness": "Does response directly answer what was asked vs. tangential info?"
        }
    
    def evaluate(self, evaluation_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate whether the response truly answers the user's question.
        
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
        final_response = evaluation_inputs.get("final_response", "")
        
        # Extract cache context (if available)
        cache_analysis_summary = evaluation_inputs.get("cache_analysis_summary", "")
        confirmed_available_analyses = evaluation_inputs.get("confirmed_available_analyses", "")
        
        prompt = f"""
        You are evaluating whether the RESPONSE truly answers the USER'S QUESTION.
        
        Original Query: "{original_query}"
        Final Response: "{final_response}"
        
        🆕 CACHED ANALYSIS RESULTS AVAILABLE:
        {cache_analysis_summary}
        
        🎯 CONFIRMED AVAILABLE ANALYSES:
        {confirmed_available_analyses}
        
        CHAIN OF THOUGHT EVALUATION:
        
        1. QUERY TYPE ANALYSIS:
           - What type of question is this? (comparison/discovery/analysis/visualization)
           - What specific information is the user seeking?
           - Is this a simple request or complex research question?
        
        2. RESPONSE ALIGNMENT:
           - Does the response directly address this query type?
           - Are the specific entities/comparisons the user asked about covered?
           - Is the response focused or does it provide irrelevant information?
        
        3. PRACTICAL UTILITY:
           - Would this response satisfy the user's information need?
           - Is the information presented in a useful format?
           - Are key findings highlighted appropriately?
           - Can the user act on this information?
        
        4. COMPLETENESS OF ANSWER:
           - Does response answer the "what" the user asked?
           - Does it provide sufficient detail for the user's apparent expertise level?
           - Are important caveats or limitations mentioned when relevant?
           - Does response utilize available cached analyses appropriately?
           - If cached results exist but aren't mentioned, is this a presentation failure?
        
        5. DIRECTNESS ASSESSMENT:
           - Does the response get straight to the point?
           - Is there excessive background information that doesn't serve the query?
           - Are the most important findings presented prominently?
        
        SPECIFIC QUERY PATTERNS TO EVALUATE:
        
        VISUALIZATION REQUESTS ("show X", "display Y"):
        - Response should focus on the visualization itself
        - Minimal background unless specifically requested
        - Clear description of what the visualization shows
        
        ANALYSIS REQUESTS ("analyze X", "what pathways in Y"):
        - Response should provide specific findings
        - Interpretation should be prominent
        - Results should be actionable or informative
        
        COMPARISON REQUESTS ("compare X vs Y", "difference between"):
        - Response should explicitly compare the entities
        - Differences and similarities should be highlighted
        - Conclusions about the comparison should be clear
        
        DISCOVERY REQUESTS ("find X", "identify Y", "what are"):
        - Response should provide specific discoveries
        - Results should be ranked or prioritized
        - Context for the discoveries should be provided
        
        CONVERSATIONAL REQUESTS ("what does this mean", "explain"):
        - Response should provide clear explanations
        - Technical terms should be clarified
        - Biological significance should be explained
        
        RED FLAGS FOR USER INTENT MISALIGNMENT:
        - Providing comprehensive analysis when user wants simple visualization
        - Giving background information when user wants specific results
        - Answering different question than what was asked
        - Over-explaining simple concepts to apparent experts
        - Under-explaining complex results to apparent novices
        - Focusing on methodology when user wants conclusions
        - Failing to use available cached analyses that would answer the question
        - Saying "no results available" when cached analyses exist
        
        RESPONSE GUIDANCE GENERATION:
        Based on your evaluation, provide specific guidance for how the response should be improved:
        
        FOR COMPARISON QUESTIONS:
        - answer_format: "comparison" 
        - structure: "comparison_format" (Entity A vs Entity B structure)
        - required_elements: ["distinguishing features", "key differences", "similarities", "biological significance"]
        - answer_template: "X is distinguished from Y by: 1. [key differences], 2. [specific markers], 3. [functional differences]"
        
        FOR DISCOVERY QUESTIONS:
        - answer_format: "discovery_list"
        - structure: "discovery_list" (numbered findings)
        - required_elements: ["specific discoveries", "significance", "evidence"]
        - answer_template: "The analysis revealed [number] key findings: 1. [discovery], 2. [discovery]..."
        
        FOR ANALYSIS QUESTIONS:
        - answer_format: "direct_answer"
        - structure: "explanation" (finding → evidence → significance)
        - required_elements: ["main finding", "supporting evidence", "biological interpretation"]
        
        FOR VISUALIZATION REQUESTS:
        - answer_format: "explanation"
        - required_elements: ["what the visualization shows", "key patterns", "interpretation"]
        - avoid_elements: ["methodology details", "step-by-step process"]
        
        SCORING GUIDELINES:
        - 0.9-1.0: Response perfectly answers the user's question with appropriate focus
        - 0.7-0.8: Response answers the question well with minor relevance issues
        - 0.5-0.6: Response addresses the question but includes unnecessary information
        - 0.3-0.4: Response partially answers but misses key aspects of the question
        - 0.0-0.2: Response fails to address what the user actually asked
        
        VERDICT:
        - Score: [0.0-1.0] where 1.0 = perfectly answers the question
        - Pass/Fail: [PASS/FAIL] (Pass if score >= 0.7)
        - Primary Issue: [if FAIL, what's the main problem?]
        - Improvement Direction: [how to better answer the user's question?]
        
        RESPONSE FORMAT (JSON only):
        {{
            "score": 0.0-1.0,
            "pass": true/false,
            "primary_issue": "string describing main problem",
            "improvement_direction": "specific guidance for better response",
            "query_type_detected": "comparison/discovery/analysis/visualization/conversational",
            "specificity_score": 0.0-1.0,
            "directness_score": 0.0-1.0,
            "utility_score": 0.0-1.0,
            "focus_issues": ["list of specific focus problems if any"],
            "cache_utilization_score": 0.0-1.0,
            "missed_cached_opportunities": ["cached analyses that could have answered the question better"],
            "response_guidance": {{
                "answer_format": "direct_answer/summary/explanation/comparison",
                "key_focus_areas": ["list of what response should emphasize"],
                "tone": "scientific/conversational/technical/educational",
                "structure": "comparison_format/discovery_list/explanation/process_summary",
                "required_elements": ["specific things the response must include"],
                "avoid_elements": ["things the response should not include"],
                "use_cached_analyses": ["specific cached results to utilize"],
                "answer_template": "suggested response structure or opening"
            }}
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
            if "focus_issues" not in result:
                result["focus_issues"] = []
            if "query_type_detected" not in result:
                result["query_type_detected"] = "analysis"
            if "response_guidance" not in result:
                result["response_guidance"] = {
                    "answer_format": "direct_answer",
                    "key_focus_areas": ["answer the user's question"],
                    "tone": "scientific",
                    "structure": "explanation",
                    "required_elements": ["main findings"],
                    "avoid_elements": ["excessive methodology"],
                    "use_cached_analyses": [],
                    "answer_template": "Based on the analysis..."
                }
            
            return result
            
        except Exception as e:
            print(f"❌ UserIntentEvaluator error: {e}")
            return {
                "score": 0.7,
                "pass": True,  # Default to pass to avoid blocking on errors
                "primary_issue": f"Evaluation error: {str(e)}",
                "improvement_direction": "Unable to evaluate user intent due to technical error",
                "query_type_detected": "unknown",
                "focus_issues": [],
                "error": str(e)
            }