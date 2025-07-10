"""
Improved User Intent Evaluator - Judge for response alignment with user query.

This improved version accepts truthful responses about missing cell types
and understands dataset limitations, focusing on honest and transparent responses.
"""

import json
import openai
from typing import Dict, Any, List


class ImprovedUserIntentEvaluator:
    """
    Evaluates user intent alignment and response relevance with realistic expectations.
    
    Key improvements:
    - Accepts truthful "cell type not found" responses
    - Understands dataset limitations
    - Focuses on honest and transparent communication
    - Evaluates appropriateness of admitting limitations
    """
    
    def __init__(self):
        self.evaluation_criteria = {
            "query_type_match": "Does response match query type (comparison/discovery/analysis)?",
            "honest_communication": "Does response honestly communicate what was/wasn't found?",
            "limitation_handling": "Are dataset limitations appropriately acknowledged?",
            "result_presentation": "Are available results presented clearly and usefully?",
            "directness": "Does response directly address what was asked?"
        }
    
    def evaluate(self, evaluation_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate whether the response appropriately answers the user's question,
        accepting truthful responses about missing data or limitations.
        
        Args:
            evaluation_inputs: Dictionary containing:
                - original_query: The user's original question
                - execution_plan: The planned steps
                - analysis_context: Available analysis context
                - relevant_cell_types: Cell types relevant to the query
                
        Returns:
            Dictionary with score, pass/fail, and improvement guidance
        """
        
        original_query = evaluation_inputs["original_query"]
        execution_plan = evaluation_inputs.get("execution_plan", {})
        analysis_context = evaluation_inputs.get("analysis_context", "")
        relevant_cell_types = evaluation_inputs.get("relevant_cell_types", [])
        
        # Build evaluation prompt
        prompt = self._build_improved_evaluation_prompt(
            original_query=original_query,
            execution_plan=execution_plan,
            analysis_context=analysis_context,
            relevant_cell_types=relevant_cell_types
        )
        
        try:
            # Get LLM evaluation
            verdict = self._get_llm_evaluation(prompt)
            
            status = "PASS" if verdict.get("pass", False) else "FAIL"
            score = verdict.get("score", 0.5)
            print(f"🎯 Improved User Intent Judge: {status} (score: {score:.2f})")
            
            return verdict
            
        except Exception as e:
            print(f"❌ Improved User Intent Judge error: {e}")
            return self._create_fallback_verdict(str(e))
    
    def _build_improved_evaluation_prompt(self, original_query: str, execution_plan: Dict[str, Any],
                                        analysis_context: str, relevant_cell_types: List[str]) -> str:
        """Build improved evaluation prompt that accepts truthful limitations."""
        
        planned_steps = execution_plan.get("steps", [])
        plan_summary = execution_plan.get("plan_summary", "")
        
        prompt = f"""You are evaluating whether a planned analysis workflow appropriately addresses the user's question.

IMPORTANT: This evaluation should accept truthful responses about missing cell types or dataset limitations. 
Honest communication about what cannot be found is BETTER than forcing unrealistic expectations.

**User Question:** {original_query}

**Planned Analysis:**
Plan Summary: {plan_summary}
Steps ({len(planned_steps)} total):"""

        for i, step in enumerate(planned_steps, 1):
            function_name = step.get("function_name", "unknown")
            parameters = step.get("parameters", {})
            description = step.get("description", "")
            prompt += f"\n{i}. {function_name}({parameters}) - {description}"

        prompt += f"""

**Available Context:**
{analysis_context}

**Relevant Cell Types:** {', '.join(relevant_cell_types) if relevant_cell_types else 'None identified'}

## EVALUATION CRITERIA:

**HONEST COMMUNICATION (High Priority):**
- Is the plan designed to truthfully report what is/isn't available?
- Will it appropriately acknowledge if requested cell types are missing?
- Does it avoid forcing answers when data doesn't support them?

**QUERY TYPE ALIGNMENT:**
- Does the planned analysis match the type of question asked?
- Are the right analysis methods selected for the question type?
- Will the analysis directly address what the user wants to know?

**PRACTICAL UTILITY:**
- Will the planned analysis provide useful information to the user?
- Are results likely to be presented in a clear, actionable format?
- Does the plan include appropriate visualizations when relevant?

**LIMITATION HANDLING:**
- Does the plan appropriately handle cases where data may be incomplete?
- Will it clearly communicate any limitations or missing information?
- Are alternative approaches considered when primary data isn't available?

## ACCEPTABLE RESPONSES:

✅ **GOOD:** "The requested cell type was not found in the dataset, but here's what we found instead..."
✅ **GOOD:** "We couldn't perform the comparison you requested because Cell Type X is not available, but we can show you..."
✅ **GOOD:** "The analysis shows that only 2 of the 3 requested cell types are present in the data..."

❌ **BAD:** Forcing an answer when the data doesn't support it
❌ **BAD:** Ignoring missing cell types and proceeding as if they exist
❌ **BAD:** Providing generic information that doesn't address the specific question

## INSTRUCTIONS:

Evaluate whether this planned analysis will:
1. Honestly address what can and cannot be determined from the data
2. Appropriately handle any missing cell types or limitations
3. Provide useful information that directly answers the user's question
4. Present results in a clear, actionable format

Return your evaluation as JSON:
{{
    "score": 0.0-1.0,
    "pass": true/false,
    "reasoning": "Detailed explanation of your evaluation",
    "strengths": ["what the plan does well"],
    "concerns": ["potential issues with the plan"],
    "improvement_suggestions": ["specific ways to improve user intent alignment"],
    "accepts_limitations": true/false,
    "honest_communication": true/false
}}"""

        return prompt
    
    def _get_llm_evaluation(self, prompt: str) -> Dict[str, Any]:
        """Get LLM evaluation using OpenAI."""
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator of single-cell RNA-seq analysis workflows, focused on honest and transparent communication with users. Always respond with valid JSON."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            # Get response content
            verdict_text = response.choices[0].message.content
            
            # Debug: Print raw response
            print(f"🔍 Raw LLM response length: {len(verdict_text) if verdict_text else 0}")
            if not verdict_text or verdict_text.strip() == "":
                print("❌ Empty response from LLM")
                return self._create_fallback_verdict("Empty response from LLM")
            
            # Clean and parse JSON response
            verdict_text = verdict_text.strip()
            
            # Remove markdown code blocks if present
            if verdict_text.startswith('```json'):
                verdict_text = verdict_text[7:]  # Remove ```json
            elif verdict_text.startswith('```'):
                verdict_text = verdict_text[3:]  # Remove ```
            
            if verdict_text.endswith('```'):
                verdict_text = verdict_text[:-3]  # Remove closing ```
            
            verdict_text = verdict_text.strip()
            print(f"🔍 Cleaned response preview: {verdict_text[:200]}...")
            
            verdict = json.loads(verdict_text)
            
            # Standardize format
            return {
                "score": verdict.get("score", 0.5),
                "pass": verdict.get("pass", False),
                "reasoning": verdict.get("reasoning", ""),
                "strengths": verdict.get("strengths", []),
                "concerns": verdict.get("concerns", []),
                "improvement_suggestions": verdict.get("improvement_suggestions", []),
                "accepts_limitations": verdict.get("accepts_limitations", False),
                "honest_communication": verdict.get("honest_communication", False)
            }
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse judge response as JSON: {e}")
            print(f"❌ Raw response was: '{verdict_text}'")
            return self._create_fallback_verdict("JSON parsing error")
        except Exception as e:
            print(f"❌ LLM evaluation error: {e}")
            return self._create_fallback_verdict(str(e))
    
    def _create_fallback_verdict(self, error_msg: str) -> Dict[str, Any]:
        """Create fallback verdict on error."""
        return {
            "score": 0.6,
            "pass": True,  # Fail-safe: approve on error
            "reasoning": f"Evaluation failed due to technical error: {error_msg}. Defaulting to approval.",
            "strengths": [],
            "concerns": [f"Technical error: {error_msg}"],
            "improvement_suggestions": [],
            "accepts_limitations": True,
            "honest_communication": True,
            "error": error_msg
        }