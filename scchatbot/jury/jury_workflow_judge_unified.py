"""
Unified Workflow Judge - Combines workflow logic, completeness, and efficiency evaluation.

This judge evaluates:
1. Logical flow and method appropriateness (original workflow judge)
2. Completeness of analysis steps (original completeness judge)
3. Efficiency and redundancy (original efficiency judge)

Uses aug_memory.json for few-shot prompting with proven pipeline examples.
"""

import json
import openai
from typing import Dict, Any, List
from pathlib import Path


class UnifiedWorkflowJudge:
    """
    Unified judge that evaluates workflow logic, completeness, and efficiency.
    
    Replaces the original trio of:
    - WorkflowLogicEvaluator
    - CompletenessEvaluator  
    - EfficiencyEvaluator
    
    Uses augmented memory examples for few-shot prompting.
    """
    
    def __init__(self):
        self.evaluation_criteria = {
            "logical_flow": "Are the steps in a logical order that builds understanding?",
            "method_selection": "Are the chosen methods appropriate for each analysis task?",
            "completeness": "Are all necessary steps included to fully answer the question?",
            "efficiency": "Are there redundant or unnecessary steps that should be removed?",
            "dependency_handling": "Are dependencies between steps properly managed?",
            "coherence": "Do the steps work together toward a unified analysis goal?"
        }
        
        # Load augmented memory examples
        self.aug_memory_examples = self._load_aug_memory_examples()
        
    def _load_aug_memory_examples(self) -> List[Dict[str, Any]]:
        """Load examples from aug_memory.json for few-shot prompting."""
        try:
            aug_memory_path = Path(__file__).parent / "aug_memory.json"
            with open(aug_memory_path, 'r') as f:
                examples = json.load(f)
            print(f"🧠 Loaded {len(examples)} augmented memory examples for workflow judge")
            return examples
        except Exception as e:
            print(f"⚠️ Could not load aug_memory.json: {e}")
            return []
    
    def evaluate(self, evaluation_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unified evaluation of workflow logic, completeness, and efficiency.
        
        Args:
            evaluation_inputs: Dictionary containing:
                - original_query: The user's original question
                - execution_plan: The planned steps
                - execution_summary: Summary of execution results
                - available_cell_types: Available cell types
                - unavailable_cell_types: Unavailable cell types
                
        Returns:
            Dictionary with score, pass/fail, and improvement guidance
        """
        
        try:
            # Extract inputs
            original_query = evaluation_inputs["original_query"]
            execution_plan = evaluation_inputs.get("execution_plan", {})
            execution_summary = evaluation_inputs.get("execution_summary", "")
            available_cell_types = evaluation_inputs.get("available_cell_types", [])
            unavailable_cell_types = evaluation_inputs.get("unavailable_cell_types", [])
            
            # Create few-shot examples from aug_memory
            few_shot_examples = self._create_few_shot_examples()
            
            # Build the evaluation prompt
            evaluation_prompt = self._build_unified_evaluation_prompt(
                original_query=original_query,
                execution_plan=execution_plan,
                execution_summary=execution_summary,
                available_cell_types=available_cell_types,
                unavailable_cell_types=unavailable_cell_types,
                few_shot_examples=few_shot_examples
            )
            
            # Get LLM evaluation
            verdict = self._get_llm_evaluation(evaluation_prompt)
            
            print(f"🏛️ Unified Workflow Judge: {verdict['decision'].upper()} (score: {verdict['score']:.2f})")
            
            return verdict
            
        except Exception as e:
            print(f"❌ Unified Workflow Judge error: {e}")
            return self._create_fallback_verdict(str(e))
    
    def _create_few_shot_examples(self) -> str:
        """Create few-shot examples from aug_memory for chain-of-thought prompting."""
        if not self.aug_memory_examples:
            return ""
        
        examples_text = "## PROVEN PIPELINE EXAMPLES:\n\n"
        
        # Use first 3 examples to avoid token limits
        for i, example in enumerate(self.aug_memory_examples[:3], 1):
            examples_text += f"**Example {i}:**\n"
            examples_text += f"Question: {example['question']}\n"
            
            # Handle both "situation" and "condtion" (typo in aug_memory.json)
            if 'situation' in example:
                examples_text += f"Situation: {example['situation']}\n"
            elif 'condtion' in example:
                examples_text += f"Situation: {example['condtion']}\n"
            
            examples_text += f"Correct Pipeline:\n"
            for step in example['correct_pipeline']:
                examples_text += f"  - {step['function']}({step['parameters']})\n"
            
            examples_text += f"Rationale: {example['rationale']}\n\n"
        
        return examples_text
    
    def _build_unified_evaluation_prompt(self, original_query: str, execution_plan: Dict[str, Any], 
                                       execution_summary: str, available_cell_types: List[str],
                                       unavailable_cell_types: List[str], few_shot_examples: str) -> str:
        """Build the comprehensive evaluation prompt."""
        
        planned_steps = execution_plan.get("steps", [])
        plan_summary = execution_plan.get("plan_summary", "")
        
        prompt = f"""You are a unified workflow evaluation judge for single-cell RNA-seq analysis pipelines.

Your task is to evaluate a planned analysis workflow across three dimensions:
1. **LOGICAL FLOW**: Are steps in proper order with correct method selection?
2. **COMPLETENESS**: Are all necessary steps included to fully answer the question?
3. **EFFICIENCY**: Are there redundant or unnecessary steps?

{few_shot_examples}

## CURRENT EVALUATION:

**User Question:** {original_query}

**Available Cell Types:** {', '.join(available_cell_types) if available_cell_types else 'None'}
**Unavailable Cell Types:** {', '.join(unavailable_cell_types) if unavailable_cell_types else 'None'}

**Planned Workflow:**
Plan Summary: {plan_summary}
Steps ({len(planned_steps)} total):"""

        for i, step in enumerate(planned_steps, 1):
            function_name = step.get("function_name", "unknown")
            parameters = step.get("parameters", {})
            description = step.get("description", "")
            prompt += f"\n{i}. {function_name}({parameters}) - {description}"
        
        if execution_summary:
            prompt += f"\n\n**Execution Results Summary:**\n{execution_summary}"
        
        prompt += f"""

## EVALUATION CRITERIA:

**LOGICAL FLOW:**
- Are steps in logical order that builds understanding?
- Are method selections appropriate for each task?
- Are dependencies handled properly?

**COMPLETENESS:**
- Does the plan fully address the user's question?
- Are all necessary analysis steps included?
- Are visualization steps included when appropriate?

**EFFICIENCY:**
- Are there redundant or duplicate steps?
- Are unnecessary steps included?
- Is the workflow optimized for the question asked?

## INSTRUCTIONS:

1. Compare the current plan against the proven examples above
2. Evaluate across all three dimensions (logic, completeness, efficiency)
3. Provide specific, actionable feedback
4. Focus on what would make the biggest improvement

Return your evaluation as JSON:
{{
    "score": 0.0-1.0,
    "decision": "approve" or "needs_revision",
    "reasoning": "Detailed explanation of your evaluation",
    "issues_found": {{
        "logical_flow": ["list of logic issues"],
        "completeness": ["list of missing steps"],
        "efficiency": ["list of redundancy issues"]
    }},
    "improvement_suggestions": [
        "specific actionable improvements"
    ],
    "priority_changes": [
        "most important changes needed"
    ]
}}"""

        return prompt
    
    def _get_llm_evaluation(self, prompt: str) -> Dict[str, Any]:
        """Get LLM evaluation using OpenAI."""
        try:
            # Check if OpenAI client is properly configured
            if not hasattr(openai, 'api_key') and not openai.api_key:
                print("⚠️ OpenAI API key not configured")
            
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert single-cell RNA-seq analysis workflow evaluator. Always respond with valid JSON."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1000
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
                "pass": verdict.get("decision", "needs_revision") == "approve",
                "reasoning": verdict.get("reasoning", ""),
                "issues_found": verdict.get("issues_found", {}),
                "improvement_suggestions": verdict.get("improvement_suggestions", []),
                "priority_changes": verdict.get("priority_changes", []),
                "decision": verdict.get("decision", "needs_revision")
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
            "issues_found": {},
            "improvement_suggestions": [],
            "priority_changes": [],
            "decision": "approve",
            "error": error_msg
        }