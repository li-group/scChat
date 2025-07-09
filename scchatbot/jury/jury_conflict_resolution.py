"""
Conflict Resolution Engine for the jury-based critic system.

This module handles conflicts between different judges and determines
appropriate resolution strategies, particularly for presentation vs analysis issues.
"""

import json
import openai
from typing import Dict, Any, List


class ConflictResolutionEngine:
    """
    Resolves conflicts between jury members and determines revision strategies.
    
    Key scenarios:
    - User Intent fails, Technical judges pass: Presentation issue
    - Technical judges fail, User Intent passes: Analysis issue  
    - Mixed failures: Complex revision needed
    """
    
    def __init__(self):
        self.resolution_strategies = {
            "presentation_only": self._handle_presentation_issue,
            "analysis_revision": self._handle_analysis_issue,
            "mixed_revision": self._handle_mixed_issues,
            "accept_with_warnings": self._handle_minor_issues
        }
    
    def resolve_conflicts(self, jury_verdicts: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze jury verdicts and determine appropriate resolution strategy.
        
        Args:
            jury_verdicts: Results from all jury members
            state: Current chat state
            
        Returns:
            Resolution strategy with specific actions
        """
        
        # Analyze the pattern of passes and failures
        technical_judges = ["workflow_judge", "efficiency_judge", "completeness_judge"]
        user_intent_judge = "user_intent_judge"
        
        technical_results = {judge: jury_verdicts.get(judge, {}).get("pass", True) 
                           for judge in technical_judges}
        user_intent_result = jury_verdicts.get(user_intent_judge, {}).get("pass", True)
        
        failed_judges = [name for name, verdict in jury_verdicts.items() 
                        if not verdict.get("pass", True)]
        
        print(f"🏛️ Jury Conflict Analysis:")
        print(f"   Technical judges: {technical_results}")
        print(f"   User intent: {user_intent_result}")
        print(f"   Failed judges: {failed_judges}")
        
        # Determine resolution strategy
        if len(failed_judges) == 0:
            return {"strategy": "accept", "action": "proceed_to_response"}
        
        elif failed_judges == [user_intent_judge]:
            # Only user intent failed - presentation issue
            print("🎯 Detected: Presentation issue (analysis correct, response wrong)")
            return self._handle_presentation_issue(jury_verdicts, state)
        
        elif user_intent_judge not in failed_judges and len(failed_judges) > 0:
            # Technical issues but user intent is satisfied
            print("🔬 Detected: Technical issues (analysis needs improvement)")
            return self._handle_analysis_issue(jury_verdicts, state)
        
        elif len(failed_judges) > 1:
            # Multiple issues
            print("🔄 Detected: Mixed issues (both analysis and presentation)")
            return self._handle_mixed_issues(jury_verdicts, state)
        
        else:
            # Single technical judge failed
            print("⚠️ Detected: Minor issue (single technical judge)")
            return self._handle_minor_issues(jury_verdicts, state)
    
    def _handle_presentation_issue(self, jury_verdicts: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle cases where analysis is correct but response presentation is wrong.
        """
        user_intent_feedback = jury_verdicts.get("user_intent_judge", {})
        
        return {
            "strategy": "response_reframing",
            "action": "regenerate_response_with_intent_guidance",
            "guidance": user_intent_feedback.get("improvement_direction", "Improve response focus"),
            "preserve_analysis": True,
            "revision_type": "presentation_only",
            "priority": "high"
        }
    
    def _handle_analysis_issue(self, jury_verdicts: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle cases where analysis needs improvement but user intent is satisfied.
        """
        failed_judges = [name for name, verdict in jury_verdicts.items() 
                        if not verdict.get("pass", True)]
        
        primary_issues = []
        for judge in failed_judges:
            issue = jury_verdicts.get(judge, {}).get("primary_issue", "Unknown issue")
            primary_issues.append(f"{judge}: {issue}")
        
        return {
            "strategy": "analysis_revision",
            "action": "revise_execution_plan",
            "failed_aspects": failed_judges,
            "issues": primary_issues,
            "preserve_analysis": False,
            "revision_type": "targeted_analysis",
            "priority": "high"
        }
    
    def _handle_mixed_issues(self, jury_verdicts: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle cases where both analysis and presentation need improvement.
        """
        failed_judges = [name for name, verdict in jury_verdicts.items() 
                        if not verdict.get("pass", True)]
        
        # Prioritize analysis issues first
        technical_failures = [j for j in failed_judges 
                            if j in ["workflow_judge", "efficiency_judge", "completeness_judge"]]
        
        if technical_failures:
            return {
                "strategy": "analysis_revision",
                "action": "revise_execution_plan", 
                "failed_aspects": failed_judges,
                "revision_type": "comprehensive",
                "preserve_analysis": False,
                "priority": "high",
                "follow_up": "presentation_review"  # Review presentation after analysis fix
            }
        else:
            return self._handle_presentation_issue(jury_verdicts, state)
    
    def _handle_minor_issues(self, jury_verdicts: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle cases with minor single-judge failures.
        """
        failed_judges = [name for name, verdict in jury_verdicts.items() 
                        if not verdict.get("pass", True)]
        
        if len(failed_judges) == 1:
            failed_judge = failed_judges[0]
            issue = jury_verdicts.get(failed_judge, {}).get("primary_issue", "Minor issue")
            
            # For efficiency issues, might still proceed if score is close
            if failed_judge == "efficiency_judge":
                score = jury_verdicts.get(failed_judge, {}).get("score", 0.0)
                if score >= 0.6:  # Close to passing threshold
                    return {
                        "strategy": "accept_with_warnings",
                        "action": "proceed_to_response",
                        "warnings": [f"Efficiency concern: {issue}"],
                        "revision_type": "none"
                    }
            
            # For other single failures, do targeted revision
            return {
                "strategy": "minor_revision",
                "action": "targeted_fix",
                "failed_aspect": failed_judge,
                "issue": issue,
                "revision_type": "targeted",
                "priority": "medium"
            }
        
        return {"strategy": "accept", "action": "proceed_to_response"}
    
    def regenerate_response_with_intent_guidance(self, state: Dict[str, Any], intent_guidance: str) -> str:
        """
        Regenerate final response using user intent judge guidance
        without redoing the analysis.
        """
        
        original_query = state.get("execution_plan", {}).get("original_question", "")
        execution_history = state.get("execution_history", [])
        
        # Extract analysis results from execution history
        analysis_results = self._extract_analysis_results(execution_history)
        
        reframing_prompt = f"""
        You need to REFRAME the response to better answer the user's question.
        
        Original User Query: "{original_query}"
        
        Available Analysis Results: {json.dumps(analysis_results, indent=2)}
        
        User Intent Judge Feedback: "{intent_guidance}"
        
        INSTRUCTIONS:
        1. Use the SAME analysis results (don't request new analyses)
        2. Reframe the response to directly address what the user asked
        3. Follow the improvement direction from the user intent judge
        4. Maintain scientific accuracy while improving user focus
        5. Present information in the most useful format for this specific question
        
        RESPONSE REQUIREMENTS:
        - Be direct and focused on the user's specific question
        - Highlight the most relevant findings prominently
        - Minimize background information unless specifically needed
        - Use appropriate technical level for the apparent user expertise
        - Include any visualizations or data that were generated
        
        Generate a response that better answers: "{original_query}"
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": reframing_prompt}],
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ Error regenerating response: {e}")
            return f"Error improving response presentation: {str(e)}"
    
    def _extract_analysis_results(self, execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract key results from execution history for response reframing.
        """
        
        results = {
            "completed_analyses": [],
            "visualizations": [],
            "key_findings": [],
            "errors": []
        }
        
        for execution in execution_history:
            if execution.get("success", False):
                function_name = execution.get("function_name", "")
                result = execution.get("result", "")
                
                if "enrichment" in function_name:
                    results["completed_analyses"].append({
                        "type": "enrichment",
                        "function": function_name,
                        "summary": str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                    })
                elif "display" in function_name or "visualization" in function_name:
                    results["visualizations"].append({
                        "type": "visualization",
                        "function": function_name,
                        "result": result
                    })
                else:
                    results["completed_analyses"].append({
                        "type": "analysis",
                        "function": function_name,
                        "summary": str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                    })
            else:
                results["errors"].append({
                    "function": execution.get("function_name", ""),
                    "error": execution.get("error", "Unknown error")
                })
        
        return results