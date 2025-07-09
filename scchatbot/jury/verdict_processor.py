"""
Jury verdict processing and decision making logic.

This module contains verdict processing methods extracted from jury_system_main.py:
- _make_jury_decision(): Make overall jury decision based on individual verdicts
- route_from_jury(): Determine routing based on jury decision
- conflict_resolution_node(): Handle presentation-only revisions
"""

from typing import Dict, Any, Literal


class VerdictProcessorMixin:
    """
    Verdict processing mixin for making jury decisions and routing.
    """

    def _make_jury_decision(self, jury_verdicts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make overall jury decision based on individual verdicts.
        
        Args:
            jury_verdicts: Results from all judges
            
        Returns:
            Overall jury decision with reasoning
        """
        
        # Calculate overall statistics
        scores = [verdict.get("score", 0.0) for verdict in jury_verdicts.values()]
        passes = [verdict.get("pass", False) for verdict in jury_verdicts.values()]
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        pass_count = sum(passes)
        total_judges = len(jury_verdicts)
        pass_rate = pass_count / total_judges if total_judges > 0 else 0.0
        
        # Determine overall decision
        if pass_rate >= 0.75:  # At least 3/4 judges pass
            decision = "ACCEPT"
            reasoning = f"Strong consensus: {pass_count}/{total_judges} judges passed"
        elif pass_rate >= 0.5:  # At least half pass
            decision = "REVISE_MINOR"
            reasoning = f"Mixed verdict: {pass_count}/{total_judges} judges passed"
        else:
            decision = "REVISE_MAJOR"
            reasoning = f"Poor performance: only {pass_count}/{total_judges} judges passed"
        
        # Special handling for user intent judge
        user_intent_pass = jury_verdicts.get("user_intent_judge", {}).get("pass", True)
        if not user_intent_pass and pass_count >= 2:
            # Technical aspects pass but user intent fails
            decision = "REVISE_PRESENTATION"
            reasoning = "Technical analysis acceptable but response doesn't answer user's question"
        
        return {
            "decision": decision,
            "reasoning": reasoning,
            "avg_score": avg_score,
            "pass_rate": pass_rate,
            "total_judges": total_judges,
            "passing_judges": pass_count
        }
    
    def route_from_jury(self, state: Dict[str, Any]) -> Literal["accept", "revise_analysis", "revise_presentation"]:
        """
        Determine routing based on jury decision.
        
        Args:
            state: Current chat state with jury verdicts
            
        Returns:
            Routing decision for the workflow
        """
        
        jury_decision = state.get("jury_decision", {})
        decision = jury_decision.get("decision", "REVISE_MAJOR")
        
        if decision == "ACCEPT":
            print("🏤 Jury accepts - proceeding to final response")
            return "accept"
        
        elif decision == "REVISE_PRESENTATION":
            print("🎯 User intent issue - revising presentation only")
            return "revise_presentation"
        
        else:  # REVISE_MINOR or REVISE_MAJOR
            print("🔬 Technical issues - revising analysis plan")
            return "revise_analysis"
    
    def conflict_resolution_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle presentation-only revisions using conflict resolution.
        
        Args:
            state: Current chat state
            
        Returns:
            Updated state with improved response
        """
        
        jury_verdicts = state.get("jury_verdicts", {})
        
        # Get user intent guidance
        user_intent_verdict = jury_verdicts.get("user_intent_judge", {})
        intent_guidance = user_intent_verdict.get("improvement_direction", "Improve response focus")
        
        print(f"🎯 Applying presentation fix: {intent_guidance}")
        
        try:
            # Get the enhanced user intent guidance (not just improvement_direction)
            user_intent_guidance = user_intent_verdict.get("response_guidance", {})
            if not user_intent_guidance:
                # Fallback to basic guidance
                user_intent_guidance = {
                    "answer_format": "comparison" if "compar" in intent_guidance.lower() else "direct_answer",
                    "required_elements": ["distinguishing features", "specific markers", "functional differences"] if "compar" in intent_guidance.lower() else ["main findings"],
                    "key_focus_areas": ["cell type differences"] if "compar" in intent_guidance.lower() else ["analysis results"],
                    "answer_template": "X is distinguished from Y by:" if "compar" in intent_guidance.lower() else ""
                }
            
            # Apply the guidance to state for response generator
            state["user_intent_guidance"] = user_intent_guidance
            state["conflict_resolution_applied"] = True
            
            print(f"🎯 Applied user intent guidance: {user_intent_guidance.get('answer_format', 'direct_answer')}")
            
            # Mark as completed so it goes to response generator
            state["conversation_complete"] = True
            
            return state
            
        except Exception as e:
            print(f"❌ Error in conflict resolution: {e}")
            # Fallback: just proceed with basic guidance
            state["user_intent_guidance"] = {
                "answer_format": "direct_answer",
                "required_elements": ["main findings"],
                "key_focus_areas": ["analysis results"],
                "answer_template": ""
            }
            state["conflict_resolution_applied"] = True
            state["conversation_complete"] = True
            return state
