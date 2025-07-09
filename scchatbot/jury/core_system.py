"""
Core JurySystem class with initialization and basic structure.

This module contains the main JurySystem class extracted from jury_system_main.py:
- JurySystem.__init__(): Initialize the jury system with infrastructure components
- _initialize_jury_members(): Setup individual jury members
- jury_evaluation_node(): Main entry point for jury evaluation
- _create_fallback_verdict(): Create fallback verdict for errors
"""

import asyncio
import json
from typing import Dict, Any, List, Literal

from .jury_workflow_judge import WorkflowLogicEvaluator
from .jury_efficiency_judge import EfficiencyEvaluator
from .jury_completeness_judge import CompletenessEvaluator
from .jury_user_intent_judge import UserIntentEvaluator
from .jury_conflict_resolution import ConflictResolutionEngine


class CoreSystemMixin:
    """
    Core JurySystem functionality including initialization and main evaluation entry point.
    """
    
    def __init__(self, simple_cache=None, hierarchy_manager=None, history_manager=None, function_descriptions=None, existing_critic_agent=None):
        """
        Initialize the jury system with infrastructure components.
        
        Args:
            simple_cache: Cache manager for analysis results
            hierarchy_manager: Cell type hierarchy manager
            history_manager: Function execution history manager
            function_descriptions: Available function descriptions
            existing_critic_agent: DEPRECATED - for backward compatibility only
        """
        
        # Accept parameters directly or from existing critic agent (backward compatibility)
        if existing_critic_agent:
            self.simple_cache = existing_critic_agent.simple_cache
            self.hierarchy_manager = existing_critic_agent.hierarchy_manager
            self.history_manager = existing_critic_agent.history_manager
            self.function_descriptions = getattr(existing_critic_agent, 'function_descriptions', [])
        else:
            self.simple_cache = simple_cache
            self.hierarchy_manager = hierarchy_manager
            self.history_manager = history_manager
            self.function_descriptions = function_descriptions or []
        
        # Initialize jury members
        self.jury_members = self._initialize_jury_members()
        
        # Initialize conflict resolution engine
        self.conflict_resolver = ConflictResolutionEngine()
        
        # Track jury evaluation counts
        self.jury_evaluation_count = 0
        
        print("\n🏤 Jury System initialized with 4 specialized judges")
        print("   • Workflow Logic Evaluator")
        print("   • Efficiency Evaluator")
        print("   • Completeness Evaluator")
        print("   • User Intent Evaluator")
        print("   • Conflict Resolution Engine")

    def _initialize_jury_members(self) -> Dict[str, Any]:
        """Initialize all jury members with their specialized evaluators"""
        return {
            "workflow_judge": WorkflowLogicEvaluator(),
            "efficiency_judge": EfficiencyEvaluator(),
            "completeness_judge": CompletenessEvaluator(),
            "user_intent_judge": UserIntentEvaluator()
        }

    def jury_evaluation_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main jury evaluation node that coordinates all judges.
        
        This is the primary entry point for jury evaluation, called by the LangGraph workflow.
        It orchestrates the evaluation process and manages the verdict aggregation.
        
        Args:
            state: Current workflow state containing execution plan and history
            
        Returns:
            Updated state with jury verdicts and decision
        """
        print("\n🏤 ==================== JURY EVALUATION START ====================")
        
        # Increment jury evaluation count
        self.jury_evaluation_count += 1
        state["jury_iteration"] = self.jury_evaluation_count
        
        print(f"🏤 Jury Evaluation #{self.jury_evaluation_count}")
        print(f"🏤 Evaluating execution plan with {len(state.get('execution_plan', {}).get('steps', []))} steps")
        
        # Prepare evaluation inputs with cache awareness
        evaluation_inputs = self._prepare_cache_aware_evaluation_inputs(state)
        
        # Run jury evaluation
        jury_verdicts = self._run_jury_evaluation_sync(evaluation_inputs)
        
        # Make final jury decision
        jury_decision = self._make_jury_decision(jury_verdicts)
        
        # Store results in state
        state["jury_verdicts"] = jury_verdicts
        state["jury_decision"] = jury_decision
        
        # Log final decision
        decision_action = jury_decision.get("decision", "unknown")
        confidence = jury_decision.get("confidence", 0.0)
        print(f"🏤 Final Jury Decision: {decision_action.upper()} (confidence: {confidence:.2f})")
        
        # Add user intent guidance for response generation
        if "user_intent" in jury_verdicts and jury_verdicts["user_intent"].get("approved", False):
            user_intent_verdict = jury_verdicts["user_intent"]
            state["user_intent_guidance"] = {
                "answer_format": user_intent_verdict.get("answer_format", "direct_answer"),
                "required_elements": user_intent_verdict.get("required_elements", []),
                "key_focus_areas": user_intent_verdict.get("key_focus_areas", []),
                "answer_template": user_intent_verdict.get("answer_template", "")
            }
            print(f"🏤 Added user intent guidance: {state['user_intent_guidance']['answer_format']}")
        
        print("🏤 ==================== JURY EVALUATION END ====================\n")
        
        return state

    def _create_fallback_verdict(self, error_msg: str = "") -> Dict[str, Any]:
        """Create a fallback verdict when jury evaluation fails"""
        return {
            "workflow_logic": {"approved": True, "reasoning": "Fallback approval due to evaluation error", "confidence": 0.5},
            "efficiency": {"approved": True, "reasoning": "Fallback approval due to evaluation error", "confidence": 0.5},
            "completeness": {"approved": True, "reasoning": "Fallback approval due to evaluation error", "confidence": 0.5},
            "user_intent": {"approved": True, "reasoning": "Fallback approval due to evaluation error", "confidence": 0.5},
            "evaluation_error": error_msg
        }