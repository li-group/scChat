"""
Data structures and type definitions for cell type management and chatbot state.

This module contains all the core data structures, enums, and type definitions
used throughout the chatbot system for cell type management and workflow state.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, TypedDict
from langchain_core.messages import BaseMessage


class CellTypeRelation(Enum):
    """Enumeration of possible relationships between cell types"""
    ANCESTOR = "ancestor"
    DESCENDANT = "descendant" 
    SIBLING = "sibling"
    SAME = "same"
    UNRELATED = "unrelated"


@dataclass
class CellTypeLineage:
    """Represents the lineage and processing history of a cell type"""
    cell_id: str
    current_type: str
    lineage_path: List[str]
    processing_history: List[Dict[str, Any]]


class ChatState(TypedDict):
    """State that flows through the LangGraph workflow"""
    # Input/Output
    messages: List[BaseMessage]
    current_message: str
    response: str
    
    # Analysis context
    available_cell_types: List[str]
    adata: Optional[Any]
    
    # Planning and execution
    initial_plan: Optional[Dict[str, Any]]  # Initial plan before validation
    execution_plan: Optional[Dict[str, Any]]  # Validated plan after status checking
    current_step_index: int
    execution_history: List[Dict[str, Any]]
    
    # Function execution
    function_result: Optional[Any]
    function_name: Optional[str]
    function_args: Optional[Dict[str, Any]]
    
    # Memory and awareness
    function_history_summary: Optional[Dict[str, Any]]
    missing_cell_types: List[str]
    required_preprocessing: List[Dict[str, Any]]
    
    # Conversation management
    conversation_complete: bool
    errors: List[str]
    
    # 🆕 CRITIC AGENT FIELDS
    critic_iterations: int
    critic_feedback_history: List[Dict[str, Any]]
    plan_revision_history: List[Dict[str, Any]]
    original_execution_complete: bool
    cumulative_analysis_results: Dict[str, Any]
    impossible_request_detected: bool
    degradation_strategy: Optional[Dict[str, Any]]
    error_recovery_strategy: Optional[Dict[str, Any]]
    revision_applied: bool


@dataclass
class ExecutionStep:
    """Represents a single step in the execution plan"""
    step_type: str
    function_name: str
    parameters: Dict[str, Any]
    description: str
    expected_outcome: Optional[str] = None
    target_cell_type: Optional[str] = None
    expected_children: Optional[List[str]] = None


@dataclass
class ExecutionPlan:
    """Represents the complete execution plan"""
    steps: List[ExecutionStep]
    original_question: str
    plan_summary: str
    estimated_steps: int


@dataclass
class CriticEvaluation:
    """Structured critic evaluation result"""
    relevance_score: float
    completeness_score: float  
    missing_analyses: List[str]
    recommendations: List[str]
    needs_revision: bool
    reasoning: str
    evaluation_type: str