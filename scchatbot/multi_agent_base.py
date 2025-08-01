"""
Core multi-agent chatbot base class.

This module provides the base class for the multi-agent chatbot system,
including initialization, component management, and workflow orchestration.
"""

import os
import json
import shutil
from typing import Dict, Any, List, Literal

import openai
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from .cell_types.annotation_pipeline import initial_cell_annotation
from .analysis.visualizations import (
    display_dotplot,
    display_cell_type_composition,
    display_gsea_dotplot,
    display_umap,
    display_processed_umap,
    display_enrichment_barplot,
    display_enrichment_dotplot,
    display_enrichment_visualization
)
from .cell_types.utils import clear_directory
from .cell_types.models import ChatState
from .workflow.function_history import FunctionHistoryManager
from .cell_types.hierarchy_manager import HierarchicalCellTypeManager, CellTypeExtractor
from .analysis.analysis_wrapper import AnalysisFunctionWrapper
from .workflow import WorkflowNodes


class MultiAgentChatBot:
    """
    Base class for the multi-agent chatbot system.
    
    This class provides the core infrastructure for a multi-agent chatbot
    that can perform complex single-cell RNA-seq analysis tasks through
    orchestrated workflow execution.
    """
    
    def __init__(self):
        # Initialize core components
        self._initialize_directories()
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        self.adata = None
        
        # Initialize memory and awareness systems
        self.history_manager = FunctionHistoryManager("conversation_history")
        
        
        # Initialize hierarchical management components
        self.hierarchy_manager = None
        self.analysis_wrapper = None
        
        # Setup functions and initialize data
        self.setup_functions()
        self._initialize_annotation()
        # Initialize hierarchical management after adata is ready
        self._initialize_hierarchical_management()
        
        # Initialize cell type extractor
        self.cell_type_extractor = None
        self._initialize_cell_type_extractor()
        print("1")
        
        # Initialize workflow nodes
        self.workflow_nodes = WorkflowNodes(
            self.initial_annotation_content,
            self.initial_cell_types,
            self.adata,
            self.history_manager,
            self.hierarchy_manager,
            self.cell_type_extractor,
            self.function_descriptions,
            self.function_mapping,
            self.visualization_functions
        )
        print("2")
        
        # Create LangGraph workflow
        self.workflow = self._create_workflow()
        print("3")
        # Initialize session state management for conversation continuity
        self.session_states = {}  # Dict to store state for each session
        print("✅ Session state management initialized")
        
    def _initialize_directories(self):
        """Clean all directories at initialization"""
        directories_to_clear = [
            'figures', 'process_cell_data', 'scchatbot/annotated_adata',
            'scchatbot/enrichment', 'umaps/annotated', 'scchatbot/runtime_data/basic_data/',
            'scchatbot/deg_res', 'conversation_history'
        ]
        for directory in directories_to_clear:
            clear_directory(directory)
            
        # Clear execution history file specifically
        execution_history_file = "conversation_history/execution_history.json"
        if os.path.exists(execution_history_file):
            os.remove(execution_history_file)
            print(f"🧹 Cleared execution history: {execution_history_file}")

    def _initialize_annotation(self):
        """Initialize or load annotation data"""
        gene_dict, marker_tree, adata, explanation, annotation_result = initial_cell_annotation()
        
        # Handle case where h5ad file is not provided
        if adata is None:
            print(f"⚠️ Warning: {explanation}")
            # Set defaults for missing data
            self.adata = None
            self.initial_cell_types = []
            self.initial_annotation_content = f"Warning: {explanation}"
        else:
            self.adata = adata
            self.initial_cell_types = self._extract_initial_cell_types(annotation_result)
            self.initial_annotation_content = (
                f"Initial annotation complete.\n"
                f"• Annotation Result: {annotation_result}\n" 
                f"• Top‐genes per cluster: {gene_dict}\n"
                f"• Marker‐tree: {marker_tree}\n"
                f"• Explanation: {explanation}"
            )

    def _initialize_hierarchical_management(self):
        """Initialize hierarchical cell type management"""
        if self.adata is not None:
            self.hierarchy_manager = HierarchicalCellTypeManager(self.adata)
            self.analysis_wrapper = AnalysisFunctionWrapper(self.hierarchy_manager)
            print("✅ Unified hierarchical cell type management initialized")
        else:
            print("⚠️ Cannot initialize hierarchical management without adata")
    
    def _initialize_cell_type_extractor(self):
        """Initialize centralized cell type extractor"""
        if self.adata is not None:
            self.cell_type_extractor = CellTypeExtractor(
                hierarchy_manager=self.hierarchy_manager,
                adata=self.adata
            )
            # Add historical function access
            self.cell_type_extractor._get_historical_cell_types = self._get_historical_cell_types_for_extractor
            print("✅ Unified cell type extractor initialized")
        else:
            print("⚠️ Cannot initialize cell type extractor without adata")
    
    def _get_historical_cell_types_for_extractor(self) -> List[str]:
        """Get cell types from historical function executions for the extractor"""
        cell_types = set()
        
        # Get recent analyses from history manager
        recent_analyses = self.history_manager.get_recent_executions("perform_enrichment_analyses", limit=10)
        recent_analyses.extend(self.history_manager.get_recent_executions("dea_split_by_condition", limit=10))
        recent_analyses.extend(self.history_manager.get_recent_executions("process_cells", limit=10))
        
        for execution in recent_analyses:
            if execution.get("success") and execution.get("parameters"):
                cell_type = execution["parameters"].get("cell_type")
                if cell_type and cell_type != "overall":
                    cell_types.add(cell_type)
        
        return list(cell_types)

    def _extract_initial_cell_types(self, annotation_result: str) -> List[str]:
        """Extract cell types from initial annotation"""
        cell_types = []
        if isinstance(annotation_result, dict):
            for cluster, cell_type in annotation_result.items():
                if cell_type not in cell_types:
                    cell_types.append(cell_type)
        else:
            import re
            matches = re.findall(r"'([^']+)':\s*'([^']+)'", str(annotation_result))
            for _, cell_type in matches:
                if cell_type not in cell_types:
                    cell_types.append(cell_type)
        return cell_types

    def setup_functions(self):
        """Setup function descriptions and mappings"""
        self.visualization_functions = {
            "display_dotplot", "display_cell_type_composition", "display_gsea_dotplot",
            "display_umap", "display_processed_umap", "display_enrichment_barplot", 
            "display_enrichment_dotplot", "display_enrichment_visualization"
        }

        self.function_descriptions = [
            {
                "name": "display_dotplot",
                "description": "Display dotplot for the annotated results. Use when user wants to see gene expression patterns across cell types.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell_type": {"type": "string", "description": "The cell type to focus on, or 'Overall cells' for all cells"}
                    },
                    "required": []
                },
            },
            {
                "name": "display_cell_type_composition", 
                "description": "Display cell type composition graph. Use when user wants to see the proportion of different cell types.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "display_gsea_dotplot",
                "description": "Display GSEA dot plot. Use when user wants to see gene set enrichment analysis results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell_type": {"type": "string", "description": "The cell type to analyze, or 'overall' for all cells"},
                        "condition": {"type": "string", "description": "Optional specific condition folder"},
                        "top_n": {"type": "integer", "default": 20, "description": "Number of top terms to display"}
                    },
                    "required": []
                },
            },
            {
                "name": "display_umap",
                "description": "Display UMAP that is NOT annotated with cell types. Use when user wants basic dimensionality reduction visualization.",
                "parameters": {
                    "type": "object",
                    "properties": {"cell_type": {"type": "string", "description": "The cell type to focus on, or 'overall' for all cells"}},
                    "required": ["cell_type"],
                },
            },
            {
                "name": "display_processed_umap",
                "description": "Display UMAP that IS annotated with cell types. Use when user wants to see cell type annotations on UMAP.",
                "parameters": {
                    "type": "object", 
                    "properties": {"cell_type": {"type": "string", "description": "The cell type to focus on, or 'overall' for all cells"}},
                    "required": ["cell_type"],
                },
            },
            {
                "name": "perform_enrichment_analyses",
                "description": "Run enrichment analyses on DE genes for a cell type. Use for pathway analysis. Supports: REACTOME (pathways), GO (gene ontology), KEGG (pathways), GSEA (gene set enrichment). When user mentions specific analysis types (e.g., 'GSEA', 'GO', 'KEGG'), use those; otherwise run all four analyses by default.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell_type": {"type": "string", "description": "The cell type to analyze."},
                        "analyses": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["reactome", "go", "kegg", "gsea"]},
                            "description": "Which enrichment analyses to run. If user mentions 'GSEA'->['gsea'], 'GO'->['go'], 'KEGG'->['kegg'], 'REACTOME'->['reactome']. If no specific type mentioned, use ['reactome', 'go', 'kegg', 'gsea'] as default.",
                            "default": ["reactome", "go", "kegg", "gsea"]
                        },
                        "logfc_threshold": {"type": "number", "description": "Minimum absolute log2 fold change."},
                        "pval_threshold": {"type": "number", "description": "Adjusted p‑value cutoff."},
                        "top_n_terms": {"type": "integer", "description": "How many top enriched terms to return."}
                    },
                    "required": ["cell_type"]
                }
            },
            {
                "name": "process_cells",
                "description": "Recluster and further annotate cells based on cell type. Use when user wants to find subtypes within a cell type.",
                "parameters": {
                    "type": "object",
                    "properties": {"cell_type": {"type": "string", "description": "The cell type to process for subtype discovery."}},
                    "required": ["cell_type"]
                }
            },
            {
                "name": "display_enrichment_barplot",
                "description": "Show ONLY barplot of enriched terms. DEPRECATED: Use display_enrichment_visualization instead for better results. For GO analysis, domain defaults to BP if not specified.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "analysis": {"type": "string", "enum": ["reactome","go","kegg","gsea"], "description": "Type of enrichment analysis to visualize"},
                        "cell_type": {"type": "string"},
                        "top_n": {"type": "integer", "default": 10},
                        "domain": {"type": "string", "enum": ["BP","MF","CC"], "description": "Required for GO analysis. BP=Biological Process, MF=Molecular Function, CC=Cellular Component"},
                        "condition": {"type": "string"}
                    },
                    "required": ["analysis","cell_type"]
                }
            },
            {
                "name": "display_enrichment_dotplot", 
                "description": "Show ONLY dotplot of enriched terms. DEPRECATED: Use display_enrichment_visualization instead for better results. For GO analysis, domain defaults to BP if not specified.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "analysis": {"type": "string", "enum": ["reactome","go","kegg","gsea"], "description": "Type of enrichment analysis to visualize"},
                        "cell_type": {"type": "string"},
                        "top_n": {"type": "integer", "default": 10}, 
                        "domain": {"type": "string", "enum": ["BP","MF","CC"], "description": "Required for GO analysis. BP=Biological Process, MF=Molecular Function, CC=Cellular Component"},
                        "condition": {"type": "string"}
                    },
                    "required": ["analysis","cell_type"]
                }
            },
            {
                "name": "display_enrichment_visualization",
                "description": "PREFERRED: Show comprehensive enrichment visualization with both barplot and dotplot (default), or individual plots. Use this function for ALL enrichment visualization requests unless user specifically asks for only one plot type. For GO analysis, domain defaults to BP if not specified.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "analysis": {"type": "string", "enum": ["reactome","go","kegg","gsea"], "description": "Type of enrichment analysis to visualize"},
                        "cell_type": {"type": "string"},
                        "plot_type": {"type": "string", "enum": ["bar","dot","both"], "default": "both"},
                        "top_n": {"type": "integer", "default": 10}, 
                        "domain": {"type": "string", "enum": ["BP","MF","CC"], "description": "Required for GO analysis. BP=Biological Process, MF=Molecular Function, CC=Cellular Component"},
                        "condition": {"type": "string"}
                    },
                    "required": ["analysis","cell_type"]
                }
            },
            {
                "name": "dea_split_by_condition",
                "description": "Perform differential expression analysis (DEA) split by condition. Use when comparing conditions. This function enables the agent to answer questions related to differential expression genes.",
                "parameters": {
                    "type": "object",
                    "properties": {"cell_type": {"type": "string"}},
                    "required": ["cell_type"]
                }
            },
            {
                "name": "compare_cell_counts",
                "description": "Compare cell counts between experimental conditions for specific cell type(s). Use when analyzing how cell type abundance differs across conditions (e.g., pre vs post treatment). Supports both single cell type ('T cell') and multi-cell type comparisons ('B cell vs T cell' or 'B cell and T cell').", 
                "parameters": {
                    "type": "object",
                    "properties": {"cell_type": {"type": "string", "description": "The cell type(s) to analyze. Can be a single cell type like 'T cell' or multiple cell types like 'B cell vs T cell' or 'B cell and T cell'."}},
                    "required": ["cell_type"]
                }
            },
            {
                "name": "search_enrichment_semantic",
                "description": "Search all enrichment terms semantically to find specific pathways or biological processes regardless of their ranking in top results. Use when user asks about specific pathways, terms, or biological processes that might not appear in standard top-5 visualizations. Provides comprehensive search across all indexed enrichment data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The biological pathway, process, or term to search for (e.g., 'cell cycle', 'apoptosis', 'DNA repair')"},
                        "cell_type": {"type": "string", "description": "The cell type to search within. Can be inferred from conversation context if not explicitly mentioned."},
                        "analysis_type_filter": {"type": "string", "enum": ["go", "kegg", "reactome", "gsea"], "description": "Optional: limit search to specific analysis type"},
                        "condition_filter": {"type": "string", "description": "Optional: filter by specific experimental condition"},
                        "k": {"type": "integer", "default": 10, "description": "Number of top results to return"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "validate_processing_results",
                "description": "Validate that process_cells discovered expected cell types. Internal validation step.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "processed_parent": {"type": "string", "description": "The parent cell type that was processed"},
                        "expected_children": {"type": "array", "items": {"type": "string"}, "description": "Expected child cell types"}
                    },
                    "required": ["processed_parent", "expected_children"]
                }
            }
        ]
        
        # Function mappings
        self.function_mapping = {
            "display_dotplot": self._wrap_visualization(display_dotplot),
            "display_cell_type_composition": self._wrap_visualization(display_cell_type_composition),
            "display_gsea_dotplot": self._wrap_visualization(display_gsea_dotplot),
            "display_umap": self._wrap_visualization(display_umap),
            "display_processed_umap": self._wrap_visualization(display_processed_umap),
            "display_enrichment_barplot": self._wrap_visualization(display_enrichment_barplot),
            "display_enrichment_dotplot": self._wrap_visualization(display_enrichment_dotplot),
            "display_enrichment_visualization": self._wrap_visualization(display_enrichment_visualization),
            "perform_enrichment_analyses": self._wrap_enrichment_analysis,
            "process_cells": self._wrap_process_cells,
            "dea_split_by_condition": self._wrap_dea_analysis,
            "compare_cell_counts": self._wrap_compare_cells,
            "search_enrichment_semantic": self._wrap_search_enrichment_semantic,
            "validate_processing_results": self._wrap_validate_processing_results,
        }

    def _create_workflow(self) -> StateGraph:
        """Create simplified workflow without jury system"""
        workflow = StateGraph(ChatState)
        
        # Add workflow nodes
        workflow.add_node("input_processor", self.workflow_nodes.input_processor_node)
        workflow.add_node("planner", self.workflow_nodes.planner_node)
        workflow.add_node("executor", self.workflow_nodes.executor_node)
        workflow.add_node("step_evaluator", self.workflow_nodes.evaluator_node)  # NEW
        workflow.add_node("final_evaluator", self.workflow_nodes.evaluator_node)
        workflow.add_node("response_generator", self.workflow_nodes.unified_response_generator_node)
        
        # Set entry point
        workflow.set_entry_point("input_processor")
        
        # Main workflow connections - NEW PIPELINE: planner -> executor -> step_evaluator -> (continue|complete)
        workflow.add_edge("input_processor", "planner")
        workflow.add_edge("planner", "executor")
        
        # Modified routing: executor → step_evaluator → (continue|complete)
        workflow.add_conditional_edges(
            "executor",
            self.route_from_executor,
            {
                "step_evaluate": "step_evaluator",
                "complete": "final_evaluator"
            }
        )
        
        workflow.add_conditional_edges(
            "step_evaluator", 
            self.route_from_step_evaluator,
            {
                "continue": "executor",
                "complete": "final_evaluator",
                "abort": "response_generator"
            }
        )
        
        # Final evaluator reviews execution and routes to response generation or back to executor
        workflow.add_conditional_edges(
            "final_evaluator",
            self.route_from_final_evaluator,
            {
                "continue_execution": "executor",
                "generate_response": "response_generator"
            }
        )
        
        # Direct path: response generator goes directly to END
        workflow.add_edge("response_generator", END)
        
        return workflow.compile()

    def route_from_executor(self, state: ChatState) -> Literal["step_evaluate", "complete"]:
        """Always route to step evaluator unless no steps executed"""
        
        execution_history = state.get("execution_history", [])
        current_step_index = state.get("current_step_index", 0)
        total_steps = len(state.get("execution_plan", {}).get("steps", []))
        
        if execution_history and current_step_index <= total_steps:
            return "step_evaluate"
        else:
            return "complete"

    def route_from_step_evaluator(self, state: ChatState) -> Literal["continue", "complete", "abort"]:
        """Route based on step evaluation results"""
        
        step_evaluation = state.get("last_step_evaluation", {})
        current_step_index = state.get("current_step_index", 0) 
        total_steps = len(state.get("execution_plan", {}).get("steps", []))
        
        if step_evaluation.get("critical_failure", False):
            return "abort"
        elif current_step_index >= total_steps:
            return "complete"
        else:
            return "continue"

    def route_from_final_evaluator(self, state: ChatState) -> Literal["continue_execution", "generate_response"]:
        """Route from final evaluator based on post-execution evaluation results"""
        
        # If conversation_complete is False, it means supplementary steps were added
        if not state.get("conversation_complete", True):
            print("🔄 Routing: Post-execution evaluation added supplementary steps, continuing execution")
            return "continue_execution"
        else:
            print("🔄 Routing: Post-execution evaluation complete, generating response")
            return "generate_response"

    def send_message(self, message: str, session_id: str = "default") -> str:
        """Send a message to the chatbot and get response with conversation tracking"""
        try:
            # Get or create session state
            if session_id in self.session_states:
                # Reuse existing session state to preserve discovered cell types
                initial_state = self.session_states[session_id].copy()
                initial_state["current_message"] = message
                initial_state["response"] = ""
                initial_state["conversation_complete"] = False
                initial_state["errors"] = []
                print(f"🔄 Reusing session state for '{session_id}' with {len(initial_state.get('available_cell_types', []))} available cell types")
            else:
                # Create new initial state for new session
                initial_state: ChatState = {
                    "messages": [],
                    "current_message": message,
                    "response": "",
                    "available_cell_types": [],
                    "adata": None,
                    "execution_plan": None,
                    "current_step_index": 0,
                    "execution_history": [],
                    "function_result": None,
                    "function_name": None,
                    "function_args": None,
                    "function_history_summary": {},
                    "missing_cell_types": [],
                    "required_preprocessing": [],
                    "conversation_complete": False,
                    "errors": [],
                    "session_id": session_id  # Add session tracking
                }
                print(f"🆕 Creating new session state for '{session_id}'")
            
            # Invoke the workflow with recursion limit
            config = RunnableConfig(recursion_limit=100)
            final_state = self.workflow.invoke(initial_state, config=config)
            
            # Extract response
            response = final_state.get("response", "Analysis completed, but no response generated.")
            
            # Record conversation in vector database if using enhanced history manager
            if hasattr(self.history_manager, 'record_conversation_with_vector'):
                try:
                    # Extract clean response text
                    if response.startswith('{'):
                        response_data = json.loads(response)
                        response_text = response_data.get("response", response)
                    else:
                        response_text = response
                    
                    # Get analysis context for richer metadata
                    analysis_context = {
                        "execution_steps": len(final_state.get("execution_history", [])),
                        "successful_analyses": len([h for h in final_state.get("execution_history", []) 
                                                  if h.get("success", False)]),
                        "available_cell_types": final_state.get("available_cell_types", []),
                        "has_plots": bool(final_state.get("available_plots", []))
                    }
                    
                    # Record in vector database
                    self.history_manager.record_conversation_with_vector(
                        user_message=message,
                        bot_response=response_text,
                        session_id=session_id,
                        analysis_context=analysis_context
                    )
                    print("✅ Conversation recorded in vector database")
                    
                except Exception as e:
                    print(f"⚠️ Failed to record conversation in vector database: {e}")
            
            # Save the final state for this session to preserve discovered cell types
            self.session_states[session_id] = final_state
            print(f"💾 Session state saved for '{session_id}' with {len(final_state.get('available_cell_types', []))} available cell types")
            
            return response
                
        except Exception as e:
            print(f"❌ Error in workflow execution: {e}")
            return f"I encountered an error: {e}"

    def cleanup(self):
        """Cleanup resources and clear analysis results"""
        print("🧹 Starting cleanup...")
        
        # Close hierarchy manager connection
        if self.hierarchy_manager:
            self.hierarchy_manager.close()
            
        # Clear analysis result directories
        directories_to_clear = [
            'scchatbot/enrichment', 'scchatbot/deg_res',
            'figures', 'umaps/annotated'
        ]
        
        for directory in directories_to_clear:
            try:
                if os.path.exists(directory):
                    clear_directory(directory)
                    print(f"🧹 Cleared directory: {directory}")
            except Exception as e:
                print(f"⚠️ Failed to clear {directory}: {e}")
        
        # Remove execution history file
        execution_history_file = "conversation_history/execution_history.json"
        if os.path.exists(execution_history_file):
            try:
                os.remove(execution_history_file)
                print(f"🧹 Removed execution history: {execution_history_file}")
            except Exception as e:
                print(f"⚠️ Failed to remove execution history: {e}")
                
        print("✅ Cleanup completed")
            

    # ========== Function Wrappers ==========
    
    def _wrap_visualization(self, func):
        """Wrapper for visualization functions"""
        def wrapper(**kwargs):
            try:
                func_name = func.__name__
                
                # DEBUG: Log all parameters passed to visualization wrapper
                print(f"🔍 VIZ WRAPPER DEBUG: {func_name} called with kwargs: {kwargs}")
                
                # Handle different function signatures properly
                if func_name == 'display_dotplot':
                    # Now takes cell_type parameter
                    cell_type = kwargs.get('cell_type', 'Overall cells')
                    return func(cell_type)
                elif func_name == 'display_cell_type_composition':
                    # Takes no parameters  
                    return func()
                elif func_name == 'display_gsea_dotplot':
                    # Takes cell_type and other optional parameters
                    cell_type = kwargs.get('cell_type', 'overall')
                    condition = kwargs.get('condition', None)
                    top_n = kwargs.get('top_n', 20)
                    return func(cell_type=cell_type, condition=condition, top_n=top_n)
                elif func_name in ['display_umap', 'display_processed_umap']:
                    # Takes cell_type as positional argument
                    cell_type = kwargs.get('cell_type', 'overall')
                    return func(cell_type)
                elif func_name in ['display_enrichment_barplot', 'display_enrichment_dotplot']:
                    # Takes analysis and cell_type as positional arguments, other args as kwargs
                    analysis = kwargs.pop('analysis', 'go')  # Default to go if not provided
                    cell_type = kwargs.pop('cell_type', 'overall')  # Default if not provided
                    
                    # Handle GO domain - default to BP if not specified
                    if analysis.lower() == 'go' and 'domain' not in kwargs:
                        kwargs['domain'] = 'BP'
                        
                    return func(analysis, cell_type, **kwargs)
                elif func_name == 'display_enrichment_visualization':
                    # Takes analysis, cell_type and other keyword arguments
                    analysis = kwargs.pop('analysis', 'go')
                    cell_type = kwargs.pop('cell_type', 'overall')
                    plot_type = kwargs.pop('plot_type', 'both')
                    
                    # CRITICAL FIX: Don't default to 'overall' if we have a more specific cell type
                    if cell_type == 'overall':
                        print(f"⚠️ WARNING: cell_type defaulted to 'overall' - this may indicate planner issue")
                    
                    # Handle GO domain - default to BP if not specified
                    if analysis.lower() == 'go' and 'domain' not in kwargs:
                        kwargs['domain'] = 'BP'
                    
                    print(f"🎨 Visualization call: {func_name}(analysis='{analysis}', cell_type='{cell_type}', plot_type='{plot_type}', kwargs={kwargs})")
                        
                    return func(analysis, cell_type, plot_type, **kwargs)
                else:
                    # Default behavior
                    return func(**kwargs)
            except Exception as e:
                print(f"❌ Visualization error: {e}")
                return f"Visualization failed: {e}"
        return wrapper

    def _wrap_enrichment_analysis(self, **kwargs):
        """Enrichment analysis wrapper"""
        return self._compute_enrichment_analysis(**kwargs)
    
    def _compute_enrichment_analysis(self, **kwargs):
        """Internal method to compute enrichment analysis (extracted for caching)"""
        # Set default analyses if not specified
        if "analyses" not in kwargs or not kwargs["analyses"]:
            kwargs["analyses"] = ["reactome", "go", "kegg", "gsea"]
            print(f"🧬 No specific analyses requested, using default: {kwargs['analyses']}")
        
        if self.analysis_wrapper:
            try:
                return self.analysis_wrapper.perform_enrichment_analyses_hierarchical(**kwargs)
            except Exception as e:
                print(f"❌ Hierarchical analysis failed: {e}")
                # Fallback to direct function call
                from .analysis.enrichment_analysis import perform_enrichment_analyses
                return perform_enrichment_analyses(self.adata, **kwargs)
        else:
            # Fallback to direct function call
            from .analysis.enrichment_analysis import perform_enrichment_analyses
            return perform_enrichment_analyses(self.adata, **kwargs)

    def _wrap_dea_analysis(self, **kwargs):
        """DEA analysis wrapper"""
        return self._compute_dea_analysis(**kwargs)
    
    def _compute_dea_analysis(self, **kwargs):
        """Internal method to compute DEA analysis (extracted for caching)"""
        if self.analysis_wrapper:
            try:
                return self.analysis_wrapper.dea_split_by_condition_hierarchical(**kwargs)
            except Exception as e:
                print(f"❌ Hierarchical DEA failed: {e}")
                # Fallback to direct function call
                from .cell_types.utils import dea_split_by_condition
                return dea_split_by_condition(self.adata, **kwargs)
        else:
            # Fallback to direct function call
            from .cell_types.utils import dea_split_by_condition
            return dea_split_by_condition(self.adata, **kwargs)

    def _wrap_process_cells(self, **kwargs):
        """Process cells wrapper with hierarchy awareness"""
        try:
            from .cell_types.annotation_pipeline import handle_process_cells_result
            cell_type = kwargs.get("cell_type", "unknown")
            resolution = kwargs.get("resolution", None)
            
            # Use handle_process_cells_result which internally calls process_cells
            result = handle_process_cells_result(self.adata, cell_type, resolution)
            
            if result is None:
                # This means there was an error or special case (leaf node, no cells, etc.)
                return "Process cells completed, but no new cell types were discovered."
            
            # Result is a string with the annotation explanation
            # Note: The adata is modified in-place by process_cells, so we don't need to update it
            # But we should update workflow nodes to be safe
            if self.workflow_nodes:
                self.workflow_nodes.adata = self.adata
            
            return result
            
        except Exception as e:
            print(f"❌ Process cells error: {e}")
            return f"Process cells failed: {e}"

    def _wrap_compare_cells(self, **kwargs):
        """Compare cells wrapper"""
        cell_type = kwargs.get("cell_type")
        
        # Debug: Print what we received
        print(f"🔍 compare_cell_counts called with cell_type='{cell_type}', kwargs={kwargs}")
        
        # If no cell_type provided, this might be a generic comparison request
        if not cell_type:
            return "Error: No cell type specified for comparison. Please specify which cell type(s) to compare."
        
        # Handle multi-cell-type comparisons (e.g., "B cell vs T cell")
        if self._is_multi_cell_type_comparison(cell_type):
            print(f"🎯 Detected multi-cell-type comparison: {cell_type}")
            return self._handle_multi_cell_type_comparison(cell_type, **kwargs)
        
        # Validate single cell type exists
        if cell_type == "overall":
            return "Error: 'overall' is not a valid cell type for comparison. Please specify a specific cell type like 'T cell' or 'B cell'."
        
        print(f"🔄 Processing single cell type comparison: {cell_type}")
        
        # Remove cell_type from kwargs to avoid "multiple values" error
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "cell_type"}
        
        if self.analysis_wrapper:
            return self.analysis_wrapper.compare_cell_count_hierarchical(cell_type, **filtered_kwargs)
        else:
            # Fallback to direct function call
            from .cell_types.utils import compare_cell_count
            return compare_cell_count(self.adata, cell_type)

    def _wrap_validate_processing_results(self, **kwargs):
        """Validate processing results wrapper"""
        return self.workflow_nodes.validate_processing_results(
            kwargs.get("processed_parent"),
            kwargs.get("expected_children", [])
        )
    
    def _is_multi_cell_type_comparison(self, cell_type: str) -> bool:
        """Check if cell_type contains multiple cell types for comparison"""
        if not isinstance(cell_type, str):
            return False
        
        # Common patterns for multi-cell-type comparisons
        comparison_indicators = [
            " vs ", " versus ", " and ", " & ", ",", " compared to ", " against "
        ]
        
        return any(indicator in cell_type.lower() for indicator in comparison_indicators)
    
    def _handle_multi_cell_type_comparison(self, cell_type: str, **kwargs):
        """Handle comparisons between multiple cell types"""
        try:
            # Parse cell types from the comparison string
            cell_types = self._parse_comparison_cell_types(cell_type)
            
            if len(cell_types) < 2:
                return f"Could not identify multiple cell types in: {cell_type}"
            
            # Remove cell_type from kwargs to avoid conflicts
            filtered_kwargs = {k: v for k, v in kwargs.items() if k != "cell_type"}
            
            results = []
            for ct in cell_types:
                ct = ct.strip()
                try:
                    if self.analysis_wrapper:
                        result = self.analysis_wrapper.compare_cell_count_hierarchical(ct, **filtered_kwargs)
                    else:
                        from .cell_types.utils import compare_cell_count
                        result = compare_cell_count(self.adata, ct)
                    
                    results.append(f"\\n=== {ct} ===\\n{result}")
                except Exception as e:
                    results.append(f"\\n=== {ct} ===\\nError: {e}")
            
            comparison_summary = f"Cell Count Comparison Results:\\n" + "\\n".join(results)
            comparison_summary += f"\\n\\n📊 Summary: Compared cell counts across conditions for {len(cell_types)} cell types: {', '.join(cell_types)}"
            
            return comparison_summary
            
        except Exception as e:
            return f"Error in multi-cell-type comparison: {e}"
    
    def _parse_comparison_cell_types(self, cell_type: str) -> list:
        """Parse cell types from a comparison string"""
        # Split by common separators
        separators = [" vs ", " versus ", " and ", " & ", ",", " compared to ", " against "]
        
        cell_types = [cell_type]
        for separator in separators:
            new_types = []
            for ct in cell_types:
                if separator in ct.lower():
                    split_types = ct.split(separator)
                    new_types.extend([t.strip() for t in split_types])
                else:
                    new_types.append(ct)
            cell_types = new_types
        
        # Clean up and deduplicate
        cleaned_types = []
        for ct in cell_types:
            ct = ct.strip()
            if ct and ct not in cleaned_types:
                cleaned_types.append(ct)
        
        return cleaned_types
    
    def _wrap_search_enrichment_semantic(self, **kwargs):
        """
        Conversation-aware semantic search wrapper for enrichment results.
        
        Implements Option C: fallback parameter resolution using conversation context.
        """
        query = kwargs.get("query", "")
        cell_type = kwargs.get("cell_type", "")
        analysis_type_filter = kwargs.get("analysis_type_filter")
        condition_filter = kwargs.get("condition_filter") 
        k = kwargs.get("k", 10)
        
        # Option C: Infer missing cell_type from conversation context
        if not cell_type or cell_type == "unknown" or cell_type == "overall":
            print(f"🔄 Cell type missing or generic ('{cell_type}'), inferring from context...")
            
            # Try to get cell_type from recent enrichment analyses
            recent_analyses = self.history_manager.get_recent_executions("perform_enrichment_analyses", limit=3)
            if recent_analyses:
                inferred_cell_type = recent_analyses[-1]["parameters"].get("cell_type", "")
                if inferred_cell_type and inferred_cell_type != "unknown":
                    cell_type = inferred_cell_type
                    print(f"✅ Inferred cell_type from recent analysis: '{cell_type}'")
            
            # Fallback: try to get from recent process_cells operations
            if not cell_type or cell_type == "unknown":
                recent_process = self.history_manager.get_recent_executions("process_cells", limit=2)
                if recent_process:
                    # Look for newly discovered cell types in the results
                    for execution in reversed(recent_process):
                        if execution.get("success") and execution.get("result"):
                            result_str = str(execution["result"])
                            if "Discovered new cell type:" in result_str:
                                # Extract discovered cell types
                                import re
                                discoveries = re.findall(r"Discovered new cell type: ([^\\n]+)", result_str)
                                if discoveries:
                                    cell_type = discoveries[-1].strip()  # Use the last discovered type
                                    print(f"✅ Inferred cell_type from process_cells result: '{cell_type}'")
                                    break
        
        # Validate that we have required parameters
        if not query:
            return {"error": "Search query is required", "results": []}
        
        if not cell_type or cell_type == "unknown":
            return {
                "error": "Could not determine cell type from context. Please specify the cell type for semantic search.",
                "query": query,
                "results": []
            }
        
        print(f"🔍 Semantic search: '{query}' in '{cell_type}' (k={k})")
        
        try:
            # Call the vector database search
            search_results = self.history_manager.search_enrichment_semantic(
                query=query,
                cell_type=cell_type,
                condition_filter=condition_filter,
                analysis_type_filter=analysis_type_filter,
                k=k
            )
            
            # Return structured data for response generation (instead of formatted string)
            if search_results.get("total_matches", 0) > 0:
                return {
                    "search_results": search_results,
                    "query": query,
                    "cell_type": cell_type,
                    "total_matches": search_results.get("total_matches", 0),
                    "formatted_summary": self._format_semantic_search_results(search_results)
                }
            else:
                return {
                    "message": f"No enrichment terms found matching '{query}' in {cell_type}",
                    "query": query,
                    "cell_type": cell_type,
                    "total_matches": 0,
                    "suggestions": f"Try broader terms or check if enrichment analysis was performed for {cell_type}"
                }
        
        except Exception as e:
            print(f"❌ Semantic search failed: {e}")
            return {
                "error": f"Semantic search failed: {e}",
                "query": query,
                "cell_type": cell_type
            }
    
    def _format_semantic_search_results(self, search_results: dict) -> str:
        """Format semantic search results for user-friendly display"""
        
        if not search_results.get("results"):
            return "No matching enrichment terms found."
        
        query = search_results.get("query", "")
        cell_type = search_results.get("cell_type", "")
        total_matches = search_results.get("total_matches", 0)
        results = search_results.get("results", [])
        
        # Build formatted response
        response_parts = []
        response_parts.append(f"🔍 Found {total_matches} enrichment terms matching '{query}' in {cell_type}:")
        response_parts.append("")
        
        # Group results by analysis type for better organization
        by_analysis = {}
        for result in results[:10]:  # Show top 10
            analysis_type = result.get("analysis_type", "unknown").upper()
            if analysis_type not in by_analysis:
                by_analysis[analysis_type] = []
            by_analysis[analysis_type].append(result)
        
        # Format each analysis type group
        for analysis_type, analysis_results in by_analysis.items():
            response_parts.append(f"📊 {analysis_type} Results:")
            
            for i, result in enumerate(analysis_results, 1):
                term_name = result.get("term_name", "")
                rank = result.get("rank", "?")
                similarity = result.get("similarity_score", 0)
                p_value = result.get("adj_p_value", "")
                genes = result.get("intersecting_genes", "")
                
                # Format the result line
                result_line = f"  {i}. {term_name}"
                result_line += f" (rank: {rank}, similarity: {similarity:.3f}"
                if p_value:
                    result_line += f", p-value: {p_value:.2e}" if isinstance(p_value, float) else f", p-value: {p_value}"
                result_line += ")"
                
                response_parts.append(result_line)
                
                # Add genes if available (first few only)
                if genes:
                    gene_list = genes.split(',')[:5]  # First 5 genes
                    gene_str = ', '.join(gene_list)
                    if len(genes.split(',')) > 5:
                        gene_str += "..."
                    response_parts.append(f"     Genes: {gene_str}")
            
            response_parts.append("")  # Empty line between analysis types
        
        # Add summary footer
        if total_matches > 10:
            response_parts.append(f"... and {total_matches - 10} more results")
        
        response_parts.append(f"💡 These terms were found regardless of their ranking in standard top-5 results.")
        
        return "\n".join(response_parts)