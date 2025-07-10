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

from .annotation import initial_cell_annotation
from .visualizations import (
    display_dotplot,
    display_cell_type_composition,
    display_gsea_dotplot,
    display_umap,
    display_processed_umap,
    display_enrichment_barplot,
    display_enrichment_dotplot,
    display_enrichment_visualization
)
from .utils import clear_directory
from .cell_type_models import ChatState
from .function_history import FunctionHistoryManager
from .cache_manager import SimpleIntelligentCache
from .cell_type_hierarchy import HierarchicalCellTypeManager, CellTypeExtractor
from .analysis_wrapper import AnalysisFunctionWrapper
from .workflow_nodes import WorkflowNodes


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
        self.api_key = os.getenv("OPENAI_API_KEY", "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")
        openai.api_key = self.api_key
        self.adata = None
        
        # Initialize memory and awareness systems
        self.history_manager = FunctionHistoryManager()
        
        # Initialize intelligent cache with insights
        self.simple_cache = SimpleIntelligentCache()
        self.simple_cache.ensure_cache_directories()
        
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
        
        # Initialize jury system (evaluation system)
        from .jury_system_main import JurySystem
        self.jury_system = JurySystem(
            simple_cache=self.simple_cache,
            hierarchy_manager=self.hierarchy_manager,
            history_manager=self.history_manager,
            function_descriptions=self.function_descriptions
        )
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
            self.visualization_functions,
            self.simple_cache
        )
        # Create LangGraph workflow
        self.workflow = self._create_workflow()
        
    def _initialize_directories(self):
        """Clean all directories at initialization"""
        directories_to_clear = [
            'figures', 'process_cell_data', 'scchatbot/annotated_adata',
            'scchatbot/enrichment', 'umaps/annotated', 'scchatbot/runtime_data/basic_data/',
            'scchatbot/deg_res', 'function_history'
        ]
        for directory in directories_to_clear:
            clear_directory(directory)
            
        # Clear execution history file specifically
        execution_history_file = "function_history/execution_history.json"
        if os.path.exists(execution_history_file):
            os.remove(execution_history_file)
            print(f"🧹 Cleared execution history: {execution_history_file}")

    def _initialize_annotation(self):
        """Initialize or load annotation data"""
        gene_dict, marker_tree, adata, explanation, annotation_result = initial_cell_annotation()
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
                "name": "conversational_response",
                "description": "Provide a conversational response without function calls. Use for greetings, clarifications, explanations, interpretive questions about analysis results, or when no analysis is needed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "response_type": {"type": "string", "enum": ["greeting", "explanation", "clarification", "general", "interpretation", "analysis_summary"]},
                        "cell_type": {"type": "string", "description": "Cell type for interpretation/analysis questions"},
                        "question_context": {"type": "string", "description": "Additional context for interpretation questions"}
                    },
                    "required": ["response_type"]
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
            "conversational_response": self._wrap_conversational_response,
            "validate_processing_results": self._wrap_validate_processing_results,
        }

    def _create_workflow(self) -> StateGraph:
        """Create enhanced workflow with jury-based evaluation system"""
        workflow = StateGraph(ChatState)
        
        # Add workflow nodes
        workflow.add_node("input_processor", self.workflow_nodes.input_processor_node)
        workflow.add_node("planner", self.workflow_nodes.planner_node)
        workflow.add_node("evaluator", self.workflow_nodes.evaluator_node)
        workflow.add_node("executor", self.workflow_nodes.executor_node)
        workflow.add_node("response_generator", self.workflow_nodes.unified_response_generator_node)  # NEW: Use unified generator
        workflow.add_node("plot_integration", self.workflow_nodes.add_plots_to_final_response)  # NEW: Add plots post-jury
        
        # Jury system nodes (new evaluation system)
        workflow.add_node("jury_evaluation", self.jury_system.jury_evaluation_node)
        workflow.add_node("conflict_resolution", self.jury_system.conflict_resolution_node)
        workflow.add_node("targeted_revision", self.jury_system.targeted_revision_node)
        
        
        # Set entry point
        workflow.set_entry_point("input_processor")
        
        # Main workflow connections
        workflow.add_edge("input_processor", "planner")
        workflow.add_edge("planner", "evaluator")
        
        # Routing from evaluator to continue execution or generate response
        workflow.add_conditional_edges(
            "evaluator", 
            self.route_from_evaluator,
            {
                "continue": "executor",
                "to_response": "response_generator"  # NEW: Route to response generator when execution complete
            }
        )
        
        # Routing from executor - either continue executing or generate response when complete
        workflow.add_conditional_edges(
            "executor",
            self.route_from_executor,
            {
                "continue": "executor",  # Continue with next step
                "complete": "response_generator"  # NEW: All steps done, generate response first
            }
        )
        
        # NEW: Response generator always goes to jury for evaluation
        workflow.add_edge("response_generator", "jury_evaluation")
        
        # Jury system routing
        workflow.add_conditional_edges(
            "jury_evaluation",
            self.jury_system.route_from_jury,
            {
                "accept": "plot_integration",  # NEW: Accepted responses go to plot integration first
                "revise_analysis": "targeted_revision",
                "revise_presentation": "conflict_resolution"
            }
        )
        
        # Jury revision flows
        workflow.add_edge("conflict_resolution", "response_generator")  # Presentation fixes go back to response generator
        workflow.add_edge("targeted_revision", "evaluator")  # Analysis revisions restart from evaluator
        
        # Final response generation with plots
        workflow.add_edge("plot_integration", END)  # NEW: Plot integration is the final step
        
        return workflow.compile()

    def route_from_evaluator(self, state: ChatState) -> Literal["continue", "to_response"]:
        """Route from evaluator to continue execution or go to jury"""
        
        # Defensive check: ensure execution_plan exists
        if not state.get("execution_plan") or not state["execution_plan"].get("steps"):
            print("⚠️ Routing: No execution plan or steps found, generating response")
            state["conversation_complete"] = True
            return "to_response"  # Route to response generator to handle the error
        
        current_step_index = state.get("current_step_index", 0)
        total_steps = len(state["execution_plan"]["steps"])
        
        # Check if all steps are complete
        if current_step_index >= total_steps:
            print("🏁 All execution steps complete - routing to response generation")
            return "to_response"
        
        # If plan needs processing and hasn't been processed yet, we'll process it and then continue
        if not state.get("plan_processed"):
            print("🔧 Plan needs processing - enhanced evaluator will process then continue")
            return "continue"  # Enhanced evaluator will process the plan and set plan_processed=True
        
        # If there are more steps, continue execution
        print(f"🔄 Continuing execution - step {current_step_index + 1}/{total_steps}")
        return "continue"

    def route_from_executor(self, state: ChatState) -> Literal["continue", "complete"]:
        """Route from executor - continue with next step or complete to response generation"""
        
        # Check if execution is complete
        current_step_index = state.get("current_step_index", 0)
        execution_plan = state.get("execution_plan", {})
        total_steps = len(execution_plan.get("steps", []))
        
        if current_step_index >= total_steps:
            print("🏁 All execution steps complete - routing to response generation")
            return "complete"
        else:
            print(f"🔄 Executor continuing - next step {current_step_index + 1}/{total_steps}")
            return "continue"

    def send_message(self, message: str) -> str:
        """Send a message to the chatbot and get response"""
        try:
            # Create initial state with all required fields
            initial_state: ChatState = {
                "messages": [],
                "current_message": message,
                "response": "",
                "available_cell_types": [],
                "adata": None,
                "initial_plan": None,
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
                
                
                # Jury system fields
                "jury_verdicts": None,
                "jury_decision": None,
                "revision_type": None,
                "jury_iteration": 0,
                "conflict_resolution_applied": False
            }
            
            # Invoke the workflow with recursion limit
            config = RunnableConfig(recursion_limit=100)
            final_state = self.workflow.invoke(initial_state, config=config)
            
            # Extract response - return it directly as the original did
            return final_state.get("response", "Analysis completed, but no response generated.")
                
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
            'scchatbot/enrichment', 'scchatbot/deg_res', 'function_history',
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
        execution_history_file = "function_history/execution_history.json"
        if os.path.exists(execution_history_file):
            try:
                os.remove(execution_history_file)
                print(f"🧹 Removed execution history: {execution_history_file}")
            except Exception as e:
                print(f"⚠️ Failed to remove execution history: {e}")
                
        print("✅ Cleanup completed")
            
    def manage_cache(self, action: str, **kwargs):
        """Manage cache operations"""
        if action == "invalidate":
            self.simple_cache.invalidate_cache(**kwargs)
        elif action == "stats":
            return self.simple_cache.get_cache_stats()
        elif action == "insights":
            cell_type = kwargs.get("cell_type")
            if cell_type:
                return self.simple_cache.get_analysis_insights(cell_type)
            else:
                return "Cell type required for insights"

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
        """🧠 CACHED: Enrichment analysis with simple intelligent caching"""
        
        # Remove analysis_type from kwargs to avoid "multiple values" error
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "analysis_type"}
        
        # Use cache-aware wrapper
        return self.simple_cache.cache_aware_function_wrapper(
            function_name="perform_enrichment_analyses",
            analysis_type="enrichment", 
            compute_function=lambda: self._compute_enrichment_analysis(**kwargs),
            **filtered_kwargs
        )
    
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
                from .enrichment import perform_enrichment_analyses
                return perform_enrichment_analyses(self.adata, **kwargs)
        else:
            # Fallback to direct function call
            from .enrichment import perform_enrichment_analyses
            return perform_enrichment_analyses(self.adata, **kwargs)

    def _wrap_dea_analysis(self, **kwargs):
        """🧠 CACHED: DEA analysis with simple intelligent caching"""
        
        # Use cache-aware wrapper
        return self.simple_cache.cache_aware_function_wrapper(
            function_name="dea_split_by_condition",
            analysis_type="dea", 
            compute_function=lambda: self._compute_dea_analysis(**kwargs),
            **kwargs
        )
    
    def _compute_dea_analysis(self, **kwargs):
        """Internal method to compute DEA analysis (extracted for caching)"""
        if self.analysis_wrapper:
            try:
                return self.analysis_wrapper.dea_split_by_condition_hierarchical(**kwargs)
            except Exception as e:
                print(f"❌ Hierarchical DEA failed: {e}")
                # Fallback to direct function call
                from .utils import dea_split_by_condition
                return dea_split_by_condition(self.adata, **kwargs)
        else:
            # Fallback to direct function call
            from .utils import dea_split_by_condition
            return dea_split_by_condition(self.adata, **kwargs)

    def _wrap_process_cells(self, **kwargs):
        """Process cells wrapper with hierarchy awareness"""
        try:
            from .annotation import handle_process_cells_result
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
            from .utils import compare_cell_count
            return compare_cell_count(self.adata, cell_type)

    def _wrap_conversational_response(self, **kwargs):
        """Enhanced conversational response wrapper with analysis interpretation capabilities"""
        response_type = kwargs.get("response_type", "general")
        cell_type = kwargs.get("cell_type", None)
        question_context = kwargs.get("question_context", "")
        
        if response_type == "greeting":
            return "Hello! I'm here to help you analyze your single-cell RNA-seq data. What would you like to explore?"
        
        elif response_type == "explanation":
            return "I can help you with various analyses including differential expression, enrichment analysis, cell type discovery, and visualizations."
        
        elif response_type == "clarification":
            return "Could you please provide more details about what you'd like to analyze?"
        
        elif response_type == "interpretation":
            return self._generate_analysis_interpretation(cell_type, question_context)
        
        elif response_type == "analysis_summary":
            return self._generate_analysis_summary(cell_type)
        
        else:
            return "I'm ready to help with your single-cell analysis. Please let me know what you'd like to do."
    
    def _generate_analysis_interpretation(self, cell_type: str = None, question_context: str = "") -> str:
        """Generate interpretation of analysis results for a specific cell type"""
        try:
            # Get available analysis results from history
            available_results = self.history_manager.get_available_results()
            
            if not cell_type:
                # Provide general interpretation based on all available results
                return self._generate_general_interpretation(available_results, question_context)
            
            # Get specific insights for the requested cell type
            if self.simple_cache:
                insights = self.simple_cache.get_analysis_insights(cell_type)
            else:
                insights = {}
            
            # Check if we have any data for this cell type
            if not insights.get("enrichment_insights") and not insights.get("dea_insights"):
                available_cell_types = self._get_available_cell_types()
                suggestion = ""
                if available_cell_types:
                    # Check if the requested cell type is similar to available ones
                    similar_types = [ct for ct in available_cell_types if cell_type.lower() in ct.lower() or ct.lower() in cell_type.lower()]
                    if similar_types:
                        suggestion = f" Did you mean one of these: {', '.join(similar_types)}?"
                    else:
                        suggestion = f" Available cell types include: {', '.join(available_cell_types[:5])}{'...' if len(available_cell_types) > 5 else ''}."
                
                return f"I don't have any analysis results for '{cell_type}' yet.{suggestion} Would you like me to run some analyses first?"
            
            # Generate interpretation based on available insights
            interpretation = self._format_cell_type_interpretation(cell_type, insights, question_context)
            return interpretation
            
        except Exception as e:
            print(f"⚠️ Error generating interpretation: {e}")
            return f"I'm having trouble accessing the analysis results. Could you please be more specific about what you'd like to know?"
    
    def _generate_analysis_summary(self, cell_type: str = None) -> str:
        """Generate a summary of available analysis results"""
        try:
            available_results = self.history_manager.get_available_results()
            
            if not available_results:
                return "No analysis results are available yet. Would you like me to run some analyses?"
            
            summary_parts = []
            
            # Summarize processed cell types
            if "processed_cell_types" in available_results:
                processed = available_results["processed_cell_types"]
                summary_parts.append(f"📊 Processed cell types: {', '.join(processed)}")
            
            # Summarize enrichment analyses
            if "enrichment_analyses" in available_results:
                enrichment = available_results["enrichment_analyses"]
                for ct, analyses in enrichment.items():
                    summary_parts.append(f"🧬 {ct}: {', '.join(analyses)} enrichment")
            
            # Summarize DEA analyses
            if "dea_analyses" in available_results:
                dea = available_results["dea_analyses"]
                summary_parts.append(f"📈 DEA completed for: {', '.join(dea)}")
            
            # If specific cell type requested, add detailed insights
            if cell_type and self.simple_cache:
                insights = self.simple_cache.get_analysis_insights(cell_type)
                if insights.get("summary"):
                    summary_parts.append(f"\n🎯 {cell_type} specific insights:")
                    for insight in insights["summary"]:
                        summary_parts.append(f"  • {insight}")
            
            if summary_parts:
                return "\n".join(summary_parts)
            else:
                return "No analysis results are available yet. Would you like me to run some analyses?"
                
        except Exception as e:
            print(f"⚠️ Error generating summary: {e}")
            return "I'm having trouble accessing the analysis results."
    
    def _generate_general_interpretation(self, available_results: dict, question_context: str) -> str:
        """Generate general interpretation when no specific cell type is provided"""
        if not available_results:
            # Still show available cell types if no analyses have been run
            available_cell_types = self._get_available_cell_types()
            if available_cell_types:
                return f"I don't have any analysis results to interpret yet, but I can see {len(available_cell_types)} available cell types: {', '.join(available_cell_types[:5])}{'...' if len(available_cell_types) > 5 else ''}. Would you like me to run some analyses first?"
            else:
                return "I don't have any analysis results to interpret yet. Would you like me to run some analyses first?"
        
        # Build context about what's available
        context_parts = []
        
        if "processed_cell_types" in available_results:
            context_parts.append(f"I have processed {len(available_results['processed_cell_types'])} cell types")
        
        if "enrichment_analyses" in available_results:
            enrichment_count = sum(len(analyses) for analyses in available_results["enrichment_analyses"].values())
            context_parts.append(f"completed {enrichment_count} enrichment analyses")
        
        if "dea_analyses" in available_results:
            context_parts.append(f"performed differential expression analysis on {len(available_results['dea_analyses'])} cell types")
        
        base_response = f"I have {', '.join(context_parts)}. "
        
        # Add available cell types for context
        available_cell_types = self._get_available_cell_types()
        if available_cell_types:
            base_response += f"Available cell types include: {', '.join(available_cell_types[:5])}{'...' if len(available_cell_types) > 5 else ''}. "
        
        if question_context:
            base_response += f"Regarding your question about {question_context}, could you specify which cell type you're interested in? "
        
        base_response += "Which specific analysis results would you like me to interpret?"
        
        return base_response
    
    def _format_cell_type_interpretation(self, cell_type: str, insights: dict, question_context: str) -> str:
        """Format detailed interpretation for a specific cell type"""
        interpretation_parts = [f"🔬 Analysis interpretation for {cell_type}:"]
        
        # Enrichment analysis interpretation
        if insights.get("enrichment_insights"):
            interpretation_parts.append("\n📊 Enrichment Analysis Results:")
            for analysis_type, data in insights["enrichment_insights"].items():
                top_terms = data.get("top_terms", [])
                significant_count = data.get("total_significant", 0)
                
                if top_terms:
                    interpretation_parts.append(f"  • {analysis_type.upper()}: {significant_count} significant terms")
                    interpretation_parts.append(f"    Top pathways: {', '.join(top_terms[:3])}")
                    
                    # Add biological interpretation
                    if analysis_type == "reactome":
                        interpretation_parts.append(f"    These pathways suggest specific biological processes are active in {cell_type}.")
                    elif analysis_type == "go":
                        interpretation_parts.append(f"    These GO terms indicate the molecular functions and biological processes characteristic of {cell_type}.")
                    elif analysis_type == "kegg":
                        interpretation_parts.append(f"    These KEGG pathways show the metabolic and signaling networks involved in {cell_type}.")
        
        # DEA interpretation
        if insights.get("dea_insights"):
            interpretation_parts.append("\n📈 Differential Expression Analysis:")
            for condition, data in insights["dea_insights"].items():
                significant = data.get("significant_genes", 0)
                upregulated = data.get("upregulated", 0)
                downregulated = data.get("downregulated", 0)
                top_genes = data.get("top_genes", [])
                
                interpretation_parts.append(f"  • {condition}: {significant} differentially expressed genes")
                interpretation_parts.append(f"    ({upregulated} upregulated, {downregulated} downregulated)")
                if top_genes:
                    interpretation_parts.append(f"    Top upregulated genes: {', '.join(top_genes[:3])}")
                
                # Add biological context
                if upregulated > downregulated:
                    interpretation_parts.append(f"    This suggests {cell_type} may be more active or stressed in this condition.")
                elif downregulated > upregulated:
                    interpretation_parts.append(f"    This suggests {cell_type} may be less active or in a different state in this condition.")
        
        # Add context-specific interpretation if provided
        if question_context:
            interpretation_parts.append(f"\n🎯 Regarding your question about {question_context}:")
            interpretation_parts.append("Based on the available data, the analysis results suggest specific biological patterns. Would you like me to run additional analyses to explore this further?")
        
        # General summary
        if insights.get("summary"):
            interpretation_parts.append(f"\n📋 Summary: {' | '.join(insights['summary'])}")
        
        return "\n".join(interpretation_parts)
    
    def _get_available_cell_types(self) -> List[str]:
        """Get available cell types from initial annotation and processed results"""
        available_types = []
        
        # Add initial cell types
        if hasattr(self, 'initial_cell_types') and self.initial_cell_types:
            available_types.extend(self.initial_cell_types)
        
        # Add processed cell types from history
        if self.history_manager:
            available_results = self.history_manager.get_available_results()
            if "processed_cell_types" in available_results:
                for ct in available_results["processed_cell_types"]:
                    if ct not in available_types:
                        available_types.append(ct)
        
        # Add cell types from hierarchy manager if available
        if self.hierarchy_manager and hasattr(self.hierarchy_manager, 'get_available_cell_types'):
            try:
                hierarchy_types = self.hierarchy_manager.get_available_cell_types()
                for ct in hierarchy_types:
                    if ct not in available_types:
                        available_types.append(ct)
            except:
                pass  # Ignore errors if method doesn't exist
        
        return available_types

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
                        from .utils import compare_cell_count
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