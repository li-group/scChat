from .annotation import initial_cell_annotation, process_cells, handle_process_cells_result
from .visualizations import (
    display_dotplot,
    display_cell_type_composition,
    display_gsea_dotplot,
    display_umap,
    display_processed_umap,
    display_enrichment_barplot,
    display_enrichment_dotplot
)
from .enrichment import perform_enrichment_analyses
from .utils import (
    clear_directory,
    dea_split_by_condition,
    compare_cell_count,
    repeat
)
import os
import json
import openai
import pickle
from typing import Dict, Any, Tuple, Optional
import openai



class ChatBot:
    def __init__(self):
        # Initialization code remains the same...
        self._initialize_directories()
        self.conversation_history = []
        self.api_key = os.getenv("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")
        openai.api_key = self.api_key
        self.adata = None
        
        # Load or create initial annotation
        self._initialize_annotation()
        
        # Define function categories for consistent handling
        self.visualization_functions = {
            "display_dotplot", "display_cell_type_composition", "display_gsea_dotplot",
            "display_umap", "display_processed_umap", "display_enrichment_barplot",
            "display_enrichment_dotplot"
        }
        
        self.analysis_functions = {
            "perform_enrichment_analyses", "process_cells", "dea_split_by_condition",
            "compare_cell_counts"
        }
        
        self.setup_functions()

    def _initialize_directories(self):
        """Clean all directories at initialization"""
        directories_to_clear = [
            'annotated_adata', 'figures', 'process_cell_data',
            'schatbot/enrichment/go_bp', 'schatbot/enrichment/go_cc',
            'schatbot/enrichment/go_mf', 'schatbot/enrichment/gsea',
            'schatbot/enrichment/kegg', 'schatbot/enrichment/reactome',
            'umaps/annotated'
        ]
        
        for directory in directories_to_clear:
            clear_directory(directory)
        
        # Clean enrichment files
        enrichment_dir = 'schatbot/enrichment'
        for filename in os.listdir(enrichment_dir):
            file_path = os.path.join(enrichment_dir, filename)
            if os.path.isfile(file_path):
                try:
                    os.unlink(file_path)
                except Exception:
                    pass

    def _initialize_annotation(self):
        """Initialize or load annotation data"""
        pth = "annotated_adata/Overall cells_annotated_adata.pkl"
        if os.path.exists(pth):
            with open(pth, "rb") as f:
                self.adata = pickle.load(f)
                print("Loaded annotated adata from pickle file")
            gene_dict, marker_tree, _, explanation, annotation_result = initial_cell_annotation()
        else:
            gene_dict, marker_tree, adata, explanation, annotation_result = initial_cell_annotation()
            self.adata = adata
            
        # Add initial annotation to conversation history
        self._add_initial_annotation_to_history(gene_dict, marker_tree, explanation, annotation_result)

    def _add_initial_annotation_to_history(self, gene_dict, marker_tree, explanation, annotation_result):
        """Add initial annotation results to conversation history"""
        initial_content = (
            "Initial annotation complete.\n"
            f"• Annotation Result: {annotation_result}\n"
            f"• Top‐genes per cluster: {gene_dict}\n"
            f"• Marker‐tree: {marker_tree}\n"
            f"• Explanation: {explanation}"
        )
        self.conversation_history.append({
            "role": "assistant",
            "content": initial_content
        })

    def setup_functions(self):
        """Setup function descriptions and mappings"""
        # Function descriptions remain the same...
        self.function_descriptions = [
            {
                "name": "initial_cell_annotation",
                "description": """
                                Do the initial cell type annotations. 
                                Returns three values: (1) gene dictionary mapping clusters to their top genes, 
                                                      (2) marker tree containing cell type markers, and 
                                                      (3) annotated AnnData object with cell type labels.
                                This function will be called only once.
                """,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "display_dotplot",
                "description": """
                                Display dotplot for the annotated results.
                                This function will be called as the user asked to generate/visualize/display/show the dotplot.
                """,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "display_cell_type_composition",
                "description": """
                                Display cell type composition graph.
                                This function will be called as the user asked to generate/visualize/display/show the cell type composition graph.
                """,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "display_gsea_dotplot",
                "description": """
                                Display GSEA dot plot.
                                This function will be called as the user asked to generate/visualize/display/show the GSEA dot plot.
                """,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "repeat",
                "description": "Repeat given sentence",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "display_umap",
                "description": """
                                Displays UMAP that is NOT annotated with the cell types. 
                                Use overall cells if no cell type is specified.
                                This function will be called as the user asked to generate/visualize/display/show the UMAP.
                """,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell_type": {
                            "type": "string",
                            "description": "The cell type"
                        }
                    },
                    "required": ["cell_type"],
                },
            },
            {
                "name": "display_processed_umap",
                "description": """
                                Displays UMAP that IS annotated with the cell types. 
                                Use overall cells if no cell type is specified.
                                This function will be called as the user asked to generate/visualize/display/show the UMAP.
                """,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell_type": {
                            "type": "string",
                            "description": "The cell type"
                        }
                    },
                    "required": ["cell_type"],
                },
            },
            {
                "name": "perform_enrichment_analyses",
                "description": """
                                Run one or more enrichment analyses (reactome, go, kegg, gsea) on the DE genes for a given cell type.
                                This function will be called as the user asked to generate/visualize/display the enrichment analyses.
                """,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell_type": {
                            "type": "string",
                            "description": "The cell type to analyze."
                        },
                        "analyses": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["reactome", "go", "kegg", "gsea"]
                            },
                            "description": "Which enrichment analyses to run. Defaults to all if omitted."
                        },
                        "logfc_threshold": {
                            "type": "number",
                            "description": "Minimum absolute log2 fold change to call a gene significant."
                        },
                        "pval_threshold": {
                            "type": "number",
                            "description": "Adjusted p‑value cutoff for significant genes."
                        },
                        "top_n_terms": {
                            "type": "integer",
                            "description": "How many top enriched terms to return/plot."
                        }
                    },
                    "required": ["cell_type"]
                }
            },
            {
                "name": "process_cells",
                "description": """
                                Recluster the cells based on the given cell type.
                                Further annotate the cells based on the top marker genes.
                                Returns three values: (1) gene dictionary mapping clusters to their top genes, 
                                                      (2) marker tree containing cell type markers, and 
                                                      (3) annotated AnnData object with cell type labels.
                                This function will be called whenever the user asks to further annotate a specific cell type.
                """,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell_type": { "type": "string", "description": "The cell type to process (e.g. 'T cells')." }
                    },
                    "required": ["cell_type"]
                }
            },
            {
                "name": "display_enrichment_barplot",
                "description": """
                                Show a barplot of top enriched terms from one of reactome/go/kegg/gsea for a given cell type.
                                This function will be called as the user asked to generate/visualize/display/show the enrichment barplot.
                """,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "analysis": {
                            "type": "string",
                            "enum": ["reactome","go","kegg","gsea"]
                        },
                        "cell_type": {"type": "string"},
                        "top_n": {"type": "integer", "default": 10},
                        "domain": {
                            "type": "string",
                            "enum": ["BP","MF","CC"],
                            "description": "Only for GO"
                        }
                    },
                    "required": ["analysis","cell_type"]
                }
            },
            {
                "name": "display_enrichment_dotplot",
                "description": """
                                Show a dotplot (gene ratio vs. term) of top enriched terms.
                                This function will be called as the user asked to generate/visualize/display/show the enrichment dotplot.
                """,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "analysis": {
                            "type": "string",
                            "enum": ["reactome","go","kegg","gsea"]
                        },
                        "cell_type": {"type": "string"},
                        "top_n": {"type": "integer", "default": 10},
                        "domain": {
                            "type": "string",
                            "enum": ["BP","MF","CC"],
                            "description": "Only for GO"
                        }
                    },
                    "required": ["analysis","cell_type"]
                }
            },
            {
                "name": "dea_split_by_condition",
                "description": """
                                Perform differential expression analysis split by condition.
                                Specifical when the dataset itself has the different conditions in the metadata ex. (p1_pre, p1_post, p2_pre, p2_post), and the user asks to do the differentail expression genes analysis.                              
                                You will have to call this function.
                """,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell_type": {"type": "string"}
                        },
                    "required": ["cell_type"]
                }
            },
            {
                "name": "compare_cell_counts",
                "description": """
                                Compare cell counts between two conditions.
                                You will have to specify the cell type and the two conditions ex. (p1_pre, p1_post) or (p2_pre, p1_pre).
                                If user don't know how to specify the conditions, you can show them the example.
                                This function will be called as the user asked to compare the cell counts between two conditions.
                """,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell_type": {"type": "string"},
                        "sample1": {"type": "string"},
                        "sample2": {"type": "string"}
                    },
                    "required": ["cell_type", "sample1", "sample2"]
                }
            }
        ]
        
        # Create function mappings with proper wrappers
        self.function_mapping = {
            "initial_cell_annotation": self._wrap_initial_cell_annotation,
            "display_dotplot": self._wrap_visualization(display_dotplot),
            "display_cell_type_composition": self._wrap_visualization(display_cell_type_composition),
            "display_gsea_dotplot": self._wrap_visualization(display_gsea_dotplot),
            "display_umap": self._wrap_visualization(display_umap),
            "display_processed_umap": self._wrap_visualization(display_processed_umap),
            "display_enrichment_barplot": self._wrap_visualization(display_enrichment_barplot),
            "display_enrichment_dotplot": self._wrap_visualization(display_enrichment_dotplot),
            "perform_enrichment_analyses": self._wrap_enrichment_analysis,
            "process_cells": self._wrap_process_cells,
            "dea_split_by_condition": self._wrap_dea_analysis,
            "compare_cell_counts": self._wrap_compare_cells,
            "repeat": repeat
        }

    def _wrap_initial_cell_annotation(self, **kwargs):
        """Wrapper for initial cell annotation"""
        return initial_cell_annotation()

    def _wrap_visualization(self, func):
        """Generic wrapper for visualization functions"""
        def wrapper(**kwargs):
            return func(**kwargs)
        return wrapper

    def _wrap_enrichment_analysis(self, **kwargs):
        """Wrapper for enrichment analysis with proper parameter handling"""
        if self.adata is None:
            _, _, self.adata = initial_cell_annotation()
        
        return perform_enrichment_analyses(
            self.adata,
            cell_type=kwargs.get("cell_type"),
            analyses=kwargs.get("analyses"),
            logfc_threshold=kwargs.get("logfc_threshold", 1.0),
            pval_threshold=kwargs.get("pval_threshold", 0.05),
            top_n_terms=kwargs.get("top_n_terms", 10),
        )

    def _wrap_process_cells(self, **kwargs):
        """Wrapper for process cells using the new handle_process_cells_result function"""
        cell_type = kwargs.get("cell_type")
        resolution = kwargs.get("resolution")
        return handle_process_cells_result(self.adata, cell_type, resolution)

    def _wrap_dea_analysis(self, **kwargs):
        """Wrapper for differential expression analysis"""
        cell_type = kwargs.get("cell_type")
        adata_pre, adata_post, pre_significant_genes, post_significant_genes = dea_split_by_condition(self.adata, cell_type)
        
        return {
            "summary": f"DEA split by condition for {cell_type} complete.",
            "pre_significant_genes": pre_significant_genes,
            "post_significant_genes": post_significant_genes,
            "adata_pre": adata_pre,
            "adata_post": adata_post
        }

    def _wrap_compare_cells(self, **kwargs):
        """Wrapper for cell count comparison"""
        return compare_cell_count(
            self.adata,
            kwargs.get("cell_type"),
            kwargs.get("sample1"),
            kwargs.get("sample2")
        )

    def _ensure_system_message(self):
        """Ensure system message is in conversation history"""
        if not self.conversation_history or self.conversation_history[0]["role"] != "system":
            system_message = {
                "role": "system",
                "content": """
                You are an expert assistant for single-cell RNA-seq analysis. 
                Your primary goal is to help users analyze, visualize, and interpret single-cell data by calling the appropriate functions from the available toolkit.

                Guidelines:
                1. Function Selection: Carefully read the user's request and select the function that most closely matches the user's intent.
                2. Parameter Extraction: Extract all required parameters from the user's message.
                3. Always provide clear, concise answers and call appropriate functions when needed.
                
                Remember all previous analyses and their results to provide contextual responses.
                """
            }
            self.conversation_history.insert(0, system_message)

    def send_message(self, message: str) -> str:
        """
        Optimized message handling with consistent function calling and history management
        """
        self._ensure_system_message()
        
        # Step 1: Try to get a function call with current context
        function_call_result = self._attempt_function_call(message)
        
        if function_call_result:
            function_name, function_args, result = function_call_result
            return self._handle_function_result(message, function_name, function_args, result)
        else:
            # Step 2: Handle as regular conversation
            return self._handle_regular_conversation(message)

    def _attempt_function_call(self, message: str) -> Optional[Tuple[str, Dict[str, Any], Any]]:
        """Attempt to get a function call from the user message"""
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=self.conversation_history + [{"role": "user", "content": message}],
                functions=self.function_descriptions,
                function_call="auto",
                temperature=0.1
            )
            
            output = response.choices[0].message
            
            if output.function_call:
                function_name = output.function_call.name
                function_args = json.loads(output.function_call.arguments) if output.function_call.arguments else {}
                
                if function_name in self.function_mapping:
                    result = self.function_mapping[function_name](**function_args)
                    return function_name, function_args, result
                    
        except Exception as e:
            print(f"Function call error: {e}")
            
        return None

    def _handle_function_result(self, message: str, function_name: str, function_args: Dict[str, Any], result: Any) -> str:
        """Handle function results consistently"""
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Handle different function types
        if function_name in self.visualization_functions:
            return self._handle_visualization_result(function_name, function_args, result)
        elif function_name in self.analysis_functions:
            return self._handle_analysis_result(function_name, function_args, result)
        else:
            return self._handle_other_function_result(function_name, function_args, result)

    def _handle_visualization_result(self, function_name: str, function_args: Dict[str, Any], result: Any) -> str:
        """Handle visualization function results"""
        # For visualizations, we don't add the HTML to conversation history
        # but we do record that the visualization was created
        viz_summary = f"Created {function_name} visualization"
        if "cell_type" in function_args:
            viz_summary += f" for {function_args['cell_type']}"
        
        self.conversation_history.append({"role": "assistant", "content": viz_summary})
        
        return json.dumps({"response": viz_summary, "graph_html": result})

    def _handle_analysis_result(self, function_name: str, function_args: Dict[str, Any], result: Any) -> str:
        """Handle analysis function results"""
        # Format the result for conversation history
        if function_name == "perform_enrichment_analyses":
            formatted_result = result.get("formatted_summary", str(result))
        elif function_name == "process_cells":
            # Handle the new return format from handle_process_cells_result
            if result is None:
                # This means one of the early exit conditions was met (leaf node, no cells, insufficient markers)
                formatted_result = f"Processing of {function_args.get('cell_type')} completed with special condition."
            else:
                formatted_result = result
        elif function_name == "dea_split_by_condition":
            formatted_result = f"DEA analysis completed for {function_args.get('cell_type')}\n"
            formatted_result += f"Pre-condition genes: {result.get('pre_significant_genes', [])}\n"
            formatted_result += f"Post-condition genes: {result.get('post_significant_genes', [])}"
        else:
            formatted_result = str(result)
        
        # Add result to conversation history
        self.conversation_history.append({"role": "assistant", "content": formatted_result})
        
        # Get AI interpretation of the results
        interpretation = self._get_ai_interpretation()
        self.conversation_history.append({"role": "assistant", "content": interpretation})
        
        # Handle special cases that need visualization
        if function_name == "process_cells" and result is not None:
            # Only try to display UMAP if we have a successful result
            cell_type = function_args.get("cell_type", "Overall cells")
            umap_html = display_processed_umap(cell_type=cell_type)
            return json.dumps({"response": interpretation, "graph_html": umap_html})
        
        return interpretation

    def _handle_other_function_result(self, function_name: str, function_args: Dict[str, Any], result: Any) -> str:
        """Handle other function results"""
        formatted_result = str(result)
        self.conversation_history.append({"role": "assistant", "content": formatted_result})
        
        interpretation = self._get_ai_interpretation()
        self.conversation_history.append({"role": "assistant", "content": interpretation})
        
        return interpretation

    def _get_ai_interpretation(self) -> str:
        """Get AI interpretation of the current conversation state"""
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=self.conversation_history,
                temperature=0.2,
                top_p=0.4
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Analysis completed. Error in interpretation: {e}"

    def _handle_regular_conversation(self, message: str) -> str:
        """Handle regular conversation without function calls"""
        self.conversation_history.append({"role": "user", "content": message})
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=self.conversation_history,
                temperature=0.2,
                top_p=0.4
            )
            
            ai_response = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            return ai_response
            
        except Exception as e:
            error_response = f"Sorry, I encountered an error: {e}"
            self.conversation_history.append({"role": "assistant", "content": error_response})
            return error_response

    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation for debugging"""
        summary = f"Conversation has {len(self.conversation_history)} messages:\n"
        for i, msg in enumerate(self.conversation_history):
            summary += f"{i}: {msg['role']} - {msg['content'][:100]}...\n"
        return summary

    def _chat_only(self, user_message: str) -> str:
        """Ask gpt-4o directly, with no function schemas."""
        resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                self.conversation_history[0],
                {"role": "user", "content": user_message}
            ]
        )
        return resp.choices[0].message.content 