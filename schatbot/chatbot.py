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
from typing import Dict, Any, Tuple, Optional, List
import openai
import re



class ChatBot:
    def __init__(self):
        # Initialization code remains the same...
        self._initialize_directories()
        self.conversation_history = []
        self.api_key = os.getenv("OPENAI_API_KEY", "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")
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
            # 'schatbot/annotated_adata',  # Keep cached annotations
            'figures', 
            'process_cell_data',
            'schatbot/annotated_adata',
            'schatbot/enrichment/',
            'umaps/annotated',
            'schatbot/runtime_data/basic_data/'
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
        gene_dict, marker_tree, adata, explanation, annotation_result = initial_cell_annotation()
        self.adata = adata
            
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
            # {
            #     "name": "display_dotplot",
            #     "description": """
            #                     Display dotplot for the annotated results.
            #                     This function will be called as the user asked to generate/visualize/display/show the dotplot.
            #     """,
            #     "parameters": {"type": "object", "properties": {}, "required": []},
            # },
            # {
            #     "name": "display_cell_type_composition",
            #     "description": """
            #                     Display cell type composition graph.
            #                     This function will be called as the user asked to generate/visualize/display/show the cell type composition graph.
            #     """,
            #     "parameters": {"type": "object", "properties": {}, "required": []},
            # },
            # {
            #     "name": "display_gsea_dotplot",
            #     "description": """
            #                     Display GSEA dot plot.
            #                     This function will be called as the user asked to generate/visualize/display/show the GSEA dot plot.
            #     """,
            #     "parameters": {"type": "object", "properties": {}, "required": []},
            # },
            # {
            #     "name": "repeat",
            #     "description": "Repeat given sentence",
            #     "parameters": {"type": "object", "properties": {}, "required": []},
            # },
            # {
            #     "name": "display_umap",
            #     "description": """
            #                     Displays UMAP that is NOT annotated with the cell types. 
            #                     Use overall cells if no cell type is specified.
            #                     This function will be called as the user asked to generate/visualize/display/show the UMAP.
            #     """,
            #     "parameters": {
            #         "type": "object",
            #         "properties": {
            #             "cell_type": {
            #                 "type": "string",
            #                 "description": "The cell type"
            #             }
            #         },
            #         "required": ["cell_type"],
            #     },
            # },
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
                        "analysis_type": {
                            "type": "string",
                            "enum": ["GSEA", "GO", "KEGG", "reactome"],
                            "description": "Specific type of enrichment analysis to perform when called from automated workflow."
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
            # {
            #     "name": "display_enrichment_barplot",
            #     "description": """
            #                     Show a barplot of top enriched terms from one of reactome/go/kegg/gsea for a given cell type.
            #                     This function will be called as the user asked to generate/visualize/display/show the enrichment barplot.
            #     """,
            #     "parameters": {
            #         "type": "object",
            #         "properties": {
            #             "analysis": {
            #                 "type": "string",
            #                 "enum": ["reactome","go","kegg","gsea"]
            #             },
            #             "cell_type": {"type": "string"},
            #             "top_n": {"type": "integer", "default": 10},
            #             "domain": {
            #                 "type": "string",
            #                 "enum": ["BP","MF","CC"],
            #                 "description": "Only for GO"
            #             }
            #         },
            #         "required": ["analysis","cell_type"]
            #     }
            # },
            # {
            #     "name": "display_enrichment_dotplot",
            #     "description": """
            #                     Show a dotplot (gene ratio vs. term) of top enriched terms.
            #                     This function will be called as the user asked to generate/visualize/display/show the enrichment dotplot.
            #     """,
            #     "parameters": {
            #         "type": "object",
            #         "properties": {
            #             "analysis": {
            #                 "type": "string",
            #                 "enum": ["reactome","go","kegg","gsea"]
            #             },
            #             "cell_type": {"type": "string"},
            #             "top_n": {"type": "integer", "default": 10},
            #             "domain": {
            #                 "type": "string",
            #                 "enum": ["BP","MF","CC"],
            #                 "description": "Only for GO"
            #             }
            #         },
            #         "required": ["analysis","cell_type"]
            #     }
            # },
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
                        "cell_type": {"type": "string"}
                    },
                    "required": ["cell_type"]
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
        
        # Handle analysis_type parameter from automated workflow
        analyses = kwargs.get("analyses")
        if not analyses and "analysis_type" in kwargs:
            analysis_type = kwargs["analysis_type"].lower()
            if analysis_type == "gsea":
                analyses = ["gsea"]
            elif analysis_type == "go":
                analyses = ["go"]
            elif analysis_type == "kegg":
                analyses = ["kegg"]
            elif analysis_type == "reactome":
                analyses = ["reactome"]
            else:
                analyses = None  # Use default (all analyses)
        
        return perform_enrichment_analyses(
            self.adata,
            cell_type=kwargs.get("cell_type"),
            analyses=analyses,
            logfc_threshold=kwargs.get("logfc_threshold", 1.0),
            pval_threshold=kwargs.get("pval_threshold", 0.05),
            top_n_terms=kwargs.get("top_n_terms", 10),
        )

    def _wrap_process_cells(self, **kwargs):
        """
        FIXED: Wrapper for process cells that ensures annotation results are properly returned
        """
        cell_type = kwargs.get("cell_type")
        resolution = kwargs.get("resolution")
        
        print(f"🔄 Processing cell type: {cell_type}")
        
        # Call the original process_cells function directly (not the wrapper)
        result = process_cells(self.adata, cell_type, resolution)
        
        # Handle the different return types properly
        if isinstance(result, dict) and "status" in result:
            # Handle special status returns
            if result["status"] == "leaf_node":
                return f"✅ '{cell_type}' is a leaf node with no subtypes. This is the most specific level available."
            elif result["status"] == "no_cells_found":
                return f"❌ No cells found with type '{cell_type}' in the dataset."
            elif result["status"] == "insufficient_markers":
                return f"⚠️ Only {len(result.get('available_markers', []))} markers available for '{cell_type}' subtypes. Insufficient for reliable annotation."
        
        elif isinstance(result, str):
            # This is the normal case - annotation results as formatted string
            print(f"✅ Process cells completed successfully")
            print(f"📝 Result preview: {result[:200]}...")
            return result
        
        else:
            # Fallback for unexpected return types
            print(f"⚠️ Unexpected return type from process_cells: {type(result)}")
            return f"Processing of {cell_type} completed with unexpected result format."

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
            kwargs.get("cell_type")
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
        """
        IMPROVED: Handle analysis function results with explicit result tracking
        """
        # Initialize variables
        formatted_result = str(result)
        found_cell_types = []

        # Format the result for conversation history
        if function_name == "perform_enrichment_analyses":
            formatted_result = result.get("formatted_summary", str(result))
        elif function_name == "process_cells":
            cell_type = function_args.get('cell_type', 'Unknown')
            
            if result is None:
                formatted_result = f"Processing of {cell_type} completed with special condition (leaf node, no cells, or insufficient markers)."
            elif isinstance(result, str):
                formatted_result = result
                
                # IMPROVED: Extract and explicitly track annotation results
                annotation_match = re.search(r'Annotation Result:\s*([^•\n]+)', result)
                if annotation_match:
                    annotation_text = annotation_match.group(1).strip()
                    print(f"📝 Extracted annotation: {annotation_text}")
                    
                    # Parse the actual cell types from annotation
                    if "group_to_cell_type" in annotation_text:
                        dict_match = re.search(r"\{([^}]+)\}", annotation_text)
                        if dict_match:
                            dict_content = dict_match.group(1)
                            value_pattern = r":\s*['\"]([^'\"]+)['\"]"
                            cell_types_from_dict = re.findall(value_pattern, dict_content)
                            found_cell_types = list(set(cell_types_from_dict))
                            
                            # Add explicit tracking message
                            tracking_msg = f"ANNOTATION_COMPLETED: {cell_type} → Found cell types: {found_cell_types}"
                            self.conversation_history.append({
                                "role": "system", 
                                "content": tracking_msg
                            })
                            print(f"🎯 Tracking: {tracking_msg}")
            else:
                formatted_result = f"Processing of {cell_type} completed. Result: {str(result)}"
                
        elif function_name == "dea_split_by_condition":
            formatted_result = f"DEA analysis completed for {function_args.get('cell_type')}\n"
            formatted_result += f"Pre-condition genes: {result.get('pre_significant_genes', [])}\n"
            formatted_result += f"Post-condition genes: {result.get('post_significant_genes', [])}"
        
        # Add result to conversation history
        self.conversation_history.append({"role": "assistant", "content": formatted_result})
        
        # Get CONSTRAINED AI interpretation
        interpretation = self._get_ai_interpretation()
        self.conversation_history.append({"role": "assistant", "content": interpretation})
        
        # Prepare the final JSON response
        response_data = {
            "response": interpretation,
            "extracted_cell_types": found_cell_types
        }
        
        # Handle visualization for successful process_cells
        if function_name == "process_cells" and result is not None and isinstance(result, str):
            cell_type = function_args.get("cell_type", "Overall cells")
            try:
                umap_html = display_processed_umap(cell_type=cell_type)
                response_data["graph_html"] = umap_html
            except Exception as e:
                print(f"⚠️ UMAP display failed: {e}")
        
        return json.dumps(response_data)

    def _handle_other_function_result(self, function_name: str, function_args: Dict[str, Any], result: Any) -> str:
        """Handle other function results"""
        formatted_result = str(result)
        self.conversation_history.append({"role": "assistant", "content": formatted_result})
        
        interpretation = self._get_ai_interpretation()
        self.conversation_history.append({"role": "assistant", "content": interpretation})
        
        return interpretation

    def _get_ai_interpretation(self) -> str:
        """
        FIXED: Get AI interpretation constrained to actual results only
        """
        try:
            # Extract the most recent annotation results to constrain the interpretation
            recent_annotations = self._extract_recent_annotation_results()
            
            # Create a constrained prompt that focuses only on actual results
            constraint_prompt = ""
            if recent_annotations:
                constraint_prompt = f"""
IMPORTANT CONSTRAINT: Only discuss the cell types that were ACTUALLY identified in the results: {recent_annotations}
Do NOT mention any cell types from the original question unless they appear in these actual results.
Do NOT assume or infer cell types that are not explicitly listed in the results.
"""
            
            # Build messages with constraints
            messages = self.conversation_history.copy()
            
            # Add constraint as system message if we have recent annotations
            if constraint_prompt:
                messages.append({
                    "role": "system", 
                    "content": constraint_prompt
                })
            
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.1,  # Lower temperature for more factual responses
                top_p=0.3,        # More focused responses
                max_tokens=300    # Limit length to prevent rambling
            )
            
            interpretation = response.choices[0].message.content
            
            # Post-process to validate no hallucinated cell types
            validated_interpretation = self._validate_interpretation_against_results(
                interpretation, recent_annotations
            )
            
            return validated_interpretation
            
        except Exception as e:
            return f"Analysis completed. Error in interpretation: {e}"

    def _extract_recent_annotation_results(self) -> List[str]:
        """Extract cell types from the most recent annotation results"""
        if not self.conversation_history:
            return []
        
        # Look for recent annotation results in conversation history
        recent_cell_types = []
        
        # Check last few messages for annotation dictionaries
        for msg in reversed(self.conversation_history[-5:]):  # Last 5 messages
            content = msg.get("content", "")
            if "group_to_cell_type" in content:
                # Extract from annotation dictionary
                annotation_pattern = r"group_to_cell_type\s*=\s*\{([^}]+)\}"
                match = re.search(annotation_pattern, content)
                if match:
                    dict_content = match.group(1)
                    # Extract values (cell types) from dictionary
                    value_pattern = r":\s*['\"]([^'\"]+)['\"]"
                    cell_types = re.findall(value_pattern, dict_content)
                    recent_cell_types.extend(cell_types)
                    break  # Use most recent annotation
        
        # Remove duplicates and return
        return list(set(recent_cell_types))

    def _validate_interpretation_against_results(self, interpretation: str, actual_results: List[str]) -> str:
        """
        Validate that interpretation doesn't mention cell types not in actual results
        """
        if not actual_results or not interpretation:
            return interpretation
        
        # Convert actual results to lowercase for case-insensitive matching
        actual_lower = [ct.lower() for ct in actual_results]
        
        # Check for common hallucinated cell types from the original question
        problematic_mentions = [
            "regulatory t cell", "conventional memory cd4 t cell", 
            "memory cd4 t cell", "effector t cell", "central memory t cell"
        ]
        
        # Flag if interpretation mentions cell types not in actual results
        interpretation_lower = interpretation.lower()
        hallucinated_types = []
        
        for prob_type in problematic_mentions:
            if prob_type in interpretation_lower:
                # Check if this type is actually in results
                found_in_results = any(prob_type in actual.lower() for actual in actual_lower)
                if not found_in_results:
                    hallucinated_types.append(prob_type)
        
        # If hallucinations detected, provide a corrected interpretation
        if hallucinated_types:
            print(f"⚠️ Detected hallucinated cell types in interpretation: {hallucinated_types}")
            print(f"✅ Actual results: {actual_results}")
            
            corrected_interpretation = f"""
                                        The analysis has been completed and identified the following cell types: {', '.join(actual_results)}.
                                        The annotation successfully distinguished between different cell populations based on their marker expression patterns. 
                                        Further analysis can be performed on any of these identified cell types.
                                        """
            return corrected_interpretation.strip()
        
        return interpretation

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