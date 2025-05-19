import os
import json
import openai
from .sc_analysis import generate_umap
from .visualizations import (
    display_dotplot,
    display_cell_type_composition,
    display_gsea_dotplot
)
from .image_processing import read_image
from .differential_expression import sample_differential_expression_genes_comparison
from .cluster_labeling import label_clusters
from .file_utils import clear_directory, find_file_with_extension
from .sc_analysis import *
from .visualizations import *
class ChatBot:
    def __init__(self):
        self.conversation_history = []
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        self.adata = None 
        gene_dict, marker_tree, adata = generate_umap()
        # self.adata = adata
        pth = "annotated_adata/Overall cells_annotated_adata.pkl"
        if os.path.exists(pth):
            with open(pth, "rb") as f:
                self.adata = pickle.load(f)
        else:
            # fallback: regenerate & annotate in memory
            _, _, adata = generate_umap()
            self.adata = adata

        self.function_descriptions = [
            {
                "name": "generate_umap",
                "description": "Generate UMAP for RNA analysis.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "display_dotplot",
                "description": "Display dotplot for the sample.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "display_cell_type_composition",
                "description": "Display cell type composition graph.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "display_gsea_dotplot",
                "description": "Display GSEA dot plot.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "read_image",
                "description": "Process an image and return a description.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "repeater",
                "description": "repeat given sentence",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "sample_differential_expression_genes_comparison",
                "description": "Compare differential expression between two samples for a given cell type.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell_type": {"type": "string", "description": "Cell type"},
                        "sample_1": {"type": "string", "description": "First patient"},
                        "sample_2": {"type": "string", "description": "Second patient"},
                    },
                    "required": ["cell_type", "sample_1", "sample_2"],
                },
            },
            {
                "name": "label_clusters",
                "description": "Annotate clusters for a given cell type.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell_type": {"type": "string", "description": "Cell type"}
                    },
                    "required": ["cell_type"],
                },
            },
            {
                "name": "display_umap",
                "description": "Displays UMAP that is NOT annotated. Use overall cells if no cell type is specified.",
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
                "description": "Displays UMAP that IS annotated. Use overall cells if no cell type is specified.",
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
                "description": "Run one or more enrichment analyses (reactome, go, kegg, gsea) on the DE genes for a given cell type.",
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
                "description": "Process a specific cell type: recluster, rank genes, save UMAP and dot-plot data, and return top marker genes.",
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
            "description": "Show a barplot of top enriched terms from one of reactome/go/kegg/gsea for a given cell type.",
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
            "description": "Show a dotplot (avg log2FC vs. term size) of top enriched terms.",
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
        }
        ]

        def _wrap_process_cells(cell_type, resolution=None):
            print ("Here now", cell_type)
            # call your standalone function
            annotation_str = process_cells(self.adata, cell_type, resolution)
            print ("ANNOTATION STR", annotation_str)
            return annotation_str
        
        # Map function names to actual functions
        self.function_mapping = {
            "generate_umap": generate_umap,
            "display_dotplot": display_dotplot,
            "display_cell_type_composition": display_cell_type_composition,
            "display_gsea_dotplot": display_gsea_dotplot,
            "read_image": read_image,
            "sample_differential_expression_genes_comparison": sample_differential_expression_genes_comparison,
            "label_clusters": label_clusters,
            "repeat": repeat,
            "display_umap": display_umap,
            "display_processed_umap": display_processed_umap,
            "perform_enrichment_analyses": perform_enrichment_analyses,
            # "display_reactome_barplot": display_reactome_barplot,
            "display_enrichment_barplot": display_enrichment_barplot,
            "display_enrichment_dotplot": display_enrichment_dotplot,
            # "process_cells":process_cells
            # "process_cells": lambda cell_type, resolution=None: process_cells(
            #     self.adata, cell_type, resolution
            # ),
            # "process_cells": lambda cell_type, resolution=None: (
            #     process_cells(cell_type, resolution)
            # ),
            "process_cells": _wrap_process_cells

        }

    def send_message(self, message: str) -> str:
        """
        Uses minimal context (system message and current user message) to call ChatGPT.
        If a function call is returned, process it immediately without adding it to
        the conversation history. Otherwise, append the exchange to the conversation
        history and reprompt using the full history for a final response.
        """
        # Ensure we have a system message in our full conversation history.
        if not self.conversation_history:
            system_message = {
                "role": "system",
                "content": ("You are a chatbot specialized in Single Cell RNA Analysis. "
                            "Provide clear, concise answers and call appropriate functions when needed.")
            }
            self.conversation_history.append(system_message)

        # Build a minimal conversation with just the system message and the current user message.
        minimal_history = [self.conversation_history[0], {"role": "user", "content": message}]
        
        # Call ChatGPT with minimal history
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=minimal_history,
            functions=self.function_descriptions,
            function_call="auto"
        )
        output = response.choices[0].message
        ai_response = output.content

        # If a function call is produced with minimal context, process it.
        if output.function_call:
            print ("OUTPUT ", output)
            function_name = output.function_call.name
            function_args = output.function_call.arguments
            if function_args:
                try:
                    function_args = json.loads(function_args)
                except Exception:
                    function_args = {}
            else:
                function_args = {}
            # if function_name in self.function_mapping:
            #     result = self.function_mapping[function_name](**function_args)
            #     print("Made a function call to", function_name)
            if function_name in self.function_mapping:
                # SPECIAL‐CASE the enrichment call so we can inject self.adata
                if function_name == "perform_enrichment_analyses":
                    # ensure we have an AnnData loaded
                    if self.adata is None:
                        _, _, self.adata = generate_umap()
                    result = perform_enrichment_analyses(
                    self.adata,
                    cell_type       = function_args.get("cell_type"),
                    analyses        = function_args.get("analyses"),
                    logfc_threshold = function_args.get("logfc_threshold", 1.0),
                    pval_threshold  = function_args.get("pval_threshold", 0.05),
                    top_n_terms     = function_args.get("top_n_terms", 10),
                )

                    # 1) Push the function response into the convo history
                    self.conversation_history.append({"role": "function",
                                                    "name": function_name,
                                                    "content": json.dumps(result)})

                    # 2) Now ask ChatGPT to analyze that data:
                    followup = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=self.conversation_history,
                        temperature=0.2,
                        top_p=0.4
                    )
                    answer = followup.choices[0].message.content
                    self.conversation_history.append({"role": "assistant",
                                                    "content": answer})
                    return answer
                else:
                    # your existing generic dispatch
                    result = self.function_mapping[function_name](**function_args)

                print("Made a function call to", function_name)
                # … then the rest of your if/elif tree handling display vs text‐based results …
                if function_name in ["display_umap", "display_processed_umap", "display_dotplot", "display_cell_type_composition", "display_gsea_dotplot", "display_enrichment_barplot","display_enrichment_dotplot"]:
                    # Do NOT add the visualization result to conversation history.
                    return json.dumps({"response": "", "graph_html": result})
                elif function_name != "generate_umap" and function_name != "process_cells":
                    self.conversation_history.append({"role": "user", "content": message})
                    self.conversation_history.append({"role": "assistant", "content": result})
                    new_response = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=self.conversation_history,
                        temperature=0.2,
                        top_p=0.4
                    )
                    final_response = new_response.choices[0].message.content
                    self.conversation_history.append({"role": "assistant", "content": final_response})
                    return final_response
                else:
                    final_response = "Annotation is complete."
                    self.conversation_history.append({"role": "assistant", "content": result})
                    self.conversation_history.append({"role": "assistant", "content": final_response})
                    return final_response
            else:
                return f"Function {function_name} not found."
        else:
            # No function call was returned from the minimal context.
            # Append the minimal exchange (user + assistant text) to the full conversation history.
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            # Now reprompt with full conversation history.
            new_response = openai.chat.completions.create(
                model="gpt-4o",
                messages=self.conversation_history,
                temperature=0.2,
                top_p=0.4
            )
            final_response = new_response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": final_response})
            return final_response