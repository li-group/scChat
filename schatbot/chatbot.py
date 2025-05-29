from .annotation import initial_cell_annotation, process_cells
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

class ChatBot:
    def __init__(self):
        # Clean all specified folders and files at initialization
        clear_directory('annotated_adata')
        clear_directory('figures')
        clear_directory('process_cell_data')
        clear_directory('schatbot/enrichment/go_bp')
        clear_directory('schatbot/enrichment/go_cc')
        clear_directory('schatbot/enrichment/go_mf')
        clear_directory('schatbot/enrichment/gsea')
        clear_directory('schatbot/enrichment/kegg')
        clear_directory('schatbot/enrichment/reactome')
        # Remove files directly inside schatbot/enrichment (not subfolders)
        enrichment_dir = 'schatbot/enrichment'
        for filename in os.listdir(enrichment_dir):
            file_path = os.path.join(enrichment_dir, filename)
            if os.path.isfile(file_path):
                try:
                    os.unlink(file_path)
                except Exception:
                    pass
        # Clean runtime_data subfolders if they exist
        for subdir in ['runtime_data/basic_data', 'runtime_data/figures', 'runtime_data/process_cell_data']:
            if os.path.exists(subdir):
                clear_directory(subdir)

        self.conversation_history = []
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        self.adata = None 
        gene_dict, marker_tree, adata, explanation, annotation_result = initial_cell_annotation()

        self.conversation_history.append({
           "role": "assistant",
           "content": (
               "Initial annotation complete.\n"
               f"• Annotation Result: {annotation_result}\n"
               f"• Top‐genes per cluster: {gene_dict}\n"
               f"• Marker‐tree: {marker_tree}\n"
               f"• Explanation: {explanation}"
           )
       })

        pth = "annotated_adata/Overall cells_annotated_adata.pkl"
        if os.path.exists(pth):
            with open(pth, "rb") as f:
                self.adata = pickle.load(f)
        else:
            # fallback: regenerate & annotate in memory
            _, _, adata = initial_cell_annotation()
            self.adata = adata

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
                                This function will be called as the user asked to generate/visualize/display the dotplot.
                """,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "display_cell_type_composition",
                "description": """
                                Display cell type composition graph.
                                This function will be called as the user asked to generate/visualize/display the cell type composition graph.
                """,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "display_gsea_dotplot",
                "description": """
                                Display GSEA dot plot.
                                This function will be called as the user asked to generate/visualize/display the GSEA dot plot.
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
                                This function will be called as the user asked to generate/visualize/display the UMAP.
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
                                This function will be called as the user asked to generate/visualize/display the UMAP.
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
                                This function will be called as the user asked to generate/visualize/display the enrichment barplot.
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
                                This function will be called as the user asked to generate/visualize/display the enrichment dotplot.
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

        def _wrap_process_cells(cell_type, resolution=None):
            annotation_str = process_cells(self.adata, cell_type, resolution)
            return annotation_str
        
        def _wrap_dea_split_by_condition(cell_type):
            results = dea_split_by_condition(self.adata, cell_type)
            return results
        
        def _wrap_compare_cell_counts(cell_type, sample1, sample2):
            results = compare_cell_count(self.adata, cell_type, sample1, sample2)
            return results
        
        # Map function names to actual functions
        self.function_mapping = {
            "initial_cell_annotation": initial_cell_annotation,
            "display_dotplot": display_dotplot,
            "display_cell_type_composition": display_cell_type_composition,
            "display_gsea_dotplot": display_gsea_dotplot,
            "repeat": repeat,
            "display_umap": display_umap,
            "display_processed_umap": display_processed_umap,
            "perform_enrichment_analyses": perform_enrichment_analyses,
            "display_enrichment_barplot": display_enrichment_barplot,
            "display_enrichment_dotplot": display_enrichment_dotplot,
            "process_cells": _wrap_process_cells,
            "dea_split_by_condition": _wrap_dea_split_by_condition,
            "compare_cell_counts": _wrap_compare_cell_counts
        }

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
                "content": """
                    You are an expert assistant for single-cell RNA-seq analysis. 
                    Your primary goal is to help users analyze, visualize, and interpret single-cell data by calling the appropriate functions from the available toolkit.

                    Guidelines:
                    1. Function Selection: Carefully read the user's request and select the function that most closely matches the user's intent, based on the function descriptions provided. 
                       If the user asks for a plot or visualization (e.g., dotplot, UMAP, enrichment barplot/dotplot, cell type composition), call the corresponding display function. 
                       If the user requests an analysis (e.g., enrichment, differential expression, cell count comparison), call the relevant analysis function. 
                       If the user asks to annotate, further annotate, further process cells, use the annotation functions.

                    2. Parameter Extraction: Extract all required parameters from the user's message. 
                                             If a parameter is not specified but is required, use sensible defaults as described in the function's documentation. 
                                             For optional parameters, use defaults unless the user specifies otherwise.

                    3. Enrichment Analysis: When the user asks for enrichment (Reactome, GO, KEGG, GSEA), call perform_enrichment_analyses and specify the analyses as needed. 
                                            For enrichment visualizations, use display_enrichment_barplot or display_enrichment_dotplot with the correct analysis type and cell type.

                    4. Visualization: For UMAPs, dotplots, and cell type composition, use the corresponding display function. 
                                      If the user asks for a plot for a specific cell type, ensure the cell_type parameter is set.

                    5. Annotation: For further annotation, use process_cells as appropriate.

                    6. Differential Expression and Cell Counts: For DE analysis split by condition, use dea_split_by_condition. 
                                                                For comparing cell counts between conditions, especially when user asks to compare the cell counts between two conditions, use compare_cell_counts.

                    8. Response Formatting: For visualization functions, return the graph HTML as specified. For analysis functions, return the results in a clear, concise format.

                    Always:
                    - Be precise in mapping user intent to function calls.
                    - Use the function descriptions as your reference for what each function does and what parameters it needs.

                    When a user asks a question, select and call the most appropriate function from the provided toolkit, using the function descriptions and required parameters. 
                    Extract all necessary information from the user's message, use defaults where appropriate, and return results or visualizations as specified. 
                    If the user's request is ambiguous, ask for clarification. Always provide clear, concise answers and call appropriate functions when needed.
                """

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
            function_name = output.function_call.name
            function_args = output.function_call.arguments
            if function_args:
                try:
                    function_args = json.loads(function_args)
                except Exception:
                    function_args = {}
            else:
                function_args = {}
            if function_name in self.function_mapping:
                # SPECIAL‐CASE the enrichment call so we can inject self.adata
                if function_name == "perform_enrichment_analyses":
                    # ensure we have an AnnData loaded
                    if self.adata is None:
                        _, _, self.adata = initial_cell_annotation()
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

                # … then the rest of your if/elif tree handling display vs text‐based results …
                if function_name in ["display_umap", "display_processed_umap", "display_dotplot", "display_cell_type_composition", "display_gsea_dotplot", "display_enrichment_barplot","display_enrichment_dotplot"]:
                    # Do NOT add the visualization result to conversation history.
                    return json.dumps({"response": "", "graph_html": result})
                elif function_name != "initial_cell_annotation" and function_name != "process_cells":
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
                    if function_name == "process_cells":
                        self.conversation_history.append({"role": "assistant", "content": result})
                    cell = function_args.get("cell_type")
                    umap_html = display_processed_umap(cell_type=cell)
                    # we treat it like any other graph return:
                    return json.dumps({"response": "", "graph_html": umap_html})
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


    # def send_message(self, message: str) -> str:
    #     """
    #     1) o4-mini reasoning layer decides: direct-function / rewrite / pass-through
    #     2) If direct-function: we call it immediately & return
    #     3) Otherwise we rewrite (if requested) or use original message
    #     4) Then your existing GPT-4O logic runs as before
    #     """
    #     import json

    #     # ── Stage 1: reasoning layer ──
    #     reasoning = openai.chat.completions.create(
    #         model="o4-mini",
    #         messages=[
    #             {"role":"system","content":"""
    #                 You are a router for single-cell RNA-seq requests.
    #                 Given the raw user message, output a JSON object with exactly one of:
    #                 1) { "function": "<function_name>", "args": { ... } }
    #                    → to bypass GPT and call that function immediately
    #                 2) { "rewrite": "<new user prompt>" }
    #                    → to modify the prompt before GPT handles it
    #                 3) { "pass": true }
    #                    → to let the normal GPT-4O pipeline handle things
    #                 Do NOT include anything else.
    #             """},
    #             {"role":"user","content": message}
    #         ],
    #         # temperature=0
    #     )
    #     print ("reasoning ", reasoning)
    #     try:
    #         decision = json.loads(reasoning.choices[0].message.content)
    #     except Exception:
    #         decision = {"pass": True}

    #     # ── Stage 2: direct function override ──
    #     if decision.get("function"):
    #         fn_name = decision["function"]
    #         fn_args = decision.get("args", {})
    #         if fn_name in self.function_mapping:
    #             result = self.function_mapping[fn_name](**fn_args)
    #             # if it’s a viz, return graph_html
    #             if fn_name in [
    #                 "display_umap", "display_processed_umap",
    #                 "display_dotplot", "display_cell_type_composition",
    #                 "display_gsea_dotplot",
    #                 "display_enrichment_barplot", "display_enrichment_dotplot"
    #             ]:
    #                 return json.dumps({"response":"", "graph_html": result})
    #             return result
    #         else:
    #             return f"Function {fn_name} not found."

    #     # ── Stage 3: rewrite or pass-through ──
    #     user_prompt = decision.get("rewrite", message)

    #     # ── Stage 4: your existing GPT-4O pipeline ──

    #     # ensure system message
    #     if not self.conversation_history:
    #         self.conversation_history.append({
    #             "role":"system","content":"""
    #                 You are an expert assistant for single-cell RNA-seq analysis. 
    #                 Your primary goal is to help users analyze, visualize, and interpret single-cell data by calling the appropriate functions from the available toolkit.

    #                 Guidelines:
    #                 1. Function Selection: Carefully read the user's request and select the function that most closely matches the user's intent...
    #                 ... (rest of your system prompt here) ...
    #             """
    #         })

    #     minimal_history = [
    #         self.conversation_history[0],
    #         {"role":"user","content": user_prompt}
    #     ]

    #     gpt4o_resp = openai.chat.completions.create(
    #         model="gpt-4o",
    #         messages=minimal_history,
    #         functions=self.function_descriptions,
    #         function_call="auto"
    #     )
    #     output = gpt4o_resp.choices[0].message

    #     # ── If GPT-4O wants to call a function ──
    #     if output.function_call:
    #         fn_name = output.function_call.name
    #         try:
    #             fn_args = json.loads(output.function_call.arguments or "{}")
    #         except:
    #             fn_args = {}

    #         # enrichment special‐case
    #         if fn_name == "perform_enrichment_analyses":
    #             if self.adata is None:
    #                 _, _, self.adata = initial_cell_annotation()
    #             result = perform_enrichment_analyses(
    #                 self.adata,
    #                 cell_type       = fn_args.get("cell_type"),
    #                 analyses        = fn_args.get("analyses"),
    #                 logfc_threshold = fn_args.get("logfc_threshold", 1.0),
    #                 pval_threshold  = fn_args.get("pval_threshold", 0.05),
    #                 top_n_terms     = fn_args.get("top_n_terms", 10),
    #             )
    #             self.conversation_history.append({
    #                 "role":"function","name":fn_name,
    #                 "content": json.dumps(result)
    #             })
    #             followup = openai.chat.completions.create(
    #                 model="gpt-4o",
    #                 messages=self.conversation_history,
    #                 temperature=0.2,
    #                 top_p=0.4
    #             )
    #             answer = followup.choices[0].message.content
    #             self.conversation_history.append({"role":"assistant","content":answer})
    #             return answer

    #         # generic dispatch
    #         result = self.function_mapping[fn_name](**fn_args)

    #         # viz vs analysis branching
    #         if fn_name in [
    #             "display_umap","display_processed_umap",
    #             "display_dotplot","display_cell_type_composition",
    #             "display_gsea_dotplot",
    #             "display_enrichment_barplot","display_enrichment_dotplot"
    #         ]:
    #             return json.dumps({"response":"", "graph_html": result})

    #         elif fn_name not in ["initial_cell_annotation","process_cells"]:
    #             self.conversation_history.append({"role":"user","content":message})
    #             self.conversation_history.append({"role":"assistant","content":result})
    #             follow = openai.chat.completions.create(
    #                 model="gpt-4o",
    #                 messages=self.conversation_history,
    #                 temperature=0.2,
    #                 top_p=0.4
    #             )
    #             final = follow.choices[0].message.content
    #             self.conversation_history.append({"role":"assistant","content":final})
    #             return final

    #         else:
    #             # process_cells → show annotated UMAP
    #             cell = fn_args.get("cell_type")
    #             umap_html = display_processed_umap(cell_type=cell)
    #             return json.dumps({"response":"", "graph_html": umap_html})

    #     # ── No function call: full‐history chat ──
    #     else:
    #         self.conversation_history.append({"role":"user","content":message})
    #         self.conversation_history.append({"role":"assistant","content":output.content})

    #         full = openai.chat.completions.create(
    #             model="gpt-4o",
    #             messages=self.conversation_history,
    #             temperature=0.2,
    #             top_p=0.4
    #         )
    #         reply = full.choices[0].message.content
    #         self.conversation_history.append({"role":"assistant","content":reply})
    #         return reply


#     def send_message(self, message: str) -> str:
#         """
#         Uses a reasoning-capable model (o4-mini) to pick & call functions directly.
#         If the model returns a function_call, we dispatch it immediately.
#         Otherwise we just return its content.
#         """
#         import json

#         # 1) Ensure we have your original system message
#         if not self.conversation_history:
#             self.conversation_history.append({
#                 "role": "system",
#                 "content": """
# You are an expert assistant for single-cell RNA-seq analysis. 
# Your primary goal is to help users analyze, visualize, and interpret single-cell data by calling the appropriate functions from the available toolkit.

# When the user asks for a plot or analysis, pick and invoke the matching function with the right parameters. 
# If the model does not emit a function_call, simply return its content as free text.
# """
#             })

#         # 2) Build minimal context
#         minimal = [
#             self.conversation_history[0],
#             {"role": "user", "content": message}
#         ]

#         # 3) Single reasoning / function-calling step:
#         resp = openai.chat.completions.create(
#             model="o4-mini",
#             messages=minimal,
#             functions=self.function_descriptions,
#             function_call="auto"
#         )
#         out = resp.choices[0].message

#         # 4) If it picked a function, dispatch:
#         if out.function_call:
#             name = out.function_call.name
#             raw_args = out.function_call.arguments or "{}"
#             try:
#                 args = json.loads(raw_args)
#             except:
#                 args = {}

#             # special‐case enrichment so we ensure self.adata
#             if name == "perform_enrichment_analyses":
#                 if self.adata is None:
#                     _, _, self.adata = initial_cell_annotation()
#                 result = perform_enrichment_analyses(
#                     self.adata,
#                     cell_type       = args.get("cell_type"),
#                     analyses        = args.get("analyses"),
#                     logfc_threshold = args.get("logfc_threshold", 1.0),
#                     pval_threshold  = args.get("pval_threshold", 0.05),
#                     top_n_terms     = args.get("top_n_terms", 10),
#                 )
#             else:
#                 result = self.function_mapping[name](**args)

#             # if it’s a viz, return graph_html
#             if name in [
#                 "display_umap", "display_processed_umap",
#                 "display_dotplot", "display_cell_type_composition",
#                 "display_gsea_dotplot",
#                 "display_enrichment_barplot", "display_enrichment_dotplot"
#             ]:
#                 return json.dumps({"response": "", "graph_html": result})

#             # otherwise return the raw result
#             return result

#         # 5) No function_call → just return content
#         return out.content