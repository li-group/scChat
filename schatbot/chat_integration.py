import os
import json
import openai
from .file_utils import clear_directory, find_file_with_extension
from .sc_analysis import generate_umap
from .image_processing import *
from .visualizations import *
# Global variables used in chat integration
function_flag = False
display_flag = False
first_try = True
clear_data = True
conversation_history2 = None

# Use the same function_descriptions you originally provided
function_descriptions = [
    {
        "name": "generate_umap",
        "description": "Used to Generate UMAP for unsupervised clustering for RNA analysis. Generates a UMAP visualization based on the given RNA sequencing data",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "display_dotplot",
        "description": "Displays the dotplot for the sample",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "display_cell_type_composition",
        "description": "Displays the cell type composition for the sample",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_image",
        "description": "Reads and processes an image from the 'media' folder and returns a description of what's in the image.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "process_cells",
        "description": "process cells",
        "parameters": {
            "type": "object",
            "properties": {
                "cell_type": {"type": "string", "description": "the cell type"}
            },
            "required": ["cell_type"],
        },
    },
    {
        "name": "display_umap",
        "description": "displays umap that is NOT annotated. This function should be called whenever the user asks for a umap that is not annotated. In the case that the user does not specify cell type, use overall cells. This function can be called multiple times. This function should not be called when asked to GENERATE umap.",
        "parameters": {
            "type": "object",
            "properties": {
                "cell_type": {"type": "string", "description": "the cell type"}
            },
            "required": ["cell_type"],
        },
    },
    {
        "name": "display_processed_umap",
        "description": "displays umap that IS annotated. This function should be called whenever the user asks for a umap that IS annotated. In the case that the user does not specify cell type, use overall cells. This function can be called multiple times. This function should not be called when asked to GENERATE umap.",
        "parameters": {
            "type": "object",
            "properties": {
                "cell_type": {"type": "string", "description": "the cell type"}
            },
            "required": ["cell_type"],
        },
    },
    {
        "name": "display_cell_population_change",
        "description": "displays cell population change graph",
        "parameters": {
            "type": "object",
            "properties": {
                "cell_type": {"type": "string", "description": "the cell type"}
            },
            "required": ["cell_type"],
        },
    },
    {
        "name": "patient_differential_expression_genes_comparison",
        "description": "Function is designed to perform a differential gene expression analysis for a specified cell type between two patient conditions (pre and post-treatment or two different conditions)",
        "parameters": {
            "type": "object",
            "properties": {
                "cell_type": {"type": "string", "description": "The type of cell to be compared"},
                "patient_1": {"type": "string", "description": "Identifier for the first patient"},
                "patient_2": {"type": "string", "description": "Identifier for the second patient"}
            },
            "required": ["cell_type", "patient_1", "patient_2"],
        },
    },
    {
        "name": "remap_reso",
        "description": "remap_reso",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "calculate_cell_population_change",
        "description": "calculate_cell population change for percentage changes in cell populations or samples before and after treatment. This calculation can be done for any cell type. This is to see the changes in the population before and after treatment.",
        "parameters": {
            "type": "object",
            "properties": {
                "cell_type": {"type": "string", "description": "the cell type"}
            },
            "required": ["cell_type"],
        },
    },
    {
        "name": "gsea_analysis",
        "description": "Performs Gene Set Enrichment Analysis (GSEA) on a dataset of significant genes. This function ranks the genes based on their adjusted p-values and log-fold changes, performs the GSEA analysis using multiple gene set libraries, filters the results for significant terms (FDR ≤ 0.05), and generates several output files including enrichment score plots, a dot plot, and CSV files with the GSEA results and ranked gene list.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "label_clusters",
        "description": "This function can be called multiple times. This function is to label and/or annotate clusters. It can be done for any type of cells that is mentioned by the user. If the user does not mention the cell type, use overall cells. This function can be called multiple times.",
        "parameters": {
            "type": "object",
            "properties": {
                "cell_type": {"type": "string", "description": "the cell type"}
            },
            "required": ["cell_type"],
        },
    },
]

def start_chat2_wrapper(user_input: str) -> str:
    """
    Sends the user input to the OpenAI API using the legacy style,
    then returns the response (or function call result) as a string.
    """
    conversation_history = [{"role": "user", "content": user_input}]
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=conversation_history,
        functions=function_descriptions,
        function_call="auto"
    )
    output = response.choices[0].message
    ai_response = output.content
    if output.function_call:
        function_name = output.function_call.name
        # Assuming no arguments are needed, call the function from globals
        summary = globals()[function_name]()
        if summary:
            conversation_history.append({"role": "user", "content": summary})
            return summary
    return ai_response

def start_chat2() -> None:
    """
    An interactive command-line chat loop.
    """
    conversation_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        conversation_history.append({"role": "user", "content": user_input})
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=conversation_history,
            functions=function_descriptions,
            function_call="auto"
        )
        output = response.choices[0].message
        ai_response = output.content
        print("received ai response:", ai_response)
        if output.function_call:
            print(f"Making a function call to: {output.function_call.name}")
            function_name = output.function_call.name
            print("Assistant: All values returned from", function_name, "have been received")
            summary = globals()[function_name]()
            if summary:
                conversation_history.append({"role": "user", "content": summary})
        else:
            print("Assistant:", ai_response)
            conversation_history.append({"role": "assistant", "content": ai_response})

def start_chat2_web(user_input: str, conversation_history: list):
    """
    Processes user input in a web context by integrating function calls.
    Returns a tuple: (final_response, updated_conversation_history, display_flag).
    """
    global function_flag, display_flag, first_try, clear_data, conversation_history2
    scchat_context1 = " You are a chatbot for helping in Single Cell RNA Analysis. Respond with a greeting."
    base_conversation_history = []
    if first_try and clear_data:
        clear_directory("annotated_adata")
        clear_directory("basic_data")
        clear_directory("umaps")
        clear_directory("process_cell_data")
        research_context_path = find_file_with_extension("media", ".txt")
        if research_context_path:
            with open(research_context_path, "r") as rcptr:
                research_context = rcptr.read()
            conversation_history.append({"role": "user", "content": research_context})
        conversation_history.append({"role": "user", "content": scchat_context1})
        first_try = False
    display_flag = False
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history2 = conversation_history
    openai.api_key = os.getenv("OPENAI_API_KEY")
    base_conversation_history.append({"role": "user", "content": user_input})
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=base_conversation_history,
        functions=function_descriptions,
        function_call="auto",
        temperature=0.2,
        top_p=0.4
    )
    output = response.choices[0].message
    main_flag = False
    if response and output.function_call:
        main_flag = True
    if not main_flag:
        fin_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=conversation_history,
            functions=function_descriptions,
            function_call="auto",
            temperature=0.2,
            top_p=0.4
        )
        output = fin_response.choices[0].message
        ai_response = output.content
    if output.function_call and main_flag:
        function_name = output.function_call.name
        function_args = output.function_call.arguments
        function_flag = True
        try:
            function_response = "Function did not execute."
            print(f"Making a function call to: {function_name}")
            if function_args is None:
                function_response = globals()[function_name]()
            else:
                function_args = json.loads(function_args)
                print(f"Parsed function arguments: {function_args}")
                function_response = globals()[function_name](**function_args)
            if display_flag:
                return function_response, conversation_history, display_flag
            if function_name == "label_clusters":
                conversation_history.append({"role": "assistant", "content": "Annotation is complete."})
                final_response = "Annotation is complete."
            else:
                conversation_history.append({"role": "user", "content": function_response})
                new_response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=conversation_history,
                    temperature=0.2,
                    top_p=0.4
                )
                final_response = new_response.choices[0].message.content if new_response.choices[0] else "Interesting"
                conversation_history.append({"role": "assistant", "content": final_response})
            function_flag = False
            return final_response, conversation_history, display_flag
        except KeyError as e:
            return f"Function {function_name} not found.", conversation_history, display_flag
    else:
        conversation_history.append({"role": "assistant", "content": ai_response})
        return ai_response, conversation_history, display_flag