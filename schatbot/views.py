
from django.http import JsonResponse
import json
from django.http import JsonResponse
from django.shortcuts import render
from .forms import MyForm
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import openai
from django.core.files.storage import default_storage

from django.http import HttpResponse
from .new_function_calling import generate_umap

from .chat import ask  
from .new_function_calling import start_chat2
from .new_function_calling import start_chat2_web
from .new_function_calling import display_dotplot
from .new_function_calling import display_cell_type_composition
from .new_function_calling import convert_into_labels
from .new_function_calling import display_annotated_umap
from .new_function_calling import sample_differential_expression_genes_comparison
from .new_function_calling import generate_umap
from .new_function_calling import display_gsea_dotplot
from .gemini import browse_web

from django.shortcuts import render, redirect
from .forms import UploadFileForm
from .models import FileUpload


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def index(request):
    print("my view is called")
    return render(request, 'schatbot/index.html')

def file_upload(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('success_url')  # Redirect after POST
    else:
        form = UploadFileForm()
    return render(request, 'myapp/upload.html', {'form': form})


@csrf_exempt  # Temporarily disable CSRF token for testing, ensure you handle CSRF properly in production
@require_http_methods(["POST"])
def upload_file(request):
    if request.FILES:
        uploaded_file = request.FILES['file']
        file_path = default_storage.save(uploaded_file.name, uploaded_file)
        print("File saved to:", file_path)
        
        # Optionally, store the file path in the session if needed later
        request.session['uploaded_file_path'] = file_path
        
        return JsonResponse({
            'status': 'success',
            'message': f'File {uploaded_file.name} uploaded successfully.'
        })
    else:
        return JsonResponse({
            'status': 'error',
            'message': 'No file was uploaded.'
        }, status=400)



# Check need to search on for internet
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def classify_intent(user_message):
    search_keywords = {'search', 'find', 'lookup', 'google', 'web search'}
    words = set(word_tokenize(user_message.lower()))
    if words & search_keywords:
        return 'web_search'
    return 'chat'

def web_search(query):
    # Implement the web search functionality here.
    return "Web search results for: " + query

def umap_leiden_view(request):
    summary, json_data = generate_umap()
    plot_html = display_umap_leiden(json_data)
    return HttpResponse(plot_html)



function_descriptions = [
    {
        "name": "generate_umap",
        "description": "Used to Generate UMAP for unsupervised clustering for RNA analysis. Generates a UMAP visualization based on the given RNA sequencing data",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    
    {
        "name": "display_dotplot",
        "description": "Displays the dotplot for the sample",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "display_cell_type_composition",
        "description": "Displays the cell type composition for the sample",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    
    {
        "name": "display_myeloid_umap",
        "description": "displays myeloid umap",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    
    {
        "name": "read_image",
        "description": "Reads and processes an image from the 'media' folder and returns a description of what's in the image.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "process_cells",
        "description": "process cells",
        "parameters": {
            "type": "object",
            "properties": 
                {"cell_type":{
                    "type":"string","description":"the cell type"}
                 },
            "required": ["cell_type"],
        },
    },
    {
        "name": "display_umap",
        "description": "displays umap that is NOT annotated. This function should be called whenever the user asks for a umap that is not annotated. In the case that the user does not specify cell type, use overall cells. This function can be called multiple times. This function should not be called when asked to GENERATE umap.",
        "parameters": {
            "type": "object",
            "properties":
                {"cell_type":{
                    "type":"string","description":"the cell type"}
                 },
            "required": ["cell_type"],
        },
    },
    {
        "name": "display_annotated_umap",
        "description": "displays umap that IS annotated. This function should be called whenever the user asks for a umap that IS annotated. In the case that the user does not specify cell type, use overall cells. This function can be called multiple times. This function should not be called when asked to GENERATE umap.",
        "parameters": {
            "type": "object",
            "properties":
                {"cell_type":{
                    "type":"string","description":"the cell type"}
                 },
            "required": ["cell_type"],
        },
    },
    {
        "name": "display_cell_population_change",
        "description": "displays cell population change graph",
        "parameters": {
            "type": "object",
            "properties": 
                {"cell_type":{
                    "type":"string","description":"the cell type"}
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
            "cell_type": {
                "type": "string",
                "description": "The type of cell to be compared"
            },
            "patient_1": {
                "type": "string",
                "description": "Identifier for the first patient"
            },
            "patient_2": {
                "type": "string",
                "description": "Identifier for the second patient"
            }
        },
        "required": ["cell_type", "patient_1", "patient_2"]
    }
    },
    {
        "name": "remap_reso",
        "description": "remap_reso",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "calculate_cell_population_change",
        "description": "calculate_cell population change for percentage changes in cell populations or samples before and after treatment. This calculation can be done for any cell type. This is to see the changes in the population before and after treatment.",
        "parameters": {
            "type": "object",
            "properties": 
                {"cell_type":{
                    "type":"string","description":"the cell type"}
                 },
            "required": ["cell_type"],
        },
    },
    {
        "name": "gsea_analysis",
        "description": "Performs Gene Set Enrichment Analysis (GSEA) on a dataset of significant genes. This function ranks the genes based on their adjusted p-values and log-fold changes, performs the GSEA analysis using multiple gene set libraries, filters the results for significant terms (FDR ≤ 0.05), and generates several output files including enrichment score plots, a dot plot, and CSV files with the GSEA results and ranked gene list.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "label_clusters",
        "description": "This function can be called multiple times. this function is to label and or annotate clusters. It can be done for any type of cells that is mentioned by the user. If the user does not mention the cell type use overall cells. This function can be called multiple times.",
        "parameters": {
            "type": "object",
            "properties": 
                {"cell_type":{
                    "type":"string","description":"the cell type"}
                 },
            "required": ["cell_type"],
        },
    },
]
def start_chat2_wrapper(user_input):
    conversation_history = [{"role": "user", "content": user_input}]

    # Generate a chat response with potential function invocation
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=conversation_history,
        functions=function_descriptions,
        function_call="auto"  # Enable function calling automatically
    )

    output = response.choices[0].message
    ai_response = output.content

    if output.function_call:
        function_name = output.function_call.name
        summary = globals()[function_name]()  # Assuming no arguments are needed
        if summary:
            conversation_history.append({"role": "user", "content": summary})
            return summary

    return ai_response


@csrf_exempt  # Consider CSRF implications depending on your deployment
@require_http_methods(["POST"])  # Ensure only POST requests are handled
def chat_with_ai(request):
    try:
        data = json.loads(request.body)
        user_message = data.get('message', '')
        print(f"Received message: {user_message}")

        # Retrieve conversation history from the session
        conversation_history = request.session.get('conversation_history', [])
        # print(f"Loaded conversation history: {conversation_history}")

        # Web search if asked by user
        if classify_intent(user_message) == 'web_search':
            response = browse_web(user_message)
            print(f"Web search response: {response}")

        
        elif "display dotplot" in user_message.lower():
            graph_json = display_dotplot()
            response = 'displaying dotplot'
            return JsonResponse({"response": response, "graph_json": graph_json})

        elif "display cell type composition" in user_message.lower():
            graph_json = display_cell_type_composition()
            response = 'displaying cell type composition'
            return JsonResponse({"response": response, "graph_json": graph_json})
        

        
        elif "display gsea plot" in user_message.lower():
            graph_json = display_gsea_dotplot()
            response = 'displaying gsea plot'
            return JsonResponse({"response": response, "graph_json": graph_json})
        
    

        elif "set resolution" in user_message.lower():
            print ("setting resolution")
            graph_json = remap_reso(user_message)
            response = 'setting resolution'
            return JsonResponse({"response": response, "graph_json": graph_json})
        elif "sample_mapping" in user_message.lower():
            print ("setting sample mapping")
            graph_json = set_map(user_message)
            response = 'setting sample mapping'
            return JsonResponse({"response": response, "graph_json": graph_json})
        
        else:
            response, conversation_history, display_flag = start_chat2_web(user_message, conversation_history)
            if display_flag == True:
                return JsonResponse({"response": "Displaying", "graph_json": response})
            else:
                print(f"Chat response: {response}")
        
        # Store updated conversation history in the session
        request.session['conversation_history'] = conversation_history
        # print(f"Updated conversation history: {conversation_history}")
            
        return JsonResponse({"response": response})

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        print(f"Error in chat_with_ai: {e}")
        return JsonResponse({"error": str(e)}, status=500)


# forms.py
def my_view(request):
    if request.method == 'POST':
        form = MyForm(request.POST)
        if form.is_valid():
            # Process the data in form.cleaned_data
            data = form.cleaned_data['my_field']
            # Do something with the data
    else:
        form = MyForm()

    return render(request, 'index.html', {'form': form})

#mock django interactive graph
import json
import umap
import pandas as pd
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

def get_umap_data():
    try:
        # Dummy data for illustration; replace with your actual data
        data = pd.DataFrame({
            'feature1': [0.1, 0.2, 0.3, 0.4],
            'feature2': [0.2, 0.1, 0.4, 0.3],
            'label': ['A', 'B', 'C', 'D']
        })

        # Set n_neighbors to a sensible value based on the size of your dataset
        n_neighbors = min(15, len(data) - 1)

        reducer = umap.UMAP(n_neighbors=n_neighbors)
        embedding = reducer.fit_transform(data[['feature1', 'feature2']])
        data['umap1'] = embedding[:, 0]
        data['umap2'] = embedding[:, 1]

        return data
    except Exception as e:
        raise

def generate_umap_plot(data):
    try:
        fig = px.scatter(
            data,
            x='umap1',
            y='umap2',
            hover_data={'label': True},
            title='UMAP Projection'
        )
        fig.update_layout(
            hovermode='closest',
            xaxis_title='Component 1',
            yaxis_title='Component 2'
        )

        # Convert the figure to JSON
        graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
        return graph_json
    except Exception as e:
        raise
