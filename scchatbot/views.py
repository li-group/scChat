import json
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .chatbot import ChatBot

# Create a global ChatBot instance
chatbot_instance = ChatBot()



@csrf_exempt
@require_http_methods(["POST"])
def upload_file(request):
    """
    Saves an uploaded file using Django's default storage.
    """
    if request.FILES:
        uploaded_file = request.FILES['file']
        file_path = default_storage.save(uploaded_file.name, uploaded_file)
        request.session['uploaded_file_path'] = file_path
        return JsonResponse({'status': 'success', 'message': f'File {uploaded_file.name} uploaded successfully.'})
    else:
        return JsonResponse({'status': 'error', 'message': 'No file was uploaded.'}, status=400)

# @csrf_exempt
# @require_http_methods(["GET", "POST"])
# def chat_with_ai(request):
#     """
#     Combined view for chatbot UI (GET) and chat message processing (POST).
#     """
#     if request.method == "GET":
#         return render(request, "scchatbot/index.html")
#     elif request.method == "POST":
#         try:
#             data = json.loads(request.body)
#             user_message = data.get('message', '')
#             if classify_intent(user_message) == 'web_search':
#                 response_text = browse_web(user_message)
#                 return JsonResponse({"response": response_text})
#             response_text = chatbot_instance.send_message(user_message)
#             return JsonResponse({"response": response_text})
#         except json.JSONDecodeError:
#             return JsonResponse({"error": "Invalid JSON"}, status=400)
#         except Exception as e:
#             return JsonResponse({"error": str(e)}, status=500)
#


@csrf_exempt
@require_http_methods(["GET", "POST"])
def chat_with_ai(request):
    print("chat_with_ai: Received a request with method", request.method)
    if request.method == "GET":
        print("chat_with_ai: Handling GET request - rendering index.html")
        return render(request, "scchatbot/index.html")
    elif request.method == "POST":
        try:
            body = request.body.decode("utf-8")
            print("chat_with_ai: Raw request body (first 300 chars):", body[:300])
            data = json.loads(body)
            print("chat_with_ai: Parsed JSON data:", data)
            
            user_message = data.get('message', '')
            print("chat_with_ai: User message:", user_message)
            
            # if classify_intent(user_message) == 'web_search':
            #     response_text = browse_web(user_message)
            #     print("chat_with_ai: Web search response (first 300 chars):", response_text[:300])
            #     return JsonResponse({"response": response_text})
            
            response_text = chatbot_instance.send_message(user_message)
            print("chat_with_ai: Chatbot response (first 300 chars):", response_text[:300])
            
            try:
                parsed_response = json.loads(response_text)
                # Trim the output for logging purposes.
                trimmed_response = {k: (v[:300] + '...') if isinstance(v, str) and len(v) > 300 else v for k, v in parsed_response.items()}
                print("chat_with_ai: Parsed chatbot response as JSON (trimmed):", trimmed_response)
            except json.JSONDecodeError as e:
                print("chat_with_ai: JSON decode error - response is not valid JSON (first 300 chars):", response_text[:300], "Error:", e)
                parsed_response = {"response": response_text}
            
            print("chat_with_ai: Returning JSON response (keys):", list(parsed_response.keys()))
            return JsonResponse(parsed_response)
        except json.JSONDecodeError as e:
            print("chat_with_ai: JSONDecodeError:", e)
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            print("chat_with_ai: Exception occurred:", e)
            return JsonResponse({"error": str(e)}, status=500)
        

        
from django.shortcuts import render
# from .your_visualizations_module import display_umap_html  # Update the import path as needed
from .analysis.visualizations import display_umap
def show_umap(request, cell_type="Overall"):
    plot_html = display_umap(cell_type)
    return render(request, "scchatbot/umap_template.html", {"plot_html": plot_html})


# In views.py
@csrf_exempt
@require_http_methods(["GET"])
def get_umap_plot(request):
    cell_type = request.GET.get("cell_type", "Overall")
    plot_html = display_umap(cell_type)
    return JsonResponse({"plot_html": plot_html})