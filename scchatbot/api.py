import json
from django.http import StreamingHttpResponse
from ninja import Router
import openai

# To use the OpenAI API key from settings
from django.conf import settings

router = Router()

@router.get("/stream", tags=_TGS)
def create_stream(request):
    user_content = request.GET.get('content', '')  # Get the content from the query parameter

    def event_stream():
        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        
        for chunk in client.chat.completions.create(
            model='gpt-4o',
            messages=[{
                "role": "user",
                "content": f"{user_content}. The response should be returned in markdown formatting."
            }],
            stream=True,
        ):
            chatcompletion_delta = chunk.choices[0].delta
            # Convert delta to dict for JSON serialization
            delta_dict = {}
            if chatcompletion_delta.content:
                delta_dict["content"] = chatcompletion_delta.content
            if chatcompletion_delta.role:
                delta_dict["role"] = chatcompletion_delta.role
            
            data = json.dumps(delta_dict)
            print(data)
            yield f'data: {data}\\n\\n'

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response['X-Accel-Buffering'] = 'no'  # Disable buffering in nginx
    response['Cache-Control'] = 'no-cache'  # Ensure clients don't cache the data
    return response