import json
from django.http import StreamingHttpResponse
from ninja import Router
import openai

# To use the OpenAI API key from settings
from django.conf import settings
openai.api_key = settings.OPENAI_API_KEY

router = Router()

@router.get("/stream", tags=_TGS)
def create_stream(request):
    user_content = request.GET.get('content', '')  # Get the content from the query parameter

    def event_stream():
        for chunk in openai.ChatCompletion.create(
            model='gpt-4',
            messages=[{
                "role": "user",
                "content": f"{user_content}. The response should be returned in markdown formatting."
            }],
            stream=True,
        ):
            chatcompletion_delta = chunk["choices"][0].get("delta", {})
            data = json.dumps(dict(chatcompletion_delta))
            print(data)
            yield f'data: {data}\\n\\n'

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response['X-Accel-Buffering'] = 'no'  # Disable buffering in nginx
    response['Cache-Control'] = 'no-cache'  # Ensure clients don't cache the data
    return response