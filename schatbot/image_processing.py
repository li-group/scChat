import os
import base64
import json
import requests
from .utils import find_file_with_extension

def read_image() -> str:
    """
    Encodes the first JPG image found in the media folder and sends it to the OpenAI API.
    Returns the API's description of the image.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    image_path = find_file_with_extension("media", ".jpg")
    if not image_path:
        return "No image found in the folder."
    base64_image = encode_image(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 300
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    try:
        return response.json()['choices'][0]['message']['content']
    except KeyError:
        return "Error in processing the image."