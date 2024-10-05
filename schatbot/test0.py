import os # Included to Python
# from openai import OpenAI # OpenAI official Python package
import openai

# from IPython.display import Audio # Included to Python
# client = OpenAI(
    # api_key=os.getenv("openaikey"))
openai.api_key = "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"

response = openai.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What you think the person in the image is doing?"},
                {
                    "type": "image_url",
                    "image_url": "https://drive.google.com/drive/u/0/search?q=.jpg",
                },
            ],
        }
    ],
    max_tokens=300,
)
print(response.choices[0].message.content)
