import google.generativeai as genai
import re

def browse_web(query):
    API_KEY = ''

    genai.configure(api_key = API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(query + ". Give me the references for it with the links.")
    text = response.text

    print(text)
    
    # Process the input text
    formatted_text = clean_markdown_text(text)
    print(formatted_text)
   
    return formatted_text

def clean_markdown_text(text):
    # Convert markdown bold headers (e.g., **Section Title**) to a more readable format with a colon
    cleaned_text = re.sub(r'\*\*([^*]+)\*\*', r'\n\n\1:\n', text)

    # Replace markdown list items, changing '*' to a dash '-'
    cleaned_text = re.sub(r'\n\*', '\n -', cleaned_text)

    # Extract URLs from markdown link syntax and format them nicely
    cleaned_text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'\n\1: \2', cleaned_text)

    # Remove excessive newlines, ensuring that no more than one newline occurs in a row
    cleaned_text = re.sub(r'\n{2,}', '\n\n', cleaned_text)
    
    # Regular expression for matching URLs
    url_pattern = re.compile(
        r'(https?://[^\s]+)',
        re.IGNORECASE
    )
    # Replace found URLs with an HTML anchor tag
    formatted_text = re.sub(url_pattern, r'<a href="\1" target="_blank">\1</a>', cleaned_text)

    return formatted_text




