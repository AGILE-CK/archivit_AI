import google.generativeai as genai

# API Configuration
api_key = "AIzaSyAj9Hqe-mhE3SNitroUHlLczvHqII49ZFE"
genai.configure(api_key=api_key)

# Model and Generation Configuration
model_name = "gemini-pro"
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Prefix for content generation
prefix = "(답변은 100단어로만 해주세요) 주어진 연설 내용 및 발표자를 요약하고 결론 내려주세요:  \n"

def generate_content(prompt_parts):
    # Incorporate the prefix with the provided prompt parts
    full_prompt = [prefix] + prompt_parts
    
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    
    response = model.generate_content(full_prompt)
    return response.text