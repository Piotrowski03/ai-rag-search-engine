from google import genai
from dotenv import load_dotenv
import os
class LLMClient:
    def __init__(self, dotenv_path, model_name="gemini-3-flash-preview"):
        load_dotenv(dotenv_path)
        api_key = os.getenv('GEMINI_API_KEY')
        print(api_key)
        if not api_key:
            raise Exception('GEMINI_API_KEY environment variable is not set')
        
        self.client = genai.Client(api_key=api_key)

        self.model = model_name

    def generate(self, prompt: str):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")
        try:
            response = self.client.models.generate_content(model=self.model,contents=prompt)
            return response.text
        except Exception as e:
            return f"LLM error: {str(e)}"