import requests
import openai
import os
from dotenv import load_dotenv

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:3.8b"

load_dotenv()
openai.api_key = os.getenv("openAIAPIKey")
openaiModel = "gpt-4o-mini"

def call_LLM(prompt: str, mode="openai") -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    if mode == "openai":
        print("Using OpenAI API")
        response = openai.chat.completions.create(
            model=openaiModel,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant specialized in UAE VAT and tax invoices. You must keep in mind any previous context and questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=1000
        )

        return response.choices[0].message.content or "No response from LLM"
    else:
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=150)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            print("Ollama call failed:", e)
            return "Error: Could not get a response from the LLM"