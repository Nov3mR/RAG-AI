import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:3.8b"

def call_LLM(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=150)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print("Ollama call failed:", e)
        return "Error: Could not get a response from the LLM"

