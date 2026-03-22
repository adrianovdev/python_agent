# models.py
import numpy as np

def embed_text(text: str) -> list:
    np.random.seed(abs(hash(text)) % (10**6))
    return np.random.rand(768).tolist()

def call_llm(prompt: str) -> str:
    # Replace with OpenAI, Azure, local LLM, etc.
    return f"(Stub LLM) {prompt[:400]}..."
