import ollama
import os
import streamlit as st
from app.config import OLLAMA_CHAT_MODEL,TEMPERATURE,MAX_TOKEN,LOW_TEMP,TOP_P


MODEL_NAME = OLLAMA_CHAT_MODEL

def generate(prompt: str, max_tokens: int = MAX_TOKEN, temperature: float = LOW_TEMP) -> str:
    """
    Generate a full response from Ollama (non-streaming).
    Returns the full text after completion.
    """
    try:
        response = ollama.chat(
    model=MODEL_NAME,
    messages=[{"role": "user", "content": prompt}],
    options={
        "temperature": TEMPERATURE,       # 0.2–0.4 for factual; 0.7+ for creativity
        "num_predict": MAX_TOKEN,       # increase to allow longer responses
        "top_p": TOP_P,             # smoother diversity
    },
)

        if response and "message" in response and "content" in response["message"]:
            return response["message"]["content"].strip()
        return str(response)
    except Exception as e:
        return f"[Error running Ollama model: {e}]"


def generate_stream(prompt: str, max_tokens: int = MAX_TOKEN, temperature: float = LOW_TEMP) -> str:
    """
    Stream tokens from Ollama directly into the Streamlit UI.
    Returns the final combined response string.
    """
    try:
        full_response = []
        placeholder = st.empty()

        # Stream token-by-token output from Ollama
        for part in ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": temperature, "num_predict": max_tokens},
        ):
            chunk = part["message"]["content"]
            full_response.append(chunk)
            placeholder.markdown("".join(full_response))

        # Return final response text for saving to chat history
        return "".join(full_response).strip()
    
    except Exception as e:
        return f"[Error running Ollama model: {e}]"
