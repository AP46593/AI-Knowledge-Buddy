import ollama
import os
import streamlit as st

MODEL_NAME = os.environ.get("OLLAMA_MODEL", "llama3:8B")


def generate(prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """
    Generate a full response from Ollama (non-streaming).
    Returns the full text after completion.
    """
    try:
        response = ollama.chat(
    model=MODEL_NAME,
    messages=[{"role": "user", "content": prompt}],
    options={
        "temperature": 0.3,       # 0.2–0.4 for factual; 0.7+ for creativity
        "num_predict": 700,       # increase to allow longer responses
        "top_p": 0.9,             # smoother diversity
    },
)

        if response and "message" in response and "content" in response["message"]:
            return response["message"]["content"].strip()
        return str(response)
    except Exception as e:
        return f"[Error running Ollama model: {e}]"


def generate_stream(prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """
    Stream tokens from Ollama directly into the Streamlit UI.
    Returns the final combined response string.
    """
    try:
        full_response = []
        placeholder = st.empty()  # placeholder for live updates
        text = ""

        # Stream token-by-token output from Ollama
        for part in ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": temperature, "num_predict": max_tokens},
        ):
            chunk = part["message"]["content"]
            text += chunk
            placeholder.markdown(text)  # live update
            full_response.append(chunk)

        return "".join(full_response).strip()
    except Exception as e:
        return f"[Error running Ollama model: {e}]"
