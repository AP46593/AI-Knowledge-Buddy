import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(dotenv_path=Path(".env"), override=True)

# === Ollama Models ===
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:latest")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision:11b")

# === Chunking ===
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 400))

# === Retrieval ===
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 10))
TOP_P = float(os.getenv("TOP_P", 0.9))
MAX_TOKEN = int(os.getenv("MAX_TOKEN", 1000))
LOW_TOKEN = int(os.getenv("LOW_TOKEN", 512))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.4))
LOW_TEMP = float(os.getenv("LOW_TEMP", 0.2))
HIGH_TEMP = float(os.getenv("HIGH_TEMP", 0.7))
MEM_LIMIT = int(os.getenv("MEM_LIMIT", 10))  # number of recent turns to keep in context

# === Paths ===
INDEX_DIR = Path(os.getenv("INDEX_DIR", "data/index"))
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# === UI ===
STREAMLIT_TITLE = os.getenv("STREAMLIT_TITLE", "Knowledge & Incident Copilot")