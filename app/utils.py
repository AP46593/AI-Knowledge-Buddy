from pathlib import Path
from docx import Document
from pypdf import PdfReader
import openpyxl
from app.config import CHUNK_SIZE, CHUNK_OVERLAP, OLLAMA_VISION_MODEL
from PIL import Image
import logging
import base64
import os
from typing import Union

try:
    import ollama  # type: ignore
except ImportError:
    ollama = None  # type: ignore


def extract_text_from_file(path: Union[str, Path]) -> str:
    """
    Extract text from supported file types:
    - PDF, DOCX, XLSX/XLS
    - Plain text
    - Images (JPG, PNG, JPEG)
    Uses Ollama Vision (LLaVA) for image-to-text extraction.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    text = ""

    try:
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            for p in reader.pages:
                text += p.extract_text() or ""

        elif suffix == ".docx":
            doc = Document(str(path))
            for p in doc.paragraphs:
                text += p.text + "\n"

        elif suffix in (".xlsx", ".xls"):
            wb = openpyxl.load_workbook(str(path), data_only=True)
            for sheet in wb.worksheets:
                for row in sheet.iter_rows(values_only=True):
                    row_text = "\t".join([str(c) if c else "" for c in row])
                    text += row_text + "\n"

        elif suffix == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")

        elif suffix in (".jpg", ".jpeg", ".png", ".tiff"):
            # Use Ollama Vision model (e.g., llava:latest or llava3)
            if ollama is not None:
                try:
                    with open(path, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode("utf-8")
                    result = ollama.chat(
                        model=OLLAMA_VISION_MODEL,
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    "You are an OCR assistant. "
                                    "Extract and return all visible text clearly from this image. "
                                    "If the image includes forms, tables, or handwritten notes, "
                                    "return them as text in logical reading order."
                                ),
                                "images": [img_b64],
                            }
                        ],
                    )
                    # Handle potential structure of the response
                    if isinstance(result, dict):
                        message = result.get("message", {})
                        text = message.get("content", "").strip()
                    else:
                        text = str(result).strip()
                except Exception as e:
                    logging.warning(f"Ollama vision failed on {path.name}: {e}")
                    text = ""
            else:
                logging.warning("Ollama not available for image OCR.")
                text = ""

        else:
            # Unknown file type, try plain read
            text = path.read_text(encoding="utf-8", errors="ignore")

    except Exception as e:
        logging.exception(f"Error extracting text from {path}: {e}")

    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Very simple char-based chunker. Returns list of dicts with 'id' and 'text'."""
    chunks = []
    i = 0
    n = len(text)
    idx = 0
    while i < n:
        chunk = text[i:i + chunk_size]
        chunks.append({"id": f"chunk_{idx}", "text": chunk})
        i += chunk_size - overlap
        idx += 1
    return chunks
