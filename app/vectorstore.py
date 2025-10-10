# pyright: reportOptionalMemberAccess=false, reportAttributeAccessIssue=false
import os
import json
import logging
from pathlib import Path
from typing import Any, Optional, List
from app.config import INDEX_DIR, OLLAMA_EMBED_MODEL,TOP_K_RESULTS
from tqdm import tqdm

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import ollama  # type: ignore
except ImportError:
    ollama = None  # type: ignore

# --- Config ---
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)
META_PATH = INDEX_DIR / "meta.json"
SOURCES_TXT = INDEX_DIR / "sources.txt"

EMBEDDING_MODEL = OLLAMA_EMBED_MODEL

# --- Metadata ---
meta: dict[str, Any] = {"docs": [], "embs": [], "emb_type": "tfidf"}
if META_PATH.exists():
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        meta.setdefault("docs", [])
        meta.setdefault("embs", [])
        meta.setdefault("emb_type", "tfidf")
    except Exception:
        logging.exception("Failed to load meta.json; starting fresh.")

_vectorizer: Optional[TfidfVectorizer] = None


def _get_vectorizer() -> TfidfVectorizer:
    """Return TF-IDF vectorizer, fitting if needed."""
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = TfidfVectorizer(stop_words="english", max_features=2048)
    if not getattr(_vectorizer, "vocabulary_", None) and meta["docs"]:
        _vectorizer.fit([d["text"] for d in meta["docs"]])
    return _vectorizer


def save_index() -> None:
    """Persist metadata to disk."""
    try:
        META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        logging.exception("Failed to save meta.json")


def _embed_texts(texts: List[str]) -> Optional[np.ndarray]:
    """Use Ollama embeddings (nomic-embed-text) if available."""
    if ollama is None:
        return None
    try:
        result = ollama.embed(model=EMBEDDING_MODEL, input=texts)  # type: ignore[attr-defined]
        if isinstance(result, dict) and "embeddings" in result:
            return np.array(result["embeddings"], dtype="float32")
        if isinstance(result, (list, tuple)):
            return np.array(result, dtype="float32")
        raise ValueError(f"Unexpected embedding result type: {type(result)}")
    except Exception as e:
        logging.warning(f"Ollama embedding failed: {e}")
        return None


def index_document(source_path: str, chunks: list[dict[str, str]]) -> None:
    """Index document chunks using Ollama embeddings (preferred) or TF-IDF fallback."""
    if not chunks:
        return
    existing_sources = {d["source"] for d in meta.get("docs", [])}
    if source_path in existing_sources:
        return

    print(f"📄 Indexing {len(chunks)} chunks from: {source_path}")

    # Add chunks with progress bar
    for c in tqdm(chunks, desc=f"Indexing {Path(source_path).name}", unit="chunk"):
        meta["docs"].append({"source": source_path, "chunk_id": c["id"], "text": c["text"][:1000]})

    all_texts = [d["text"] for d in meta["docs"]]
    print("⚙️ Computing embeddings...")
    embeddings = _embed_texts(all_texts)

    if embeddings is not None and embeddings.shape[0] == len(all_texts):
        meta.update({"embs": embeddings.tolist(), "emb_type": "embed"})
        save_index()
        print(f"✅ Indexed {len(all_texts)} text chunks using embeddings.")
    else:
        vec = _get_vectorizer()
        dense = vec.fit_transform(all_texts).toarray()
        meta.update({"embs": dense.tolist(), "emb_type": "tfidf"})
        save_index()
        print(f"✅ Indexed {len(all_texts)} text chunks using TF-IDF fallback.")

    # Update sources.txt
    existing = set(SOURCES_TXT.read_text().splitlines()) if SOURCES_TXT.exists() else set()
    if source_path not in existing:
        existing.add(source_path)
        SOURCES_TXT.write_text("\n".join(sorted(existing)), encoding="utf-8")


def query_vectors(query: str, top_k: int = TOP_K_RESULTS) -> list[dict[str, Any]]:
    """Retrieve similar chunks using Ollama embeddings or TF-IDF."""
    if not meta.get("docs") or not meta.get("embs"):
        return []

    emb_type = meta.get("emb_type", "tfidf")

    # --- Embedding-based retrieval ---
    if emb_type == "embed":
        try:
            corpus_embs = np.array(meta["embs"], dtype="float32")
            q_embs = _embed_texts([query])
            if q_embs is not None:
                sims = cosine_similarity(q_embs, corpus_embs)[0]
                top_idx = np.argsort(sims)[::-1][:top_k]
                return [{**meta["docs"][i], "_score": float(sims[i])} for i in top_idx]
        except Exception as e:
            logging.warning(f"Embedding query failed: {e}")

    # --- TF-IDF fallback ---
    vec = _get_vectorizer()
    embs = np.array(meta["embs"], dtype="float32")
    q_vec = vec.transform([query])
    dense_q = q_vec.toarray() if hasattr(q_vec, "toarray") else np.array(q_vec)  # type: ignore
    sims = cosine_similarity(dense_q, embs)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [{**meta["docs"][i], "_score": float(sims[i])} for i in top_idx]

from shutil import rmtree

def clear_all_indexes() -> None:
    """
    Clear all stored embeddings, metadata, and source references.
    This removes all indexed data and resets the vectorstore.
    """
    global meta, _vectorizer
    try:
        # Remove metadata and source files
        for f in [META_PATH, SOURCES_TXT]:
            if f.exists():
                f.unlink()

        # --- Delete local uploads folder ---
        upload_dir = Path("./tmp_uploads")
        if upload_dir.exists():
            rmtree(upload_dir)
            upload_dir.mkdir(exist_ok=True)
            print("🧹 Cleared tmp_uploads directory.")

        # Reset in-memory data
        meta = {"docs": [], "embs": [], "emb_type": "tfidf"}
        _vectorizer = None

        print("🗑️ Cleared all vector indexes and metadata.")
    except Exception as e:
        logging.exception(f"Error clearing index: {e}")