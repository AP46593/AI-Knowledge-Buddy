# pyright: reportAttributeAccessIssue=false
from pathlib import Path
import json
import numpy as np
from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import spmatrix

# --- Globals ---
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)
META_PATH = INDEX_DIR / "meta.json"

meta: dict[str, Any] = {"docs": [], "embs": []}
if META_PATH.exists():
    try:
        meta = json.loads(META_PATH.read_text())
    except Exception:
        pass

_vectorizer: TfidfVectorizer | None = None


def _get_vectorizer() -> TfidfVectorizer:
    """Return the global TF-IDF vectorizer, fitted on all docs if needed."""
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = TfidfVectorizer(stop_words="english", max_features=2048)

    # Refit if not fitted yet and we have docs
    if not getattr(_vectorizer, "vocabulary_", None) and meta["docs"]:
        texts = [d["text"] for d in meta["docs"]]
        _vectorizer.fit(texts)
    return _vectorizer


def save_index() -> None:
    """Persist metadata to disk."""
    META_PATH.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def index_document(source_path: str, chunks: list[dict[str, str]]) -> None:
    """Index document chunks using TF-IDF embeddings."""
    if not chunks:
        return

    # --- Prevent duplicate chunk entries ---
    existing_sources = {d["source"] for d in meta.get("docs", [])}
    if source_path in existing_sources:
        return  # already indexed once, skip re-adding

    meta["docs"].extend(
        {"source": source_path, "chunk_id": c["id"], "text": c["text"][:1000]}
        for c in chunks
    )

    # Fit vectorizer and compute embeddings
    vec = _get_vectorizer()
    all_texts = [d["text"] for d in meta["docs"]]
    tfidf_matrix: spmatrix = vec.fit_transform(all_texts)
    dense = tfidf_matrix.toarray() if hasattr(tfidf_matrix, "toarray") else np.array(tfidf_matrix)
    meta["embs"] = dense.tolist()
    save_index()

    # --- Update unique source list for UI display ---
    src_file = INDEX_DIR / "sources.txt"
    existing = set()
    if src_file.exists():
        existing.update(line.strip() for line in src_file.read_text().splitlines() if line.strip())

    if source_path not in existing:
        existing.add(source_path)
        src_file.write_text("\n".join(sorted(existing)), encoding="utf-8")


def query_vectors(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve similar chunks using cosine similarity on TF-IDF embeddings."""
    if not meta.get("docs") or not meta.get("embs"):
        return []

    vec = _get_vectorizer()

    # Safety: ensure vectorizer fitted
    if not getattr(vec, "vocabulary_", None):
        texts = [d["text"] for d in meta["docs"]]
        if not texts:
            return []
        vec.fit(texts)

    embs = np.array(meta["embs"], dtype="float32")
    q_vec = vec.transform([query])
    q_dense = q_vec.toarray() if hasattr(q_vec, "toarray") else np.array(q_vec)
    sims = cosine_similarity(q_dense, embs)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [meta["docs"][i] for i in top_idx]
