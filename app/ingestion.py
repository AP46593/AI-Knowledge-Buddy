from pathlib import Path
from app.vectorstore import index_document
from app.utils import extract_text_from_file, chunk_text


TMP_INDEX_META = Path("data/index/metadata.json")
Path("data/index").mkdir(parents=True, exist_ok=True)




def handle_upload_and_index(filepath: str):
    """Extract text, chunk, compute embeddings, and add to FAISS index."""
    text = extract_text_from_file(filepath)
    chunks = chunk_text(text)
    # index_document will compute embeddings and add to FAISS
    index_document(source_path=filepath, chunks=chunks)
    return True




def list_indexed_documents():
    """Return list of indexed source file names from index folder."""
    from glob import glob
    files = glob("data/index/*index*")
    # simpler approach: list saved metadata filenames in data/index/sources.txt
    meta = Path("data/index/sources.txt")
    if not meta.exists():
        return []
    return [l.strip() for l in meta.read_text().splitlines() if l.strip()]