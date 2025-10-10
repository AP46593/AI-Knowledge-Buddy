from typing import List, Optional
from pathlib import Path
from app.vectorstore import query_vectors
from app.ollama_client import generate_stream, generate


class RagAgent:
    """
    Hybrid Chat + RAG Agent.
    - Handles normal Banking & Insurance conversations
    - Uses document retrieval when user or UI requests
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or "llama3:8b"

    # --- Intent detection ---
    def _wants_document_context(self, text: str) -> bool:
        """Detect if the user explicitly refers to uploaded documents."""
        keywords = [
            "document", "policy", "file", "refer", "upload", "attached",
            "section", "clause", "claim", "terms", "coverage"
        ]
        text_lower = text.lower()
        return any(k in text_lower for k in keywords)

    # --- Prompt Builders ---
    def _build_rag_prompt(self, question: str, retrieved_docs: List[dict]) -> str:
        """Prompt for RAG-based answer."""
        context_text = "\n\n".join(
            [f"Source {i+1}:\n{doc.get('text','')}" for i, doc in enumerate(retrieved_docs)]
        )
        return f"""
You are a professional assistant specialized in Banking and Insurance.
Answer the user's question using the provided document context.
If the context does not contain the answer, clearly say so.

QUESTION:
{question}

DOCUMENT CONTEXT:
{context_text}

ANSWER:
""".strip()

    def _build_general_prompt(self, question: str) -> str:
        """Prompt for general conversation."""
        return f"""
You are a friendly and knowledgeable assistant specializing in Banking and Insurance.
You can explain topics like loans, claims, policies, and coverage in simple terms.
Do NOT reference uploaded documents unless the user explicitly asks.

User: {question}
Assistant:
""".strip()

    # --- Core logic ---
    def answer(
        self,
        question: str,
        context_docs: Optional[List[str]] = None,
        force_rag: bool = False,   # ✅ new argument
    ) -> str:
        """
        Answer user query using either general chat or document-based retrieval.
        force_rag=True overrides heuristics (used by sidebar toggle in UI)
        """
        # --- Decide if we use RAG ---
        use_rag = force_rag or self._wants_document_context(question)

        if use_rag:
            # Retrieve top relevant chunks
            if not context_docs:
                retrieved = query_vectors(question, top_k=8)
            else:
                candidates = query_vectors(question, top_k=20)
                retrieved = [
                    c for c in candidates
                    if Path(c.get("source", "")).name in context_docs
                ][:8]

            if not retrieved:
                return "I couldn’t find relevant information in the selected documents."

            prompt = self._build_rag_prompt(question, retrieved)
        else:
            prompt = self._build_general_prompt(question)

        # --- Generate response ---
        try:
            resp = generate_stream(prompt, max_tokens=800, temperature=0.4)
        except Exception:
            try:
                resp = generate(prompt, max_tokens=800, temperature=0.4)
            except Exception as e:
                resp = f"[Error contacting model: {e}]"

        return str(resp)
