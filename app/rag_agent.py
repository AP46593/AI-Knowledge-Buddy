from typing import List, Optional
from pathlib import Path
from app.vectorstore import query_vectors
from app.ollama_client import generate_stream, generate
from app.config import TOP_K_RESULTS, MAX_TOKEN, TEMPERATURE, MEM_LIMIT


class RagAgent:
    """
    Hybrid Chat + RAG Agent with short-term memory.
    - Handles normal Banking & Insurance conversations
    - Uses document retrieval when requested
    - Retains conversational context across turns
    """

    def __init__(self, model_name: Optional[str] = None, memory_limit: int = MEM_LIMIT) -> None:
        self.model_name = model_name or "llama3:8b"
        self.chat_history: List[dict] = []  # stores {"role": "user"/"assistant", "content": "..."}
        self.memory_limit = memory_limit  # number of turns to keep in context

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
    def _format_history(self) -> str:
        """Convert recent chat history into a string for prompt context."""
        if not self.chat_history:
            return ""
        recent = self.chat_history[-self.memory_limit:]
        formatted = "\n".join([f"{h['role'].capitalize()}: {h['content']}" for h in recent])
        return f"\nConversation so far:\n{formatted}\n"

    def _build_rag_prompt(self, question: str, retrieved_docs: List[dict]) -> str:
        """Prompt for RAG-based answer, with memory context."""
        context_text = "\n\n".join(
            [f"Source {i+1}:\n{doc.get('text','')}" for i, doc in enumerate(retrieved_docs)]
        )
        memory_block = self._format_history()
        return f"""
You are a professional assistant specialized in Banking and Insurance.
Answer the user's question using the provided document context and prior conversation.
If the context does not contain the answer, clearly say so.

{memory_block}

QUESTION:
{question}

DOCUMENT CONTEXT:
{context_text}

ANSWER:
""".strip()

    def _build_general_prompt(self, question: str) -> str:
        """Prompt for general conversation with memory context."""
        memory_block = self._format_history()
        return f"""
You are a friendly and knowledgeable assistant specializing in Banking and Insurance.
You can explain topics like loans, claims, policies, and coverage in simple terms.
Do NOT reference uploaded documents unless the user explicitly asks.
Accept only queries related to Banking, Insurance, Finance, claims, policies, loans, or coverage and related topics.
Following are the capabilities you currently support:
- Explain banking and insurance concepts in simple terms
- Provide general advice on banking and insurance topics
- Answer FAQs about banking and insurance products
- Guide users on how to file claims or manage policies
- Suggest best practices for financial health and insurance coverage
- Ingest and reference uploaded documents when asked
if the question is not related to these topics, politely inform the user that you can only assist with Banking and Insurance queries.

{memory_block}

User: {question}
Assistant:
""".strip()

    # --- Core logic ---
    def answer(
        self,
        question: str,
        context_docs: Optional[List[str]] = None,
        force_rag: bool = False,
    ) -> str:
        """
        Answer user query using either general chat or document-based retrieval.
        force_rag=True overrides heuristics (used by sidebar toggle in UI)
        Retains chat history context for better continuity.
        """
        # --- Decide if we use RAG ---
        use_rag = force_rag or self._wants_document_context(question)

        if use_rag:
            # Retrieve top relevant chunks
            if not context_docs:
                retrieved = query_vectors(question, top_k=TOP_K_RESULTS)
            else:
                candidates = query_vectors(question, top_k=TOP_K_RESULTS)
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
            resp = generate_stream(prompt, max_tokens=MAX_TOKEN, temperature=TEMPERATURE)
        except Exception:
            try:
                resp = generate(prompt, max_tokens=MAX_TOKEN, temperature=TEMPERATURE)
            except Exception as e:
                resp = f"[Error contacting model: {e}]"

        # --- Update memory ---
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": str(resp)})

        # Trim memory to avoid overflow
        if len(self.chat_history) > self.memory_limit * 2:
            self.chat_history = self.chat_history[-self.memory_limit * 2 :]

        return str(resp)
