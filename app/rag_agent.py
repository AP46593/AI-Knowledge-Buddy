from typing import List, Optional
from pathlib import Path
from app.vectorstore import query_vectors
from app.ollama_client import generate_stream, generate
from app.config import (
    TOP_K_RESULTS,
    MAX_TOKEN,
    TEMPERATURE,
    MEM_LIMIT,
    OLLAMA_CHAT_MODEL,
)
from app.inc_helper import generate_ticket_number, create_incident_payload, save_incident_json


class RagAgent:
    """
    Hybrid Chat + RAG Agent with short-term memory and task-oriented capabilities.
    - Handles Banking & Insurance conversations (general + document-grounded)
    - Retains conversational context
    - Supports mock ServiceNow ticket creation flow
    """

    def __init__(self, model_name: Optional[str] = None, memory_limit: int = MEM_LIMIT) -> None:
        self.model_name = model_name or OLLAMA_CHAT_MODEL
        self.chat_history: List[dict] = []
        self.memory_limit = memory_limit

        # State for ServiceNow ticket creation
        self.active_ticket = False
        self.pending_field = None
        self.collected_data = {}
        self.awaiting_confirmation = False

        # Ticket field sequence
        self.ticket_fields = [
            ("username", "Please provide your username:"),
            ("team_name", "Which team or department are you from?"),
            ("impacted_count", "How many people are impacted by this issue?"),
            ("application", "Which application or system is affected?"),
            ("short_description", "Please provide a short summary of the issue:"),
            ("long_description", "Could you describe the issue in more detail?"),
            ("contact_email", "What is your contact email ID?"),
        ]

    # --- Intent Detection ---
    def _wants_document_context(self, text: str) -> bool:
        keywords = [
            "document", "policy", "file", "refer", "upload", "attached",
            "section", "clause", "claim", "terms", "coverage",
        ]
        return any(k in text.lower() for k in keywords)

    def _wants_ticket_creation(self, text: str) -> bool:
        """Detect intent to raise a ServiceNow or incident ticket."""
        keywords = [
            "raise incident", "raise a ticket", "open ticket", "log ticket",
            "create ticket", "create incident", "open incident", "servicenow",
            "snow ticket", "report an issue", "log an issue",
            "help me raise an incident", "report problem",
        ]
        return any(k in text.lower() for k in keywords)

    # --- Prompt Builders ---
    def _format_history(self) -> str:
        """Convert recent chat history into a string for prompt context."""
        if not self.chat_history:
            return ""
        recent = self.chat_history[-self.memory_limit:]
        formatted = "\n".join(
            [f"{h['role'].capitalize()}: {h['content']}" for h in recent]
        )
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
- Raise a ServiceNow (SNOW) ticket by collecting required details interactively

If the question is not related to these topics, politely inform the user that you can only assist with Banking and Insurance queries — except when the user is asking to raise an incident or ServiceNow ticket.

{memory_block}

User: {question}
Assistant:
""".strip()

    # --- Ticket Flow Logic ---
    def _start_ticket_flow(self) -> str:
        """Begin ticket creation flow."""
        self.active_ticket = True
        self.collected_data = {}
        self.pending_index = 0
        self.awaiting_confirmation = False
        field_name, field_prompt = self.ticket_fields[self.pending_index]
        self.pending_field = field_name
        return f"Sure! Let's create a ServiceNow ticket.\n{field_prompt}"

    def _get_next_question(self) -> Optional[str]:
        """Move to the next question and return it."""
        self.pending_index += 1
        if self.pending_index < len(self.ticket_fields):
            self.pending_field = self.ticket_fields[self.pending_index][0]
            return self.ticket_fields[self.pending_index][1]
        return None

    def _summarize_ticket(self) -> str:
        """Generate human-readable summary for confirmation."""
        summary = "\n".join(
            [f"- **{k.replace('_', ' ').capitalize()}**: {v}" for k, v in self.collected_data.items()]
        )
        return f"Here's what I've collected so far:\n\n{summary}\n\nWould you like me to proceed with ticket creation? (yes/no)"

    def _finalize_ticket(self) -> str:
        """Generate ticket, save JSON, reset state."""
        payload = create_incident_payload(self.collected_data)
        ticket_number = generate_ticket_number()
        save_incident_json(payload, ticket_number)

        self.active_ticket = False
        self.awaiting_confirmation = False
        self.pending_field = None
        self.pending_index = 0
        self.collected_data = {}

        return f"✅ Your incident has been created successfully.\nTicket Number: **{ticket_number}**"

    def _handle_ticket_conversation(self, user_input: str) -> str:
        """Manage multi-turn ServiceNow conversation."""
        text = user_input.strip().lower()

        # Handle confirmation phase
        if self.awaiting_confirmation:
            if any(k in text for k in ["yes", "proceed", "confirm", "ok", "sure"]):
                return self._finalize_ticket()
            elif any(k in text for k in ["no", "edit", "change"]):
                self.awaiting_confirmation = False
                # Re-ask the last field
                field_prompt = self.ticket_fields[self.pending_index][1]
                return f"Okay, let's update it.\n{field_prompt}"
            else:
                return "Please confirm — should I proceed with ticket creation? (yes/no)"

        # Collect the answer for the current field
        if self.pending_field:
            self.collected_data[self.pending_field] = user_input.strip()

        # Move to the next question
        next_q = self._get_next_question()
        if next_q:
            return next_q

        # All fields completed → show summary
        self.awaiting_confirmation = True
        return self._summarize_ticket()

    # --- Main Chat Logic ---
    def answer(
        self,
        question: str,
        context_docs: Optional[List[str]] = None,
        force_rag: bool = False,
    ) -> str:
        """Main entry point: routes between RAG, general chat, and ticket creation."""
        q_lower = question.lower().strip()

        # 1️⃣ If user is already in ticket flow
        if self.active_ticket:
            response = self._handle_ticket_conversation(question)
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": response})
            return response

        # 2️⃣ Detect new ticket creation intent first
        if self._wants_ticket_creation(q_lower):
            self.chat_history.append({"role": "user", "content": question})
            response = self._start_ticket_flow()
            self.chat_history.append({"role": "assistant", "content": response})
            return response

        # 3️⃣ Otherwise normal RAG / General chat
        use_rag = force_rag or self._wants_document_context(q_lower)

        if use_rag:
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

        try:
            resp = generate_stream(prompt, max_tokens=MAX_TOKEN, temperature=TEMPERATURE)
        except Exception:
            try:
                resp = generate(prompt, max_tokens=MAX_TOKEN, temperature=TEMPERATURE)
            except Exception as e:
                resp = f"[Error contacting model: {e}]"

        # Update memory
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": str(resp)})

        if len(self.chat_history) > self.memory_limit * 2:
            self.chat_history = self.chat_history[-self.memory_limit * 2 :]

        return str(resp)
