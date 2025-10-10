import streamlit as st
from pathlib import Path
from app.ingestion import handle_upload_and_index, list_indexed_documents
from app.rag_agent import RagAgent
from app.config import STREAMLIT_TITLE

# --- Setup directories ---
TMP_DIR = Path("./tmp_uploads")
TMP_DIR.mkdir(exist_ok=True)

# --- Page Config ---
st.set_page_config(page_title=STREAMLIT_TITLE, layout="wide")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = RagAgent()
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()
if "selected_docs" not in st.session_state:
    st.session_state.selected_docs = []
if "use_docs" not in st.session_state:
    st.session_state.use_docs = False

# --- SIDEBAR: Document Control Panel ---
with st.sidebar:
    st.header("⚙️ Assistant Settings")


    # --- Retrieval Mode at top ---
    use_docs = st.toggle(
        "Enable document-based answers",
        value=st.session_state.get("use_docs", False),
        help="Turn ON to let the AI use selected documents for context-based answers.",
        disabled=st.session_state.get("busy", False),
    )
    st.session_state.use_docs = use_docs

    st.divider()

    st.header("📁 Document Control Panel")
    st.caption("Manage your uploaded Banking & Insurance documents and retrieval context.")

    # Upload section
    with st.expander("📤 Upload Documents", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload new file(s):",
            type=["pdf", "docx", "xlsx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            disabled=st.session_state.get("busy", False),
        )

        if uploaded_files and not st.session_state.get("busy", False):
            for f in uploaded_files:
                file_path = TMP_DIR / f.name
                if f.name not in st.session_state.indexed_files:
                    with open(file_path, "wb") as out:
                        out.write(f.getbuffer())
                    handle_upload_and_index(str(file_path))
                    st.session_state.indexed_files.add(f.name)
                    st.success(f"✅ {f.name} indexed successfully!")
                else:
                    st.info(f"ℹ️ {f.name} already indexed, skipping re-upload.")

    st.divider()

    # --- Indexed docs section ---
    indexed_docs_full = sorted(set(list_indexed_documents()))
    indexed_docs = [Path(doc).name for doc in indexed_docs_full]

    if indexed_docs:
        st.subheader("🧠 Select Documents for Context")

        with st.expander("📄 Choose Indexed Files", expanded=True):
            selected_files = []
            for doc in indexed_docs:
                checked = st.checkbox(
                    doc,
                    value=(doc in st.session_state.selected_docs),
                    key=f"chk_{doc}",
                    disabled=st.session_state.get("busy", False),
                )
                if checked:
                    selected_files.append(doc)

            st.session_state.selected_docs = selected_files

            if selected_files:
                st.success(f"{len(selected_files)} document(s) selected.")
            else:
                st.info("No documents selected for retrieval.")

    else:
        st.warning("No documents indexed yet. Upload some files to begin.")

    st.divider()
    # --- Clear all indexed documents ---
    st.subheader("🗑️ Manage Index")
    if st.button("Clear All Documents & Index", type="primary", use_container_width=True):
        from app.vectorstore import clear_all_indexes
        clear_all_indexes()
        st.session_state.indexed_files.clear()
        st.session_state.selected_docs.clear()
        st.success("✅ All indexed documents and embeddings have been cleared.")
        st.rerun()

    st.markdown(f"**Total Indexed:** {len(indexed_docs)}")
    st.caption("💡 Keep retrieval OFF for general chat; turn it ON for document Q&A.")

# --- MAIN CHAT AREA ---
st.title(STREAMLIT_TITLE)
st.caption(
    "Chat freely with your AI assistant. "
    "When you want answers grounded in your uploaded documents, "
    "select them in the sidebar and enable document mode."
)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
user_input = st.chat_input("Ask a question or say hello...")

if user_input:
    # Record user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Determine context usage
    if st.session_state.use_docs and st.session_state.selected_docs:
        context_docs = st.session_state.selected_docs
        mode_label = "📘 Using selected documents"
    else:
        context_docs = None
        mode_label = "💬 General Chat"

    # --- Mark busy state ---
    st.session_state.busy = True

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = st.session_state.agent.answer(
                question=user_input,
                context_docs=context_docs,
                force_rag=st.session_state.use_docs,
            )

        # Save first — before displaying, so Streamlit knows state
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Now display once
        st.caption(mode_label)

    # --- Clear busy state ---
    st.session_state.busy = False

    # Display chat history
    #for msg in st.session_state.messages:
    #    with st.chat_message(msg["role"]):
    #        st.markdown(msg["content"])


# --- Footer ---
st.markdown("---")
st.caption(
    "This hybrid assistant can hold normal Banking & Insurance conversations "
    "and switch seamlessly to document-grounded Q&A. "
    "All processing runs locally — no data leaves your system."
)
