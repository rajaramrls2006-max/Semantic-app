# app.py — Chat-only UI, auto indexing & persistence (Gemini + LlamaIndex)
import os
import streamlit as st
from dotenv import load_dotenv

# LlamaIndex core
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    load_index_from_storage,
)
from llama_index.core.storage.storage_context import StorageContext

# Gemini backends
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# ----------------- CONFIG -----------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Shastri Chat System", layout="wide")
st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 900px;}
.stChatMessage { margin-bottom: .75rem; }
</style>
""", unsafe_allow_html=True)

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is not set. Add it to .env and restart.")
    st.stop()

# LLM + Embeddings (Gemini)
Settings.llm = Gemini(model="models/gemini-1.5-flash", api_key=GOOGLE_API_KEY, temperature=0)
Settings.embed_model = GeminiEmbedding(model_name="models/text-embedding-004", api_key=GOOGLE_API_KEY)

# Absolute paths to avoid cwd quirks
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "storage")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ----------------- HELPERS -----------------
def get_pdf_paths() -> list[str]:
    """Return list of absolute PDF paths in DATA_DIR."""
    try:
        return [
            os.path.join(DATA_DIR, f)
            for f in os.listdir(DATA_DIR)
            if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(DATA_DIR, f))
        ]
    except FileNotFoundError:
        return []

def save_uploaded_files(files) -> list[str]:
    """Save uploaded PDFs to DATA_DIR; return saved absolute paths."""
    saved = []
    for f in files:
        dest = os.path.join(DATA_DIR, f.name)
        with open(dest, "wb") as out:
            out.write(f.getbuffer())
        saved.append(dest)
    return saved

def build_and_persist_index():
    """(Re)build the vector index from PDFs and persist to disk. Returns index or None."""
    pdfs = get_pdf_paths()
    if not pdfs:
        return None
    # Use input_files to avoid SimpleDirectoryReader raising 'No files found'
    docs = SimpleDirectoryReader(input_files=pdfs, errors="ignore").load_data()
    if not docs:
        return None
    index = VectorStoreIndex.from_documents(docs, show_progress=True)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    return index

def load_index_if_exists():
    """Load a previously persisted index; return index or None."""
    try:
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        return load_index_from_storage(storage_context)
    except Exception:
        return None

def ensure_index(auto_rebuild: bool = False):
    """
    Ensure we have a ready-to-use index.
    - If auto_rebuild=False: try load; if missing, return None.
    - If auto_rebuild=True: rebuild from PDFs and persist; return index or None.
    """
    if not auto_rebuild:
        return load_index_if_exists()
    return build_and_persist_index()

def set_query_engine_stream(index):
    """Create streaming query engine and stash in session state."""
    st.session_state.qe_stream = index.as_query_engine(streaming=True)

# ----------------- SESSION STATE -----------------
# Only store assistant messages (answers) to keep UI answers-only
if "answers" not in st.session_state:
    st.session_state.answers = []
if "qe_stream" not in st.session_state:
    idx = ensure_index(auto_rebuild=False)  # Load if persisted
    if idx:
        set_query_engine_stream(idx)

# ----------------- SIDEBAR (upload only; indexing is automatic) -----------------
with st.sidebar:
    st.markdown("### 📄 Upload PDFs")
    files = st.file_uploader("Drop PDFs here", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    if files:
        saved = save_uploaded_files(files)
        with st.spinner("Indexing…"):
            idx = ensure_index(auto_rebuild=True)  # Auto rebuild & persist after upload
            if idx:
                set_query_engine_stream(idx)
                st.success(f"Indexed {len(saved)} file(s).")
            else:
                st.warning("No valid PDFs found to index.")
    if st.button("Clear answers", use_container_width=True):
        st.session_state.answers = []
        st.experimental_rerun()

# ----------------- MAIN: CHAT (answers only) -----------------
st.title("Shastri chat system")

# Show previous assistant answers only
for text in st.session_state.answers:
    with st.chat_message("assistant"):
        st.markdown(text)

prompt = st.chat_input("Ask about your uploaded PDFs…")
if prompt:
    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = ""

        # Ensure query engine exists (maybe user typed before uploading)
        if "qe_stream" not in st.session_state:
            # Try load existing, else build if PDFs are present
            idx = ensure_index(auto_rebuild=False) or ensure_index(auto_rebuild=True)
            if idx:
                set_query_engine_stream(idx)

        if "qe_stream" in st.session_state:
            try:
                resp = st.session_state.qe_stream.query(prompt)
                for token in resp.response_gen:
                    answer += token
                    placeholder.markdown(answer)
            except Exception as e:
                answer = f"⚠️ {e}"
                placeholder.markdown(answer)
        else:
            answer = "Please upload PDFs in the sidebar so I can build the index."
            placeholder.markdown(answer)

    # Save only the assistant's answer
    st.session_state.answers.append(answer)
