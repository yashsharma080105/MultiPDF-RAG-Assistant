"""
MultiPDF RAG Assistant — Streamlit UI
================================================================
Single self-contained app: upload any PDF(s) from your file manager and the
FULL pipeline (text extraction with OCR fallback -> chunking -> embedding ->
FAISS indexing) runs automatically, incrementally, right when you save the
files. No separate main.py / manual pipeline run needed.

Run with:  streamlit run app.py
"""

import os
import re
import io
import sys
import json
import time
import shutil

import streamlit as st
import numpy as np
import faiss
from dotenv import load_dotenv

# ----------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="MultiPDF RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header {visibility: hidden;}

    .stApp {
        background: radial-gradient(1200px 600px at 10% -10%, #1b2340 0%, #0e1220 45%, #0a0d16 100%);
    }

    /* Hero header */
    .hero {
        padding: 1.6rem 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #6C5CE7 0%, #4834d4 55%, #341f97 100%);
        box-shadow: 0 20px 45px -20px rgba(76, 52, 212, 0.65);
        margin-bottom: 1.4rem;
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: rgba(255,255,255,0.08);
        border-radius: 50%;
    }
    .hero h1 { color: white; font-weight: 800; font-size: 2.1rem; margin: 0 0 0.25rem 0; letter-spacing: -0.02em; }
    .hero p { color: rgba(255,255,255,0.85); font-size: 0.98rem; margin: 0; }
    .hero .badges { margin-top: 0.9rem; display:flex; gap:0.5rem; flex-wrap:wrap; }
    .pill {
        display:inline-block; background: rgba(255,255,255,0.14); color: white;
        padding: 0.28rem 0.75rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
        border: 1px solid rgba(255,255,255,0.25); backdrop-filter: blur(4px);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #10131f; border-right: 1px solid rgba(255,255,255,0.06); }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #a29bfe; font-size: 0.85rem; text-transform: uppercase;
        letter-spacing: 0.08em; font-weight: 700; margin-top: 1.2rem;
    }
    .stat-grid { display:flex; gap:0.6rem; }
    .stat-box {
        flex:1; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px; padding: 0.7rem; text-align:center;
    }
    .stat-box .num { font-size:1.3rem; font-weight:800; color:#a29bfe; }
    .stat-box .lbl { font-size:0.68rem; color:#9aa0b4; text-transform:uppercase; letter-spacing:0.05em; }

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px; padding: 0.4rem 0.2rem; margin-bottom: 0.6rem;
    }

    /* Source cards */
    .src-card {
        background: linear-gradient(135deg, rgba(108,92,231,0.10), rgba(255,255,255,0.02));
        border: 1px solid rgba(162,155,254,0.25); border-radius: 12px; padding: 0.65rem 0.9rem; margin-bottom: 0.55rem;
    }
    .src-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem; }
    .src-name { font-weight:700; color:#e8e8ff; font-size:0.86rem; }
    .src-page {
        font-size: 0.7rem; background: rgba(162,155,254,0.18); color:#c9c3ff;
        padding: 0.12rem 0.55rem; border-radius: 999px; font-weight:600;
    }
    .src-text { color:#b8bcd0; font-size:0.82rem; line-height:1.45; }
    .score-track { height:5px; border-radius:4px; background:rgba(255,255,255,0.08); margin-top:0.45rem; overflow:hidden; }
    .score-fill { height:100%; background: linear-gradient(90deg,#6C5CE7,#a29bfe); border-radius:4px; }

    .stChatInput textarea, div[data-testid="stChatInput"] { border-radius: 14px !important; }
    .stButton>button { border-radius: 10px; font-weight: 600; border: 1px solid rgba(255,255,255,0.12); }

    .scope-badge {
        display:inline-block; margin-top: 0.3rem; font-size: 0.72rem; color:#a29bfe;
        background: rgba(162,155,254,0.12); border: 1px solid rgba(162,155,254,0.3);
        padding: 0.15rem 0.6rem; border-radius: 999px;
    }
    .new-badge {
        display:inline-block; font-size: 0.65rem; color:#2ecc71;
        background: rgba(46,204,113,0.12); border: 1px solid rgba(46,204,113,0.35);
        padding: 0.05rem 0.45rem; border-radius: 999px; margin-left: 0.4rem;
    }
    div[data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px dashed rgba(162,155,254,0.4) !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Paths / project layout
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "pdfs")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
EMBED_FOLDER = os.path.join(BASE_DIR, "embeddings")
VECTOR_DB_FOLDER = os.path.join(BASE_DIR, "vector_db")

EXTRACTED_FILE = os.path.join(OUTPUT_FOLDER, "extracted.json")
CHUNKS_FILE = os.path.join(OUTPUT_FOLDER, "chunks.json")
EMBEDDINGS_FILE = os.path.join(EMBED_FOLDER, "embeddings.npy")
METADATA_FILE = os.path.join(EMBED_FOLDER, "metadata.json")
INDEX_FILE = os.path.join(VECTOR_DB_FOLDER, "faiss.index")

for d in (PDF_FOLDER, OUTPUT_FOLDER, EMBED_FOLDER, VECTOR_DB_FOLDER):
    os.makedirs(d, exist_ok=True)

os.chdir(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

load_dotenv(os.path.join(BASE_DIR, ".env"))

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
OCR_LANGUAGE = "eng+hin"
OCR_ZOOM = 2
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50


# ========================================================================
# CACHED MODELS (loaded once per session)
# ========================================================================
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_gemini_model():
    import google.generativeai as genai
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


@st.cache_resource(show_spinner=False)
def load_faiss_index():
    if not os.path.exists(INDEX_FILE):
        return None
    return faiss.read_index(INDEX_FILE)


@st.cache_resource(show_spinner=False)
def load_metadata():
    if not os.path.exists(METADATA_FILE):
        return []
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_embeddings():
    if not os.path.exists(EMBEDDINGS_FILE):
        return np.zeros((0, 0), dtype="float32")
    emb = np.load(EMBEDDINGS_FILE).astype("float32")
    faiss.normalize_L2(emb)
    return emb


def clear_index_caches():
    load_faiss_index.clear()
    load_metadata.clear()
    load_embeddings.clear()


# ========================================================================
# STEP 1 — TEXT EXTRACTION (with OCR fallback for broken/scanned pages)
# ========================================================================
def ocr_page(page):
    import fitz
    from PIL import Image
    import pytesseract

    pix = page.get_pixmap(matrix=fitz.Matrix(OCR_ZOOM, OCR_ZOOM))
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    text = pytesseract.image_to_string(image, lang=OCR_LANGUAGE)
    return text.strip()


def extract_pdf_pages(pdf_path):
    """Returns a list of {page, text} dicts for a single PDF, OCR-ing pages
    whose native text layer is empty/broken."""
    import fitz

    pages_out = []
    doc = fitz.open(pdf_path)
    ocr_count = 0

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if len(text) < 10:
            try:
                text = ocr_page(page)
                ocr_count += 1
            except Exception:
                pass
        pages_out.append({"page": page_number, "text": text})

    doc.close()
    return pages_out, ocr_count


# ========================================================================
# STEP 2 — CHUNKING (word-based, with overlap, page-aware)
# ========================================================================
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def chunk_words(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append({"text": " ".join(words[start:end]), "start_word": start})
        if end == len(words):
            break
        start = end - overlap
    return chunks


def chunk_pdf_pages(pages, pdf_name, start_chunk_id):
    """Turns a single PDF's extracted pages into page-tagged chunks."""
    pages = sorted(pages, key=lambda p: p["page"])

    full_words, page_boundaries = [], []
    for page in pages:
        words = clean_text(page["text"]).split()
        page_boundaries.append((len(full_words), page["page"]))
        full_words.extend(words)

    full_text = " ".join(full_words)
    word_chunks = chunk_words(full_text)

    result, chunk_id = [], start_chunk_id
    for wc in word_chunks:
        chunk_page = page_boundaries[0][1] if page_boundaries else 1
        for boundary_start, page_num in page_boundaries:
            if boundary_start <= wc["start_word"]:
                chunk_page = page_num
            else:
                break
        result.append({
            "chunk_id": chunk_id,
            "pdf_name": pdf_name,
            "page": chunk_page,
            "text": wc["text"],
        })
        chunk_id += 1

    return result, chunk_id


# ========================================================================
# STEP 3 + 4 — EMBED NEW CHUNKS & UPDATE THE FAISS INDEX (incremental)
# ========================================================================
def _load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def process_new_pdfs(filenames, progress_cb=None):
    """
    Full pipeline for a list of newly-added PDF filenames (must already exist
    in PDF_FOLDER): extract -> chunk -> embed -> merge into the FAISS index.
    Re-uploading a filename that was already indexed replaces its old chunks.
    Only the given files are read/embedded — everything else is left as-is,
    so this stays fast no matter how large the existing library is.
    """
    def report(msg):
        if progress_cb:
            progress_cb(msg)

    model = load_embedding_model()

    # ---- Load existing state ----
    extracted_docs = _load_json(EXTRACTED_FILE, [])
    all_chunks = _load_json(CHUNKS_FILE, [])
    metadata = _load_json(METADATA_FILE, [])
    old_embeddings = (
        np.load(EMBEDDINGS_FILE).astype("float32")
        if os.path.exists(EMBEDDINGS_FILE) else np.zeros((0, 0), dtype="float32")
    )

    reprocess_set = set(filenames)

    # Drop any prior entries for files being (re)processed, to avoid duplicates
    extracted_docs = [d for d in extracted_docs if d["pdf_name"] not in reprocess_set]
    keep_mask = np.array([m["pdf_name"] not in reprocess_set for m in metadata], dtype=bool)
    metadata = [m for m, keep in zip(metadata, keep_mask) if keep]
    if old_embeddings.shape[0] == len(keep_mask):
        old_embeddings = old_embeddings[keep_mask] if keep_mask.size else old_embeddings
    all_chunks = [c for c in all_chunks if c["pdf_name"] not in reprocess_set]

    next_chunk_id = (max((m["chunk_id"] for m in metadata), default=0)) + 1

    new_chunks = []
    total_ocr_pages = 0

    for filename in filenames:
        report(f"📖 Reading {filename}…")
        pdf_path = os.path.join(PDF_FOLDER, filename)
        pages, ocr_count = extract_pdf_pages(pdf_path)
        total_ocr_pages += ocr_count
        extracted_docs.extend({"pdf_name": filename, **p} for p in pages)

        report(f"✂️ Chunking {filename}…")
        file_chunks, next_chunk_id = chunk_pdf_pages(pages, filename, next_chunk_id)
        new_chunks.extend(file_chunks)
        all_chunks.extend(file_chunks)

    if not new_chunks:
        report("Nothing to embed (no extractable text found).")
        return {"new_chunks": 0, "ocr_pages": total_ocr_pages}

    report(f"🧠 Generating embeddings for {len(new_chunks)} new chunk(s)…")
    texts = [c["text"] for c in new_chunks]
    new_vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")

    if old_embeddings.size and old_embeddings.shape[1] == new_vectors.shape[1]:
        combined = np.vstack([old_embeddings, new_vectors])
    else:
        combined = new_vectors

    metadata.extend(new_chunks)

    report("📦 Updating the FAISS index…")
    normed = combined.copy()
    faiss.normalize_L2(normed)
    index = faiss.IndexFlatIP(normed.shape[1])
    index.add(normed)

    # ---- Persist everything ----
    _save_json(EXTRACTED_FILE, extracted_docs)
    _save_json(CHUNKS_FILE, all_chunks)
    _save_json(METADATA_FILE, metadata)
    np.save(EMBEDDINGS_FILE, combined)
    faiss.write_index(index, INDEX_FILE)

    clear_index_caches()
    report("✅ Done.")
    return {"new_chunks": len(new_chunks), "ocr_pages": total_ocr_pages}


# ========================================================================
# ANSWER GENERATION (Gemini)
# ========================================================================
def generate_answer(question, context, model):
    prompt = f"""
You are an AI assistant that answers questions ONLY using the provided document context.

Instructions:
1. Use only the given context.
2. Do not make up information.
3. If the answer is not found, reply:
   "I couldn't find the answer in the uploaded documents."
4. Keep the answer clear and concise.

Context:
{context}

Question:
{question}

Answer:
"""
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.2, "max_output_tokens": 512},
    )
    return response.text


def ask_question(query, k=8, selected_pdfs=None):
    """selected_pdfs=None or covering every indexed file -> fast FAISS search.
    A smaller subset -> brute-force cosine search restricted to those files."""
    model = load_embedding_model()
    metadata = load_metadata()
    llm = load_gemini_model()

    if llm is None:
        return "⚠️ No `GOOGLE_API_KEY` found in `.env`. Add a valid Gemini API key to get answers.", []
    if not metadata:
        return "No documents have been indexed yet. Upload a PDF to get started.", []

    all_pdfs = {m["pdf_name"] for m in metadata}
    scoped = selected_pdfs is not None and set(selected_pdfs) != all_pdfs

    query_embedding = model.encode(query, convert_to_numpy=True).astype("float32")
    query_embedding = np.expand_dims(query_embedding, axis=0)
    faiss.normalize_L2(query_embedding)

    if not scoped:
        index = load_faiss_index()
        if index is None:
            return "No documents have been indexed yet. Upload a PDF to get started.", []
        distances, indices = index.search(query_embedding, k)
        hits = list(zip(distances[0].tolist(), indices[0].tolist()))
    else:
        embeddings = load_embeddings()
        selected_set = set(selected_pdfs)
        subset_idx = [i for i, m in enumerate(metadata) if m["pdf_name"] in selected_set]
        if not subset_idx:
            return "No documents selected. Choose at least one PDF in the sidebar.", []
        subset_emb = embeddings[subset_idx]
        scores = subset_emb @ query_embedding[0]
        top_local = np.argsort(-scores)[:k]
        hits = [(float(scores[i]), subset_idx[i]) for i in top_local]

    context, sources = "", []
    for score, idx in hits:
        if idx == -1:
            continue
        chunk = metadata[idx]
        context += chunk["text"] + "\n\n"
        sources.append({
            "pdf_name": chunk["pdf_name"],
            "page": chunk["page"],
            "text": chunk["text"][:280].replace("\n", " ").strip(),
            "score": float(score),
        })

    if context.strip() == "":
        return "No relevant information found in the selected document(s).", []

    answer = generate_answer(query, context, llm)
    return answer, sources


# ========================================================================
# SIDEBAR
# ========================================================================
with st.sidebar:
    pdf_files = []
    if os.path.isdir(PDF_FOLDER):
        pdf_files = sorted(f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf"))

    metadata_now = load_metadata()
    indexed_pdfs = {m["pdf_name"] for m in metadata_now}

    # ---- Upload: auto-processes on save, no separate rebuild step ----
    st.markdown("### ➕ Add PDFs")
    new_files = st.file_uploader(
        "Upload any PDF(s) from your computer — processed automatically",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if new_files:
        label = f"⚡ Process {len(new_files)} file(s) now"
        if st.button(label, use_container_width=True, type="primary"):
            saved_names = []
            for uf in new_files:
                dest = os.path.join(PDF_FOLDER, uf.name)
                with open(dest, "wb") as out:
                    shutil.copyfileobj(uf, out)
                saved_names.append(uf.name)

            status_box = st.empty()

            def report(msg):
                status_box.info(msg)

            with st.spinner("Running pipeline…"):
                try:
                    result = process_new_pdfs(saved_names, progress_cb=report)
                    status_box.empty()
                    st.success(
                        f"✅ Indexed {result['new_chunks']} new chunk(s) from "
                        f"{len(saved_names)} file(s)"
                        + (f" ({result['ocr_pages']} page(s) needed OCR)" if result["ocr_pages"] else "")
                        + ". Ready to query!"
                    )
                    time.sleep(1.2)
                    st.rerun()
                except Exception as e:
                    status_box.empty()
                    st.error(f"Processing failed: {e}")

    st.markdown("### 🎯 Search Scope")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Select all", use_container_width=True):
            for f in pdf_files:
                st.session_state[f"doc_{f}"] = True
    with col_b:
        if st.button("Clear all", use_container_width=True):
            for f in pdf_files:
                st.session_state[f"doc_{f}"] = False

    selected_pdfs = []
    for f in pdf_files:
        key = f"doc_{f}"
        if key not in st.session_state:
            st.session_state[key] = True
        checked = st.checkbox(f, key=key)
        size_mb = os.path.getsize(os.path.join(PDF_FOLDER, f)) / (1024 * 1024)
        note = " ⚠️ not indexed yet — upload it above to process" if f not in indexed_pdfs else ""
        st.markdown(
            f'<div style="margin:-0.5rem 0 0.5rem 1.7rem;font-size:0.72rem;color:#9aa0b4;">'
            f'{size_mb:.1f} MB{note}</div>',
            unsafe_allow_html=True,
        )
        if checked:
            selected_pdfs.append(f)

    if pdf_files:
        if len(selected_pdfs) == len(pdf_files):
            st.markdown('<span class="scope-badge">Searching all documents</span>', unsafe_allow_html=True)
        elif len(selected_pdfs) == 1:
            st.markdown(f'<span class="scope-badge">Searching only: {selected_pdfs[0]}</span>', unsafe_allow_html=True)
        elif selected_pdfs:
            st.markdown(f'<span class="scope-badge">Searching {len(selected_pdfs)} of {len(pdf_files)} documents</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="scope-badge">⚠️ No documents selected</span>', unsafe_allow_html=True)

    n_chunks = len(metadata_now)
    st.markdown("### 📊 Index Stats")
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-box"><div class="num">{len(pdf_files)}</div><div class="lbl">PDFs</div></div>
        <div class="stat-box"><div class="num">{n_chunks}</div><div class="lbl">Chunks</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Chunks to retrieve (k)", min_value=3, max_value=15, value=8)
    show_sources = st.toggle("Show source citations", value=True)

    with st.expander("Advanced"):
        st.caption("Rebuilds everything from scratch — re-reads and re-embeds every PDF in `pdfs/`. Use this only if you changed chunking settings or suspect the index is out of sync.")
        if st.button("🔁 Full rebuild (all files)", use_container_width=True):
            with st.spinner("Rebuilding from scratch — this re-processes every PDF…"):
                try:
                    # Wipe derived state, then reprocess every file in the folder
                    for p in (EXTRACTED_FILE, CHUNKS_FILE, METADATA_FILE, EMBEDDINGS_FILE, INDEX_FILE):
                        if os.path.exists(p):
                            os.remove(p)
                    clear_index_caches()
                    process_new_pdfs(pdf_files)
                    st.success("Full rebuild complete.")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Rebuild failed: {e}")

    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    api_key_present = bool(os.getenv("GOOGLE_API_KEY"))
    status_color = "#2ecc71" if api_key_present else "#e74c3c"
    status_text = "Gemini API connected" if api_key_present else "Missing GOOGLE_API_KEY"
    st.markdown(
        f'<div style="margin-top:1rem;font-size:0.8rem;color:{status_color};">'
        f'● {status_text}</div>',
        unsafe_allow_html=True,
    )

# ========================================================================
# HERO HEADER
# ========================================================================
st.markdown("""
<div class="hero">
    <h1>📚 MultiPDF RAG Assistant</h1>
    <p>Upload any PDF and it's automatically read, chunked, embedded, and indexed — then ask questions grounded in the actual text, with page-level citations.</p>
    <div class="badges">
        <span class="pill">🔍 FAISS Vector Search</span>
        <span class="pill">🌐 Multilingual Embeddings</span>
        <span class="pill">✨ Gemini 2.5 Flash</span>
        <span class="pill">📄 OCR Fallback</span>
        <span class="pill">⚡ Auto-indexing on upload</span>
    </div>
</div>
""", unsafe_allow_html=True)

if not pdf_files:
    st.info("👋 No PDFs yet — upload one from the sidebar and it'll be processed automatically.")

# ========================================================================
# CHAT
# ========================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "✨"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources") and show_sources:
            with st.expander(f"📎 {len(msg['sources'])} source(s) used"):
                for s in msg["sources"]:
                    pct = max(0, min(100, (s["score"] + 1) / 2 * 100))
                    st.markdown(f"""
                    <div class="src-card">
                        <div class="src-top">
                            <span class="src-name">📄 {s['pdf_name']}</span>
                            <span class="src-page">Page {s['page']}</span>
                        </div>
                        <div class="src-text">{s['text']}…</div>
                        <div class="score-track"><div class="score-fill" style="width:{pct:.0f}%;"></div></div>
                    </div>
                    """, unsafe_allow_html=True)

chat_placeholder = (
    f"Ask something about {selected_pdfs[0]}…" if len(selected_pdfs) == 1
    else "Ask something about your PDFs…"
)

if prompt := st.chat_input(chat_placeholder, disabled=not selected_pdfs):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="✨"):
        placeholder = st.empty()
        placeholder.markdown("_Searching documents and thinking…_")
        try:
            start = time.time()
            answer, sources = ask_question(prompt, k=top_k, selected_pdfs=selected_pdfs)
            elapsed = time.time() - start
        except Exception as e:
            answer, sources, elapsed = f"⚠️ Something went wrong: `{e}`", [], 0

        placeholder.markdown(answer)
        st.caption(f"⏱️ {elapsed:.1f}s · retrieved {len(sources)} chunk(s)")

        if sources and show_sources:
            with st.expander(f"📎 {len(sources)} source(s) used", expanded=False):
                for s in sources:
                    pct = max(0, min(100, (s["score"] + 1) / 2 * 100))
                    st.markdown(f"""
                    <div class="src-card">
                        <div class="src-top">
                            <span class="src-name">📄 {s['pdf_name']}</span>
                            <span class="src-page">Page {s['page']}</span>
                        </div>
                        <div class="src-text">{s['text']}…</div>
                        <div class="score-track"><div class="score-fill" style="width:{pct:.0f}%;"></div></div>
                    </div>
                    """, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
