import streamlit as st
import os
import json
import numpy as np
from src.pdf_reader import extract_text_from_pdf
from src.text_splitter import chunk_text
from src.embedder import get_embedding
from src.search_engine import search
from src.ollama_integration import ask_llm_with_context, is_ollama_available
from src.summarizer import summarize_all_documents

# Page config
st.set_page_config(page_title="Smart PDF Explorer", layout="wide")

# Simple, clean CSS
st.markdown("""
<style>
    /* Clean header */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
    }
    
    .main-header p {
        color: white;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Simple stat boxes */
    .stat-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Clean message boxes */
    .user-msg {
        background: #667eea;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .ai-msg {
        background: #f8f9fa;
        color: #1e293b;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    
    .ai-msg p {
        color: #1e293b;
        margin: 0.5rem 0;
    }
    
    .ai-msg strong {
        color: #667eea;
    }
    
    /* Source box */
    .source-box {
        background: #fff8e1;
        padding: 0.8rem;
        border-radius: 6px;
        border-left: 3px solid #ffa726;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    /* Buttons */
    .stButton>button {
        background: #667eea;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        border: none;
    }
    
    .stButton>button:hover {
        background: #5568d3;
    }
</style>
""", unsafe_allow_html=True)

# File paths
CHUNKS_FILE = "data/outputs/chunks.json"
EMBEDDINGS_FILE = "data/embeddings/chunks.npy"

# Session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'total_chunks' not in st.session_state:
    st.session_state.total_chunks = 0

# Load existing data
def load_existing_data():
    if os.path.exists(CHUNKS_FILE) and os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            if len(chunks) > 0:
                files = list(set([chunk['file'] for chunk in chunks]))
                st.session_state.processed = True
                st.session_state.processed_files = files
                st.session_state.total_chunks = len(chunks)
                return True
        except:
            pass
    return False

load_existing_data()

# Header
st.markdown("""
<div class="main-header">
    <h1>📚 Smart PDF Explorer</h1>
    <p>Upload PDFs, ask questions, and get AI-powered answers</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Upload & Settings")
    
    # Ollama status
    ollama_status = is_ollama_available()
    if ollama_status:
        st.success("✅ Ollama Connected")
    else:
        st.error("❌ Ollama Not Available")
    
    st.markdown("---")
    
    # File upload
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process Documents", use_container_width=True):
            with st.spinner("Processing..."):
                os.makedirs("data/outputs", exist_ok=True)
                os.makedirs("data/embeddings", exist_ok=True)
                
                all_chunks = []
                all_embeddings = []
                
                progress_bar = st.progress(0)
                for idx, file in enumerate(uploaded_files):
                    pdf_path = f"uploaded_{file.name}"
                    with open(pdf_path, "wb") as f:
                        f.write(file.read())
                    
                    text = extract_text_from_pdf(pdf_path)
                    os.remove(pdf_path)
                    
                    if text:
                        chunks = chunk_text(text)
                        for chunk in chunks:
                            all_chunks.append({"file": file.name, "text": chunk})
                            all_embeddings.append(get_embedding(chunk))
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                if all_chunks:
                    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
                        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
                    np.save(EMBEDDINGS_FILE, np.array(all_embeddings))
                    
                    st.session_state.processed = True
                    st.session_state.processed_files = list(set([c['file'] for c in all_chunks]))
                    st.session_state.total_chunks = len(all_chunks)
                    
                    st.success(f"✅ Processed {len(all_chunks)} chunks!")
                    st.rerun()
                else:
                    st.error("No text extracted")
    
    st.markdown("---")
    
    # Settings
    st.subheader("Settings")
    top_k = st.slider("Results per document", 1, 5, 2)
    
    st.markdown("---")
    
    # Documents
    if st.session_state.processed_files:
        st.subheader("Loaded Documents")
        for file in st.session_state.processed_files:
            st.text(f"📄 {file}")
    
    st.markdown("---")
    
    # Clear buttons
    if st.button("Clear All", use_container_width=True):
        st.session_state.processed = False
        st.session_state.chat_history = []
        st.session_state.processed_files = []
        st.session_state.total_chunks = 0
        if os.path.exists(CHUNKS_FILE):
            os.remove(CHUNKS_FILE)
        if os.path.exists(EMBEDDINGS_FILE):
            os.remove(EMBEDDINGS_FILE)
        st.rerun()

# Stats
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-number">{len(st.session_state.processed_files)}</div>
        <div class="stat-label">Documents</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-number">{st.session_state.total_chunks}</div>
        <div class="stat-label">Chunks</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-number">{len(st.session_state.chat_history)}</div>
        <div class="stat-label">Queries</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main content
if st.session_state.processed:
    tab1, tab2 = st.tabs(["💬 Ask Questions", "📊 Summarize"])
    
    with tab1:
        st.subheader("Ask Questions")
        
        # Chat history
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                st.markdown(f'<div class="user-msg"><strong>You:</strong> {chat["question"]}</div>', 
                          unsafe_allow_html=True)
                st.markdown(f'<div class="ai-msg"><strong>AI:</strong><br>{chat["answer"]}</div>', 
                          unsafe_allow_html=True)
                
                if chat.get('sources'):
                    with st.expander(f"📚 View {len(chat['sources'])} sources"):
                        for idx, src in enumerate(chat['sources'], 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Source {idx}:</strong> {src['file']}<br>
                                <strong>Score:</strong> {int(src['score']*100)}%<br>
                                {src['text'][:200]}...
                            </div>
                            """, unsafe_allow_html=True)
        
        # Query input
        st.markdown("---")
        query = st.text_input("Your question:", placeholder="Ask the question?")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            search_clicked = st.button("🔍 Search", use_container_width=True)
        with col2:
            if 'last_results' in st.session_state and st.session_state.get('last_results'):
                generate_clicked = st.button("🤖 Answer", use_container_width=True)
            else:
                generate_clicked = False
        
        if search_clicked and query:
            with st.spinner("🔍 Searching..."):
                results = search(query, top_k_per_doc=top_k)
                st.session_state.last_results = results
            
            if results:
                st.success(f"✅ Found {len(results)} relevant chunks")
                with st.expander("📄 View retrieved chunks"):
                    for idx, r in enumerate(results, 1):
                        st.markdown(f"**{idx}. {r['file']}** (Score: {int(r['score']*100)}%)")
                        st.text(r['text'][:300] + "..." if len(r['text']) > 300 else r['text'])
                        st.markdown("---")
            else:
                st.warning("No relevant chunks found")
                st.session_state.last_results = []
        
        if generate_clicked and st.session_state.get('last_results'):
            if ollama_status:
                with st.spinner("🤖 Generating answer..."):
                    answer = ask_llm_with_context(query, st.session_state.last_results)
                    st.session_state.chat_history.append({
                        'question': query,
                        'answer': answer,
                        'sources': st.session_state.last_results
                    })
                    st.session_state.last_results = []
                    st.rerun()
            else:
                st.error("Ollama not available")
    
    with tab2:
        st.subheader("Generate Summary")
        st.info("Generate a summary of all uploaded documents")
        
        if st.button("Generate Summary", use_container_width=True):
            if ollama_status:
                with st.spinner("📊 Generating summary..."):
                    summary = summarize_all_documents()
                    st.markdown(f'<div class="ai-msg">{summary}</div>', unsafe_allow_html=True)
                    st.session_state.chat_history.append({
                        'question': '📊 Document Summary',
                        'answer': summary,
                        'sources': []
                    })
            else:
                st.error("Ollama not available")

else:
    st.info("👈 Upload PDF files from the sidebar to get started")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📤 Upload")
        st.write("Upload one or multiple PDF files")
    with col2:
        st.markdown("### 💬 Ask")
        st.write("Ask questions about your documents")
    with col3:
        st.markdown("### 📊 Analyze")
        st.write("Get AI-powered answers and summaries")

# --- 🧠 Mind Map Section ---
st.markdown("---")
st.subheader("🧠 Mind Map: How Smart PDF Explorer Works")

st.markdown("""
<style>
.mindmap {
    display: flex;
    flex-direction: column;
    align-items: center;
    font-family: 'Inter', sans-serif;
    color: #333;
}

.mindmap-node {
    background: #f8f9fa;
    border-left: 4px solid #667eea;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin: 0.6rem 0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    width: 80%;
}

.mindmap-node strong {
    color: #667eea;
}
</style>

<div class="mindmap">

<div class="mindmap-node">
<strong>📤 Step 1 – Upload PDFs:</strong>  
You upload one or multiple PDF files. The app extracts their text using <code>extract_text_from_pdf()</code>.
</div>

<div class="mindmap-node">
<strong>🧩 Step 2 – Chunk Text:</strong>  
The extracted text is split into smaller, meaningful sections using <code>chunk_text()</code>.  
This helps the app process large documents efficiently.
</div>

<div class="mindmap-node">
<strong>🧠 Step 3 – Generate Embeddings:</strong>  
Each text chunk is converted into a numerical vector using <code>get_embedding()</code>  
(from a Sentence Transformer model). These embeddings capture semantic meaning.
</div>

<div class="mindmap-node">
<strong>🔍 Step 4 – Semantic Search:</strong>  
When you ask a question, the app finds the most relevant chunks using cosine similarity in <code>search()</code>.
</div>

<div class="mindmap-node">
<strong>🤖 Step 5 – Context-Aware Answer:</strong>  
The retrieved chunks are passed to the local LLM (Ollama).  
Function <code>ask_llm_with_context()</code> composes a detailed answer using both your question and the context.
</div>

<div class="mindmap-node">
<strong>📚 Step 6 – Display Results:</strong>  
The AI-generated answer is displayed along with document sources and confidence scores.  
Your chat history and retrieved chunks are stored for re-use.
</div>

</div>
""", unsafe_allow_html=True)

# Optional: Mermaid visual diagram (uncomment to use)
# st.markdown("""
# ```mermaid
# graph TD
#     A[📄 Upload PDFs] --> B[🧩 Chunk Text]
#     B --> C[🔢 Generate Embeddings]
#     C --> D[🔍 Search Relevant Chunks]
#     D --> E[🤖 Ask LLM with Context]
#     E --> F[🧠 Display AI Answer + Sources]
# ```
# """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Smart PDF Explorer | Powered by Ollama & Sentence Transformers</div>",
    unsafe_allow_html=True
)
