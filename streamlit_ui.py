import streamlit as st
import os
import json
import numpy as np
from src.pdf_reader import extract_text_from_pdf
from src.text_splitter import chunk_text
from src.embedder import get_embedding
from src.search_engine import search
from src.ollama_integration import ask_llm_with_context
from src.summarizer import summarize_all_documents

st.set_page_config(page_title="Smart PDF Inquiry Hub", layout="wide")
st.title("Smart PDF Exploration & Summarization")
st.markdown(
    "Upload multiple PDFs, process them, ask intelligent questions, or generate a combined summary offline."
)

CHUNKS_FILE = "data/outputs/chunks.json"
EMBEDDINGS_FILE = "data/embeddings/chunks.npy"

# -------------------- Upload PDFs --------------------
uploaded_files = st.file_uploader(
    "Upload PDF files", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    if st.button("Process Documents"):
        with st.spinner("Processing PDFs..."):
            os.makedirs("data/outputs", exist_ok=True)
            os.makedirs("data/embeddings", exist_ok=True)

            all_chunks = []
            all_embeddings = []

            for uploaded_file in uploaded_files:
                pdf_path = f"uploaded_{uploaded_file.name}"
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.read())

                raw_text = extract_text_from_pdf(pdf_path)
                os.remove(pdf_path)

                if len(raw_text) > 0:
                    chunks = chunk_text(raw_text)
                    for chunk in chunks:
                        all_chunks.append({"file": uploaded_file.name, "text": chunk})
                        all_embeddings.append(get_embedding(chunk))

            if all_chunks:
                # Save chunks and embeddings
                with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
                    json.dump(all_chunks, f, ensure_ascii=False, indent=2)
                np.save(EMBEDDINGS_FILE, np.array(all_embeddings))
                st.success(
                    f" {len(all_chunks)} chunks created from {len(uploaded_files)} PDFs!"
                )
                st.session_state['processed'] = True
            else:
                st.error("No chunks could be created.")
else:
    st.info("Upload PDFs to get started.")

# -------------------- Ask Questions & Generate Answer --------------------
if 'processed' in st.session_state and st.session_state['processed']:
    st.subheader(" Ask a Question")
    query = st.text_input("Type your question here")

    if query:
        with st.spinner("Finding relevant information..."):
            results = search(query, top_k_per_doc=2)

        # Display top chunks
        with st.expander(" Relevant Chunks (click to view)", expanded=False):
            for r in results:
                st.markdown(f"** {r['file']}** — *Score: {r['score']:.2f}*")
                st.markdown(r["text"])
                st.markdown("---")

        # Generate AI answer
        st.subheader(" Generate AI Answer")
        if st.button("Generate AI Answer"):
            with st.spinner("Generating AI Answer..."):
                answer = ask_llm_with_context(query, results)
                bullet_points = [bp.strip() for bp in answer.replace("\n", ". ").split(".") if bp.strip()]
                for bp in bullet_points:
                    st.markdown(f"- {bp}")

    # -------------------- Combined Summary --------------------
    st.subheader("Generate Combined Summary")
    if st.button("Generate Combined Summary"):
        with st.spinner("Generating combined summary..."):
            summary = summarize_all_documents()
            st.markdown(f" Combined Summary\n{summary}")

else:
    st.info("Upload and process PDFs to get started.")
