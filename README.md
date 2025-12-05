

# Intelligent PDF Exploration Using Natural Language Queries

Offline PDF Question Answering, Summarization & Semantic Search

This application allows users to extract knowledge from PDF documents using **natural language queries**, perform **semantic search**, and generate **summaries**, all running **offline** for maximum privacy.
The system uses **SentenceTransformer embeddings**, **NumPy similarity**, and optional **local LLM refinement using the Mistral model via Ollama**.



# Features

* Offline question answering from one or multiple PDFs
* Semantic search using vector embeddings
* Automatic summarization of documents
* Optional LLM-enhanced answers using **Mistral (via Ollama)**
* Fast NumPy-based cosine similarity
* Clean and minimal Streamlit interface



#🧠 Tech Stack

* Python 3.10+
* Streamlit
* SentenceTransformers
* NumPy
* PyPDF2
* Ollama (Mistral model)



# Installation

#Clone the Repository

```bash
git clone https://github.com/Nikhil786890/Intelligent-PDF-Exploration-Using-Natural-Language-Queries
cd Intelligent-PDF-Exploration-Using-Natural-Language-Queries
```

#Create & Activate Virtual Environmen

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# OR
source venv/bin/activate   # Linux/Mac
```

#Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Optional: Enable LLM Mode (Mistral via Ollama)

You can enhance answer quality using **Mistral**, running completely offline.

#Install Ollama

[https://ollama.ai/](https://ollama.ai/)

#Download the Mistral model

```bash
ollama pull mistral
```

Your app will automatically use it when LLM mode is enabled.

---

# Usage

Start the application:

```bash
streamlit run app.py
```

Then:

1. Upload one or multiple PDFs
2. Ask questions using natural language
3. View semantic search results
4. Generate summaries
5. (Optional) Toggle LLM mode for deeper Mistral-powered reasoning

---

#Architecture Overview

### **PDF Loader**

Extracts raw text from uploaded PDFs.

### **Text Splitter**

Breaks long documents into manageable chunks for embedding.

### **Embedding Generator**

Uses SentenceTransformers to convert text chunks into numerical vectors.

### **Semantic Search Engine**

Uses NumPy cosine similarity to find the most relevant chunks.

### **LLM Integration (Mistral)**

If enabled, Ollama’s Mistral model refines, rewrites, or expands responses.

### **Streamlit UI**

Provides an interactive, minimal user interface.

---

# Project Structure

```
project/
│── app.py
│── README.md
│── requirements.txt
│── .gitignore
│
└── src/
      ├── __init__.py
      ├── embedder.py
      ├── pdf_reader.py
      ├── search_engine.py
      ├── summarizer.py
      ├── text_splitter.py
      └── ollama_integration.py
```

---

# Future Improvements

* GPU-accelerated embedding support
* Advanced summarization pipeline
* Multi-language support
* Vector store caching for large PDF collections
* Chat-style conversation memory

---

#Contributing

Feedback, enhancements, and bug reports are welcome.
Create an issue or submit a pull request anytime.

---

#License

Open-source under the **MIT License**.

---


