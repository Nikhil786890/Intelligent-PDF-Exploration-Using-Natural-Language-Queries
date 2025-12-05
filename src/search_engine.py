import json
import numpy as np
from src.embedder import get_embedding

CHUNKS_FILE = "data/outputs/chunks.json"
EMBEDDINGS_FILE = "data/embeddings/chunks.npy"

def load_chunks():
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)  # Returns list of {"file":..., "text":...}

def load_embeddings():
    return np.load(EMBEDDINGS_FILE)

def search(query: str, top_k_per_doc: int = 1):
    """
    Cross-paper search: pick top_k chunks per PDF based on similarity to query.
    Returns list of dicts: {"file": ..., "text": ..., "score": ...}
    """
    chunks = load_chunks()
    embeddings = load_embeddings()

    query_embedding = get_embedding(query)
    similarities = np.dot(embeddings, query_embedding)

    chunk_scores = [
        {"file": chunks[i]["file"], "text": chunks[i]["text"], "score": similarities[i]}
        for i in range(len(chunks))
    ]

    # Group by file
    grouped = {}
    for entry in chunk_scores:
        grouped.setdefault(entry["file"], []).append(entry)

    # Pick top chunks per file
    top_results = []
    for file, entries in grouped.items():
        entries.sort(key=lambda x: x["score"], reverse=True)
        top_results.extend(entries[:top_k_per_doc])

    # Sort globally by similarity
    top_results.sort(key=lambda x: x["score"], reverse=True)
    return top_results
