#to generate a summary of the docuemnt uploaded
import json
from src.ollama_integration import ask_llm_with_context

def summarize_all_documents():
    # Load all chunks
    with open("data/outputs/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Concatenate all text
    combined_text = "\n\n".join([c["text"] for c in chunks])

    # Call LLM
    result = ask_llm_with_context("Summarize all documents", [{"text": combined_text}])
    return result
