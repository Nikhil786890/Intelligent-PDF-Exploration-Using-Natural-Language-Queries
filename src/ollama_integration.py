import subprocess

def is_ollama_available() -> bool:
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, check=True
        )
        return True
    except Exception:
        return False

def ask_llm_with_context(query, context_chunks, model="mistral"):
    if not is_ollama_available():
        return "Ollama local model is not available. Please ensure it is installed and running."

    context_text = "\n\n".join([c["text"] for c in context_chunks])

    prompt = (
        f"You are an AI assistant analyzing multiple research papers or documents.\n"
        f"Context from the documents:\n{context_text}\n\n"
        f"Question: {query}\n"
        f"Answer concisely and intelligently, extracting relevant info from each document. "
        f"Provide a structured, readable format."
    )

    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return result.stdout.decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        return f"Error calling Ollama: {e.stderr.decode('utf-8', errors='ignore')}"
