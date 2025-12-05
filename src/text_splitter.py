import re
from typing import List

def chunk_text(text: str, max_chunk_size: int = 500) -> List[str]:
    """
    Splits text into chunks of approximately max_chunk_size characters.
    Keeps sentence boundaries when possible.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
