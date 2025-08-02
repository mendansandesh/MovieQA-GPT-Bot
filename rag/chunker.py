def chunk_text(cleaned_text: str, max_chunk_tokens: int = 500, overlap_tokens: int = 50) -> list:
    """
    Splits text into overlapping chunks based on approximate token limits.

    Returns list of dicts: {'chunk_id': int, 'text': str}
    """
    words = cleaned_text.split()
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(words):
        end = start + max_chunk_tokens
        chunk_words = words[start:end]
        chunks.append({'chunk_id': chunk_id, 'text': ' '.join(chunk_words)})
        chunk_id += 1
        # overlap
        start = end - overlap_tokens
    return chunks