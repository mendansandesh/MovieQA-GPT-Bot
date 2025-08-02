from sentence_transformers import SentenceTransformer

# Load local embedding model (Hugging Face)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(chunks: list) -> list:
    """
    Generates local embeddings using MiniLM.
    Returns list of dicts: {'chunk_id': int, 'embedding': np.ndarray}
    """
    texts = [chunk['text'] for chunk in chunks]
    vectors = embedding_model.encode(texts, convert_to_numpy=True)

    return [
        {'chunk_id': chunk['chunk_id'], 'embedding': vector}
        for chunk, vector in zip(chunks, vectors)
    ]