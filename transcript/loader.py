import sys
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from sentence_transformers import SentenceTransformer

# Constants for chunking
MAX_CHUNK_TOKENS = 500  # approx token count per chunk
OVERLAP_TOKENS = 50     # approximate tokens of overlap

# Load local embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def fetch_transcript(video_id: str, languages: list = ['en']) -> str:
    """
    Fetches the transcript for a given YouTube video ID.
    Returns raw concatenated transcript.
    """
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        for lang in languages:
            try:
                transcript = transcript_list.find_transcript([lang])
                fetched = transcript.fetch()
                return " ".join([s.text for s in fetched.snippets])
            except NoTranscriptFound:
                continue
        # Fallback to first available
        transcript = transcript_list.find_transcript(
            [t.language_code for t in transcript_list]
        )
        fetched = transcript.fetch()
        return " ".join([s.text for s in fetched.snippets])
    except (VideoUnavailable, TranscriptsDisabled, NoTranscriptFound):
        raise ValueError(f"Transcript not available for video ID: {video_id}")

def clean_transcript(raw_text: str) -> str:
    """
    Cleans raw transcript: removes annotations, timestamps, extra whitespace.
    """
    text = re.sub(r"\[.*?\]", "", raw_text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(cleaned_text: str, max_chunk_tokens: int = MAX_CHUNK_TOKENS,
               overlap_tokens: int = OVERLAP_TOKENS) -> list:
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
        start = end - overlap_tokens
    return chunks

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