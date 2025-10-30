# transcript/loader.py
import os
import json
import re
from typing import List, Dict, Optional

# Use the same API you currently have available:
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable

# LangChain splitter (already in your repo)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Local embedding model (you already use MiniLM)
from sentence_transformers import SentenceTransformer
import numpy as np

# Configuration (tweak if needed)
CACHE_DIR = "./transcript"
CHUNK_CHAR_LIMIT = 1000   # ~ safe for MiniLM (<512 tokens typically)
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # local HF model you already use

# initialize embedding model once
_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def _snippets_to_serializable(snippets) -> List[Dict]:
    """
    Convert FetchedTranscript / FetchedTranscriptSnippet objects into serializable dicts:
    [{'text': ..., 'start': float, 'duration': float}, ...]
    """
    out = []
    for s in snippets:
        # snippet may expose attributes as .text, .start, .duration or as dict-like;
        # handle both gracefully
        try:
            text = s.text
            start = float(s.start) if hasattr(s, "start") else None
            duration = float(s.duration) if hasattr(s, "duration") else None
        except Exception:
            # fallback: if s is a dict-like
            text = s.get("text") if isinstance(s, dict) else str(s)
            start = float(s.get("start")) if isinstance(s, dict) and s.get("start") is not None else None
            duration = float(s.get("duration")) if isinstance(s, dict) and s.get("duration") is not None else None

        out.append({"text": text, "start": start, "duration": duration})
    return out


def fetch_transcript_snippets(video_id: str, languages: List[str] = ["en"]) -> List[Dict]:
    """
    Fetch (and cache) transcript as list of snippets: [{'text', 'start', 'duration'}, ...]
    Uses the api instance + transcript_list.find_transcript(...).fetch() flow (no get_transcript()).
    Raises ValueError if transcript not available.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{video_id}.json")

    # Try cache first
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # verify it's a list of dicts
            if isinstance(data, list):
                return data
        except Exception:
            # fall through and refetch
            pass

    # Fetch using the API methods you have
    try:
        api = YouTubeTranscriptApi()
        # transcript_list = api.list(video_id)  # older style you had used
        # prefer to use api.list() since that's available on your install
        transcript_list = api.list(video_id)

        # try languages in order
        for lang in languages:
            try:
                transcript = transcript_list.find_transcript([lang])
                fetched = transcript.fetch()  # returns FetchedTranscript (object with .snippets)
                snippets = getattr(fetched, "snippets", None)
                if snippets is None:
                    # If structure is different, try to_raw_data()
                    try:
                        raw = fetched.to_raw_data()
                        snippets = raw
                    except Exception:
                        snippets = fetched  # fallback: iterate fetched directly
                serial = _snippets_to_serializable(snippets)
                # cache
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(serial, f, ensure_ascii=False, indent=2)
                return serial
            except NoTranscriptFound:
                continue

        # fallback: try to pick first available transcript in the list
        try:
            first_langs = [t.language_code for t in transcript_list]
            transcript = transcript_list.find_transcript(first_langs)
            fetched = transcript.fetch()
            snippets = getattr(fetched, "snippets", None) or fetched
            serial = _snippets_to_serializable(snippets)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(serial, f, ensure_ascii=False, indent=2)
            return serial
        except Exception as e:
            raise ValueError(f"No transcript found for video {video_id}: {e}")

    except (VideoUnavailable, TranscriptsDisabled, NoTranscriptFound) as e:
        raise ValueError(f"Transcript not available for video {video_id}: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error fetching transcript for {video_id}: {e}")


def fetch_transcript_text(video_id: str, languages: List[str] = ["en"]) -> str:
    """
    Convenience: returns concatenated transcript text (all snippets joined by space).
    """
    snippets = fetch_transcript_snippets(video_id, languages=languages)
    return " ".join([s["text"] for s in snippets if s.get("text")])


def clean_transcript(raw_text: str) -> str:
    """
    Basic cleaning: remove bracketed annotations, parentheses, timestamps, collapse whitespace.
    """
    if raw_text is None:
        return ""
    text = re.sub(r"\[.*?\]", "", raw_text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(cleaned_text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[Dict]:
    """
    Split cleaned transcript into chunks (dicts with chunk_id + text).
    Uses LangChain's RecursiveCharacterTextSplitter to avoid producing chunks too large for embedding models.
    """
    if chunk_size is None:
        chunk_size = CHUNK_CHAR_LIMIT
    if overlap is None:
        overlap = CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    pieces = splitter.split_text(cleaned_text)
    return [{"chunk_id": i, "text": p} for i, p in enumerate(pieces)]


def embed_chunks(chunks: list) -> list:
    """
    Generates local embeddings using MiniLM.
    Returns list of dicts: {'chunk_id': int, 'embedding': np.ndarray}
    """
    texts = [chunk['text'][:480] for chunk in chunks]  #truncate long texts
    vectors = embedding_model.encode(texts, convert_to_numpy=True)

    return [
        {'chunk_id': chunk['chunk_id'], 'embedding': vector}
        for chunk, vector in zip(chunks, vectors)
    ]
