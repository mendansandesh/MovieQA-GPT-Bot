import sys
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from rag.embedder import embed_chunks
from rag.chunker import chunk_text

# Constants for chunking
MAX_CHUNK_TOKENS = 500  # approx token count per chunk
OVERLAP_TOKENS = 50     # approximate tokens of overlap

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
        # Try to fetch the first available transcript if none found for preferred languages
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transcript/loader.py <YouTube_VIDEO_ID>")
        sys.exit(1)

    vid = sys.argv[1]
    raw = fetch_transcript(vid)
    clean = clean_transcript(raw)
    chunks = chunk_text(clean)
    embeddings = embed_chunks(chunks)

    print(f"Clean transcript for video ID {vid}: " + clean)
    print(f"Created {len(chunks)} chunks and embeddings.")
    print("🔍 Preview Embedding Vector for chunk 0:")
    print(embeddings[0])