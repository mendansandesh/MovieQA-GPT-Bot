import sys
from transcript.loader import fetch_transcript, clean_transcript
from rag.chunker import chunk_text
from rag.embedder import embed_chunks
from vectorstore.indexer import get_or_create_collection, upsert_chunks

def main():
    if len(sys.argv) != 2:
        print("Usage: python transcript/loader.py <YouTube_VIDEO_ID>")
        sys.exit(1)

    video_id = sys.argv[1]
    raw = fetch_transcript(video_id)
    clean = clean_transcript(raw)
    chunks = chunk_text(clean)
    embeddings = embed_chunks(chunks)
    collection = get_or_create_collection()
    upsert_chunks(collection, chunks, embeddings, video_id)
    
    print(f"Clean transcript for video ID {video_id}: " + clean)
    print(f"Created {len(chunks)} chunks and embeddings.")
    print("Preview Embedding Vector for chunk 0:")
    print(embeddings[0])    

if __name__ == "__main__":
    main()    