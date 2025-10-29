# app.py (CLI version)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
from transcript.loader import fetch_transcript, clean_transcript
from rag.chunker import chunk_text
from rag.embedder import embed_chunks
from rag.qa_engine import generate_answer
from vectorstore.indexer import (
    get_or_create_collection,
    upsert_chunks,
    get_retriever
)

def load_and_index_transcript(video_id: str):
    print("-----------START------------------------------------")
    print(f"Fetching transcript for video ID: {video_id}")
    raw = fetch_transcript(video_id)
    clean = clean_transcript(raw)
    print(f"Transcript Length: {len(clean)} characters")
    chunks = chunk_text(clean)
    print(f"Chunked into {len(chunks)} chunks")
    for chunk in chunks:
        print(f"\n[Chunk {chunk['chunk_id']}]: {chunk['text'][:100]}...\n")

    embeddings = embed_chunks(chunks)
    
    upsert_chunks(chunks, embeddings, video_id)

    print(f"Indexed {len(chunks)} chunks for video ID {video_id}")
    return chunks

def answer_question(video_id: str, question: str):
    top_k = 3
    retriever = get_retriever(video_id=video_id, k=top_k)
    docs = retriever.get_relevant_documents(question)
    
    print(f"\nVideoID: {video_id}")
    print(f"\nQuestion: {question}")
    print("\nTop Relevant Chunks:\n")
    relevant_chunks = []
    for i, doc in enumerate(docs, 1):
        text = doc.page_content.strip()
        relevant_chunks.append(text)
        print(f"[Chunk {i}]\n{doc.page_content[:500]}\n{'-' * 60}")

    print("\n🤖 Generating Answer (using local model)...")
    answer = generate_answer(question, relevant_chunks, top_k=top_k)

    print("\n💡 Final Answer:")
    print(answer)    

def main():
    if len(sys.argv) != 3:
        print("Usage: python app.py <YouTube_VIDEO_ID> <User_Question>")
        sys.exit(1)

    video_id = sys.argv[1]
    user_question = sys.argv[2]

    load_and_index_transcript(video_id)
    answer_question(video_id, user_question)

if __name__ == "__main__":
    main()