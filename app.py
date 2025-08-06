# app.py (CLI version)
import sys
from transcript.loader import fetch_transcript, clean_transcript
from rag.chunker import chunk_text
from rag.embedder import embed_chunks
from vectorstore.indexer import get_or_create_collection, upsert_chunks
from rag.retriever import Retriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


def load_and_index_transcript(video_id: str):
    print(f"Fetching transcript for video ID: {video_id}")
    raw = fetch_transcript(video_id)
    clean = clean_transcript(raw)
    print(f"Transcript Length: {len(clean)} characters")
    chunks = chunk_text(clean)
    embeddings = embed_chunks(chunks)

    collection = get_or_create_collection()
    upsert_chunks(collection, chunks, embeddings, video_id)

    print(f"Indexed {len(chunks)} chunks for video ID {video_id}")
    return collection


def main():
    if len(sys.argv) != 3:
        print("Usage: python app.py <YouTube_VIDEO_ID> <User_Question>")
        sys.exit(1)

    video_id = sys.argv[1]
    user_question = sys.argv[2]

    collection = load_and_index_transcript(video_id)

    # Initialize retriever
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./vectorstore", embedding_function=embedding_model)
    retriever = Retriever(vectorstore, embedding_model)

    print(f"Question: {user_question}")
    top_docs = retriever.retrieve_top_k(user_question)

    print("Top Retrieved Transcript Chunks:\n")
    for idx, doc in enumerate(top_docs, 1):
        print(f"[Chunk {idx}]\n{doc.page_content[:500]}\n{'-' * 60}")

if __name__ == "__main__":
    main()