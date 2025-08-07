from typing import List, Dict
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
import os

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "movieqa_chunks"

# Load local embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_or_create_collection() -> Chroma:
    return Chroma(
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )

def delete_video_chunks(video_id: str):
    """
    Removes all chunks for a given video_id from the Chroma collection.
    """
    collection = get_or_create_collection()
    try:
        collection.delete(filter={"video_id": video_id})
        print(f"Deleted existing chunks for video ID: {video_id}")
    except Exception as e:
        print(f"Failed to delete chunks for {video_id}: {str(e)}")

def upsert_chunks(chunks: List[Dict], embeddings: List[Dict], video_id: str):
    """
    Inserts new chunks into the Chroma vector store after clearing existing ones.
    """
    # 1. Clear old chunks for the same video_id
    delete_video_chunks(video_id)

    # 2. Prepare texts and metadata
    texts = [c['text'] for c in chunks]
    metadatas = [{"video_id": video_id, "chunk_id": c['chunk_id']} for c in chunks]

    # 3. Add to Chroma
    collection = get_or_create_collection()
    collection.add_texts(texts=texts, metadatas=metadatas)

    print(f"Upserted {len(chunks)} chunks into Chroma for video ID: {video_id}")

def get_retriever(video_id: str = None, k: int = 3) -> VectorStoreRetriever:
    """
    Returns a retriever configured to fetch top-k documents.
    """
    collection = get_or_create_collection()
    retriever = collection.as_retriever(
        search_kwargs={
            "k": k,
            "filter": {"video_id": video_id} if video_id else {}
        }
    )
    return retriever

