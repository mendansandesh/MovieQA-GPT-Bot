from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from typing import List

CHROMA_DIR = "./chroma_db"  # persistent directory

class Retriever:
    def __init__(self, persist_dir: str = CHROMA_DIR):
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
        )
        self.retriever: VectorStoreRetriever = self.vectorstore.as_retriever()

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        return self.retriever.invoke(query)
