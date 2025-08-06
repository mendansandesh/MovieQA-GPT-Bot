from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


class Retriever:
    def __init__(self, vectorstore: Chroma, embedding_model: HuggingFaceEmbeddings, top_k: int = 5):
        """
        Wrapper class to retrieve top-k relevant transcript chunks for a user question.
        """
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model
        self.top_k = top_k

        # LangChain retriever interface
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",  # you can use "mmr" or "similarity_score_threshold" if needed
            search_kwargs={"k": self.top_k}
        )

    def retrieve_top_k(self, user_question: str) -> List[Document]:
        """
        Given a user question, retrieve top-k relevant transcript chunks from the vector DB.
        Returns:
            List[Document]: A list of LangChain Document objects.
        """
        print(f"Invoking retriever with question: [{user_question}]")
        docs = self.retriever.invoke(user_question)
        return docs
