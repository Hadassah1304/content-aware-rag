import os

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import dotenv
import uuid
import numpy as np

class Storage:
    """
    Chroma DB storage
    """

    def __init__(self,collection_name: str):
        dotenv.load_dotenv()

        self.chroma_client = chromadb.PersistentClient(path=f".chromadb")
        self.collection_name = collection_name

        # Google Embedding Function by Default
        self.embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model_name="gemini-embedding-001"
        )

    @staticmethod
    def generate_random_ids(n: int) -> list[str]:
        """
        Generate n random ids
        """
        return [str(uuid.uuid4()) for _ in range(n)]

    @staticmethod
    def create_collection(collection_name: str):
        """
        Create a new collection in Chroma DB
        """
        chroma_client = chromadb.PersistentClient(path=f".chromadb")
        embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model_name="gemini-embedding-001"
        )
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        return {"status": "success", "message": f"Created collection {collection.name}"}

    @staticmethod
    def delete_collection(collection_name: str):
        """
        Delete a collection from Chroma DB
        """
        chroma_client = chromadb.PersistentClient(path=f".chromadb")
        chroma_client.delete_collection(name=collection_name)

        return {"status": "success", "message": f"Deleted collection {collection_name}"}

    def add_document(self, document_contents: list[str], metadatas: list[dict] = None, embeddings: list[list[float]] = None):
        """
        Add a document to a collection
        """
        collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )

        document_ids = self.generate_random_ids(len(document_contents))

        collection.add(
            documents=document_contents,
            metadatas=metadatas,
            ids=document_ids,
            embeddings=embeddings
        )

        return {
            "status": "success",
            "message": f"Added {len(document_ids)} documents to collection {self.collection_name}",
            "document_ids": document_ids
        }

    def query_collection(self, query_texts: list[str], n_results: int = 5):
        """
        Query a collection
        """
        collection = self.chroma_client.get_collection(name=self.collection_name, embedding_function=self.embedding_function)
        results = collection.query(
            query_texts=query_texts,
            n_results=n_results
        )

        return results["documents"]