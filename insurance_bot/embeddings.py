from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

class NomicEmbeddings(Embeddings):
    """Use Nomic retrieval task prefixes consistently for indexing and queries."""
    def __init__(self, settings):
        self.client = OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url,
                                       client_kwargs={"timeout": 120})
        self.dimensions = settings.embedding_dimensions

    def _validate(self, vectors):
        if any(len(vector) != self.dimensions for vector in vectors):
            raise ValueError("Ollama vector dimensions do not match EMBEDDING_DIMENSIONS. Use a matching Pinecone index.")
        return vectors

    def embed_documents(self, texts):
        return self._validate(self.client.embed_documents(["search_document: " + text for text in texts]))

    def embed_query(self, text):
        return self._validate([self.client.embed_query("search_query: " + text)])[0]
