import logging
from .config import Settings
from .data import load_documents
from .retrieval import HybridRetriever, vector_store
from .graph import build_graph, fallback

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, settings=None):
        self.settings = settings or Settings()
        documents = load_documents(self.settings.data_dir)
        store = vector_store(self.settings, documents)
        self.graph = build_graph(self.settings, HybridRetriever(self.settings, documents, store))

    def ask(self, question, history=None):
        if not question.strip():
            raise ValueError("Question must not be blank")
        try:
            return self.graph.invoke({"question": question, "history": history or []})["result"]
        except Exception as exc:
            # Avoid logging questions, evidence, provider payloads, or credentials.
            logger.error("Chat pipeline failed (%s)", type(exc).__name__)
            result = fallback(self.settings)
            result["answer"] = "The support assistant is temporarily unavailable. " + result["answer"].split("information. ", 1)[-1]
            return result
