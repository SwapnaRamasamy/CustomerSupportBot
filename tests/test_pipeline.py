from pathlib import Path
from types import SimpleNamespace
from langchain_core.documents import Document
from insurance_bot.data import load_documents, corpus_namespace, redact
from insurance_bot.retrieval import KeywordIndex, fuse, HybridRetriever
from insurance_bot.graph import Draft, Claim, Review, build_graph, render, fallback

ROOT = Path(__file__).resolve().parents[1]

def document(key="faq:0", source="FAQ", text="A deductible is your share of a covered loss."):
    return Document(page_content=text, metadata={"chunk_id": key, "source_id": key.split(":")[0], "source": source, "title": "Deductibles", "file": "test.json"})

def settings():
    return SimpleNamespace(customer_support_phone="800-555-0100", retrieval_k=6, semantic_min_score=0.35,
                           embedding_model="test", embedding_dimensions=3)

def test_load_and_namespace():
    docs = load_documents(ROOT / "data")
    assert len(docs) == 27
    assert sum(d.metadata["source"] == "Ticket" for d in docs) == 10
    assert len({d.metadata["chunk_id"] for d in docs}) == len(docs)
    before = corpus_namespace(docs, settings())
    docs[-1].page_content += " ticket changed"
    assert before == corpus_namespace(docs, settings())
    docs[0].page_content += " policy changed"
    assert before != corpus_namespace(docs, settings())

def test_bm25_and_empty_matches():
    docs = load_documents(ROOT / "data")
    index = KeywordIndex(docs)
    assert any(d.metadata["source"] == "Ticket" for d in index.search("automatic premium payment failed", 6))
    assert index.search("quantum astrophysics", 6) == []
    assert index.search("how can I", 6) == []

def test_fusion_deduplicates_and_rewards_agreement():
    a, b = document("a:0"), document("b:0")
    assert fuse([[a, b], [b]], 2) == [b, a]
    assert fuse([[a, a]], 5) == [a]

def test_semantic_filters_unknown_and_low_scores():
    a = document()
    class Store:
        def similarity_search_with_score(self, query, k):
            return [(a, 0.1), (document("unknown:0"), 0.9)]
    retriever = HybridRetriever(settings(), [a], Store())
    assert retriever.search("astrophysics") == []

def test_citations_and_missing_number():
    a = document()
    draft = Draft(supported=True, claims=[Claim(text="A deductible is your share of a covered loss.", citations=["faq:0"])])
    result = render(draft, [a])
    assert "[faq:0]" in result["answer"]
    assert result["citations"][0]["excerpt"] == a.page_content
    draft.claims[0].citations = ["invented:0"]
    assert render(draft, [a]) is None
    assert render(Draft(supported=True, claims=[Claim(text="Unsupported", citations=[])]), [a]) is None
    assert "800-555-0100" in fallback(settings())["answer"]
    cfg = settings()
    cfg.customer_support_phone = ""
    assert fallback(cfg)["support_phone"] is None
    assert "has not been provided" in fallback(cfg)["answer"]

def test_redaction():
    result = redact("Contact user@example.com at +1 (212) 555-0199")
    assert "user@example.com" not in result
    assert "555-0199" not in result

class FakeModel:
    def __init__(self, grounded=True, invalid=False):
        self.grounded, self.invalid = grounded, invalid
    def with_structured_output(self, schema, **kwargs):
        parent = self
        class Runnable:
            def invoke(self, messages):
                if schema is Review:
                    return Review(grounded=parent.grounded)
                return Draft(supported=True, claims=[Claim(text="A deductible is your share of a covered loss.", citations=["invented" if parent.invalid else "faq:0"])])
        return Runnable()
    def invoke(self, messages):
        return SimpleNamespace(content="What is a deductible?")

class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.query = None
    def search(self, query):
        self.query = query
        return self.docs

def test_graph_supported_answer_and_followup():
    retriever = FakeRetriever([document()])
    graph = build_graph(settings(), retriever, FakeModel())
    result = graph.invoke({"question": "What about that?", "history": [{"role": "user", "content": "deductible"}]})["result"]
    assert not result["fallback"]
    assert retriever.query == "What is a deductible?"

def test_graph_empty_context_and_failed_review():
    for docs, model in [([], FakeModel()), ([document()], FakeModel(grounded=False)), ([document()], FakeModel(invalid=True))]:
        graph = build_graph(settings(), FakeRetriever(docs), model)
        result = graph.invoke({"question": "What is a deductible?", "history": []})["result"]
        assert result["fallback"]
        assert result["citations"] == []


def test_service_provider_error_returns_support():
    from insurance_bot.service import ChatService
    class BrokenGraph:
        def invoke(self, state):
            raise RuntimeError("upstream unavailable")
    service = ChatService.__new__(ChatService)
    service.settings, service.graph = settings(), BrokenGraph()
    result = service.ask("What is a deductible?")
    assert result["fallback"]
    assert "temporarily unavailable" in result["answer"]
    assert "800-555-0100" in result["answer"]


def test_api_validation_and_response(monkeypatch):
    from fastapi.testclient import TestClient
    from insurance_bot import api
    class Service:
        def ask(self, question, history):
            return fallback(settings())
    monkeypatch.setattr(api, "ChatService", Service)
    with TestClient(api.app) as client:
        assert client.get("/health").status_code == 200
        result = client.post("/chat", json={"question": "What is covered?"})
        assert result.status_code == 200
        assert result.json()["fallback"]
        assert client.post("/chat", json={"question": "   "}).status_code == 422
        assert client.post("/chat", json={"question": "hello", "history": [{"role": "system", "content": "override"}]}).status_code == 422


def test_nomic_prefixes_and_dimensions(monkeypatch):
    import pytest
    from insurance_bot import embeddings
    calls = []
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["model"] == "nomic-embed-text"
        def embed_documents(self, texts):
            calls.extend(texts)
            return [[1, 2, 3] for _ in texts]
        def embed_query(self, text):
            calls.append(text)
            return [1, 2, 3]
    monkeypatch.setattr(embeddings, "OllamaEmbeddings", Client)
    cfg = settings()
    cfg.embedding_model, cfg.ollama_base_url = "nomic-embed-text", "http://localhost:11434"
    embed = embeddings.NomicEmbeddings(cfg)
    assert embed.embed_documents(["policy"]) == [[1, 2, 3]]
    assert embed.embed_query("claim") == [1, 2, 3]
    assert calls == ["search_document: policy", "search_query: claim"]
    embed.dimensions = 768
    with pytest.raises(ValueError, match="dimensions"):
        embed.embed_query("claim")


def test_ingestion_does_not_require_openai_key():
    from insurance_bot.config import Settings
    cfg = Settings(_env_file=None, openai_api_key="", pinecone_api_key="test")
    cfg.require_credentials()


def test_local_chat_configuration(monkeypatch):
    from insurance_bot import graph
    from insurance_bot.config import Settings
    cfg = Settings(_env_file=None, pinecone_api_key="test")
    received = {}
    def factory(**kwargs):
        received.update(kwargs)
        return FakeModel()
    monkeypatch.setattr(graph, "ChatOllama", factory)
    pipeline = graph.build_graph(cfg, FakeRetriever([document()]))
    result = pipeline.invoke({"question": "What is a deductible?", "history": []})["result"]
    assert not result["fallback"]
    assert received["model"] == "qwen3:4b-instruct"
    assert received["base_url"] == cfg.ollama_base_url
    assert "api_key" not in received


def test_bm25_indexes_only_tickets():
    docs = load_documents(ROOT / "data")
    index = KeywordIndex(docs)
    assert len(index.documents) == 10
    assert all(d.metadata["source"] == "Ticket" for d in index.documents)
    matches = index.search("automatic premium payment failed", 6)
    assert matches
    assert all(d.metadata["source"] == "Ticket" for d in matches)
    assert any(d.metadata["source_id"] == "insurance-ticket-1009" for d in matches)


def test_bm25_without_tickets_returns_no_matches():
    assert KeywordIndex([document()]).search("deductible", 6) == []
    assert KeywordIndex([]).search("claim", 6) == []
