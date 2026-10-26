import re
import time
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi
from .embeddings import NomicEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from .data import corpus_namespace

STOPWORDS = set("a an the is are was were i my me you your do does did can could how what when where why to of in on for and or it this that please tell about".split())

def tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]

class KeywordIndex:
    def __init__(self, documents):
        self.documents = [d for d in documents if d.metadata.get("source") == "Ticket"]
        self.tokenized = [tokens(d.page_content) for d in self.documents]
        self.index = BM25Okapi(self.tokenized) if self.tokenized and any(self.tokenized) else None

    def search(self, query, k):
        query_tokens = tokens(query)
        if not query_tokens or self.index is None:
            return []
        scores = self.index.get_scores(query_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        return [self.documents[i] for i in ranked if scores[i] > 0 and set(query_tokens).intersection(self.tokenized[i])][:k]

def fuse(rankings, k):
    scores, docs = {}, {}
    for ranking in rankings:
        seen = set()
        for rank, doc in enumerate(ranking, 1):
            key = doc.metadata["chunk_id"]
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
            docs[key] = doc
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    return [docs[key] for key in ordered[:k]]

def vector_store(settings, documents, create=False):
    settings.require_credentials()
    # Check local Ollama before creating any remote resources.
    embeddings = NomicEmbeddings(settings)
    embeddings.embed_query("connection check")
    client = Pinecone(api_key=settings.pinecone_api_key)
    if settings.pinecone_index not in client.list_indexes().names():
        if not create:
            raise ValueError("Pinecone index missing. Run python -m insurance_bot ingest first.")
        client.create_index(name=settings.pinecone_index, dimension=settings.embedding_dimensions,
                            metric="cosine", spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region))
    deadline = time.monotonic() + 120
    description = client.describe_index(settings.pinecone_index)
    while not description.status["ready"]:
        if time.monotonic() > deadline:
            raise TimeoutError("Pinecone index is not ready; retry later.")
        time.sleep(2)
        description = client.describe_index(settings.pinecone_index)
    if description.dimension != settings.embedding_dimensions or description.metric != "cosine":
        raise ValueError("Pinecone index must use cosine similarity and the configured embedding dimensions.")
    return PineconeVectorStore(index=client.Index(settings.pinecone_index), embedding=embeddings,
                               namespace=corpus_namespace(documents, settings))

class HybridRetriever:
    def __init__(self, settings, documents, store):
        self.settings, self.store = settings, store
        self.known = {d.metadata["chunk_id"]: d for d in documents if d.metadata["source"] != "Ticket"}
        self.keyword = KeywordIndex(documents)

    def search(self, query):
        def semantic():
            hits = self.store.similarity_search_with_score(query, k=self.settings.retrieval_k)
            return [self.known[d.metadata["chunk_id"]] for d, score in hits
                    if score >= self.settings.semantic_min_score and d.metadata.get("chunk_id") in self.known]
        with ThreadPoolExecutor(max_workers=2) as pool:
            dense = pool.submit(semantic)
            sparse = pool.submit(self.keyword.search, query, self.settings.retrieval_k)
            return fuse([dense.result(), sparse.result()], self.settings.retrieval_k)
