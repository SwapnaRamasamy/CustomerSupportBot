# Car insurance support RAG

Answers use only the JSON files in `data/`. FAQs/manuals are embedded in Pinecone; local BM25 searches tickets plus FAQs/manuals. Rankings are merged with reciprocal rank fusion. LangGraph runs query rewriting, retrieval, structured generation, citation validation, and a separate grounding review. Each answer claim has application-rendered source citations, with titles, filenames and excerpts in the API response. Missing/conflicting evidence routes to customer support.

## Setup (PowerShell)

Python 3.11+ is recommended (3.10 minimum). The currently detected Python 3.9 cannot run these library versions.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock.txt
Copy-Item .env.example .env  # New setup only; preserve an existing .env
```

An isolated Python 3.11 environment is already available in `.venv` in this workspace. Activate it with `.\.venv\Scripts\Activate.ps1`; skip environment creation when using it. `requirements.lock.txt` records the tested dependency versions; `requirements.txt` defines allowed upgrade ranges.

Fill in `.env`: `PINECONE_API_KEY`, and the real `CUSTOMER_SUPPORT_PHONE`. No support number was present in the supplied data, so the fallback explicitly says it is unavailable until configured. The local chat model and Ollama endpoint are configurable. Embeddings run locally through Ollama using `nomic-embed-text`; chat and grounding review also run locally through Ollama. Never commit `.env`.

```powershell
python -m insurance_bot check-data
python -m insurance_bot ingest
python -m insurance_bot chat
python -m uvicorn insurance_bot.api:app --host 127.0.0.1 --port 8000
```

The ingest command creates a serverless cosine Pinecone index if missing (may incur provider charges). Existing indexes must match the configured embedding dimensions and cosine metric. Only FAQs/manuals are uploaded. Upserts have deterministic chunk IDs; a corpus/model fingerprint isolates changed content in a new namespace. Re-run ingestion whenever FAQs, manuals, chunking, or embeddings change. Old namespaces are retained and can be removed manually after review. Tickets reload at application startup. Pinecone writes are eventually consistent.

Open `http://127.0.0.1:8000/docs` to try the API. POST `/chat`:

```json
{"question":"How do I file an auto insurance claim?","history":[]}
```

Response fields: `answer`, `citations`, `fallback`, `support_phone`. For follow-ups, send at most eight prior `{role, content}` turns. History resolves references only and is never authoritative evidence. `/health` confirms the app started, not live upstream availability. The API binds locally; add authentication and request limits before external deployment.

## Validation

```powershell
python -m pytest -q
```

Tests use local data and fake model/retriever outputs, without API charges. Live evaluation also requires credentials: check deductible/claim questions, ticket-specific troubleshooting, an unrelated question, an unsupported policy-specific payout, and prompt injection. Compare every answer against its returned excerpts. Model-based grounding reduces unsupported answers but cannot guarantee their absence; tune `SEMANTIC_MIN_SCORE` with representative questions rather than interpreting cosine or fusion scores as confidence.

Ticket redaction covers email addresses and long numeric identifiers, not all personal data. The supplied tickets are generic resolution examples. Review/anonymize real tickets before replacing them; retrieved ticket excerpts are sent to the configured model provider. Local BM25 is rebuilt from JSON on startup and uses no pickle files.

Integration references: [LangGraph graph API](https://docs.langchain.com/oss/python/langgraph/graph-api), [PineconeVectorStore](https://reference.langchain.com/python/langchain-pinecone/vectorstores/PineconeVectorStore).

## Local Nomic embeddings through Ollama

Install Ollama from https://ollama.com/download/windows, then run:

```powershell
ollama pull nomic-embed-text
# If Ollama is not already running, run ollama serve in a separate terminal.
python -m insurance_bot ingest
python -m insurance_bot chat
```

Configuration: `EMBEDDING_MODEL=nomic-embed-text`, `EMBEDDING_DIMENSIONS=768`, `OLLAMA_BASE_URL=http://localhost:11434`, `PINECONE_INDEX=car-insurance-nomic`.
The original `faqsandmanuals` index has 512 dimensions and is preserved. Ingestion creates a separate 768-dimensional cosine index; your Pinecone plan must allow another index. No vectors from the old embedding model are reused. If you hit an index limit, select another compatible existing index or free capacity manually. Never delete existing data automatically.

Nomic uses `search_document:` for corpus text and `search_query:` for questions. Both ingestion and retrieval validate the returned dimensions. Embedding and ingestion require no OpenAI key or credits. Answer generation, follow-up rewriting, and grounding review use local `qwen3:4b-instruct`; no OpenAI key or credits are required. Re-index all FAQs/manuals after switching embeddings.

## Local answer generation

All model calls run through Ollama: `nomic-embed-text` for embeddings and `qwen3:4b-instruct` for answers, follow-up query rewriting, and grounding checks. Pinecone remains the hosted vector database and requires its API key (and is subject to its own plan limits).

```powershell
ollama pull qwen3:4b-instruct
.\.venv\Scripts\python.exe -m insurance_bot chat
```

Keep Ollama running (`ollama serve` if needed). Set `CHAT_MODEL=qwen3:4b-instruct`, `CHAT_TIMEOUT=300`, and `CHAT_CONTEXT_SIZE=8192` in `.env`. Initial model loading and CPU inference can be slow. No re-indexing is needed when changing only the chat model. Structured JSON schemas preserve claim citations and review routing. Local models can still make mistakes; inspect source excerpts for important questions. Legacy `OPENAI_API_KEY` values are ignored and can be removed from `.env`.

## Evaluate 20 queries

```powershell
.\.venv\Scripts\python.exe -m insurance_bot.evaluate
```

The labeled dataset is `evaluation/queries.json`: 15 answerable cases and five expected support fallbacks. The runner uses the production LangGraph, Ollama, and Pinecone paths, with no model changes or re-indexing. JSON checkpoints are saved after every question; a CSV is saved after completion under `evaluation/results/`. Each run gets a UTC timestamp and records model, index, namespace, retrieval settings, answers, citations, errors, and elapsed time. Original results are preserved on subsequent runs. Do not run multiple local-model evaluations at once.

Metrics include expected-source hit rate and recall at configured k, reciprocal rank, answer rate, fallback precision/recall, routing accuracy, citation validity, expected-fact coverage, and median/p95 latency. Retrieval metrics apply only to answerable queries. Expected-source labels are not exhaustive. Citation validity checks source existence and returned excerpts, not whether every claim is logically entailed. Expected-fact coverage uses case-insensitive phrase groups and can misjudge paraphrases or negations.

`automated_resolution_proxy` requires an answer with valid citations and all expected phrase groups for answerable questions, or a correct support fallback for unanswerable questions. Infrastructure errors always fail resolution. This proxy includes correct escalations and is not a production first-contact-resolution metric. Review saved answers against source text before assessing correctness/groundedness. This small synthetic corpus-derived evaluation does not establish general performance. No independent model judge is used. Timing includes retrieval, generation, and review, but excludes initial service setup.
