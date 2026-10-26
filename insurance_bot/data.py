import hashlib
import json
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

FILES = {"FAQ": "car_insurance_faqs.json", "Manual": "car_insurance_manuals.json", "Ticket": "car_insurance_tickets.json"}

def redact(text):
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL REDACTED]", text)
    return re.sub(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)", "[NUMBER REDACTED]", text)

def load_documents(data_dir):
    documents, seen = [], set()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    for source, filename in FILES.items():
        rows = json.loads((data_dir / filename).read_text(encoding="utf-8-sig"))
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{filename} must contain a non-empty JSON array")
        for row in rows:
            if any(not isinstance(row.get(key), str) or not row[key].strip() for key in ("id", "title", "content")):
                raise ValueError(f"Invalid document in {filename}")
            if row["id"] in seen:
                raise ValueError(f"Duplicate source ID: {row['id']}")
            seen.add(row["id"])
            title = redact(row["title"]) if source == "Ticket" else row["title"]
            content = redact(row["content"]) if source == "Ticket" else row["content"]
            chunks = splitter.split_text(content) if source == "Manual" else [content]
            for position, chunk in enumerate(chunks):
                documents.append(Document(page_content=f"{title}\n{chunk}", metadata={
                    "source_id": row["id"], "chunk_id": f"{row['id']}:{position}",
                    "source": source, "title": title, "file": filename,
                }))
    return documents

def corpus_namespace(documents, settings):
    # A changed corpus/model gets a new namespace; old vectors cannot leak into retrieval.
    payload = [(d.metadata, d.page_content) for d in documents if d.metadata["source"] != "Ticket"]
    payload += ["ollama-nomic-prefix-v1", settings.embedding_model, settings.embedding_dimensions]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
    return f"insurance-{digest}"
