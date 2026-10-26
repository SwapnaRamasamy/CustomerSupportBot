import argparse
import json
from .config import Settings
from .data import load_documents, corpus_namespace

def main():
    parser = argparse.ArgumentParser(description="Grounded car insurance RAG")
    parser.add_argument("command", choices=["ingest", "chat", "check-data"])
    args = parser.parse_args()
    settings = Settings()
    documents = load_documents(settings.data_dir)
    if args.command == "check-data":
        print(json.dumps({source: sum(d.metadata["source"] == source for d in documents) for source in ("FAQ", "Manual", "Ticket")}))
        return
    if args.command == "ingest":
        from .retrieval import vector_store
        store = vector_store(settings, documents, create=True)
        indexed = [d for d in documents if d.metadata["source"] != "Ticket"]
        store.add_documents(indexed, ids=[d.metadata["chunk_id"] for d in indexed])
        print(f"Upserted {len(indexed)} FAQ/manual chunks into {corpus_namespace(documents, settings)}. Tickets remain local.")
        print("Pinecone is eventually consistent; allow a short delay before chatting.")
        return
    from .service import ChatService
    service, history = ChatService(settings), []
    print("Car insurance support. Type exit to quit.")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = service.ask(question, history[-8:])
        print(result["answer"])
        history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": result["answer"]}])

if __name__ == "__main__":
    main()
