"""Run a repeatable, local-model evaluation against the actual RAG graph.
Expected-fact coverage is a lexical proxy, not an independent correctness judge.
"""
import argparse
import csv
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from .config import Settings, ROOT
from .data import load_documents, corpus_namespace
from .retrieval import HybridRetriever, vector_store
from .graph import build_graph, fallback


def divide(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize(rows):
    answerable = [r for r in rows if not r["should_fallback"]]
    answered = [r for r in rows if not r["result"]["fallback"] and not r["error"]]
    expected_fallback = [r for r in rows if r["should_fallback"]]
    fallback_rows = [r for r in rows if r["result"]["fallback"] and not r["error"]]
    true_fallback = sum(r["should_fallback"] for r in fallback_rows)
    times = sorted(r["latency_seconds"] for r in rows)
    return {
        "queries": len(rows), "answerable_queries": len(answerable), "expected_fallback_queries": len(expected_fallback),
        "errors": sum(bool(r["error"]) for r in rows),
        "retrieval_hit_at_k": divide(sum(r["retrieval_hit"] for r in answerable), len(answerable)),
        "mean_reciprocal_rank_at_k": divide(sum(r["reciprocal_rank"] for r in answerable), len(answerable)),
        "expected_source_recall_at_k": divide(sum(r["expected_source_recall"] for r in answerable), len(answerable)),
        "answer_rate_on_answerable": divide(sum(not r["result"]["fallback"] and not r["error"] for r in answerable),len(answerable)),
        "fallback_precision": divide(true_fallback,len(fallback_rows)),
        "fallback_recall": divide(true_fallback,len(expected_fallback)),
        "routing_accuracy": divide(sum(not r["error"] and r["should_fallback"] == r["result"]["fallback"] for r in rows),len(rows)),
        "citation_validity_on_answers": divide(sum(r["citations_valid"] for r in answered),len(answered)),
        "mean_expected_fact_coverage_on_answerable": divide(sum(r["fact_coverage"] for r in answerable),len(answerable)),
        "automated_resolution_proxy": divide(sum(r["resolution_proxy"] for r in rows),len(rows)),
        "latency_median_seconds": statistics.median(times) if times else None,
        "latency_p95_seconds": times[max(0, math.ceil(0.95*len(times))-1)] if times else None,
        "total_query_seconds": sum(times),
    }


def score(case, state, elapsed, error=None):
    result = state.get("result", fallback(Settings()))
    documents = state.get("documents", [])
    retrieved = list(dict.fromkeys(d.metadata["source_id"] for d in documents))
    expected = set(case["expected_source_ids"])
    ranks = [i+1 for i,key in enumerate(retrieved) if key in expected]
    chunks = {d.metadata["chunk_id"]: d for d in documents}
    citations = result.get("citations", [])
    valid = bool(citations) and all(c["id"] in chunks and c["excerpt"] == chunks[c["id"]].page_content and f"[{c['id']}]" in result["answer"] for c in citations)
    text = result["answer"].lower()
    groups = case["expected_fact_groups"]
    fact_hits = [any(term in text for term in group) for group in groups]
    coverage = sum(fact_hits)/len(groups) if groups and not result["fallback"] else 0.0
    resolved = not error and (result["fallback"] if case["should_fallback"] else not result["fallback"] and valid and coverage == 1)
    return {**case, "query": state.get("query",case["question"]), "retrieved_source_ids": retrieved,
            "retrieved_chunk_ids": list(chunks), "result": result, "error": error,
            "latency_seconds": round(elapsed,3), "retrieval_hit": bool(ranks),
            "reciprocal_rank": 1/min(ranks) if ranks else 0.0,
            "expected_source_recall": len(expected.intersection(retrieved))/len(expected) if expected else 0.0,
            "citations_valid": valid, "fact_groups_matched": fact_hits,
            "fact_coverage": coverage, "resolution_proxy": bool(resolved)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=ROOT/"evaluation"/"queries.json")
    parser.add_argument("--output",type=Path,default=ROOT/"evaluation"/"results")
    args = parser.parse_args()
    cases = json.loads(args.dataset.read_text(encoding="utf-8"))
    settings = Settings()
    documents = load_documents(settings.data_dir)
    retriever = HybridRetriever(settings, documents, vector_store(settings, documents))
    graph = build_graph(settings,retriever)
    args.output.mkdir(parents=True,exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.output/f"run-{run_id}.json"
    rows=[]
    metadata = {"run_id":run_id,"chat_model":settings.chat_model,"embedding_model":settings.embedding_model,
                "pinecone_index":settings.pinecone_index,"namespace":corpus_namespace(documents,settings),
                "retrieval_k":settings.retrieval_k,"semantic_min_score":settings.semantic_min_score,
                "note":"Single run on 20 synthetic cases derived from the corpus; lexical resolution proxy requires human review. Source labels are non-exhaustive. No external judge used. Setup time excluded from latency."}
    for number,case in enumerate(cases,1):
        started=time.perf_counter()
        error=None
        try:
            state=graph.invoke({"question":case["question"],"history":case.get("history",[])})
        except Exception as exc:
            error=type(exc).__name__
            state={"result":fallback(settings)}
        row=score(case,state,time.perf_counter()-started,error)
        rows.append(row)
        path.write_text(json.dumps({"metadata":metadata,"summary":summarize(rows),"results":rows},indent=2),encoding="utf-8")
        print(f"{number}/{len(cases)} {case['id']}: fallback={row['result']['fallback']} retrieval_hit={row['retrieval_hit']} fact_coverage={row['fact_coverage']:.2f} seconds={row['latency_seconds']:.1f} error={error}",flush=True)
    csv_path=path.with_suffix(".csv")
    with csv_path.open("w",newline="",encoding="utf-8") as handle:
        fields=["id","question","should_fallback","fallback","retrieval_hit","reciprocal_rank","fact_coverage","citations_valid","resolution_proxy","latency_seconds","error","answer"]
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key:row.get(key,row["result"].get(key)) for key in fields})
    print(json.dumps(summarize(rows),indent=2),flush=True)
    print(f"Report: {path}",flush=True)
    print(f"CSV: {csv_path}",flush=True)

if __name__ == "__main__":
    main()
