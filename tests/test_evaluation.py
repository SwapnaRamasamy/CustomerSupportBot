from insurance_bot.evaluate import score, summarize
from insurance_bot.graph import fallback
from insurance_bot.config import Settings
from langchain_core.documents import Document


def test_scoring_retrieval_citations_and_fact_coverage():
    case = {"id":"test", "question":"deductible?", "should_fallback":False,
            "expected_source_ids":["faq"], "expected_fact_groups":[["amount"],["covered"]]}
    doc = Document(page_content="An amount paid for a covered loss", metadata={"source_id":"faq","chunk_id":"faq:0"})
    state = {"documents":[doc], "result":{"answer":"An amount paid for a covered loss [faq:0]", "fallback":False,
             "citations":[{"id":"faq:0","excerpt":doc.page_content}]}}
    row=score(case,state,2)
    assert row["resolution_proxy"]
    assert row["reciprocal_rank"] == 1
    assert row["fact_coverage"] == 1
    state["result"]["citations"][0]["id"]="invented"
    assert not score(case,state,2)["resolution_proxy"]


def test_errors_do_not_count_as_successful_fallbacks():
    case={"id":"unknown","question":"unknown?","should_fallback":True,
          "expected_source_ids":[],"expected_fact_groups":[]}
    state={"result":fallback(Settings(_env_file=None))}
    ok=score(case,state,1)
    failed=score(case,state,3,error="TimeoutError")
    result=summarize([ok,failed])
    assert result["fallback_recall"] == 0.5
    assert result["routing_accuracy"] == 0.5
    assert result["automated_resolution_proxy"] == 0.5
    assert result["retrieval_hit_at_k"] is None
    assert result["errors"] == 1
    assert result["latency_p95_seconds"] == 3
