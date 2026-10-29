from app.ingestion import (
    chunk_by_structure,
    chunk_fixed_window,
    compute_content_hash,
    process_documents,
)
from app.rag_agent import InvestigationAgent, decompose_query, detect_contradictions
from app.search import HybridSearchEngine


def test_deduplication_and_chunking_strategies():
    docs = [
        {"doc_id": "d1", "text": "First paragraph about payment gateway.\n\nSecond paragraph about timeout."},
        {"doc_id": "d2", "text": "First paragraph about payment gateway.\n\nSecond paragraph about timeout."},
    ]
    chunks_struct = process_documents(docs, strategy="structure")
    # d2 დუბლიკატია და უნდა გაიფილტროს
    assert len(chunks_struct) == 1
    assert chunks_struct[0]["doc_id"] == "d1"

    long_text = " ".join([f"word{i}" for i in range(350)])
    fixed_chunks = chunk_fixed_window(long_text, chunk_size=180, overlap=40)
    struct_chunks = chunk_by_structure(long_text, max_words=200)
    assert len(fixed_chunks) >= 2
    assert len(struct_chunks) >= 2


def test_query_decomposition_and_contradictions():
    sub = decompose_query(
        "What system change caused the March outage and which downstream services were damaged?"
    )
    assert len(sub) >= 2

    conflicting_chunks = [
        {
            "doc_id": "report_a",
            "text": "The Redis cache cluster remained healthy with 0% packet loss and was ruled out.",
        },
        {
            "doc_id": "report_b",
            "text": "A contradiction shows the Redis cache cluster failed with 42% packet loss and outage.",
        },
    ]
    conflicts = detect_contradictions(conflicting_chunks)
    assert len(conflicts) >= 1
    assert conflicts[0]["source_a"] == "report_a"


def test_hybrid_search_and_rag_abstention():
    engine = HybridSearchEngine(collection_name="pytest_collection")
    sample_docs = [
        {
            "doc_id": "inc_100",
            "title": "Payment Gateway Outage",
            "text": "Release v4.8.2 exhausted PostgreSQL connections in payment-gateway-service causing 504 errors.",
        }
    ]
    chunks = process_documents(sample_docs, strategy="structure")
    engine.index_chunks(chunks)

    hits = engine.search("Which release caused 504 errors in payment-gateway-service?", top_k=3, mode="hybrid")
    assert len(hits) >= 1
    assert hits[0]["doc_id"] == "inc_100"

    agent = InvestigationAgent(engine)
    agent.api_key = ""

    ans = agent.answer_question("Which release caused 504 errors in payment-gateway-service?")
    assert ans["abstained"] is False
    assert "inc_100" in ans["citations"]

    unanswerable = agent.answer_question("What was the Q4 2026 cryptocurrency wallet revenue in Tokyo?")
    assert unanswerable["abstained"] is True