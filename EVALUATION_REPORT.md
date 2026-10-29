# Evaluation Report — AI Investigation Engine

## 1. Evaluation Methodology
We built an automated evaluation suite (`scripts/evaluate.py`) comprising **42 test questions** across four distinct categories:
- **Simple Single-Document Questions (14)**: Direct technical lookups (versions, timeouts, error codes).
- **Multi-Hop Cross-Document Questions (12)**: Questions requiring connecting 2–3 documents (e.g., matching a changelog release to an incident post-mortem and architecture spec).
- **Contradiction Detection Questions (8)**: Queries targeting conflicting reports (e.g., SRE audit vs external vendor memo on Redis packet loss; Kafka disk saturation vs FinOps S3 claim).
- **Unanswerable / Out-of-Scope Questions (8)**: Queries with no supporting evidence in the corpus to verify abstention and hallucination prevention.

## 2. Quantitative Comparison of Two Configurations

| Metric | Config A: Fixed Chunking + Semantic Only | Config B: Structure Chunking + Hybrid (Semantic + BM25 + Custom Ranker) | Delta (Improvement) |
| :--- | :---: | :---: | :---: |
| **Recall@5** | 0.9706 | **1.0** | +0.0294 |
| **MRR (Mean Reciprocal Rank)** | 0.9162 | **0.9706** | +0.0544 |
| **Answer Correctness** | 0.8929 | **0.9048** | +0.0119 |
| **Citation Accuracy** | 0.9524 | **0.9762** | +0.0238 |
| **Contradiction Detection Rate** | 1.0 | **1.0** | +0.0 |
| **Abstention Accuracy (Unanswerable)** | 0.875 | **0.875** | +0.0 |

## 3. Error Analysis: Where the System Fails, Why, and How to Improve

### Where Config A (Semantic Only + Fixed Window) Fails:
1. **Exact Identifier & Version Misses**: Pure dense embeddings (`BAAI/bge-small-en-v1.5`) occasionally rank general incident descriptions above exact version/ticket identifiers like `CHG-884`, `v1.11.3`, or `800ms`.
2. **Multi-Hop Recall Drops**: When a question asks about both a date (`2025-03-14`) and downstream architectural impact, semantic-only search retrieves the primary incident report at Rank 1 but sometimes pushes the secondary changelog or architecture spec below Rank 3.

### Why Config B (Structure Chunking + Hybrid RRF + Custom Ranker) Performs Better:
- Combining **BM25 lexical scores** with **Semantic cosine similarity** via our custom **Weighted Reciprocal Rank Fusion (RRF)** and **Numeric/Title Boost** ensures exact ticket IDs (`INC-309`, `CHG-884`) and dates surface in the top results.
- Multi-step query decomposition splits compound questions into focused sub-queries, improving multi-document `Recall@5` and `Citation Accuracy`.

### Remaining Limitations & Future Improvements:
- **Subtle Semantic Contradictions**: Our rule + numeric conflict detector catches explicit status inversions (`healthy` vs `outage`) and numeric discrepancies (`0%` vs `42%`), but subtle logical contradictions phrased without contrasting terms require an NLI (Natural Language Inference) cross-encoder model.
- **Cross-Encoder Re-ranking**: Adding a lightweight cross-encoder (`bge-reranker-base`) on top of the Top-20 hybrid candidates would further improve `MRR` on 3-hop queries.
