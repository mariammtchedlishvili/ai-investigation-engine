import os
import re
import uuid
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from fastembed import TextEmbedding


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+|[\u10A0-\u10FF]+", text.lower())


class HybridSearchEngine:
    def __init__(
        self,
        collection_name: str = "investigation_docs",
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.embedder = TextEmbedding(model_name=embedding_model_name)
        self.vector_size = 384

        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        qdrant_key = os.getenv("QDRANT_API_KEY", "").strip()
        if not hasattr(HybridSearchEngine, '_shared_client'):
            if qdrant_url:
                HybridSearchEngine._shared_client = QdrantClient(url=qdrant_url, api_key=qdrant_key or None)
            else:
                os.makedirs("data/qdrant_storage", exist_ok=True)
                HybridSearchEngine._shared_client = QdrantClient(path="data/qdrant_storage")
        self.client = HybridSearchEngine._shared_client

        self._ensure_collection()
        self.chunks_cache: list[dict] = []
        self.bm25: BM25Okapi | None = None
        self._sync_bm25_from_db()

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def _chunk_id_to_uuid(self, chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    def _sync_bm25_from_db(self):
        all_points = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for r in records:
                if r.payload:
                    all_points.append(r.payload)
            if offset is None:
                break

        self.chunks_cache = all_points
        if self.chunks_cache:
            tokenized_corpus = [tokenize(c.get("text", "")) for c in self.chunks_cache]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    def index_chunks(self, chunks: list[dict], batch_size: int = 128) -> int:
        if not chunks:
            return 0
        texts = [c["text"] for c in chunks]
        embeddings = list(self.embedder.embed(texts, batch_size=batch_size))

        points = []
        for chunk, vec in zip(chunks, embeddings):
            point_id = self._chunk_id_to_uuid(chunk["chunk_id"])
            points.append(
                PointStruct(id=point_id, vector=vec.tolist(), payload=chunk)
            )

        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + batch_size],
            )

        self._sync_bm25_from_db()
        return len(points)

    def delete_document(self, doc_id: str) -> int:
        before_count = len(self.chunks_cache)
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )
        self._sync_bm25_from_db()
        return max(0, before_count - len(self.chunks_cache))

    def list_documents(self) -> list[dict]:
        docs = {}
        for ch in self.chunks_cache:
            did = ch.get("doc_id", "unknown")
            if did not in docs:
                docs[did] = {
                    "doc_id": did,
                    "title": ch.get("title", did),
                    "source": ch.get("source", "unknown"),
                    "file_type": ch.get("file_type", "text"),
                    "date": ch.get("date", "unknown"),
                    "chunk_count": 0,
                }
            docs[did]["chunk_count"] += 1
        return list(docs.values())

    def semantic_search(self, query: str, top_k: int = 15) -> list[dict]:
        query_vec = list(self.embedder.embed([query]))[0].tolist()
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=top_k,
            with_payload=True,
        ).points

        results = []
        for rank, hit in enumerate(res, start=1):
            item = dict(hit.payload)
            item["semantic_score"] = float(hit.score)
            item["semantic_rank"] = rank
            results.append(item)
        return results

    def keyword_search(self, query: str, top_k: int = 15) -> list[dict]:
        if not self.bm25 or not self.chunks_cache:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)), key=lambda idx: scores[idx], reverse=True
        )[:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            if scores[idx] <= 0:
                continue
            item = dict(self.chunks_cache[idx])
            item["bm25_score"] = float(scores[idx])
            item["bm25_rank"] = rank
            results.append(item)
        return results

    def custom_hybrid_ranker(
        self,
        query: str,
        sem_results: list[dict],
        kw_results: list[dict],
        top_k: int = 6,
        alpha: float = 0.6,
        rrf_k: int = 60,
    ) -> list[dict]:
        merged: dict[str, dict] = {}
        max_bm25 = max((r.get("bm25_score", 0.0) for r in kw_results), default=1.0) or 1.0

        for item in sem_results:
            cid = item["chunk_id"]
            merged[cid] = dict(item)
            merged[cid].setdefault("bm25_score", 0.0)
            merged[cid].setdefault("bm25_rank", 999)

        for item in kw_results:
            cid = item["chunk_id"]
            if cid not in merged:
                merged[cid] = dict(item)
                merged[cid]["semantic_score"] = 0.0
                merged[cid]["semantic_rank"] = 999
            else:
                merged[cid]["bm25_score"] = item["bm25_score"]
                merged[cid]["bm25_rank"] = item["bm25_rank"]

        query_tokens = set(tokenize(query))
        numeric_tokens = {t for t in query_tokens if any(ch.isdigit() for ch in t)}

        final_list = []
        for cid, doc in merged.items():
            sem_rank = doc.get("semantic_rank", 999)
            bm25_rank = doc.get("bm25_rank", 999)

            rrf_sem = (1.0 / (rrf_k + sem_rank)) if sem_rank < 999 else 0.0
            rrf_kw = (1.0 / (rrf_k + bm25_rank)) if bm25_rank < 999 else 0.0
            rrf_score = alpha * rrf_sem + (1.0 - alpha) * rrf_kw

            norm_sem = max(0.0, min(1.0, doc.get("semantic_score", 0.0)))
            norm_bm25 = doc.get("bm25_score", 0.0) / max_bm25
            blended_score = alpha * norm_sem + (1.0 - alpha) * norm_bm25

            doc_tokens = set(tokenize(doc.get("text", "")))
            title_tokens = set(tokenize(doc.get("title", "")))

            coverage = (
                len(query_tokens & doc_tokens) / len(query_tokens)
                if query_tokens
                else 0.0
            )
            numeric_bonus = (
                0.12
                if numeric_tokens and numeric_tokens.issubset(doc_tokens | title_tokens)
                else 0.0
            )
            title_bonus = 0.10 if (query_tokens & title_tokens) else 0.0

            final_score = (
                0.50 * blended_score
                + 0.30 * (rrf_score * rrf_k)
                + 0.20 * coverage
                + numeric_bonus
                + title_bonus
            )

            doc["final_score"] = round(final_score, 4)
            doc["coverage_score"] = round(coverage, 3)
            final_list.append(doc)

        final_list.sort(key=lambda x: x["final_score"], reverse=True)
        return final_list[:top_k]

    def search(self, query: str, top_k: int = 6, mode: str = "hybrid") -> list[dict]:
        if mode == "semantic":
            res = self.semantic_search(query, top_k=top_k)
            for r in res:
                r["final_score"] = round(r.get("semantic_score", 0.0), 4)
            return res
        if mode == "keyword":
            res = self.keyword_search(query, top_k=top_k)
            for r in res:
                r["final_score"] = round(r.get("bm25_score", 0.0), 4)
            return res
        sem_res = self.semantic_search(query, top_k=top_k * 3)
        kw_res = self.keyword_search(query, top_k=top_k * 3)
        return self.custom_hybrid_ranker(query, sem_res, kw_res, top_k=top_k)