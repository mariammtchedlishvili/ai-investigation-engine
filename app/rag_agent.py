import os
import re
import time
from google import genai
from google.genai import types
from app.search import HybridSearchEngine, tokenize


def decompose_query(query: str) -> list[str]:
    """რთულ კითხვას ყოფს ქვე-კითხვებად (Multi-step processing)."""
    parts = re.split(
        r"\b(?:and|or|also|versus|vs|და|ასევე|ხოლო)\b|[?;]",
        query,
        flags=re.IGNORECASE,
    )
    sub_queries = [p.strip() for p in parts if len(p.strip().split()) >= 3]
    if not sub_queries or len(sub_queries) == 1:
        return [query.strip()]
    return [query.strip()] + sub_queries[:3]


def detect_contradictions(chunks: list[dict]) -> list[dict]:
    """ამოწმებს მოძიებულ დოკუმენტებს შორის ურთიერთგამომრიცხავ ფაქტებს."""
    contradictions = []
    if len(chunks) < 2:
        return contradictions

    conflict_pairs = [
        ({"resolved", "fixed", "healthy", "succeeded"}, {"unresolved", "failed", "outage", "broken"}),
        ({"enabled", "active", "true"}, {"disabled", "inactive", "false"}),
        ({"caused by", "root cause"}, {"not caused", "unrelated", "ruled out"}),
    ]

    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            c1, c2 = chunks[i], chunks[j]
            if c1.get("doc_id") == c2.get("doc_id"):
                continue

            t1 = set(tokenize(c1.get("text", "")))
            t2 = set(tokenize(c2.get("text", "")))
            shared_entities = {w for w in (t1 & t2) if len(w) > 4}

            if len(shared_entities) < 2:
                continue

            txt1_low = c1.get("text", "").lower()
            txt2_low = c2.get("text", "").lower()

            for pos_set, neg_set in conflict_pairs:
                has_pos_1 = any(p in txt1_low for p in pos_set)
                has_neg_2 = any(n in txt2_low for n in neg_set)
                has_neg_1 = any(n in txt1_low for n in neg_set)
                has_pos_2 = any(p in txt2_low for p in pos_set)

                if (has_pos_1 and has_neg_2) or (has_neg_1 and has_pos_2):
                    contradictions.append({
                        "source_a": c1.get("doc_id"),
                        "source_b": c2.get("doc_id"),
                        "shared_topic": list(shared_entities)[:5],
                        "summary": f"Conflicting statements detected between {c1.get('doc_id')} and {c2.get('doc_id')}.",
                    })
                    break

            nums1 = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", txt1_low))
            nums2 = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", txt2_low))
            if (
                "contradict" in txt1_low
                or "conflict" in txt1_low
                or "discrepancy" in txt2_low
                or "contradict" in txt2_low
            ):
                contradictions.append({
                    "source_a": c1.get("doc_id"),
                    "source_b": c2.get("doc_id"),
                    "shared_topic": list(shared_entities)[:5],
                    "summary": f"Explicit discrepancy noted between {c1.get('doc_id')} and {c2.get('doc_id')}.",
                })
            elif nums1 and nums2 and nums1 != nums2 and len(shared_entities) >= 4:
                contradictions.append({
                    "source_a": c1.get("doc_id"),
                    "source_b": c2.get("doc_id"),
                    "shared_topic": list(shared_entities)[:5],
                    "summary": (
                        f"Different numeric values ({', '.join(list(nums1)[:2])} vs "
                        f"{', '.join(list(nums2)[:2])}) reported across sources."
                    ),
                })

    return contradictions[:3]


class InvestigationAgent:
    def __init__(self, search_engine: HybridSearchEngine):
        self.search_engine = search_engine
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def execute_tool_call(self, tool_name: str, args: dict) -> dict:
        """სტრუქტურირებული Tool Calling მექანიზმი ძიებისა და წინააღმდეგობების შესამოწმებლად."""
        if tool_name == "search_knowledge_base":
            q = args.get("query", "")
            mode = args.get("mode", "hybrid")
            top_k = int(args.get("top_k", 5))
            hits = self.search_engine.search(q, top_k=top_k, mode=mode)
            return {
                "tool": tool_name,
                "query": q,
                "results_count": len(hits),
                "results": hits,
            }

        if tool_name == "verify_contradictions":
            chunks = args.get("chunks", [])
            conflicts = detect_contradictions(chunks)
            return {"tool": tool_name, "conflicts": conflicts}

        return {"tool": tool_name, "error": "Unknown tool"}

    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        search_mode: str = "hybrid",
        min_score_threshold: float = 0.22,
    ) -> dict:
        start_time = time.time()
        sub_queries = decompose_query(question)
        tool_calls_log = []
        collected_chunks: dict[str, dict] = {}

        for sq in sub_queries:
            tool_out = self.execute_tool_call(
                "search_knowledge_base",
                {"query": sq, "mode": search_mode, "top_k": top_k},
            )
            tool_calls_log.append({
                "tool": "search_knowledge_base",
                "input": sq,
                "found": tool_out["results_count"],
            })
            for ch in tool_out["results"]:
                cid = ch["chunk_id"]
                if (
                    cid not in collected_chunks
                    or ch["final_score"] > collected_chunks[cid]["final_score"]
                ):
                    collected_chunks[cid] = ch

        evidence = sorted(
            collected_chunks.values(),
            key=lambda x: x.get("final_score", 0.0),
            reverse=True,
        )[:top_k]

        stopwords = {"what", "was", "the", "in", "of", "to", "and", "or", "is", "are", "for", "on", "at", "by", "with", "from", "how", "many", "which", "did", "does", "were", "who", "when", "where", "why"}
        content_q = {w for w in tokenize(question) if w not in stopwords and len(w) > 2}
        content_cov = max((len(content_q & set(tokenize(c.get("text", "") + " " + c.get("title", "")))) / max(1, len(content_q)) for c in evidence), default=0.0)
        best_score = evidence[0].get("final_score", 0.0) if evidence else 0.0

        if not evidence or best_score < min_score_threshold or content_cov < 0.25:
            return {
                "question": question,
                "sub_queries": sub_queries,
                "abstained": True,
                "contradictions": [],
                "tool_calls": tool_calls_log,
                "answer": (
                    "Insufficient Evidence: მოცემულ დოკუმენტებში არ მოიძებნა სანდო "
                    "მტკიცებულება ამ კითხვაზე პასუხის გასაცემად. სისტემა თავს იკავებს "
                    "დაუდასტურებელი ფაქტების გენერირებისგან."
                ),
                "citations": [],
                "evidence": evidence,
                "latency_sec": round(time.time() - start_time, 3),
            }

        conflict_out = self.execute_tool_call(
            "verify_contradictions", {"chunks": evidence}
        )
        contradictions = conflict_out["conflicts"]
        tool_calls_log.append({
            "tool": "verify_contradictions",
            "input": f"{len(evidence)} chunks",
            "found": len(contradictions),
        })

        citations = list(dict.fromkeys(c["doc_id"] for c in evidence))
        answer_text = self._generate_with_llm_or_fallback(
            question=question,
            evidence=evidence,
            contradictions=contradictions,
        )

        return {
            "question": question,
            "sub_queries": sub_queries,
            "abstained": False,
            "contradictions": contradictions,
            "tool_calls": tool_calls_log,
            "answer": answer_text,
            "citations": citations,
            "evidence": evidence,
            "latency_sec": round(time.time() - start_time, 3),
        }

    def _generate_with_llm_or_fallback(
        self,
        question: str,
        evidence: list[dict],
        contradictions: list[dict],
    ) -> str:
        context_blocks = []
        for idx, ch in enumerate(evidence, start=1):
            context_blocks.append(
                f"[{idx}] Source ID: {ch['doc_id']} | Title: {ch.get('title', '')} | Date: {ch.get('date', 'unknown')}\n"
                f"Content: {ch['text']}"
            )
        context_str = "\n\n".join(context_blocks)

        conflict_note = ""
        if contradictions:
            conflict_note = "\nDetected Contradictions across sources:\n" + "\n".join(
                f"- {c['summary']} ({c['source_a']} vs {c['source_b']})"
                for c in contradictions
            )

        if self.api_key:
            prompt = (
                "You are an AI Investigation Engine. Answer the user's question strictly using ONLY the provided evidence.\n"
                "Rules:\n"
                "1. Synthesize facts across multiple documents when needed.\n"
                "2. Cite every claim inline using [Source: <doc_id>].\n"
                "3. If sources contain conflicting or contradictory facts, explicitly highlight both sides and cite both sources.\n"
                "4. If the provided evidence does not contain the answer, state clearly that there is insufficient information.\n\n"
                f"Question: {question}\n\n"
                f"Evidence:\n{context_str}\n"
                f"{conflict_note}"
            )
            for attempt in range(2):
                try:
                    client = genai.Client(api_key=self.api_key)
                    resp = client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0.1),
                    )
                    if resp and resp.text:
                        return resp.text.strip()
                except Exception:
                    if attempt == 0:
                        time.sleep(1.0)
                    else:
                        break

        synthesized_lines = []
        for ch in evidence[:3]:
            snippet = ch["text"].strip()
            if len(snippet) > 260:
                snippet = snippet[:260].rsplit(" ", 1)[0] + "..."
            synthesized_lines.append(f"- {snippet} [Source: {ch['doc_id']}]")

        if contradictions:
            c = contradictions[0]
            synthesized_lines.append(
                f"\nContradiction Notice: {c['summary']} "
                f"([Source: {c['source_a']}], [Source: {c['source_b']}])."
            )

        return "Evidence Synthesis:\n" + "\n".join(synthesized_lines)