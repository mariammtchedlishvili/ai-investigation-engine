import os
import tempfile
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from app.ingestion import extract_text_from_file, process_documents
from app.search import HybridSearchEngine
from app.rag_agent import InvestigationAgent

app = FastAPI(title="AI Investigation Engine", version="1.0.0")

search_engine = HybridSearchEngine()
agent = InvestigationAgent(search_engine)


INITIAL_DOCS = [
    {
        "doc_id": "changelog_2025_03",
        "title": "System Changelog - March 2025",
        "source": "changelog_march.md",
        "date": "2025-03-14",
        "file_type": "md",
        "text": (
            "On 2025-03-14 at 09:15 UTC, release v4.8.2 was deployed to production. "
            "This change updated the PostgreSQL connection pooler configuration in payment-gateway-service, "
            "reducing max_idle_connections from 120 to 15 and enabling strict TLS timeout of 800ms. "
            "Deployment author: DevOps Team."
        ),
    },
    {
        "doc_id": "incident_inc_309",
        "title": "Incident Report INC-309: March Payment Processing Outage",
        "source": "inc_309_postmortem.md",
        "date": "2025-03-14",
        "file_type": "md",
        "text": (
            "Incident INC-309 occurred in March (2025-03-14 09:22 UTC to 11:05 UTC). "
            "Payment processing experienced severe delays and HTTP 504 timeouts. "
            "Root cause investigation confirmed that deployment v4.8.2 in payment-gateway-service exhausted database "
            "connections under peak load. The damaged downstream services were checkout-api, billing-worker, "
            "and merchant-settlement-service. Rolling back v4.8.2 resolved the outage."
        ),
    },
    {
        "doc_id": "arch_spec_payments",
        "title": "Payment Architecture & Dependency Specification",
        "source": "architecture_spec.pdf",
        "date": "2025-01-20",
        "file_type": "pdf",
        "text": (
            "The payment-gateway-service acts as the core routing layer. "
            "Any latency or connection exhaustion in payment-gateway-service directly impacts synchronous callers "
            "checkout-api and asynchronous consumers billing-worker and merchant-settlement-service. "
            "Maximum retry window is 30 seconds before transaction failure."
        ),
    },
    {
        "doc_id": "audit_note_march_A",
        "title": "Internal SRE Audit Report - Redis Cache Cluster",
        "source": "sre_audit_a.json",
        "date": "2025-03-18",
        "file_type": "json",
        "text": (
            "Audit of the March 14 incident shows that the Redis cache cluster remained healthy with 99.98% uptime "
            "and 0% packet loss. Redis was ruled out and was not caused by cache eviction; the outage was caused by "
            "v4.8.2 database connection pool exhaustion."
        ),
    },
    {
        "doc_id": "vendor_report_march_B",
        "title": "External Network Vendor Preliminary Memo",
        "source": "vendor_memo_b.txt",
        "date": "2025-03-15",
        "file_type": "txt",
        "text": (
            "Preliminary external memo claims a contradiction: it states the March 14 outage was caused by "
            "a Redis cache cluster failure with 42% packet loss and that database connection pools remained healthy "
            "and unresolved until 14:00 UTC."
        ),
    },
]


def seed_initial_documents():
    if not search_engine.list_documents():
        chunks = process_documents(INITIAL_DOCS, strategy="structure")
        search_engine.index_chunks(chunks)


seed_initial_documents()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    search_mode: str = "hybrid"


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "indexed_chunks": len(search_engine.chunks_cache),
        "documents_count": len(search_engine.list_documents()),
    }


@app.get("/api/documents")
def get_documents():
    return {"documents": search_engine.list_documents()}


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    strategy: str = Form("structure"),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in [".pdf", ".md", ".txt", ".json"]:
        raise HTTPException(
            status_code=400,
            detail="Supported formats are .pdf, .md, .txt, and .json",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        raw_docs = extract_text_from_file(tmp_path)
        for d in raw_docs:
            d["source"] = file.filename
            if len(raw_docs) == 1:
                d["doc_id"] = Path(file.filename).stem
                d["title"] = Path(file.filename).stem
        chunks = process_documents(raw_docs, strategy=strategy)
        indexed = search_engine.index_chunks(chunks)
        return {
            "status": "indexed",
            "filename": file.filename,
            "strategy": strategy,
            "chunks_added": indexed,
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.delete("/api/documents/{doc_id}")
def remove_document(doc_id: str):
    removed = search_engine.delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id, "chunks_removed": removed}


@app.post("/api/query")
def query_engine(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    return agent.answer_question(
        question=req.question,
        top_k=req.top_k,
        search_mode=req.search_mode,
    )


@app.get("/", response_class=HTMLResponse)
def web_ui():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>AI Investigation Engine</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; margin: 0; background: #f6f8fa; color: #1f2328; }
    header { background: #0f172a; color: #fff; padding: 16px 28px; display: flex; justify-content: space-between; align-items: center; }
    .container { max-width: 1150px; margin: 24px auto; padding: 0 16px; display: grid; grid-template-columns: 340px 1fr; gap: 20px; }
    .card { background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
    h2, h3 { margin-top: 0; }
    input, select, textarea, button { width: 100%; padding: 9px; margin-top: 6px; margin-bottom: 12px; border-radius: 6px; border: 1px solid #d0d7de; box-sizing: border-box; font-size: 14px; }
    button { background: #2563eb; color: #fff; border: none; font-weight: 600; cursor: pointer; }
    button:hover { background: #1d4ed8; }
    .btn-del { background: #dc2626; width: auto; padding: 4px 10px; font-size: 12px; margin: 0; }
    .doc-item { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #eee; font-size: 13px; }
    .badge { background: #e2e8f0; color: #334155; padding: 2px 7px; border-radius: 4px; font-size: 12px; margin-right: 6px; }
    .conflict { background: #fef2f2; border-left: 4px solid #dc2626; padding: 10px; margin: 10px 0; font-size: 13px; }
    .source-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px; margin-top: 8px; font-size: 13px; }
    .sample-btn { background: #f1f5f9; color: #0f172a; border: 1px solid #cbd5e1; text-align: left; font-weight: normal; margin-bottom: 6px; }
    .sample-btn:hover { background: #e2e8f0; }
  </style>
</head>
<body>
  <header>
    <div><strong>AI Investigation Engine</strong> — Multi-Document Hybrid RAG</div>
    <div id="stats" style="font-size: 13px;">Loading...</div>
  </header>
  <div class="container">
    <div>
      <div class="card">
        <h3>Upload Document</h3>
        <input type="file" id="fileInput" accept=".pdf,.md,.txt,.json" />
        <label style="font-size:13px;">Chunking Strategy:</label>
        <select id="chunkStrategy">
          <option value="structure">Structure / Paragraph-Aware</option>
          <option value="fixed">Fixed Window (180 words, 40 overlap)</option>
        </select>
        <button onclick="uploadFile()">Upload & Index</button>
        <div id="uploadStatus" style="font-size:13px; color:#15803d;"></div>
      </div>
      <div class="card">
        <h3>Indexed Documents</h3>
        <div id="docList"></div>
      </div>
    </div>
    <div>
      <div class="card">
        <h3>Investigation Query</h3>
        <div style="margin-bottom:8px; font-size:13px; color:#475569;">Try sample multi-hop / contradiction / unanswerable queries:</div>
        <button class="sample-btn" onclick="setQuery('What system change in March caused the payment processing outage, which downstream services were damaged, and which documents confirm this?')">
          1. Multi-hop: March payment outage root cause & damaged services
        </button>
        <button class="sample-btn" onclick="setQuery('Was the March 14 outage caused by a Redis cache cluster failure or database connection pools? Are there contradictions between reports?')">
          2. Contradiction Detection: Redis cache vs Database pool reports
        </button>
        <button class="sample-btn" onclick="setQuery('What was the Q4 2026 cryptocurrency hardware wallet revenue in Tokyo?')">
          3. Unanswerable (Abstention Test): Out-of-scope question
        </button>
        <textarea id="questionInput" rows="3" placeholder="Enter a complex investigation question..."></textarea>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 12px;">
          <div>
            <label style="font-size:13px;">Search Mode:</label>
            <select id="searchMode">
              <option value="hybrid">Hybrid (Semantic + BM25 + Custom Ranker)</option>
              <option value="semantic">Semantic Only (Qdrant Vectors)</option>
              <option value="keyword">Keyword Only (BM25)</option>
            </select>
          </div>
          <div>
            <label style="font-size:13px;">Top-K Evidence Chunks:</label>
            <input type="number" id="topK" value="5" min="1" max="15" />
          </div>
        </div>
        <button onclick="runQuery()">Run Investigation</button>
      </div>
      <div class="card" id="resultCard" style="display:none;">
        <h3>Investigation Findings</h3>
        <div id="metaInfo" style="margin-bottom:10px;"></div>
        <div id="conflictArea"></div>
        <div id="answerArea" style="white-space: pre-wrap; line-height: 1.5; background:#f8fafc; padding:12px; border-radius:6px; border:1px solid #e2e8f0;"></div>
        <h4 style="margin-top:18px; margin-bottom:8px;">Retrieved Evidence & Citations</h4>
        <div id="evidenceArea"></div>
      </div>
    </div>
  </div>
  <script>
    function setQuery(q) { document.getElementById('questionInput').value = q; }
    async function loadDocs() {
      const res = await fetch('/api/documents');
      const data = await res.json();
      const health = await (await fetch('/api/health')).json();
      document.getElementById('stats').innerText = `Docs: ${health.documents_count} | Chunks: ${health.indexed_chunks}`;
      const list = document.getElementById('docList');
      list.innerHTML = data.documents.map(d => `
        <div class="doc-item">
          <div>
            <strong>${d.doc_id}</strong> <span class="badge">${d.file_type}</span><br/>
            <span style="color:#64748b;">${d.chunk_count} chunks | ${d.date}</span>
          </div>
          <button class="btn-del" onclick="deleteDoc('${d.doc_id}')">Delete</button>
        </div>
      `).join('');
    }
    async function uploadFile() {
      const input = document.getElementById('fileInput');
      if (!input.files.length) return alert('Select a file first');
      const formData = new FormData();
      formData.append('file', input.files[0]);
      formData.append('strategy', document.getElementById('chunkStrategy').value);
      document.getElementById('uploadStatus').innerText = 'Indexing...';
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await res.json();
      document.getElementById('uploadStatus').innerText = `Indexed ${data.chunks_added} chunks (${data.strategy})`;
      loadDocs();
    }
    async function deleteDoc(docId) {
      await fetch(`/api/documents/${docId}`, { method: 'DELETE' });
      loadDocs();
    }
    async function runQuery() {
      const question = document.getElementById('questionInput').value;
      if (!question.trim()) return;
      const top_k = parseInt(document.getElementById('topK').value || '5');
      const search_mode = document.getElementById('searchMode').value;
      document.getElementById('resultCard').style.display = 'block';
      document.getElementById('answerArea').innerText = 'Investigating across documents...';
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, top_k, search_mode })
      });
      const data = await res.json();
      document.getElementById('metaInfo').innerHTML = `
        <span class="badge">Latency: ${data.latency_sec}s</span>
        <span class="badge">Sub-queries: ${data.sub_queries.length}</span>
        <span class="badge">Tool Calls: ${data.tool_calls.length}</span>
        <span class="badge">Abstained: ${data.abstained}</span>
      `;
      document.getElementById('conflictArea').innerHTML = (data.contradictions || []).map(c =>
        `<div class="conflict"><strong>Contradiction Detected:</strong> ${c.summary} (${c.source_a} vs ${c.source_b})</div>`
      ).join('');
      document.getElementById('answerArea').innerText = data.answer;
      document.getElementById('evidenceArea').innerHTML = (data.evidence || []).map(e => `
        <div class="source-box">
          <strong>[Source: ${e.doc_id}]</strong> (${e.title}) —
          <span class="badge">Final Score: ${e.final_score}</span>
          <span class="badge">Sem: ${(e.semantic_score || 0).toFixed(3)}</span>
          <span class="badge">BM25: ${(e.bm25_score || 0).toFixed(2)}</span>
          <div style="margin-top:6px;">${e.text}</div>
        </div>
      `).join('');
    }
    loadDocs();
  </script>
</body>
</html>
"""