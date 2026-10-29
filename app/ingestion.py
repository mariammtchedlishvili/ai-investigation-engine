import hashlib
import json
import re
from pathlib import Path
from pypdf import PdfReader


def compute_content_hash(text: str) -> str:
    """ტექსტის სუფთა ვერსიიდან ქმნის უნიკალურ MD5 კოდს დუბლიკატების გამოსავლენად."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def extract_text_from_file(file_path: str) -> list[dict]:
    """კითხულობს .pdf, .md/.txt და .json ფაილებს და აბრუნებს ტექსტს მეტამონაცემებით."""
    path = Path(file_path)
    ext = path.suffix.lower()
    records = []

    if ext == ".pdf":
        reader = PdfReader(str(path))
        pages_text = []
        for page_num, page in enumerate(reader.pages, start=1):
            txt = page.extract_text() or ""
            if txt.strip():
                pages_text.append(f"[Page {page_num}]\n{txt}")
        records.append({
            "doc_id": path.stem,
            "source": path.name,
            "title": path.stem,
            "date": "unknown",
            "file_type": "pdf",
            "text": "\n\n".join(pages_text),
        })

    elif ext in [".md", ".txt"]:
        content = path.read_text(encoding="utf-8", errors="ignore")
        records.append({
            "doc_id": path.stem,
            "source": path.name,
            "title": path.stem,
            "date": "unknown",
            "file_type": ext.lstrip("."),
            "text": content,
        })

    elif ext == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw if isinstance(raw, list) else [raw]
        for idx, item in enumerate(items):
            text = item.get("text") or item.get("content") or ""
            if not text.strip():
                continue
            records.append({
                "doc_id": str(item.get("id", f"{path.stem}_{idx}")),
                "source": item.get("source", path.name),
                "title": item.get("title", path.stem),
                "date": item.get("date", "unknown"),
                "file_type": "json",
                "text": text,
            })
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return records


def chunk_fixed_window(text: str, chunk_size: int = 180, overlap: int = 40) -> list[str]:
    """სტრატეგია 1: ჭრის ტექსტს ფიქსირებული სიტყვების რაოდენობით (overlap-ით)."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        piece = " ".join(words[i : i + chunk_size])
        if piece.strip():
            chunks.append(piece)
        if i + chunk_size >= len(words):
            break
    return chunks


def chunk_by_structure(text: str, max_words: int = 200) -> list[str]:
    """სტრატეგია 2: ჭრის ტექსტს აბზაცებისა და სექციების მიხედვით."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current_words = []

    for para in paragraphs:
        para_words = para.split()
        if len(para_words) > max_words:
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = []
            chunks.extend(chunk_fixed_window(para, chunk_size=max_words, overlap=30))
            continue

        if len(current_words) + len(para_words) <= max_words:
            current_words.extend(para_words)
        else:
            chunks.append(" ".join(current_words))
            current_words = para_words

    if current_words:
        chunks.append(" ".join(current_words))
    return chunks


def process_documents(
    raw_docs: list[dict],
    strategy: str = "structure",
    seen_hashes: set | None = None,
) -> list[dict]:
    """ფილტრავს დუბლიკატებს და აბრუნებს დაჭრილ chunk-ებს მეტამონაცემებით."""
    if seen_hashes is None:
        seen_hashes = set()

    processed_chunks = []
    for doc in raw_docs:
        text = doc.get("text", "").strip()
        if not text:
            continue

        doc_hash = compute_content_hash(text)
        if doc_hash in seen_hashes:
            continue
        # doc_hash checked separately

        raw_chunks = (
            chunk_fixed_window(text)
            if strategy == "fixed"
            else chunk_by_structure(text)
        )

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_hash = compute_content_hash(chunk_text)
            if chunk_hash in seen_hashes:
                continue
            seen_hashes.add(chunk_hash)

            processed_chunks.append({
                "chunk_id": f"{doc['doc_id']}_ch_{idx}",
                "doc_id": str(doc["doc_id"]),
                "source": doc.get("source", "unknown"),
                "title": doc.get("title", str(doc["doc_id"])),
                "date": doc.get("date", "unknown"),
                "file_type": doc.get("file_type", "text"),
                "chunk_index": idx,
                "strategy": strategy,
                "content_hash": chunk_hash,
                "text": chunk_text,
            })

    return processed_chunks