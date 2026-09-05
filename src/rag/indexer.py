"""
Module B: Qualitative Legal & Policy Vector Store Indexer (Qdrant + BM25)

Processes all preprocessed qualitative trade policy notifications, DGFT export/import circulars,
scanned OCR legal gazettes, and PIB press releases into:
  1. Dense Semantic Vector Index (`Qdrant` collection using BAAI/bge-small-en-v1.5 embeddings)
  2. Sparse Keyword Index (`rank_bm25.BM25Okapi` serialized to pickle for fast exact HS code/circular lookups)
"""

import os
import re
import json
import joblib
import argparse
import lzma
import pickle
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from rank_bm25 import BM25Okapi

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
FALLBACK_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "trade_policy_compliance"

# bge-small truncates input at 512 tokens (~2000 chars). Whole gazette
# notifications blow past that, so the dense vector only ever represented the
# opening boilerplate. Paragraph-aware chunking keeps every embedded unit
# inside the model's real context window.
MAX_CHUNK_CHARS = 1600
CHUNK_OVERLAP_CHARS = 200
SINGLE_CHUNK_THRESHOLD = int(MAX_CHUNK_CHARS * 1.2)


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    """Split text into overlapping chunks on paragraph boundaries.

    Greedily merges paragraphs up to max_chars; paragraphs longer than the
    limit are hard-split with character overlap so no content is dropped.
    """
    text = text.strip()
    if len(text) <= SINGLE_CHUNK_THRESHOLD:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n{1,}", text) if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush():
        nonlocal current, current_len
        if current:
            chunks.append("\n".join(current))
            current, current_len = [], 0

    for para in paragraphs:
        if len(para) > max_chars:
            flush()
            step = max(1, max_chars - overlap)
            for start in range(0, len(para), step):
                piece = para[start:start + max_chars]
                if len(piece) < overlap // 2 and chunks:
                    break
                chunks.append(piece)
            continue
        if current_len + len(para) + 1 > max_chars:
            flush()
        current.append(para)
        current_len += len(para) + 1
    flush()

    return chunks if chunks else [text[:max_chars]]

def load_all_policy_chunks(data_dir: str) -> List[Dict[str, Any]]:
    """
    Loads and normalizes qualitative chunks from processed JSONL files:
      - dgft_policy_chunks.jsonl
      - dgft_ocr_chunks.jsonl
      - pib_press_releases.jsonl
    """
    files_to_load = [
        ("dgft_policy_chunks.jsonl", "DGFT_Policy"),
        ("dgft_ocr_chunks.jsonl", "DGFT_OCR"),
        ("pib_press_releases.jsonl", "PIB_Press_Release")
    ]

    normalized_docs = []
    print(f"[*] Scanning qualitative policy directory -> {data_dir}")

    for filename, doc_type in files_to_load:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"  [Warning] File not found: {filepath}. Skipping.")
            continue

        print(f"  [*] Ingesting {doc_type} records from {filename}...")
        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                text = obj.get("clean_text") or obj.get("text") or ""
                if not text or len(text.strip()) < 15:
                    continue

                doc_id = str(obj.get("chunk_id") or obj.get("prid") or f"{doc_type}_{count}")

                title = obj.get("subject") or obj.get("title") or f"{doc_type} Notification {obj.get('notification_no', '')}"

                payload = {
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "title": str(title)[:300],
                    "notification_no": str(obj.get("notification_no", "")),
                    "date": str(obj.get("date", "")),
                    "ministry": str(obj.get("ministry", "DGFT / Ministry of Commerce")),
                    "pdf_path": str(obj.get("pdf_path", obj.get("url", ""))),
                    "text": text.strip(),
                    "char_length": len(text.strip())
                }

                normalized_docs.append(payload)
                count += 1
        print(f"    -> Extracted {count} valid policy chunks from {filename}")

    print(f"\n[OK] Total normalized qualitative policy documents ready for indexing: {len(normalized_docs)}")
    return normalized_docs

def explode_docs_into_chunks(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand whole-document payloads into paragraph-level chunk payloads.

    Parent metadata (doc_type, title, notification_no, date, ministry,
    pdf_path) is inherited by every chunk so citations survive chunking.
    """
    chunk_docs: List[Dict[str, Any]] = []
    for doc in docs:
        pieces = chunk_text(doc["text"])
        for idx, piece in enumerate(pieces):
            chunk_docs.append({
                **doc,
                "doc_id": f"{doc['doc_id']}::c{idx}",
                "parent_doc_id": doc["doc_id"],
                "chunk_index": idx,
                "num_chunks": len(pieces),
                "text": piece,
                "char_length": len(piece),
            })
    multi = sum(1 for d in docs if d["char_length"] > SINGLE_CHUNK_THRESHOLD)
    print(f"[*] Chunking: {len(docs)} documents ({multi} long) -> {len(chunk_docs)} chunks "
          f"(max {MAX_CHUNK_CHARS} chars, overlap {CHUNK_OVERLAP_CHARS})")
    return chunk_docs


def build_dense_and_sparse_indexes(
    docs: List[Dict[str, Any]],
    qdrant_path: str,
    bm25_output_path: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64
):
    """
    Encodes documents into dense vectors (Qdrant) and sparse inverted token counts (BM25).
    """
    if not docs:
        raise ValueError("Document list is empty! Cannot build vector/keyword indexes.")

    docs = explode_docs_into_chunks(docs)

    print(f"\n[*] Initializing dense embedding model: {model_name}...")
    try:
        embedder = SentenceTransformer(model_name)
    except Exception as e:
        print(f"  [Warning] Failed to load {model_name} ({e}). Falling back to {FALLBACK_EMBEDDING_MODEL}...")
        embedder = SentenceTransformer(FALLBACK_EMBEDDING_MODEL)

    embedding_dim = embedder.get_sentence_embedding_dimension()
    print(f"[OK] Embedding model loaded | Vector Dimension: {embedding_dim}")

    os.makedirs(qdrant_path, exist_ok=True)
    print(f"\n[*] Initializing persistent Qdrant vector database at -> {qdrant_path}")
    client = QdrantClient(path=qdrant_path)

    collections = client.get_collections().collections
    if any(c.name == COLLECTION_NAME for c in collections):
        print(f"  [*] Recreating Qdrant collection: '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
    )

    print(f"[*] Vectorizing {len(docs)} qualitative chunks in batches of {batch_size}...")
    texts = [d["text"] for d in docs]

    points = []
    for i in tqdm(range(0, len(docs), batch_size), desc="Encoding & Upserting to Qdrant"):
        batch_docs = docs[i:i + batch_size]
        batch_texts = texts[i:i + batch_size]

        embeddings = embedder.encode(batch_texts, show_progress_bar=False, normalize_embeddings=True)

        for j, doc in enumerate(batch_docs):
            point_id = i + j
            points.append(PointStruct(
                id=point_id,
                vector=embeddings[j].tolist(),
                payload=doc
            ))

    print("[*] Writing dense vector payloads to Qdrant storage...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"[SUCCESS] Qdrant Dense Vector Index Built! Total Indexed Points: {collection_info.points_count}")

    print("\n[*] Constructing Sparse BM25 Keyword Index for exact HS Code & Circular lookup...")
    tokenized_corpus = []
    for text in tqdm(texts, desc="Tokenizing corpus for BM25"):

        tokens = [word.lower().strip(",.()[]{}\"'") for word in text.split() if len(word) > 1]
        tokenized_corpus.append(tokens)

    bm25 = BM25Okapi(tokenized_corpus)

    os.makedirs(os.path.dirname(bm25_output_path), exist_ok=True)
    joblib.dump({
        "bm25": bm25,
        "docs": docs,
        "tokenized_corpus": tokenized_corpus,
        "indexed_at": datetime.now().isoformat()
    }, bm25_output_path)
    print(f"[SUCCESS] Sparse BM25 Index & Corpus serialized to -> {bm25_output_path}")

    # lzma-compressed twin for production: the whole pickle is ~21MB of
    # repetitive legal text (2.6MB compressed) — small enough to track in git
    # and COPY into the Render Docker image, so the API always has a real
    # sparse retriever even without a live Qdrant cluster.
    lzma_path = bm25_output_path + ".lzma"
    with open(bm25_output_path, "rb") as f_in:
        raw = f_in.read()
    with lzma.open(lzma_path, "wb", preset=6) as f_out:
        f_out.write(raw)
    print(f"[Exported] Compressed prod BM25 artifact -> {lzma_path} "
          f"({len(raw) // 1024}KB -> {os.path.getsize(lzma_path) // 1024}KB)")

    meta_path = os.path.join(os.path.dirname(bm25_output_path), "rag_index_meta.json")
    meta_data = {
        "index_timestamp": datetime.now().isoformat(),
        "total_documents": len(docs),
        "chunking": {
            "strategy": "paragraph-aware, overlapping",
            "max_chunk_chars": MAX_CHUNK_CHARS,
            "overlap_chars": CHUNK_OVERLAP_CHARS,
            "parents_split_into_chunks": len({d["parent_doc_id"] for d in docs if d.get("num_chunks", 1) > 1}),
        },
        "doc_types_breakdown": {
            "DGFT_Policy": sum(1 for d in docs if d["doc_type"] == "DGFT_Policy"),
            "DGFT_OCR": sum(1 for d in docs if d["doc_type"] == "DGFT_OCR"),
            "PIB_Press_Release": sum(1 for d in docs if d["doc_type"] == "PIB_Press_Release")
        },
        "dense_index": {
            "engine": "Qdrant Persistent Disk Storage",
            "path": qdrant_path,
            "collection_name": COLLECTION_NAME,
            "embedding_model": model_name,
            "vector_dimension": embedding_dim,
            "points_count": collection_info.points_count
        },
        "sparse_index": {
            "engine": "BM25Okapi",
            "path": bm25_output_path
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)
    print(f"[Exported] RAG Index metadata -> {meta_path}")
    print("\n=== PHASE 4 (QUALITATIVE RAG INDEXING) PIPELINE COMPLETE ===")

def main():
    parser = argparse.ArgumentParser(description="Build Qualitative Legal & Policy Vector/BM25 Index (Module B)")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Directory containing qualitative JSONL files")
    parser.add_argument("--qdrant-path", type=str, default="data/cache/qdrant_index", help="Persistent Qdrant database folder")
    parser.add_argument("--bm25-path", type=str, default="data/cache/bm25_index.pkl", help="Serialized BM25 index file path")
    parser.add_argument("--model", type=str, default=DEFAULT_EMBEDDING_MODEL, help="HuggingFace SentenceTransformer model")
    args = parser.parse_args()

    docs = load_all_policy_chunks(args.data_dir)
    build_dense_and_sparse_indexes(
        docs=docs,
        qdrant_path=args.qdrant_path,
        bm25_output_path=args.bm25_path,
        model_name=args.model
    )

if __name__ == "__main__":
    main()

