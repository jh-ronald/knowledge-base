#!/usr/bin/env python3
"""Local RAG knowledge base with optimized indexing and query pipeline."""

import os
import shutil
import argparse
import base64
import hashlib
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import log
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

load_dotenv()

# ─── Constants ────────────────────────────────────────────────────────────────

DATA_PATH = str(Path(__file__).resolve().parent / "docs")
CHROMA_PATH = "chroma_db"
HASH_MANIFEST_FILE = "doc_hashes.json"

DEFAULT_OCR_PROMPT = (
    "Extract all readable text from this PDF page, including chart titles, axis labels, "
    "legends, table headers/cells, and annotations. Preserve key structure with short "
    "headings and bullet points."
)
LOCAL_CONTEXT_MIN_CHARS_FOR_NO_WEB_FALLBACK = 1200

TERM_ALIASES = {
    """Insert domain-specific term aliases here"""
}

# ─── File Hashing (for incremental indexing) ──────────────────────────────────

def compute_file_hash(file_path):
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _manifest_path(persist_directory=CHROMA_PATH):
    """Store manifest as a sibling of the chroma_db directory."""
    db_path = Path(persist_directory).resolve()
    return str(db_path.parent / HASH_MANIFEST_FILE)


def load_hash_manifest(persist_directory=CHROMA_PATH):
    path = _manifest_path(persist_directory)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_hash_manifest(manifest, persist_directory=CHROMA_PATH):
    path = _manifest_path(persist_directory)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def generate_chunk_id(chunk, index=0):
    """Generate deterministic ID for a chunk to prevent duplicates."""
    source = chunk.metadata.get("source", "")
    page = str(chunk.metadata.get("page", "0"))
    content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:12]
    return f"{source}:p{page}:c{content_hash}:{index}"


# ─── Wikipedia Fallback (parallelized) ────────────────────────────────────────

def extract_candidate_terms(question, max_terms=3):
    """Extracts likely terminology terms for Wikipedia fallback lookup."""
    stop_terms = {
        "Tell", "What", "How", "When", "Where", "Why", "Which", "And", "Or", "The",
        "A", "An", "About", "Role", "Flow", "Relationship", "Between", "In",
    }
    terms = []
    terms.extend(re.findall(r"\b[A-Z]{2,}\b", question))
    terms.extend(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", question))

    cleaned = []
    for term in terms:
        value = term.strip()
        if not value or value in stop_terms:
            continue
        if value not in cleaned:
            cleaned.append(value)
        if len(cleaned) >= max_terms:
            break
    return cleaned


def fetch_wikipedia_summary(term, timeout_seconds=6, max_summary_chars=500):
    """Fetches a concise Wikipedia summary for a single term."""
    encoded_term = urllib.parse.quote(term, safe="")
    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_term}"
    request = urllib.request.Request(
        summary_url,
        headers={"User-Agent": "local-rag/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    summary = (payload.get("extract") or "").strip()
    if not summary:
        return None
    if len(summary) > max_summary_chars:
        summary = summary[:max_summary_chars].rstrip() + "..."

    page_url = payload.get("content_urls", {}).get("desktop", {}).get("page", "")
    title = payload.get("title") or term
    return {"title": title, "summary": summary, "url": page_url}


def _fetch_term_with_aliases(term, timeout_seconds, max_summary_chars):
    """Try a term and its aliases, return first successful result."""
    candidates = [term] + TERM_ALIASES.get(term.upper(), [])
    for candidate in candidates:
        result = fetch_wikipedia_summary(
            term=candidate,
            timeout_seconds=timeout_seconds,
            max_summary_chars=max_summary_chars,
        )
        if result:
            return result
    return None


def build_web_context(question, max_terms=3, max_summary_chars=500, timeout_seconds=6):
    """Builds Wikipedia context with parallel lookups."""
    terms = extract_candidate_terms(question, max_terms=max_terms)
    if not terms:
        return "", []

    # Fetch all terms in parallel
    entries = []
    with ThreadPoolExecutor(max_workers=min(len(terms), 4)) as pool:
        futures = {
            pool.submit(
                _fetch_term_with_aliases, term, timeout_seconds, max_summary_chars
            ): term
            for term in terms
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                entries.append(result)

    if not entries:
        return "", []

    lines = [
        f"- {e['title']}: {e['summary']} (Source: {e['url'] or 'Wikipedia'})"
        for e in entries
    ]
    return "\n".join(lines), entries


# ─── Lightweight Reranker (TF-IDF, no extra deps) ────────────────────────────

def rerank_documents(query, documents, top_k=None):
    """Rerank retrieved documents by TF-IDF cosine relevance to the query."""
    if not documents:
        return documents

    query_terms = query.lower().split()
    if not query_terms:
        return documents[:top_k] if top_k else documents

    num_docs = len(documents)
    scored = []

    for doc in documents:
        doc_text = doc.page_content.lower()
        doc_terms = doc_text.split()
        doc_len = max(len(doc_terms), 1)
        term_counts = Counter(doc_terms)

        score = 0.0
        for qt in query_terms:
            tf = term_counts.get(qt, 0) / doc_len
            docs_with_term = sum(1 for d in documents if qt in d.page_content.lower())
            idf = log((num_docs + 1) / (docs_with_term + 1)) + 1.0
            score += tf * idf

        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    limit = top_k if top_k else num_docs
    return [doc for _, doc in scored[:limit]]


# ─── Context Formatting ──────────────────────────────────────────────────────

def format_local_context(retrieved_docs):
    """Formats retrieved local documents into a single context string."""
    blocks = []
    for idx, doc in enumerate(retrieved_docs, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("source", "unknown_source")
        page = metadata.get("page", metadata.get("page_number", "?"))
        blocks.append(
            f"[Local Doc {idx}] Source: {source}, Page: {page}\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


def format_conversation_history(history):
    """Formats conversation history deque into a prompt section."""
    if not history:
        return ""
    lines = ["Previous conversation:"]
    for entry in history:
        # Truncate long answers to save context window space
        answer = entry["a"]
        if len(answer) > 400:
            answer = answer[:400].rstrip() + "..."
        lines.append(f"Q: {entry['q']}")
        lines.append(f"A: {answer}")
        lines.append("")
    return "\n".join(lines)


def find_missing_terms_in_local_context(question, local_context, max_terms=3):
    """Finds extracted terms from the question that do not appear in local context."""
    lower_context = local_context.lower()
    missing = []
    for term in extract_candidate_terms(question, max_terms=max_terms):
        if term.lower() not in lower_context:
            missing.append(term)
    return missing


# ─── PDF Loading ──────────────────────────────────────────────────────────────

def find_pdf_files(data_path=DATA_PATH, recursive=False):
    """Finds all PDF files in a directory."""
    base_path = Path(data_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {base_path}")
    if not base_path.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {base_path}")

    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = sorted(base_path.glob(pattern))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {base_path} (recursive={recursive})")
    return pdf_files


def load_pdf_standard(pdf_file):
    """Loads a single PDF with PyPDFLoader (lightweight, fast)."""
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(str(pdf_file))
    documents = loader.load()
    for doc in documents:
        doc.metadata = doc.metadata or {}
        doc.metadata["source"] = str(pdf_file)
        # PyPDFLoader uses 0-indexed pages; convert to 1-indexed
        if "page" in doc.metadata:
            doc.metadata["page"] = doc.metadata["page"] + 1
        doc.metadata.setdefault("ingest_method", "standard")
    return documents


def _group_standard_docs_by_page(documents):
    """Groups standard-loaded docs by page number when metadata is available."""
    page_to_text = {}
    for doc in documents:
        metadata = doc.metadata or {}
        page_value = metadata.get("page", metadata.get("page_number"))
        if isinstance(page_value, int):
            page_number = page_value
        elif isinstance(page_value, str) and page_value.isdigit():
            page_number = int(page_value)
        else:
            continue
        page_to_text.setdefault(page_number, []).append(doc.page_content or "")
    return {page: "\n".join(texts).strip() for page, texts in page_to_text.items()}


def _call_ollama_ocr(image_bytes, model_name, prompt, host):
    """Calls local Ollama chat API with an image for OCR."""
    endpoint = host.rstrip("/") + "/api/chat"
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_base64],
            }
        ],
        "stream": False,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Failed to call Ollama OCR endpoint at {endpoint}. "
            f"Ensure `ollama serve` is running and model '{model_name}' is installed."
        ) from error

    content = response_data.get("message", {}).get("content", "")
    return content.strip()


def _render_pdf_pages_to_png(pdf_file, max_pages=None, dpi_scale=1.5):
    """Renders PDF pages to PNG bytes. Uses dpi_scale=1.5 by default (108 DPI)."""
    import fitz  # PyMuPDF

    pages = []
    with fitz.open(str(pdf_file)) as pdf_document:
        total_pages = pdf_document.page_count
        page_limit = total_pages if max_pages is None else min(max_pages, total_pages)
        for page_index in range(page_limit):
            page = pdf_document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi_scale, dpi_scale))
            pages.append((page_index + 1, pixmap.tobytes("png"), total_pages))
    return pages


def _ocr_single_page(page_data, ocr_model, ocr_prompt, ocr_host, pdf_file):
    """OCR a single page — designed to run inside a thread pool."""
    page_number, image_bytes, total_pages = page_data
    print(f"  OCR page {page_number}/{total_pages} from {Path(pdf_file).name}")
    page_text = _call_ollama_ocr(
        image_bytes=image_bytes,
        model_name=ocr_model,
        prompt=ocr_prompt,
        host=ocr_host,
    )
    if not page_text:
        page_text = "[OCR produced no text for this page]"
    return Document(
        page_content=page_text,
        metadata={
            "source": str(pdf_file),
            "page": page_number,
            "ocr_model": ocr_model,
            "ingest_method": "ocr",
        },
    )


def load_pdf_with_ocr(
    pdf_file,
    ocr_model="glm-ocr",
    ocr_prompt=DEFAULT_OCR_PROMPT,
    ocr_host="http://127.0.0.1:11434",
    ocr_max_pages=None,
    ocr_workers=2,
    dpi_scale=1.5,
):
    """Loads a single PDF using per-page OCR via Ollama (parallelized)."""
    page_images = _render_pdf_pages_to_png(pdf_file, max_pages=ocr_max_pages, dpi_scale=dpi_scale)

    if ocr_workers <= 1 or len(page_images) <= 1:
        # Sequential fallback
        documents = []
        for page_data in page_images:
            doc = _ocr_single_page(page_data, ocr_model, ocr_prompt, ocr_host, pdf_file)
            documents.append(doc)
        return documents

    # Parallel OCR
    documents = [None] * len(page_images)
    with ThreadPoolExecutor(max_workers=ocr_workers) as pool:
        future_to_idx = {
            pool.submit(
                _ocr_single_page, page_data, ocr_model, ocr_prompt, ocr_host, pdf_file
            ): idx
            for idx, page_data in enumerate(page_images)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            documents[idx] = future.result()

    return documents


def load_pdf_hybrid(
    pdf_file,
    ocr_model="glm-ocr",
    ocr_prompt=DEFAULT_OCR_PROMPT,
    ocr_host="http://127.0.0.1:11434",
    ocr_max_pages=None,
    min_page_chars=120,
    ocr_workers=2,
    dpi_scale=1.5,
):
    """Uses standard extraction first, then OCR for sparse/empty pages.

    Improvement: pre-renders ALL pages up front so the PDF is only opened once
    for rendering, instead of reopening per sparse page.
    """
    standard_docs = load_pdf_standard(pdf_file)
    grouped_page_text = _group_standard_docs_by_page(standard_docs)

    import fitz  # PyMuPDF
    with fitz.open(str(pdf_file)) as pdf_document:
        total_pages = pdf_document.page_count

    if not grouped_page_text:
        total_standard_chars = sum(len((doc.page_content or "").strip()) for doc in standard_docs)
        threshold = min_page_chars * max(1, total_pages // 2)
        if total_standard_chars >= threshold:
            for doc in standard_docs:
                doc.metadata = doc.metadata or {}
                doc.metadata["ingest_method"] = "hybrid_standard"
            print(f"  Hybrid kept standard extraction for {Path(pdf_file).name} (page metadata unavailable).")
            return standard_docs
        print(f"  Hybrid fallback: OCR all pages for {Path(pdf_file).name} (low extracted text).")
        return load_pdf_with_ocr(
            pdf_file=pdf_file,
            ocr_model=ocr_model,
            ocr_prompt=ocr_prompt,
            ocr_host=ocr_host,
            ocr_max_pages=ocr_max_pages,
            ocr_workers=ocr_workers,
            dpi_scale=dpi_scale,
        )

    page_limit = total_pages if ocr_max_pages is None else min(ocr_max_pages, total_pages)

    # Identify which pages need OCR
    pages_needing_ocr = []
    for page_number in range(1, page_limit + 1):
        standard_text = grouped_page_text.get(page_number, "").strip()
        if len(standard_text) < min_page_chars:
            pages_needing_ocr.append(page_number)

    # Pre-render ONLY the sparse pages in a single PDF open (instead of reopening per page)
    ocr_page_images = {}
    if pages_needing_ocr:
        all_images = _render_pdf_pages_to_png(pdf_file, max_pages=ocr_max_pages, dpi_scale=dpi_scale)
        for page_number, image_bytes, tp in all_images:
            if page_number in pages_needing_ocr:
                ocr_page_images[page_number] = (page_number, image_bytes, tp)

    # Parallel OCR for sparse pages
    ocr_results = {}
    if ocr_page_images:
        ocr_items = list(ocr_page_images.values())
        if ocr_workers <= 1 or len(ocr_items) <= 1:
            for page_data in ocr_items:
                doc = _ocr_single_page(page_data, ocr_model, ocr_prompt, ocr_host, pdf_file)
                ocr_results[page_data[0]] = doc
        else:
            with ThreadPoolExecutor(max_workers=ocr_workers) as pool:
                future_to_page = {
                    pool.submit(
                        _ocr_single_page, pd, ocr_model, ocr_prompt, ocr_host, pdf_file
                    ): pd[0]
                    for pd in ocr_items
                }
                for future in as_completed(future_to_page):
                    pn = future_to_page[future]
                    ocr_results[pn] = future.result()

    # Assemble final document list in page order
    documents = []
    for page_number in range(1, page_limit + 1):
        if page_number in ocr_results:
            doc = ocr_results[page_number]
            doc.metadata["ingest_method"] = "hybrid_ocr"
            documents.append(doc)
        else:
            standard_text = grouped_page_text.get(page_number, "").strip()
            if standard_text:
                documents.append(
                    Document(
                        page_content=standard_text,
                        metadata={
                            "source": str(pdf_file),
                            "page": page_number,
                            "ingest_method": "hybrid_standard",
                        },
                    )
                )
    return documents


def load_single_pdf(pdf_file, ingest_method="standard", **kwargs):
    """Routes a single PDF to the appropriate loader."""
    if ingest_method == "standard":
        return load_pdf_standard(pdf_file)
    elif ingest_method == "ocr":
        return load_pdf_with_ocr(pdf_file=pdf_file, **kwargs)
    elif ingest_method == "hybrid":
        return load_pdf_hybrid(pdf_file=pdf_file, **kwargs)
    else:
        raise ValueError(f"Unknown ingest method: {ingest_method}")


def load_documents(
    data_path=DATA_PATH,
    recursive=False,
    ingest_method="standard",
    ocr_model="glm-ocr",
    ocr_prompt=DEFAULT_OCR_PROMPT,
    ocr_host="http://127.0.0.1:11434",
    ocr_max_pages=None,
    ocr_min_page_chars=120,
    ocr_workers=2,
    dpi_scale=1.5,
):
    """Loads all PDF documents using standard, OCR, or hybrid ingestion."""
    pdf_files = find_pdf_files(data_path=data_path, recursive=recursive)
    all_documents = []

    print(f"Found {len(pdf_files)} PDF file(s). Starting load with method: {ingest_method}")
    loader_kwargs = dict(
        ocr_model=ocr_model,
        ocr_prompt=ocr_prompt,
        ocr_host=ocr_host,
        ocr_max_pages=ocr_max_pages,
        ocr_workers=ocr_workers,
        dpi_scale=dpi_scale,
    )
    if ingest_method == "hybrid":
        loader_kwargs["min_page_chars"] = ocr_min_page_chars

    for pdf_file in pdf_files:
        documents = load_single_pdf(pdf_file, ingest_method=ingest_method, **loader_kwargs)
        print(f"  Loaded {len(documents)} page(s) from {Path(pdf_file).name}")
        all_documents.extend(documents)

    print(f"Total loaded pages/chunks before splitting: {len(all_documents)}")
    return all_documents


# ─── Text Splitting ──────────────────────────────────────────────────────────

def split_documents(documents, chunk_size=700, chunk_overlap=150):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    all_splits = text_splitter.split_documents(documents)
    print(f"Split into {len(all_splits)} chunks")
    return all_splits


# ─── Embeddings ──────────────────────────────────────────────────────────────

def get_embedding_function(model_name="qwen3-embedding:latest", batch_size=32):
    """Initializes Ollama embeddings with batch support."""
    embeddings = OllamaEmbeddings(model=model_name)
    # OllamaEmbeddings doesn't have a native batch_size param, but we store it
    # for use in manual batching during indexing.
    embeddings._batch_size = batch_size
    print(f"Initialized Ollama embeddings: model={model_name}, batch_size={batch_size}")
    return embeddings


def warmup_embedding(embedding_function):
    """Send a dummy embed to pre-load the model into Ollama memory."""
    try:
        embedding_function.embed_query("warmup")
        print("Embedding model warmed up.")
    except Exception as e:
        print(f"Embedding warmup failed (non-fatal): {e}")


# ─── Vector Store ─────────────────────────────────────────────────────────────

def get_vector_store(embedding_function, persist_directory=CHROMA_PATH):
    """Initializes or loads the Chroma vector store."""
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
    )
    print(f"Vector store loaded from: {persist_directory}")
    return vectorstore


def index_documents(chunks, embedding_function, persist_directory=CHROMA_PATH):
    """Indexes document chunks into Chroma with deterministic IDs (prevents duplicates)."""
    if not chunks:
        raise RuntimeError(
            "No chunks were produced from loaded documents. "
            "Try '--ingest-method hybrid' or '--ingest-method ocr' for scanned/image-heavy PDFs."
        )

    print(f"Indexing {len(chunks)} chunks...")

    # Generate deterministic IDs to prevent duplicate entries
    ids = [generate_chunk_id(chunk, idx) for idx, chunk in enumerate(chunks)]

    # Batch insertion for better throughput
    batch_size = getattr(embedding_function, "_batch_size", 32)
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
    )

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        vectorstore.add_documents(batch_chunks, ids=batch_ids)
        done = min(i + batch_size, len(chunks))
        print(f"  Indexed {done}/{len(chunks)} chunks")

    print(f"Indexing complete. Data saved to: {persist_directory}")
    return vectorstore


def incremental_index(args, embedding_function, persist_directory=CHROMA_PATH):
    """Incrementally index only new/changed documents, skip unchanged ones."""
    pdf_files = find_pdf_files(data_path=args.data_path, recursive=args.recursive)
    manifest = load_hash_manifest(persist_directory)
    new_manifest = {}

    files_to_process = []
    unchanged_count = 0

    for pdf_file in pdf_files:
        file_key = str(pdf_file)
        file_hash = compute_file_hash(pdf_file)
        new_manifest[file_key] = file_hash

        if manifest.get(file_key) == file_hash:
            unchanged_count += 1
        else:
            files_to_process.append(pdf_file)

    # Detect deleted files
    current_keys = set(new_manifest.keys())
    deleted_sources = set(manifest.keys()) - current_keys

    print(f"Incremental index: {len(files_to_process)} changed/new, "
          f"{unchanged_count} unchanged, {len(deleted_sources)} deleted")

    # Get or create vector store
    vectorstore = get_vector_store(embedding_function, persist_directory)

    # Remove vectors for changed and deleted files
    sources_to_remove = deleted_sources | {
        str(f) for f in files_to_process if str(f) in manifest
    }
    if sources_to_remove:
        try:
            collection = vectorstore._collection
            for source in sources_to_remove:
                try:
                    collection.delete(where={"source": source})
                    print(f"  Removed old vectors for: {Path(source).name}")
                except Exception:
                    pass
        except Exception:
            pass

    if not files_to_process:
        # Verify the vector store actually has content (guards against first run with stale manifest)
        try:
            count = vectorstore._collection.count()
        except Exception:
            count = 0
        if count == 0:
            print("Warning: vector store is empty but manifest says all files unchanged. "
                  "Run with --mode reset to do a full rebuild.")
        else:
            print(f"All documents unchanged. Nothing to reindex ({count} existing chunks).")
        save_hash_manifest(new_manifest, persist_directory)
        return vectorstore

    # Load only changed/new files
    loader_kwargs = dict(
        ocr_model=args.ocr_model,
        ocr_prompt=args.ocr_prompt,
        ocr_host=args.ocr_host,
        ocr_max_pages=args.ocr_max_pages,
        ocr_workers=args.ocr_workers,
        dpi_scale=args.ocr_dpi_scale,
    )
    if args.ingest_method == "hybrid":
        loader_kwargs["min_page_chars"] = args.ocr_min_page_chars

    all_docs = []
    for pdf_file in files_to_process:
        docs = load_single_pdf(pdf_file, ingest_method=args.ingest_method, **loader_kwargs)
        print(f"  Loaded {len(docs)} page(s) from {Path(pdf_file).name}")
        all_docs.extend(docs)

    chunks = split_documents(all_docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    if chunks:
        ids = [generate_chunk_id(chunk, idx) for idx, chunk in enumerate(chunks)]
        batch_size = getattr(embedding_function, "_batch_size", 32)
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            vectorstore.add_documents(batch_chunks, ids=batch_ids)
        print(f"  Added {len(chunks)} new chunks.")

    save_hash_manifest(new_manifest, persist_directory)
    return vectorstore


def reset_vector_db(persist_directory=CHROMA_PATH):
    """Deletes the existing Chroma vector database and hash manifest."""
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"Deleted existing vector DB at: {persist_directory}")
    else:
        print(f"No existing vector DB found at: {persist_directory}")

    manifest_path = _manifest_path(persist_directory)
    if os.path.exists(manifest_path):
        os.remove(manifest_path)
        print(f"Deleted hash manifest: {manifest_path}")


# ─── RAG Chain & Query ───────────────────────────────────────────────────────

def create_rag_chain(
    vector_store,
    llm_model_name="qwen3.5:9b",
    context_window=16384,
    retrieval_type="mmr",
    retrieval_k=8,
    retrieval_fetch_k=30,
    retrieval_lambda_mult=0.5,
    source_filter=None,
):
    """Creates retriever, prompt, and LLM components for hybrid answering."""
    llm = ChatOllama(
        model=llm_model_name,
        temperature=0,
        num_ctx=context_window,
    )
    print(f"Initialized ChatOllama: model={llm_model_name}, ctx={context_window}")

    # Build retriever search kwargs
    search_kwargs = {"k": retrieval_k}
    if retrieval_type == "mmr":
        search_kwargs["fetch_k"] = retrieval_fetch_k
        search_kwargs["lambda_mult"] = retrieval_lambda_mult

    # Source filtering: restrict retrieval to documents matching the filter
    if source_filter:
        try:
            collection = vector_store._collection
            all_meta = collection.get(include=["metadatas"])["metadatas"]
            matching = list(set(
                m["source"] for m in all_meta
                if source_filter.lower() in m.get("source", "").lower()
            ))
            if matching:
                search_kwargs["filter"] = {"source": {"$in": matching}}
                print(f"Source filter matched {len(matching)} document(s) containing '{source_filter}'")
            else:
                print(f"Warning: source filter '{source_filter}' matched no documents.")
        except Exception as e:
            print(f"Warning: source filtering failed ({e}), proceeding without filter.")

    retriever = vector_store.as_retriever(
        search_type=retrieval_type,
        search_kwargs=search_kwargs,
    )
    print(f"Retriever: type={retrieval_type}, k={retrieval_k}, "
          f"fetch_k={search_kwargs.get('fetch_k')}, lambda={search_kwargs.get('lambda_mult')}")

    template = """You must answer using local context as the primary source.
Use web context only for terminology definitions that are missing from local docs.
If both local and web context do not contain enough information, say exactly:
"I don't have enough information in the provided context."

{conversation_history}Local Context:
{local_context}

Web Terminology Context (Wikipedia fallback):
{web_context}

Question: {question}

Return this format:
1) Answer
2) Sources
   - Local: list local document sources used (or "None")
   - Web: list Wikipedia URLs used (or "None")
"""
    prompt = ChatPromptTemplate.from_template(template)
    print("RAG chain ready.")
    return {"llm": llm, "retriever": retriever, "prompt": prompt}


def query_rag(
    rag_components,
    question,
    web_fallback="off",
    web_max_terms=3,
    web_max_summary_chars=500,
    web_timeout_seconds=6,
    rerank=False,
    conversation_history=None,
):
    """Queries local RAG with streaming output.

    Improvements over original:
    - Web context is gathered BEFORE the LLM call (no double invocation)
    - LLM output is streamed token-by-token
    - Optional reranking of retrieved documents
    - Conversation history for follow-up questions
    """
    print(f"\nQuestion: {question}")

    retriever = rag_components["retriever"]
    prompt = rag_components["prompt"]
    llm = rag_components["llm"]

    # 1. Retrieve documents
    retrieved_docs = retriever.invoke(question)

    # 2. Optional reranking
    if rerank and retrieved_docs:
        retrieved_docs = rerank_documents(question, retrieved_docs)
        print(f"  Reranked {len(retrieved_docs)} documents.")

    # 3. Format local context
    local_context = format_local_context(retrieved_docs)

    # 4. Gather web context BEFORE the LLM call (eliminates double invocation)
    web_context = ""
    web_sources = []
    if web_fallback == "on":
        missing_terms = find_missing_terms_in_local_context(
            question=question,
            local_context=local_context,
            max_terms=web_max_terms,
        )
        context_is_sparse = len(local_context.strip()) < LOCAL_CONTEXT_MIN_CHARS_FOR_NO_WEB_FALLBACK
        if context_is_sparse or missing_terms:
            if context_is_sparse:
                print("  Local context sparse — running Wikipedia fallback...")
            else:
                print(f"  Terms not in local context ({', '.join(missing_terms)}) — running Wikipedia fallback...")
            web_context, web_sources = build_web_context(
                question=question,
                max_terms=web_max_terms,
                max_summary_chars=web_max_summary_chars,
                timeout_seconds=web_timeout_seconds,
            )
            if web_sources:
                print(f"  Wikipedia added {len(web_sources)} definition(s).")

    # 5. Build conversation history context
    conv_history_text = format_conversation_history(conversation_history) if conversation_history else ""

    # 6. Invoke LLM with streaming
    messages = prompt.invoke(
        {
            "local_context": local_context if local_context.strip() else "None",
            "web_context": web_context if web_context.strip() else "None",
            "question": question,
            "conversation_history": conv_history_text,
        }
    )

    print("\nResponse:")
    full_response = ""
    for chunk in llm.stream(messages):
        token = getattr(chunk, "content", str(chunk))
        print(token, end="", flush=True)
        full_response += token
    print()  # Final newline

    return full_response


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Local RAG pipeline (optimized)")

    # Mode
    parser.add_argument(
        "--mode", choices=["reuse", "reindex", "reset"], default="reuse",
        help="reuse: load existing DB; reindex: incremental rebuild; reset: full rebuild",
    )

    # Document ingestion
    parser.add_argument("--data-path", default=DATA_PATH, help="Directory containing PDF files")
    parser.add_argument("--recursive", action="store_true", help="Scan subdirectories for PDFs")
    parser.add_argument(
        "--ingest-method", choices=["standard", "ocr", "hybrid"], default="standard",
        help="Ingestion method: standard (fast text), ocr (vision model), hybrid (auto-fallback)",
    )

    # OCR settings
    parser.add_argument("--ocr-model", default="glm-ocr", help="Ollama OCR model name")
    parser.add_argument("--ocr-prompt", default=DEFAULT_OCR_PROMPT, help="Prompt for OCR extraction")
    parser.add_argument("--ocr-host", default="http://127.0.0.1:11434", help="Ollama API base URL")
    parser.add_argument("--ocr-max-pages", type=int, default=None, help="Max pages per PDF for OCR")
    parser.add_argument("--ocr-min-page-chars", type=int, default=120, help="Hybrid: min chars before OCR fallback")
    parser.add_argument("--ocr-workers", type=int, default=2, help="Parallel OCR workers (default: 2)")
    parser.add_argument("--ocr-dpi-scale", type=float, default=1.5, help="Render DPI scale for OCR (default: 1.5)")

    # Text splitting
    parser.add_argument("--chunk-size", type=int, default=700, help="Chunk size for splitting")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap for splitting")

    # Models
    parser.add_argument("--llm-model", default="qwen3.5:9b", help="Ollama LLM model for answering")
    parser.add_argument("--embedding-model", default="qwen3-embedding:latest", help="Ollama embedding model")
    parser.add_argument("--embedding-batch-size", type=int, default=32, help="Embedding batch size for indexing")

    # LLM & retrieval
    parser.add_argument("--context-window", type=int, default=16384, help="LLM context window (num_ctx)")
    parser.add_argument("--retrieval-type", choices=["similarity", "mmr"], default="mmr", help="Search strategy")
    parser.add_argument("--retrieval-k", type=int, default=8, help="Chunks returned to prompt")
    parser.add_argument("--retrieval-fetch-k", type=int, default=30, help="MMR candidate pool size")
    parser.add_argument("--retrieval-lambda-mult", type=float, default=0.5, help="MMR diversity (0-1)")

    # Reranking
    parser.add_argument("--rerank", choices=["on", "off"], default="off", help="Enable TF-IDF reranking")

    # Source filtering
    parser.add_argument("--source-filter", default=None, help="Filter retrieval to sources containing this substring")

    # Conversation memory
    parser.add_argument("--memory-size", type=int, default=3, help="Conversation history turns to keep (0 to disable)")

    # Web fallback
    parser.add_argument("--web-fallback", choices=["on", "off"], default="off", help="Wikipedia terminology fallback")
    parser.add_argument("--web-max-terms", type=int, default=3, help="Max terms to look up per question")
    parser.add_argument("--web-max-summary-chars", type=int, default=500, help="Max chars per Wikipedia summary")
    parser.add_argument("--web-timeout-seconds", type=int, default=6, help="Timeout per Wikipedia request")

    args = parser.parse_args()

    # Validation
    if args.ocr_max_pages is not None and args.ocr_max_pages < 1:
        parser.error("--ocr-max-pages must be >= 1")
    if args.ocr_min_page_chars < 1:
        parser.error("--ocr-min-page-chars must be >= 1")
    if args.ocr_workers < 1:
        parser.error("--ocr-workers must be >= 1")
    if args.ocr_dpi_scale < 0.5 or args.ocr_dpi_scale > 4.0:
        parser.error("--ocr-dpi-scale must be between 0.5 and 4.0")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be >= 1")
    if args.chunk_overlap < 0:
        parser.error("--chunk-overlap must be >= 0")
    if args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-overlap must be less than --chunk-size")
    if args.context_window < 1024:
        parser.error("--context-window must be >= 1024")
    if args.retrieval_k < 1:
        parser.error("--retrieval-k must be >= 1")
    if args.retrieval_fetch_k < 1:
        parser.error("--retrieval-fetch-k must be >= 1")
    if args.retrieval_fetch_k < args.retrieval_k:
        parser.error("--retrieval-fetch-k must be >= --retrieval-k")
    if not 0 <= args.retrieval_lambda_mult <= 1:
        parser.error("--retrieval-lambda-mult must be between 0 and 1")
    if args.embedding_batch_size < 1:
        parser.error("--embedding-batch-size must be >= 1")
    if args.memory_size < 0:
        parser.error("--memory-size must be >= 0")
    if args.web_max_terms < 1:
        parser.error("--web-max-terms must be >= 1")
    if args.web_max_summary_chars < 100:
        parser.error("--web-max-summary-chars must be >= 100")
    if args.web_timeout_seconds < 1:
        parser.error("--web-timeout-seconds must be >= 1")

    return args


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    embedding_function = get_embedding_function(
        model_name=args.embedding_model,
        batch_size=args.embedding_batch_size,
    )
    warmup_embedding(embedding_function)

    print(
        f"Config: mode={args.mode}, ingest={args.ingest_method}, "
        f"chunk={args.chunk_size}/{args.chunk_overlap}, "
        f"retrieval={args.retrieval_type} k={args.retrieval_k}, "
        f"rerank={args.rerank}, memory={args.memory_size}, "
        f"web_fallback={args.web_fallback}"
    )

    if args.mode == "reuse":
        print("Mode: reuse existing vector DB.")
        vector_store = get_vector_store(embedding_function)

    elif args.mode == "reindex":
        print("Mode: incremental reindex.")
        vector_store = incremental_index(args, embedding_function)

    elif args.mode == "reset":
        print("Mode: reset and full rebuild.")
        reset_vector_db()
        docs = load_documents(
            data_path=args.data_path,
            recursive=args.recursive,
            ingest_method=args.ingest_method,
            ocr_model=args.ocr_model,
            ocr_prompt=args.ocr_prompt,
            ocr_host=args.ocr_host,
            ocr_max_pages=args.ocr_max_pages,
            ocr_min_page_chars=args.ocr_min_page_chars,
            ocr_workers=args.ocr_workers,
            dpi_scale=args.ocr_dpi_scale,
        )
        chunks = split_documents(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        vector_store = index_documents(chunks, embedding_function)
        # Save manifest for future incremental runs
        pdf_files = find_pdf_files(data_path=args.data_path, recursive=args.recursive)
        manifest = {str(f): compute_file_hash(f) for f in pdf_files}
        save_hash_manifest(manifest)

    rag_components = create_rag_chain(
        vector_store,
        llm_model_name=args.llm_model,
        context_window=args.context_window,
        retrieval_type=args.retrieval_type,
        retrieval_k=args.retrieval_k,
        retrieval_fetch_k=args.retrieval_fetch_k,
        retrieval_lambda_mult=args.retrieval_lambda_mult,
        source_filter=args.source_filter,
    )

    # Conversation memory and query cache
    conversation_history = deque(maxlen=args.memory_size) if args.memory_size > 0 else None
    query_cache = {}

    query_question = input("Enter a question (type '/end' to quit): \n")

    while query_question != "/end":
        # Check cache for identical questions
        if query_question in query_cache:
            print("\n(cached) Response:")
            print(query_cache[query_question])
        else:
            response = query_rag(
                rag_components,
                query_question,
                web_fallback=args.web_fallback,
                web_max_terms=args.web_max_terms,
                web_max_summary_chars=args.web_max_summary_chars,
                web_timeout_seconds=args.web_timeout_seconds,
                rerank=(args.rerank == "on"),
                conversation_history=conversation_history,
            )

            # Cache the response
            query_cache[query_question] = response

            # Update conversation history
            if conversation_history is not None:
                conversation_history.append({"q": query_question, "a": response})

        query_question = input("Enter a question (type '/end' to quit): \n")

    print("Ending program.")
