# Local RAG Guide

This guide explains the structure of `local_rag.py` and how to run it from Terminal.

## What This Script Does

`local_rag.py` builds a local Retrieval-Augmented Generation (RAG) workflow:

1. Loads PDF files from a folder
2. Splits text into chunks
3. Embeds chunks with Ollama embeddings
4. Stores/retrieves chunks from ChromaDB
5. Answers questions using an Ollama chat model with retrieved context

The ingestion pipeline supports:

- `standard`: fast text extraction via `PyPDFLoader`
- `ocr`: page-by-page OCR through Ollama vision model (for example `glm-ocr`), parallelized across workers
- `hybrid`: standard extraction first, OCR fallback on low-text pages

## Code Structure

Main file: `local_rag.py`

- `find_pdf_files(data_path, recursive)` — Finds all `.pdf` files in the target folder.
- `load_pdf_standard(pdf_file)` — Fast text extraction using PyPDFLoader.
- `load_pdf_with_ocr(pdf_file, ...)` — Renders pages to PNG and sends to Ollama OCR model (parallel workers).
- `load_pdf_hybrid(pdf_file, ...)` — Standard extraction first, OCR fallback on sparse pages. Pre-renders all pages in a single pass.
- `load_single_pdf(pdf_file, ingest_method, ...)` — Routes a single PDF to the appropriate loader.
- `load_documents(data_path, ...)` — Loads all PDFs using the chosen ingestion method.
- `split_documents(documents, chunk_size, chunk_overlap)` — Splits loaded documents into chunks (defaults: `chunk_size=700`, `chunk_overlap=150`).
- `get_embedding_function(model_name, batch_size)` — Creates Ollama embedding function with configurable batch size.
- `warmup_embedding(embedding_function)` — Pre-loads the embedding model into Ollama memory to avoid cold-start latency.
- `get_vector_store(...)` — Loads an existing Chroma vector database.
- `index_documents(...)` — Indexes chunks into Chroma with deterministic IDs (prevents duplicates) and batched insertion.
- `incremental_index(...)` — SHA-256 hash-based incremental indexing: only processes new or changed files, removes vectors for deleted files.
- `reset_vector_db(...)` — Deletes the persisted Chroma database and hash manifest.
- `rerank_documents(query, documents, top_k)` — Lightweight TF-IDF reranker (no extra dependencies).
- `create_rag_chain(...)` — Creates the retrieval + prompt + LLM chain with optional source filtering.
- `query_rag(...)` — Queries RAG with streaming output, optional reranking, and conversation memory.
- `parse_args()` — CLI options for all runtime configuration.

Helper utilities:

- `compute_file_hash(file_path)` — SHA-256 hash for incremental indexing.
- `build_web_context(...)` — Parallel Wikipedia fallback lookups for missing terminology.
- `format_conversation_history(history)` — Formats previous Q&A turns for the prompt.
- `generate_chunk_id(chunk, index)` — Deterministic chunk IDs to prevent duplicates in ChromaDB.

## Folder and Data Layout

Default PDF folder:

- `./docs`

Default vector DB folder (auto-created):

- `chroma_db` (in the working directory where you run the script)

Hash manifest (auto-created, used for incremental indexing):

- `doc_hashes.json` (sibling of the `chroma_db` directory)

Recommended layout:

```text
data/
  local_rag.py
  requirements.txt
  docs/
    file1.pdf
    file2.pdf
    subfolder/
      file3.pdf
```

## Prerequisites

1. Python 3.10+ installed
2. Ollama installed and running
3. Required Python packages installed
4. PyMuPDF (`pymupdf`) is required for OCR and hybrid ingestion

## Terminal Setup

From Terminal:

```bash
cd data
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
python3 -m pip install -r requirements.txt
```

Start Ollama server in another Terminal if needed:

```bash
ollama serve
```

Pull required models:

```bash
ollama pull qwen3-embedding:latest
ollama pull qwen3.5:9b
ollama pull glm-ocr:latest
```

## How to Run

Go to the project folder:

```bash
cd data
source .venv/bin/activate
```

### Mode 1: Reuse existing vector DB

Uses existing `chroma_db` without re-indexing:

```bash
python3 local_rag.py --mode reuse
```

### Mode 2: Incremental reindex

Only processes new or changed PDFs. Unchanged files are skipped automatically using SHA-256 hashes:

```bash
python3 local_rag.py --mode reindex --recursive --ingest-method hybrid --ocr-model glm-ocr
```

### Mode 3: Reset and full rebuild

Deletes previous vectors and hash manifest, then rebuilds everything from scratch:

```bash
python3 local_rag.py --mode reset --recursive --ingest-method hybrid --ocr-model glm-ocr
```

### Use a custom PDF folder

```bash
python3 local_rag.py --mode reset --data-path "./docs"
```

### Include PDFs from nested subfolders

```bash
python3 local_rag.py --mode reset --recursive
```

### OCR ingestion with `glm-ocr`

Use OCR for every page (best for scanned/graph-heavy docs):

```bash
python3 local_rag.py --mode reset --ingest-method ocr --ocr-model glm-ocr
```

### Hybrid ingestion (recommended for mixed PDFs)

Use normal text extraction where possible, OCR fallback on sparse pages:

```bash
python3 local_rag.py --mode reset --ingest-method hybrid --ocr-model glm-ocr
```

### OCR tuning examples

Limit OCR to first 20 pages per PDF:

```bash
python3 local_rag.py --mode reindex --ingest-method ocr --ocr-model glm-ocr --ocr-max-pages 20
```

Use 4 parallel OCR workers for faster processing:

```bash
python3 local_rag.py --mode reset --ingest-method ocr --ocr-model glm-ocr --ocr-workers 4
```

Reduce DPI scale for faster OCR (lower quality, smaller images):

```bash
python3 local_rag.py --mode reset --ingest-method ocr --ocr-model glm-ocr --ocr-dpi-scale 1.0
```

Override OCR prompt for charts and tables:

```bash
python3 local_rag.py --mode reindex --ingest-method ocr --ocr-model glm-ocr --ocr-prompt "Extract chart legends, axis labels, values, and table cells in structured bullets."
```

Use a custom Ollama host:

```bash
python3 local_rag.py --mode reindex --ingest-method ocr --ocr-host "http://127.0.0.1:11434"
```

## Model Selection

Both the LLM and embedding model are configurable via CLI flags, so you can swap to faster or larger models without editing code:

- `--llm-model` (default `qwen3.5:9b`) — the chat model used for answering
- `--embedding-model` (default `qwen3-embedding:latest`) — the embedding model used for vectorization
- `--embedding-batch-size` (default `32`) — batch size for embedding during indexing

Example using a smaller, faster LLM:

```bash
python3 local_rag.py --mode reuse --llm-model qwen2.5:3b --context-window 8192
```

## Retrieval and Context Tuning

The script supports runtime tuning for chunking, retrieval, and model context window:

- `--chunk-size` (default `700`) — Characters per chunk before embedding. Recommended max: `1200` to `1500` for technical docs.
- `--chunk-overlap` (default `150`) — Overlap between neighboring chunks. Keep around 15 to 25 percent of chunk size.
- `--context-window` (default `16384`) — Ollama `num_ctx` (total token budget). Practical range: `8192` to `32768`.
- `--retrieval-type` (`similarity` or `mmr`, default `mmr`) — `similarity` returns top nearest chunks; `mmr` balances relevance with diversity.
- `--retrieval-k` (default `8`) — Number of chunks sent to the LLM. Recommended max: `12` to `16`.
- `--retrieval-fetch-k` (default `30`) — MMR candidate pool size. Recommended max: `50` to `80`. Must be `>= retrieval-k`.
- `--retrieval-lambda-mult` (default `0.5`) — MMR relevance vs diversity trade-off (`0` to `1`). Recommended range: `0.3` to `0.8`.

### Balanced preset (recommended starting point)

```bash
python3 local_rag.py --mode reset --recursive --ingest-method hybrid --ocr-model glm-ocr --chunk-size 700 --chunk-overlap 150 --retrieval-type mmr --retrieval-k 8 --retrieval-fetch-k 30 --retrieval-lambda-mult 0.5 --context-window 16384
```

### High-recall preset (broader retrieval)

```bash
python3 local_rag.py --mode reset --recursive --ingest-method hybrid --ocr-model glm-ocr --chunk-size 700 --chunk-overlap 150 --retrieval-type mmr --retrieval-k 12 --retrieval-fetch-k 50 --retrieval-lambda-mult 0.35 --context-window 16384
```

### Lower-latency preset

```bash
python3 local_rag.py --mode reindex --retrieval-type similarity --retrieval-k 5 --chunk-size 800 --chunk-overlap 120 --context-window 8192 --llm-model qwen2.5:3b
```

## Reranking

After retrieval, an optional TF-IDF reranker reorders chunks by relevance to the query. This can improve answer quality without adding external dependencies.

- `--rerank` (`on` or `off`, default `off`)

```bash
python3 local_rag.py --mode reuse --rerank on
```

## Source Filtering

Restrict retrieval to documents whose source path contains a given substring. Useful when your knowledge base has multiple topics and you want to scope queries to a specific area.

- `--source-filter` (default: none)

```bash
python3 local_rag.py --mode reuse --source-filter "slides"
python3 local_rag.py --mode reuse --source-filter "diagrams"
```

## Conversation Memory

The query loop maintains a rolling window of previous Q&A turns, allowing follow-up questions like "tell me more about that" or "how does that relate to the previous topic".

- `--memory-size` (default `3`, set to `0` to disable)

Previous answers are truncated to 400 characters in the prompt to conserve context window space.

```bash
python3 local_rag.py --mode reuse --memory-size 5
python3 local_rag.py --mode reuse --memory-size 0   # disable memory
```

## Query Caching

Identical questions within the same session are served instantly from an in-memory cache. No configuration needed — this is always active.

## Web Terminology Fallback (Wikipedia)

When local docs do not contain enough terminology definitions, you can enable fallback Wikipedia enrichment. Lookups are now parallelized across terms for faster response.

- `--web-fallback` (`off` or `on`, default `off`) — Enable/disable Wikipedia fallback.
- `--web-max-terms` (default `3`) — Max terms to look up per question. Recommended max: `5` to `8`.
- `--web-max-summary-chars` (default `500`) — Max chars per Wikipedia summary. Recommended max: `600` to `800`.
- `--web-timeout-seconds` (default `6`) — Timeout per Wikipedia request. Recommended max: `8` to `10`.

Behavior:

1. Local retrieval runs first.
2. Web context is gathered before the LLM call (not after) to avoid double invocation.
3. If local context is sparse, the script extracts likely terms from the question (for example, `SBC`, `Call Proxy`, `SIPREC`) and fetches Wikipedia summaries in parallel.
4. The final answer is generated with explicit source split: local documents and web (Wikipedia).

### Baseline (local docs only)

```bash
python3 local_rag.py --mode reuse --web-fallback off
```

### Enable fallback Wikipedia enrichment

```bash
python3 local_rag.py --mode reuse --web-fallback on --web-max-terms 3 --web-max-summary-chars 500 --web-timeout-seconds 6
```

## Verification Workflow for Recall Issues

1. Reindex with a balanced preset.
2. Prepare 5 to 10 previously failing questions.
3. Run with `--retrieval-k 8`, then compare with `--retrieval-k 12`.
4. Try `--rerank on` to see if reranking improves answers.
5. Keep the lowest-latency config that still answers all target questions.
6. If docs are scanned/diagram-heavy, use `--ingest-method hybrid` or `ocr`.
7. If terminology is missing from docs, compare `--web-fallback off` vs `on` using the same question.

## Interactive Querying

After startup, the script prompts for questions. LLM responses are streamed token-by-token so you see results as they are generated. Repeated identical questions are served instantly from cache.

- Enter a question to query the RAG chain
- Type `/end` to exit

Example:

```text
Enter a question (type '/end' to quit):
Summarize my work experience from the documents.
```

## Troubleshooting

- `No PDF files found...` — Confirm PDFs exist in `./docs` (or your `--data-path`).

- Import warnings in editor (`could not be resolved`) — Usually means dependencies are not installed in the interpreter selected by your IDE.

- Ollama/model errors — Ensure `ollama serve` is running and models (`qwen3-embedding:latest`, `qwen3.5:9b`, `glm-ocr`) are pulled.

- OCR import errors (`No module named fitz`) — Install PyMuPDF in your active environment: `pip install pymupdf`

- OCR slow — Increase `--ocr-workers` (try `3` or `4`), reduce `--ocr-dpi-scale` (try `1.0`), or limit pages with `--ocr-max-pages`.

- Slow indexing — Use `--mode reindex` for incremental indexing (only processes changed files). For full rebuilds, indexing time is dominated by embedding; increase `--embedding-batch-size` if your system can handle it.

- Weak answers despite indexing — Try `--rerank on`, increase `--retrieval-k`, use `--retrieval-type mmr`, and reindex with tuned `--chunk-size`/`--chunk-overlap`.

- Vector store empty warning on reindex — Run `--mode reset` once to do a full rebuild, then use `--mode reindex` for subsequent updates.

## Full CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `reuse` | `reuse`, `reindex` (incremental), or `reset` (full rebuild) |
| `--data-path` | `./docs` | PDF folder path |
| `--recursive` | off | Scan subdirectories |
| `--ingest-method` | `standard` | `standard`, `ocr`, or `hybrid` |
| `--ocr-model` | `glm-ocr` | Ollama vision model for OCR |
| `--ocr-prompt` | (built-in) | Custom OCR extraction prompt |
| `--ocr-host` | `http://127.0.0.1:11434` | Ollama API base URL |
| `--ocr-max-pages` | none | Max pages per PDF for OCR |
| `--ocr-min-page-chars` | `120` | Hybrid: min chars before OCR fallback |
| `--ocr-workers` | `2` | Parallel OCR workers |
| `--ocr-dpi-scale` | `1.5` | Render scale for OCR images |
| `--chunk-size` | `700` | Characters per chunk |
| `--chunk-overlap` | `150` | Overlap between chunks |
| `--llm-model` | `qwen3.5:9b` | Ollama LLM for answering |
| `--embedding-model` | `qwen3-embedding:latest` | Ollama embedding model |
| `--embedding-batch-size` | `32` | Batch size for embedding |
| `--context-window` | `16384` | LLM context window (num_ctx) |
| `--retrieval-type` | `mmr` | `similarity` or `mmr` |
| `--retrieval-k` | `8` | Chunks sent to prompt |
| `--retrieval-fetch-k` | `30` | MMR candidate pool |
| `--retrieval-lambda-mult` | `0.5` | MMR diversity (0-1) |
| `--rerank` | `off` | TF-IDF reranking (`on`/`off`) |
| `--source-filter` | none | Filter sources by substring |
| `--memory-size` | `3` | Conversation history turns (0 = off) |
| `--web-fallback` | `off` | Wikipedia fallback (`on`/`off`) |
| `--web-max-terms` | `3` | Terms to look up per question |
| `--web-max-summary-chars` | `500` | Max chars per summary |
| `--web-timeout-seconds` | `6` | Timeout per request |

## My Commands

Max Context + Website Search (full rebuild):
`python3 local_rag.py --mode reset --recursive --ingest-method hybrid --ocr-model glm-ocr --ocr-workers 4 --chunk-size 1200 --chunk-overlap 200 --retrieval-type mmr --retrieval-k 12 --retrieval-fetch-k 50 --retrieval-lambda-mult 0.4 --context-window 16384 --rerank on --web-fallback on --web-max-terms 3 --web-max-summary-chars 600 --web-timeout-seconds 10`

Max Context + Website Search (without rebuilding):
`python3 local_rag.py --mode reuse --retrieval-type mmr --retrieval-k 12 --retrieval-fetch-k 50 --retrieval-lambda-mult 0.4 --context-window 16384 --rerank on --web-fallback on --web-max-terms 3 --web-max-summary-chars 600 --web-timeout-seconds 6`

Min Latency:
`python3 local_rag.py --mode reuse --retrieval-type similarity --retrieval-k 5 --context-window 8192 --memory-size 0`

Balanced Mode:
`python3 local_rag.py --mode reuse --retrieval-type mmr --retrieval-k 8 --retrieval-fetch-k 30 --retrieval-lambda-mult 0.5 --context-window 16384 --rerank on`