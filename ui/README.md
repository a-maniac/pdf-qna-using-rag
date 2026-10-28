# Document QnA — Web UI

A web UI for the RAG pipeline: upload a PDF, ask questions, and see the answer with the retrieved chunks, their page numbers, and their rerank and vector scores.

It improves on `../main.py` with recursive token-based chunking and a cross-encoder reranker. See `../README.md` for how it works and `../advance_rag.md` for planned improvements.

## Run

From the project root (uses the same `.env` and a running Qdrant on `localhost:6333`):

```bash
source .venv/bin/activate
pip install -r requirements.txt
uvicorn ui.server:app --reload
```

Open http://localhost:8000

The first start downloads the reranker model (~1 GB).
