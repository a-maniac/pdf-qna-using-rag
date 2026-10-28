import io
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel
from pypdf import PdfReader
from qdrant_client import QdrantClient, models
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

COLLECTION_NAME = "rag_ui"
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"

llm = ChatOpenAI(model="deepseek/deepseek-v4.1-flash", api_key=API_KEY, base_url=BASE_URL)
embedding_model = OpenAIEmbeddings(
    model="openai/text-embedding-3-small", api_key=API_KEY, base_url=BASE_URL
)
qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

app = FastAPI(title="Document QnA")
state = {"filename": None, "chunks": 0}


def chunk_text(text, size=500, overlap=100):
    step = size - overlap if size > overlap else size
    return [text[i : i + size] for i in range(0, len(text), step)]


def chunk_text_using_recursive_text_splitter(pages: list[str], size=500, overlap=75):
    """Split each page on paragraphs -> sentences -> words, sized in tokens, keeping the page number."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=size, chunk_overlap=overlap
    )
    chunks = []
    for page_no, page_text in enumerate(pages, start=1):
        for piece in splitter.split_text(page_text):
            if piece.strip():
                chunks.append({"text": piece, "page": page_no})
    return chunks


class Question(BaseModel):
    question: str
    top_k: int = 3


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/api/status")
def status():
    return state


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")

    reader = PdfReader(io.BytesIO(await file.read()))
    pages = [page.extract_text() or "" for page in reader.pages]
    chunks = chunk_text_using_recursive_text_splitter(pages)
    if not chunks:
        raise HTTPException(400, "No extractable text found in this PDF.")

    embeddings = embedding_model.embed_documents([c["text"] for c in chunks])

    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=len(embeddings[0]), distance=models.Distance.COSINE
        ),
    )
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=i,
                vector=emb,
                payload={"chunk_id": i, "filename": file.filename, **chunk},
            )
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ],
    )

    state.update(filename=file.filename, chunks=len(chunks))
    return {**state, "pages": len(reader.pages)}


# pip install sentence-transformers
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")  # load once, at module level

def retrieve_and_rerank(question: str, top_k=3, fetch_k=20, min_score=None):
    """Returns (point, rerank_score) pairs. Rerank scores are 0-1 and on a different scale than cosine."""
    candidates = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding_model.embed_query(question),
        limit=fetch_k,
        with_payload=True,
    ).points
    if not candidates:
        return []
    scores = reranker.predict([(question, c.payload["text"]) for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [
        (c, float(s)) for c, s in ranked[:top_k] if min_score is None or s >= min_score
    ]

@app.post("/api/ask")
def ask(q: Question):
    if not state["filename"]:
        raise HTTPException(400, "Upload a PDF first.")

    results = retrieve_and_rerank(q.question, q.top_k)

    context = "\n\n".join(f"[p. {r.payload['page']}] {r.payload['text']}" for r, _ in results)
    prompt = f"""
Use the following context to answer the user question.

Context:
{context}

Question: {q.question}

Answer briefly and based only on the context.
"""
    answer = llm.invoke(prompt).content
    sources = [
        {
            "chunk_id": r.payload["chunk_id"],
            "page": r.payload["page"],
            "vector_score": round(r.score, 3),
            "rerank_score": round(rerank_score, 4),
            "text": r.payload["text"],
        }
        for r, rerank_score in results
    ]
    return {"answer": answer, "sources": sources}
