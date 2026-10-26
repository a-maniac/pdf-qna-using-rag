from pypdf import PdfReader
from dotenv import load_dotenv
load_dotenv()
reader=PdfReader("sample-text-only-pdf-a4-size.pdf")
import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient,models

llm_key=os.getenv("OPENROUTER_API_KEY")

llm=ChatOpenAI(
    model="deepseek/deepseek-v4.1-flash",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
    )

embedding_model=OpenAIEmbeddings(
    model='openai/text-embedding-3-small',
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

#print(llm_key)

# print(len(reader.pages))
# print(reader.pages[0].extract_text())
# print(reader.pages[0].images)

def chunk_text(text,size=500, overlap=100):
    print(len(text))
    step=size-overlap
    if step <= 0:
        step = size
    #for in range (0,57307,400)
    return [text[i:i+size] for i in range(0, len(text), step)]

text = "\n".join(page.extract_text() or "" for page in reader.pages)

# all_pages_text = []

# for page in reader.pages:
#     page_text = page.extract_text()
#     if page_text is None:
#         page_text = ""
#     all_pages_text.append(page_text)

# text = "\n".join(all_pages_text)

chunks = chunk_text(text)

print(len(chunks))

embeddings =embedding_model.embed_documents(chunks)

response = llm.invoke("What is Pune?")
#print(response.content)

vec = embedding_model.embed_query("Pune is in Maharashtra")
print("Vector type:", type(vec))
print("Vector length:", len(vec))
print("First 5 values:", vec[:5])
print("Number of dimensions:", len(vec))

qdrant=QdrantClient(
    url='http://localhost:6333'
)

COLLECTION_NAME="rag_fundamentals_1"

if qdrant.collection_exists(COLLECTION_NAME):
    qdrant.delete_collection(COLLECTION_NAME)

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config= models.VectorParams(
        size=1536,
        distance=models.Distance.COSINE
    )
    )

print("Collection created : ", COLLECTION_NAME)

points=[
    models.PointStruct(
        id=i,
        vector=embedding,
        payload={
            "chunk_id":i,
            "text": chunk
        }
    )
    for i,(chunk,embedding) in enumerate(zip(chunks,embeddings))
]

qdrant.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print(f"Inserted {len(chunks)} chunks into {COLLECTION_NAME}")

user_query="What is the main theme of the pdf?"

embedded_user_query=embedding_model.embed_query(user_query)

#find the ans of embedded_user_query from thr qdrant vector search

ans = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=embedded_user_query,
    limit=3,
    with_payload=True,
).points

# for result in ans:
#     print("Score:", result.score)
#     print(result.payload["text"])
#     print("-" * 50)

context = "\n\n".join(result.payload["text"] for result in ans)
prompt = f"""
Use the following context to answer the user question.

Context:
{context}

Question: {user_query}

Answer briefly and based only on the context.
"""

final_answer = llm.invoke(prompt)
print(final_answer.content)


