import sys
import os
import pandas as pd
import time
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
from pathlib import Path
from loadChunks import create_chunks
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from promptLLM import call_LLM
from buildQuery import returnQuery

APP_DIR = Path(__file__).resolve().parent
CACHE_PATH = APP_DIR / "embeddings.parquet"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

force_refresh = "--refresh" in sys.argv

if os.path.exists(CACHE_PATH) and not force_refresh:
    print("Loading cached embeddings")
    df = pd.read_parquet(CACHE_PATH)
    chunks = df["chunk"].tolist()
    embeddings = np.array(df["embedding"].tolist())
else:
    print("Generating embeddings")
    chunks, metas = create_chunks()
    embeddings = model.encode(chunks, show_progress_bar=True)

    df = pd.DataFrame({
        "chunk": chunks,
        "embedding": embeddings.tolist(),
        "source_file": [meta["source_file"] for meta in metas]
    })
    df.to_parquet(CACHE_PATH, index=False)
    print("Embeddings cached")

persist_dir = str(Path(__file__).resolve().parent / "chromaStore")
client = chromadb.PersistentClient(path=persist_dir)

collection = client.get_or_create_collection(name="vat_chunks")

documents = df["chunk"].astype(str).tolist()
embeddings = df["embedding"].tolist()
metadatas = df[["source_file"]].to_dict(orient="records") 


if "id" not in df.columns:
    df["id"] = ["chunk_" + str(i) for i in range(len(df))]

ids = df["id"].tolist()


collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)


def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = model.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0], results["metadatas"][0]


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

@app.post('/query')
def returnResults(q: Query):
    query = q.query

    topChunks, topMetas = retrieve_relevant_chunks(query=query)

    formatted_meta = "\n".join([f"- Source: {m['source_file']}" for m in topMetas])

    context = "\n\n".join(topChunks)

    prompt = returnQuery(context=context, meta=formatted_meta, query=query)

    print("Calling LLM")
    startTime = time.time()
    answer = call_LLM(prompt=prompt)

    endTime = time.time()
    totalTime = endTime - startTime

    print(f"Time elapsed: {totalTime}")

    return {
        "answer": answer,
        "prompt": prompt,
        "chunks": topChunks,
        "metadata": formatted_meta
    }
