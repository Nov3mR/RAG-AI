import sys
import os
import re
import pandas as pd
import time
from sentence_transformers import SentenceTransformer, util
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
from formatExamples import format_examples


os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

APP_DIR = Path(__file__).resolve().parent
CACHE_PATH = APP_DIR / "embeddings.parquet"
model_name = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(model_name)
model2 = SentenceTransformer("all-mpnet-base-v2")
model3 = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
model4 = SentenceTransformer("msmarco-distilbert-base-v4")
model5 = SentenceTransformer("BAAI/bge-base-en-v1.5")

force_refresh = "--refresh" in sys.argv
db_refresh = "--db-refresh" in sys.argv

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
        "source_file": [meta["source_file"] for meta in metas],
        "invoice_no": [meta["invoice_no"] for meta in metas],
        "format": [meta["format"] for meta in metas],
        "id": [f"chunk_{i}" for i in range(len(chunks))]
    })
    df.to_parquet(CACHE_PATH, index=False)
    print("Embeddings cached")

persist_dir = str(Path(__file__).resolve().parent / "chromaStore")
client = chromadb.PersistentClient(path=persist_dir)

if "id" not in df.columns:
    df["id"] = ["chunk_" + str(i) for i in range(len(df))]

documents = df["chunk"].astype(str).tolist()
embeddings = df["embedding"].tolist()
metadatas = df[["source_file", "invoice_no", 'format']].to_dict(orient="records")
ids = df["id"].tolist()


if db_refresh:

    print(f"Rebuilding collection")
    collection = client.get_or_create_collection(name="vat_chunks")

    collection.upsert(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print("ChromaDB collection refreshed.")

else:
    collection = client.get_or_create_collection(name="vat_chunks")
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )


def find_query_format(query_text):
    threshold = 0.3
    top_k = 7

    formats = []
    formatExamples = format_examples

    
    example_texts = []
    example_labels = []

    for label, examples in formatExamples.items():
        for example in examples:
            example_texts.append(example)
            example_labels.append(label)

    example_embeddings = model2.encode(example_texts, normalize_embeddings=True)
    query_embedding = model2.encode(query_text, normalize_embeddings=True)

    similarities = util.cos_sim(query_embedding, example_embeddings)[0]

    scored = list(zip(example_labels, similarities))
    scored.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    scores = []

    for label, score in scored:
        if score > threshold and label not in seen:
            formats.append(label)
            scores.append(score)
            seen.add(label)
        if len(formats) >= top_k:
            break

    print(formats, scores)
    
    if formats and formats[0] == "SAR":
        return ["SAR"]
    elif formats and formats[0] == "Common":
        if 'vat' in formats:
            return ["Box 1", "Box 3", "Box 6/7", "Box 9", "OOS"]
        else:
            return ["Box 1", "Box 3", "Box 4", "Box 6/7", "Box 9", "OOS"]
    else:
        return formats

def query_chroma(collection, query_text, n_results):
    
    embedded_query = model.encode([query_text]).tolist()

    formats = find_query_format(query_text)
    allFormats = ["SAR", "Box 1", "Box 3", "Box 4", "Box 6/7", "Box 9", "OOS"]

    match = re.search(r"\binvoice(?: number| no)?[:\s]*([a-z0-9\-]+)", query_text.lower())

    
    if match:
        invoice_no = match.group(1)
        print(f"Detected invoice number in query: {invoice_no}")
        results = collection.query(
            query_embeddings=embedded_query,
            n_results=n_results,
            where={"invoice_no": invoice_no}
        )
    else:
        print("No invoice number detected, running general semantic search.")
        results = collection.query(
            query_embeddings=embedded_query,
            n_results=n_results,
            where={"format": {"$in": formats if formats else allFormats}}
        )

    print(formats)
    return results

def retrieve_relevant_chunks(query, top_k=5):
    query = query.lower()
    print(query)
    results = query_chroma(collection, query, n_results=top_k)
    return results["documents"][0], results["metadatas"][0], results["ids"][0]


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

    topChunks, topMetas, topIds = retrieve_relevant_chunks(query=query)

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
        "metadata": formatted_meta,
        "ids": topIds
    }



if __name__ == "__main__":
    chunks, metas, ids = retrieve_relevant_chunks("Give me the sales report for last month")
    print("CHUNKS: ", chunks)
    print("METAS: ", metas)
    print("IDS: ", ids)
