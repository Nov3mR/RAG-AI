import os
from pathlib import Path
import sys
import pandas as pd
import re
import numpy as np
import math
import chromadb
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from formatExamples import format_examples
from loadChunks import create_chunks

os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

APP_DIR = Path(__file__).resolve().parent
CACHE_PATH = APP_DIR / "embeddings.parquet"
FORMAT_PATH = APP_DIR / "formats.parquet"

force_refresh = "--refresh" in sys.argv
db_refresh = "--db-refresh" in sys.argv
format_refresh = "--format-refresh" in sys.argv
    
model1 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer("intfloat/e5-small-v2")
model2 = SentenceTransformer("all-mpnet-base-v2")


def loadOrCreateEmbeddings():
    if os.path.exists(CACHE_PATH) and not force_refresh:
        print("Loading cached embeddings")
        df = pd.read_parquet(CACHE_PATH)
    else:
        print("Generating embeddings")
        chunks, metas = create_chunks()
        embeddings = model.encode(chunks, show_progress_bar=True)

        df = pd.DataFrame({
            "chunk": chunks,
            "embedding": embeddings.tolist(),
            "source_file": [meta["source_file"] for meta in metas],
            "invoice_no": [meta["invoice_no"] for meta in metas],
            "prefix": [meta["prefix"] for meta in metas],
            "trn": [meta["trn"] for meta in metas],
            "company_name": [meta["company_name"] for meta in metas],
            "format": [meta["format"] for meta in metas],
            "raw": [meta["chunk"] for meta in metas],
            "id": [f"chunk_{i}" for i in range(len(chunks))]
            # "company_name": metas["company_name"][0] if "company_name" in metas else None,  # Assuming all chunks have the same company name
            # "trn": metas["trn"][0] if "trn" in metas else None,  # Assuming all chunks have the same TRN
        })
        df.to_parquet(CACHE_PATH, index=False)
        print("Embeddings cached")

    return df
    
def chromaDBSetup(df):
    persist_dir = str(Path(__file__).resolve().parent / "chromaStore")
    client = chromadb.PersistentClient(path=persist_dir)

    if "id" not in df.columns:
        df["id"] = ["chunk_" + str(i) for i in range(len(df))]

    documents = df["chunk"].astype(str).tolist()
    embeddings = df["embedding"].tolist()
    metadatas = df[["source_file", "invoice_no", 'prefix', 'trn', 'company_name', 'format', "raw"]].to_dict(orient="records")
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

    return collection

def find_query_format(query_text):
    threshold = 0.3
    top_k = 7

    formats = []
    formatExamples = format_examples
    
    example_texts = []
    example_labels = []

    if os.path.exists(FORMAT_PATH) and not format_refresh:
        print("Loading cached examples")
        df = pd.read_parquet(FORMAT_PATH)
        example_texts = df["text"].tolist()
        example_labels = df["label"].tolist()
        example_embeddings = np.array(df["embedding"].tolist(), dtype=np.float32)

    else:
        print("Generating example embeddings")

        for label, examples in formatExamples.items():
            for example in examples:
                example_texts.append(example)
                example_labels.append(label)

        example_embeddings = model2.encode(example_texts, normalize_embeddings=True, show_progress_bar=True)

        df = pd.DataFrame({
            "text": example_texts,
            "label": example_labels,
            "embedding": example_embeddings.tolist()
        })

        df.to_parquet(FORMAT_PATH, index=False)


    customerLabels = ["Box 1", "Box 4", "OOS"]
    supplierLabels = ["Box 3", "Box 6/7", "Box 9", "SAR"]


    CSmatches = re.findall(r'\b(customer|supplier)\b', query_text, re.IGNORECASE)
    roles = set(match.lower() for match in CSmatches)

    customer = False
    supplier = False

    if 'customer' in roles and 'supplier' in roles:
        customer = True
        supplier = True
        print("Both customer and supplier roles detected.")
    elif 'customer' in roles:
        customer = True
        print("Customer role detected.")
    elif 'supplier' in roles:
        supplier = True
        print("Supplier role detected.")


    query_embedding = model2.encode(query_text, normalize_embeddings=True, show_progress_bar=True)

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
                print("added", label)
        if len(formats) >= top_k:
            break

    print(formats, scores)

    if formats:
            
        
        if 'vat' in formats and "Box 4" in formats:
            formats.remove("Box 4")
            formats.remove('vat')
        else:
            filtered = []
            for label in formats:
                if (customer and label in customerLabels) or \
                (supplier and label in supplierLabels) or \
                (not customer and not supplier) or \
                (customer and supplier):
                    filtered.append(label)
                else:
                    print("Removing:", label)

            formats = filtered if filtered else []

    if 'vat' in formats:
        formats.remove('vat')
    

    print(formats)
    return formats

def rerank_chunks(query, retrieved_chunks, metadatas):
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # print(retrieved_chunks)

    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = model.predict(pairs)

    chunk_meta_score = list(zip(retrieved_chunks, metadatas, scores))
    sorted_results = sorted(chunk_meta_score, key=lambda x: x[2], reverse=True)

    ranked_raw_texts = [meta['raw'] for _, meta, _ in sorted_results]
    ranked_scores = [score for _, _, score in sorted_results]

    return ranked_raw_texts, ranked_scores

def query_by_format_priority(collection, query_text, embedded_query, format_priority_list, top_k, n_results):

    raw = []

    results = {
        "documents": [],
        "metadatas": [],
        "ids": [],
        "distances": [],
    }

    for fmt in format_priority_list:

        if n_results <= top_k:
            top_k = n_results

        filtered_result = collection.query(
                query_embeddings=embedded_query,
                n_results=20,
                where={"format": fmt},
                include=["documents", "metadatas", "distances"]
        )

        rankedChunks, rankedScores = rerank_chunks(query_text, filtered_result["documents"][0], filtered_result["metadatas"][0])

        for text in rankedChunks[:top_k]:
            raw.append(text)
        print(f"Added {top_k} chunks for {fmt}")

        if filtered_result["documents"]:
            results["documents"].extend(filtered_result["documents"][0])
            results["metadatas"].extend(filtered_result["metadatas"][0])
            results["ids"].extend(filtered_result["ids"][0])
            results["distances"].extend(filtered_result["distances"][0])

        n_results -= top_k

        if n_results == 0:
            break

    return results, raw

def rank_other_chunks(query, chunks):
    model = SentenceTransformer("intfloat/e5-small-v2")

    chunksCopy = [chunk.lower() for chunk in chunks]

    query_embedding = model.encode(query, convert_to_tensor=True)
    chunk_embeddings = model.encode(chunksCopy, convert_to_tensor=True)

    cos_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]

    top_results = sorted(zip(chunks, cos_scores), key=lambda x: x[1], reverse=True)

  
    ranked_chunks = [chunk for chunk, score in top_results]

    return ranked_chunks[:5]

def query_chroma(collection, query_text, n_results):

    raw = []
    
    embedded_query = model.encode([query_text]).tolist()

    formats = find_query_format(query_text)
    allFormats = ["SAR", "Box 1", "Box 3", "Box 4", "Box 6/7", "Box 9", "OOS"]

    pattern = r"\b(?:invoice(?: number| no)?|tax invoice(?:/tax credit note)?(?: number| no)?)[:\s]*([0-9][A-Z0-9/\-]*)"

    matches = re.findall(pattern, query_text, re.IGNORECASE)

    if len(matches) > 0:
        print(f"Detected invoice numbers in query: {match}" for match in matches)
        results = collection.query(
            query_embeddings=embedded_query,
            n_results=n_results,
            where={"invoice_no": {"$in": matches}},
        )
        raw = [meta["raw"] for sublist in results["metadatas"] for meta in sublist]
    elif len(formats) > 0:
        print("No invoice number detected, searching priority formats.")
        print(formats)

        if len(formats) >= n_results:
            top_k = 1
        elif len(formats) == 1:
            top_k = n_results
        else:
            top_k = math.ceil(n_results / len(formats))

        print(top_k, n_results)

        results, raw = query_by_format_priority(collection, query_text, embedded_query, formats, top_k, n_results)
    else:
        print("No formats detected, running general semantic search.")
        results, raw = query_by_format_priority(collection, query_text, embedded_query, allFormats, 1, 7)
        raw = rank_other_chunks(query_text, raw)

    print(formats)
    return results, raw

def retrieve_relevant_chunks(query, top_k=5):
    query = query.lower()
    print(query)
    df = loadOrCreateEmbeddings()
    collection = chromaDBSetup(df)
    results, raw = query_chroma(collection, query, n_results=top_k)
    # originalText = [meta["raw"] for sublist in results["metadatas"] for meta in sublist]

    originalText = []

    for item in results["metadatas"]:
        if isinstance(item, dict):
            originalText.append(item.get("raw", ""))
        elif isinstance(item, list):
            for subitem in item:
                if isinstance(subitem, dict):
                    originalText.append(subitem.get("raw", ""))


    return results["documents"], results["metadatas"], results["ids"], raw, originalText


if __name__ == "__main__":
    docs, metas, ids, raw, originalText = retrieve_relevant_chunks("bjsdbgfbsjfgrgyshfd")
    for i, text in enumerate(raw):
        print(f"\n\nText {i}: {text}")