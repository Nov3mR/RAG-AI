import os
from pathlib import Path
import sys
import pandas as pd
import re
import numpy as np
import math
import torch.nn.functional as F
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
            "trn": [meta["trn"] for meta in metas],
            "prefix": [meta["prefix"] for meta in metas],
            "company_trn": [meta["company_trn"] for meta in metas],
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
    metadatas = df[["source_file", "invoice_no", "trn", 'prefix', 'company_trn', 'company_name', 'format', "raw"]].to_dict(orient="records")
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

def rerank_chunks(query, retrieved_chunks, metadatas, ids, distances):
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # print(retrieved_chunks)

    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = model.predict(pairs)

    print(type(ids), type(distances))

    chunk_meta_score = list(zip(retrieved_chunks, metadatas, scores, ids, distances))
    sorted_results = sorted(chunk_meta_score, key=lambda x: x[2], reverse=True)

    ranked_chunks = [chunk for chunk, _, _, _, _ in sorted_results]
    ranked_metas = [meta for _, meta, _, _, _ in sorted_results]
    ranked_scores = [score for _, _, score, _, _ in sorted_results]
    ranked_ids = [id for _, _, _, id, _ in sorted_results]
    ranked_distances = [distance for _, _, _, _, distance in sorted_results]

    return ranked_chunks, ranked_metas, ranked_scores, ranked_ids, ranked_distances


def rank_other_chunks(query, raw, chunks, metadatas, ids, distances):
    model = SentenceTransformer("intfloat/e5-small-v2")

    results = {
        "documents": [],
        "metadatas": [],
        "ids": [],
        "distances": [],
    }

    query_embedding = model.encode(query, convert_to_tensor=True)
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

    cos_scores = F.cosine_similarity(query_embedding.unsqueeze(0), chunk_embeddings, dim=1) 
    cos_scores = cos_scores.tolist() 

    # cos_scores = util.cos_sim(query_embedding, chunk_embeddings)


    top_results = sorted(list(zip(raw, cos_scores, chunks, metadatas, ids, distances)), key=lambda x: x[1], reverse=True)

    ranked_raw = [r for r, _, _, _, _, _ in top_results]
    ranked_chunks = [chunk for _, score, chunk, _, _, _ in top_results]
    ranked_metas = [meta for _, _, _, meta, _, _ in top_results]
    ranked_ids = [id for _, _, _, _, id, _ in top_results]
    ranked_distances = [distance for _, _, _, _, _, distance in top_results]

    results["documents"].extend(ranked_chunks[:5])
    results["metadatas"].extend(ranked_metas[:5])
    results["ids"].extend(ranked_ids[:5])
    results["distances"].extend(ranked_distances[:5])

    return results, ranked_raw

def query_by_format_priority(collection, query_text, embedded_query, format_priority_list, top_k, n_results):

    raw = []
    ids = []

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

        rankedChunks, rankedMetas, rankedScores, rankedIds, rankedDistances = rerank_chunks(query_text, filtered_result["documents"][0], filtered_result["metadatas"][0], filtered_result["ids"][0], filtered_result["distances"][0])

        for text, id in zip(rankedMetas[:top_k], rankedIds[:top_k]):
            raw.append(text["raw"])
            ids.append(id)
        print(f"Added {top_k} chunks for {fmt}")

        if filtered_result["documents"]:
            results["documents"].extend(rankedChunks[:top_k])
            results["metadatas"].extend(rankedMetas[:top_k])
            results["ids"].extend(rankedIds[:top_k])
            results["distances"].extend(rankedDistances[:top_k])

        n_results -= top_k

        if n_results == 0:
            break

    return results, raw

def query_chroma(collection, query_text, n_results):

    raw = []
    
    embedded_query = model.encode([query_text]).tolist()

    formats = find_query_format(query_text)
    allFormats = ["SAR", "Box 1", "Box 3", "Box 4", "Box 6/7", "Box 9", "OOS"]

    invoice_pattern = r"\b(?:invoice(?: number| no)?|tax invoice(?:/tax credit note)?(?: number| no)?)[:\s]*([0-9][A-Z0-9/\-]*)"
    trn_pattern = r'\b\d{15}\b|\b\d{5,6}[Xx]{4,6}\d{4,6}\b'

    invoice_matches = re.findall(invoice_pattern, query_text, re.IGNORECASE)
    trn_matches = re.findall(trn_pattern, query_text, re.IGNORECASE)

    print(trn_matches)

    if len(invoice_matches) > 0:
        print([f"Detected invoice numbers in query: {match}" for match in invoice_matches])
        results = collection.query(
            query_embeddings=embedded_query,
            n_results=n_results,
            where={"invoice_no": {"$in": invoice_matches}},
        )
        raw = [meta["raw"] for sublist in results["metadatas"] for meta in sublist]

    elif len(trn_matches) > 0:
        print([f"Detected TRN numbers in query: {match}" for match in trn_matches])
        results = collection.query(
            query_embeddings=embedded_query,
            n_results=n_results,
            where={"trn": {"$in": trn_matches}},
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
        results, raw = rank_other_chunks(query_text, raw, results["documents"], results["metadatas"], results["ids"], results["distances"])

    print(formats)
    return results, raw

def retrieve_relevant_chunks(query, top_k=5):
    query = query.lower()
    print(query)
    df = loadOrCreateEmbeddings()
    collection = chromaDBSetup(df)
    results, raw = query_chroma(collection, query, n_results=top_k)

    originalText = []

    for item, text in zip(results["metadatas"], raw):
        if isinstance(item, dict) and text == item.get("raw", ""):
            originalText.append(item.get("raw", ""))
        elif isinstance(item, list):
            for subitem in item:
                if isinstance(subitem, dict) and text == subitem.get("raw", ""):
                    originalText.append(subitem.get("raw", ""))


    return results["documents"], results["metadatas"], results["ids"], originalText


if __name__ == "__main__":
    docs, metas, ids, originalText = retrieve_relevant_chunks("Invoice Number 240530045")
    for i, text in enumerate(originalText):
        print(f"\n\nText {i}: {text}")