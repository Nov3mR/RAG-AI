import json
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
from loadChunks import *
from getQuery import *
from buildQuery import *
from promptLLM import *

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
            "format": [meta["format"] for meta in metas],
            "prefix": [meta["prefix"] for meta in metas],
            "company_trn": [meta["company_trn"] for meta in metas],
            "company_name": [meta["company_name"] for meta in metas],
            "name": [meta["name"] for meta in metas],
            "voucher_no": [meta["voucher_no"] for meta in metas],
            "invoice_no": [meta["invoice_no"] for meta in metas],
            "invoice_date": [meta["invoice_date"] for meta in metas],
            "due_date": [meta["due_date"] for meta in metas],
            "total_invoice_amount": [meta["total_invoice_amount"] for meta in metas],
            "vat_amount": [meta["vat_amount"] for meta in metas],
            "amount_paid": [meta["amount_paid"] for meta in metas],
            "amount_pending": [meta["amount_pending"] for meta in metas],
            "days_due": [meta["days_due"] for meta in metas],
            "date_received": [meta["date_received"] for meta in metas],
            "trn": [meta["trn"] for meta in metas],
            "location": [meta["location"] for meta in metas],
            "customs_auth": [meta["customs_auth"] for meta in metas],
            "customs_number": [meta["customs_number"] for meta in metas],
            "vat_recovered": [meta["vat_recovered"] for meta in metas],
            "vat_adjustments": [meta["vat_adjustments"] for meta in metas],
            "month": [meta["month"] for meta in metas],
            "raw": [meta["chunk"] for meta in metas],
            "id": [f"chunk_{i}" for i in range(len(chunks))]
            # "company_name": metas["company_name"][0] if "company_name" in metas else None,  # Assuming all chunks have the same company name
            # "trn": metas["trn"][0] if "trn" in metas else None,  # Assuming all chunks have the same TRN
        })
        df.to_parquet(CACHE_PATH, index=False)
        print("Embeddings cached")

    return df
    
def chromaDBSetup(df, batch_size=5000):
    persist_dir = str(Path(__file__).resolve().parent / "chromaStore")
    client = chromadb.PersistentClient(path=persist_dir)

    if "id" not in df.columns:
        df["id"] = ["chunk_" + str(i) for i in range(len(df))]

    documents = df["chunk"].astype(str).tolist()
    embeddings = df["embedding"].tolist()
    metadatas = df[["source_file", "format", "prefix", "company_trn", "company_name", "name", "voucher_no", "invoice_no", "invoice_date", "due_date", "total_invoice_amount", "vat_amount", "amount_paid", "amount_pending", "days_due", "date_received", "trn", "location", "customs_auth", "customs_number", "vat_recovered", "vat_adjustments", "month", "raw"]].to_dict(orient="records")
    ids = df["id"].tolist()

    collection = client.get_or_create_collection(name="vat_chunks")


    def batch_upsert():
        for i in range(0, len(documents), batch_size):
            collection.upsert(
                documents=documents[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )

    def batch_add():
        for i in range(0, len(documents), batch_size):
            collection.add(
                documents=documents[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )

    if db_refresh:

        print(f"Rebuilding collection")
        batch_upsert()
        print("ChromaDB collection refreshed.")

    else:
        batch_add()

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
    scored.sort(key=lambda x: float(x[1]), reverse=True)
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

    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = model.predict(pairs)

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

def query_by_format_priority(collection, query_text, embedded_query, format_priority_list, top_k, n_results, months):

    raw = []
    ids = []

    results = {
        "documents": [],
        "metadatas": [],
        "ids": [],
        "distances": [],
    }

    for fmt in format_priority_list:

        fmtMonths = ["sar"] if fmt == "SAR" else (months if isinstance(months, list) else [months])
        if n_results <= top_k:
            top_k = n_results

        print("Format:", fmt, "\nMonths:", fmtMonths)
        filtered_result = collection.query(
                query_embeddings=embedded_query,
                n_results=20,
                where={
                    "$and": [
                        {"format": fmt},
                        {"month": {"$in": fmtMonths}}
                    ]
                },
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

def query_chroma(collection, query_text, original_query, n_results, json_query):
    query_text = query_text.lower()
    original_query = original_query.lower()

    raw = []
    jsonMonths = []
    invoice_matches = []
    
    embedded_query = model.encode([query_text]).tolist()

    formats = find_query_format(query_text)
    allFormats = ["SAR", "Box 1", "Box 3", "Box 4", "Box 6/7", "Box 9", "OOS"]

    allMonths = ["may", "june", "july", "august", "september", "sar"]
    months = []
    if json_query != "":
        for col, data in json_query.items():
            if col == "Month":
                if isinstance(data, list):
                    jsonMonths.extend([item.lower() for item in data])
                else:
                    jsonMonths.append(data.lower())

    if len(jsonMonths) > 0 and isinstance(jsonMonths, list):
        months = jsonMonths[0]

    print(months)
    print(jsonMonths)
       

    invoice_pattern = r"\b(?:invoice(?: number| no)?|tax invoice(?:/tax credit note)?(?: number| no)?)[:\s]*([0-9][A-Z0-9/\-]*)"
    json_invoice_pattern = r"invoice\s*(?:no|number)\s*:\s*([^\s|,]+)"
    trn_pattern = r'\b\d{15}\b|\b\d{5,6}[Xx]{4,6}\d{4,6}\b'
    month_pattern = r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b'

    print(original_query)
    invoice_matches = re.findall(invoice_pattern, original_query, re.IGNORECASE)
    json_invoice_matches = re.findall(json_invoice_pattern, original_query, re.IGNORECASE)
    trn_matches = re.findall(trn_pattern, original_query, re.IGNORECASE)
    month_matches = re.findall(month_pattern, original_query, re.IGNORECASE)

    print("REGEX: ", month_matches)

    print("REGEX: ", trn_matches)

    if len(months) < len(month_matches):
        months = month_matches

    json_invoices = []
    if json_invoice_matches:
        for invoice in json_invoice_matches:
            json_invoices.append(invoice[2])

    print("Detected invoice numbers in query: ", invoice_matches)
    print("Detected JSON invoice numbers in query: ", json_invoices)

    if len(invoice_matches) > 0 or len(json_invoice_matches) > 0:
        print([f"Detected invoice numbers in query: {match}" for match in invoice_matches])
        results = collection.query(
            query_embeddings=embedded_query,
            n_results=n_results,
            where={
                "$and": [
                    {"invoice_no": {"$in": invoice_matches if invoice_matches else json_invoices}},
                    {"month": {"$in": months if months else allMonths}}
                ]
            }
        )

        raw = [meta["raw"] for sublist in results["metadatas"] for meta in sublist]

    elif len(trn_matches) > 0:
        print([f"Detected TRN numbers in query: {match}" for match in trn_matches])
        results = collection.query(
            query_embeddings=embedded_query,
            n_results=n_results,
            where={
                "$and": [
                    {"trn": {"$in": trn_matches}},
                    {"month": {"$in": months if months else allMonths}}
                ]
            }
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

        results, raw = query_by_format_priority(collection, query_text, embedded_query, formats, top_k, n_results, months=months if months else allMonths)
    else:
        print("No formats detected, running general semantic search.")
        results, raw = query_by_format_priority(collection, query_text, embedded_query, allFormats, 1, 7, months=months if months else allMonths)
        results, raw = rank_other_chunks(query_text, raw, results["documents"], results["metadatas"], results["ids"], results["distances"])

    print(formats)
    return results, raw

from typing import Union

def perform_arithmetic_from_llm(df: pd.DataFrame, llm_json: Union[str, dict]):

    query_json = llm_json
    if isinstance(query_json, str):
        query_json = json.loads(query_json)
    operation = query_json["operation"]
    field = query_json["field"]
    filters = query_json.get("filters", {})

    filtered = 0

    df_filtered = df.copy()

    def clean_string(s):
        return re.sub(r'[^\w\s]', '', s).strip().lower()  # removes punctuation

    print(df_filtered.columns.tolist())

    for col, condition in filters.items():
        try:
            if condition == "" or (isinstance(condition, str) and condition.startswith("<") and condition.endswith(">")):
                continue

            if isinstance(condition, list) and len(condition) == 1:
                condition = condition[0]

            if df_filtered[col].dtype == object:
                df_filtered[col] = df_filtered[col].astype(str)


            if isinstance(condition, list):
                # Case-insensitive match for lists
                if df_filtered[col].dtype == object:
                    df_filtered = df_filtered[
                    df_filtered[col].str.strip().str.lower().isin([str(v).strip().lower() for v in condition])
                ]
                else:
                    df_filtered = df_filtered[
                    pd.to_numeric(df_filtered[col], errors='coerce').isin([float(v) for v in condition])
                ]
            elif isinstance(condition, str) and condition.startswith(">="):
                value = float(condition[2:])
                df_filtered = df_filtered[pd.to_numeric(df_filtered[col], errors='coerce') >= value]
            elif isinstance(condition, str) and condition.startswith("<="):
                value = float(condition[2:])
                df_filtered = df_filtered[pd.to_numeric(df_filtered[col], errors='coerce') <= value]
            elif isinstance(condition, str) and condition.startswith(">"):
                print(f'Col: {col}, Condition: {condition}')
                value = float(condition[1:])
                print(value)
                df_filtered = df_filtered[pd.to_numeric(df_filtered[col], errors='coerce') > value]
            elif isinstance(condition, str) and condition.startswith("<"):
                value = float(condition[1:])
                df_filtered = df_filtered[pd.to_numeric(df_filtered[col], errors='coerce') < value]
            elif isinstance(condition, str) and (condition.strip().replace('.', '', 1).isdigit() or condition.strip().lstrip('-').replace('.', '', 1).isdigit()):
                value = float(condition)
                df_filtered = df_filtered[pd.to_numeric(df_filtered[col], errors='coerce') == value]
            else:
                print(f'Col: {col}, Condition: {condition}')
                df_filtered = df_filtered[df_filtered[col].apply(lambda x: clean_string(str(x))) == clean_string(condition)]
            filtered += 1
            print("DataFrame: ", df_filtered[field])

            if df_filtered.empty and filtered < len(filters):
                print(f"No data found for filter: {col}")
                df_filtered = df.copy()  # Reset to original DataFrame if no data found
            elif df_filtered.empty:
                print("No data found after applying all filters.")
                return -1, ""
        except Exception as e:
            print(f"Error filtering {col} with condition {condition}: {e}")

    resultRow = []


    if operation in ["sum", "average", "min", "max"]:
        df_filtered[field] = df_filtered[field].astype(str).str.replace(",", "").str.strip()
        df_filtered[field] = pd.to_numeric(df_filtered[field], errors='coerce') 

    #add option for multiple rows
    if operation == "count":
        result = df_filtered[field].count()
        if result <= 6:
            for _, row in df_filtered.iterrows():
                resultRow.append(row.to_dict())
    elif operation == "sum":
        result = df_filtered[field].sum()
        if len(df_filtered) <= 6:
            for _, row in df_filtered.iterrows():
                resultRow.append(row.to_dict())
    elif operation == "average":
        result = df_filtered[field].mean()
        if len(df_filtered) <= 6:
            for _, row in df_filtered.iterrows():
                resultRow.append(row.to_dict())
    elif operation == "max":
        index = df_filtered[field].idxmax()
        result = df_filtered[field].max()
        resultRow.append(df_filtered.loc[index].to_dict())
    elif operation == "min":
        index = df_filtered[field].idxmin()
        result = df_filtered[field].min()
        resultRow.append(df_filtered.loc[index].to_dict())
    else:
        raise ValueError(f"Unsupported operation: {operation}")
    
    
    print("Result: ", result)

    return result, resultRow if resultRow != [] else ""

def retrieve_relevant_chunks(query, top_k=5):

    df = loadOrCreateEmbeddings()
    collection = chromaDBSetup(df)

    jsonQuery, newQuery, parsed, searchQuery = "", "", "", ""

    jsonQuery, newQuery = returnNewQuery(query=query)

    try:
        match = re.search(r"\{.*?\}", jsonQuery, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in response")

        jsonQuery = match.group(0)
        parsed = json.loads(jsonQuery)  
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    searchQuery = " | ".join(f"{k}: {v}" for k, v in parsed.items() if k != "query" and v is not None and v not in ["", "null"] and str(v).strip() != "")

    print(searchQuery)

    results, raw = query_chroma(collection, searchQuery.lower() if searchQuery != "" else query.lower(), original_query=query.lower(), n_results=top_k, json_query=parsed) # searchQuery.lower() if searchQuery != "" else newQuery
    
    needArithmetic = False
    if len(results["ids"]) > 1:
        print("Multiple results found. Might Need Arithmetic")

        arithmeticData = returnArithmeticData(query=query)

        try:
            if isinstance(arithmeticData, str):
                arithmeticData = json.loads(arithmeticData.strip())
            # If it's already a dict, do nothing
        except json.JSONDecodeError:
            print("Invalid JSON received from LLM:", arithmeticData)
            arithmeticData = {}


        if arithmeticData.get("field", "").lower() != "none" and arithmeticData.get("operation", "").lower() != "none" and arithmeticData != {}:
            arithmeticResult, resultRow = perform_arithmetic_from_llm(df, arithmeticData)
            needArithmetic = True
            if arithmeticResult and arithmeticResult == -1:
                needArithmetic = False
            print([result['raw'] for result in resultRow] if resultRow != "" else "")

    originalText = []

    for item, text in zip(results["metadatas"], raw):
        if isinstance(item, dict) and text == item.get("raw", ""):
            originalText.append(item.get("raw", ""))
        elif isinstance(item, list):
            for subitem in item:
                if isinstance(subitem, dict) and text == subitem.get("raw", ""):
                    originalText.append(subitem.get("raw", ""))


    return results["documents"], results["metadatas"], results["ids"], originalText, query, [arithmeticResult, [result['raw'] for result in resultRow] if resultRow != "" else ""] if needArithmetic else None

def getResult(query, newFile=None):

    topChunks, topMetas, topIds, originalText, query, arithmetic = retrieve_relevant_chunks(query=query)
    arithmeticResult = None
    arithmeticRow = ""

    if arithmetic:
        arithmeticResult = arithmetic[0] if arithmetic else None
        arithmeticRow = arithmetic[1] if arithmetic[1] != "" else ""

    company_info = ""
    for meta in topMetas:
        if isinstance(meta, dict) and meta['format'] != "SAR":
            company_info = f"""
                    Our company name: {meta['company_name']}
                    Our TRN: {meta['trn']}"""
            break
        else:
            meta = meta[0] if isinstance(meta, list) else meta
            company_info = f"""
                    Our company name: {meta['company_name']}
                    Our TRN: {meta['trn']}"""
            break

    
    formatted_meta = f"""
                    Arithmetic Result: {arithmeticResult if arithmeticResult else "This query might not require an arithmetic result."}
                    """

    if topMetas and isinstance(topMetas[0], list):
        
        invoice_info = "\n".join([f"""- Invoice Number: {m[0]['invoice_no']}
            - Type of Transaction: {m[0]['prefix']}""" for m in topMetas])
    else:
        invoice_info = "\n".join([f"""- Invoice Number: {m['invoice_no']}
            - Type of Transaction: {m['prefix']}""" for m in topMetas])

    formatted_meta = company_info + "\n" + formatted_meta
    meta = formatted_meta + "\n" + invoice_info

    context = "\n\n".join(originalText)

    # if newFile:
    #     newFileResults = ""
    #     context += f"\n\n {newFileResults["documents"][0]}"

    prompt = ""

    print(arithmeticResult, arithmeticRow)

    if arithmeticResult and isinstance(arithmeticResult, float) and (arithmeticRow == "" or arithmeticRow is None):
        answer = f"The result is: {arithmeticResult:,.2f}"
    elif arithmeticResult and (arithmeticRow == "" or arithmeticRow is None):
        answer = f"The result is: {arithmeticResult}"
    elif arithmeticRow != "" and arithmeticRow is not None:
        context = "\n".join(arithmeticRow.split())
        meta = formatted_meta
        query = query
        prompt = returnArithmeticQuery(context=context, meta=meta, query=query, arithmetic=arithmeticResult)
        print("Calling LLM for Arithmetic Prompt")
        answer = f"The result is: {arithmeticResult}"
        llmAnswer = call_LLM(prompt=prompt)
        answer += f"\n{llmAnswer}"
    else:
        prompt = returnQuery(context=context, meta=meta, query=query)
        print("Calling LLM")
        answer = call_LLM(prompt=prompt)

    return originalText, answer, prompt, topChunks, formatted_meta, topIds


if __name__ == "__main__":
    docs, metas, ids, originalText, query, arithmeticResult = retrieve_relevant_chunks("Which transactions had a VAT amount above AED 100k in June")
    for i, text in enumerate(originalText):
        print(f"\n\nText {i}: {text}")
        print(f"Metadata {i}: {metas[i]['format']}")
    if arithmeticResult:
        print(f"{result}" for result in arithmeticResult)