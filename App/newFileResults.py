import pandas as pd
from io import BytesIO
import re
import chromadb
import json
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
from promptLLM import call_LLM
from buildQuery import returnQuery
from fuzzywuzzy import process, fuzz

def new_df_to_chunks(df, filename=None):
    documents = []
    metadatas = []
    ids = []

    for idx, row in df.iterrows():
        text = " | ".join(f"{k}: {v}" for k, v in row.items() if pd.notnull(v))
        documents.append(text)
        metadatas.append({
            "source": filename,
            "row_index": idx
        })
        ids.append(f"{filename}_{idx}")
    
    return documents, metadatas, ids

def map_columns(cols, fields):
    prompt = f"""You are a smart data engineer. Your job is to match incoming column names to a target database schema.

    Incoming columns: {cols}

    Target fields: {fields}

    Output a dictionary mapping each incoming column name to a field in the schema. If a column does not match any field, map it to "raw".

    Return only a JSON object mapping the file column names to the DB fields. Do not explain anything else. Example format:

    {{
    "Supplier Name": "name",
    "Voucher No": "voucher_no"
    }}
    """
    print("Mapping Cols")
    response = call_LLM(prompt=prompt)
    print(response)

    cleaned = re.sub(r"^```(?:json)?\n?", "", response.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```$", "", cleaned.strip())
    cleaned = re.sub(r'//.*', '', cleaned)
    cleaned = re.sub(r",\s*(\}|\])", r"\1", cleaned)
    print("Cleaned Response:")
    print(cleaned)  

    return json.loads(cleaned)


def fuzzy_map_columns(cols, fields):
    threshold = 80
    
    mapping = {}
    unmapped = []
    suggestions = {}

    for col in cols:
        best_match = None
        best_score = 0
        for db_field in fields:
            score = fuzz.ratio(col.lower(), db_field.lower())
            if score > best_score:
                best_match = db_field
                best_score = score

        if best_score >= threshold:
            mapping[col] = best_match
        else:
            unmapped.append(col)
            suggestions[col] = best_match  # optional

    return mapping, unmapped, suggestions

def add_file_to_db(contents, fileName):
    df  = pd.read_csv(BytesIO(contents), encoding='utf-8', on_bad_lines='skip') if fileName.endswith('.csv') else pd.read_excel(BytesIO(contents), engine='openpyxl')
    
    df_columns = df.columns.tolist()
    db_fields = ["name", "voucher_no", "invoice_no", "invoice_date", "due_date", "total_invoice_amount", "vat_amount", "amount_paid", "amount_pending", "days_due", "date_received", "trn", "location", "customs_auth", "customs_number", "vat_recovered", "vat_adjustments"]
    print(df_columns)

    # "source_file", "format", "prefix", "company_trn", "company_name"
    # "month", "raw"

    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    mapping = map_columns(df_columns, db_fields)
    fuzzy_mapping, unmapped, suggestions = fuzzy_map_columns(df_columns, db_fields)

    print(fuzzy_mapping, unmapped, suggestions)

    if len(unmapped) > 0:
        # Handle unmapped columns (e.g., log them, map them to "raw", etc.)
        print("Unmapped columns found:")
        for col in unmapped:
            print(f" - {col} -> raw")
        return {
            "status": "partial_mapping",
            "unmapped": unmapped
        }

    new_documents, new_metadatas, new_ids = new_df_to_chunks(df=df, filename=fileName)
    collection_name = fileName.replace(".", "_")

    client = chromadb.Client()
    collection = client.get_or_create_collection(name=collection_name)

    for doc, meta, doc_id in zip(new_documents, new_metadatas, new_ids):
        collection.add(
            documents=[doc],
            metadatas=[meta],
            ids=[doc_id]
        )
    print("Added New file to DB")
    return {"status": "success", "message": f"{fileName} added to database."}

def query_new_collection(query_text, file_name):
    client = chromadb.Client()
    collection = client.get_collection(name=file_name.replace(".", "_"))
    model = SentenceTransformer("all-MiniLM-L6-v2") 

    embedded_query = model.encode(query_text).tolist()

    results = collection.query(
        query_embeddings=[embedded_query],
        n_results=5,
        include=["documents", "metadatas"]
    )

    print(results["documents"])

    documents = results.get("documents")
    metadatas = results.get("metadatas")
    if documents and len(documents) > 0 and documents[0]:
        context_text = "\n\n".join(documents[0])
    else:
        context_text = "No relevant documents found."

    prompt = returnQuery(
        context=context_text,
        meta=metadatas,
        query=query_text
    )
    answer = call_LLM(prompt=prompt)

    return answer