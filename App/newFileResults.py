import pandas as pd
from io import BytesIO
import re
import chromadb
import json
from pdf2image import convert_from_bytes
import pdfplumber
import pytesseract
from PIL import Image
import mimetypes
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
from promptLLM import call_LLM
from buildQuery import returnQuery
from fuzzywuzzy import process, fuzz
from getResults import *
from getQuery import *
import json
import re

def extract_json_from_llm_response(response: str):
    """
    Strips markdown formatting like ```json ... ``` and returns parsed JSON.
    Handles both single and multiple JSON blocks.
    """
    # Extract content between ```json and ```
    matches = re.findall(r"```json(.*?)```", response, re.DOTALL)

    if not matches:
        # fallback: maybe no code block wrapper
        matches = [response.strip()]

    result = []
    for match in matches:
        try:
            # clean whitespace and parse JSON
            parsed = json.loads(match.strip())
            if isinstance(parsed, list):
                result.extend(parsed)
            else:
                result.append(parsed)
        except json.JSONDecodeError as e:
            print(f"Failed to decode: {match[:100]}...")  # optional debug
            raise e
    return result




def extract_text_from_pdf(contents):
    try:
        with pdfplumber.open(BytesIO(contents)) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
            return text.strip()
    except:
        return ""

def extract_text_from_scanned_pdf(contents):
    images = convert_from_bytes(contents, dpi=300)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text.strip()

def extract_text_from_image(contents):
    image = Image.open(BytesIO(contents))
    return pytesseract.image_to_string(image).strip()


def extract_structured_row(lines, fields, fileType):
    prompt = f"""Extract invoice data fields from the following line of text:

    {" | ".join(lines)}

    This text was extracted from a {fileType} file.

    These are the fields you can use: {" , ".join(fields or [])}

    Return a JSON object like:
    {{
    "invoice_no": "INV1001",
    "invoice_date": "14-11-2023",
    "total_invoice_amount": "1000.00",
    "vat_amount": "100.00"
    }}


    Only return valid JSON. If field is not present, set it to "N/A". Any text that is not a field should be added to the "other" field."""

    print("Calling LLM for structured row extraction")
    print(prompt)
    response = call_LLM(prompt, mode="openai")
    print("LLM Response:", response)
    response = extract_json_from_llm_response(response)

    try:
        if isinstance(response, list):
            return response
    except:
        return None


def extract_metadata_from_row(row, aliases=None):
    result = {}
    other_fields = []

    if isinstance(aliases, dict):
        for col, var in aliases.items():
            value = "n/a"
            if col in row and pd.notnull(row[col]):
                value = str(row[col]).strip()
                if value == "-":
                    value = "0"
                else:
                    value = re.sub(r'(?<=\d),(?=\d)', '', value)

            if var == "other":
                if value != "N/A":
                    other_fields.append(f"{col}: {value}")
            else:
                result[var] = value

    # Combine all 'other' values into a single field
    if other_fields:
        result["other"] = " | ".join(other_fields)

    return result

def new_df_to_chunks(df, filename=None, fullMapping=None):
    chunks = []
    metadatas = []


    for idx, row in df.iterrows():

        print(fullMapping)

        result = extract_metadata_from_row(row, aliases=fullMapping)

        print("Result:", result)

        if fullMapping is not None:
            companyTrn = result["trn"] if "trn" in result.keys() else "N/A"
            companyName = result["name"] if "name" in result.keys() else "N/A"
            name = result["name"] if "name" in result.keys() else "N/A"
            voucherNo = result["voucher_no"] if "voucher_no" in result.keys() else "N/A"
            invoiceNo = result["invoice_no"] if "invoice_no" in result.keys() else "N/A"
            invoiceDate = result["invoice_date"] if "invoice_date" in result.keys() else "N/A"
            dueDate = result["due_date"] if "due_date" in result.keys() else "N/A"
            totalInvoiceAmount = result["total_invoice_amount"] if "total_invoice_amount" in result.keys() else "N/A"
            vatAmount = result["vat_amount"] if "vat_amount" in result.keys() else "N/A"
            amountPaid = result["amount_paid"] if "amount_paid" in result.keys() else "N/A"
            amountPending = result["amount_pending"] if "amount_pending" in result.keys() else "N/A"
            daysDue = result["days_due"] if "days_due" in result.keys() else "N/A"
            dateReceived = result["date_received"] if "date_received" in result.keys() else "N/A"
            trn = result["trn"] if "trn" in result.keys() else "N/A"
            location = result["location"] if "location" in result.keys() else "N/A"
            customsAuth = result["customs_auth"] if "customs_auth" in result.keys() else "N/A"
            customsNumber = result["customs_number"] if "customs_number" in result.keys() else "N/A"
            vatRecovered = result["vat_recovered"] if "vat_recovered" in result.keys() else "N/A"
            vatAdjustments = result["vat_adjustments"] if "vat_adjustments" in result.keys() else "N/A"
            month = result["month"] if "month" in result.keys() else "other"
            other = result["other"] if "other" in result.keys() else "N/A"

        chunk = " | ".join(f"{k}: {v}" for k, v in row.items() if pd.notnull(v))

        chunks.append(chunk.lower())
        metadatas.append({
            "chunk": chunk,
            "source_file": filename,
            "format": "New",
            "prefix": "new_file",
            "company_trn": companyTrn.lower(),
            "company_name": companyName.lower(),
            "name": name.lower(),
            "voucher_no": voucherNo.lower(),
            "invoice_no": invoiceNo.lower(),
            "invoice_date": invoiceDate.lower(),
            "due_date": dueDate.lower(),
            "total_invoice_amount": totalInvoiceAmount.lower(),
            "vat_amount": vatAmount.lower(),
            "amount_paid": amountPaid.lower(),
            "amount_pending": amountPending.lower(),
            "days_due": str(daysDue).lower(),
            "date_received": dateReceived.lower(),
            "trn": trn.lower(),
            "location": location.lower(),
            "customs_auth": customsAuth.lower(),
            "customs_number": str(customsNumber).lower(),
            "vat_recovered": vatRecovered.lower(),
            "vat_adjustments": vatAdjustments.lower(),
            "month": month.lower() if month.lower() != "n/a" else "other",
            "other": other.lower()
        })

        
    
    return chunks, metadatas

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
            fields.remove(best_match)  # Remove matched field to avoid reusing
        else:
            unmapped.append(col)
            suggestions[col] = best_match  # optional

    return mapping, unmapped, suggestions, fields

def add_file_to_db(contents, fileName, fullMapping=None, batch_size=500):

    displayNames = {
                "name": "Customer Name",
                "voucher_no": "Voucher Number",
                "invoice_no": "Invoice Number",
                "invoice_date": "Invoice Date",
                "due_date": "Due Date",
                "total_invoice_amount": "Total Invoice Amount",
                "vat_amount": "VAT Amount",
                "amount_paid": "Amount Paid",
                "amount_pending": "Amount Pending",
                "days_due": "Days Due",
                "date_received": "Date Received",
                "trn": "TRN (Tax Registration Number)",
                "location": "Location",
                "customs_auth": "Customs Authority",
                "customs_number": "Customs Number",
                "vat_recovered": "VAT Recovered",
                "vat_adjustments": "VAT Adjustments",
                "month": "Invoice Month",
                "other": "Other / Unclassified",
              }
    
    db_fields = list(displayNames.keys())

    file_ext = fileName.lower().split(".")[-1]

    if file_ext in ['csv', 'xls', 'xlsx']:
        df = pd.read_csv(BytesIO(contents), encoding='utf-8', on_bad_lines='skip') if file_ext == 'csv' else pd.read_excel(BytesIO(contents), engine='openpyxl')
    
    elif file_ext == 'pdf':
        print("Processing PDF file")
        text = extract_text_from_pdf(contents)
        if not text.strip():
            print("Fallback to OCR for scanned PDF")
            text = extract_text_from_scanned_pdf(contents)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        print("Extracted text from PDF:", lines)  # Print first 1000 characters for debugging

        # for line in lines:
        #     extracted = extract_structured_row(line)
        #     if extracted:
        #         structured_data.append(extracted)

        extracted = extract_structured_row(lines, db_fields, fileType="PDF")
        

        if not extracted:
            raise ValueError("Could not extract any structured rows from text")

        print("Structured data extracted from PDF:", extracted)  # Print for debugging

        df = pd.DataFrame(extracted)

    elif file_ext in ['jpg', 'jpeg', 'png']:
        print("Processing image file")
        text = extract_text_from_image(contents)
        print("Extracted text from image:", text)  # Print first 1000 characters for debugging
        lines = [line.strip() for line in text.split('\n') if line.strip()]


        # for line in lines:
        #     extracted = extract_structured_row(line)
        #     if extracted:
        #         structured_data.append(extracted)
        extracted = extract_structured_row(lines, db_fields, fileType="image")

        if not extracted:
            raise ValueError("Could not extract any structured rows from text")

        df = pd.DataFrame(extracted)

    else:
        raise ValueError("Unsupported file type")
    
    df_columns = df.columns.tolist()

    
    print(df_columns)
    print("Available DB fields: ", list(displayNames.keys()))


    # "source_file", "format", "prefix", "company_trn", "company_name"
    # "month", "raw"

    model = SentenceTransformer("intfloat/e5-small-v2")

    # mapping = map_columns(df_columns, db_fields)
    if not fullMapping or len(fullMapping) != len(df_columns):
        mapping, unmapped, suggestions, fieldsLeft = fuzzy_map_columns(df_columns, db_fields) 
        print("mapped: ", mapping, "\n unmapped: ", unmapped, "\n fields: ", fieldsLeft, "\n suggestions: ", suggestions)


        if len(unmapped) > 0:
            # Handle unmapped columns (e.g., log them, map them to "raw", etc.)
            print("Unmapped columns found:")
            for col in unmapped:
                print(f" - {col} -> raw")
            return {
                "status": "partial_mapping",
                "unmapped": unmapped,
                "mapping": mapping,
                "suggestions": suggestions,
                "fields_left": fieldsLeft,
                "contents": df,
                "fieldDisplayNames": displayNames
            }
        else:
            fullMapping = mapping


    chunks, metas = new_df_to_chunks(df=df, filename=fileName, fullMapping=fullMapping)
    collection_name = fileName.replace(".", "_")

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
            "other": [meta["other"] for meta in metas],
            "raw": [meta["chunk"] for meta in metas],
            "id": [f"chunk_{i}" for i in range(len(chunks))]
        })

    client = chromadb.Client()
    collection = client.get_or_create_collection(name=collection_name)

    documents = df["chunk"].astype(str).tolist()
    embeddings = df["embedding"].tolist()
    metadatas = df[["source_file", "format", "prefix", "company_trn", "company_name", "name", "voucher_no", "invoice_no", "invoice_date", "due_date", "total_invoice_amount", "vat_amount", "amount_paid", "amount_pending", "days_due", "date_received", "trn", "location", "customs_auth", "customs_number", "vat_recovered", "vat_adjustments", "month", "raw", "other"]].to_dict(orient="records")
    ids = df["id"].tolist()

    # def batch_upsert():
    #     for i in range(0, len(documents), batch_size):

    #         batch_metadatas = []
    #         for meta in metadatas[i:i+batch_size]:
    #             clean_meta = {str(k): (v if isinstance(v, (str, int, float, bool)) or v is None else str(v)) for k, v in meta.items()}
    #             batch_metadatas.append(clean_meta)

    #         collection.upsert(
    #             documents=documents[i:i+batch_size],
    #             embeddings=embeddings[i:i+batch_size],
    #             metadatas=batch_metadatas,
    #             ids=ids[i:i+batch_size]
    #         )

    def batch_add():
        for i in range(0, len(documents), batch_size):
            # Ensure each metadata dict has only str keys and values of allowed types
            batch_metadatas = []
            for meta in metadatas[i:i+batch_size]:
                clean_meta = {str(k): (v if isinstance(v, (str, int, float, bool)) or v is None else str(v)) for k, v in meta.items()}
                batch_metadatas.append(clean_meta)

            collection.add(
                documents=documents[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                metadatas=batch_metadatas,
                ids=ids[i:i+batch_size]
            )


    batch_add()

    
    print("Added New file to DB")
    return {"status": "success", "message": f"{fileName} added to database.", "mapping": fullMapping}

def query_new_collection(query_text, file_name, mode):
    client = chromadb.Client()
    collection = client.get_collection(name=file_name.replace(".", "_"))
    if mode == "file_only":
        top_k = 5
    else:
        top_k = 2

    jsonQuery, newQuery, parsed, searchQuery = "", "", "", ""

    jsonQuery, newQuery = returnNewQuery(query=query_text)
    print("JSON QUERY:", jsonQuery)

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

    results, raw = query_chroma(collection, searchQuery.lower() if searchQuery != "" else query_text.lower(), original_query=query_text.lower(), n_results=top_k, json_query=parsed, format="New") # searchQuery.lower() if searchQuery != "" else newQuery

    #Can add ability to do arithmetic operations here if needed by adding code from retrieve_relevant_chunks

    context_text = []

    for item, text in zip(results["metadatas"], raw):
        if isinstance(item, dict) and text == item.get("raw", ""):
            context_text.append(item.get("raw", ""))
        elif isinstance(item, list):
            for subitem in item:
                if isinstance(subitem, dict) and text == subitem.get("raw", ""):
                    context_text.append(subitem.get("raw", ""))

    metadatas = results["metadatas"]

    print("Results from new file: ", context_text)


    return context_text, metadatas