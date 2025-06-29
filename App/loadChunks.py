import os
import pandas as pd
from pathlib import Path
import json
import re


DATA_DIR = Path(__file__).resolve().parent.parent / "Data"

def load_format_a(df):

    df = df.dropna(how='all').dropna(axis=1, how='all')
    df.columns = df.columns.astype(str).str.strip()
    df = df[~df.iloc[:, 0].astype(str).str.lower().str.contains("total|supplies|difference", na=False)]
    return df

def load_format_b(df):

    df = df.dropna(how='all').dropna(axis=1, how='all')
    df.columns = df.columns.astype(str).str.strip()
    return df

def detect_format(df):

    cols = df.columns.str.lower().tolist()

    if "supplier name" in cols and "voucher no" in cols:
        return "SAR"


    if 'customer name' in cols: #1, 4, OOS
        if 'vat amount aed' not in cols:
            return 'Box 4'
        elif 'reason of out-of-scope sales treatment' in cols:
            return 'OOS'
        else:
            return 'Box 1'
    elif 'supplier name' in cols: #3, 6, 7, 9
        if 'supplier trn' in cols:
            return 'Box 9'
        elif "customs declaration number" not in cols:
            return "Box 3"
        else:
            return "Box 6/7"
    else:
        return "unknown"

def load_all_data():

    files = os.listdir(DATA_DIR)
   
    vat_dfs = []
    aging_dfs = []

    for file in files:
        if not file.endswith(".csv"):
            continue

        path = DATA_DIR / file
        df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
        format = detect_format(df)

        if format == "SAR":
            df = load_format_b(df)
            aging_dfs.append((df, format))
        elif format == "unknown":
            print(f"Skipping unknown format: {file}")
        else:
            df = load_format_a(df)
            vat_dfs.append((file, df, format))

    # aging_dfs = pd.concat(aging_dfs, ignore_index=True) if aging_dfs else pd.DataFrame()

    return vat_dfs, aging_dfs


def row_to_chunk_json_with_invoice(row):
    data = {
        k.strip(): v
        for k, v in row.items()
        if pd.notnull(v) and not str(v).strip().lower() in ["-", "nan", "none"]
    }


    invoice_no_key = next(
        (key for key in data if "invoice" in key.lower() and "no" in key.lower()), 
        None
    )

    invoice_no = str(data[invoice_no_key]).strip().lower() if invoice_no_key else None

    return invoice_no


def clean_vals(val):
    valStr = str(val)
    if re.match(r'^[\d,\.]+$', valStr):
        return valStr.replace(",", "")
    return re.sub(r'(?<=\d),(?=\d)', '', valStr)

def row_to_chunk(row):
    return " and ".join(f"{col.strip()} is {clean_vals(val).strip()}" for col, val in row.items() if pd.notnull(val) and not str(val).strip().lower() in ["-", "nan", "none"])

def df_to_chunks(df, prefix=None, filename=None, format=None):
    chunks = []
    metadatas = []
    for _, row in df.iterrows():
        chunk = row_to_chunk(row=row)
        chunk = chunk.replace("\n", " ").replace("  ", " ")
        invoiceNo = row_to_chunk_json_with_invoice(row=row)
        invoiceNo = invoiceNo.lower() if invoiceNo else None
        if prefix:
            chunk = f"{prefix} | {chunk}"
        chunks.append(chunk.lower())
        metadatas.append({
            "chunk": chunk,
            "source_file": filename or prefix or "unknown",
            "prefix": prefix if prefix else None,
            "invoice_no": invoiceNo,
            'format': format if format else None
        })

    return chunks, metadatas


vatFiles, agingFiles = load_all_data()


def create_chunks():
    allChunks = []
    allMetadatas = []
    for fname, df, format in vatFiles:
        chunks, metas = df_to_chunks(df, filename=fname, format=format)
        allChunks.extend(chunks)
        allMetadatas.extend(metas)

    for df, format in agingFiles:
        agingChunks, agingMetas = df_to_chunks(df, prefix="Aging Report", format=format)
        allChunks.extend(agingChunks)
        allMetadatas.extend(agingMetas)

    return allChunks, allMetadatas



if __name__ == "__main__":
    create_chunks()