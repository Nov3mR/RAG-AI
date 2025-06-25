import os
import pandas as pd
from pathlib import Path


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
    if "transaction type" in cols and 'taxpayer trn' in cols:
        return "VAT File"
    elif "supplier name" in cols and "voucher no" in cols:
        return "Supplier Aging Report"
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

        if format == "VAT File":
            df = load_format_a(df)
            vat_dfs.append((file, df))
        elif format == "Supplier Aging Report":
            df = load_format_b(df)
            aging_dfs.append(df)
        else:
            print(f"Skipping unknown format: {file}")

    aging_dfs = pd.concat(aging_dfs, ignore_index=True) if aging_dfs else pd.DataFrame()

    return vat_dfs, aging_dfs


def row_to_chunk(row):
    return " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notnull(val))

def df_to_chunks(df, prefix=None, filename=None):
    chunks = []
    metadatas = []
    for _, row in df.iterrows():
        chunk = " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notnull(val))
        if prefix:
            chunk = f"{prefix} | {chunk}"
        chunks.append(chunk)
        metadatas.append(({"source_file": filename or prefix or "unknown"}))


    return chunks, metadatas


vatFiles, agingFiles = load_all_data()

def create_chunks():
    allChunks = []
    allMetadatas = []
    for fname, df in vatFiles:
        chunks, metas = df_to_chunks(df, filename=fname)
        allChunks.extend(chunks)
        allMetadatas.extend(metas)

    agingChunks, agingMetas = df_to_chunks(agingFiles, prefix="Aging Report")
    
    allChunks.extend(agingChunks)
    allMetadatas.extend(agingMetas)

    return allChunks, allMetadatas



if __name__ == "__main__":
    print(create_chunks())