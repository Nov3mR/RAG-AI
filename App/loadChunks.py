import os
import pandas as pd
from pathlib import Path
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
    vat_dfs = []
    aging_dfs = []

    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if not file.endswith(".csv"):
                continue

            path = Path(root) / file
            df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')

            format = detect_format(df)

            if format == "SAR":
                df = load_format_b(df)
                aging_dfs.append((df, format))
            elif format == "unknown":
                print(f"Skipping unknown format: {file}")
            else:
                month = Path(path).parent.name
                # add if condition for specific months
                df = load_format_a(df)
                vat_dfs.append((file, df, format, month))   
               
    return vat_dfs, aging_dfs

# def row_to_chunk_json_with_invoice(row):
#     data = {
#         k.strip(): v
#         for k, v in row.items()
#         if pd.notnull(v) and not str(v).strip().lower() in ["-", "nan", "none"]
#     }


#     invoice_no_key = next(
#         (key for key in data if "invoice" in key.lower() and "no" in key.lower()), 
#         None
#     )

#     invoice_no = str(data[invoice_no_key]).strip().lower() if invoice_no_key else None

#     trn_key = next(
#         (key for key in data if "supplier trn" in key.lower() or "customer trn" in key.lower()), 
#         None
#     )

#     if trn_key and trn_key in data:
#         val = data[trn_key]
#         try:
#             # Try to convert to float then int to remove trailing .0 if numeric
#             trn = str(int(float(val)))
#         except (ValueError, TypeError):
           
#             val_str = str(val).strip().lower()
#             if val_str.endswith(".0"):
#                 val_str = val_str[:-2]
#             trn = val_str
#     else:
#         trn = "N/A"

#     # N/A for SAR, 3, 6, 7

#     return invoice_no, trn

def clean_vals(val):
    valStr = str(val)
    if re.match(r'^[\d,\.]+$', valStr):
        return valStr.replace(",", "")
    return re.sub(r'(?<=\d),(?=\d)', '', valStr)

def row_to_chunk(row):

    row_items = list(row.items())

    if clean_vals(row_items[0][0]).strip() == "Transaction Type":
        prefix = clean_vals(row_items[0][1]).strip() if pd.notnull(row_items[0][1]) else ""
        trn = clean_vals(row_items[1][1]).strip() if pd.notnull(row_items[1][1]) else ""
        company_name = clean_vals(row_items[2][1]).strip() if pd.notnull(row_items[2][1]) else ""
        chunk = " | ".join(
            f"{col.strip()}: {clean_vals(val).strip()}"
            for col, val in row_items[3:]
            if pd.notnull(val) and str(val).strip().lower() not in ["-", "nan", "none"]
        )

        chunk = chunk.replace("Tax Invoice/Tax credit note No", "Invoice No")
    else:
        prefix = None
        trn = None
        company_name = None
        chunk = " | ".join(
            f"{col.strip()}: {clean_vals(val).strip()}"
            for col, val in row_items[1:]
            if pd.notnull(val) and str(val).strip().lower() not in ["-", "nan", "none"]
        )

    return chunk, prefix, trn, company_name


def extract_metadata_from_row(row, format):

    aliases = {
    "name": ["Supplier Name", "Customer Name"],
    "voucherNo": ["Voucher No"],
    "invoiceNo": ["Invoice No", "Tax Invoice/Tax credit note No"],
    "invoiceDate": ["Invoice Date", "Tax Invoice/Tax credit note Date"],
    "dueDate": ["Due Date"],
    "totalInvoiceAmount": ["Total Invoice Amount (AED)", "Tax Invoice/Tax credit note Amount AED (before VAT)"],
    "vatAmount": ["VAT Amount (AED)", "VAT Amount AED"],
    "amountPaid": ["Amount Paid (AED)"],
    "amountPending": ["Pending Amount (AED)"],
    "daysDue": ["Total days due from due date as on 16 May 2025"],
    "dateReceived": ["Tax Invoice/Tax credit note Received Date"],
    "trn": ["Supplier TRN", "Customer TRN"],
    "location": ["Location of the Supplier", "Location of the Customer"],
    "customsAuth": ["Name of the Customs Authority"],
    "customsNumber": ["Customs Declaration Number"],
    "vatRecovered": ["VAT Amount Recovered AED"],
    "vatAdjustments": ["VAT adjustments"]
    }

    result = {}
    for var, possible_cols in aliases.items():
        value = "N/A"
        for col in possible_cols:
            if col in row and pd.notnull(row[col]):
                value = str(row[col]).strip()
                if value == "-":
                    value = "0"
                else:
                    value = re.sub(r'(?<=\d),(?=\d)', '', value)
                break
        result[var] = value

    return result

def df_to_chunks(df, prefix=None, filename=None, format=None, month=None):
    chunks = []
    metadatas = []
    #add date, amount, vat amount, name, location, received date/vatRecovered/adjustments (Box 9), customsAuth/Number (6/7) 
    #for later, description, OOS

    """
    Supplier Aging Report: Supplier Name, Voucher No, Invoice No, Invoice Date, Due Date, Total Invoice Amount (AED), VAT Amount (AED), Amount Paid (AED), Pending Amount (AED), Upto 30 Days Due, 30-60 Days Due, 61-90 Days Due, More than 90 Days Due, Pending >180 Days, Total days due from due date as on 16 May 2025
    VAT Sheets: Transaction Type, Taxpayer TRN, Company Name, Tax Invoice/Credit Note Date, Tax Invoice/Credit Note Received Date, Clear description of the supply, Clear description of the transaction, Customer Name, Supplier Name, Customer TRN, Supplier TRN, Location of the Supplier, Location of the Customer, Customs Declaration Number, VAT Amount Recovered AED, VAT adjustments, Reason of Out-of-Scope Sales treatment, Month

    """
    #ADD DESCRIPTION AND OOS LATER

    for _, row in df.iterrows():

        values = extract_metadata_from_row(row=row, format=format)

        name = values["name"]
        voucherNo = values["voucherNo"]
        invoiceNo = values["invoiceNo"]
        invoiceDate = values["invoiceDate"]
        dueDate = values["dueDate"]
        totalInvoiceAmount = values["totalInvoiceAmount"]
        vatAmount = values["vatAmount"]
        amountPaid = values["amountPaid"]
        amountPending = values["amountPending"]
        daysDue = values["daysDue"]
        dateReceived = values["dateReceived"]
        trn = values["trn"]
        location = values["location"]
        customsAuth = values["customsAuth"]
        customsNumber = values["customsNumber"]
        vatRecovered = values["vatRecovered"]
        vatAdjustments = values["vatAdjustments"]

        # TRN normalization

        if trn != "N/A":
            try:
                trn = str(int(float(trn)))
            except (ValueError, TypeError):
                val_str = str(trn).strip().lower()
                if val_str.endswith(".0"):
                    val_str = val_str[:-2]
                trn = val_str

        # VAT Adjustment normalization

       

        chunk, tType, company_trn, company_name = row_to_chunk(row=row)
        chunk = chunk.replace("\n", " ").replace("  ", " ")
        # invoiceNo, trn = row_to_chunk_json_with_invoice(row=row)
        # invoiceNo = invoiceNo.lower() if invoiceNo else None
        chunks.append(chunk.lower())
        metadatas.append({
            "chunk": chunk,
            "source_file": filename or prefix or "unknown",
            "format": format if format else "unknown",
            "prefix": prefix.lower() if prefix else tType.lower(),
            "company_trn": company_trn.lower() if company_trn else "Aging Report - No TRN",
            "company_name": company_name.lower() if company_name else "Aging Report - No Company Name",
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
            "month": month.lower() if month else "sar"
        })

    # if format == "SAR":
    #     for meta in metadatas:
    #         print(meta["amount_pending"])

    return chunks, metadatas

def create_chunks():
    vatFiles, agingFiles = load_all_data()  
    allChunks = []
    allMetadatas = []
    for fname, df, format, month in vatFiles:
        chunks, metas = df_to_chunks(df, filename=fname, format=format, month=month)
        allChunks.extend(chunks)
        allMetadatas.extend(metas)

    for df, format in agingFiles:
        agingChunks, agingMetas = df_to_chunks(df, prefix="Aging Report", format=format)
        allChunks.extend(agingChunks)
        allMetadatas.extend(agingMetas)

    return allChunks, allMetadatas


if __name__ == "__main__":
    create_chunks()