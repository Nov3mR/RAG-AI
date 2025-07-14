colNames = """
    Supplier Aging Report: Supplier Name, Voucher No, Invoice No, Invoice Date, Due Date, Total Invoice Amount (AED), VAT Amount (AED), Amount Paid (AED), Pending Amount (AED), Total days due from due date as on 16 May 2025
    VAT Files: Customer Name, Supplier Name, Tax Invoice/Tax credit note No, Tax Invoice/credit note Date, Tax Invoice/Credit Note Received Date, Tax Invoice/Credit Note Amount AED (before VAT), Clear description of the supply, Clear description of the transaction, Customer TRN, Supplier TRN, Location of the Supplier, Location of the Customer, Customs Declaration Number, VAT Amount Recovered AED, VAT adjustments, Reason of Out-of-Scope Sales treatment, Month
    """


arithmeticColNames = """
    If the query is about Supplier Aging Report, use these column names:
    'name', 'voucher_no', 'invoice_no', 'total_invoice_amount', 'vat_amount', 'amount_paid', 'amount_pending', 'days_due'
    
    If Supplier Aging Report is not mentioned, use these column names:
    'name', 'voucher_no', 'invoice_no', 'total_invoice_amount', 'vat_amount', 'trn', 'location', 'customs_auth', 'customs_number', 'vat_recovered', 'vat_adjustments', 'month'

    DO NOT INCLUDE ANY OTHER COLUMN NAMES

    """


def getJsonPrompt(query: str) -> dict:
   
    jsonPrompt = f"""
    Extract structured info from the following user query:
    
    Query: "{query}"

    Respond in JSON format with keys mentioned below. Only return keys that are present in the user query.

    KEYS: 
    {colNames}

    DO NOT INCLUDE KEYS THAT ARE NOT PRESENT IN THE USER QUERY

    DO NOT INCLUDE ANY EXPLANATION OR ADDITIONAL TEXT

    Make sure you return the keys as they are

    Convert any dates into the form MM/D/2024

    AGAIN DO NOT INCLUDE KEYS THAT ARE NOT PRESENT IN THE USER QUERY 

    This MUST be the format
    DO NOT ADD ANY PLAINTEXT

    {{
    "<key_name_1>": "<value_1>",
    "<key_name_2>": "<value_2>"
    }}

    DO NOT ADD ANY EXPLANATION OR EXTRA TEXT
    """

    return jsonPrompt.strip()


def getNewPrompt(query: str) -> str:

    # newPrompt = f"""
    # You are a system that rewrites natural language queries into a concise and searchable format. This query is about data from te year 2024

    # Instructions:
    # - Rewrite the query using relevant keywords such as "invoice number", "vat amount", "supplier name", "customer name", etc.
    # - Keep the query short, clear, and aligned with how invoice data is stored.
    # - Remove all unnecessary filler words.
    # - Return only the rewritten query as plain text. No JSON, no explanations.

    # Examples:

    # User: What year was invoice number 300545025110003 generated?  
    # Response: invoice number 300545025110003 generation year

    # User: How much VAT did we pay to ABC LTD in May?  
    # Response: vat paid to abc ltd may

    # User: Which invoices from June have pending VAT?  
    # Response: pending vat invoices june

    # Now rewrite this query: {query}
    # """


    newPrompt = f"""
    You are a helpful assistant. Your task is to rewrite the following user query to make it better for searching a structured document database.

    Focus only on the essential searchable terms. Do not explain anything. Do not include reasoning. Do not include any JSON formatting.

    ❗Return only a plain string — no punctuation, no JSON, no explanation.

    For context, these are the column names of the database. IT IS NOT DATA
    {colNames}

    Examples:

    User: What year was the invoice number 300545025110003 generated?
    Rewritten: invoice number 300545025110003 invoice date year

    User: How much VAT did we pay to supplier ACER in May?
    Rewritten: vat amount supplier acer may

    User: Why was the invoice from EMPO on 9th September out-of-scope?
    Rewritten: invoice empo 9 september out-of-scope reason

    Now rewrite this:
    \"\"\"{query}\"\"\"

    Rewritten:
"""
    return newPrompt.strip()

def getArithmeticPrompt(query: str) -> dict:
    arithmeticPrompt = f"""

    You are an intelligent assistant helping identify if a user's query requires performing arithmetic operations (like sum, average, count, min, or max) on tabular data.

    Your job is to return a JSON object in the following format:

    {{
    "operation": "<operation>",      
    "field": "<field_name>",         
    "filters": {{                     
        "<column_name_1>": "<value_1>",
        "<column_name_2>": "<value_2>"
    }}
    }}

    Rules:
    - If no arithmetic is needed, set "operation": "none", "field": "none", and "filters": {{}}.
    - If arithmetic applies to the whole column (no filters), set "filters": {{}}.
    - Always return only this JSON object — no explanations or surrounding text.

    These are the column names of the database you can possibly use for the filters if needed
    {arithmeticColNames}

    Assume query is about VAT sheets unless Supplier Aging Report is mentioned.
    Important: Output must be pure JSON. Do not include any comments, explanations, or natural language outside of the JSON structure.

    Examples

    User Query 1:  
    What is the total VAT amount for May from ABC LTD?

    Output:
    {{
    "operation": "sum",
    "field": "vat_amount",
    "filters": {{
        "company_name": "ABC LTD",
        "month": "May"
    }}
    }}

    User Query 2:  
    How much pending amount do we have overall?

    Output:
    {{
    "operation": "sum",
    "field": "amount_pending",
    "filters": {{}}
    }}

    User Query 3:  
    List all invoices from PETR.

    Output:
    {{
    "operation": "none",
    "field": "none",
    "filters": {{}}
    }}

    Output a JSON object with the following format:
    {{
    "operation": "<sum | count | average | etc.>",
    "field": "<name of the field to apply the operation on>",
    "filters": {{
        "<column_name>": "<filter condition>"  // e.g., "amount_pending": "0"
    }}  
    }}

    Important rules:
    - All filter conditions must be separate key-value pairs inside the "filters" dictionary.
    - Do not write expressions like "Pending Amount (AED) > 0" as keys.
    - Keys must be exact field names from the dataset.
    - The JSON must be clean, valid, and directly usable with Python's json.loads().
    - Do not include comments, extra explanations, or natural language outside the JSON.
    - Integer values cannot be negative.

    THE PURPOSE OF THE FILTERS IS TO DECREASE THE RESULT SET TO MAKE IT MORE ACCURATE AND TO DO ARITHMETIC ON
    DO NOT RETURN A COLUMN NAME IN FILTERS IF THERE IS NO CONDITION OR IF IT IS NOT SPECIFIED IN THE QUERY.

    


    Now analyze this user query and return only the JSON:
    \"\"\"{query}\"\"\"
    """

    return arithmeticPrompt.strip()
