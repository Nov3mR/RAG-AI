colNames = """
    Supplier Aging Report: Supplier Name, Voucher No, Invoice No, Invoice Date, Due Date, Total Invoice Amount (AED), VAT Amount (AED), Amount Paid (AED), Pending Amount (AED), Total days due from due date as on 16 May 2025
    VAT Files: Customer Name, Supplier Name, Tax Invoice/Tax credit note No, Tax Invoice/credit note Date, Tax Invoice/Credit Note Received Date, Tax Invoice/Credit Note Amount AED (before VAT), Clear description of the supply, Clear description of the transaction, Customer TRN, Supplier TRN, Location of the Supplier, Location of the Customer, Customs Declaration Number, VAT Amount Recovered AED, VAT adjustments, Reason of Out-of-Scope Sales treatment, Month
    """


arithmeticColNames = """
    If the query is about Supplier Aging Report, use these column names:
    'name', 'voucher_no', 'invoice_no', 'total_invoice_amount', 'vat_amount', 'amount_paid', 'amount_pending', 'days_due'
    
    If Supplier Aging Report is not mentioned, use these column names:
    'name', 'invoice_no', 'total_invoice_amount', 'vat_amount', 'trn', 'location', 'customs_auth', 'customs_number', 'vat_recovered', 'vat_adjustments', 'month'

    DO NOT INCLUDE ANY OTHER COLUMN NAMES

    """


def getJsonPrompt(query: str) -> str:

    jsonPrompt = f"""
    You are a data extraction system.

    Your task is to extract only explicit field-value pairs from the query below.

    Only include a field in the JSON if its value is **clearly present in the user query**. 
    If the query only refers to a field (e.g. asks about it), do NOT include that key unless a value is explicitly mentioned.

    Query:
    "{query}"

    Return a JSON object where:
    - Keys must be from this list only: {colNames}
    - Values must be clearly stated in the query.


    - ONLY include keys if their values are explicitly present.
    - DO NOT include empty strings as values.
    - DO NOT add any explanation or commentary.
    - DO NOT infer or assume anything.
    - DO NOT guess missing values.

    INCLUDE Month as a key if the query mentions a specific month. If there are multiple months, return as a list of months.

    IF there are multiple values for a key, return them as a list.

    Examples of Months are: January, February, March, April, May, June, July, August, September, October, November, December

    If query is about the Supplier Aging Report, the month key should have the value "SAR"

    If the query has no explicit field-value pairs, return exactly:
    {{}}

    Examples:

    1. Query: "Show me the VAT amount for invoice 300545025110003"
    → Output:
    {{
    "Invoice No": "300545025110003"
    }}

    2. Query: "When is the due date for supplier EMPO?"
    → Output:
    {{
    "Supplier Name": "empo"
    }}

    3. Query: "How many invoices have not been paid?"
    → Output:
    {{}}

    4. Query: "Show me the VAT amount for invoices in June"
    → Output:
    {{
    "Month": "June"
    }}

    Return only the JSON. Do not add markdown or text around it.
    AGAIN: DO NOT include fields with blank or missing values. NO comments. NO explanations. ONLY the valid JSON.

    NO EXPLANATIONS IN YOUR RESPONSE

    """

    return jsonPrompt.strip()


def getNewPrompt(query: str, conversationHistory: str) -> str:
    prompt = f"""
You are a helpful assistant that rewrites user questions to include necessary context from the previous conversation, 
so that they can be better understood when retrieving information from a database.

Conversation history:
{conversationHistory}

User's current question:
{query}

Rewrite the user's current question to include relevant context from the conversation history in a concise way.
Return only the rewritten question.
"""
    return prompt.strip()

def getArithmeticPrompt(query: str) -> str:
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
        "name": "ABC LTD",
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
    - Return non numeric filter conditions for 1 column name as a list of conditions.
    - If condition is numeric or contains a comparison operator, DO NOT RETURN AS A LIST
    - There must only be one operation which must be a string
    - Always make it easier to filter the data. For example, fully paid would mean "amount_pending": "0" instead of "amount_paid": "full" and not fully paid would mean "amount_pending": ">0"
    - Possible comparison operators in filters: >, >=, <, <=
    - If filter has a comparison operator, IT MUST CONTAIN A NUMBER AFTER IT AND NOT LETTERS
    - DO NOT RETURN A COMPARISON OPERATOR WITHOUT A NUMBER
    - Make sure to know which operation is most relevant. For example, how many invoices would mean operation: "count", field: "invoice_no" but how many days due would mean operation: "sum", field: "days_due".
    - Do not add a filter name within the condition
    - Only use column names from the column names list provided above
    - Use the most accurate filters and field. For example, VAT would refer to field "vat_amount" but vat that was recovered would refer to field "vat_recovered".
    
    
    THE PURPOSE OF THE FILTERS IS TO DECREASE THE RESULT SET TO MAKE IT MORE ACCURATE AND TO DO ARITHMETIC ON
    DO NOT RETURN A COLUMN NAME IN FILTERS IF THERE IS NO CONDITION OR IF IT IS NOT SPECIFIED IN THE QUERY.

    


    Now analyze this user query and return only the JSON:
    \"\"\"{query}\"\"\"
    """

    return arithmeticPrompt.strip()
