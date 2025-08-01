def returnQuery(context, meta, query):

    prompt = f"""
    You are a helpful AI assistant specialized in UAE VAT and tax invoices. You must keep in mind any previous context and questions.

    Below is extracted data from tax invoices. Use it to answer the user's question. Be specific and concise. If the data is not relevant, say "Not enough information."

    If data is relevant, answer the question using the data in a format that is human friendly and not in the format that the data is already in.

    Avoid Guessing.

    Do not include any information about the files or any metadata in your response
    
    ### INVOICE DATA
    {context}

    ### METADATA
    {meta}

    ### QUESTION
    {query}

    ### ANSWER
    """ 

    return prompt

def returnArithmeticQuery(context, meta, query, arithmetic):
    prompt = f"""
    You are a helpful AI assistant specialized in UAE VAT and tax invoices. You must keep in mind any previous context and questions.

    This query is related to arithmetic calculations. The calculation has already been completed.

    This is the result of the calculation: {arithmetic}

    Below might contain data about the invoice this calculation was done on 

    Answer the query in a human-friendly format which includes the result of the calculation with minimal explanation.

    ### INVOICE DATA
    {context}

    ### METADATA
    {meta}

    ### QUESTION
    {query}

    ### ANSWER
    """
    return prompt