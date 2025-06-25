def returnQuery(context, meta, query):

    prompt = f"""
    You are a helpful AI assistant specialized in UAE VAT and tax invoices.

    Below is extracted data from tax invoices. Use it to answer the user's question. Be specific and concise. If the data is not relevant, say "Not enough information."

    If data is relevant, answer the question using the data in a format that is human friendly and not in the format that the data is already in.

    Avoid Guessing.
    
    ### INVOICE DATA
    {context}

    ### METADATA
    {meta}

    ### QUESTION
    {query}

    ### ANSWER
    """ 

    return prompt