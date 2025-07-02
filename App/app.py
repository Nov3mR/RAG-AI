import os
import time
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from promptLLM import call_LLM
from buildQuery import returnQuery
from getResults import retrieve_relevant_chunks 

os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

model3 = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
model4 = SentenceTransformer("msmarco-distilbert-base-v4")
model5 = SentenceTransformer("BAAI/bge-base-en-v1.5")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

@app.post('/query')
async def returnResults(q: Query):

    
    query = q.query

    topChunks, topMetas, topIds, originalText = retrieve_relevant_chunks(query=query)

    formatted_meta = f"""
                    Our company name: {topMetas[0]['company_name']}
                    Our TRN: {topMetas[0]['trn']} """

    formatted_meta = "\n".join([f"""- Invoice Number: {m['invoice_no']}
        - Raw: {m['raw']}
        - Format: {m['format']}
        - Type of Transaction: {m['prefix']}
        """ for m in topMetas])

    print(formatted_meta)

    context = "\n\n".join(originalText)

    prompt = returnQuery(context=context, meta=formatted_meta, query=query)

    print("Calling LLM")
    startTime = time.time()
    answer = call_LLM(prompt=prompt)   

    endTime = time.time()
    totalTime = endTime - startTime

    print(f"Time elapsed: {totalTime}")
    print(f"Raw: {originalText}")

    return {
        "answer": answer,
        "prompt": prompt,
        "chunks": topChunks,
        "metadata": formatted_meta,
        "ids": topIds
    }


if __name__ == "__main__":
    chunks, metas, ids, originalText = retrieve_relevant_chunks("what is the location of customer bauh")
    meta = metas[0]
    print(meta['invoice_no'])