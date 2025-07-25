import os
import time
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from getResults import *
from newFileResults import *

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

app.state.userAddedFile = False
app.state.fileName = ""

@app.post('/query')
async def returnResults(q: Query):

    query = q.query

    startTime = time.time()

    if app.state.userAddedFile:
        answer = query_new_collection(query_text=query, file_name=app.state.fileName)
    else:
        originalText, answer, prompt, topChunks, formatted_meta, topIds = getResult(query)

    endTime = time.time()
    totalTime = endTime - startTime

    print(f"Time elapsed: {totalTime}")
    # print(f"Raw: {originalText}")

    return {
        "answer": answer,
        # "prompt": prompt,
        # "chunks": topChunks,
        # "metadata": formatted_meta,
        # "ids": topIds
    }


@app.post('/upload')
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    result = add_file_to_db(contents, file.filename)
    app.state.userAddedFile = True
    app.state.fileName = file.filename

    if result.get("status") == "partial_mapping":
        print("Partial mapping occurred:")
        return {
            "status": "partial_mapping",
            "unmapped": result["unmapped"]
        }

    return {"filename": app.state.fileName, "contents": contents.decode("utf-8"), "message": result["message"]}

