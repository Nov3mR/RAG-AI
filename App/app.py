import os
import time
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
from getResults import *
from newFileResults import *
from getLLMAnswer import *

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

app.state.userAddedFile = False
app.state.fileName = ""
app.state.contents = b""

class Query(BaseModel):
    query: str
    fileMode: Literal["file_only", "db_only", "both"] = "both" if app.state.userAddedFile else "db_only"
    llmMode: Literal["openai", "ollama"] = "openai"

@app.post('/query')
async def returnResults(q: Query):

    query = q.query
    mode = q.fileMode
    llm = q.llmMode

    startTime = time.time()

    context, meta, arithmeticResult, arithmeticRow, newContext, newMeta = None, None, None, None, None, None

    if app.state.userAddedFile and mode in ["both", "file_only"]:
        newContext, newMeta = query_new_collection(query_text=query, file_name=app.state.fileName, mode=mode)

    if mode in ["db_only", "both"]:
        context, meta, arithmeticResult, arithmeticRow = getResult(query)

    answer = returnLLMAnswer(query=query, context=context if context else newContext, meta=meta if meta else newMeta, arithmeticResult=arithmeticResult, arithmeticRow=arithmeticRow, newContext=newContext if context else None, newMeta=newMeta if meta else None, llmMode=llm)

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
    app.state.contents = await file.read()
    result = add_file_to_db(app.state.contents, file.filename)
    app.state.userAddedFile = True
    app.state.fileName = file.filename

    if result.get("status") == "partial_mapping":
        print("Partial mapping occurred:")
        return {
            "status": "partial_mapping",
            "unmapped": result["unmapped"],
            "mapping": result["mapping"],
            "suggestions": result["suggestions"],
            "fields_left": result["fields_left"],
            "fieldDisplayNames": result.get("fieldDisplayNames", {}),
            "contents": app.state.contents,
            "filename": app.state.fileName
        }

    return {"filename": app.state.fileName, "contents": "Uploaded", "message": result["message"], "mapping": result.get("mapping", {})}

@app.post('/finalize-mapping')
async def finalize_mapping(payload: dict):
    # Here you would finalize the mapping in your database or processing pipeline
    print("Finalizing mapping:")
    print(payload["mapping"])
    result = add_file_to_db(app.state.contents, app.state.fileName, fullMapping=payload["mapping"])
    app.state.userAddedFile = True

    if result.get("status") == "success":
        print("Mapping finalized successfully.")
        return {"status": "success", "message": f"Mapping finalized. {app.state.fileName} added to database."}
    else:
        print("Error finalizing mapping.")
        return {
            "status": "partial_mapping",
            "message": "Error finalizing mapping.",
            "unmapped": result["unmapped"],
            "mapping": result["mapping"],
            "suggestions": result["suggestions"],
            "fields_left": result["fields_left"],
            "contents": app.state.contents,
            "filename": app.state.fileName
            }
