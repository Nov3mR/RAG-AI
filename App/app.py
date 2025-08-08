import os
import time
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
import uuid
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

fileStore = {}
conversationHistory = []

def format_history(history, n=6):
    formatted = ""
    for item in history[-n:]:
        role = item.get("role", "unknown")
        msg = item.get("content", "")
        formatted += f"{role}: {msg.strip()}\n"
    return formatted

def needs_rag(query: str) -> bool:
    prompt = f"Does the following user query need retrieval-augmented generation (i.e., access to invoice data)? Reply with 'yes' or 'no'.\n\nQuery: {query}"
    response = call_LLM(prompt, mode="openai").lower()
    return "yes" in response


class Query(BaseModel):
    query: str
    fileMode: Literal["file_only", "db_only", "both"] = "both" if app.state.userAddedFile else "db_only"
    llmMode: Literal["openai", "ollama"] = "openai"

@app.post('/query')
async def returnResults(q: Query):
    global fileStore, conversationHistory

    query = q.query
    mode = q.fileMode
    llm = q.llmMode


    startTime = time.time()
    conversationHistory.append({"role": "user", "content": query})

    context, meta, arithmeticResult, arithmeticRow, newContext, newMeta = None, None, None, None, None, None

    formattedConversationHistory = format_history(conversationHistory, n=6)

    history_context = formattedConversationHistory if formattedConversationHistory else ""
    if history_context:
        history_context += "\n\n"  

    print("History Context:")
    print(history_context)

    if needs_rag(query):
        if app.state.userAddedFile and mode in ["both", "file_only"]:
            file_names = [entry["fileName"] for entry in fileStore.values()]
            file_ids = list(fileStore.keys())
            newContext, newMeta = query_new_collection(query_text=query, file_names=file_names, mode=mode, file_ids=file_ids, conversationHistory=history_context)

        if mode in ["db_only", "both"]:
            context, meta, arithmeticResult, arithmeticRow = getResult(query, conversationHistory=history_context)

        answer = returnLLMAnswer(query=query, context=context if context else newContext, meta=meta if meta else newMeta, arithmeticResult=arithmeticResult, arithmeticRow=arithmeticRow, newContext=newContext if context else None, newMeta=newMeta if meta else None, llmMode=llm, history_context=history_context)
    else:
        print("No need for RAG, using LLM directly.")
        answer = returnLLMAnswer(query, llmMode=llm, history_context=history_context)

    conversationHistory.append({"role": "assistant", "content": answer})


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
    global fileStore

    contents = await file.read()

    fileId = str(uuid.uuid4())
    fileStore[fileId] = {
        "fileName": file.filename,
        "contents": contents
    }


    result = add_file_to_db(contents, file.filename)

    print(f"File {fileStore[fileId]['fileName']} uploaded with ID: {fileId}")
    print("Result of adding file to DB:", fileStore)
    app.state.userAddedFile = True

    if result.get("status") == "partial_mapping":
        print("Partial mapping occurred:")
        return {
            "status": "partial_mapping",
            "unmapped": result["unmapped"],
            "mapping": result["mapping"],
            "suggestions": result["suggestions"],
            "fields_left": result["fields_left"],
            "fieldDisplayNames": result.get("fieldDisplayNames", {}),
            "contents": contents,
            "fileId": fileId
        }

    return {
        "fileId": list(fileStore.keys()),
        "fileName": file.filename,
        "message": result["message"],
        "mapping": result.get("mapping", {})
    }

@app.post('/finalize-mapping')
async def finalize_mapping(payload: dict):
    global fileStore
    # Here you would finalize the mapping in your database or processing pipeline
    print("Finalizing mapping:")
    print(payload["mapping"])
    fileId = payload.get("fileId", None)
    if not fileId or fileId not in fileStore:
        return {"status": "error", "message": "Invalid fileId or file not found."}
    else:
        result = add_file_to_db(fileStore[fileId]["contents"], fileStore[fileId]["fileName"], fullMapping=payload["mapping"])

    if result.get("status") == "success":
        print("Mapping finalized successfully.")
        print(f"File {fileStore[fileId]['fileName']} uploaded with ID: {fileId}")
        print("Result of adding file to DB:", fileStore)
        return {"status": "success", "fileName": fileStore[fileId]['fileName'], "message": f"Mapping finalized. {fileStore[fileId]['fileName']} added to database.", "fileIds": list(fileStore.keys())}
    else:
        print("Error finalizing mapping.")
        return {
            "status": "partial_mapping",
            "message": "Error finalizing mapping.",
            "unmapped": result["unmapped"],
            "mapping": result["mapping"],
            "suggestions": result["suggestions"],
            "fields_left": result["fields_left"],
            "contents": fileStore[fileId]["contents"],
            "filename": fileStore[fileId]["fileName"]
            }
