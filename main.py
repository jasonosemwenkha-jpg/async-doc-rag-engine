import os
import shutil
from dotenv import load_dotenv

load_dotenv()  # Now Python knows what load_dotenv is!

from fastapi import FastAPI, UploadFile, File, HTTPException
from app.ingestion import process_pdf
from app.rag import answer_query


app = FastAPI(title="RAG Engine API")

@app.get("/")
def read_root():
    return {"message": "RAG Engine API is running!"}

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    os.makedirs("./data", exist_ok=True)
    temp_path = f"./data/{file.filename}"
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        chunk_count = process_pdf(temp_path)
        return {"status": "success", "chunks_processed": chunk_count}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/query")
def query_rag(question: str):
    try:
        answer = answer_query(question)
        return {"question": question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))