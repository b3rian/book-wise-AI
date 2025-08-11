import os
import logging
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv 
import chromadb
from api.query import rag_query

# Load environment variables
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PERSIST_DIR = os.getenv("PERSIST_DIR", r"D:\Documents\chromadb\nietzsche_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "nietzsche_books")

# ---------- FastAPI App ----------
app = FastAPI(
    title="Nietzsche RAG API",
    description="Retrieve and answer questions based on Nietzsche's works.",
    version="1.0.0",
)

# ---------- Models ----------
class QueryRequest(BaseModel):
    question: str
    n_results: Optional[int] = 3

class QueryResponse(BaseModel):
    answer: str

# ---------- Startup ----------
@app.on_event("startup")
def startup_event():
    global chroma_client, chroma_collection
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = client.get_collection(name=COLLECTION_NAME)
    logger.info("ChromaDB client initialized successfully.")

# ---------- Endpoints ----------
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    answer = rag_query(req.question, req.n_results)
    return QueryResponse(answer=answer)