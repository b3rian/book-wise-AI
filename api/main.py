import os
import logging
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv 
import chromadb
from api.query import rag_query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

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

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class QueryRequest(BaseModel):
    prompt: str
    n_results: Optional[int] = 3

class QueryResponse(BaseModel):
    answer: str

# ---------- Startup ----------
@app.on_event("startup")
def startup_event():
    global chroma_client, chroma_collection
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = chroma_client.get_collection(name=COLLECTION_NAME)
    logger.info("ChromaDB client initialized successfully.")

# ---------- Endpoints ----------
@app.post("/prompt", response_model=QueryResponse)
async def rag_endpoint(request: QueryRequest):
    try:
        answer = rag_query(
            request.prompt,
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_NAME,
            n_results=request.n_results
        )
        return QueryResponse(answer=answer)
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Error processing RAG query")
        raise HTTPException(status_code=500, detail="Internal server error")