import os
import logging
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq, GroqError
import chromadb
from chromadb.config import Settings
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import json
import asyncio

# Load environment variables from .env
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Config ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PERSIST_DIR = os.getenv("PERSIST_DIR", r"D:\Documents\chromadb\nietzsche_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "nietzsche_books")
MODEL_NAME = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# ---------- FastAPI App ----------
app = FastAPI(
    title="Nietzsche RAG API",
    description="Retrieve and answer questions based on Nietzsche's works.",
    version="1.0.0",
)

# ---------- Models ----------
class QueryRequest(BaseModel):
    prompt: str
    n_results: Optional[int] = 3

class QueryResponse(BaseModel):
    answer: str

class StreamingResponseModel(BaseModel):
    chunk: str

def query_chromadb(persist_directory, collection_name, query_text, n_results=3):
    """
    Query a ChromaDB persisted database.

    Args:
        persist_directory (str): Path to the ChromaDB storage folder.
        collection_name (str): Name of the collection to query.
        query_text (str): The text you want to search for.
        n_results (int): Number of top results to return.

    Returns:
        dict: Query results containing IDs, documents, and metadata.
    """
    # Load the persisted ChromaDB client
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    # Get the collection
    collection = client.get_collection(name=COLLECTION_NAME)

    # Perform the query
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )

    return results

def get_groq_client() -> Groq:
    """Initialize and return the Groq client with error handling."""
    api_key = GROQ_API_KEY

    if not api_key:
        logger.error("GROQ_API_KEY is not set in the environment.")
        raise EnvironmentError("Missing GROQ_API_KEY in environment.")
    
    try:
        return Groq(api_key=api_key)
    except GroqError as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        raise

def generate_completion(
    prompt: str,
    model: str,
    role: str = "user",
    system_prompt: Optional[str] = None
) -> str:
    """Generate a completion from Groq API using a given prompt and model."""
    client = get_groq_client()
     
    messages: List[Dict[str, str]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": role, "content": prompt})

    try:
        logger.info("Sending request to Groq API...")
        response = client.chat.completions.create(
            messages=messages,
            model=MODEL_NAME
        )
        result = response.choices[0].message.content
        logger.info("Response received successfully.")
        return result
    except Exception as e:
        logger.error(f"Error generating completion: {e}")
        raise

def clean_title(filename: str) -> str:
    name_without_ext = filename.rsplit('.', 1)[0]  # remove extension
    cleaned = name_without_ext.replace('_', ' ')   # replace underscores with spaces
    bold_title = f"**{cleaned.strip().title()}**"  # add bold formatting and title case
    return bold_title

# ---------- RAG Pipeline ----------
def rag_query(user_query: str, persist_directory: str, collection_name: str, n_results: int = 3) -> str:
    # Step 1: Retrieve relevant chunks
    search_results = query_chromadb(persist_directory, collection_name, user_query, n_results=n_results)

    # Step 2: Combine retrieved chunks into context
    context_with_titles = []
    for doc, meta in zip(search_results["documents"][0], search_results["metadatas"][0]):
        raw_title = meta.get("source", "Unknown Source")
        book_title = clean_title(raw_title)  # Clean the title
        context_with_titles.append(f"From '{book_title}':\n{doc}")

    context_text = "\n\n".join(context_with_titles)

    # Step 3: Build final prompt
    prompt = (
        f"Use the following excerpts from Nietzsche's works to answer the question.\n\n"
        f"{context_text}\n\n"
        f"Question: {user_query}\n\n"
        f"Answer:"
    )

    # Step 4: Call the LLM
    answer = generate_completion(
        prompt=prompt,
        system_prompt="You are a philosophical assistant specializing in Friedrich Nietzsche's works. Always cite the book title when using excerpts.",
        model=MODEL_NAME
    )

    return answer

# ---------- Startup ----------
@app.on_event("startup")
def startup_event():
    global chroma_client, chroma_collection
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = chroma_client.get_collection(name=COLLECTION_NAME)
    logger.info("ChromaDB client initialized successfully.")

# ---------- Endpoints ----------
@app.post("/prompt", response_model=QueryResponse)
def rag_endpoint(request: QueryRequest):
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
