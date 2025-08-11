import chromadb
from chromadb.config import Settings
import os
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
from groq import Groq, GroqError
import argparse

# Load environment variables from .env
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    client = chromadb.PersistentClient(path=persist_directory)

    # Get the collection
    collection = client.get_collection(name=collection_name)

    # Perform the query
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )

    return results

def get_groq_client() -> Groq:
    """Initialize and return the Groq client with error handling."""
    api_key = os.getenv("GROQ_API_KEY")

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
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
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
            model=model
        )
        result = response.choices[0].message.content
        logger.info("Response received successfully.")
        return result
    except Exception as e:
        logger.error(f"Error generating completion: {e}")
        raise

# ---------- RAG Pipeline ----------
def rag_query(user_query: str, persist_directory: str, collection_name: str, n_results: int = 3) -> str:
    # Step 1: Retrieve relevant chunks
    search_results = query_chromadb(persist_directory, collection_name, user_query, n_results=n_results)

    # Step 2: Combine retrieved chunks into context
    context_with_titles = []
    for doc, meta in zip(search_results["documents"][0], search_results["metadatas"][0]):
        book_title = meta.get("source", "Unknown Source")
        context_with_titles.append(f"From '{book_title}':\n{doc}")

    context_text = "\n\n".join(context_with_titles)

    # Step 3: Build final prompt
    prompt = (
        f"Use the following excerpts from Nietzsche's works to answer the question.\n\n"
        f"{context_text}\n\n"
        f"Question: {user_query}\n\n"
        f"Answer:"
    )

    # Step 3: Build final prompt
    prompt = f"Use the following excerpts from Nietzsche's works to answer the question.\n\nContext:\n{context_text}\n\nQuestion: {user_query}\n\nAnswer:"

    # Step 4: Call the LLM
    answer = generate_completion(
        prompt=prompt,
        system_prompt="You are a philosophical assistant specializing in Friedrich Nietzsche's works. Always cite relevant excerpts where possible."
    )

    return answer

# ---------- Example Usage ----------
if __name__ == "__main__":
    PERSIST_DIR = r"D:\Documents\chromadb\nietzsche_db"  # Path to your ChromaDB folder
    COLLECTION_NAME = "nietzsche_books"

    user_question = "What does Nietzsche say about the meaning of life?"
    response = rag_query(user_question, PERSIST_DIR, COLLECTION_NAME, n_results=5)

    print("\n--- LLM Response ---\n")
    print(response)


