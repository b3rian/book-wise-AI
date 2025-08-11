import chromadb
from chromadb.config import Settings
from inference.generate_text import generate_completion, get_groq_client

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

# ---------- RAG Pipeline ----------
def rag_query(user_query: str, persist_directory: str, collection_name: str, n_results: int = 3) -> str:
    # Step 1: Retrieve relevant chunks
    search_results = query_chromadb(persist_directory, collection_name, user_query, n_results=n_results)

    # Step 2: Combine retrieved chunks into context
    context_chunks = search_results["documents"][0]
    context_text = "\n\n".join(context_chunks)

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
    PERSIST_DIR = "./nietzsche_db"  # Path to your ChromaDB folder
    COLLECTION_NAME = "nietzsche_collection"

    user_question = "What does Nietzsche say about the meaning of life?"
    response = rag_query(user_question, PERSIST_DIR, COLLECTION_NAME, n_results=5)

    print("\n--- LLM Response ---\n")
    print(response)


