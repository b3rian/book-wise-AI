import chromadb
from chromadb.config import Settings

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


# Example usage
if __name__ == "__main__":
    persist_dir = r"D:\Documents\chromadb\nietzsche_db" 
    collection_name = "nietzsche_books"
    query = "What is plato?"

    results = query_chromadb(persist_dir, collection_name, query, n_results=3)

    # Pretty print results
    for i, doc in enumerate(results["documents"][0]):
        print(f"Result {i+1}:")
        print(f"Document: {doc}")
        print(f"Metadata: {results['metadatas'][0][i]}")
        print(f"ID: {results['ids'][0][i]}")
        print("-" * 40)


