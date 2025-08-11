import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.create_collection(name="knowledge_base")

collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents = [
    "God is dead. God remains dead. And we have killed him.",
    "He who has a why to live can bear almost any how.",
    "To live is to suffer, to survive is to find some meaning in the suffering."
])

results = collection.query(
    query_texts=["Finding purpose in suffering"],
    n_results=2
)
print(results)

