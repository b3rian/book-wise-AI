import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
chromadb_client = chromadb.Client()
collection = chromadb_client.create_collection(name="test_collection")

docs =[
    "God is dead. God remains dead. And we have killed him.",
    "He who has a why to live can bear almost any how.",
    "To live is to suffer, to survive is to find some meaning in the suffering."
]

embeddings = model.encode(docs).tolist()
collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents = docs,
    embeddings=embeddings
)
     
query_embedding = model.encode(["Finding purpose in suffering"]).tolist()
results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)
print(results)

