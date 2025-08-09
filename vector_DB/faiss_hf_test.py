 



# 2. Example documents
docs = [
    "God is dead. God remains dead. And we have killed him.",
    "He who has a why to live can bear almost any how.",
    "To live is to suffer, to survive is to find some meaning in the suffering."
]


# 4. Create FAISS index
dim = embeddings.shape[1]  # embedding dimension
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

print(f"FAISS index contains {index.ntotal} vectors.")

# 5. Test a query
query = "Finding purpose in suffering"
query_emb = model.encode([query]).astype("float32")

# 6. Search
k = 2  # top results
distances, indices = index.search(query_emb, k)

print("\nQuery:", query)
print("Top matches:")
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. {docs[idx]} (distance: {distances[0][i]:.4f})")