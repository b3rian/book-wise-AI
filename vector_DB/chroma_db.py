import os
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# CONFIGs
BOOKS_FOLDER = "/path/to/nietzsche_books"
DB_PATH = "/path/to/chroma_storage/nietzsche_db"  # fixed storage location
COLLECTION_NAME = "nietzsche_books"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 1. Initialize persistent Chroma client
client = chromadb.PersistentClient(path=DB_PATH)

# If collection exists, use it; otherwise, create it
try:
    collection = client.get_collection(COLLECTION_NAME)
except:
    collection = client.create_collection(name=COLLECTION_NAME)

# ====== 2. Initialize Chroma's built-in embedding function ======
embedding_fn = DefaultEmbeddingFunction()

# ====== 3. Helper: Split text into chunks ======
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ====== 4. Process all books ======
doc_id_counter = 1

for filename in os.listdir(BOOKS_FOLDER):
    if filename.endswith(".txt"):
        file_path = os.path.join(BOOKS_FOLDER, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Clean and chunk
        text = text.strip().replace("\n", " ")
        chunks = chunk_text(text)

        # Generate embeddings
        embeddings = embedding_fn(chunks)

        for i, chunk in enumerate(chunks):
            print(f"\n📄 File: {filename} | Chunk {i+1}/{len(chunks)}")
            print(f"Text chunk: {chunk[:100]}...")  # preview first 100 chars
            print(f"Embedding dims: {len(embeddings[i])}")
            print(f"First 10 values: {embeddings[i][:10]}")

        # Store in Chroma
        ids = [f"doc{doc_id_counter + i}" for i in range(len(chunks))]
        metadatas = [{"source": filename} for _ in chunks]

        collection.add(
            documents=chunks,
            embeddings=embeddings,  # explicitly pass embeddings
            ids=ids,
            metadatas=metadatas
        )

        doc_id_counter += len(chunks)
        print(f"✅ Stored {len(chunks)} chunks from {filename}")

print("\n🎯 All Nietzsche books embedded and stored in ChromaDB!")
