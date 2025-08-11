import os
import chromadb

# ====== CONFIG ======
BOOKS_FOLDER = r"D:\Documents\text_files" 
DB_PATH = r"D:\Documents\chromadb"  # fixed location for persistence
COLLECTION_NAME = "nietzsche_books"
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50  # characters overlap between chunks

# ====== 1. Initialize Chroma persistent client ======
client = chromadb.PersistentClient(path=DB_PATH)

# If collection exists, use it; otherwise, create it
try:
    collection = client.get_collection(COLLECTION_NAME)
except:
    collection = client.create_collection(name=COLLECTION_NAME)

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

doc_id_counter = 1

for filename in os.listdir(BOOKS_FOLDER):
    if filename.endswith(".txt"):
        file_path = os.path.join(BOOKS_FOLDER, filename)
        
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Remove extra spaces/newlines
        text = text.strip().replace("\n", " ")
        
        # Split into chunks
        chunks = chunk_text(text)
        
        # Add each chunk to Chroma (built-in embedding will be used)
        ids = [f"doc{doc_id_counter + i}" for i in range(len(chunks))]
        metadatas = [{"source": filename} for _ in chunks]
        
        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        
        doc_id_counter += len(chunks)
        print(f"Stored {len(chunks)} chunks from {filename}")

print("✅ All books embedded and stored in ChromaDB!")
