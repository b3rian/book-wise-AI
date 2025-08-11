import os
import chromadb

# ====== CONFIG ======
BOOKS_FOLDER = "D:\Documents\text_files" 
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
