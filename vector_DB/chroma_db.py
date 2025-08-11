import os
import chromadb

# ====== CONFIG ======
BOOKS_FOLDER = "/path/to/nietzsche_books"
DB_PATH = "/path/to/chroma_storage/nietzsche_db"  # fixed location for persistence
COLLECTION_NAME = "nietzsche_books"
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50  # characters overlap between chunks

# ====== 1. Initialize Chroma persistent client ======
client = chromadb.PersistentClient(path=DB_PATH)
