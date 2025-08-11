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
