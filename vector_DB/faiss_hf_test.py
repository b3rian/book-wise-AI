from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Load Hugging Face embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")