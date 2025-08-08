from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Load Hugging Face embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Example documents
docs = [
    "God is dead. God remains dead. And we have killed him.",
    "He who has a why to live can bear almost any how.",
    "To live is to suffer, to survive is to find some meaning in the suffering."
]