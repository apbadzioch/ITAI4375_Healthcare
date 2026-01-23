import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2

class HealthcareRAG:
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        self.embedder = SentenceTransformer(embedding_model)
        self.index = None
        self.documents = []


