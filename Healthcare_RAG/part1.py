# -----------------------------------------
# Custom Retrieval_Augmented Generation (RAG) Class
# 1. Loading and chunking PDF documents
# 2. Creating embeddings using a transformer model
# 3. Storing embeddings in a FAISS vector index
# 4. Retrieving relevant text based on a user query
# ------------------------------------------


import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
import time

class HealthcareRAG:
    # model from Hugging Face
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        # load the pretrained sentence embedding model, convert text into dense numeric vectors
        self.embedder = SentenceTransformer(embedding_model)
        # FAISS index initialized after documents are added
        # FAISS = Facebook AI Similarity Search
        self.index = None
        # stores the original text chunks to map vector search results back to readable text
        self.documents = []

    def add_documents(self, file_paths):
        """
        Reads PDF files, extracts text, chunks the text, generates embeddings,
        and stores them in FAISS
        """
        all_chunks = []
        # loop through PDF files
        for path in file_paths:
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                # extract text
                text = ''.join(
                    page.extract_text()
                    for page in reader.pages
                    if page.extract_text()
                )
            # split long text into fixed-size chunks (500 chars)
            # chunking improve retrieval accuracy and memory efficiency
            chunks = [
                text[i:i+500]
                for i in range(0, len(text), 500)
                if text[i:i+500].strip()
            ]
            all_chunks.extend(chunks)
            # Save chunks to be retrieved later
            self.documents = all_chunks
            # convert text chinks into vector embeddings
            embeddings = self.embedder.encode(all_chunks)
            # get dimensionality of the embeddings
            dim = embeddings.shape[1]
            # create FAISS index using cosine similarity (via inner product)
            self.index = faiss.IndexFlatIP(dim)
            # normalize vectors so cosine similarity works correctly
            faiss.normalize_L2(embeddings)
            # add embeddings into the FAISS index
            self.index.add(embeddings.astype('float32'))
            print(f"Check custom: Indexed {len(all_chunks)} chunks")

    def retrieve(self, query, k=3):
        """
        Converts a query into an embedding and retrieves the
        top-k most relevant document chunks.
        """
        # Encode the query into a vector
        query_emb = self.embedder.encode([query])
        # normalize query vector for cosine similarity
        faiss.normalize_L2(query_emb)
        # search the FAISS index for the closest vectors
        scores, indices = self.index.search(query_emb.astype('float32'), k)
        # map the retrieved indices back to text chunks
        return [self.documents[i] for i in indices[0]]

    def generate(self, query):
        """
        Retrieves relevant context for the query.
        (In a full RAG system, this context would be sent to an LLM.)
        """
        # retrieve top relevant chunks
        context = '\n---\n'.join(self.retrieve(query))
        # print preview of retrieved content for debugging
        # print("Custom context:", context[:200] + "...")
        # simulated response(placeholder instead of real LLM)
        return f"Custom RAG: {context[:200]}..."

# -----------------------------------
# Initialize and test the custom RAG pipeline
# -----------------------------------
# create an instance of the RAG system
custom_rag = HealthcareRAG()
# load and index PDF documents
custom_rag.add_documents(["data/diabetes.pdf", "data/standards.pdf"])

queries = [
    "what are metformin side effects?",
    "A1C target range for type 2 diabetes?",
    "How to treat hypoglycemia?",
    "When to check blood glucose?",
    "Foot care recommendations for diabetics?"
]
for q in queries:
    print(f"\n Testing: {q}")
    start = time.time()
    response = custom_rag.generate(q)
    elapsed = time.time() - start
    print(response)
    print(f"RAG time: {elapsed:.3f} seconds.")







