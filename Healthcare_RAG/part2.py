
"""
# Part 2: Langchain RAG Implementation
# Goal:
# 1. Load PDFs as Langchain Documents
# 2. Chunk the text with overlap (better context continuity)
# 3. Embed chunks into vectors
# 4. Store vectors in a FAISS vector database
# 5. Retrieve top_k chunks for a query and run a QA chain
"""
#--------------------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# old/deprecated >>> from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# had to look for update on stack overflow, old deprecated
# original code for class >>> from langchain.chains import RetrievalQA
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

# added the use of real local LLM instead of "mock_llm"
# old/deprecated >>> from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM

import time

# ---------------------------------------------
# 1. Loading PDFs using PyPDFLoader loading each page as a document with metadata (page number, source, etc.)
# ---------------------------------------------
# the original code from class
# loader = PyPDFLoader("data/diabetes.pdf")
# docs = loader.load()
# ---------------------------------------------
# updated to load multiple PDFs
docs = []
pdf_files = [
     "data/diabetes.pdf",
     "data/standards.pdf"
 ]
for pdf in pdf_files:
     loader = PyPDFLoader(pdf)
     docs.extend(loader.load())

# ---------------------------------------------
# 2. Split text into chunks
# chunk_size: how large each chunk is
# chunk_overlap: repeated text between chunks to preserve context across boundaries
# RecursiveCharacterTextSplitter is common default for general text chunking
# ---------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# ---------------------------------------------
# 3. Create embeddings
# HuggingFaceEmbeddings will generate vectors for each chunk
# (Default model is usually fine, but can also specify one explicitly)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------------------------
# 4. Build the vectorstore (FAISS)
# stores the chunk embeddings and supports fast similarity search
# ---------------------------------------------
vectorstore = FAISS.from_documents(splits, embeddings)

# ---------------------------------------------
# 5. create ollama backed LLM for Langchain
# Ollama MUST be installed and running on your computer
# make sure to use the model pulled: 'ollama pull gemma3:270m' or whatever
# ---------------------------------------------
llm = OllamaLLM(model="gemma3:1b", temperature=0.2)

# ---------------------------------------------
# 6. Build RetrievalQA chain
# retriever fetches top_k relevant chunks into the prompt context
# ---------------------------------------------
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(search_kwargs={"k": 3}))

# ---------------------------------------------
# 7. run test queries + timing
# ---------------------------------------------
queries = [
    "What are metformin side effects?",
    "A1C target range for type 2 diabetes?",
    "How to treat hypoglycemia?",
    "When to check blood glucose?",
    "Foot care recommendations for diabetics?"
]
for q in queries:
    print(f"\n Testing: {q}")
    start = time.perf_counter()
    answer = qa_chain.run(q)
    end = time.perf_counter() - start

    print("LLM Answer:\n", answer)
    print(f"Time: {end:.2f} seconds")



