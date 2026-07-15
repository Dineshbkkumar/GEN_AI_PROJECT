import os
from dotenv import load_dotenv

load_dotenv()

# PDF Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Retrieval & Ranking
TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 5

# Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

# Weaviate
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_COLLECTION = "RAGDocuments"

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# LLM Settings
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 2000
