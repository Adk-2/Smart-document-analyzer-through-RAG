from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

for dotenv_path in (PROJECT_ROOT / ".env", Path.cwd() / ".env"):
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        break


EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

TOP_K = 8
ANSWER_TOP_K = 10
BM25_TOP_K = 8
FETCH_K = 25
RAG_CONTEXT_MAX_DOCS = ANSWER_TOP_K
RAG_CONTEXT_MAX_CHARS = 16000

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHROMA_COLLECTION_NAME = "documents"
VECTOR_STORE_PATH = PROJECT_ROOT / "data" / "vector_store"

GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_TEMPERATURE = 0.1
GROQ_MAX_TOKENS = 1024
GROQ_REQUEST_TIMEOUT_SECONDS = 90
RETRIEVAL_TIMEOUT_SECONDS = 30
