import os
import logging
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


# Project root (directory containing backend/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load .env if present
load_dotenv(BASE_DIR / ".env")


# API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()


# Model configuration
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# RAG configuration
CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "500")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "50")
)

RETRIEVAL_K = int(
    os.getenv("RETRIEVAL_K", "10")
)

RERANK_TOP_K = int(
    os.getenv("RERANK_TOP_K", "3")
)


# Context window & Memory policy configuration
CONTEXT_WINDOW_SIZE = int(
    os.getenv("CONTEXT_WINDOW_SIZE", "10")
)

MEMORY_WINDOW_SIZE = int(
    os.getenv("MEMORY_WINDOW_SIZE", "5")
)


# Storage & upload limits
DOCUMENTS_DIR = BASE_DIR / "documents"
MAX_UPLOAD_SIZE_BYTES = int(
    os.getenv("MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024))  # 10 MB default
)
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}


# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


# Authentication & JWT configuration
_DEFAULT_INSECURE_KEY = "rorak-ai-v2.6-insecure-default-secret-key-change-in-prod"
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", _DEFAULT_INSECURE_KEY)
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()

if JWT_SECRET_KEY == _DEFAULT_INSECURE_KEY and ENVIRONMENT in {"production", "prod"}:
    raise RuntimeError("Insecure default JWT_SECRET_KEY cannot be used in production. Set JWT_SECRET_KEY in environment.")
elif JWT_SECRET_KEY == _DEFAULT_INSECURE_KEY:
    logger.warning("Using default insecure JWT_SECRET_KEY for development. Change this in production!")

JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", str(60 * 24))
)


# Validation
if not GROQ_API_KEY:
    logging.warning(
        "GROQ_API_KEY is not configured in environment or .env. "
        "Generation features will be unavailable."
    )

# Agent Workspace & Search Configuration
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
WORKSPACES_DIR = Path(os.getenv("WORKSPACES_DIR", str(BASE_DIR / "workspaces"))).resolve()

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if not DATABASE_URL:
    logger.warning("DATABASE_URL is not configured")
