from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────
    APP_NAME: str = "Nova Agent"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    # Set to True on Render/Prod to enable strict security checks
    PROD_MODE: bool = False
    # Bearer token for API authentication. Set in .env to enable protection.
    # Leave empty in development to skip auth entirely.
    API_TOKEN: str = ""
    # List of allowed origins for CORS. Defaults to "*" for dev.
    # In production, set this to your Render frontend URL.
    ALLOWED_ORIGINS: list[str] = ["*"]

    # ── LLM / OpenRouter ─────────────────────────────────
    OPENROUTER_API_KEY: str  # required — will raise ValueError at startup if missing
    LLM_MODEL: str = "openai/gpt-3.5-turbo"
    LLM_API_BASE: str = "https://openrouter.ai/api/v1"
    LLM_TEMPERATURE: float = 0.0
    HTTP_REFERER: str = "http://localhost:5173"
    # Hard cap for OpenRouter "max_tokens".
    # OpenRouter may reject requests when credits/quota are low and the requested
    # max_tokens exceeds what the account can afford.
    LLM_MAX_TOKENS: int = 2048

    # ── LangSmith Observability ───────────────────────────
    # Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY to enable.
    # All LangGraph / LangChain calls are auto-traced — no code changes needed.
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "nova-agent"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"

    # ── Search ────────────────────────────────────────────
    # Optional at startup — raises clear error at tool call time if missing
    TAVILY_API_KEY: str = ""
    TAVILY_MAX_RESULTS: int = 5
    TAVILY_SEARCH_DEPTH: str = "basic"  # "basic" | "advanced"
    TAVILY_TIMEOUT: float = 10.0

    # ── Redis Cache ───────────────────────────────────────
    # Leave empty to disable Redis and use the in-memory fallback instead.
    REDIS_URL: str = ""               # e.g. redis://localhost:6379/0
    REDIS_KEY_PREFIX: str = "nova"    # namespace prefix for all cache keys
    SEARCH_CACHE_TTL: int = 300       # seconds to cache search results (5 min)

    # ── Agent ─────────────────────────────────────────────
    AGENT_MAX_ITERATIONS: int = 5
    MEMORY_WINDOW_SIZE: int = 15
    MEMORY_DB_PATH: str = "sqlite:///chat_memory.db"

    # ── Vector memory (FAISS + sentence-transformers) ────
    VECTOR_MEMORY_DIR: str = "vector_memory_data"
    VECTOR_MEMORY_TOP_K: int = 4
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ── RAG embeddings (singleton via rag.embeddings) ─────
    # huggingface: sentence-transformers / HuggingFaceEmbeddings (uses EMBEDDING_MODEL)
    # openai: OpenAIEmbeddings — OpenAI or any OpenAI-compatible API (e.g. OpenRouter)
    EMBEDDING_PROVIDER: str = "huggingface"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    # If empty, OpenAI path uses OPENROUTER_API_KEY. Set EMBEDDING_OPENAI_API_BASE for OpenRouter/Azure.
    EMBEDDING_OPENAI_API_KEY: str = ""
    EMBEDDING_OPENAI_API_BASE: str = ""

    # ── RAG vector store (FAISS or Chroma via rag.vectorstore) ──
    RAG_VECTORSTORE_BACKEND: str = "faiss"  # faiss | chroma
    RAG_FAISS_DIR: str = "rag_data/faiss_index"
    RAG_CHROMA_DIR: str = "rag_data/chroma"
    RAG_CHROMA_COLLECTION: str = "rag_documents"
    RAG_CHUNK_SIZE: int = 500
    RAG_CHUNK_OVERLAP: int = 100
    RAG_RETRIEVAL_K: int = 4
    RAG_FETCH_K: int = 20

    # ── File upload (RAG indexing) ───────────────────────
    UPLOAD_DIR: str = "uploads"
    UPLOAD_MAX_BYTES: int = 25 * 1024 * 1024  # 25 MiB

    # ── Token Pricing (per token, not per 1k) ────────────
    # Default: GPT-3.5-turbo proxy pricing as used on OpenRouter standard tier
    COST_PROMPT_PER_TOKEN: float = 0.0000005     # $0.50  / 1M tokens
    COST_COMPLETION_PER_TOKEN: float = 0.0000015  # $1.50  / 1M tokens
    
    # ── Feature Toggles ──────────────────────────────────
    ENABLE_TRACING: bool = True
    ENABLE_METRICS: bool = True
    ENABLE_EVAL: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def is_auth_enabled(self) -> bool:
        return bool(self.API_TOKEN.strip())

    def validate_prod_security(self):
        """Raise error if in PROD_MODE but security is disabled."""
        if self.PROD_MODE and not self.is_auth_enabled:
            raise ValueError(
                "CRITICAL SECURITY ERROR: PROD_MODE is enabled but API_TOKEN is empty. "
                "You MUST set a secure API_TOKEN in your production environment variables."
            )
        if self.PROD_MODE and "*" in self.ALLOWED_ORIGINS:
             import logging
             logging.warning("SECURITY WARNING: PROD_MODE is on but ALLOWED_ORIGINS contains '*'.")


# No try/except: fail fast at startup if OPENROUTER_API_KEY is missing.
# A 401 at runtime is far harder to debug than a clear startup error.
settings = Settings()

