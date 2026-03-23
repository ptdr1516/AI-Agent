import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.chat import router as chat_router
from api.tools import router as tools_router
from api.upload import router as upload_router
from api.rag import router as rag_router
from api.documents import router as documents_router
from api.metrics import router as metrics_router
from core.logger import log
from core.errors import setup_exception_handlers
from core.config import settings
from core.tracing import configure_langsmith

log.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
settings.validate_prod_security()
configure_langsmith()

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)
setup_exception_handlers(app)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stable paths for clients (do not rename without a frontend migration):
#   POST /api/chat, POST /api/chat/stream, POST /api/upload, POST /api/rag/query, …
app.include_router(chat_router, prefix="/api")
app.include_router(tools_router, prefix="/api")
app.include_router(upload_router, prefix="/api")
app.include_router(rag_router, prefix="/api")
app.include_router(documents_router, prefix="/api")
app.include_router(metrics_router, prefix="/api")

@app.get("/")
def root():
    return {"status": "ok", "message": "Nova Agent API is running", "docs": "/docs"}

@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
