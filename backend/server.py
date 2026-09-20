from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import chat, health, ingestion, sources
from backend.database.db import init_db
from backend.utils.error_handlers import register_exception_handlers
from backend.utils.logging import configure_logging


configure_logging()

app = FastAPI(
    title="RAG Analyzer API",
    description="Backend API for RAG research workspace",
    version="0.1.0"
)

# Enable CORS for local frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    init_db()


register_exception_handlers(app)

app.include_router(health.router)
app.include_router(ingestion.router)
app.include_router(sources.router)
app.include_router(chat.router)
