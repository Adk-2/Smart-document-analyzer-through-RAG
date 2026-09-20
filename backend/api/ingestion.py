from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.models.schemas import IngestURLRequest
from backend.services.ingestion_service import ingest_pdf_uploads, ingest_webpage


router = APIRouter()


@router.post("/ingest/url")
def ingest_webpage_endpoint(
    request: IngestURLRequest,
    db: Session = Depends(get_db),
):
    return ingest_webpage(db, request)


@router.post("/ingest/pdf")
def ingest_pdf_endpoint(
    files: list[UploadFile] = File(...),
    workspace_id: str | None = Form(None),
    workspace_name: str | None = Form(None),
    db: Session = Depends(get_db),
):
    return ingest_pdf_uploads(db, files, workspace_id, workspace_name)
