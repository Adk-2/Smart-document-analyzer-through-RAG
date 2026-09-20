import uuid
import shutil
import tempfile
from pathlib import Path

from fastapi import HTTPException
from fastapi import UploadFile
from sqlalchemy.orm import Session

from backend.database import crud
from backend.models.schemas import IngestURLRequest
from backend.utils.logging import get_logger
from src.pipeline import ingest_pdfs, ingest_url
from src.rag_pipeline import refresh_retriever
from src.workspace_manager import WorkspaceManager


logger = get_logger(__name__)


def _resolve_workspace(workspace_id: str | None = None, workspace_name: str | None = None) -> str:
    return (
        (workspace_id or "").strip()
        or (workspace_name or "").strip()
        or WorkspaceManager.get_workspace()
        or f"session_{uuid.uuid4().hex[:8]}"
    )


def _validate_pdf_upload(file: UploadFile) -> str:
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="PDF filename is required")

    is_pdf = (
        file.content_type == "application/pdf"
        or filename.lower().endswith(".pdf")
    )
    if not is_pdf:
        raise HTTPException(status_code=400, detail=f"Only PDF uploads are supported: {filename}")

    return filename


def ingest_webpage(db: Session, request: IngestURLRequest) -> dict:
    source_url = request.url
    workspace = _resolve_workspace(request.workspace_id, request.workspace_name)

    logger.info(
        "Ingestion request received",
        extra={
            "event": "ingestion_request",
            "context": {
                "workspace_name": workspace,
                "source_url": source_url,
            },
        },
    )

    WorkspaceManager.set_workspace(workspace)
    logger.info(
        "Workspace activated",
        extra={
            "event": "workspace_operation",
            "context": {
                "workspace_name": workspace,
                "operation": "set_workspace",
            },
        },
    )

    try:
        result = ingest_url(source_url)
        refresh_retriever(workspace_name=workspace)

        source = crud.create_source(
            db,
            workspace_name=workspace,
            source_type="webpage",
            source_name=result.get("title", "Unknown"),
            source_url=source_url,
            status="indexed",
        )
        crud.record_ingestion_history(
            db,
            workspace_name=workspace,
            source_id=source.id,
            source_type="webpage",
            source_name=source.source_name,
            source_url=source.source_url,
            status="success",
            message="URL indexed successfully",
        )
        logger.info(
            "Ingestion completed",
            extra={
                "event": "ingestion_request",
                "context": {
                    "workspace_name": workspace,
                    "source_id": source.id,
                    "source_url": source_url,
                    "status": "success",
                    "total_chunks_created": result.get("chunks", 0),
                    "vector_insertion": result.get("vector_insertion", {}),
                    "vector_total_documents": result.get("vector_state", {}).get("total_documents"),
                    "chunk_previews": result.get("chunk_diagnostics", [])[:5],
                },
            },
        )

        return {
            "status": "success",
            "workspace": workspace,
            "title": result.get("title", "Unknown"),
            "url": source_url,
            "chunks": result.get("chunks", 0),
            "vector_insertion": result.get("vector_insertion", {}),
            "vector_total_documents": result.get("vector_state", {}).get("total_documents"),
            "source_id": source.id,
        }
    except Exception as exc:
        crud.record_ingestion_history(
            db,
            workspace_name=workspace,
            source_type="webpage",
            source_url=source_url,
            status="failed",
            message=str(exc),
        )
        logger.exception(
            "Ingestion failed",
            extra={
                "event": "api_failure",
                "context": {
                    "workspace_name": workspace,
                    "source_url": source_url,
                    "status": "failed",
                },
            },
        )
        raise HTTPException(
            status_code=502,
            detail=f"URL ingestion failed: {str(exc)}",
        ) from exc


def ingest_pdf_uploads(
    db: Session,
    files: list[UploadFile],
    workspace_id: str | None = None,
    workspace_name: str | None = None,
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required")

    workspace = _resolve_workspace(workspace_id, workspace_name)
    filenames = [_validate_pdf_upload(file) for file in files]

    logger.info(
        "PDF ingestion request received",
        extra={
            "event": "ingestion_request",
            "context": {
                "workspace_name": workspace,
                "source_type": "pdf",
                "file_count": len(filenames),
            },
        },
    )

    WorkspaceManager.set_workspace(workspace)

    with tempfile.TemporaryDirectory(prefix="rag_pdf_ingest_") as temp_dir:
        temp_paths: list[str] = []
        for index, file in enumerate(files):
            filename = filenames[index]
            temp_path = Path(temp_dir) / f"{index}.pdf"
            try:
                file.file.seek(0)
                with temp_path.open("wb") as destination:
                    shutil.copyfileobj(file.file, destination)
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not read uploaded PDF {filename}: {str(exc)}",
                ) from exc
            temp_paths.append(str(temp_path))

        try:
            source_names = {
                temp_path: filenames[index]
                for index, temp_path in enumerate(temp_paths)
            }
            chunks = ingest_pdfs(temp_paths, source_names=source_names)
            refresh_retriever(workspace_name=workspace)

            sources = []
            for filename in filenames:
                source = crud.create_source(
                    db,
                    workspace_name=workspace,
                    source_type="pdf",
                    source_name=filename,
                    source_url=filename,
                    status="indexed",
                )
                crud.record_ingestion_history(
                    db,
                    workspace_name=workspace,
                    source_id=source.id,
                    source_type="pdf",
                    source_name=source.source_name,
                    source_url=source.source_url,
                    status="success",
                    message="PDF indexed successfully",
                )
                sources.append(
                    {
                        "id": source.id,
                        "filename": filename,
                        "source_name": source.source_name,
                        "source_url": source.source_url,
                    }
                )

            logger.info(
                "PDF ingestion completed",
                extra={
                    "event": "ingestion_request",
                    "context": {
                        "workspace_name": workspace,
                        "source_type": "pdf",
                        "file_count": len(filenames),
                        "total_chunks_created": chunks,
                    },
                },
            )

            return {
                "status": "success",
                "workspace": workspace,
                "file_count": len(filenames),
                "files": sources,
                "chunks": chunks,
            }
        except HTTPException:
            raise
        except Exception as exc:
            for filename in filenames:
                crud.record_ingestion_history(
                    db,
                    workspace_name=workspace,
                    source_type="pdf",
                    source_name=filename,
                    source_url=filename,
                    status="failed",
                    message=str(exc),
                )
            logger.exception(
                "PDF ingestion failed",
                extra={
                    "event": "api_failure",
                    "context": {
                        "workspace_name": workspace,
                        "source_type": "pdf",
                        "file_count": len(filenames),
                    },
                },
            )
            raise HTTPException(
                status_code=502,
                detail=f"PDF ingestion failed: {str(exc)}",
            ) from exc
