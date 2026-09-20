from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.services.source_service import get_sources_response


router = APIRouter()


@router.get("/sources")
def get_sources(
    workspace_id: str | None = None,
    workspace_name: str | None = None,
    db: Session = Depends(get_db),
):
    return get_sources_response(db, workspace_id, workspace_name)
