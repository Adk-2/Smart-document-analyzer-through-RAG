from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models.database import IngestionHistory, Source, Workspace


def get_or_create_workspace(db: Session, name: str) -> Workspace:
    clean_name = name.strip()
    workspace = db.scalar(select(Workspace).where(Workspace.name == clean_name))
    if workspace:
        return workspace

    workspace = Workspace(name=clean_name)
    db.add(workspace)
    db.commit()
    db.refresh(workspace)
    return workspace


def create_source(
    db: Session,
    *,
    workspace_name: str,
    source_type: str,
    source_name: str,
    source_url: str,
    status: str = "indexed",
) -> Source:
    get_or_create_workspace(db, workspace_name)

    source = Source(
        id=str(uuid4()),
        workspace_name=workspace_name.strip(),
        source_type=source_type,
        source_name=source_name,
        source_url=source_url,
        status=status,
        indexed_at=datetime.now(timezone.utc),
    )
    db.add(source)
    db.commit()
    db.refresh(source)
    return source


def list_sources(db: Session, workspace_name: str | None = None) -> list[Source]:
    statement = select(Source).order_by(Source.indexed_at.desc())
    if workspace_name:
        statement = statement.where(Source.workspace_name == workspace_name.strip())
    return list(db.scalars(statement).all())


def record_ingestion_history(
    db: Session,
    *,
    workspace_name: str,
    source_type: str,
    source_url: str,
    status: str,
    source_id: str | None = None,
    source_name: str | None = None,
    message: str | None = None,
) -> IngestionHistory:
    get_or_create_workspace(db, workspace_name)

    history = IngestionHistory(
        workspace_name=workspace_name.strip(),
        source_id=source_id,
        source_type=source_type,
        source_name=source_name,
        source_url=source_url,
        status=status,
        message=message,
    )
    db.add(history)
    db.commit()
    db.refresh(history)
    return history
