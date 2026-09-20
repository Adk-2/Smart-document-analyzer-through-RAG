from sqlalchemy.orm import Session

from backend.database import crud
from backend.utils.logging import get_logger
from backend.utils.serializers import serialize_source
from src.workspace_manager import WorkspaceManager


logger = get_logger(__name__)


def get_sources_response(
    db: Session,
    workspace_id: str | None = None,
    workspace_name: str | None = None,
) -> dict:
    workspace = (
        (workspace_id or "").strip()
        or (workspace_name or "").strip()
        or WorkspaceManager.get_workspace()
    )
    sources = [
        serialize_source(source)
        for source in crud.list_sources(db, workspace)
    ]
    logger.info(
        "Sources fetched",
        extra={
            "event": "workspace_operation",
            "context": {
                "workspace_name": workspace,
                "operation": "list_sources",
                "count": len(sources),
            },
        },
    )
    return {
        "workspace": workspace,
        "sources": sources,
        "count": len(sources),
    }
