from backend.models.database import Source


def serialize_source(source: Source) -> dict:
    indexed_at = source.indexed_at.isoformat() if source.indexed_at else None
    return {
        "id": source.id,
        "workspace_name": source.workspace_name,
        "source_type": source.source_type,
        "source_name": source.source_name,
        "source_url": source.source_url,
        "status": source.status,
        "indexed_at": indexed_at,
        # Backward-compatible fields consumed by the current frontend.
        "title": source.source_name,
        "url": source.source_url,
        "workspace": source.workspace_name,
        "type": source.source_type,
        "chunks": 0,
    }
