import time

from fastapi import HTTPException

from backend.models.schemas import ChatRequest
from backend.utils.logging import get_logger
from src.rag_pipeline import answer_question
from src.workspace_manager import WorkspaceManager


logger = get_logger(__name__)
_conversation_memory: dict[str, list[dict[str, str]]] = {}
MAX_MEMORY_MESSAGES = 10


def _conversation_key(request: ChatRequest, workspace_name: str) -> str:
    return request.conversation_id or f"workspace:{workspace_name}"


def _get_conversation_history(request: ChatRequest, workspace_name: str) -> list[dict[str, str]]:
    if request.history:
        return request.history[-MAX_MEMORY_MESSAGES:]
    return _conversation_memory.get(_conversation_key(request, workspace_name), [])[-MAX_MEMORY_MESSAGES:]


def _remember_turn(request: ChatRequest, workspace_name: str, answer: str) -> None:
    key = _conversation_key(request, workspace_name)
    messages = _conversation_memory.setdefault(key, [])
    messages.extend(
        [
            {"role": "user", "content": request.question},
            {"role": "assistant", "content": answer},
        ]
    )
    _conversation_memory[key] = messages[-MAX_MEMORY_MESSAGES:]


def ask_question(request: ChatRequest, request_id: str) -> dict:
    request_started_at = time.perf_counter()
    workspace_name = (
        request.workspace_id
        or request.workspace_name
        or WorkspaceManager.get_workspace()
    )

    try:
        logger.info(
            "Chat request received",
            extra={
                "event": "chat_request",
                "context": {
                    "request_id": request_id,
                    "question_length": len(request.question),
                    "workspace_name": workspace_name,
                },
            },
        )

        conversation_history = _get_conversation_history(request, workspace_name)
        result = answer_question(
            request.question,
            conversation_history=conversation_history,
            workspace_name=workspace_name,
            request_id=request_id,
        )

        response_payload = {
            "success": True,
            "request_id": request_id,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "summary": result.get("summary"),
            "workspace": workspace_name,
            "rewritten_query": result.get("rewritten_query"),
            "query_was_rewritten": result.get("query_was_rewritten", False),
        }
        _remember_turn(request, workspace_name, response_payload["answer"])
        logger.info(
            "Chat response ready",
            extra={
                "event": "chat_response_ready",
                "context": {
                    "request_id": request_id,
                    "workspace_name": workspace_name,
                    "answer_length": len(response_payload["answer"]),
                    "source_count": len(response_payload["sources"]),
                    "duration_ms": round((time.perf_counter() - request_started_at) * 1000),
                },
            },
        )
        return response_payload
    except ValueError as exc:
        logger.warning(
            "Chat request failed validation",
            extra={
                "event": "validation_failure",
                "context": {
                    "request_id": request_id,
                    "error": str(exc),
                },
            },
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "Chat request failed",
            extra={
                "event": "api_failure",
                "context": {
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                },
            },
        )
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(exc)}",
        ) from exc
