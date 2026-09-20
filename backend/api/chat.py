import time
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from backend.models.schemas import ChatRequest
from backend.services.chat_service import ask_question
from backend.utils.logging import get_logger


router = APIRouter()
logger = get_logger(__name__)


@router.post("/chat")
def chat_endpoint(request: Request, chat_request: ChatRequest):
    request_started_at = time.perf_counter()
    request_id = request.headers.get("x-request-id") or f"api-{uuid4().hex}"
    logger.info(
        "Chat API request received",
        extra={
            "event": "chat_api_request_received",
            "context": {
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
            },
        },
    )
    payload = ask_question(chat_request, request_id=request_id)
    serialization_started_at = time.perf_counter()
    logger.info(
        "Chat API response serialization started",
        extra={
            "event": "chat_response_serialization_start",
            "context": {
                "request_id": request_id,
                "workspace_name": payload.get("workspace"),
            },
        },
    )
    encoded_payload = jsonable_encoder(payload)
    response = JSONResponse(content=encoded_payload)
    response.headers["X-Request-ID"] = request_id
    serialization_duration_ms = round((time.perf_counter() - serialization_started_at) * 1000)
    logger.info(
        "Chat API response serialization completed",
        extra={
            "event": "chat_response_serialization_end",
            "context": {
                "request_id": request_id,
                "workspace_name": payload.get("workspace"),
                "duration_ms": serialization_duration_ms,
                "response_bytes": len(response.body),
            },
        },
    )
    logger.info(
        "Chat API response send",
        extra={
            "event": "chat_response_send",
            "context": {
                "request_id": request_id,
                "workspace_name": payload.get("workspace"),
                "duration_ms": round((time.perf_counter() - request_started_at) * 1000),
            },
        },
    )
    return response
