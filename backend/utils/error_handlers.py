from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.utils.logging import get_logger


logger = get_logger(__name__)


def _build_error_response(status_code: int, error_message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": error_message,
        },
    )


def _extract_validation_message(exc: RequestValidationError) -> str:
    if not exc.errors():
        return "Invalid request"

    error = exc.errors()[0]
    location = error.get("loc", [])
    if "url" in location and error.get("type") == "missing":
        return "URL is required"

    message = error.get("msg", "Invalid request")
    if message.startswith("Value error, "):
        return message.removeprefix("Value error, ")
    return message


def _get_request_id(request: Request) -> str | None:
    return request.headers.get("x-request-id")


async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    error_message = _extract_validation_message(exc)
    logger.warning(
        "Request validation failed",
        extra={
            "event": "validation_failure",
            "context": {
                "request_id": _get_request_id(request),
                "method": request.method,
                "path": request.url.path,
                "errors": exc.errors(),
            },
        },
    )
    return _build_error_response(422, error_message)


async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    logger.warning(
        "HTTP exception raised",
        extra={
            "event": "api_failure",
            "context": {
                "request_id": _get_request_id(request),
                "method": request.method,
                "path": request.url.path,
                "status_code": exc.status_code,
                "detail": exc.detail,
            },
        },
    )
    return _build_error_response(exc.status_code, str(exc.detail))


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.exception(
        "Unhandled API exception",
        extra={
            "event": "api_failure",
            "context": {
                "request_id": _get_request_id(request),
                "method": request.method,
                "path": request.url.path,
                "error_type": type(exc).__name__,
            },
        },
    )
    return _build_error_response(500, "Internal server error")


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
