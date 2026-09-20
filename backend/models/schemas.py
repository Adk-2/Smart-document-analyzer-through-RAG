from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field, field_validator


class WorkspaceRequest(BaseModel):
    workspace_id: str = ""
    workspace_name: str | None = None

    @field_validator("workspace_id")
    @classmethod
    def sanitize_workspace_id(cls, value: str) -> str:
        return (value or "").strip()

    @field_validator("workspace_name")
    @classmethod
    def sanitize_workspace_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        sanitized_value = value.strip()
        return sanitized_value or None


class UrlRequest(WorkspaceRequest):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        sanitized_url = (value or "").strip()
        if not sanitized_url:
            raise ValueError("Invalid URL")
        if any(char.isspace() for char in sanitized_url):
            raise ValueError("Invalid URL")

        parsed = urlsplit(sanitized_url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Invalid URL")

        return urlunsplit(
            (
                parsed.scheme.lower(),
                parsed.netloc,
                parsed.path,
                parsed.query,
                parsed.fragment,
            )
        )


class IngestURLRequest(UrlRequest):
    pass


class ChatRequest(WorkspaceRequest):
    question: str
    conversation_id: str | None = None
    history: list[dict[str, str]] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        sanitized_question = (value or "").strip()
        if not sanitized_question:
            raise ValueError("Question is required")
        return sanitized_question

    @field_validator("conversation_id")
    @classmethod
    def sanitize_conversation_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        sanitized_value = value.strip()
        return sanitized_value[:128] or None

    @field_validator("history")
    @classmethod
    def sanitize_history(cls, value: list[dict[str, str]] | None) -> list[dict[str, str]]:
        sanitized_history = []
        for item in value or []:
            role = (item.get("role") or "").strip().lower()
            content = (item.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            sanitized_history.append({
                "role": role,
                "content": content[:2000],
            })
        return sanitized_history[-10:]
