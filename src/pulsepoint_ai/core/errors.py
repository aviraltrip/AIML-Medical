from __future__ import annotations


class PulsePointError(Exception):
    """Base for all custom errors. Always carries a stable `code` for clients."""

    code: str = "internal_error"
    http_status: int = 500

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(PulsePointError):
    code = "validation_error"
    http_status = 400


class AuthError(PulsePointError):
    code = "auth_error"
    http_status = 401


class RateLimitError(PulsePointError):
    code = "rate_limited"
    http_status = 429


class UpstreamError(PulsePointError):
    """Raised when an external API (LLM/STT/embeddings) fails."""

    code = "upstream_error"
    http_status = 502


class ModelNotLoadedError(PulsePointError):
    code = "model_not_loaded"
    http_status = 503


class HallucinationDetected(PulsePointError):
    """Raised internally when guard blocks output. NOT returned to clients —
    the cleaned output is returned instead, with the hallucination logged."""

    code = "hallucination_detected"
    http_status = 200
