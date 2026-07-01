"""Shared provider utilities."""

from __future__ import annotations

import re

from personal_agent.exceptions import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

# Patterns that may carry credentials in raw provider/SDK error strings.
_REDACT_PATTERNS = [
    # access_token / token query params (e.g. Baidu Qianfan puts it in the URL)
    # \b prevents matching "key" inside words like "monkey=banana" -> "mon***REDACTED***"
    re.compile(r"\b(access_token|token|api_key|apikey|key)=([^&\s]+)", re.IGNORECASE),
    # Authorization headers
    re.compile(r"(authorization)\s*[:=]\s*(bearer\s+)?(\S+)", re.IGNORECASE),
    # Password in URLs: scheme://user:pass@host
    re.compile(r"(://[^/\s:@]+:)([^@\s]+)(@)"),
]


def _sanitize(error: Exception) -> str:
    """Return a sanitized string representation of ``error``.

    Strips bearer tokens, query-string credentials, and Authorization headers
    so they cannot leak into agent conversation history or user-facing messages
    via ProviderError messages.
    """
    text = str(error)
    # access_token / token / api_key query params (Baidu Qianfan puts the OAuth
    # access_token in the request URL, which httpx may include in error text).
    text = _REDACT_PATTERNS[0].sub(r"\1=***REDACTED***", text)
    # Authorization: Bearer <key> headers
    text = _REDACT_PATTERNS[1].sub(r"\1: ***REDACTED***", text)
    # Password://user:pass@host credentials
    text = _REDACT_PATTERNS[2].sub(r"\1***REDACTED***\3", text)
    return text


def raise_provider_error(error: Exception) -> None:
    """Classify and re-raise provider errors.

    Shared across all provider implementations to avoid code duplication.
    """
    # Pass through existing provider errors to avoid double-wrapping
    if isinstance(error, (ProviderAuthError, ProviderRateLimitError, ProviderTimeoutError, ProviderError)):
        raise

    error_str = str(error).lower()
    # Match HTTP status codes as standalone tokens (e.g., "401", "status 401",
    # "401 unauthorized") but NOT as substrings of larger numbers/words like
    # "4012" or "port 40193".
    if re.search(r"\b401\b", error_str) or "unauthorized" in error_str or "invalid api key" in error_str or "authentication error" in error_str:
        raise ProviderAuthError(_sanitize(error)) from error
    if re.search(r"\b429\b", error_str) or "rate limit" in error_str:
        raise ProviderRateLimitError(_sanitize(error)) from error
    if "timeout" in error_str or "timed out" in error_str:
        raise ProviderTimeoutError(_sanitize(error)) from error
    raise ProviderError(_sanitize(error)) from error