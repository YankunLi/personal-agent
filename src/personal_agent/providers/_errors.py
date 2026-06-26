"""Shared provider utilities."""

from __future__ import annotations

from personal_agent.exceptions import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)


def raise_provider_error(error: Exception) -> None:
    """Classify and re-raise provider errors.

    Shared across all provider implementations to avoid code duplication.
    """
    error_str = str(error).lower()
    if "401" in error_str or "unauthorized" in error_str or "invalid api key" in error_str or "authentication error" in error_str:
        raise ProviderAuthError(str(error)) from error
    if "429" in error_str or "rate limit" in error_str:
        raise ProviderRateLimitError(str(error)) from error
    if "timeout" in error_str or "timed out" in error_str:
        raise ProviderTimeoutError(str(error)) from error
    raise ProviderError(str(error)) from error