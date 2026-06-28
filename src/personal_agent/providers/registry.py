"""Provider registry and factory."""

from __future__ import annotations

import logging

from personal_agent.config import ProviderCredentials, Settings
from personal_agent.exceptions import ConfigError
from personal_agent.providers.base import Provider
from personal_agent.providers.openai_compat import OpenAICompatibleProvider

logger = logging.getLogger(__name__)

# Pre-configured provider map: provider_name -> (class, default_base_url, default_model)
PROVIDER_REGISTRY: dict[str, dict] = {
    "openai": {
        "class": "openai_compat",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
    },
    "deepseek": {
        "class": "openai_compat",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
    },
    "qwen": {
        "class": "openai_compat",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
    },
    "zhipu": {
        "class": "openai_compat",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4-plus",
    },
    "hunyuan": {
        "class": "openai_compat",
        "base_url": "https://api.hunyuan.cloud.tencent.com/v1",
        "default_model": "hunyuan-pro",
    },
    "anthropic": {
        "class": "anthropic",
        "base_url": None,
        "default_model": "claude-sonnet-4-6",
    },
    "wenxin": {
        "class": "baidu",
        "base_url": None,
        "default_model": "ernie-4.0-turbo-128k",
    },
}


def create_provider(
    provider_name: str = "openai",
    model: str = "",
    api_key: str = "",
    base_url: str | None = None,
    timeout: float = 120.0,
    max_retries: int = 3,
    credentials: ProviderCredentials | None = None,
) -> Provider:
    """Create a provider instance.

    Args:
        provider_name: Provider key (openai, deepseek, qwen, etc.)
        model: Model name. If empty, uses the provider's default.
        api_key: API key.
        base_url: Override the default base URL.
        timeout: HTTP timeout in seconds.
        max_retries: Max HTTP retries.
        credentials: ProviderCredentials object (overrides api_key/base_url if provided).

    Returns:
        A Provider instance.
    """
    # Merge credentials if provided
    if credentials:
        api_key = credentials.api_key or api_key
        base_url = credentials.api_base or base_url
        timeout = credentials.timeout or timeout
        max_retries = credentials.max_retries or max_retries

    if provider_name not in PROVIDER_REGISTRY:
        raise ConfigError(
            f"Unknown provider '{provider_name}'. "
            f"Available: {list(PROVIDER_REGISTRY.keys())}"
        )

    meta = PROVIDER_REGISTRY[provider_name]

    if not model:
        model = meta["default_model"]

    if not base_url:
        base_url = meta.get("base_url")

    if not api_key:
        logger.warning(
            "No API key provided for provider '%s'. "
            "Set it via PA_PROVIDERS__%s__API_KEY or in your config file.",
            provider_name, provider_name.upper(),
        )

    if meta["class"] == "openai_compat":
        return OpenAICompatibleProvider(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    if meta["class"] == "anthropic":
        from personal_agent.providers.anthropic import AnthropicProvider

        return AnthropicProvider(
            model=model,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    if meta["class"] == "baidu":
        from personal_agent.providers.baidu import BaiduProvider

        return BaiduProvider(
            model=model,
            api_key=api_key,
            timeout=timeout,
        )

    raise ConfigError(f"Unknown provider class '{meta['class']}' for '{provider_name}'")


def create_provider_from_settings(settings: Settings) -> Provider:
    """Create a provider from the Settings object."""
    agent = settings.agent
    creds = settings.get_provider_credentials()
    return create_provider(
        provider_name=agent.provider,
        model=agent.model,
        credentials=creds,
    )


def register_provider(
    name: str,
    class_name: str,
    base_url: str | None = None,
    default_model: str = "",
) -> None:
    """Register a custom provider at runtime."""
    PROVIDER_REGISTRY[name] = {
        "class": class_name,
        "base_url": base_url,
        "default_model": default_model,
    }


def list_providers() -> list[str]:
    """Return a list of registered provider names."""
    return list(PROVIDER_REGISTRY.keys())