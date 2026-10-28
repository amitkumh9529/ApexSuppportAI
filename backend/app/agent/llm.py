"""Provider-agnostic chat model factory for ApexSupport-AI.

Returns a LangChain chat model for the configured provider. Clients are cached
per process so HTTP connections to the provider are pooled and reused across
requests, with explicit timeouts and retries.

`LLM_PROVIDER=fake` returns a scripted tool-calling model with realistic
latency — used by load tests and local zero-token verification.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings

FAKE_MODEL_NAME = "scripted-fake"


def _build(provider: str, max_retries: int | None = None):
    settings = get_settings()
    provider = provider.lower()
    retries = settings.llm_max_retries if max_retries is None else max_retries

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.gemini_api_key:
            raise RuntimeError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set.")
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            max_tokens=settings.max_output_tokens,
            timeout=settings.llm_timeout_s,
            max_retries=retries,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.anthropic_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set."
            )
        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
            max_tokens=settings.max_output_tokens,
            timeout=settings.llm_timeout_s,
            max_retries=retries,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set.")
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
            max_tokens=settings.max_output_tokens,
            timeout=settings.llm_timeout_s,
            max_retries=retries,
        )

    if provider == "omniroute":
        import httpx
        from langchain_openai import ChatOpenAI

        if not (settings.omniroute_api_key and settings.omniroute_base_url):
            raise RuntimeError(
                "LLM_PROVIDER=omniroute needs OMNIROUTE_BASE_URL and OMNIROUTE_API_KEY."
            )
        return ChatOpenAI(
            model=settings.omniroute_model,
            base_url=settings.omniroute_base_url,
            api_key=settings.omniroute_api_key,
            max_tokens=settings.max_output_tokens,
            timeout=httpx.Timeout(settings.llm_timeout_s, connect=settings.llm_connect_timeout_s),
            max_retries=retries,
        )

    if provider == "fake":
        from app.agent.llm_fake import ScriptedFakeChatModel

        return ScriptedFakeChatModel(
            latency_ms=settings.fake_llm_latency_ms,
            error_rate=settings.fake_llm_error_rate,
        )

    raise RuntimeError(
        f"Unknown LLM provider '{provider}'. "
        "Use 'gemini', 'anthropic', 'openai', 'omniroute' or 'fake'."
    )


@lru_cache
def get_model(provider: str, max_retries: int | None = None):
    return _build(provider, max_retries)


def get_chat_model():
    settings = get_settings()
    if get_fallback_model() is not None:
        return get_model(settings.llm_provider, min(settings.llm_max_retries, 1))
    return get_model(settings.llm_provider)


@lru_cache
def get_fallback_model():
    settings = get_settings()
    fallback = settings.llm_fallback_provider
    if not fallback or fallback == settings.llm_provider:
        return None
    return get_model(fallback)


def primary_model_name() -> str:
    settings = get_settings()
    return {
        "gemini": settings.gemini_model,
        "anthropic": settings.anthropic_model,
        "openai": settings.openai_model,
        "omniroute": settings.omniroute_model.rsplit("/", 1)[-1],
        "fake": FAKE_MODEL_NAME,
    }.get(settings.llm_provider, "")


def use_prompt_caching() -> bool:
    settings = get_settings()
    return settings.llm_provider == "anthropic" and settings.llm_fallback_provider in ("", "fake")


def model_info() -> dict:
    settings = get_settings()
    return {
        "provider": settings.llm_provider,
        "model": primary_model_name(),
        "fallback_provider": settings.llm_fallback_provider or None,
    }
