from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent

# Only ever used when AUTH_MODE=dev. Startup refuses to run in prod with it.
DEV_JWT_SECRET = "apex-dev-insecure-secret-key-32chars-min-prod"


def _async_db_url(url: str) -> str:
    """Normalise a DB URL to its async driver (asyncpg / aiosqlite)."""
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        return _asyncpg_query("postgresql+asyncpg://" + url[len("postgresql://") :])
    if url.startswith("sqlite:///"):
        return "sqlite+aiosqlite:///" + url[len("sqlite:///") :]
    return url


def _asyncpg_query(url: str) -> str:
    """Translate libpq query params (as in Neon/Supabase URLs) for asyncpg."""
    parts = urlsplit(url)
    if not parts.query:
        return url
    params = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key == "sslmode":
            params.append(("ssl", value))
        elif key not in ("channel_binding", "options"):
            params.append((key, value))
    return urlunsplit(parts._replace(query=urlencode(params)))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "ApexSupport AI"
    app_version: str = "1.0.0"

    # "dev" enables the dev-token endpoint, in-process Redis fallback and
    # permissive defaults. "prod" requires real secrets, Redis and Postgres.
    env: str = "dev"

    # LLM provider: "gemini", "anthropic", "openai", "omniroute" (OpenAI-compatible) or "fake"
    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    omniroute_base_url: str = ""
    omniroute_api_key: str = ""
    omniroute_model: str = "cx/gpt-5.6-sol-medium"
    omniroute_judge_model: str = "cx/gpt-5.6-luna-low"

    # Provider failover & resilience
    llm_connect_timeout_s: float = 5.0
    llm_fallback_cooldown_s: float = 30.0
    llm_fallback_provider: str = ""
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2

    # Fake provider knobs (LLM_PROVIDER=fake)
    fake_llm_latency_ms: int = 200
    fake_llm_error_rate: float = 0.0

    # Storage: Default to SQLite with WAL mode for zero-docker local dev
    database_url: str = f"sqlite:///{BACKEND_ROOT / 'var' / 'apexsupport.db'}"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout_s: float = 10.0
    seed_on_startup: bool = False

    # Redis. Empty in dev/tests -> in-process fakeredis
    redis_url: str = ""

    # Auth
    auth_mode: str = "dev"
    jwt_secret: str = ""
    jwt_jwks_url: str = ""
    jwt_issuer: str = ""
    jwt_audience: str = ""
    dev_token_ttl_s: int = 8 * 3600

    # Demo mode
    demo_session_rate_limit: str = "5/hour"
    demo_token_ttl_s: int = 2 * 3600
    demo_data_ttl_hours: int = 24
    demo_global_daily_token_budget: int = 0
    turnstile_secret: str = ""

    # HTTP & Security
    cors_origins: str = ""
    max_request_bytes: int = 16_384
    trusted_proxy_hops: int = 0

    # Refund policy deterministic parameters
    return_window_days: int = 30
    escalation_threshold: float = 500.0

    # Guardrails
    max_message_chars: int = 2000
    max_conversation_turns: int = 30
    max_conversation_tokens: int = 60000
    customer_daily_token_budget: int = 300_000
    agent_recursion_limit: int = 12
    max_output_tokens: int = 1024
    chat_rate_limit: str = "15/minute"
    max_concurrent_turns: int = 200
    turn_timeout_s: float = 120.0

    # Sentiment & Proactive Escalation Upgrade
    sentiment_analysis_enabled: bool = True
    sentiment_frustration_threshold: float = -0.5

    # Outbound Webhook Dispatcher Upgrade
    webhook_url: str = ""
    webhook_secret: str = ""
    webhook_timeout_s: float = 5.0

    # Data retention
    retention_days: int = 90

    # Observability
    log_level: str = "INFO"
    log_json: bool = True
    sentry_dsn: str = ""
    metrics_token: str = ""
    opik_url_override: str = ""
    opik_api_key: str = ""
    opik_workspace: str = ""
    opik_project_name: str = "apexsupport-ai"

    @model_validator(mode="after")
    def _check_prod(self) -> "Settings":
        if self.env == "prod":
            problems = []
            if self.auth_mode not in ("jwt", "demo"):
                problems.append("AUTH_MODE must be 'jwt' or 'demo'")
            if not (self.jwt_jwks_url or self.jwt_secret):
                problems.append("JWT_SECRET or JWT_JWKS_URL is required")
            if self.auth_mode == "demo" and (self.jwt_jwks_url or len(self.jwt_secret) < 32):
                problems.append("AUTH_MODE=demo needs JWT_SECRET (>= 32 chars) and no JWKS")
            if self.jwt_secret == DEV_JWT_SECRET:
                problems.append("JWT_SECRET must not be the dev secret")
            if not self.redis_url:
                problems.append("REDIS_URL is required")
            if self.async_database_url.startswith("sqlite"):
                problems.append("DATABASE_URL must point at Postgres in production")
            if problems:
                raise ValueError("invalid production config: " + "; ".join(problems))
        return self

    @property
    def async_database_url(self) -> str:
        return _async_db_url(self.database_url)

    @property
    def effective_jwt_secret(self) -> str:
        if self.jwt_secret:
            return self.jwt_secret
        return DEV_JWT_SECRET if self.auth_mode in ("dev", "demo") else ""

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def opik_enabled(self) -> bool:
        return bool(self.opik_url_override or self.opik_api_key)

    @property
    def is_demo(self) -> bool:
        return self.auth_mode == "demo"

    @property
    def has_llm_key(self) -> bool:
        if self.llm_provider == "gemini":
            return bool(self.gemini_api_key)
        if self.llm_provider == "anthropic":
            return bool(self.anthropic_api_key)
        if self.llm_provider == "openai":
            return bool(self.openai_api_key)
        if self.llm_provider == "omniroute":
            return bool(self.omniroute_api_key and self.omniroute_base_url)
        return self.llm_provider == "fake"


@lru_cache
def get_settings() -> Settings:
    return Settings()
