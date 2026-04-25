from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(default="aio2025", alias="OPENAI_API_KEY")
    vllm_api_base_url: str = Field(default="http://127.0.0.1:8000/v1", alias="VLLM_API_BASE_URL")
    tensorrt_llm_api_base_url: str = Field(
        default="http://127.0.0.1:8002/v1",
        alias="TENSORRT_LLM_API_BASE_URL",
    )
    routing_config_path: str = Field(
        default=str(ROOT_DIR / "configs" / "routing_config.yaml"),
        alias="ROUTING_CONFIG_PATH",
    )
    request_timeout_seconds: int = Field(default=120, alias="REQUEST_TIMEOUT_SECONDS")
    cache_enabled: bool = Field(default=False, alias="CACHE_ENABLED")
    cache_redis_url: str | None = Field(default=None, alias="CACHE_REDIS_URL")
    cache_namespace: str = Field(default="default", alias="CACHE_NAMESPACE")
    classifier_cache_enabled: bool = Field(default=True, alias="CLASSIFIER_CACHE_ENABLED")
    response_cache_enabled: bool = Field(default=True, alias="RESPONSE_CACHE_ENABLED")
    cache_ttl_seconds: int | None = Field(default=None, alias="CACHE_TTL_SECONDS")
    cache_socket_timeout_seconds: int = Field(default=2, alias="CACHE_SOCKET_TIMEOUT_SECONDS")
    app_title: str = "LLM Gateway Service"
    app_description: str = "Config-driven FastAPI gateway for routed LLM serving"
    version: str = "2.0.0"

    @property
    def api_key(self) -> str:
        return self.openai_api_key

    @property
    def api_base(self) -> str:
        return self.vllm_api_base_url


@lru_cache()
def get_settings() -> Settings:
    return Settings()
