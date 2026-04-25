import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, model_validator

from core.config import get_settings


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


class ProviderConfig(BaseModel):
    api_base: str
    api_key: str | None = None
    temperature: float = 0
    max_tokens: int = 256


class RouteConfig(BaseModel):
    provider: str
    model: str
    description: str
    system_prompt: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    classifier_enabled: bool = True


class ClassifierConfig(BaseModel):
    provider: str
    model: str
    instructions: str
    temperature: float = 0
    max_tokens: int = 64


class RoutingConfig(BaseModel):
    default_route: str
    providers: dict[str, ProviderConfig]
    classifier: ClassifierConfig
    routes: dict[str, RouteConfig]

    @model_validator(mode="after")
    def validate_references(self) -> "RoutingConfig":
        if self.default_route not in self.routes:
            raise ValueError(f"Unknown default route: {self.default_route}")

        provider_names = set(self.providers.keys())
        missing = []
        if self.classifier.provider not in provider_names:
            missing.append(self.classifier.provider)

        for route_name, route_config in self.routes.items():
            if route_config.provider not in provider_names:
                missing.append(f"{route_name}:{route_config.provider}")

        if missing:
            raise ValueError(f"Routing config references unknown providers: {', '.join(missing)}")
        return self


def load_routing_config(config_path: str | Path) -> RoutingConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    expanded_config = _expand_env(raw_config)
    return RoutingConfig.model_validate(expanded_config)


def get_routing_config_fingerprint(routing_config: RoutingConfig) -> str:
    payload = json.dumps(
        routing_config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@lru_cache()
def get_routing_config() -> RoutingConfig:
    settings = get_settings()
    return load_routing_config(settings.routing_config_path)