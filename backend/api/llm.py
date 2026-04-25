import hashlib
import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from time import perf_counter
from typing import Callable, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import traceable
from prometheus_client import Counter, Histogram
from pydantic import BaseModel

from core.config import get_settings
from core.cache import CacheBackend, NullCacheBackend, build_cache_backend
from core.routing import ProviderConfig, RouteConfig, RoutingConfig, get_routing_config, get_routing_config_fingerprint


RouteClassifier = Callable[[Sequence[dict[str, str]]], dict[str, str] | str]
RouteInvoker = Callable[[RouteConfig, ProviderConfig, Sequence[dict[str, str]]], str]


ROUTED_REQUESTS = Counter(
    "llm_gateway_routed_requests_total",
    "Total LLM gateway requests by route and provider.",
    ["route", "provider", "model"],
)
ROUTED_LATENCY = Histogram(
    "llm_gateway_route_latency_seconds",
    "Latency for routed LLM requests.",
    ["route", "provider", "model"],
)
CLASSIFICATIONS = Counter(
    "llm_gateway_classifications_total",
    "Total classifier decisions emitted by the gateway.",
    ["route", "classifier_model"],
)
CACHE_OPERATIONS = Counter(
    "llm_gateway_cache_operations_total",
    "Cache operations performed by the LLM gateway.",
    ["cache", "result"],
)


class RouteDecision(BaseModel):
    route: str
    reason: str


@dataclass
class RoutedLLMResult:
    content: str
    route: str
    provider: str
    model: str
    classifier_model: str
    reason: str


def _normalize_messages(messages: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "").strip()
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _hash_payload(payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _to_langchain_messages(messages: Sequence[dict[str, str]]):
    converted = []
    for message in _normalize_messages(messages):
        role = message["role"]
        content = message["content"]
        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    return converted


def _flatten_messages(messages: Sequence[dict[str, str]]) -> str:
    lines = []
    for message in _normalize_messages(messages):
        lines.append(f"{message['role']}: {message['content']}")
    return "\n".join(lines)


def _strip_markdown_fence(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return cleaned


def _extract_route_from_output(raw_output: str, allowed_routes: set[str]) -> str | None:
    cleaned = _strip_markdown_fence(raw_output)
    if not cleaned:
        return None

    # Prefer JSON route extraction when the model returns JSON-like content.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            route_value = parsed.get("route")
            if isinstance(route_value, str) and route_value in allowed_routes:
                return route_value
    except Exception:
        pass

    # Common fallback: model returns just the route token or a sentence containing it.
    for route_name in sorted(allowed_routes, key=len, reverse=True):
        if cleaned == route_name:
            return route_name
        if re.search(rf"\b{re.escape(route_name)}\b", cleaned):
            return route_name
    return None


@lru_cache(maxsize=32)
def _build_chat_model(
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
) -> ChatOpenAI:
    return ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=api_base,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout_seconds,
    )


class LLMGateway:
    def __init__(
        self,
        settings=None,
        routing_config: RoutingConfig | None = None,
        classifier: RouteClassifier | None = None,
        invoker: RouteInvoker | None = None,
        cache_backend: CacheBackend | None = None,
    ):
        self.settings = settings or get_settings()
        self.routing_config = routing_config or get_routing_config()
        self._classifier_override = classifier
        self._invoker_override = invoker
        self._cache_backend = cache_backend or build_cache_backend(self.settings)
        self._routing_fingerprint = get_routing_config_fingerprint(self.routing_config)

    def _get_model(self, provider: ProviderConfig, model: str, temperature: float, max_tokens: int) -> ChatOpenAI:
        return _build_chat_model(
            api_base=provider.api_base,
            api_key=provider.api_key or self.settings.openai_api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=self.settings.request_timeout_seconds,
        )

    def _normalize_decision(self, decision: dict[str, str] | str) -> RouteDecision:
        if isinstance(decision, str):
            return RouteDecision(route=decision, reason="Provided by classifier override")
        return RouteDecision.model_validate(decision)

    def _cache_enabled(self, feature_enabled: bool) -> bool:
        return feature_enabled and not isinstance(self._cache_backend, NullCacheBackend)

    def _cache_get(self, cache_name: str, key: str) -> str | None:
        try:
            cached_value = self._cache_backend.get(key)
        except Exception:
            CACHE_OPERATIONS.labels(cache_name, "error").inc()
            return None

        if cached_value is None:
            CACHE_OPERATIONS.labels(cache_name, "miss").inc()
            return None

        CACHE_OPERATIONS.labels(cache_name, "hit").inc()
        return cached_value

    def _cache_set(self, cache_name: str, key: str, value: str) -> None:
        try:
            self._cache_backend.set(key, value)
        except Exception:
            CACHE_OPERATIONS.labels(cache_name, "error").inc()
            return
        CACHE_OPERATIONS.labels(cache_name, "store").inc()

    def _eligible_routes(self) -> dict[str, RouteConfig]:
        eligible_routes = {
            route_name: route_config
            for route_name, route_config in self.routing_config.routes.items()
            if route_config.classifier_enabled
        }
        if eligible_routes:
            return eligible_routes
        return self.routing_config.routes

    def _response_cache_key(self, messages: Sequence[dict[str, str]], requested_route: str | None) -> str:
        return f"response:{_hash_payload({
            'namespace': self.settings.cache_namespace,
            'routing_fingerprint': self._routing_fingerprint,
            'requested_route': requested_route or 'auto',
            'messages': _normalize_messages(messages),
        })}"

    def _classifier_cache_key(self, messages: Sequence[dict[str, str]]) -> str:
        classifier_config = self.routing_config.classifier
        eligible_routes = self._eligible_routes()
        return f"classifier:{_hash_payload({
            'namespace': self.settings.cache_namespace,
            'routing_fingerprint': self._routing_fingerprint,
            'messages': _normalize_messages(messages),
            'classifier': classifier_config.model_dump(mode='json'),
            'eligible_routes': {
                route_name: route_config.description for route_name, route_config in eligible_routes.items()
            },
        })}"

    def _record_request_metrics(self, result: RoutedLLMResult, used_classifier: bool, duration: float) -> None:
        if used_classifier:
            CLASSIFICATIONS.labels(result.route, self.routing_config.classifier.model).inc()
        ROUTED_REQUESTS.labels(result.route, result.provider, result.model).inc()
        ROUTED_LATENCY.labels(result.route, result.provider, result.model).observe(duration)

    @traceable(name="gateway_classify", run_type="chain")
    def classify(self, messages: Sequence[dict[str, str]]) -> RouteDecision:
        if self._classifier_override is not None:
            return self._normalize_decision(self._classifier_override(messages))

        parser = PydanticOutputParser(pydantic_object=RouteDecision)
        classifier_config = self.routing_config.classifier
        provider = self.routing_config.providers[classifier_config.provider]
        eligible_routes = self._eligible_routes()
        route_names = set(eligible_routes.keys())
        route_definitions = "\n".join(
            f"- {route_name}: {route_config.description}"
            for route_name, route_config in eligible_routes.items()
        )
        chat_model = self._get_model(
            provider=provider,
            model=classifier_config.model,
            temperature=classifier_config.temperature,
            max_tokens=classifier_config.max_tokens,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", classifier_config.instructions),
                (
                    "human",
                    "Available routes:\n{route_definitions}\n\nConversation:\n{conversation}\n\n{format_instructions}",
                ),
            ]
        )
        chain = prompt | chat_model | parser

        try:
            decision = chain.invoke(
                {
                    "route_definitions": route_definitions,
                    "conversation": _flatten_messages(messages),
                    "format_instructions": parser.get_format_instructions(),
                }
            )
        except Exception as exc:
            raw_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", classifier_config.instructions),
                    (
                        "human",
                        "Available routes:\n{route_definitions}\n\nConversation:\n{conversation}\n\n"
                        "Return ONLY one route name from this list: {route_list}. "
                        "No JSON, no explanation.",
                    ),
                ]
            )
            try:
                raw_response = (raw_prompt | chat_model).invoke(
                    {
                        "route_definitions": route_definitions,
                        "conversation": _flatten_messages(messages),
                        "route_list": ", ".join(sorted(route_names)),
                    }
                )
                parsed_route = _extract_route_from_output(str(raw_response.content), route_names)
                if parsed_route is not None:
                    return RouteDecision(
                        route=parsed_route,
                        reason=f"Classifier fallback recovered from {exc.__class__.__name__}",
                    )
            except Exception:
                pass

            return RouteDecision(
                route=self.routing_config.default_route,
                reason=f"Classifier fallback after error: {exc.__class__.__name__}",
            )

        normalized = RouteDecision.model_validate(decision)
        if normalized.route not in self.routing_config.routes:
            return RouteDecision(
                route=self.routing_config.default_route,
                reason=f"Classifier returned unknown route '{normalized.route}', default applied",
            )
        return normalized

    @traceable(name="gateway_invoke", run_type="llm")
    def invoke_route(
        self,
        route_config: RouteConfig,
        provider: ProviderConfig,
        messages: Sequence[dict[str, str]],
    ) -> str:
        if self._invoker_override is not None:
            return self._invoker_override(route_config, provider, messages)

        temperature = route_config.temperature if route_config.temperature is not None else provider.temperature
        max_tokens = route_config.max_tokens if route_config.max_tokens is not None else provider.max_tokens
        effective_messages = list(messages)
        if route_config.system_prompt:
            effective_messages = [{"role": "system", "content": route_config.system_prompt}, *effective_messages]

        response = self._get_model(
            provider=provider,
            model=route_config.model,
            temperature=temperature,
            max_tokens=max_tokens,
        ).invoke(_to_langchain_messages(effective_messages))

        if isinstance(response.content, str):
            return response.content
        return str(response.content)

    def route_chat(self, messages: Sequence[dict[str, str]], requested_route: str | None = None) -> RoutedLLMResult:
        if not messages:
            raise ValueError("At least one message is required")

        start_time = perf_counter()
        used_classifier = requested_route is None
        if self._cache_enabled(self.settings.response_cache_enabled):
            response_cache_key = self._response_cache_key(messages, requested_route)
            cached_result = self._cache_get("response", response_cache_key)
            if cached_result is not None:
                result = RoutedLLMResult(**json.loads(cached_result))
                duration = perf_counter() - start_time
                self._record_request_metrics(result, used_classifier, duration)
                return result
        else:
            response_cache_key = None

        if requested_route:
            decision = RouteDecision(route=requested_route, reason="Route forced by request")
        else:
            decision = None
            classifier_cache_key = None
            if self._cache_enabled(self.settings.classifier_cache_enabled):
                classifier_cache_key = self._classifier_cache_key(messages)
                cached_decision = self._cache_get("classifier", classifier_cache_key)
                if cached_decision is not None:
                    decision = RouteDecision.model_validate_json(cached_decision)

            if decision is None:
                decision = self.classify(messages)
                if (
                    classifier_cache_key is not None
                    and decision.route in self.routing_config.routes
                    and "fallback" not in decision.reason.lower()
                    and "default applied" not in decision.reason.lower()
                ):
                    self._cache_set("classifier", classifier_cache_key, decision.model_dump_json())

        route_name = decision.route
        if route_name not in self.routing_config.routes:
            route_name = self.routing_config.default_route
            decision = RouteDecision(
                route=route_name,
                reason=f"Unknown requested route '{decision.route}', default applied",
            )

        route_config = self.routing_config.routes[route_name]
        provider = self.routing_config.providers[route_config.provider]
        content = self.invoke_route(route_config, provider, messages)
        result = RoutedLLMResult(
            content=content,
            route=route_name,
            provider=route_config.provider,
            model=route_config.model,
            classifier_model=self.routing_config.classifier.model,
            reason=decision.reason,
        )
        if response_cache_key is not None:
            self._cache_set("response", response_cache_key, json.dumps(asdict(result), sort_keys=True))

        duration = perf_counter() - start_time
        self._record_request_metrics(result, used_classifier, duration)
        return result


@lru_cache()
def get_gateway() -> LLMGateway:
    return LLMGateway()
