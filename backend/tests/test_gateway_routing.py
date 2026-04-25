from collections import Counter
from types import SimpleNamespace
import unittest

from api.llm import LLMGateway
from core.cache import InMemoryCacheBackend
from core.routing import RoutingConfig


TEST_ROUTING_CONFIG = RoutingConfig.model_validate(
    {
        "default_route": "medical_qa",
        "providers": {
            "vllm": {
                "api_base": "http://vllm:8000/v1",
                "api_key": "test-key",
                "temperature": 0,
                "max_tokens": 256,
            },
            "trtllm": {
                "api_base": "http://trtllm:8000/v1",
                "api_key": "test-key",
                "temperature": 0,
                "max_tokens": 256,
            },
        },
        "classifier": {
            "provider": "vllm",
            "model": "routing-classifier",
            "instructions": "route requests",
            "temperature": 0,
            "max_tokens": 64,
        },
        "routes": {
            "math_qa": {
                "provider": "trtllm",
                "model": "mathqa-lora",
                "description": "math questions",
                "system_prompt": "solve math",
            },
            "medical_qa": {
                "provider": "vllm",
                "model": "vi-medqa-lora",
                "description": "medical questions",
                "system_prompt": "answer medicine",
            },
        },
    }
)


class LLMGatewayRoutingTest(unittest.TestCase):
    @staticmethod
    def build_settings(**overrides):
        values = {
            "openai_api_key": "test-key",
            "request_timeout_seconds": 30,
            "cache_enabled": True,
            "cache_namespace": "tests-v1",
            "classifier_cache_enabled": True,
            "response_cache_enabled": True,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_forced_route_bypasses_classifier(self):
        gateway = LLMGateway(
            settings=self.build_settings(),
            routing_config=TEST_ROUTING_CONFIG,
            classifier=lambda _messages: self.fail("classifier should not be called"),
            invoker=lambda route, _provider, _messages: f"served-by:{route.model}",
            cache_backend=InMemoryCacheBackend(),
        )

        result = gateway.route_chat(
            messages=[{"role": "user", "content": "2 + 2 = ?"}],
            requested_route="math_qa",
        )

        self.assertEqual(result.route, "math_qa")
        self.assertEqual(result.provider, "trtllm")
        self.assertEqual(result.model, "mathqa-lora")
        self.assertEqual(result.content, "served-by:mathqa-lora")

    def test_classifier_selects_medical_route(self):
        gateway = LLMGateway(
            settings=self.build_settings(),
            routing_config=TEST_ROUTING_CONFIG,
            classifier=lambda _messages: {"route": "medical_qa", "reason": "medical domain"},
            invoker=lambda route, _provider, _messages: f"served-by:{route.model}",
            cache_backend=InMemoryCacheBackend(),
        )

        result = gateway.route_chat(messages=[{"role": "user", "content": "What is the treatment for asthma?"}])

        self.assertEqual(result.route, "medical_qa")
        self.assertEqual(result.provider, "vllm")
        self.assertEqual(result.model, "vi-medqa-lora")
        self.assertEqual(result.reason, "medical domain")

    def test_unknown_route_falls_back_to_default(self):
        gateway = LLMGateway(
            settings=self.build_settings(),
            routing_config=TEST_ROUTING_CONFIG,
            classifier=lambda _messages: {"route": "not-a-route", "reason": "bad classifier output"},
            invoker=lambda route, _provider, _messages: f"served-by:{route.model}",
            cache_backend=InMemoryCacheBackend(),
        )

        result = gateway.route_chat(messages=[{"role": "user", "content": "Unclear question"}])

        self.assertEqual(result.route, "medical_qa")
        self.assertEqual(result.provider, "vllm")
        self.assertIn("default applied", result.reason)

    def test_response_cache_bypasses_classifier_and_invoker_on_repeat(self):
        call_counts = Counter()

        gateway = LLMGateway(
            settings=self.build_settings(),
            routing_config=TEST_ROUTING_CONFIG,
            classifier=lambda _messages: call_counts.update(["classifier"]) or {
                "route": "medical_qa",
                "reason": "medical domain",
            },
            invoker=lambda route, _provider, _messages: call_counts.update(["invoker"]) or f"served-by:{route.model}",
            cache_backend=InMemoryCacheBackend(),
        )

        messages = [{"role": "user", "content": "What is the treatment for asthma?"}]
        first = gateway.route_chat(messages=messages)
        second = gateway.route_chat(messages=messages)

        self.assertEqual(first.content, second.content)
        self.assertEqual(call_counts["classifier"], 1)
        self.assertEqual(call_counts["invoker"], 1)

    def test_classifier_cache_bypasses_repeat_classification_when_response_cache_disabled(self):
        call_counts = Counter()

        gateway = LLMGateway(
            settings=self.build_settings(response_cache_enabled=False),
            routing_config=TEST_ROUTING_CONFIG,
            classifier=lambda _messages: call_counts.update(["classifier"]) or {
                "route": "medical_qa",
                "reason": "medical domain",
            },
            invoker=lambda route, _provider, _messages: call_counts.update(["invoker"]) or f"served-by:{route.model}",
            cache_backend=InMemoryCacheBackend(),
        )

        messages = [{"role": "user", "content": "What is the treatment for asthma?"}]
        gateway.route_chat(messages=messages)
        gateway.route_chat(messages=messages)

        self.assertEqual(call_counts["classifier"], 1)
        self.assertEqual(call_counts["invoker"], 2)

    def test_cache_namespace_change_invalidates_shared_response_cache(self):
        shared_cache = InMemoryCacheBackend()
        call_counts = Counter()
        messages = [{"role": "user", "content": "What is the treatment for asthma?"}]

        first_gateway = LLMGateway(
            settings=self.build_settings(cache_namespace="tests-v1"),
            routing_config=TEST_ROUTING_CONFIG,
            classifier=lambda _messages: call_counts.update(["classifier"]) or {
                "route": "medical_qa",
                "reason": "medical domain",
            },
            invoker=lambda route, _provider, _messages: call_counts.update(["invoker"]) or f"served-by:{route.model}",
            cache_backend=shared_cache,
        )
        second_gateway = LLMGateway(
            settings=self.build_settings(cache_namespace="tests-v2"),
            routing_config=TEST_ROUTING_CONFIG,
            classifier=lambda _messages: call_counts.update(["classifier"]) or {
                "route": "medical_qa",
                "reason": "medical domain",
            },
            invoker=lambda route, _provider, _messages: call_counts.update(["invoker"]) or f"served-by:{route.model}",
            cache_backend=shared_cache,
        )

        first_gateway.route_chat(messages=messages)
        second_gateway.route_chat(messages=messages)

        self.assertEqual(call_counts["classifier"], 2)
        self.assertEqual(call_counts["invoker"], 2)


if __name__ == "__main__":
    unittest.main()