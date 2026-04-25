"""
Test classifier routing for Vietnamese medical questions.

Usage:
    python -m pytest backend/tests/test_classifier_routing.py -v
"""

import unittest
from types import SimpleNamespace

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
            "max_tokens": 128,
        },
        "routes": {
            "math_qa": {
                "provider": "trtllm",
                "model": "mathqa-lora",
                "description": "Solve math questions, equations, and reasoning-heavy quantitative prompts.",
                "system_prompt": "solve math",
            },
            "medical_qa": {
                "provider": "vllm",
                "model": "vi-medqa-lora",
                "description": "Answer Vietnamese and English medical questions.",
                "system_prompt": "answer medicine",
            },
            "chat": {
                "provider": "trtllm",
                "model": "chat-sft",
                "description": "General conversational chat.",
                "system_prompt": "be helpful",
            },
        },
    }
)


class ClassifierRoutingTest(unittest.TestCase):
    @staticmethod
    def build_settings(**overrides):
        values = {
            "openai_api_key": "test-key",
            "request_timeout_seconds": 30,
            "cache_namespace": "tests-classifier-v1",
            "classifier_cache_enabled": False,
            "response_cache_enabled": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_vietnamese_stomach_pain_routes_to_medical(self):
        """Test that Vietnamese medical question 'đau bụng nên xử lý như thế nào' routes to medical_qa."""
        gateway = LLMGateway(
            settings=self.build_settings(),
            routing_config=TEST_ROUTING_CONFIG,
            classifier=lambda _messages: {"route": "medical_qa", "reason": "vietnamese medical symptom"},
            invoker=lambda route, _provider, _messages: f"served-by:{route.model}",
            cache_backend=InMemoryCacheBackend(),
        )

        result = gateway.route_chat(
            messages=[{"role": "user", "content": "đau bụng nên xử lý như thế nào"}]
        )

        self.assertEqual(result.route, "medical_qa", f"Expected medical_qa but got {result.route}")
        self.assertEqual(result.model, "vi-medqa-lora")

    def test_math_problem_routes_to_math_qa(self):
        """Test that math problems route to math_qa."""
        gateway = LLMGateway(
            settings=self.build_settings(),
            routing_config=TEST_ROUTING_CONFIG,
            classifier=lambda _messages: {"route": "math_qa", "reason": "math problem"},
            invoker=lambda route, _provider, _messages: f"served-by:{route.model}",
            cache_backend=InMemoryCacheBackend(),
        )

        result = gateway.route_chat(
            messages=[{"role": "user", "content": "Solve: 2x + 3 = 7"}]
        )

        self.assertEqual(result.route, "math_qa", f"Expected math_qa but got {result.route}")
        self.assertEqual(result.model, "mathqa-lora")

    def test_general_chat_routes_to_chat(self):
        """Test that general questions route to chat."""
        gateway = LLMGateway(
            settings=self.build_settings(),
            routing_config=TEST_ROUTING_CONFIG,
            classifier=lambda _messages: {"route": "chat", "reason": "general question"},
            invoker=lambda route, _provider, _messages: f"served-by:{route.model}",
            cache_backend=InMemoryCacheBackend(),
        )

        result = gateway.route_chat(
            messages=[{"role": "user", "content": "What is the capital of France?"}]
        )

        self.assertEqual(result.route, "chat", f"Expected chat but got {result.route}")
        self.assertEqual(result.model, "chat-sft")

    def test_vietnamese_fever_routes_to_medical(self):
        """Test Vietnamese symptom 'sốt cao' (high fever) routes to medical_qa."""
        gateway = LLMGateway(
            settings=self.build_settings(),
            routing_config=TEST_ROUTING_CONFIG,
            classifier=lambda _messages: {"route": "medical_qa", "reason": "vietnamese fever symptom"},
            invoker=lambda route, _provider, _messages: f"served-by:{route.model}",
            cache_backend=InMemoryCacheBackend(),
        )

        result = gateway.route_chat(
            messages=[{"role": "user", "content": "Em bé tôi bị sốt cao, tôi nên làm gì?"}]
        )

        self.assertEqual(result.route, "medical_qa", f"Expected medical_qa but got {result.route}")

    def test_vietnamese_cough_routes_to_medical(self):
        """Test Vietnamese symptom 'ho' (cough) routes to medical_qa."""
        gateway = LLMGateway(
            settings=self.build_settings(),
            routing_config=TEST_ROUTING_CONFIG,
            classifier=lambda _messages: {"route": "medical_qa", "reason": "vietnamese cough symptom"},
            invoker=lambda route, _provider, _messages: f"served-by:{route.model}",
            cache_backend=InMemoryCacheBackend(),
        )

        result = gateway.route_chat(
            messages=[{"role": "user", "content": "Tôi bị ho liên tục, cần uống thuốc gì?"}]
        )

        self.assertEqual(result.route, "medical_qa", f"Expected medical_qa but got {result.route}")


if __name__ == "__main__":
    unittest.main()
