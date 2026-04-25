"""
Debug script to test classifier routing with various Vietnamese medical questions.

This script calls the classifier directly to see how it routes different questions.

Usage:
    python backend/debug_classifier.py

Required environment variables:
    VLLM_HOST (default: localhost)
    VLLM_PORT (default: 8000)
    OPENAI_API_KEY (default: test-key)
"""

import os
import sys
from api.llm import LLMGateway
from core.routing import get_routing_config

# Test cases: (question, expected_route, domain_description)
TEST_CASES = [
    # Vietnamese medical questions
    ("đau bụng nên xử lý như thế nào", "medical_qa", "stomach pain - should handle as medical"),
    ("Em bé tôi bị sốt cao", "medical_qa", "high fever - Vietnamese medical symptom"),
    ("Tôi bị ho liên tục, cần uống thuốc gì?", "medical_qa", "persistent cough - medication question"),
    ("Bệnh tiểu đường là gì?", "medical_qa", "diabetes question"),
    ("Cách điều trị viêm phổi", "medical_qa", "pneumonia treatment"),
    
    # Math questions
    ("Giải phương trình: 2x + 3 = 7", "math_qa", "solve equation - math problem"),
    ("Tính 15% của 200", "math_qa", "percentage calculation"),
    ("2 + 2 = ?", "math_qa", "simple arithmetic"),
    
    # General questions
    ("Thủ đô của Việt Nam là gì?", "chat", "geography question"),
    ("Hôm nay thời tiết thế nào?", "chat", "weather chat"),
]


def main():
    print("=" * 80)
    print("CLASSIFIER ROUTING DEBUG TEST")
    print("=" * 80)
    print()
    
    try:
        routing_config = get_routing_config()
        print(f"✓ Loaded routing config successfully")
        print(f"  - Classifier model: {routing_config.classifier.model}")
        print(f"  - Routes: {list(routing_config.routes.keys())}")
        print()
    except Exception as e:
        print(f"✗ Failed to load routing config: {e}")
        return 1
    
    try:
        gateway = LLMGateway()
        print(f"✓ Initialized LLM Gateway")
        print()
    except Exception as e:
        print(f"✗ Failed to initialize gateway: {e}")
        print(f"  Make sure VLLM_HOST, VLLM_PORT, and OPENAI_API_KEY are set")
        return 1
    
    print("Testing classifier routing on test cases:")
    print("-" * 80)
    
    passed = 0
    failed = 0
    
    for question, expected_route, description in TEST_CASES:
        try:
            messages = [{"role": "user", "content": question}]
            decision = gateway.classify(messages)
            
            status = "✓ PASS" if decision.route == expected_route else "✗ FAIL"
            if decision.route == expected_route:
                passed += 1
            else:
                failed += 1
            
            print(f"{status}: {description}")
            print(f"  Q: {question[:60]}{'...' if len(question) > 60 else ''}")
            print(f"  Expected: {expected_route} | Got: {decision.route} | Reason: {decision.reason}")
            print()
        except Exception as e:
            print(f"✗ ERROR: {description}")
            print(f"  Q: {question[:60]}{'...' if len(question) > 60 else ''}")
            print(f"  Exception: {e}")
            print()
            failed += 1
    
    print("-" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(TEST_CASES)} tests")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
