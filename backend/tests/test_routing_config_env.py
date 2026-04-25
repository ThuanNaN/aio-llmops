from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from core.routing import load_routing_config


class RoutingConfigEnvExpansionTest(TestCase):
    def test_provider_api_bases_expand_from_host_and_port_variables(self):
        config_text = """
default_route: medical_qa
providers:
  vllm:
    api_base: http://${VLLM_HOST}:${VLLM_PORT}/v1
    api_key: test-key
  trtllm:
    api_base: http://${TRTLLM_HOST}:${TRTLLM_PORT}/v1
    api_key: test-key
classifier:
  provider: vllm
  model: routing-classifier
  instructions: route requests
routes:
  medical_qa:
    provider: vllm
    model: vi-medqa-lora
    description: medical questions
""".strip()

        with TemporaryDirectory() as tmp_dir, patch.dict(
            "os.environ",
            {
                "VLLM_HOST": "192.168.1.101",
                "VLLM_PORT": "8000",
                "TRTLLM_HOST": "192.168.1.102",
                "TRTLLM_PORT": "8002",
            },
            clear=False,
        ):
            config_path = Path(tmp_dir) / "routing.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            routing_config = load_routing_config(config_path)

        self.assertEqual(routing_config.providers["vllm"].api_base, "http://192.168.1.101:8000/v1")
        self.assertEqual(routing_config.providers["trtllm"].api_base, "http://192.168.1.102:8002/v1")