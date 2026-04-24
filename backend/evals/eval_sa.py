import os
import requests
from dotenv import load_dotenv
from langsmith import Client


load_dotenv()

client = Client()

DATASET_NAME = os.getenv("LANGSMITH_ROUTING_DATASET", "LLM Gateway Routing Dataset")
BACKEND_URL = os.getenv("BACKEND_EVAL_URL", "http://0.0.0.0:8001/v1/chat")
API_KEY = os.getenv("OPENAI_API_KEY", "aio2025")

EXAMPLES = [
    {
        "inputs": {
            "messages": [{"role": "user", "content": "Solve 17 * 24 and explain briefly."}],
        },
        "outputs": {"route": "math_qa"},
    },
    {
        "inputs": {
            "messages": [{"role": "user", "content": "Tamoxifen is primarily used in the treatment of which condition?"}],
        },
        "outputs": {"route": "medical_qa"},
    },
    {
        "inputs": {
            "messages": [{"role": "user", "content": "What is the derivative of x^3 + 4x?"}],
        },
        "outputs": {"route": "math_qa"},
    },
    {
        "inputs": {
            "messages": [{"role": "user", "content": "Which cranial nerve carries parasympathetic fibers to the parotid gland?"}],
        },
        "outputs": {"route": "medical_qa"},
    },
]


def get_or_create_dataset(dataset_name: str):
    try:
        return client.read_dataset(dataset_name=dataset_name)
    except Exception:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Math and medical routing evaluation dataset for the FastAPI gateway.",
        )
        client.create_examples(dataset_id=dataset.id, examples=EXAMPLES)
        return dataset


dataset = get_or_create_dataset(DATASET_NAME)


def route_correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    return outputs.get("route") == reference_outputs.get("route")


def non_empty_answer(outputs: dict, reference_outputs: dict | None = None) -> bool:
    return bool(outputs.get("content", "").strip())


def my_app(messages: list[dict[str, str]]) -> dict:
    response = requests.post(
        BACKEND_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"messages": messages},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    return {"content": payload["content"], "route": payload["route"]}


def ls_target(inputs: dict) -> dict:
    return my_app(inputs["messages"])


experiment_results = client.evaluate(
    ls_target,
    data=dataset.name,
    evaluators=[route_correctness, non_empty_answer],
    experiment_prefix="llm-gateway-routing",
)
