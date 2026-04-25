"""
Evaluation script for the /medical-qa endpoint.

Loads samples from hungnm/vietnamese-medical-qa, creates (or reuses) a
LangSmith dataset, calls the backend, and runs evaluators via LangSmith.

Usage:
    python backend/evals/eval_med_qa.py

Required environment variables:
    LANGSMITH_API_KEY        – LangSmith credentials
    OPENAI_API_KEY           – Bearer token for the backend (default: aio2025)
    BACKEND_EVAL_URL         – Base URL of the running backend
                               (default: http://0.0.0.0:8001)
    MED_EVAL_DATASET_NAME    – LangSmith dataset name to create/reuse
    MED_EVAL_NUM_SAMPLES     – Number of HF samples to seed the dataset (default: 50)
"""

import os

import requests
from datasets import load_dataset
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

client = Client()

DATASET_NAME = os.getenv("MED_EVAL_DATASET_NAME", "Medical QA Eval Dataset")
BACKEND_BASE_URL = os.getenv("BACKEND_EVAL_URL", "http://0.0.0.0:8001")
ENDPOINT_URL = f"{BACKEND_BASE_URL}/v1/medical-qa"
API_KEY = os.getenv("OPENAI_API_KEY", "aio2025")
HF_DATASET_NAME = "hungnm/vietnamese-medical-qa"
NUM_SAMPLES = int(os.getenv("MED_EVAL_NUM_SAMPLES", "50"))


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _load_hf_examples(n: int) -> list[dict]:
    """Load the first *n* rows from the Vietnamese medical QA dataset."""
    ds = load_dataset(HF_DATASET_NAME, split="train")
    examples = []
    for row in ds.select(range(min(n, len(ds)))):
        examples.append(
            {
                "inputs": {"question": row["question"].strip()},
                "outputs": {"reference_answer": row["answer"].strip()},
            }
        )
    return examples


def get_or_create_dataset(dataset_name: str) -> object:
    try:
        return client.read_dataset(dataset_name=dataset_name)
    except Exception:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=f"Medical QA evaluation dataset seeded from {HF_DATASET_NAME}.",
        )
        examples = _load_hf_examples(NUM_SAMPLES)
        client.create_examples(dataset_id=dataset.id, examples=examples)
        print(f"Created dataset '{dataset_name}' with {len(examples)} examples.")
        return dataset


# ---------------------------------------------------------------------------
# Target function (calls the backend)
# ---------------------------------------------------------------------------

def call_medical_qa(inputs: dict) -> dict:
    response = requests.post(
        ENDPOINT_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={"question": inputs["question"]},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "content": payload.get("content", ""),
        "route": payload.get("route", ""),
        "provider": payload.get("provider", ""),
        "model": payload.get("model", ""),
    }


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def non_empty_answer(outputs: dict, reference_outputs: dict | None = None) -> bool:
    """Pass if the model returned a non-empty answer."""
    return bool(outputs.get("content", "").strip())


def correct_route(outputs: dict, reference_outputs: dict | None = None) -> bool:
    """Pass if the gateway routed the request to the medical_qa route."""
    return outputs.get("route") == "medical_qa"


def answer_overlap(outputs: dict, reference_outputs: dict) -> float:
    """
    Token-level F1 between the model answer and the reference answer
    (language-agnostic, character-split).  Returns a score in [0, 1].
    """
    reference = reference_outputs.get("reference_answer", "")
    content = outputs.get("content", "")

    if not reference or not content:
        return 0.0

    ref_tokens = set(reference.lower().split())
    out_tokens = set(content.lower().split())
    if not ref_tokens:
        return 0.0

    common = ref_tokens & out_tokens
    precision = len(common) / len(out_tokens) if out_tokens else 0.0
    recall = len(common) / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dataset = get_or_create_dataset(DATASET_NAME)

    experiment_results = client.evaluate(
        call_medical_qa,
        data=dataset.name,
        evaluators=[non_empty_answer, correct_route, answer_overlap],
        experiment_prefix="med-qa-eval",
        max_concurrency=4,
    )

    print("Experiment results summary:")
    print(experiment_results)
