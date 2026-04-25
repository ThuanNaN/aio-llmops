"""
Evaluation script for the /math-qa endpoint.

Loads samples from TIGER-Lab/MathInstruct, creates (or reuses) a LangSmith
dataset, calls the backend, and runs evaluators via LangSmith.

Usage:
    python backend/evals/eval_math_qa.py

Required environment variables:
    LANGSMITH_API_KEY       – LangSmith credentials
    OPENAI_API_KEY          – Bearer token for the backend (default: aio2025)
    BACKEND_EVAL_URL        – Base URL of the running backend
                              (default: http://0.0.0.0:8001)
    MATH_EVAL_DATASET_NAME  – LangSmith dataset name to create/reuse
    MATH_EVAL_NUM_SAMPLES   – Number of HF samples to seed the dataset (default: 50)
"""

import os
import re

import requests
from datasets import load_dataset
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

client = Client()

DATASET_NAME = os.getenv("MATH_EVAL_DATASET_NAME", "Math QA Eval Dataset")
BACKEND_BASE_URL = os.getenv("BACKEND_EVAL_URL", "http://0.0.0.0:8001")
ENDPOINT_URL = f"{BACKEND_BASE_URL}/v1/math-qa"
API_KEY = os.getenv("OPENAI_API_KEY", "aio2025")
HF_DATASET_NAME = "TIGER-Lab/MathInstruct"
NUM_SAMPLES = int(os.getenv("MATH_EVAL_NUM_SAMPLES", "50"))


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _load_hf_examples(n: int) -> list[dict]:
    """Stream the first *n* rows from MathInstruct and convert to LangSmith examples."""
    ds = load_dataset(HF_DATASET_NAME, split="train", streaming=True)
    examples = []
    for row in ds.take(n):
        examples.append(
            {
                "inputs": {"question": row["instruction"].strip()},
                "outputs": {"reference_answer": row["output"].strip()},
            }
        )
    return examples


def get_or_create_dataset(dataset_name: str) -> object:
    try:
        return client.read_dataset(dataset_name=dataset_name)
    except Exception:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=f"Math QA evaluation dataset seeded from {HF_DATASET_NAME}.",
        )
        examples = _load_hf_examples(NUM_SAMPLES)
        client.create_examples(dataset_id=dataset.id, examples=examples)
        print(f"Created dataset '{dataset_name}' with {len(examples)} examples.")
        return dataset


# ---------------------------------------------------------------------------
# Target function (calls the backend)
# ---------------------------------------------------------------------------

def call_math_qa(inputs: dict) -> dict:
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
    """Pass if the gateway routed the request to the math_qa route."""
    return outputs.get("route") == "math_qa"


def numeric_answer_present(outputs: dict, reference_outputs: dict) -> bool:
    """
    Soft check: pass if at least one number from the reference answer appears
    somewhere in the model's answer.  Works for simple arithmetic tasks where
    the reference contains a clear numeric result.
    """
    reference = reference_outputs.get("reference_answer", "")
    content = outputs.get("content", "")
    ref_numbers = set(re.findall(r"-?\d+(?:\.\d+)?", reference))
    if not ref_numbers:
        return True  # No numeric ground-truth to check; skip.
    out_numbers = set(re.findall(r"-?\d+(?:\.\d+)?", content))
    return bool(ref_numbers & out_numbers)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dataset = get_or_create_dataset(DATASET_NAME)

    experiment_results = client.evaluate(
        call_math_qa,
        data=dataset.name,
        evaluators=[non_empty_answer, correct_route, numeric_answer_present],
        experiment_prefix="math-qa-eval",
        max_concurrency=4,
    )

    print("Experiment results summary:")
    print(experiment_results)
