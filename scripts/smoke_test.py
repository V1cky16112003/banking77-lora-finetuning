"""Post-start smoke test (finding 9B). Run this against a running serve.py
before trusting the demo for a live interview — verifies real-model
correctness end to end, which CI cannot do (no GPU/weights in GitHub Actions).

Usage: python scripts/smoke_test.py
"""
from __future__ import annotations

import sys

import requests

API_URL = "http://127.0.0.1:8000/predict"

# A few known Banking77 examples with their expected intent. Fill in with
# real examples + expected labels once the fine-tuned model is trained —
# these are illustrative placeholders matching the dataset's style.
KNOWN_EXAMPLES = [
    {"message": "I lost my card, what should I do?", "expected_label": "card_lost"},
    {"message": "My balance hasn't updated after a transfer.", "expected_label": "balance_not_updated"},
]


def run_smoke_test() -> bool:
    all_passed = True
    for example in KNOWN_EXAMPLES:
        try:
            response = requests.post(API_URL, json={"message": example["message"]}, timeout=60)
        except requests.RequestException as exc:
            print(f"FAIL: request error for '{example['message']}': {exc}")
            all_passed = False
            continue

        if response.status_code != 200:
            print(f"FAIL: status {response.status_code} for '{example['message']}': {response.text}")
            all_passed = False
            continue

        predicted = response.json()["predicted_label"]
        if predicted != example["expected_label"]:
            print(
                f"FAIL: '{example['message']}' -> expected '{example['expected_label']}', "
                f"got '{predicted}'"
            )
            all_passed = False
        else:
            print(f"PASS: '{example['message']}' -> '{predicted}'")

    return all_passed


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
