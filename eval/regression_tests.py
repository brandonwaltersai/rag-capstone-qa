"""Fixed regression suite — five prompts that must keep behaving the same way across changes.

Run with: python -m pytest eval/regression_tests.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src import data_prep
from src.indexing import build_or_load_index
from src.pipeline import answer_query

REGRESSION_CASES = [
    ("Account Access", "I can't log in to my account", "ANSWER"),
    ("Account Access", "How do I reset my password?", "ANSWER"),
    ("Orders/Refunds", "Where is my order?", "ANSWER"),
    ("Orders/Refunds", "How do I return an item?", "ANSWER"),
    (None, "Dispute a credit card charge for my order.", "ESCALATE"),  # must always escalate — safety trigger
]


@pytest.fixture(scope="module")
def kb_and_index():
    kb = data_prep.load_or_build_kb()
    index = build_or_load_index(kb)
    return kb, index


@pytest.mark.parametrize("lane,query,expected_type", REGRESSION_CASES)
def test_regression(kb_and_index, lane, query, expected_type):
    kb, index = kb_and_index
    out = answer_query(query, kb, index, lane=lane)
    assert out["type"] == expected_type, f"{query!r} -> expected {expected_type}, got {out['type']} ({out['reason']})"
    if expected_type == "ANSWER":
        assert out["citations"], f"{query!r} answered without citations"
