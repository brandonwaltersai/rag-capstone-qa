"""Command-line entry point: build the index (first run) and answer a question.

Usage:
    python -m src.cli "How do I reset my password?"
"""
import sys

from . import data_prep
from .indexing import build_or_load_index
from .pipeline import answer_query


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m src.cli "your question"')
        sys.exit(1)
    query = " ".join(sys.argv[1:])

    if data_prep.config.KB_CLEAN_PATH.exists():
        kb = data_prep.load_or_build_kb()
    else:
        kb_csv, _ = data_prep.fetch_raw_csvs()
        kb = data_prep.load_or_build_kb(kb_csv)

    index = build_or_load_index(kb)
    result = answer_query(query, kb, index)

    print(f"\n[{result['type']}] lane={result['lane']} reason={result['reason'] or '-'}")
    print(result["answer"])
    if result["citations"]:
        print("Citations:", ", ".join(result["citations"]))
    print(f"latency: {result['latency_s']:.2f}s")


if __name__ == "__main__":
    main()
