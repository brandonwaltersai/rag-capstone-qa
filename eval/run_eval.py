"""Batch-evaluate the assistant against eval_prompts.csv and score against the project KPIs:

  - Grounded Answer Rate       — of all ANSWERs, % with valid, evidence-backed citations
  - Escalation Appropriateness — % of cases where type+reason matched the expected outcome
  - Retrieval Coverage         — of expected ANSWERs, % where relevant evidence was found
  - Latency (mean, p95)

Writes eval/logs/eval_results.csv and eval/logs/kpi_summary.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config, data_prep
from src.indexing import build_or_load_index
from src.pipeline import answer_query
from src.config import ALLOWED_LANES

EVAL_MAX_ROWS = 120


def run(max_rows: int = EVAL_MAX_ROWS) -> pd.DataFrame:
    kb = data_prep.load_or_build_kb()
    index = build_or_load_index(kb)
    eval_df = pd.read_csv(config.EVAL_PROMPTS_PATH, keep_default_na=False)
    eval_run = eval_df.head(min(max_rows, len(eval_df))).copy()

    records = []
    for r in eval_run.itertuples(index=False):
        q = str(r.utterance_text)
        expected_type = str(r.expected_type)
        lane = str(r.lane) if expected_type == "ANSWER" and str(r.lane) in ALLOWED_LANES else None
        out = answer_query(q, kb, index, lane=lane)

        has_retrieved = bool(out.get("retrieved"))
        has_citations = bool(out.get("citations"))
        citation_ok = (out.get("type") == "ANSWER") and has_citations

        records.append({
            "prompt_id": int(r.prompt_id), "expected_type": expected_type,
            "expected_reason": str(r.expected_reason), "lane_expected": str(r.lane),
            "query": q, "type": out.get("type"), "reason": out.get("reason") or "",
            "latency_s": float(out.get("latency_s") or 0.0),
            "top_kb": (out["retrieved"][0]["KB_ID"] if has_retrieved else ""),
            "top_score": (out["retrieved"][0]["score"] if has_retrieved else None),
            "has_retrieved": has_retrieved, "citation_ok": citation_ok,
        })

    results_df = pd.DataFrame(records)
    for c in ["reason", "expected_reason", "lane_expected"]:
        results_df[c] = results_df[c].fillna("")

    results_path = config.LOG_DIR / "eval_results.csv"
    results_df.to_csv(results_path, index=False)

    answer_ct = int((results_df["type"] == "ANSWER").sum())
    esc_ct = int((results_df["type"] == "ESCALATE").sum())
    grounded_answer_rate = float(results_df.loc[results_df["type"] == "ANSWER", "citation_ok"].mean()) if answer_ct else 0.0

    type_match = results_df["type"] == results_df["expected_type"]

    def _reason_match(row) -> bool:
        if row["expected_type"] != "ESCALATE":
            return True
        exp = str(row.get("expected_reason") or "").strip().lower()
        if not exp:
            return True
        return exp in str(row.get("reason") or "").strip().lower()

    reason_match = results_df.apply(_reason_match, axis=1)
    escalation_appropriateness = float((type_match & reason_match).mean()) if len(results_df) else 0.0
    retrieval_coverage = float(
        results_df.loc[results_df["expected_type"] == "ANSWER", "has_retrieved"].mean()
    ) if (results_df["expected_type"] == "ANSWER").any() else 0.0

    kpi_summary = pd.DataFrame([{
        "eval_rows": int(len(results_df)), "answer_count": answer_ct, "escalate_count": esc_ct,
        "grounded_answer_rate": round(grounded_answer_rate, 3),
        "escalation_appropriateness": round(escalation_appropriateness, 3),
        "retrieval_coverage": round(retrieval_coverage, 3),
        "latency_mean_s": round(float(results_df["latency_s"].mean()), 2),
        "latency_p95_s": round(float(results_df["latency_s"].quantile(0.95)), 2),
    }])
    kpi_summary.to_csv(config.LOG_DIR / "kpi_summary.csv", index=False)
    print(kpi_summary.to_string(index=False))
    return kpi_summary


if __name__ == "__main__":
    run()
