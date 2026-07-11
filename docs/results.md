# Evaluation Results

Results from a 120-prompt stratified evaluation run (75 in-scope utterances per
lane, sampled from real customer utterances, plus 3 fixed safety/out-of-scope
negative cases), scored against `eval/run_eval.py`.

| KPI | Result | What it measures |
|---|---:|---|
| Grounded Answer Rate | **100%** | Of all non-escalated answers, % with valid citations traceable to retrieved KB evidence |
| Escalation Appropriateness | **88.3%** | % of cases where the system's type (answer/escalate) and reason matched the expected outcome |
| Retrieval Coverage | **96.7%** | Of prompts expecting an answer, % where relevant KB evidence was actually retrieved |
| Mean latency | **1.95s** | End-to-end, retrieval + generation |
| P95 latency | **3.17s** | |

Knowledge base: 17,880 curated entries across two lanes (Orders/Refunds:
11,894, Account Access: 5,986), built from the Bitext customer-support corpus.

## Regression suite

Five fixed prompts run on every change (`eval/regression_tests.py`), including
one hard safety case ("Dispute a credit card charge for my order.") that must
always escalate regardless of retrieval results — this is enforced by the
safety-trigger check running *before* retrieval, not by hoping the model
declines.

## Reading the numbers

- **100% grounded answer rate** is a hard-gate outcome, not a soft metric: the
  system structurally cannot emit an answer without valid citations — if
  citations are missing or reference KB IDs outside what was retrieved, the
  turn is escalated instead. So this number reflects the gate working, not a
  claim that the underlying LLM never hallucinates.
- **88.3% escalation appropriateness** is the more honest signal of system
  quality — it's where false escalations (unnecessary handoffs) and missed
  escalations (should have deferred to a human, didn't) show up. The
  remaining ~12% is the concrete next-iteration target: tightening
  `ESCALATION_THRESHOLD` and lane-routing keyword coverage.
- **Latency (mean 1.95s / p95 3.17s)** is dominated by the generation call;
  retrieval itself is sub-100ms against a 17.8K-item flat FAISS index.

## Reproducing

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
python -m eval.run_eval
```

First run downloads both datasets via `kagglehub`, embeds the knowledge base
(~17.9K items, cached after first run), and writes `eval/logs/kpi_summary.csv`.
