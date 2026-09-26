# Retrieval-Grounded Service Center Chatbot

A RAG assistant for customer support that answers **only** from an approved
knowledge base, cites every claim, and escalates to a human whenever evidence
is missing, weak, or the request touches sensitive/out-of-scope territory.

## The problem

Service centers spread knowledge across FAQs, policy docs, SOPs, and
troubleshooting notes. That fragmentation drives inconsistent answers, longer
resolution times, and repeat contacts — and generative AI makes it worse
before it makes it better, because an unconstrained LLM will answer
confidently from *outside* approved policy just as readily as from inside it.
That's an operational, legal, and reputational risk, not just a quality one.

## The approach

Rather than trusting the model to "know when it doesn't know," the system
makes ungrounded answers structurally difficult to return:

1. **Retrieve** top-k evidence from a FAISS index over a curated, lane-tagged
   knowledge base (17,880 entries), with a hybrid vector + lexical rerank.
2. **Generate** an answer constrained to that evidence only — the prompt
   explicitly instructs the model to treat retrieved text as read-only and to
   ignore any instructions embedded inside it.
3. **Hard-gate on citations** — if the response is missing citations, or cites
   a KB_ID that wasn't actually retrieved, the turn is escalated instead of
   returned. This is enforced in code rather than left to prompt compliance.
4. **Escalate before generation** whenever the query matches a safety trigger
   or falls outside the two supported lanes.

Every escalation produces a structured handoff ticket with the reason,
supporting evidence, and clarifying questions rather than a dead end.

## Final evaluation

The final evaluation run used **120 prompts** across the supported lanes.

| Metric | Result |
|---|---:|
| Answers | 107 |
| Escalations | 13 |
| Grounded Answer Rate | **100%** |
| Retrieval Coverage | **98.3%** |
| Escalation Appropriateness | **89.2%** |
| Mean Latency | **2.00s** |
| p95 Latency | **3.01s** |

Escalation reasons in the final run were: 6 missing/invalid-citation cases,
5 weak-evidence/model escalations, and 2 safety escalations.

Full methodology and reproduction notes: [`docs/results.md`](docs/results.md).

## What this demonstrates

This project reflects the intersection of AI engineering, governance, and
operational deployment — building systems that can state what they know,
show where it came from, expose uncertainty, and stop when the evidence runs out.

## Project structure

```
src/
  config.py       lane routing, safety triggers, scope constants
  data_prep.py    dataset fetch + knowledge base cleaning + eval set construction
  indexing.py     embeddings + FAISS index, with content-signature caching
  pipeline.py     retrieval, generation, citation gate, escalation logic
  cli.py          ask a question from the command line
eval/
  run_eval.py         batch evaluation against the KPIs above
  regression_tests.py fixed regression suite (pytest)
docs/
  results.md      full evaluation results + how to reproduce
data/
  README.md       dataset sources and attribution
```

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env          # add your OPENAI_API_KEY
python -m src.cli "How do I reset my password?"
python -m pytest eval/regression_tests.py -v
python -m eval.run_eval
```

## Stack

Python · OpenAI · FAISS · pandas · pytest

## Author

Brandon Walters — [LinkedIn](https://www.linkedin.com/in/bw172b29208/)
