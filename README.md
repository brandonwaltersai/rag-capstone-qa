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
makes ungrounded answers structurally impossible:

1. **Retrieve** top-k evidence from a FAISS index over a curated, lane-tagged
   knowledge base (17,880 entries), with a hybrid vector + lexical rerank.
2. **Generate** an answer constrained to that evidence only — the prompt
   explicitly instructs the model to treat retrieved text as read-only and to
   ignore any instructions embedded inside it (a basic prompt-injection
   defense, since KB content is technically user-influenceable data).
3. **Hard-gate on citations** — if the response is missing citations, or cites
   a KB_ID that wasn't actually retrieved, the turn is escalated instead of
   returned. The model cannot talk its way past this gate; it's a code check,
   not a prompt instruction.
4. **Escalate before generation** whenever the query matches a safety trigger
   (payment details, SSNs, government IDs) or falls outside the two
   supported lanes — no LLM call happens on that path at all.

Every escalation produces a structured handoff ticket (reason, top evidence,
clarifying questions) — not a dead end for the customer.

## Results

| Grounded Answer Rate | Escalation Appropriateness | Retrieval Coverage | Mean Latency |
|---:|---:|---:|---:|
| 100% | 88.3% | 96.7% | 1.95s |

Full methodology and how to reproduce: [`docs/results.md`](docs/results.md).

## What this demonstrates

This project sits at the intersection of applied AI, reliability, and secure
system design: **AI systems don't fail only on the model — they fail on the
governance and controls wrapped around it.** The interesting engineering here
isn't the retrieval or the prompt, it's the citation hard-gate and the
safety-trigger-before-retrieval ordering — the parts that make the system
safer to deploy in a support workflow rather than just demo well.

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

Python · OpenAI (`text-embedding-3-small`, `gpt-4.1`) · FAISS · pandas

## Author

Brandon Walters — [LinkedIn](https://www.linkedin.com/in/bw172b29208/)
