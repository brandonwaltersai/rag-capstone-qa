# System Architecture

## Reliability-first request flow

```text
User query
   |
   v
Safety trigger check -----------------------> ESCALATE: safety
   |
   v
Lane routing -------------------------------> ESCALATE: out_of_scope
   |
   v
FAISS vector retrieval
   |
   v
Lane filter + lexical rerank
   |
   v
Evidence threshold -------------------------> ESCALATE: low_evidence
   |
   v
Evidence block with immutable KB IDs
   |
   v
LLM generation constrained to evidence
   |
   v
Citation extraction + code-level validation -> ESCALATE: citation_invalid_or_missing
   |
   v
Grounded answer + validated citations
```

## Design decisions

### 1. Safety checks happen before retrieval and generation
Requests containing configured sensitive-data triggers are routed directly to a secure human-support path. The model is not asked to improvise a response to those requests.

### 2. Retrieval is lane-aware
The system routes a query to one of the supported service lanes, retrieves candidate evidence from the FAISS index, and filters results to the selected lane. This reduces the chance that semantically similar but operationally unrelated content is used as evidence.

### 3. Retrieval combines semantic and lexical signals
FAISS cosine-similarity retrieval produces the initial candidate set. A lightweight lexical-overlap score is then used as a secondary reranking signal without adding another model dependency.

### 4. Evidence is explicitly delimited
Retrieved records are rendered into an evidence block with stable `KB_ID` markers. The system prompt tells the model to treat retrieved content as read-only reference data and to ignore instructions embedded inside it.

### 5. Citation validation is enforced in code
The model must end a non-escalated answer with cited `KB_ID` values. The application extracts those citations and verifies that every cited identifier came from the actual retrieved set. Missing or fabricated citations cause escalation rather than answer delivery.

### 6. Escalation is structured, not a dead end
Each escalation returns a reason code, top retrieved evidence where available, and clarifying questions for a human handoff. The same mechanism covers safety, scope, weak evidence, empty model output, model-requested escalation, and invalid citations.

## Failure philosophy

The system is intentionally optimized for **safe failure over confident failure**. When the evidence, scope, or citations are not strong enough to support an answer, the correct system behavior is escalation.

## Core implementation

The main request path is implemented in `src/pipeline.py`. Retrieval/index construction is separated into `src/indexing.py`, configuration and routing into `src/config.py`, and evaluation into the `eval/` package.
