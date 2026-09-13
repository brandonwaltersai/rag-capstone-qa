# Lightweight Threat Model

This document captures the security assumptions already reflected in the implementation. It is not a claim of complete LLM red-team coverage.

## Assets to protect
- Correctness of answers returned to users
- Integrity of approved knowledge-base evidence
- Sensitive customer information
- Auditability of answer and escalation decisions

## Primary failure and abuse cases

| Risk | Current control | Residual risk |
|---|---|---|
| Unsupported answer / hallucination | Answers are constrained to retrieved evidence and must pass a code-level citation gate | A valid citation can still support an incomplete or poorly synthesized answer |
| Fabricated citation | Cited `KB_ID` values must be a subset of identifiers actually retrieved for the turn | Citation validity does not by itself prove that every sentence is entailed by the evidence |
| Prompt injection embedded in knowledge-base text | Evidence is explicitly delimited and the system prompt instructs the model to treat it as read-only reference data and ignore embedded instructions | Prompt-level defenses are not a substitute for comprehensive adversarial testing |
| Sensitive-data handling | Configured safety triggers cause pre-generation escalation; no LLM call occurs on that path | Keyword rules can miss paraphrases or novel sensitive-data requests |
| Out-of-scope request | Lane routing rejects unsupported requests before generation | Router errors can misclassify borderline requests |
| Weak retrieval | Similarity threshold causes low-evidence escalation | Threshold calibration may not generalize equally across all query types |
| Empty or self-escalated model output | Application converts those outcomes into structured escalation | Provider/runtime failures still affect availability |

## Security-relevant ordering

The request flow intentionally performs high-confidence blocking decisions before generation:

1. sensitive-request check
2. scope/lane routing
3. retrieval and evidence threshold
4. model generation
5. citation validation

This ordering reduces unnecessary model exposure for requests that should never reach generation.

## What this project does not claim

The project does **not** claim comprehensive protection against indirect prompt injection, data poisoning, model extraction, denial of service, compromised dependencies, account takeover, or all OWASP/ATLAS attack classes. Those would require a broader application-security program and dedicated adversarial evaluation.

## Recommended next security tests

- Curated indirect-prompt-injection corpus embedded in KB records
- Paraphrased sensitive-data requests designed to bypass keyword triggers
- Adversarial lane-routing cases
- Evidence-entailment scoring in addition to citation-ID validity
- Dependency and secret scanning in CI
- Request-rate and malformed-input tests
