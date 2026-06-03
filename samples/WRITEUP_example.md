# Khamuel POC — Writeup

**Candidate:** [your name]
**Submitted:** [YYYY-MM-DD]

## Summary

Two-sentence description of your overall approach to the memory layer. What's the architectural shape, and what's the one most important design decision?

## Approach

Concise per-component explanation. One line each is fine.

- **Fact extraction:** [your choice — regex / spaCy NER / Ollama LLM call / hybrid — and why]
- **Rolling summary:** [your choice — hardcoded template / Ollama-generated structured JSON / skip — and why]
- **Key-fact surfacing:** [your choice — always-inject-all / recency-weighted / embedding-ranked top-K]
- **Journey state / depth tracking:** [your choice — turn-count heuristic / milestone-detection / skip]
- **Cross-session persistence:** [JSON / SQLite / pickle — why]
- **Context budget enforcement:** [how you stay under 4000 chars when older content piles up]

## What worked

One to two paragraphs on what you got right. Reference specific eval-result numbers where relevant (e.g., "fact recall went from 0/8 baseline to 5/8 after adding the embedding-ranked surfacing on top of the static-inject baseline at 2/8").

## What didn't work

One to two paragraphs on what you tried and reverted, or what your implementation still handles weakly. Be specific. "I tried X, it produced Y, here's what I think went wrong" is what we're looking for.

## One design tradeoff

The single hardest call you made and why. Be specific. Examples:

- "I chose Ollama-generated summaries over hand-templated ones because [reason], but it costs ~600 ms per summary regeneration and burns 200 tokens of model context."
- "I always inject the top-5 facts even when only 2 are relevant, because relevance ranking on a 1B model is too unreliable. The cost is ~400 wasted prompt chars per turn."

## If I had another week

One paragraph on what you'd build next.

---

## Practical notes (optional)

Anything specific about your setup that helps Moses reproduce or compare:
- Did you use OpenAI for the judge during development, or Ollama 3b?
- Did you modify `runner.py` at all? (If so, briefly why.)
- Any Python version assumptions beyond the README's 3.10+?
- Total dev hours invested.
