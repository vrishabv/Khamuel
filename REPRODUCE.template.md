# How to reproduce my POC results

**Candidate:** [your name]
**Submitted:** [date]
**Time-to-reproduce on a fresh Mac (measured):** [your number, e.g. "22 minutes"]

---

## 1. Prerequisites you assumed

- macOS [version] on [Apple Silicon / Intel]
- Python [version]
- [anything else, e.g. Homebrew]

## 2. Setup steps

```bash
# Copy/paste the exact commands Moses runs from a fresh checkout.
# Be specific: include any model pulls, env vars, dependencies.
```

## 3. Run the eval

```bash
# The single command sequence that produces eval_results/*.json
```

## 4. Expected output

After running, `eval_results/` should contain:

- `grief_20turn.json` — full transcript
- `grief_20turn_scores.json` — deterministic metrics
- `grief_20turn_judge.json` — LLM-as-judge scores
- `grief_20turn_1-10.json` — cross-session part 1
- `grief_20turn_11-20.json` — cross-session part 2

My reported scores (from my own runs):

| Metric | My score | Target |
|---|---|---|
| Topic adherence (LLM-judge) | x.x / 5 | ≥ 4.0 |
| Fact recall (turns 13+) | x of 8 | ≥ 3 |
| Restart-marker reuse (turns 17+) | x | 0 |
| Avg near-dups per response | x.xx | ≤ 0.5 |
| Cross-session continuity (LLM-judge) | x.x / 5 | ≥ 4.0 |

Moses should see scores within 5% of these.

## 5. Gotchas / known issues

- [anything Moses needs to know — e.g. "first run downloads ~80 MB of sentence-transformers model", "Ollama judge sometimes returns malformed JSON ~1 in 20 runs, just re-run", etc.]

## 6. If something doesn't reproduce

Contact: [your email / Upwork handle]. I'll respond within 24 hours.
