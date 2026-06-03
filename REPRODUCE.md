# How to reproduce my POC results

**Candidate:** Abstrabit-team
**Submitted:** 2026-06-03
**Time-to-reproduce on a fresh machine (measured):** ~25 minutes (most of it model/dep downloads)

---

## 1. Prerequisites you assumed

- macOS (Apple Silicon or Intel) **or** Windows 11 — developed on Windows 11 / Python 3.12.
- Python 3.10+
- [Ollama](https://ollama.com) installed and the daemon running
- An `OPENAI_API_KEY` (the LLM judge defaults to OpenAI `gpt-4o-mini` for reproducible
  scoring; without it the judge falls back to a noisier local model)

## 2. Setup steps

```bash
# 1. Install Ollama, then make sure the daemon is reachable
#    macOS: launch Ollama.app  |  Windows: it runs as a service after install
#    Linux/Homebrew: `ollama serve` in a separate terminal
curl -s http://localhost:11434/api/tags          # should return JSON

# 2. Pull the model under test
ollama pull llama3.2:1b-instruct-q4_K_M

# 3. Python env
python3 -m venv .venv
source .venv/bin/activate                         # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 4. Judge key (reproducible scoring)
export OPENAI_API_KEY=sk-...                       # Windows PS: $env:OPENAI_API_KEY="sk-..."

# Windows only: so the scorer's ✓/✗ glyphs print on the console
#   PowerShell: $env:PYTHONUTF8="1"
```

## 3. Run the eval

```bash
# One command runs the full suite (long-thread + cross-session); then score + judge.
python -m stubs.runner --eval --all
python -m eval.score_long_thread eval_results/grief_20turn.json
python -m eval.llm_judge        eval_results/grief_20turn.json
python -m eval.score_long_thread eval_results/grief_20turn_11-20.json
python -m eval.llm_judge        eval_results/grief_20turn_11-20.json
```

## 4. Expected output

After running, `eval_results/` contains:

- `grief_20turn.json` — full 20-turn transcript
- `grief_20turn_scores.json` — deterministic metrics
- `grief_20turn_judge.json` — LLM-as-judge scores
- `grief_20turn_1-10.json` — cross-session part 1
- `grief_20turn_11-20.json` — cross-session part 2 (+ `_scores`/`_judge`)

My reported scores (from my own runs):

| Metric | My score | Target |
|---|---|---|
| Topic adherence (LLM-judge, full) | 4 / 5 | ≥ 4.0 |
| Progressive depth (LLM-judge, full) | 4 / 5 | ≥ 4.0 |
| Fact recall (turns 13+, full) | 6 of 8 | ≥ 3 |
| Restart-marker reuse (turns 17+) | 0 | 0 |
| Avg near-dups per response | 0.00 | ≤ 0.5 |
| Cross-session continuity (LLM-judge, 11-20) | 5 / 5 | ≥ 4.0 |

Moses should see scores in the same range (see the variance note in §5 — all gated
thresholds clear every run; fact recall floats 3–6/8, judge dimensions 4–5/5).

## 5. Gotchas / known issues

- **First run downloads** the model (~0.8 GB) and the scorer's sentence-transformers
  embedder (~80 MB + torch); subsequent runs are fast.
- **Score variance:** generation temperature is 0.7 (raised from 0.4 to break a
  forced-opener parroting effect — see WRITEUP), so scores float run-to-run. All
  *gated thresholds* clear on every run; fact recall has the least margin (floor ~3/8).
  If a single run shows recall 2/8 or topic 3, re-run `--all` once.
- The Ollama judge fallback (no `OPENAI_API_KEY`) occasionally returns malformed JSON;
  setting the key avoids it.
- The kit's `.gitignore` excludes `eval_results/*.json`; if cloning my repo, the
  delivered transcripts are force-added.

## 6. If something doesn't reproduce

Contact: wearesomethingbigger@gmail.com — I'll respond within 24 hours.
