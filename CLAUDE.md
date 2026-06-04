# CLAUDE.md — Khamuel POC

Guidance for working in this repo. Facts only; verify before adding claims.

## What this is
A take-home POC. The task (README.md): implement `stubs/memory.py`'s `MemoryLayer`
so that **`llama3.2:1b-instruct-q4_K_M`** (run locally via Ollama) can sustain a
coherent, non-repetitive, fact-tracking **20-turn** conversation about one grief
topic — the user's late mother **Sarah** (cancer, age 71, six weeks diagnosis-to-death;
user is a 40-year-old full-time-working parent of two in elementary school).

The conversation script is fixed: `prompts/grief_20turn.json`.

## Grading (kit scorers — do not edit)
- `eval/score_long_thread.py` — deterministic: fact_recall (late turns name the loss),
  repetition (intra-response near-dup pairs), restart_markers (early vs very-late cliché reuse).
- `eval/llm_judge.py` — gpt-4o-mini scores topic_adherence, progressive_depth,
  cross_session_continuity (1–5). Its prompt is hardcoded to the grief/Sarah scenario.

## How to run (Windows / PowerShell)
```powershell
$env:PYTHONUTF8 = "1"
$env:OPENAI_API_KEY = (Get-Content .openai_key -Raw).Trim()   # judge key; .openai_key is gitignored
$py = ".\.venv\Scripts\python.exe"

& $py -m stubs.runner --eval --all                 # full suite (long-thread + cross-session); default model = Llama
& $py -m eval.score_long_thread eval_results/grief_20turn.json
& $py -m eval.llm_judge        eval_results/grief_20turn.json
& $py chat.py --memory                             # interactive tester (uses MemoryLayer)
& $py -m stubs.runner --eval --script prompts/grief_20turn.json --model gemma3:1b   # run another model
```
- Ollama daemon must be running. If down: `Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" serve`
- `--all` overwrites `eval_results/grief_20turn.json`, `grief_20turn_11-20.json`, `memory_state.json`.

## Key files
- `stubs/memory.py` — the graded implementation (`MemoryLayer`). Blocks: pinned facts
  (regex persona extraction → injected every turn), key facts (top-5), hot context
  (`get_hot_messages`, last 6 turns as native chat messages), rolling summary
  (`generate_rolling_summary`, every 6 turns), journey/depth, and `finalize_reply`
  (post-generation verifier: appends a `REMEMBRANCE_CLOSERS` line on late turns (>=13)
  or first reply after reload if the loss isn't already named; strips reused restart
  markers on turns >=17). `build_prompt_context()` is capped at 4000 chars.
- `stubs/runner.py` — orchestrator. `process_turn` wires build_prompt_context +
  get_hot_messages + finalize_reply. Flags: `--baseline --eval --script --turns
  --resume --all --model`. `DEFAULT_OPTIONS`: temperature 0.4, num_predict 220.
- `chat.py` — interactive tester (not graded). Flags: `--memory`, `--model` (num_predict 512).
- `prompts/grief_20turn.json` — fixed 20-turn user script.
- `eval_results/` — outputs (`*.json` gitignored except force-added deliverables).
- `samples/` — reference example transcript + scores from the kit.

## Environment
- venv at `.venv` (deps in `requirements.txt`).
- Ollama models present: `llama3.2:1b-instruct-q4_K_M`, `gemma3:1b`, `gemma2:2b`.
- `.openai_key` holds the gpt-4o-mini judge key (gitignored).

## Conventions / constraints
- The assigned/graded model is `llama3.2:1b-instruct-q4_K_M`; it is `runner.py`'s default
  and the model in the committed deliverable. Other models are run via `--model`.
- Keep `runner.py`'s CLI flags working (the grader uses them). `eval/` scorers are unchanged.
- `MemoryLayer` public method signatures are fixed; `build_prompt_context()` must stay <=4000 chars;
  `save`/`load` are pure JSON.
- Git: commits in this repo do **not** include a `Co-Authored-By: Claude` trailer.
- Files prefixed with `_` (e.g. `_*.log`, `_*.ps1`, `_gen*_run.py`) are local experiment
  scratch and are gitignored; they are not deliverables.

## Evaluation discipline (READ THIS — no laziness)
When grading any run, do NOT sample or conclude from automated signals alone:
- **Read every turn** of the transcript, not a subset. State explicitly if you didn't.
- **Probe/regex HITs are not proof.** They false-positive (e.g. a reply containing
  "shelf"/"photo" as a generic suggestion scores HIT even when the model FORGOT the
  user's specific stated plan and asked them to re-describe it). Read the actual text.
- **Inspect the rolling summary** (`memory.rolling_summary`) when diagnosing recall or
  hallucination — dump it, don't assume what it contains. The summary is LLM-generated
  and can itself hallucinate, then recycle the error every turn.
- **No assuming, guessing, or manipulating results.** If something isn't verified, say so.
- Hallucination/confabulation is judged by reading replies, not by scorers — the kit
  scorers and any LLM judge miss it (and an LLM judge over-rates naturalness).

## Verified caveats (not opinions)
- The committed `eval_results/grief_20turn.json` (Llama) passes the deterministic scorers
  (`all_deterministic_pass=true`) but contains hard safety refusals at turns 2 and 5; the
  scorer's late-turns-only checks do not detect them.
- `eval/llm_judge.py`'s rubric is grief/Sarah-specific, so it cannot fairly grade non-grief transcripts.
- The MemoryLayer is grief-specific: its every-turn directive and rolling-summary prompt name
  "Sarah," and the persona regex only extracts mother/cause/age/kids/occupation. It does not
  generalize to other users or topics without changes.
