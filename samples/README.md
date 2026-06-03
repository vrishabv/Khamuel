# Samples

These files show the exact format your deliverable should match. They're outputs from the **baseline** run (Llama 3.2 1B with no memory layer) on the included grief script. Use them to:

1. Verify the JSON shape your `--eval --all` run should produce
2. See concretely what the bar (0/8 fact recall, ~4 LLM-judge) looks like — what you're improving from
3. See the structure your `WRITEUP.md` should follow

## Files

| File | What it is |
|---|---|
| `sample_grief_20turn.json` | A real transcript from baseline. Your `eval_results/grief_20turn.json` should match this schema (same keys, same nesting). |
| `sample_grief_20turn_scores.json` | Output of `eval/score_long_thread.py` on the baseline. Note `all_deterministic_pass: false` — that's the bar to beat. |
| `sample_grief_20turn_judge.json` | Output of `eval/llm_judge.py` on the baseline (using OpenAI gpt-4o-mini). Note all three rubric dimensions scored 4/5 — your memory layer needs to hold these AND raise fact recall. |
| `WRITEUP_example.md` | Template for the 1-page `WRITEUP.md` you submit. Don't expand it beyond ~1 page; depth matters more than length. |

## Note on the sample transcript

Glance at `sample_grief_20turn.json` and you'll see Llama 3.2 1B's actual baseline output. Notice:

- **Turn 15** quotes Psalm 119:105 as "Through the word of Christ my light has shone..." — pure hallucination (actual WEB text: "Your word is a lamp to my feet, and a light for my path"). The kit doesn't grade verse accuracy in this POC, but it shows you the kind of trust-critical failure Phase 1 will need to address.
- **Late turns (13-20)** never mention "Sarah" or "my mother" — the model has lost the user's specific context completely. This is the fact-recall gap your memory layer fixes.
- **Responses are LONG** (~1000 chars each) but coherent. The 1B model isn't dumb — it's just stateless.
