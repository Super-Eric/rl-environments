# pandas-agentic-v1

Sandboxed agentic pandas work — repair, reconcile, merge-audit — graded on the artifact
the agent writes, verified by a **pandas-free** reference.

Built against the Environments Hub **Software Library Evals** contract: real code
execution in a sandbox per task, 50+ nontrivial agentic tasks (the fixed-seed benchmark
slice is 60; the stream is infinite for training), targeting a library the author uses
daily in data-engineering work.

| | |
|---|---|
| **Archetype** | Agentic, sandboxed, artifact-verified, procedural + `INFINITE` |
| **Harness** | `bash`, `claude-code`, `codex` — anything with a shell |
| **Runtime** | Container required (`NEEDS_CONTAINER = True`) |
| **Reward** | Binary per task; `produced_output` metric separates "no run" from "wrong run" |

## Task families

| Family | The work | Output | What ignoring the mess costs |
|---|---|---|---|
| `repair` | messy CSV → clean typed CSV (sentinels, `$`/thousands separators, exact dupes) | `clean.csv` | any un-normalised cell, wrong dtype, kept dupe, or reordered row fails |
| `reconcile` | clean then aggregate: per-group count / **nulls** / sum / mean, half-up at the end | `result.json` | nulls averaged in, banker's rounding, or a phantom group fails |
| `merge` | left-join audit: joined row count + unmatched keys both sides | `result.json` | missing the row-multiplication a duplicated right-side key causes fails |

Roadmap to the full catalogue (tracked here, built next): `reshape` (pivot/melt round-trip),
`timeseries` (resample with DST-safe boundaries), `dtype-forensics` (find the row that
breaks a column's type), `io-contract` (read_csv dialect traps), `window` (rolling stats
with min_periods edges).

## Design rules (house standard, proven in the four sibling environments)

- **The artifact is the answer.** Prose scores zero; `finalize` captures the file while
  the sandbox is alive and records *why* when it can't.
- **The grader never imports pandas.** References are stdlib-only (`reference.py`), so
  pandas' own conventions — NaN propagation, dtype coercion, groupby `dropna`, merge row
  multiplication — are what's being tested, not silently shared between solver and grader.
- **Answer keys re-derive.** `validate()` recomputes every key from the exact file bodies
  the agent receives; a generator bug fails validation instead of shipping a wrong key.
- **Every reward ships with a test that tries to cheat it** — echoing the messy input,
  off-by-one counts, reordered rows, missing artifacts all score zero
  (`tests/test_pandas_agentic.py`).

## Run

```bash
uv run pytest tests/test_pandas_agentic.py -q        # offline: reference + keys + anti-hack
uv run validate pandas-agentic-v1 -n 20 --runtime.type docker
uv run eval pandas-agentic-v1 -n 10 --model <id> \
  --env.agent.harness.id bash --env.agent.runtime.type docker
```

## Status

Three of eight families shipped; 12 offline tests green. Not yet run end-to-end against a
model (needs Docker + credentials) and not yet trained with prime-rl — both are the next
milestones and are required by the bounty contract this targets.
