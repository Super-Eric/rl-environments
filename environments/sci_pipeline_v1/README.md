# sci-pipeline-v1

Clean and aggregate a messy sensor time series inside a sandbox. **The artifact is the
answer** — the reward reads the file the agent wrote, not what it said.

| | |
|---|---|
| **Archetype** | Agentic, sandboxed, artifact-verified |
| **Harness** | `bash`, `claude-code`, `codex` — anything with a shell |
| **Runtime** | Container required (`NEEDS_CONTAINER = True`) |
| **Reward** | `daily_means` (0.5) + `record_counts` (0.25) + `extremes` (0.25), each exact |

## The task

`/workspace/readings.csv` holds hourly station readings. The agent must write
`/workspace/result.json` with per-day mean temperatures in Celsius, valid/missing counts,
and the extremes. An agent that describes a correct pipeline but never runs it scores zero.

Four defects, each of which silently corrupts a naive aggregation:

| Defect | Failure if ignored |
|---|---|
| `-999.0` sentinel for missing readings | Large and obvious — drags the mean to absurdity |
| Fahrenheit rows, marked only in a `unit` column | Small and **non**-obvious — the one most likely to survive the agent's own sanity check |
| Duplicate timestamps (first occurrence wins) | Slight mean shift, broken counts |
| Shuffled row order | Makes "first in file order" differ from "earliest by time" — stated in the prompt, not a trick |

`tests/test_sci_pipeline.py::test_every_injected_defect_is_load_bearing` asserts that
ignoring any single defect costs reward, so no defect is decoration.

## Design notes

**The reference pipeline is stdlib-only.** The agent will reach for pandas or polars; if
the grader used pandas too, the two would share its missing-value and rounding conventions
and the environment would quietly stop testing whether the agent handled them. The
contract lives in the prompt, not in a library's defaults.

**Half-up rounding, stated explicitly.** Python's `round()` is banker's rounding and would
send 12.345 to 12.34. The prompt promises the ordinary convention and the reference
implements it with `Decimal`.

**Reading happens in `finalize`, not in a reward.** Rewards may run after the runtime is
released. `finalize` captures the artifact while the sandbox is alive and puts both the
submission and any failure reason in `trace.info`, so a failed rollout is diagnosable from
the trace alone.

**Rewards are split three ways to be diagnostic.** Right counts with wrong means points at
aggregation; the reverse points at parsing.

## Run

```bash
uv run validate sci-pipeline-v1 -n 12 --runtime.type docker
uv run eval sci-pipeline-v1 -n 10 --model <model-id> \
  --env.agent.harness.id bash --env.agent.runtime.type docker
uv run eval sci-pipeline-v1 -n 10 --env.taskset.num-days 14 \
  --env.taskset.missing-rate 0.12     # longer series, more gaps
```

## Verified

`tests/test_sci_pipeline.py` checks the reference against a hand-computed fixture (it is
the answer key for every task, so an untested reference means an environment that is
confidently wrong at scale), then checks the rewards against the outputs three plausible
broken pipelines actually produce.

It also drives `setup` → `finalize` against a **real subprocess runtime**, covering the
file plumbing that scoring tests cannot reach: that `setup` materialises exactly one file,
that a missing artifact scores zero and records why, that a correct `result.json` scores
1.0, and that malformed JSON is reported as malformed rather than as absent. The task is
relocated into a temp directory for that test because the subprocess runtime resolves
absolute paths against the host filesystem — it does not sandbox them, which is exactly why
this environment declares `NEEDS_CONTAINER` for real runs.

A full container run has **not** been performed here; Docker Desktop would not finish
starting on this machine. Run it with `uv run validate sci-pipeline-v1 -n 12 --runtime.type docker`.
