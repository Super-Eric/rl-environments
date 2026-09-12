# sci-pipeline-v1

Clean and aggregate a messy sensor time series inside a sandbox. **The artifact is the
answer** — the reward reads the file the agent wrote, not what it said.

| | |
|---|---|
| **Archetype** | Agentic, sandboxed, artifact-verified, procedural + `INFINITE` |
| **Harness** | `bash`, `claude-code`, `codex` — anything with a shell |
| **Runtime** | Container required (`NEEDS_CONTAINER = True`); the runtime's default image, `python:3.11-slim` |
| **Reward** | `daily_means` (0.5) + `record_counts` (0.25) + `extremes` (0.25); binary per component — counts exact, temperatures within 0.01 °C |

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

The offline tests assert that ignoring any single defect costs reward, so no defect is
decoration.

## Rewards and metrics

Every reward reads the `result.json` that `finalize` captured from the sandbox. A missing
or unparseable file scores zero on all three.

| Reward | Weight | Pays 1.0 when |
|---|---|---|
| `daily_means` | 0.5 | The set of dates in `daily_mean_c` equals the reference exactly and every mean is within `tolerance_c` (default 0.01 °C) of the reference. A missing date, an extra date, or one mean out of tolerance scores 0. |
| `record_counts` | 0.25 | `n_valid` and `n_missing` both equal the reference counts. Values are passed through `int()` before comparison. |
| `extremes` | 0.25 | `max_c` and `min_c` are both within `tolerance_c` of the reference. |

Metrics carry no weight; they exist so a failed rollout is diagnosable from the trace:

| Metric | Meaning |
|---|---|
| `produced_output` | 1.0 if a parseable `result.json` existed at all — separates "no run" from "wrong run". |
| `day_coverage` | Share of expected dates present in `daily_mean_c`, whatever their values — shows truncated output that `daily_means` scores as 0. |

## Arguments

Taskset knobs are passed as `--env.taskset.<name>` (kebab-case) to `eval`, or as
`--taskset.<name>` to `validate`.

| Argument | Default | Meaning |
|---|---|---|
| `num-days` | `5` | Calendar days of hourly readings per task. |
| `missing-rate` | `0.06` | Share of readings replaced by the `-999.0` sentinel. |
| `fahrenheit-rate` | `0.15` | Share of readings recorded in Fahrenheit. |
| `duplicate-rate` | `0.05` | Share of readings re-emitted under a timestamp that already appeared. |
| `seed` | `20260826` | Generator seed; the task stream is deterministic per seed. |
| `task.tolerance-c` | `0.01` | Absolute tolerance, in Celsius, for the mean and extreme comparisons. |

## Dependencies

The package depends only on `verifiers`; the reference pipeline is standard library
(`csv`, `decimal`). The agent's container is the runtime's default image,
`python:3.11-slim`: Python and the stdlib `csv` module are enough to solve the task, and no
data-science library is preinstalled. No secrets or environment variables are needed
beyond the model provider's API key.

## Run

```bash
uv run validate sci-pipeline-v1 -n 10 --runtime.type docker      # model-free: keys re-derive
uv run eval sci-pipeline-v1 -n 10 --model <model-id> \
  --env.agent.harness.id bash --env.agent.runtime.type docker
uv run eval sci-pipeline-v1 -n 10 --model <model-id> \
  --env.agent.harness.id bash --env.agent.runtime.type docker \
  --env.taskset.num-days 14 --env.taskset.missing-rate 0.12     # longer series, more gaps
```

Always pass `--env.agent.runtime.type`. Without it `eval` falls back to the Prime sandbox
runtime, which provisions a remote, billed sandbox for every rollout.

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

## Status

Model-free validation: **10/10 valid** on the Docker runtime
(`uv run validate sci-pipeline-v1 -n 10 --runtime.type docker`) — the first
container-runtime run in this repo.

20 offline tests pass and ruff is clean. They check the reference against a hand-computed
fixture (it is the answer key for every task, so an untested reference means an
environment that is confidently wrong at scale), then check the rewards against the
outputs three plausible broken pipelines actually produce and against direct attempts to
cheat each component.

They also drive `setup` → `finalize` against a **real subprocess runtime**, covering the
file plumbing that scoring tests cannot reach: that `setup` materialises exactly one file,
that a missing artifact scores zero and records why, that a correct `result.json` scores
1.0, and that malformed JSON is reported as malformed rather than as absent. The task is
relocated into a temp directory for that test because the subprocess runtime resolves
absolute paths against the host filesystem — it does not sandbox them, which is exactly why
this environment declares `NEEDS_CONTAINER` for real runs.

Measured end-to-end on 2026-09-12 via Prime Inference (temperature 0.0, `bash` harness,
Docker runtime, 10 tasks each, every rollout completed; the reproducible command is
`uv run eval sci-pipeline-v1 -n 10 --model <model-id> --env.agent.harness.id bash --env.agent.runtime.type docker`,
plus `--env.taskset.num-days 30` for the longer setting):

| Model | `num_days` | Reward | `daily_means` | `extremes` | `record_counts` | `produced_output` |
|---|---|---|---|---|---|---|
| `openai/gpt-5-mini` | 5 | **1.000** | 1.0 | 1.0 | 1.0 | 1.0 |
| `openai/gpt-5-mini` | 30 | **1.000** | 1.0 | 1.0 | 1.0 | 1.0 |
| `openai/gpt-5-nano` | 5 | **0.875** | 0.9 | 0.8 | 0.9 | 0.9 |

Reading: gpt-5-mini saturates the task at both lengths — for a competent shell agent the
cleaning rules are fully specified and the artifact is checkable, so this environment is a
correctness floor rather than a ranking eval at that capability level. gpt-5-nano does not
saturate it (one task produced no artifact, one got the extremes wrong), so the environment
separates the two. The generator's messiness is fixed; a difficulty knob on the corruption
mix is the natural next revision if a harder ceiling is wanted.
