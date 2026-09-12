# pandas-agentic-v1

Clean, reconcile and merge-audit messy order exports inside a sandbox. **The artifact is
the answer** — the reward reads the file the agent wrote, not what it said — and the
grader never imports pandas, so pandas' own conventions are what is being tested.

| | |
|---|---|
| **Archetype** | Agentic, sandboxed, artifact-verified, procedural + `INFINITE` |
| **Harness** | `bash`, `claude-code`, `codex` — anything with a shell |
| **Runtime** | Container required (`NEEDS_CONTAINER = True`); every task declares the image `amancevice/pandas:slim-3.0.5` |
| **Reward** | `artifact_correct` (1.0), binary per task; `produced_output` metric separates "no run" from "wrong run" |

## Task families

Every task starts from a generated order export with the columns `order_id, category,
region, units, unit_price`. Missing values appear as any of `-`, `?`, `N/A`, `NA`, `null`
or an empty cell (pandas' default `na_values` does not cover all of these), numbers may
carry thousands separators or a `$`, and some rows are exact duplicates. The prompt states
all of this; nothing about the input is a hidden trick.

| Family | The work | Output | Scores 0 if |
|---|---|---|---|
| `repair` | messy CSV → clean typed CSV: drop exact duplicates keeping the first, empty every missing cell, `units` as integers, `unit_price` as floats rounded half-up to 2 decimals, same header and row order | `clean.csv` | any sentinel left in place, wrong value, kept duplicate, or reordered row |
| `reconcile` | clean, then per-`category` count / **nulls** / sum / mean of `unit_price`, rounded half-up at the end; a category whose prices are all missing is reported with count 0, sum 0.0 and mean null | `result.json` | nulls averaged in, banker's rounding, a missing or phantom category |
| `merge` | clean orders (`left.csv`) and shipments (`right.csv`), then audit a LEFT join on `order_id`: joined row count plus the unmatched keys on both sides | `result.json` | wrong row count — an order with two shipment rows contributes two joined rows — or a wrong key list |

Task `idx` selects the family (`families[idx % len(families)]`). The first 60 tasks of the
default seed are the fixed benchmark slice, 20 per family; the stream continues
indefinitely for training.

## Rewards and metrics

`finalize` reads the output file from the sandbox while it is alive (`clean.csv` as text,
`result.json` parsed) and records why when it cannot; the reward runs from what it
captured. A missing or unparseable file scores 0.

| Reward | Weight | Pays 1.0 when |
|---|---|---|
| `artifact_correct` | 1.0 | **repair**: the same number of rows as the reference, in the same order; per cell, a missing value must be an empty string, integers and strings must match exactly, floats must be within `task.float_tolerance` (default 0.001). **reconcile**: the set of categories equals the reference exactly; per category `count` and `nulls` are exact, `sum` and `mean` within the tolerance, and a null reference mean requires a null. **merge**: `joined_rows` exact and both key lists equal the reference as sets of strings (their order is not enforced). Anything else scores 0. |

The tolerance is deliberately below 0.005. A banker's-rounded value sits 0.01 from the
half-up one and an unrounded x.xx5 value 0.005 from it; both must fail regardless of
binary float noise, which a tolerance at exactly 0.01 or 0.005 would not guarantee.

| Metric | Meaning |
|---|---|
| `produced_output` | 1.0 if an output file was read at all (parseable JSON, or any CSV text) — separates "never wrote the file" from "wrote the wrong thing". |

## Arguments

Taskset knobs are passed as `--env.taskset.<name>` (kebab-case) to `eval`, or as
`--taskset.<name>` to `validate`; the list-valued `families` is easiest to set in a TOML
config passed with `@ file.toml` (`[env.taskset] families = ["repair"]`).

| Argument | Default | Meaning |
|---|---|---|
| `families` | `["repair", "reconcile", "merge"]` | Families to cycle through by task index. |
| `num-rows` | `60` | Rows per generated order table. The shipments table has half as many orders, a tenth of which get a second shipment row. |
| `mess-rate` | `0.18` | Share of `units`, `unit_price` and `region` cells replaced by a missing-value marker. |
| `seed` | `20260827` | Generator seed. All families draw from one generator, so changing any knob changes every later task, not only the family it concerns. |
| `task.float-tolerance` | `0.001` | Absolute tolerance for every float comparison in the reward. |

## Runtime

- `NEEDS_CONTAINER = True`. Every task sets `image` and `workdir`, and `validate` refuses
  the subprocess runtime, so `--runtime.type docker` (or `prime` / `modal`) is required.
- **Image**: `amancevice/pandas:slim-3.0.5` — Python 3.13 on Debian slim with pandas 3.0.5
  and numpy 2.5.2 preinstalled, published on Docker Hub for linux/amd64 and linux/arm64.
  It is a community-maintained image, not one published by the pandas project, and the
  tag is release-named rather than digest-pinned. The runtimes' default `python:3.11-slim`
  ships no pandas, which is why the task declares one. `validate` runs
  `python3 -c "import pandas"` inside the container, so a missing or substituted image
  fails validation instead of every rollout.
- **Files**: `setup` writes the input files (`raw.csv`, or `left.csv` and `right.csv`)
  under `/workspace`; `finalize` reads the output file (`clean.csv` or `result.json`, at
  most 4 MB) from the same directory.
- **Network**: the task does not restrict egress; the runtime's own policy applies
  (unrestricted by default). Nothing in the task needs the network.
- **Timeouts and resources**: no per-task `timeout` or `resources` are set, so the
  runtime and episode defaults apply. Tables are 60 rows by default, so the defaults are
  ample; raise `--env.timeout.episode` if you scale `num-rows` far beyond that.

## Dependencies

The package depends only on `verifiers>=0.3.1`; the reference implementations are
standard library (`csv`, `decimal`). pandas is needed only inside the sandbox and comes
from the image above — nothing pandas-related is installed on the host. No secrets or
environment variables are needed beyond the model provider's API key.

## Run

```bash
uv run validate pandas-agentic-v1 -n 10 --runtime.type docker      # model-free: keys re-derive, pandas imports in the container
uv run eval pandas-agentic-v1 -n 10 --model <model-id> \
  --env.agent.harness.id bash --env.agent.runtime.type docker
uv run eval pandas-agentic-v1 -n 10 --model <model-id> \
  --env.agent.harness.id bash --env.agent.runtime.type docker \
  --env.taskset.num-rows 200 --env.taskset.mess-rate 0.3       # longer tables, more corruption
```

Always pass `--env.agent.runtime.type`. Without it `eval` falls back to the Prime sandbox
runtime, which provisions a remote, billed sandbox for every rollout.

## Design notes

**The reference is stdlib-only.** The agent will reach for pandas; if the grader used
pandas too, the two would share its missing-value, dtype-coercion, `groupby(dropna=...)`
and merge-duplication conventions, and the environment would quietly stop testing whether
the agent handled them. The contract lives in the prompt, not in a library's defaults.

**Answer keys re-derive from the shipped bytes.** `validate()` recomputes every key from
the exact file bodies the agent receives, so a generator bug fails validation instead of
shipping a wrong key.

**Half-up rounding, stated explicitly.** Python's `round()` and pandas' `.round()` are
banker's rounding and would send 12.345 to 12.34. The prompt promises the ordinary
convention and the reference implements it with `Decimal`.

**Every reward ships with a test that tries to cheat it.** The offline tests in the source
repository assert that echoing the messy input, reordering rows, keeping a duplicate,
leaving a sentinel in place, off-by-one counts, a phantom category, a banker's-rounded or
unrounded mean, and a missing or malformed artifact all score zero; that every category
present in the data appears in the key even when all its prices are missing; that at least
one merge task in the benchmark slice actually multiplies joined rows; and they drive
`setup` → `finalize` through a real subprocess runtime to cover the file plumbing.

## Status

| Check | Result |
|---|---|
| `uv run validate pandas-agentic-v1 -n 10 --runtime.type docker` | **10/10 valid**, 0 errors, 29 s wall clock, on the Docker runtime with the image above: keys re-derive and `import pandas` succeeds in every container. |
| Offline tests and lint | 21 tests for this environment pass (`pytest tests/test_pandas_agentic.py`); `ruff check` and `ruff format --check` clean |
| Model eval | **Not yet run.** The reproducible command is `uv run eval pandas-agentic-v1 -n 10 --model <model-id> --env.agent.harness.id bash --env.agent.runtime.type docker`. |
| prime-rl training | **Not yet run.** |
