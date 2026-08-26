# rl-environments

Reinforcement-learning environments and evals for LLM post-training, built natively
against [verifiers](https://github.com/PrimeIntellect-ai/verifiers) **v1** and runnable
with the Prime Intellect stack (`eval`, `prime-rl`, Environments Hub).

Four environments, each chosen to demonstrate a different technical archetype rather than
to repeat the same shape four times.

| Environment | Archetype | Verification | Container |
|---|---|---|---|
| [`geo-reprojection-v1`](environments/geo_reprojection_v1) | Deterministic, procedural, `INFINITE` | PROJ (the geospatial reference implementation) | no |
| [`au-bas-reconciliation-v1`](environments/au_bas_reconciliation_v1) | Decomposed reward, long-context | Statutory GST rules, re-derived from the ledger | no |
| [`sci-pipeline-v1`](environments/sci_pipeline_v1) | Agentic, sandboxed | The artifact the agent writes to disk | yes |
| [`terminal-agentic-v1`](environments/terminal_agentic_v1) | Harbor benchmark port | Harbor's own verifier scripts | yes |

## Quickstart

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                                     # verifiers + all four environments, editable

uv run pytest tests/ -q                     # 45 tests, no model or network needed
uv run validate geo-reprojection-v1 -n 40 --runtime.type subprocess
uv run eval @ configs/geo-reprojection.toml --dry-run
```

Running a real eval needs a model endpoint:

```bash
export OPENAI_API_KEY=...
uv run eval @ configs/geo-reprojection.toml
```

## Design principles

These are the rules the repository actually enforces, in `tests/` and in CI — not
aspirations. They exist because the failure mode of an RL environment is not crashing, it
is **paying for the wrong behaviour while looking fine**.

**Every reward ships with a test that tries to cheat it.** A reward is only as good as the
answers it refuses. Each environment has tests asserting that the obvious exploit scores
zero: echoing the input back, flagging every row as suspicious, emitting a
plausible-magnitude number, scattering candidate answers in prose, or skipping the
required output format. The headline case is
[`test_flagging_everything_scores_worse_than_the_honest_answer`](tests/test_au_bas.py) —
an audit reward scored on recall alone is maximised by blanket suspicion, so it is scored
on pair-level F1 instead, and the test proves blanket suspicion loses.

**Binary reward, graded metric.** Partial credit on an exact-answer task is a
reward-hacking surface. Rewards are pass/fail against a stated threshold; the continuous
error is recorded as a `@vf.metric` so a training run still has a dense signal to inspect
without the policy being able to farm it.

**The answer key is re-derived, never remembered.** Ground truth is computed by reading
the same task data the model reads
([`rules.py`](environments/au_bas_reconciliation_v1/au_bas_reconciliation_v1/rules.py),
[`reference.py`](environments/sci_pipeline_v1/sci_pipeline_v1/reference.py)), independently
of what the generator intended to produce. A generator bug then surfaces as a
key/data disagreement in `validate` instead of shipping a confidently wrong key at scale.

**The grader does not share the solver's libraries.** `sci-pipeline-v1`'s reference is
stdlib-only. An agent will reach for pandas; if the grader used pandas too, both would
inherit its missing-value and rounding conventions and the environment would silently stop
testing whether the agent handled them.

**Procedural generation with physical tolerances.** Three of the four environments are
`INFINITE` — tasks are generated on demand from a seed, so there is no answer key to leak
and no fixed set to memorise. Tolerances are expressed in the units of the domain (metres,
cents, degrees of azimuth), so precision means the same thing across every task.

**Metrics separate "wrong" from "unparseable".** They need different fixes, and a reward
curve that conflates them hides which one is happening.

## Verification status

Honest accounting of what has actually been run here:

| Check | Status |
|---|---|
| `pytest tests/` | **45 passed** — reward, anti-hack, generator, reference-implementation, and runtime-I/O tests |
| `ruff check .` | **clean** |
| `validate geo-reprojection-v1 -n 40` | **40/40 valid** — every coordinate round-trips through PROJ to within 1 cm |
| `validate au-bas-reconciliation-v1 -n 60` | **60/60 valid** — every key recomputed from its shipped ledger |
| All four `configs/*.toml` | **resolve** under `eval --dry-run` |
| `sci-pipeline-v1` runtime I/O (`setup` → `finalize`) | **passing** — driven against a real subprocess runtime in `tests/` |
| `sci-pipeline-v1` on a *container* runtime | not yet run — Docker Desktop would not finish starting on this machine |
| `terminal-agentic-v1` end-to-end | not yet run — needs the terminal-bench-2 registry, images, and model credentials |
| Any full model rollout | not yet run — no model credentials configured in this environment |

The unrun rows are infrastructure-gated, not correctness-gated. `sci-pipeline-v1`'s reward
logic *and* its runtime file plumbing are both covered by `tests/`; a container adds
isolation, not different semantics. `terminal-agentic-v1` delegates entirely to
`HarborTaskset`, so what is unrun there is Harbor's provisioning, not code in this repo.

## Layout

```
environments/<name>/          one installable package per environment
  <name>/taskset.py           TaskData + Task (rewards) + Taskset (load)
  <name>/rules.py             independent answer-key derivation, where one is needed
  pyproject.toml              its own dependencies + Hub tags
  README.md                   the task, the design decisions, how to run it
configs/*.toml                runnable eval configurations
tests/                        offline reward + anti-hack tests (no model, no network)
AGENTS.md                     conventions for coding agents working here
```

## Publishing to the Environments Hub

```bash
prime login
prime env push <name> --visibility PUBLIC
```

## License

MIT
