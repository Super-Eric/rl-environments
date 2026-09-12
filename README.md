# rl-environments

Reinforcement-learning environments and evals for LLM post-training, built natively
against [verifiers](https://github.com/PrimeIntellect-ai/verifiers) **v1** and runnable
with the Prime Intellect stack (`eval`, `prime-rl`, Environments Hub).

Six environments, each chosen to demonstrate a different technical archetype rather than
to repeat the same shape six times.

| Environment | Archetype | Verification | Container |
|---|---|---|---|
| [`geo-reprojection-v1`](environments/geo_reprojection_v1) | Deterministic, procedural, `INFINITE` | PROJ (the geospatial reference implementation), datum operation pinned | no |
| [`au-bas-reconciliation-v1`](environments/au_bas_reconciliation_v1) | Decomposed reward, procedural, `INFINITE` | Statutory GST rules, re-derived from the ledger | no |
| [`two-hop-latent-v1`](environments/two_hop_latent_v1) | Paper port ([arXiv:2411.16353](https://arxiv.org/abs/2411.16353)), finite | The paper's own grading, ported line-for-line | no |
| [`sci-pipeline-v1`](environments/sci_pipeline_v1) | Agentic, sandboxed, procedural, `INFINITE` | The artifact the agent writes to disk, against a stdlib-only reference | yes |
| [`pandas-agentic-v1`](environments/pandas_agentic_v1) | Agentic, sandboxed, procedural, `INFINITE` | The artifact the agent writes to disk, against a pandas-free reference | yes (`amancevice/pandas:slim-3.0.5`) |
| [`terminal-agentic-v1`](environments/terminal_agentic_v1) | Harbor benchmark port (Terminal-Bench 2.0, 89 tasks) | Harbor's own per-task verifier scripts | yes |

## Quickstart

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                                     # verifiers + all six environments, editable

uv run pytest tests/ -q                     # 117 tests, no model or network needed
uv run ruff check . && uv run ruff format --check .
uv run validate geo-reprojection-v1 -n 40 --runtime.type subprocess
uv run validate sci-pipeline-v1 -n 10 --runtime.type docker     # sandboxed tasksets need Docker
uv run eval @ configs/geo-reprojection.toml --dry-run
```

If `uv run pytest` reports `ModuleNotFoundError: verifiers` (the console-script shims
break on some volumes), use the module form: `uv run python -m pytest tests/ -q`.

Running a real eval needs a model endpoint. Every config in `configs/` is the documented
"correct way to run" its environment — the tool-less ones pin the host `subprocess`
runtime (the framework default provisions a sandbox VM per rollout) and every config
carries an episode timeout:

```bash
prime login                                  # Prime Inference: PRIME_API_KEY, api.pinference.ai
export PRIME_API_KEY=...
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
on pair-level F1 instead, and the test proves blanket suspicion loses. The measured run
below confirms it live: the model flags 20.8 rows on average against ~21 true faults, at
0.949 precision.

**Binary reward, graded metric.** Partial credit on an exact-answer task is a
reward-hacking surface. Rewards are pass/fail against a stated threshold; the continuous
error is recorded as a `@vf.metric` so a training run still has a dense signal to inspect
without the policy being able to farm it.

**The answer key is re-derived, never remembered.** Ground truth is computed by reading
the same task data the model reads
([`rules.py`](environments/au_bas_reconciliation_v1/au_bas_reconciliation_v1/rules.py),
[`reference.py`](environments/sci_pipeline_v1/sci_pipeline_v1/reference.py)), independently
of what the generator intended to produce, and every `validate()` asserts the shipped key
against that re-derivation — a tampered key fails validation, and the tests prove it. A
generator bug then surfaces as a key/data disagreement in `validate` instead of shipping a
confidently wrong key at scale.

**The grader does not share the solver's libraries.** `sci-pipeline-v1`'s and
`pandas-agentic-v1`'s references are stdlib-only. An agent will reach for pandas; if the
grader used pandas too, both would inherit its missing-value and rounding conventions and
the environment would silently stop testing whether the agent handled them.

**Instructions and reward match.** A prompt clause the reward does not enforce trains the
policy to ignore instructions. Each prompt states exactly the tolerance, parsing rule and
output contract its reward applies — nothing more — and the ports keep the upstream
prompt verbatim and document the inherited mismatches instead of hiding them.

**Procedural generation with physical tolerances.** Four of the six environments are
`INFINITE` — tasks are generated on demand from a seed, so there is no answer key to leak
and no fixed set to memorise. Tolerances are expressed in the units of the domain (metres,
cents, degrees of azimuth, hundredths of a degree Celsius), so precision means the same
thing across every task.

**Metrics separate "wrong" from "unparseable".** They need different fixes, and a reward
curve that conflates them hides which one is happening.

## Verification status

Honest accounting of what has actually been run here. Model-free rows were re-run on
2026-09-12 against the current code; model rollouts are from 2026-09-11 with
`openai/gpt-5-mini` via Prime Inference, temperature 0.0, `null` harness, host
`subprocess` runtime.

| Check | Status |
|---|---|
| `pytest tests/` | **117 passed, 1 xfailed** — reward, anti-hack, generator, reference-implementation, and runtime-I/O tests (the strict `xfail` tracks a documented `strict_answer` limitation in `two-hop-latent-v1`) |
| `ruff check .` / `ruff format --check .` | **clean** |
| `validate geo-reprojection-v1 -n 40` | **40/40 valid** — every key recomputed from its row's inputs through the pinned PROJ operation |
| `validate au-bas-reconciliation-v1 -n 60` | **60/60 valid** — every key recomputed from its shipped ledger |
| `validate two-hop-latent-v1 -n 40` | **40/40 valid** — against the pinned upstream data |
| `validate sci-pipeline-v1 -n 10 --runtime.type docker` | **10/10 valid** on a real container runtime |
| `validate pandas-agentic-v1 -n 10 --runtime.type docker` | **10/10 valid** on a real container runtime, pandas available in the sandbox image |
| All six `configs/*.toml` | **resolve** under `eval --dry-run` |
| `geo-reprojection-v1` rollout, 40 tasks | reward **0.088** (3/34), `answered` 1.000, median error 92 m against a 1 m tolerance. 36 traces recorded — the run was cut after four rollouts hung in a provider retry loop (the configs now carry an episode timeout); 2 provider-error traces excluded. The prompt wording was tightened after this run; the reward is unchanged |
| `au-bas-reconciliation-v1` rollout, 30 tasks | `bas_correct` **0.267**, `fault_audit` **0.946** (precision 0.949 / recall 0.946, 20.8 rows flagged vs ~21 true faults), `parsed` 1.000; 30/30 completed |
| `two-hop-latent-v1` rollout, 60 tasks, `nocot` | `correct` **0.233** (14/60), `strict_answer` 0.200, `first_hop_mentioned` 0.017, `answer_spam` 0.500 — every correct row had zero spam and never named the bridge entity; 60/60 completed. Read as a prior baseline: the upstream protocol fine-tunes on the first-hop facts first (see the environment README) |
| `sci-pipeline-v1` / `pandas-agentic-v1` rollout | not yet run — needs a model plus Docker in one place |
| `terminal-agentic-v1` validate / rollout | not yet run — needs the Harbor CLI (`uv pip install 'verifiers[harbor]'`, which re-pins `openai` for the whole workspace, so it is installed deliberately, not by default), Docker, and model credentials |

The unrun rows are infrastructure-gated, not correctness-gated. `terminal-agentic-v1`
delegates entirely to `HarborTaskset`; its 22 offline tests cover the `solved` reward and
the Harbor wiring, and what is unrun there is Harbor's provisioning, not code in this repo.

## Layout

```
environments/<name>/          one installable package per environment
  <name>/taskset.py           TaskData + Task (rewards) + Taskset (load)
  <name>/rules.py             independent answer-key derivation, where one is needed
  <name>/reference.py         stdlib-only grader reference, where one is needed
  pyproject.toml              its own dependencies + Hub tags + license
  README.md                   the task, the design decisions, how to run it — rendered on the Hub
configs/*.toml                runnable eval configurations, one per environment
tests/                        offline reward + anti-hack tests (no model, no network)
AGENTS.md                     conventions for coding agents working here
```

## Publishing to the Environments Hub

```bash
prime login
prime env push geo-reprojection-v1 --path environments --visibility PUBLIC   # prompts once for a username
```

`prime env push` uploads the built wheel plus `README.md`, `pyproject.toml` and the
package; `outputs/`, `dist/` and hidden files are excluded. A finished v1 eval run is
attached to the published environment by re-entering it — a complete run re-runs nothing
and uploads through the config's `push = true`:

```bash
uv run eval @ outputs/<run-dir>/configs/resolved/eval.json --resume
```

## License

MIT
