# AGENTS.md

Conventions for coding agents working in this repository.

## Target verifiers v1, always

- `import verifiers.v1 as vf`. Never `import verifiers as vf`.
- v0 (`Environment`, `Rubric`, `Parser`, `SingleTurnEnv`, `MultiTurnEnv`, `ToolEnv`,
  `load_environment()`) is **deprecated** and will be removed. Most tutorials and most
  model priors still teach it. Do not follow them.
- A package exports its `vf.Taskset` subclass — and optionally one `vf.Env` and/or one
  `vf.Harness` — through `__all__`. The v1 loader resolves classes from `__all__` and
  generic bases. Do not add `load_environment()` / `load_taskset()` shims.

## Commands

```bash
uv sync                                   # installs verifiers + every environment editable
uv run init <name>                        # scaffold (never hand-roll the package layout)
uv run validate <id> --runtime.type subprocess   # model-free; use docker if NEEDS_CONTAINER
uv run eval @ configs/<name>.toml --dry-run      # check a config before spending tokens
uv run pytest tests/ -q
uv run ruff check --fix .
```

Always `uv run`, never bare `python`.

## Where things belong

| Owns | Holds |
|---|---|
| `TaskData` | Immutable serialisable row: prompts, ground truth, image, workdir, timeouts. Only this is stored on the trace. |
| `Task` | Behaviour: `setup` / `finalize` / `validate`, stops, metrics, rewards, task-scoped toolsets. |
| `Taskset` | `load()` and selection-time config (split, seed, size). Never override `__init__`. |
| Harness | The reusable agent program and how it is launched. |

## Scoring rules this repo holds itself to

1. **Deterministic verification over judges.** Reach for `vf.Judge` only when the answer
   is genuinely semantic. Every environment here is graded by rule or by artifact.
2. **Derive the answer key from the shipped task, not from generator intent.** Grade
   against a re-derivation that reads the same data the model reads, so a generator bug
   surfaces as a disagreement rather than a silently wrong key. See
   `au_bas_reconciliation_v1/rules.py` and `sci_pipeline_v1/reference.py`.
3. **Binary reward, graded metric.** Partial credit on an exact-answer task is a
   reward-hacking surface. Keep the continuous error as a `@vf.metric` for observability.
4. **Every reward needs a test that tries to cheat it.** A new reward lands with a test
   asserting that the obvious exploit — echo the input, flag everything, emit a
   plausible-magnitude number, skip the required format — scores zero. See `tests/`.
5. **`validate()` returns `bool`.** Returning `None` reads as invalid. Assert for
   diagnostics *and* `return True`.
6. **Metrics separate "wrong" from "unparseable".** They need different fixes.

## Dependencies

Add environment dependencies to that environment's own `pyproject.toml`, then register
the package in the root `pyproject.toml` under `[dependency-groups] dev` and
`[tool.uv.sources]` as `{ workspace = true }`. Never add an environment's dependency to
the root project dependencies.

## Publishing

`prime env push <name> --visibility PRIVATE|PUBLIC` is an external state change. Ask
before running it; local green is not consent to publish.
