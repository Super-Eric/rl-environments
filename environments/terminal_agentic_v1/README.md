# terminal-agentic-v1

Terminal-Bench 2.0 ([tbench.ai](https://www.tbench.ai/)) loaded through Harbor, wired for
verifiers evaluation and prime-rl training. The package is a thin wrapper over verifiers'
`HarborTaskset`: it names the Hub package, marks every task as needing a container, and
exports the harbor env as the default env. Prompts, images, network policy and verifier
scripts are the benchmark's own.

| | |
|---|---|
| **Archetype** | Harbor-backed benchmark port |
| **Harness** | `terminus-2` (Harbor's reference agent, named in the upstream README), `claude-code`, `codex`, `bash` |
| **Runtime** | Container required (`NEEDS_CONTAINER = True`) — Docker locally, Prime for scale |
| **Default env** | `HarborEnv`; identical to the single-agent env for the shared-verifier tasks tb2 ships |
| **Reward** | `solved` (weight 1.0): the verdict the task's own `tests/test.sh` leaves in `/logs/verifier/` |

## Tasks

89 Terminal-Bench 2.0 tasks, loaded from the Harbor Hub package
`terminal-bench/terminal-bench-2`. Each task ships its instruction (used as the prompt,
stripped of surrounding whitespace), a pullable Docker image, a `tests/test.sh` verifier
and timeouts; `HarborTaskset`
maps them onto `TaskData` (`prompt`, `image`, `workdir`, `network_allow`, `timeout`,
`resources`).

What was verified upstream on GitHub (`harbor-framework/terminal-bench-2`, formerly
`laude-institute/terminal-bench-2`, `main` at `2fd12b8`), across all 89 `task.toml` files:

- every task declares a pullable `docker_image = "alexgshaw/<task>:20251031"`;
- every task sets `allow_internet = true`, which becomes `network_allow = ["*"]`;
- none sets `[verifier].environment_mode`, so every task grades in the agent's own box;
- agent timeouts range 600–12000 s and verifier timeouts 360–12000 s (48 of 89 at 900 s
  on each side, 47 on both), all ignored by default (see Design).

The Hub package itself was not inspected here (the Harbor CLI is not installed — see
Dependencies), so read the above as the state of the GitHub repository, not a guarantee
about the Hub snapshot.

**Unpinned.** `dataset` carries no `@ref` because no Hub tag, revision or digest could be
verified offline: Hub refs are not Git refs, and the GitHub repository has no tags or
releases. Provenance that *was* verified: the legacy Harbor registry
(`laude-institute/harbor`, `registry.json`) pins `terminal-bench@2.0` to all 89 tasks at
commit `69671fbaac6d67a7ef0dfec016cc38a64ef7a77c` (2025-10-31) of the tb2 repository, whose
`task.toml` files use schema 1.0 and predate the `allow_internet` field. To freeze a Hub
snapshot, look a ref up with the Harbor CLI and set it yourself — `dataset` is an
ordinary config knob:

```toml
[env.taskset]
dataset = "terminal-bench/terminal-bench-2@<ref>"
```

## Reward

One reward, `solved` (weight 1.0), inherited unchanged from `HarborTask`. After the agent
phase it stages the task's `tests/` into `/tests` — first deleting
`/logs/verifier/reward.json` and `/logs/verifier/reward.txt`, so a verdict the agent planted
is gone before the grader runs — then runs `bash /tests/test.sh` and reads, in order:

1. `/logs/verifier/reward.json`: a finite number, or a non-empty JSON object of numbers. An
   object with a `reward` key scores `reward` and records every other key as a metric
   (`{"reward": 1, "tests_passed": 3}` → reward 1.0, metric `tests_passed = 3.0`); an object
   without one records each key as a reward of its own. Parsing is strict — a quoted string,
   `NaN`, an infinity or `{}` counts as absent.
2. otherwise `/logs/verifier/reward.txt`, parsed as a float (an empty file reads as 0).
3. otherwise **0.0**.

A verifier that crashes or writes nothing therefore scores exactly like a failed task, and
nothing on the trace separates the two. The two upstream tb2 verifiers inspected
(`adaptive-rejection-sampler` and `build-pov-ray` on `main`) run pytest and end in
`echo 1|0 > /logs/verifier/reward.txt`, so for those the reward is binary; the other 87 were
not read.

## Arguments

`[env.taskset]` is a `HarborConfig`; only `dataset`'s default differs from upstream. Defaults
are verifiers 0.3.1's.

| Argument | Default | Meaning |
|---|---|---|
| `dataset` | `"terminal-bench/terminal-bench-2"` | Harbor Hub package, `org/name` or `org/name@ref`. Any Harbor-format package works, so the same wiring can run another registry. |
| `repo` | `None` | Legacy `--repo` registry selector (`org/repo@ref`); with it `dataset` is a bare `name` or `name@version`. |
| `registry_path` | `None` | Legacy `--registry-path` selector. |
| `registry_url` | `None` | Legacy `--registry-url` selector for a raw `registry.json`. |
| `tasks` | `None` | Subset of task names to load (`None` = all 89). |
| `ignore_timeouts` | `True` | Drop the tasks' declared agent and verifier timeouts (see Design). |
| `timeout_multiplier` | `1.0` | Scale the tasks' timeouts; only applies with `ignore_timeouts = false`. |
| `resource_multiplier` | `1.0` | Scale CPU, memory and disk requests (GPUs unchanged). |
| `require_image` | `False` | Reject a task that declares no environment instead of running it on the runtime's default image. |
| `ignore_dockerfile` | `False` | Run a Dockerfile-only task on the harness image instead of rejecting it (the Dockerfile is not built). |
| `ignore_separate_verifier` | `False` | Grade a separate-verifier task in the agent's box anyway. |

The env is `HarborEnv`, whose config adds `verifier_runtime` (where a separate-verifier
task grades; default: derived from the agent's runtime) and `verifier_retries` (default 2).

## Dependencies

- **Python 3.12** — `requires-python = ">=3.12"`, because Harbor only resolves on 3.12+.
- **`verifiers>=0.3.1`** — the only declared dependency. The package installs and imports
  without Harbor.
- **The Harbor CLI, from `verifiers[harbor]`** — required to load *any* task: `load()` shells
  out to `harbor download`, so `validate` and `eval` fail before a single task is parsed
  without it. It is deliberately **not** declared on this package, not even as an extra:
  the extra pins `harbor==0.21.0`, whose `litellm` dependency requires `openai<3`, so it
  re-pins `openai` (3.3.1 → 2.54.0 in the workspace this was developed in), `openai-agents`
  and `websockets`. A uv workspace folds every member extra into the shared lock — declaring
  the extra here, even behind a `tool.uv.conflicts` fork, was verified to re-pin those three
  for every environment in the workspace. Install it into a venv where that is acceptable:

  ```bash
  uv pip install 'verifiers[harbor]>=0.3.1'      # next to this package, Python 3.12
  uv sync --python 3.12 --extra harbor            # inside a verifiers checkout (its own hint)
  ```
- **Docker** (local) or a **Prime** runtime — every tb2 task carries an image and the task
  class sets `NEEDS_CONTAINER`, so the subprocess runtime is refused.
- **A model API key** for the chosen harness, exported the way that harness expects (for
  example `ANTHROPIC_API_KEY` for `claude-code`). No other secrets.

## Design

`ignore_timeouts` stays at verifiers' default of `True`. Timeouts written into a Harbor task
were authored against Harbor's own runtime, so enforcing them here scores the speed of
whatever inference stack is behind the run as if it were model capability. For a
head-to-head against published numbers, set:

```toml
[env.taskset]
ignore_timeouts = false
timeout_multiplier = 2.0
```

`NEEDS_CONTAINER = True` is set on the task class rather than left to the images: it also
holds for a registry whose tasks declare no image, which the default `require_image = false`
would otherwise let onto the subprocess runtime. `HarborEnv` is exported so that a registry
declaring `[verifier].environment_mode = "separate"` grades in a fresh box instead of
raising.

## Run

Model-free check — needs the Harbor CLI and Docker, and has **not** been run here:

```bash
uv run validate terminal-agentic-v1 -n 5 --runtime.type docker
```

An eval from the CLI:

```bash
uv run eval terminal-agentic-v1 -n 5 --model <model-id> \
  --env.agent.harness.id claude-code --env.agent.runtime.type docker
```

or from a config file. This is the whole file:

```toml
model = "anthropic/claude-sonnet-5"
num_tasks = 5

[env.taskset]
id = "terminal-agentic-v1"
ignore_timeouts = true

[env.agent.harness]
id = "claude-code"

[env.agent.runtime]
type = "docker"
```

```bash
uv run eval @ terminal-bench-2.toml --dry-run   # resolves the config; does not load tasks
uv run eval @ terminal-bench-2.toml
```

`--dry-run` succeeds even without the Harbor CLI, so it is not evidence that tasks load.

## Status

| Check | Result |
|---|---|
| `ruff check`, `ruff format --check` | clean |
| import | ok — `taskset_class` resolves `TerminalBench2Taskset`, `environment_class` resolves `HarborEnv` |
| `eval --dry-run` on the config above | resolves |
| `pytest tests/test_terminal_agentic.py` | 22 passed: loader wiring, the container flag, and the `solved` reward against planted, malformed and missing verdicts on a stub runtime |
| `validate` | **not run** — the Harbor CLI is not installed (`verifiers[harbor]` re-pins `openai` 3.3.1 → 2.54.0; see Dependencies) |
| model eval | **not run** — there are no measured reward numbers for this environment |
