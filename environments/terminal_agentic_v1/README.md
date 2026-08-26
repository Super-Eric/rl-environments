# terminal-agentic-v1

[terminal-bench-2](https://www.tbench.ai/) loaded through Harbor, wired for verifiers
evaluation and prime-rl training.

| | |
|---|---|
| **Archetype** | Harbor-backed benchmark port |
| **Harness** | `claude-code`, `codex`, `bash` |
| **Runtime** | Container required — Docker locally, Prime VM for scale |
| **Reward** | Harbor's own verifier scripts, read from `reward.json` |

## Why this is nine lines

Harbor tasks already define their images, prompts, timeouts, network policy and verifier
scripts, and `HarborTaskset` already knows how to load a registry, provision the sandbox,
run the agent phase and read the verdict. Re-implementing any of that would add code whose
only achievement is a second place for the behaviour to drift. The whole taskset is a
config subclass naming the dataset; every knob a run needs already exists on `HarborConfig`.

Point `dataset` at any Harbor-format registry to reuse the same wiring.

## `ignore_timeouts`

Left at the verifiers-side default of `True`. Timeouts written into a Harbor task were
authored against Harbor's own runtime, so enforcing them here scores the speed of whatever
inference stack is behind the run as if it were model capability. Set it to `False` for a
faithful head-to-head against published numbers:

```toml
[env.taskset]
ignore_timeouts = false
timeout_multiplier = 2.0
```

## Run

```bash
uv run eval terminal-agentic-v1 -n 5 --model <model-id> \
  --env.agent.harness.id claude-code --env.agent.runtime.type docker

uv run eval @ configs/terminal-bench-2.toml --dry-run   # validate config first
uv run eval @ configs/terminal-bench-2.toml
```

## Status

Wiring is written and imports cleanly. It has **not** been run end-to-end here — a real
run pulls the terminal-bench-2 registry and its task images, which needs a warm Docker
daemon or a Prime account, and (for the agent phase) model credentials. Run the `--dry-run`
above first; it validates the config without provisioning anything.
