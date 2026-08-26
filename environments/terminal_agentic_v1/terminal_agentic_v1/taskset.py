"""terminal-agentic-v1 — Harbor-backed terminal benchmarks.

This is the deliberately thin one, and that is the point. Harbor tasks already define
their own images, prompts, timeouts, network policy and verifier scripts; `HarborTaskset`
already knows how to load a Harbor registry, provision the sandbox, run the agent phase
and read `reward.json`. A port that re-implemented any of that would be adding code whose
only achievement is a second place for the behaviour to drift.

So the whole taskset is a config subclass naming the dataset. Every knob a run needs —
which tasks, which harness, timeout and resource multipliers, whether to honour the
benchmark's own timeouts — already exists on `HarborConfig` and is reachable from the CLI
or a TOML file. See `README.md` for the runnable configs.

`ignore_timeouts` stays at Harbor's verifiers-side default of `True`: the timeouts written
into a Harbor task were authored against Harbor's own runtime, so enforcing them here
would score the speed of whatever inference stack is behind the run as if it were model
capability. Set it to `False` for a faithful head-to-head against published numbers.
"""

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTask, HarborTaskset


class TerminalBench2Config(HarborConfig):
    dataset: str = "terminal-bench/terminal-bench-2"
    """Harbor registry dataset. Point this at any Harbor-format registry to reuse the wiring."""


class TerminalBench2Taskset(HarborTaskset, vf.Taskset[HarborTask, TerminalBench2Config]):
    """terminal-bench-2, loaded straight from the Harbor registry."""


__all__ = ["TerminalBench2Taskset"]
