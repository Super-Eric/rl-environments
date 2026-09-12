"""Terminal-Bench 2.0 through verifiers' `HarborTaskset`. The config names the Harbor Hub
package and the task subclass declares that a container is required; everything else
(download, parsing, sandbox setup, running `tests/test.sh`, reading the verdict) is upstream.
"""

from collections.abc import Iterator

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTask, HarborTaskset


class TerminalBench2Config(HarborConfig):
    dataset: str = "terminal-bench/terminal-bench-2"
    """Harbor Hub package. Unpinned: append `@<ref>` (tag, revision or digest) to freeze a snapshot."""


class TerminalBench2Task(HarborTask):
    NEEDS_CONTAINER = True


class TerminalBench2Taskset(HarborTaskset, vf.Taskset[TerminalBench2Task, TerminalBench2Config]):
    """terminal-bench-2, loaded straight from the Harbor Hub."""

    def load(self) -> Iterator[TerminalBench2Task]:
        for task in super().load():
            yield TerminalBench2Task(task.data, task.config)
