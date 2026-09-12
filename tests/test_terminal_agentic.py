"""Tests for terminal-agentic-v1.

The package adds three things to `HarborTaskset`: the dataset name, a task subclass that
requires a container, and the harbor env as the default env. The loader tests pin those.

The reward is upstream `HarborTask.solved`: stage the task's `tests/` into the box, run
`tests/test.sh`, read the verdict it leaves in `/logs/verifier/reward.json` (or the older
`reward.txt`). This repo depends on that code but never asserted it, and the obvious cheat is
the agent writing the verdict file itself before the grader runs. A stub runtime — an
in-memory filesystem plus a command log — runs the whole scoring path offline, with no
Harbor CLI and no Docker, so the tests can say exactly what gets paid for.
"""

import base64

import pytest
import terminal_agentic_v1
import verifiers.v1 as vf
from conftest import make_trace
from terminal_agentic_v1.taskset import TerminalBench2Config, TerminalBench2Task, TerminalBench2Taskset
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.tasksets.harbor import HarborData, HarborEnv, HarborTask, HarborTaskset
from verifiers.v1.utils.loaders import environment_class, taskset_class

REWARD_JSON = "/logs/verifier/reward.json"
REWARD_TXT = "/logs/verifier/reward.txt"
STAGE = "rm -f /logs/verifier/reward.json /logs/verifier/reward.txt"
TEST_SH = ["bash", "/tests/test.sh"]


class StubRuntime(vf.Runtime):
    """An in-memory box. `files` is its filesystem, `commands` everything that ran, and
    `verifier` what `tests/test.sh` leaves behind when it runs."""

    def __init__(self, files: dict[str, bytes] | None = None, verifier: dict[str, bytes] | None = None):
        super().__init__()
        self.files = dict(files or {})
        self.verifier = dict(verifier or {})
        self.commands: list[list[str]] = []

    async def start(self) -> None:
        pass

    async def write(self, path: str, data: bytes) -> None:
        self.files[path] = data

    async def _read(self, path: str) -> bytes:
        if path not in self.files:
            raise vf.SandboxError(f"read {path!r}: no such file")
        return self.files[path]

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        self.commands.append(argv)
        script = argv[2] if argv[:2] == ["sh", "-c"] else ""
        if "base64 <" in script:  # Runtime.read's capped transfer: sh -c <script> sh <cap> <path>
            cap, path = int(argv[4]), argv[5]
            if path not in self.files:
                return ProgramResult(1, "", f"head: {path}: No such file or directory")
            return ProgramResult(0, base64.b64encode(self.files[path][:cap]).decode(), "")
        if STAGE in script:
            if "/tmp/tests.tgz" not in self.files:
                return ProgramResult(1, "", "tar: /tmp/tests.tgz: Cannot open")
            self.files.pop(REWARD_JSON, None)
            self.files.pop(REWARD_TXT, None)
            return ProgramResult(0, "", "")
        if argv == TEST_SH:
            self.files.update(self.verifier)
            return ProgramResult(0, "", "")
        raise AssertionError(f"stub runtime got an unexpected command: {argv}")


@pytest.fixture
def task(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    return TerminalBench2Task(HarborData(idx=0, name="stub-task", prompt="Solve it.", task_dir=str(tmp_path)))


async def solved(task, runtime) -> vf.Trace:
    trace = make_trace(task, "")
    await task.score(trace, runtime=runtime)
    return trace


# ------------------------------------------------------------- wiring


def test_loader_resolves_the_wrapper_and_the_harbor_env():
    assert taskset_class("terminal-agentic-v1") is TerminalBench2Taskset
    assert environment_class("terminal-agentic-v1") is HarborEnv
    assert sorted(terminal_agentic_v1.__all__) == ["HarborEnv", "TerminalBench2Taskset"]


def test_dataset_defaults_to_terminal_bench_2_and_stays_overridable():
    taskset = TerminalBench2Taskset(TerminalBench2Config(id="terminal-agentic-v1"))
    assert taskset.config.dataset == "terminal-bench/terminal-bench-2"
    assert taskset.config.ignore_timeouts is True
    pinned = TerminalBench2Config(id="terminal-agentic-v1", dataset="terminal-bench/terminal-bench-2@7")
    assert TerminalBench2Taskset(pinned).config.dataset == "terminal-bench/terminal-bench-2@7"


def test_tasks_require_a_container():
    assert TerminalBench2Taskset.task_type() is TerminalBench2Task
    assert TerminalBench2Task.NEEDS_CONTAINER is True
    assert HarborTask.NEEDS_CONTAINER is False  # the flag is this package's, not inherited


def test_load_yields_the_container_requiring_subclass(monkeypatch, task):
    monkeypatch.setattr(HarborTaskset, "load", lambda self: iter([HarborTask(task.data, self.config.task)]))
    tasks = list(TerminalBench2Taskset(TerminalBench2Config(id="terminal-agentic-v1")))
    assert [type(t) for t in tasks] == [TerminalBench2Task]
    assert tasks[0].data is task.data


# ------------------------------------------------------------- reward `solved`


async def test_planted_verdict_is_removed_before_the_verifier_runs(task):
    runtime = StubRuntime(files={REWARD_JSON: b"1.0", REWARD_TXT: b"1\n"})  # test.sh writes nothing
    trace = await solved(task, runtime)
    assert trace.reward == 0.0
    staged = next(i for i, argv in enumerate(runtime.commands) if argv[:2] == ["sh", "-c"] and STAGE in argv[2])
    assert staged < runtime.commands.index(TEST_SH)


async def test_verifier_verdict_beats_a_planted_one(task):
    runtime = StubRuntime(files={REWARD_JSON: b"1.0"}, verifier={REWARD_TXT: b"0\n"})
    assert (await solved(task, runtime)).reward == 0.0


@pytest.mark.parametrize("body", [b'"1"', b"NaN", b"1e309", b"{}", b"[1]", b"true", b"", b"not json"])
async def test_verdict_that_is_not_a_finite_number_scores_zero(task, body):
    runtime = StubRuntime(verifier={REWARD_JSON: body})
    trace = await solved(task, runtime)
    assert trace.reward == 0.0
    assert trace.rewards["solved"].score == 0.0


@pytest.mark.parametrize("body, expected", [(b"1", 1.0), (b"0", 0.0), (b"0.5", 0.5)])
async def test_scalar_reward_json_is_the_reward(task, body, expected):
    runtime = StubRuntime(verifier={REWARD_JSON: body})
    trace = await solved(task, runtime)
    assert trace.reward == expected
    assert trace.rewards["solved"].weight == 1.0


async def test_keyed_reward_json_records_the_other_keys_as_metrics(task):
    runtime = StubRuntime(verifier={REWARD_JSON: b'{"reward": 1.0, "tests_passed": 3.0}'})
    trace = await solved(task, runtime)
    assert trace.reward == 1.0
    assert trace.rewards["reward"].score == 1.0 and trace.rewards["reward"].weight == 1.0
    assert trace.metrics["tests_passed"] == 3.0
    assert "tests_passed" not in trace.rewards


@pytest.mark.parametrize("body, expected", [(b"1\n", 1.0), (b"0\n", 0.0)])
async def test_reward_txt_is_read_when_there_is_no_reward_json(task, body, expected):
    # The upstream tb2 verifiers inspected end in `echo 1|0 > /logs/verifier/reward.txt`.
    runtime = StubRuntime(verifier={REWARD_TXT: body})
    assert (await solved(task, runtime)).reward == expected


async def test_reward_json_wins_over_reward_txt(task):
    runtime = StubRuntime(verifier={REWARD_JSON: b"0", REWARD_TXT: b"1\n"})
    assert (await solved(task, runtime)).reward == 0.0


async def test_silent_verifier_scores_zero_like_a_failed_task(task):
    runtime = StubRuntime()
    trace = await solved(task, runtime)
    assert trace.reward == 0.0
    assert trace.rewards["solved"].score == 0.0
    assert runtime.commands[-1] != TEST_SH  # the reads after test.sh happened and found nothing
