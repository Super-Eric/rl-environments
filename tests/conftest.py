"""Shared helpers for scoring environments offline, with no model and no runtime.

`Task.score(trace)` runs every `@vf.reward` / `@vf.metric` that does not require a
runtime, so a synthetic single-reply trace is enough to assert what an environment pays
for. That is the point of these tests: an environment is only as good as the answers it
*refuses* to reward, and that property is cheap to test but almost never tested.
"""

import pytest
import verifiers.v1 as vf


def make_trace(task: vf.Task, reply: str) -> vf.Trace:
    """A finished one-turn trace whose single assistant message is `reply`."""
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        nodes=[
            vf.MessageNode(parent=None, message=vf.UserMessage(content=str(task.data.prompt or "")), sampled=False),
            vf.MessageNode(parent=0, message=vf.AssistantMessage(content=reply), sampled=True),
        ],
    )


async def score(task: vf.Task, reply: str) -> vf.Trace:
    """Score `reply` against `task` and hand back the trace carrying rewards + metrics."""
    trace = make_trace(task, reply)
    await task.score(trace)
    return trace


async def reward_of(task: vf.Task, reply: str) -> float:
    """The single weighted reward scalar for `reply`."""
    return (await score(task, reply)).reward


@pytest.fixture
def scorer():
    return reward_of
