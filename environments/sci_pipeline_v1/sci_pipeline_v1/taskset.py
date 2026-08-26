"""sci-pipeline-v1 — clean and aggregate a messy sensor time series inside a sandbox.

The agent gets a workspace containing one CSV of hourly station readings and must write
`result.json`. **The artifact is the answer.** Nothing in the transcript is scored: the
reward reads the file the agent left on disk. An agent that explains a correct pipeline
in prose and never runs it scores zero, which is the property that makes this an agentic
environment rather than a reading comprehension test with extra steps.

The CSV carries four defects, each of which silently corrupts a naive aggregation:

- **A `-999.0` sentinel** for missing readings. Averaged in rather than dropped, it drags
  a daily mean to absurdity — a large, obvious error.
- **Fahrenheit rows**, marked only in a `unit` column. Ignored, they bias the mean by a
  plausible-looking amount — a small, *non*-obvious error, and the one most likely to
  survive an agent's own sanity check.
- **Duplicate timestamps**, where the first occurrence wins. Re-weighted, they shift the
  mean slightly and break the counts.
- **Shuffled row order**, so "first occurrence" means first in file order, not earliest
  by time. This is stated explicitly in the prompt; it is a spec to follow, not a trick.

Reward is split three ways so a partially working pipeline is diagnosable: getting the
counts right while the means are wrong points at aggregation, the reverse points at
parsing. Each component is exact — a data pipeline that is approximately right about how
many records it processed is not right.
"""

import csv
import io
import json
import random
from collections.abc import Iterator
from datetime import date, timedelta

import verifiers.v1 as vf

from sci_pipeline_v1.reference import SENTINEL, reduce_readings

WORKDIR = "/workspace"
INPUT_NAME = "readings.csv"
OUTPUT_NAME = "result.json"
TOLERANCE_C = 0.01


class SciData(vf.TaskData):
    csv_text: str
    """The exact CSV written into the sandbox."""
    expected: dict
    """Reference output, derived from `csv_text` by `reference.reduce_readings`."""
    station: str


class SciTaskConfig(vf.TaskConfig):
    tolerance_c: float = TOLERANCE_C
    """Absolute tolerance, in Celsius, for the mean and extreme comparisons."""


class SciTask(vf.Task[SciData, vf.State, SciTaskConfig]):
    # The agent needs a filesystem and a shell, so the rollout needs a container.
    NEEDS_CONTAINER = True

    @property
    def workdir(self) -> str:
        """The task's own workdir. Read from `TaskData` so a run can relocate it."""
        return self.data.workdir or WORKDIR

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        """Materialise the workspace: exactly one input file, nothing else.

        Only the CSV is written. An extra README or a stray helper script in the
        workspace would reach the agent as unstated context and change the task.
        """
        await runtime.write(f"{self.workdir}/{INPUT_NAME}", self.data.csv_text.encode())

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        """Capture `result.json` while the sandbox is still alive.

        Reading happens here rather than inside a reward because rewards may run after
        the runtime is released. Both the parsed submission and any failure reason land
        in `trace.info`, so a failed rollout can be diagnosed from the trace alone.
        """
        try:
            raw = await runtime.read(f"{self.workdir}/{OUTPUT_NAME}", max_bytes=2_000_000)
        except Exception as exc:  # noqa: BLE001 - a missing artifact is a normal outcome, not a crash
            trace.info["submission_error"] = f"could not read {OUTPUT_NAME}: {exc}"
            return
        try:
            trace.info["submission"] = json.loads(raw.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            trace.info["submission_error"] = f"{OUTPUT_NAME} is not valid JSON: {exc}"

    def _submission(self, trace: vf.Trace) -> dict | None:
        sub = trace.info.get("submission")
        return sub if isinstance(sub, dict) else None

    @vf.reward(weight=0.5)
    async def daily_means(self, trace: vf.Trace) -> float:
        """Every calendar day present, and every mean within tolerance. All or nothing."""
        sub = self._submission(trace)
        if sub is None or not isinstance(sub.get("daily_mean_c"), dict):
            return 0.0
        expected = self.data.expected["daily_mean_c"]
        got = sub["daily_mean_c"]
        if set(got) != set(expected):
            return 0.0
        try:
            return float(all(abs(float(got[d]) - v) <= self.config.tolerance_c for d, v in expected.items()))
        except (TypeError, ValueError):
            return 0.0

    @vf.reward(weight=0.25)
    async def record_counts(self, trace: vf.Trace) -> float:
        """`n_valid` and `n_missing` exactly — these are counts, not estimates."""
        sub = self._submission(trace)
        if sub is None:
            return 0.0
        try:
            return float(
                int(sub["n_valid"]) == self.data.expected["n_valid"]
                and int(sub["n_missing"]) == self.data.expected["n_missing"]
            )
        except (KeyError, TypeError, ValueError):
            return 0.0

    @vf.reward(weight=0.25)
    async def extremes(self, trace: vf.Trace) -> float:
        """`max_c` and `min_c` within tolerance — catches unit and sentinel handling directly."""
        sub = self._submission(trace)
        if sub is None:
            return 0.0
        try:
            return float(
                abs(float(sub["max_c"]) - self.data.expected["max_c"]) <= self.config.tolerance_c
                and abs(float(sub["min_c"]) - self.data.expected["min_c"]) <= self.config.tolerance_c
            )
        except (KeyError, TypeError, ValueError):
            return 0.0

    @vf.metric
    async def produced_output(self, trace: vf.Trace) -> float:
        """1.0 if a parseable result.json existed at all — separates 'no run' from 'wrong run'."""
        return float(self._submission(trace) is not None)

    @vf.metric
    async def day_coverage(self, trace: vf.Trace) -> float:
        """Share of expected days present, regardless of value. Catches truncated output."""
        sub = self._submission(trace)
        expected = self.data.expected["daily_mean_c"]
        if sub is None or not isinstance(sub.get("daily_mean_c"), dict) or not expected:
            return 0.0
        return len(set(sub["daily_mean_c"]) & set(expected)) / len(expected)

    async def validate(self, runtime: vf.Runtime) -> bool:
        """Model-free: the key must be reproducible from the CSV the agent will receive."""
        rows = list(csv.DictReader(io.StringIO(self.data.csv_text)))
        assert reduce_readings(rows) == self.data.expected, f"key drift on idx={self.data.idx}"
        assert self.data.expected["n_valid"] > 0, "a task with no valid readings is ungradeable"
        return True


class SciConfig(vf.TasksetConfig):
    num_days: int = 5
    """Calendar days of hourly readings per task."""
    missing_rate: float = 0.06
    """Share of readings replaced by the -999.0 sentinel."""
    fahrenheit_rate: float = 0.15
    """Share of readings recorded in Fahrenheit."""
    duplicate_rate: float = 0.05
    """Share of readings re-emitted under a timestamp that already appeared."""
    seed: int = 20260826
    task: SciTaskConfig = SciTaskConfig()


class SciPipelineTaskset(vf.Taskset[SciTask, SciConfig]):
    """Generates messy sensor exports forever; bound a run with `-n`."""

    INFINITE = True

    def load(self) -> Iterator[SciTask]:
        c = self.config
        assert c.num_days >= 1, "need at least one day of readings"
        rng = random.Random(c.seed)
        for idx in range(10**9):
            yield SciTask(self._build(idx, rng), c.task)

    def _build(self, idx: int, rng: random.Random) -> SciData:
        c = self.config
        station = f"{rng.choice(['ARN', 'BOM', 'CDU', 'MEL', 'DRW'])}-{rng.randint(100, 999)}"
        start = date(2025, rng.randint(1, 12), rng.randint(1, 20))
        rows = []
        for day_offset in range(c.num_days):
            day = start + timedelta(days=day_offset)
            # A plausible diurnal cycle around a per-day baseline; the physics does not
            # matter to grading, but realistic values keep the task from looking synthetic.
            baseline = rng.uniform(8.0, 30.0)
            for hour in range(24):
                celsius = baseline + 6.0 * ((hour - 5) % 24 < 12) * rng.uniform(0.4, 1.0) + rng.gauss(0, 0.7)
                ts = f"{day.isoformat()}T{hour:02d}:00:00"
                if rng.random() < c.missing_rate:
                    rows.append({"timestamp": ts, "station": station, "value": f"{SENTINEL}", "unit": "C"})
                elif rng.random() < c.fahrenheit_rate:
                    rows.append({"timestamp": ts, "station": station,
                                 "value": f"{celsius * 9.0 / 5.0 + 32.0:.4f}", "unit": "F"})
                else:
                    rows.append({"timestamp": ts, "station": station, "value": f"{celsius:.4f}", "unit": "C"})

        # Duplicates are appended after the originals, then the whole file is shuffled, so
        # "first in file order" genuinely differs from "earliest timestamp".
        for src in rng.sample(rows, max(1, int(len(rows) * c.duplicate_rate))):
            clone = dict(src)
            clone["value"] = f"{float(src['value']) + rng.uniform(-2, 2):.4f}" if float(src["value"]) != SENTINEL else f"{SENTINEL}"
            rows.append(clone)
        rng.shuffle(rows)

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["timestamp", "station", "value", "unit"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        csv_text = buf.getvalue()

        return SciData(
            idx=idx,
            workdir=WORKDIR,
            prompt=self._prompt(station),
            csv_text=csv_text,
            expected=reduce_readings(list(csv.DictReader(io.StringIO(csv_text)))),
            station=station,
        )

    def _prompt(self, station: str) -> str:
        return f"""`{WORKDIR}/{INPUT_NAME}` is an hourly export from weather station {station}, with \
columns `timestamp`, `station`, `value`, `unit`. The export is not clean.

Process it in exactly this order:

1. **Deduplicate on `timestamp`, keeping the first row for each timestamp in file order.** \
Rows are not sorted by time, so "first" means first in the file, not earliest.
2. Treat a `value` of exactly `{SENTINEL}` as a missing reading in any unit. Count it, then exclude it.
3. Convert every row whose `unit` is `F` to Celsius using `(F - 32) * 5 / 9`. Rows with unit `C` are \
already Celsius.
4. Compute, over the surviving readings, the mean Celsius temperature for each calendar date.

Round every temperature to 2 decimal places, half-up, and only at the very end.

Write your answer to `{WORKDIR}/{OUTPUT_NAME}` as JSON in exactly this shape:

```json
{{
  "daily_mean_c": {{"2025-01-01": 18.42, "2025-01-02": 19.07}},
  "n_valid": 118,
  "n_missing": 6,
  "max_c": 27.31,
  "min_c": 9.84
}}
```

`n_valid` is how many readings remain after deduplication and removing missing ones. `n_missing` is \
how many `{SENTINEL}` readings remain after deduplication. `max_c` and `min_c` are over the surviving \
readings, in Celsius.

You are graded solely on the contents of `{WORKDIR}/{OUTPUT_NAME}`. Write the file."""
