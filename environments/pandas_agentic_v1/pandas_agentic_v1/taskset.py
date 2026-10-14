"""pandas-agentic-v1 — sandboxed data-cleaning work, graded on the artifact the agent writes.

Every task is a messy order export the agent has to clean before it can answer:

- ``repair``     messy CSV (sentinels, dtype chaos, dupes) -> clean typed CSV
- ``reconcile``  per-group aggregation with nulls counted, not averaged in
- ``merge``      left-join audit: joined row count + unmatched keys on both sides

Rules every family follows:

- **The artifact is the answer.** The agent writes ``result.json`` / ``clean.csv`` in
  the sandbox; prose scores zero.
- **The grader never imports pandas.** References are stdlib-only, so pandas' own
  conventions (dtype coercion, groupby dropna, merge duplication) are what is being
  tested, not silently shared.
- **Procedural and INFINITE** with a fixed seed catalogue: the first 60 tasks are the
  stable benchmark slice; the stream continues forever for training.
"""

import csv
import io
import json
import random
from collections.abc import Iterator
from typing import Literal

import verifiers.v1 as vf

from pandas_agentic_v1.reference import (
    SENTINELS,
    group_aggregate,
    left_join_unmatched,
    repair_and_typecast,
)

WORKDIR = "/workspace"
# The runtimes' default image (python:3.11-slim) ships no pandas; this one does, for amd64 and arm64.
IMAGE = "amancevice/pandas:slim-3.0.5"
Family = Literal["repair", "reconcile", "merge"]

CATEGORIES = ["hardware", "produce", "freight", "software", "apparel", "chemicals"]
REGIONS = ["north", "south", "east", "west", "central"]

MISSING_DESC = f"Missing values appear as any of {sorted(SENTINELS - {''})} or empty cells"


def _export_intro(path: str, columns: list[str]) -> str:
    return (
        f"`{path}` is a messy order export: columns {', '.join(columns)}. {MISSING_DESC}; numbers may "
        "carry thousands separators or currency symbols; some rows are exact duplicates."
    )


class PandasData(vf.TaskData):
    family: Family
    input_files: dict[str, str]
    """filename -> exact file body materialised into the sandbox."""
    expected: dict
    """Answer key derived by the pandas-free reference from the same file bodies."""
    output_file: str
    spec: dict


class PandasTaskConfig(vf.TaskConfig):
    float_tolerance: float = 0.001
    """Absolute slack on every float comparison. Kept well below 0.005 so a banker's-rounded
    value (0.01 off) and an unrounded x.xx5 value (0.005 off) both fail regardless of float noise."""


def _same_count(value, want: int) -> bool:
    """Equal as a number: 8 and 8.0 match 8; 8.9, NaN and Infinity do not. `int()` would
    truncate 8.9 to 8 and raise OverflowError on Infinity, which `json.loads` accepts."""
    return float(value) == want


class PandasTask(vf.Task[PandasData, vf.State, PandasTaskConfig]):
    NEEDS_CONTAINER = True

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        workdir = self.data.workdir or WORKDIR
        for name, body in self.data.input_files.items():
            await runtime.write(f"{workdir}/{name}", body.encode())

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        workdir = self.data.workdir or WORKDIR
        try:
            raw = await runtime.read(f"{workdir}/{self.data.output_file}", max_bytes=4_000_000)
        except Exception as exc:  # noqa: BLE001 - missing artifact is an outcome, not a crash
            trace.info["submission_error"] = f"could not read {self.data.output_file}: {exc}"
            return
        text = raw.decode("utf-8", errors="replace")
        if self.data.output_file.endswith(".json"):
            try:
                trace.info["submission"] = json.loads(text)
            except json.JSONDecodeError as exc:
                trace.info["submission_error"] = f"invalid JSON: {exc}"
        else:
            trace.info["submission_csv"] = text

    # ---- scoring ---------------------------------------------------------------

    @vf.reward(weight=1.0)
    async def artifact_correct(self, trace: vf.Trace) -> float:
        if self.data.family == "repair":
            return self._score_repair(trace)
        sub = trace.info.get("submission")
        if not isinstance(sub, dict):
            return 0.0
        if self.data.family == "reconcile":
            return self._score_reconcile(sub)
        return self._score_merge(sub)

    def _close(self, value, want: float) -> bool:
        """An acceptance test, not a rejection test: NaN fails every comparison, so
        `abs(x - want) > tol` would let it through while `<= tol` turns it away."""
        return abs(float(value) - want) <= self.config.float_tolerance

    def _score_repair(self, trace: vf.Trace) -> float:
        text = trace.info.get("submission_csv")
        if not text:
            return 0.0
        try:
            got = list(csv.DictReader(io.StringIO(text)))
        except csv.Error:
            return 0.0
        expected = self.data.expected["rows"]
        if len(got) != len(expected):
            return 0.0
        spec = self.data.spec["columns"]
        for g, e in zip(got, expected):
            for col, kind in spec.items():
                raw_value = (g.get(col) or "").strip()
                want = e[col]
                if want is None:
                    if raw_value != "":
                        return 0.0
                    continue
                try:
                    if kind == "int" and int(raw_value) != want:
                        return 0.0
                    if kind == "float" and not self._close(raw_value, want):
                        return 0.0
                    if kind == "str" and raw_value != want:
                        return 0.0
                except ValueError:
                    return 0.0
        return 1.0

    def _score_reconcile(self, sub: dict) -> float:
        expected = self.data.expected["groups"]
        if set(sub) != set(expected):
            return 0.0
        for group, want in expected.items():
            got = sub[group]
            if not isinstance(got, dict):
                return 0.0
            try:
                if not (_same_count(got["count"], want["count"]) and _same_count(got["nulls"], want["nulls"])):
                    return 0.0
                if not self._close(got["sum"], want["sum"]):
                    return 0.0
                if want["mean"] is None:
                    if got["mean"] is not None:
                        return 0.0
                elif not self._close(got["mean"], want["mean"]):
                    return 0.0
            except (KeyError, TypeError, ValueError):
                return 0.0
        return 1.0

    def _score_merge(self, sub: dict) -> float:
        want = self.data.expected
        try:
            return float(
                _same_count(sub["joined_rows"], want["joined_rows"])
                and sorted(map(str, sub["left_only_keys"])) == want["left_only_keys"]
                and sorted(map(str, sub["right_only_keys"])) == want["right_only_keys"]
            )
        except (KeyError, TypeError, ValueError):
            return 0.0

    @vf.metric
    async def produced_output(self, trace: vf.Trace) -> float:
        return float("submission" in trace.info or "submission_csv" in trace.info)

    async def validate(self, runtime: vf.Runtime | None) -> bool:
        """Answer key must be re-derivable from the shipped file bodies, and the sandbox
        must import pandas — the runtimes' default image does not."""
        d = self.data

        def _rows(name: str) -> list[dict]:
            return list(csv.DictReader(io.StringIO(d.input_files[name])))

        if d.family == "repair":
            assert repair_and_typecast(_rows("raw.csv"), d.spec) == d.expected["rows"]
        elif d.family == "reconcile":
            cleaned = repair_and_typecast(_rows("raw.csv"), d.spec)
            assert group_aggregate(cleaned, d.spec["by"], d.spec["value"]) == d.expected["groups"]
        else:
            left = repair_and_typecast(_rows("left.csv"), {"columns": d.spec["left_columns"]})
            right = repair_and_typecast(_rows("right.csv"), {"columns": d.spec["right_columns"]})
            assert left_join_unmatched(left, right, d.spec["on"]) == d.expected
        if runtime is not None:
            probe = await runtime.run(["python3", "-c", "import pandas"], {})
            assert probe.exit_code == 0, f"pandas does not import in {d.image}: {probe.stderr.strip()[-300:]}"
        return True


class PandasConfig(vf.TasksetConfig):
    families: list[Family] = ["repair", "reconcile", "merge"]
    num_rows: int = 60
    """Rows per generated table; raise for longer-context difficulty."""
    mess_rate: float = 0.18
    """Share of cells corrupted with sentinels / formatting noise."""
    seed: int = 20260827
    task: PandasTaskConfig = PandasTaskConfig()


class PandasAgenticTaskset(vf.Taskset[PandasTask, PandasConfig]):
    """First 60 tasks (fixed seed) are the stable benchmark slice; stream is infinite."""

    INFINITE = True

    def load(self) -> Iterator[PandasTask]:
        c = self.config
        assert c.families, "need at least one family"
        rng = random.Random(c.seed)
        for idx in range(10**9):
            family: Family = c.families[idx % len(c.families)]
            build = {"repair": self._build_repair, "reconcile": self._build_reconcile, "merge": self._build_merge}[
                family
            ]
            yield PandasTask(build(idx, rng), c.task)

    # ---- generation ------------------------------------------------------------

    def _messy_number(self, rng: random.Random, value: float, kind: str) -> str:
        roll = rng.random()
        if roll < 0.25:
            return f"{value:,.2f}" if kind == "float" else f"{int(value):,}"
        if roll < 0.35 and kind == "float":
            return f"${value:.2f}"
        return f"{value:.2f}" if kind == "float" else str(int(value))

    def _raw_table(self, rng: random.Random, n: int) -> tuple[list[dict], dict]:
        spec = {
            "columns": {"order_id": "str", "category": "str", "region": "str", "units": "int", "unit_price": "float"}
        }
        rows = []
        for i in range(n):
            rows.append(
                {
                    "order_id": f"ORD-{1000 + i}",
                    "category": rng.choice(CATEGORIES),
                    "region": rng.choice(REGIONS),
                    "units": self._messy_number(rng, rng.randint(1, 8000), "int"),
                    "unit_price": self._messy_number(rng, rng.uniform(0.5, 9000), "float"),
                }
            )
        mess = self.config.mess_rate
        for row in rows:
            for col in ("units", "unit_price", "region"):
                if rng.random() < mess:
                    row[col] = rng.choice(sorted(SENTINELS))
        for _ in range(max(1, int(n * 0.06))):  # exact duplicates
            rows.insert(rng.randrange(len(rows)), dict(rng.choice(rows)))
        return rows, spec

    def _csv(self, rows: list[dict]) -> str:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue()

    def _build_repair(self, idx: int, rng: random.Random) -> PandasData:
        rows, spec = self._raw_table(rng, self.config.num_rows)
        body = self._csv(rows)
        expected = repair_and_typecast(list(csv.DictReader(io.StringIO(body))), spec)
        prompt = (
            f"{_export_intro(f'{WORKDIR}/raw.csv', list(spec['columns']))}\n\n"
            "Clean it (pandas is available in the sandbox):\n"
            "1. Drop exact-duplicate rows, keeping the first occurrence in file order.\n"
            "2. Normalise every missing-value marker to an EMPTY cell.\n"
            "3. Coerce `units` to integers and `unit_price` to floats rounded half-up to 2 decimals; "
            "strip separators and currency symbols.\n\n"
            f"Write the cleaned table to `{WORKDIR}/clean.csv` with the same header, same row order. "
            "You are graded solely on that file."
        )
        return PandasData(
            idx=idx,
            image=IMAGE,
            workdir=WORKDIR,
            prompt=prompt,
            family="repair",
            input_files={"raw.csv": body},
            expected={"rows": expected},
            output_file="clean.csv",
            spec=spec,
        )

    def _build_reconcile(self, idx: int, rng: random.Random) -> PandasData:
        rows, spec_base = self._raw_table(rng, self.config.num_rows)
        body = self._csv(rows)
        spec = {**spec_base, "by": "category", "value": "unit_price"}
        cleaned = repair_and_typecast(list(csv.DictReader(io.StringIO(body))), spec_base)
        expected = group_aggregate(cleaned, "category", "unit_price")
        prompt = (
            f"{_export_intro(f'{WORKDIR}/raw.csv', list(spec_base['columns']))}\n\n"
            "After cleaning it (drop exact duplicates keeping the first, treat missing markers as missing, "
            "parse numbers), reconcile `unit_price` by `category`:\n\n"
            "For each category present, report count (non-missing prices), nulls (missing prices), "
            "sum and mean of prices, rounded half-up to 2 decimals at the end. A category whose "
            "prices are all missing has count 0, sum 0.0 and mean null.\n\n"
            f'Write `{WORKDIR}/result.json` shaped like: {{"hardware": {{"count": 12, "nulls": 2, '
            '"sum": 4210.55, "mean": 350.88}}. You are graded solely on that file.'
        )
        return PandasData(
            idx=idx,
            image=IMAGE,
            workdir=WORKDIR,
            prompt=prompt,
            family="reconcile",
            input_files={"raw.csv": body},
            expected={"groups": expected},
            output_file="result.json",
            spec=spec,
        )

    def _build_merge(self, idx: int, rng: random.Random) -> PandasData:
        left_rows, left_spec = self._raw_table(rng, self.config.num_rows)
        right = [
            {
                "order_id": f"ORD-{1000 + i}",
                "shipment_id": f"SHP-{rng.randint(10000, 99999)}",
                "carrier": rng.choice(["auspost", "tnt", "dhl", "startrack"]),
            }
            for i in rng.sample(range(self.config.num_rows + 20), self.config.num_rows // 2)
        ]
        for r in rng.sample(right, max(1, len(right) // 10)):  # a second shipment for the same order
            right.append({**r, "shipment_id": f"{r['shipment_id']}-2"})
        rng.shuffle(right)
        right_spec = {"order_id": "str", "shipment_id": "str", "carrier": "str"}
        left_body, right_body = self._csv(left_rows), self._csv(right)
        left_clean = repair_and_typecast(list(csv.DictReader(io.StringIO(left_body))), left_spec)
        right_clean = repair_and_typecast(list(csv.DictReader(io.StringIO(right_body))), {"columns": right_spec})
        expected = left_join_unmatched(left_clean, right_clean, "order_id")
        prompt = (
            f"{_export_intro(f'{WORKDIR}/left.csv', list(left_spec['columns']))} "
            f"`{WORKDIR}/right.csv` holds shipments: columns {', '.join(right_spec)}. An order may have "
            "several shipments, and a shipment may reference an order that does not exist.\n\n"
            "Clean both (drop exact duplicates keeping the first, treat missing markers as missing), then "
            "audit a LEFT join of orders to shipments on `order_id`:\n\n"
            "- `joined_rows`: row count of the left join (an order with several shipment rows contributes "
            "one joined row per shipment)\n"
            "- `left_only_keys`: order_ids with no shipment\n"
            "- `right_only_keys`: order_ids appearing only in shipments\n\n"
            f"Write `{WORKDIR}/result.json` with exactly those three keys. You are graded solely on that file."
        )
        return PandasData(
            idx=idx,
            image=IMAGE,
            workdir=WORKDIR,
            prompt=prompt,
            family="merge",
            input_files={"left.csv": left_body, "right.csv": right_body},
            expected=expected,
            output_file="result.json",
            spec={"left_columns": left_spec["columns"], "right_columns": right_spec, "on": "order_id"},
        )
