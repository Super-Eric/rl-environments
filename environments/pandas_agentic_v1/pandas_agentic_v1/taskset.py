"""pandas-agentic-v1 — sandboxed agentic pandas work, graded on the artifact.

Built for the Environments Hub "Software Library Evals" contract: sandboxed code
execution per task, 50+ nontrivial agentic tasks against a library the author uses
daily, suitable for prime-rl training. Three families ship now; the roadmap in
README.md takes the catalogue to eight.

- ``repair``     messy CSV (sentinels, dtype chaos, dupes) -> clean typed CSV
- ``reconcile``  per-group aggregation with nulls counted, not averaged in
- ``merge``      left-join audit: joined row count + unmatched keys on both sides

Every family follows the house rules proven in sci-pipeline-v1:

- **The artifact is the answer.** The agent writes ``result.json`` / ``clean.csv`` in
  the sandbox; prose scores zero.
- **The grader never imports pandas.** References are stdlib-only, so pandas' own
  conventions (dtype coercion, groupby dropna, merge duplication) are what is being
  tested, not silently shared.
- **Procedural and INFINITE** with a fixed seed catalogue: the first 60 tasks are the
  stable "benchmark" slice (>50, per the contract); the stream continues forever for
  training.
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
Family = Literal["repair", "reconcile", "merge"]

CATEGORIES = ["hardware", "produce", "freight", "software", "apparel", "chemicals"]
REGIONS = ["north", "south", "east", "west", "central"]


class PandasData(vf.TaskData):
    family: Family
    input_files: dict[str, str]
    """filename -> exact file body materialised into the sandbox."""
    expected: dict
    """Answer key derived by the pandas-free reference from the same file bodies."""
    output_file: str
    spec: dict


class PandasTaskConfig(vf.TaskConfig):
    float_tolerance: float = 0.01


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
                    if raw_value not in ("", *SENTINELS):
                        return 0.0
                    continue
                try:
                    if kind == "int" and int(raw_value) != want:
                        return 0.0
                    if kind == "float" and abs(float(raw_value) - want) > self.config.float_tolerance:
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
                if int(got["count"]) != want["count"] or int(got["nulls"]) != want["nulls"]:
                    return 0.0
                if abs(float(got["sum"]) - want["sum"]) > self.config.float_tolerance:
                    return 0.0
                if want["mean"] is None:
                    if got["mean"] is not None:
                        return 0.0
                elif abs(float(got["mean"]) - want["mean"]) > self.config.float_tolerance:
                    return 0.0
            except (KeyError, TypeError, ValueError):
                return 0.0
        return 1.0

    def _score_merge(self, sub: dict) -> float:
        want = self.data.expected
        try:
            return float(
                int(sub["joined_rows"]) == want["joined_rows"]
                and sorted(map(str, sub["left_only_keys"])) == want["left_only_keys"]
                and sorted(map(str, sub["right_only_keys"])) == want["right_only_keys"]
            )
        except (KeyError, TypeError, ValueError):
            return 0.0

    @vf.metric
    async def produced_output(self, trace: vf.Trace) -> float:
        return float("submission" in trace.info or "submission_csv" in trace.info)

    async def validate(self, runtime: vf.Runtime) -> bool:
        """Answer key must be re-derivable from the shipped file bodies."""
        d = self.data
        parse = lambda name: list(csv.DictReader(io.StringIO(d.input_files[name])))  # noqa: E731
        if d.family == "repair":
            assert repair_and_typecast(parse("raw.csv"), d.spec) == d.expected["rows"]
        elif d.family == "reconcile":
            cleaned = repair_and_typecast(parse("raw.csv"), d.spec)
            assert group_aggregate(cleaned, d.spec["by"], d.spec["value"]) == d.expected["groups"]
        else:
            left = repair_and_typecast(parse("left.csv"), {"columns": d.spec["left_columns"]})
            right = repair_and_typecast(parse("right.csv"), {"columns": d.spec["right_columns"]})
            assert left_join_unmatched(left, right, d.spec["on"]) == d.expected
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
            build = {"repair": self._build_repair, "reconcile": self._build_reconcile,
                     "merge": self._build_merge}[family]
            yield PandasTask(build(idx, rng), c.task)

    # ---- generation ------------------------------------------------------------

    def _messy_number(self, rng, value: float, kind: str) -> str:
        roll = rng.random()
        if roll < 0.25:
            return f"{value:,.2f}" if kind == "float" else f"{int(value):,}"
        if roll < 0.35 and kind == "float":
            return f"${value:.2f}"
        return f"{value:.2f}" if kind == "float" else str(int(value))

    def _raw_table(self, rng, n: int) -> tuple[list[dict], dict]:
        spec = {"columns": {"order_id": "str", "category": "str", "region": "str",
                            "units": "int", "unit_price": "float"}}
        rows = []
        for i in range(n):
            rows.append({
                "order_id": f"ORD-{1000 + i}",
                "category": rng.choice(CATEGORIES),
                "region": rng.choice(REGIONS),
                "units": self._messy_number(rng, rng.randint(1, 8000), "int"),
                "unit_price": self._messy_number(rng, rng.uniform(0.5, 9000), "float"),
            })
        mess = self.config.mess_rate
        for row in rows:
            for col in ("units", "unit_price", "region"):
                if rng.random() < mess:
                    row[col] = rng.choice(sorted(SENTINELS))
        for _ in range(max(1, int(n * 0.06))):          # exact duplicates
            rows.insert(rng.randrange(len(rows)), dict(rng.choice(rows)))
        return rows, spec

    def _csv(self, rows: list[dict]) -> str:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue()

    def _build_repair(self, idx, rng) -> PandasData:
        rows, spec = self._raw_table(rng, self.config.num_rows)
        body = self._csv(rows)
        expected = repair_and_typecast(list(csv.DictReader(io.StringIO(body))), spec)
        prompt = (
            f"`{WORKDIR}/raw.csv` is a messy order export: columns {', '.join(spec['columns'])}. "
            f"Missing values appear as any of {sorted(SENTINELS - {''})} or empty cells; numbers may "
            "carry thousands separators or currency symbols; some rows are exact duplicates.\n\n"
            "Clean it with pandas (or any tool available in the sandbox):\n"
            "1. Drop exact-duplicate rows, keeping the first occurrence in file order.\n"
            "2. Normalise every missing-value marker to an EMPTY cell.\n"
            "3. Coerce `units` to integers and `unit_price` to floats rounded half-up to 2 decimals; "
            "strip separators and currency symbols.\n\n"
            f"Write the cleaned table to `{WORKDIR}/clean.csv` with the same header, same row order. "
            "You are graded solely on that file."
        )
        return PandasData(idx=idx, workdir=WORKDIR, prompt=prompt, family="repair",
                          input_files={"raw.csv": body}, expected={"rows": expected},
                          output_file="clean.csv", spec=spec)

    def _build_reconcile(self, idx, rng) -> PandasData:
        rows, spec_base = self._raw_table(rng, self.config.num_rows)
        body = self._csv(rows)
        spec = {**spec_base, "by": "category", "value": "unit_price"}
        cleaned = repair_and_typecast(list(csv.DictReader(io.StringIO(body))), spec_base)
        expected = group_aggregate(cleaned, "category", "unit_price")
        prompt = (
            f"`{WORKDIR}/raw.csv` is the same style of messy order export (sentinel missing values, "
            "formatted numbers, duplicate rows). After cleaning it (drop exact duplicates keeping "
            "the first, treat sentinels as missing, parse numbers), reconcile `unit_price` by "
            "`category`:\n\n"
            "For each category present, report count (non-missing prices), nulls (missing prices), "
            "sum and mean of prices, rounded half-up to 2 decimals at the end. A category whose "
            "prices are all missing has mean null.\n\n"
            f'Write `{WORKDIR}/result.json` shaped like: {{"hardware": {{"count": 12, "nulls": 2, '
            '"sum": 4210.55, "mean": 350.88}}}}. You are graded solely on that file.'
        )
        return PandasData(idx=idx, workdir=WORKDIR, prompt=prompt, family="reconcile",
                          input_files={"raw.csv": body}, expected={"groups": expected},
                          output_file="result.json", spec=spec)

    def _build_merge(self, idx, rng) -> PandasData:
        left_rows, left_spec = self._raw_table(rng, self.config.num_rows)
        right = [{"order_id": f"ORD-{1000 + i}", "shipment_id": f"SHP-{rng.randint(10000, 99999)}",
                  "carrier": rng.choice(["auspost", "tnt", "dhl", "startrack"])}
                 for i in rng.sample(range(self.config.num_rows + 20), self.config.num_rows // 2)]
        for r in rng.sample(right, max(1, len(right) // 10)):     # duplicated shipment rows
            right.append(dict(r))
        rng.shuffle(right)
        right_spec = {"order_id": "str", "shipment_id": "str", "carrier": "str"}
        left_body, right_body = self._csv(left_rows), self._csv(right)
        left_clean = repair_and_typecast(list(csv.DictReader(io.StringIO(left_body))), left_spec)
        right_clean = repair_and_typecast(list(csv.DictReader(io.StringIO(right_body))),
                                          {"columns": right_spec})
        expected = left_join_unmatched(left_clean, right_clean, "order_id")
        prompt = (
            f"`{WORKDIR}/left.csv` holds orders; `{WORKDIR}/right.csv` holds shipments (possibly "
            "duplicated, possibly for orders that do not exist). Clean both (drop exact duplicates "
            "keeping the first, sentinels are missing), then audit a LEFT join of orders to "
            "shipments on `order_id`:\n\n"
            "- `joined_rows`: row count of the left join (a duplicated shipment multiplies its order's rows)\n"
            "- `left_only_keys`: sorted order_ids with no shipment\n"
            "- `right_only_keys`: sorted order_ids appearing only in shipments\n\n"
            f"Write `{WORKDIR}/result.json` with exactly those three keys. "
            "You are graded solely on that file."
        )
        return PandasData(idx=idx, workdir=WORKDIR, prompt=prompt, family="merge",
                          input_files={"left.csv": left_body, "right.csv": right_body},
                          expected=expected, output_file="result.json",
                          spec={"left_columns": left_spec["columns"], "right_columns": right_spec,
                                "on": "order_id"})
