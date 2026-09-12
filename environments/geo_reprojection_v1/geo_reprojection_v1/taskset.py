"""geo-reprojection-v1 — coordinate reprojection and geodesy, verified against PROJ.

Four procedurally generated task families, all graded against `pyproj` (PROJ) rather
than a stored answer key:

- `reproject`  — transform a geographic coordinate into a projected CRS (metres).
- `inverse`    — transform projected easting/northing back to longitude/latitude.
- `distance`   — geodesic distance between two points on the ellipsoid.
- `bearing`    — initial (forward) geodesic azimuth from the first point to the second.

Why this shape:

- **Procedural and `INFINITE`.** Coordinates are drawn at random to six decimal places
  inside each CRS's true area of use, so the task stream never repeats and cannot be
  memorised from a leaked answer key. Runs bound it with `-n`.
- **Deterministic verification.** PROJ is the reference implementation the geospatial
  world actually uses; the reward is agreement with it, not with an LLM judge. The datum
  shift behind every non-WGS 84 grid is pinned to one named EPSG operation
  (`PINNED_DATUM_SHIFT`), so the key does not depend on which grids the grading host has.
- **Binary reward, graded metric.** The reward is a hard pass/fail against a physical
  tolerance (metres, or degrees for bearing). Partial credit on a numeric answer is a
  reward-hacking surface: a model that emits the input unchanged, or a plausible-looking
  number of the right magnitude, should score zero rather than bank a fraction. The
  continuous error is still recorded as a *metric* so training runs keep a dense
  learning signal to inspect. Set `graded = true` to fold that curve into the reward
  when a run explicitly wants shaping.
- **Tolerance is physical, not textual.** Errors are compared in the destination grid's
  own metres (ground metres to within the 0.9996 scale factor on the transverse Mercator
  grids, up to about 2.9x ground metres for Web Mercator at the +/-70 deg edge of its
  sampling box), on the ellipsoid for inverse rows, and in degrees of azimuth — so
  precision requirements do not vary with the magnitude of the printed coordinate.

Ground truth is checked without a model in `validate()`: every row's answer key is
re-derived from the shipped inputs — the same numbers the model reads — and every
coordinate is round-tripped through PROJ, so a generator bug or a coordinate drawn
outside its CRS's valid domain is rejected before a model ever sees it.
"""

import itertools
import math
import random
import re
from collections.abc import Iterator
from functools import lru_cache
from typing import Literal

import verifiers.v1 as vf
from pyproj import CRS, Geod, Transformer
from pyproj.transformer import TransformerGroup

WGS84 = "EPSG:4326"
GEOD = Geod(ellps="WGS84")
# The farthest apart two points on the WGS84 ellipsoid can be: half the meridional
# circumference. Stands in for an error PROJ cannot evaluate (see `GeoTask._error`).
MAX_GEODESIC_M = 20_003_931.5

Family = Literal["reproject", "inverse", "distance", "bearing"]

# Projected CRSs paired with the lon/lat box they are actually valid inside. Sampling
# from the area of use (rather than the whole globe) is what keeps generated tasks
# physically meaningful — PROJ will happily return nonsense far outside a zone.
PROJECTED_CRS: dict[str, tuple[str, tuple[float, float, float, float]]] = {
    # GDA2020 MGA zones — the Australian national standard.
    "EPSG:7855": ("GDA2020 / MGA zone 55", (144.0, -45.0, 150.0, -28.0)),
    "EPSG:7856": ("GDA2020 / MGA zone 56", (150.0, -40.0, 156.0, -25.0)),
    "EPSG:7854": ("GDA2020 / MGA zone 54", (138.0, -45.0, 144.0, -24.0)),
    # WGS84 UTM zones.
    "EPSG:32756": ("WGS 84 / UTM zone 56S", (150.0, -40.0, 156.0, -20.0)),
    "EPSG:32633": ("WGS 84 / UTM zone 33N", (12.0, 35.0, 18.0, 60.0)),
    "EPSG:32610": ("WGS 84 / UTM zone 10N", (-126.0, 35.0, -120.0, 55.0)),
    # National / regional grids.
    "EPSG:27700": ("OSGB36 / British National Grid", (-6.0, 50.0, 1.5, 58.0)),
    "EPSG:2193": ("NZGD2000 / New Zealand Transverse Mercator", (168.0, -46.0, 178.0, -35.0)),
    "EPSG:3857": ("WGS 84 / Pseudo-Mercator", (-170.0, -70.0, 170.0, 70.0)),
}

# Datum shifts to WGS 84 are pinned to one named EPSG operation rather than left to
# PROJ's best-available choice, which depends on the grids installed on the grading host
# (or on PROJ_NETWORK) and would otherwise move the answer key for the same seed by
# metres. OSGB36 uses the 7-parameter Helmert "OSGB36 to WGS 84 (6)" (EPSG-stated
# accuracy 2 m), not the OSTN15 grid; GDA2020 and NZGD2000 use EPSG's null
# transformation, i.e. they are treated as coincident with WGS 84. The WGS 84 UTM zones
# and Web Mercator involve no datum shift.
PINNED_DATUM_SHIFT: dict[str, str] = {
    "EPSG:7855": "GDA2020 to WGS 84 (2)",
    "EPSG:7856": "GDA2020 to WGS 84 (2)",
    "EPSG:7854": "GDA2020 to WGS 84 (2)",
    "EPSG:27700": "OSGB36 to WGS 84 (6)",
    "EPSG:2193": "NZGD2000 to WGS 84 (1)",
}

# Told to the model on British National Grid rows: the two OSGB36 shifts in common use
# differ by metres, which is more than the position tolerance.
OSGB36_NOTE = (
    ' The OSGB36 datum shift is the EPSG 7-parameter Helmert transformation "OSGB36 to WGS 84 (6)", '
    "not the OSTN15 grid."
)

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@lru_cache(maxsize=64)
def _transformer(src: str, dst: str) -> Transformer:
    # always_xy keeps every axis order as (lon, lat) / (easting, northing), which is the
    # order the prompts state. EPSG axis order for several of these CRSs is otherwise
    # (lat, lon) and would silently transpose the ground truth.
    a, b = CRS.from_user_input(src), CRS.from_user_input(dst)
    shift = PINNED_DATUM_SHIFT.get(dst if src == WGS84 else src)
    if shift is None:
        return Transformer.from_crs(a, b, always_xy=True)
    pinned = [t for t in TransformerGroup(a, b, always_xy=True).transformers if shift in t.description]
    if len(pinned) != 1:
        raise RuntimeError(f"PROJ offers {len(pinned)} operations named {shift!r} for {src}->{dst}, expected one")
    return pinned[0]


def _parse_numbers(reply: str | None, count: int) -> list[float] | None:
    """Pull exactly `count` numbers out of the last <answer> block, or return None.

    Strict by design: no <answer> block, or the wrong number of values inside it, is a
    failed response rather than a salvage-what-you-can parse. Loose extraction over the
    whole reply would let a model score by scattering candidate numbers in prose.
    """
    if not reply:
        return None
    blocks = ANSWER_RE.findall(reply)
    if not blocks:
        return None
    values = [float(m) for m in NUMBER_RE.findall(blocks[-1].replace(",", " "))]
    return values if len(values) == count else None


def _bearing_error(got: float, expected: float) -> float:
    # Azimuths are circular: 359.9 deg and -0.1 deg are the same heading.
    return abs((got - expected + 180.0) % 360.0 - 180.0)


class GeoData(vf.TaskData):
    family: Family
    """Which of the four task families this row is."""
    src_crs: str
    """Authority code the inputs are expressed in."""
    dst_crs: str
    """Authority code the answer must be expressed in (equal to src_crs for geodesy rows)."""
    inputs: list[float]
    """The coordinate(s) exactly as the prompt prints them, in the order it states them."""
    expected: list[float]
    """PROJ's answer: [easting, northing], [lon, lat], [metres], or [degrees]."""
    tolerance: float
    """Pass threshold, in metres for lengths/positions and degrees for bearings."""


class GeoTaskConfig(vf.TaskConfig):
    graded: bool = False
    """Fold the error curve into the reward instead of a hard pass/fail (see module docstring)."""
    graded_halflife: float = 1.0
    """Error, in tolerance units, at which a graded reward decays to 0.5."""


class GeoTask(vf.Task[GeoData, vf.State, GeoTaskConfig]):
    def _error(self, trace: vf.Trace) -> float | None:
        """Physical error of the reply: metres for positions/lengths, degrees for bearings."""
        got = _parse_numbers(trace.last_reply, len(self.data.expected))
        if got is None:
            return None
        exp = self.data.expected
        if self.data.family == "bearing":
            err = _bearing_error(got[0], exp[0])
        elif self.data.family == "distance":
            err = abs(got[0] - exp[0])
        elif self.data.family == "inverse":
            # Compare in metres on the ellipsoid, not in degrees: a degree of longitude
            # is worth very different distances at different latitudes, so a degree-based
            # tolerance would be far stricter near the poles than at the equator.
            err = float(GEOD.inv(got[0], got[1], exp[0], exp[1])[2])
        else:
            err = ((got[0] - exp[0]) ** 2 + (got[1] - exp[1]) ** 2) ** 0.5
        if math.isfinite(err):
            return err
        # A latitude outside [-90, 90] has no point on the ellipsoid and an overflowing
        # literal is already inf, so PROJ hands back NaN. Count the reply as maximally
        # wrong rather than let a non-finite value poison the run-level metrics.
        return 180.0 if self.data.family == "bearing" else MAX_GEODESIC_M

    async def validate(self, runtime: vf.Runtime) -> bool:
        """Model-free ground-truth check: the answer key must re-derive from the shipped inputs.

        Runs via `uv run validate`, before any model is involved. The key is recomputed
        from `inputs` — the numbers the model reads — so a generator bug surfaces as a
        disagreement instead of a silently wrong key, and coordinate rows must also
        survive a forward-then-inverse transform, which fails for a coordinate outside its
        CRS's usable domain. Assertions raise rather than return False so a failure names
        the offending row and its drift in the report.
        """
        d = self.data
        assert d.tolerance > 0, f"non-positive tolerance on task {d.idx}"
        if d.family in ("distance", "bearing"):
            fwd_az, _, dist = GEOD.inv(*d.inputs)
            key = fwd_az if d.family == "bearing" else dist
            assert math.isfinite(key), f"non-finite ground truth for task {d.idx}"
            drift = _bearing_error(key, d.expected[0]) if d.family == "bearing" else abs(key - d.expected[0])
        else:
            px, py = _transformer(d.src_crs, d.dst_crs).transform(*d.inputs)
            assert math.isfinite(px) and math.isfinite(py), f"CRS {d.src_crs}->{d.dst_crs} is undefined at task {d.idx}"
            bx, by = _transformer(d.dst_crs, d.src_crs).transform(px, py)
            if d.family == "reproject":
                back = float(GEOD.inv(bx, by, *d.inputs)[2])
                drift = ((px - d.expected[0]) ** 2 + (py - d.expected[1]) ** 2) ** 0.5
            else:
                back = ((bx - d.inputs[0]) ** 2 + (by - d.inputs[1]) ** 2) ** 0.5
                drift = float(GEOD.inv(px, py, *d.expected)[2])
            assert back < 0.01, f"round-trip drift {back:.4f} m exceeds 1 cm for task {d.idx}"
        assert drift <= d.tolerance, f"answer key for task {d.idx} is {drift:.4f} off a fresh PROJ derivation"
        return True

    @vf.reward(weight=1.0)
    async def within_tolerance(self, trace: vf.Trace) -> float:
        err = self._error(trace)
        if err is None:
            return 0.0
        if not self.config.graded:
            return float(err <= self.data.tolerance)
        # Smooth decay, still anchored on the physical tolerance.
        return 1.0 / (1.0 + (err / (self.data.tolerance * self.config.graded_halflife)) ** 2)

    @vf.metric
    async def error(self, trace: vf.Trace) -> float:
        """Physical error in metres (degrees for bearings); -1.0 when no answer parsed."""
        err = self._error(trace)
        return -1.0 if err is None else err

    @vf.metric
    async def error_ratio(self, trace: vf.Trace) -> float:
        """Error in units of this row's tolerance (1.0 is the pass boundary), comparable across families; -1.0 when no answer parsed."""
        err = self._error(trace)
        return -1.0 if err is None else err / self.data.tolerance

    @vf.metric
    async def answered(self, trace: vf.Trace) -> float:
        """1.0 when a well-formed <answer> block was produced. Separates 'wrong' from 'unparseable'."""
        return float(self._error(trace) is not None)


class GeoConfig(vf.TasksetConfig):
    families: list[Family] = ["reproject", "inverse", "distance", "bearing"]
    """Task families to draw from; each generated row picks one uniformly."""
    position_tolerance_m: float = 1.0
    """Pass threshold for reproject/inverse rows, in metres."""
    distance_tolerance_m: float = 10.0
    """Pass threshold for geodesic distance rows, in metres."""
    bearing_tolerance_deg: float = 0.05
    """Pass threshold for bearing rows, in degrees."""
    seed: int = 20260826
    """Seed for the coordinate stream; a run with the same seed regenerates the same tasks."""
    task: GeoTaskConfig = GeoTaskConfig()


class GeoReprojectionTaskset(vf.Taskset[GeoTask, GeoConfig]):
    """Procedurally generates coordinate tasks forever; bound a run with `-n`."""

    INFINITE = True

    def load(self) -> Iterator[GeoTask]:
        c = self.config
        if not c.families:
            raise ValueError("need at least one family in `families`")
        rng = random.Random(c.seed)
        codes = list(PROJECTED_CRS)
        for idx in itertools.count():
            family: Family = rng.choice(c.families)
            code = rng.choice(codes)
            label, (w, s, e, n) = PROJECTED_CRS[code]
            lon, lat = round(rng.uniform(w, e), 6), round(rng.uniform(s, n), 6)
            yield GeoTask(self._build(idx, family, code, label, lon, lat, rng), c.task)

    def _build(
        self, idx: int, family: Family, code: str, label: str, lon: float, lat: float, rng: random.Random
    ) -> GeoData:
        # Every prompt states exactly what the reward checks — the tolerance it is graded
        # against and the parsing rule — and nothing about how to get there.
        c = self.config
        common = (
            "Put the final answer inside <answer></answer> tags; the last such block is graded. "
            "Separate values with commas or spaces and do not use thousands separators."
        )
        datum = OSGB36_NOTE if code == "EPSG:27700" else ""
        if family == "reproject":
            x, y = _transformer(WGS84, code).transform(lon, lat)
            prompt = (
                f"Transform the WGS84 geographic coordinate longitude {lon}, latitude {lat} "
                f"into {label} ({code}).{datum}\n\n"
                f"Give easting and northing in metres, easting first; an answer within "
                f"{c.position_tolerance_m:g} m of PROJ's result is accepted. {common}\n"
                f"Example: <answer>500000.00, 6100000.00</answer>"
            )
            return GeoData(
                idx=idx,
                prompt=prompt,
                family=family,
                src_crs=WGS84,
                dst_crs=code,
                inputs=[lon, lat],
                expected=[x, y],
                tolerance=c.position_tolerance_m,
            )
        if family == "inverse":
            # Stored at the millimetre the prompt prints, so validate() reads the same numbers the model does.
            x, y = (round(v, 3) for v in _transformer(WGS84, code).transform(lon, lat))
            prompt = (
                f"Transform the {label} ({code}) coordinate easting {x:.3f} m, northing {y:.3f} m "
                f"back into WGS84 geographic coordinates ({WGS84}).{datum}\n\n"
                f"Give longitude and latitude in decimal degrees, longitude first; an answer within "
                f"{c.position_tolerance_m:g} m of PROJ's result, measured on the ellipsoid, is accepted. {common}\n"
                f"Example: <answer>151.209900, -33.865143</answer>"
            )
            return GeoData(
                idx=idx,
                prompt=prompt,
                family=family,
                src_crs=code,
                dst_crs=WGS84,
                inputs=[x, y],
                expected=[lon, lat],
                tolerance=c.position_tolerance_m,
            )
        # Geodesy families: a second point offset by a random azimuth and distance.
        az, dist = rng.uniform(-180.0, 180.0), rng.uniform(1_000.0, 400_000.0)
        lon2, lat2, _ = GEOD.fwd(lon, lat, az, dist)
        lon2, lat2 = round(lon2, 6), round(lat2, 6)
        fwd_az, _, true_dist = GEOD.inv(lon, lat, lon2, lat2)
        a, b = f"A (longitude {lon}, latitude {lat})", f"B (longitude {lon2}, latitude {lat2})"
        if family == "distance":
            prompt = (
                f"Compute the geodesic distance on the WGS84 ellipsoid between {a} and {b}.\n\n"
                f"Give the distance in metres; an answer within {c.distance_tolerance_m:g} m of PROJ's result "
                f"is accepted. {common}\n"
                f"Example: <answer>123456.78</answer>"
            )
            return GeoData(
                idx=idx,
                prompt=prompt,
                family=family,
                src_crs=WGS84,
                dst_crs=WGS84,
                inputs=[lon, lat, lon2, lat2],
                expected=[true_dist],
                tolerance=c.distance_tolerance_m,
            )
        prompt = (
            f"Compute the initial bearing (forward geodesic azimuth) from {a} to {b} on the WGS84 ellipsoid.\n\n"
            f"Give the azimuth in degrees clockwise from true north; any angle within "
            f"{c.bearing_tolerance_deg:g} degrees of PROJ's result, modulo 360, is accepted. {common}\n"
            f"Example: <answer>142.37</answer>"
        )
        return GeoData(
            idx=idx,
            prompt=prompt,
            family=family,
            src_crs=WGS84,
            dst_crs=WGS84,
            inputs=[lon, lat, lon2, lat2],
            expected=[fwd_az],
            tolerance=c.bearing_tolerance_deg,
        )
