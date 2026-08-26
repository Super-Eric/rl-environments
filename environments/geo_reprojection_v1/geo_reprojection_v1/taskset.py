"""geo-reprojection-v1 — coordinate reprojection and geodesy, verified against PROJ.

Four procedurally generated task families, all graded against `pyproj` (PROJ) rather
than a stored answer key:

- `reproject`  — transform a geographic coordinate into a projected CRS (metres).
- `inverse`    — transform projected easting/northing back to longitude/latitude.
- `distance`   — geodesic distance between two points on the ellipsoid.
- `bearing`    — initial (forward) geodesic azimuth between two points.

Why this shape:

- **Procedural and `INFINITE`.** Coordinates are drawn at random to six decimal places
  inside each CRS's true area of use, so the task stream never repeats and cannot be
  memorised from a leaked answer key. Runs bound it with `-n`.
- **Deterministic verification.** PROJ is the reference implementation the geospatial
  world actually uses; the reward is agreement with it, not with an LLM judge.
- **Binary reward, graded metric.** The reward is a hard pass/fail against a physical
  tolerance (metres, or degrees for bearing). Partial credit on a numeric answer is a
  reward-hacking surface: a model that emits the input unchanged, or a plausible-looking
  number of the right magnitude, should score zero rather than bank a fraction. The
  continuous error is still recorded as a *metric* so training runs keep a dense
  learning signal to inspect. Set `graded = true` to fold that curve into the reward
  when a run explicitly wants shaping.
- **Tolerance is physical, not textual.** Errors are compared in metres (or degrees of
  azimuth), so precision requirements mean the same thing across CRS families rather
  than varying with the magnitude of the coordinate.

Ground truth is checked without a model in `validate()`: every generated task is
round-tripped through PROJ and rejected if it does not return to its own input, which
catches a coordinate drawn outside its CRS's valid domain before a model ever sees it.
"""

import math
import random
import re
from collections.abc import Iterator
from functools import lru_cache
from typing import Literal

import verifiers.v1 as vf
from pyproj import CRS, Geod, Transformer

WGS84 = "EPSG:4326"
GEOD = Geod(ellps="WGS84")

Family = Literal["reproject", "inverse", "distance", "bearing"]

# Projected CRSs paired with the lon/lat box they are actually valid inside. Sampling
# from the area of use (rather than the whole globe) is what keeps generated tasks
# physically meaningful — PROJ will happily return nonsense far outside a zone.
PROJECTED_CRS: dict[str, tuple[str, tuple[float, float, float, float]]] = {
    # GDA2020 MGA zones — the Australian national standard (relevant, and rarely memorised).
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

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@lru_cache(maxsize=64)
def _transformer(src: str, dst: str) -> Transformer:
    # always_xy keeps every axis order as (lon, lat) / (easting, northing), which is the
    # order the prompts state. EPSG axis order for several of these CRSs is otherwise
    # (lat, lon) and would silently transpose the ground truth.
    return Transformer.from_crs(CRS.from_user_input(src), CRS.from_user_input(dst), always_xy=True)


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


class GeoData(vf.TaskData):
    family: Family
    """Which of the four task families this row is."""
    src_crs: str
    """Authority code the inputs are expressed in."""
    dst_crs: str
    """Authority code the answer must be expressed in (equal to src_crs for geodesy rows)."""
    inputs: list[float]
    """The coordinate(s) given in the prompt, in the order the prompt states them."""
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
            # Azimuths are circular: 359.9 deg and -0.1 deg are the same heading.
            return abs((got[0] - exp[0] + 180.0) % 360.0 - 180.0)
        if self.data.family == "distance":
            return abs(got[0] - exp[0])
        if self.data.family == "inverse":
            # Compare in metres on the ellipsoid, not in degrees: a degree of longitude
            # is worth very different distances at different latitudes, so a degree-based
            # tolerance would be far stricter near the poles than at the equator.
            return float(GEOD.inv(got[0], got[1], exp[0], exp[1])[2])
        return ((got[0] - exp[0]) ** 2 + (got[1] - exp[1]) ** 2) ** 0.5

    async def validate(self, runtime: vf.Runtime) -> bool:
        """Model-free ground-truth check: PROJ must round-trip this row back to its input.

        Runs via `uv run validate`, before any model is involved. A coordinate that does
        not survive a forward-then-inverse transform is outside its CRS's usable domain,
        and its ground truth would be meaningless. Assertions raise rather than return
        False so a failure names the offending row and its drift in the report.
        """
        d = self.data
        if d.family in ("distance", "bearing"):
            assert math.isfinite(d.expected[0]), f"non-finite ground truth for task {d.idx}"
            return True
        fwd, inv = (d.src_crs, d.dst_crs) if d.family == "reproject" else (d.dst_crs, d.src_crs)
        x, y = d.inputs if d.family == "reproject" else d.expected
        px, py = _transformer(fwd, inv).transform(x, y)
        bx, by = _transformer(inv, fwd).transform(px, py)
        assert math.isfinite(px) and math.isfinite(py), f"CRS {fwd}->{inv} is undefined at task {d.idx}"
        back = float(GEOD.inv(bx, by, x, y)[2])
        assert back < 0.01, f"round-trip drift {back:.4f} m exceeds 1 cm for task {d.idx}"
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
        assert c.families, "need at least one family in `families`"
        rng = random.Random(c.seed)
        codes = list(PROJECTED_CRS)
        for idx in range(10**9):
            family: Family = rng.choice(c.families)
            code = rng.choice(codes)
            label, (w, s, e, n) = PROJECTED_CRS[code]
            lon, lat = round(rng.uniform(w, e), 6), round(rng.uniform(s, n), 6)
            yield GeoTask(self._build(idx, family, code, label, lon, lat, rng), c.task)

    def _build(self, idx, family, code, label, lon, lat, rng) -> GeoData:
        c = self.config
        common = "Reply with only the numeric answer inside <answer></answer> tags."
        if family == "reproject":
            x, y = _transformer(WGS84, code).transform(lon, lat)
            prompt = (
                f"Transform the WGS84 geographic coordinate longitude {lon}, latitude {lat} "
                f"into {label} ({code}).\n\n"
                f"Give easting and northing in metres, comma-separated, easting first. {common}\n"
                f"Example: <answer>500000.00, 6100000.00</answer>"
            )
            return GeoData(idx=idx, prompt=prompt, family=family, src_crs=WGS84, dst_crs=code,
                           inputs=[lon, lat], expected=[x, y], tolerance=c.position_tolerance_m)
        if family == "inverse":
            x, y = _transformer(WGS84, code).transform(lon, lat)
            prompt = (
                f"Transform the {label} ({code}) coordinate easting {x:.3f} m, northing {y:.3f} m "
                f"back into WGS84 geographic coordinates ({WGS84}).\n\n"
                f"Give longitude and latitude in decimal degrees, comma-separated, longitude first, "
                f"to at least six decimal places. {common}\n"
                f"Example: <answer>151.209900, -33.865143</answer>"
            )
            return GeoData(idx=idx, prompt=prompt, family=family, src_crs=code, dst_crs=WGS84,
                           inputs=[x, y], expected=[lon, lat], tolerance=c.position_tolerance_m)
        # Geodesy families: a second point offset by a random azimuth and distance.
        az, dist = rng.uniform(-180.0, 180.0), rng.uniform(1_000.0, 400_000.0)
        lon2, lat2, _ = GEOD.fwd(lon, lat, az, dist)
        lon2, lat2 = round(lon2, 6), round(lat2, 6)
        fwd_az, _, true_dist = GEOD.inv(lon, lat, lon2, lat2)
        pts = f"A (longitude {lon}, latitude {lat}) and B (longitude {lon2}, latitude {lat2})"
        if family == "distance":
            prompt = (
                f"Compute the geodesic distance on the WGS84 ellipsoid between {pts}.\n\n"
                f"Give the distance in metres. {common}\n<answer>123456.78</answer>"
            )
            return GeoData(idx=idx, prompt=prompt, family=family, src_crs=WGS84, dst_crs=WGS84,
                           inputs=[lon, lat, lon2, lat2], expected=[true_dist],
                           tolerance=c.distance_tolerance_m)
        prompt = (
            f"Compute the initial bearing (forward geodesic azimuth) from {pts} on the WGS84 ellipsoid.\n\n"
            f"Give the azimuth in degrees clockwise from true north, in the range [-180, 180]. "
            f"{common}\n<answer>142.37</answer>"
        )
        return GeoData(idx=idx, prompt=prompt, family=family, src_crs=WGS84, dst_crs=WGS84,
                       inputs=[lon, lat, lon2, lat2], expected=[fwd_az],
                       tolerance=c.bearing_tolerance_deg)
