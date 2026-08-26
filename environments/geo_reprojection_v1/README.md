# geo-reprojection-v1

Coordinate reprojection and geodesy on the WGS84 / GDA2020 ellipsoid, graded against
[PROJ](https://proj.org/) rather than a stored answer key.

| | |
|---|---|
| **Archetype** | Deterministic reward, procedurally generated, `INFINITE` |
| **Harness** | Any. Tool-less (`null`) makes it a numeric-reasoning task; `bash` makes it a tool-use task |
| **Runtime** | None required — no container |
| **Reward** | Binary pass/fail against a physical tolerance; graded mode opt-in |

## Task families

| Family | Ask | Tolerance |
|---|---|---|
| `reproject` | WGS84 lon/lat → projected easting/northing | 1 m |
| `inverse` | Projected easting/northing → WGS84 lon/lat | 1 m (compared on the ellipsoid, not in degrees) |
| `distance` | Geodesic distance between two points | 10 m |
| `bearing` | Initial forward azimuth between two points | 0.05° (circular) |

Nine projected CRSs, each sampled only inside its true area of use: GDA2020 MGA zones
54–56, WGS84 UTM zones 10N/33N/56S, British National Grid, NZTM2000, and Web Mercator.

## Design notes

**Why binary reward.** Partial credit on a numeric answer is a reward-hacking surface: a
model that echoes the input, or emits a number of roughly the right magnitude, would bank
a fraction for free. The reward is pass/fail; the continuous error is recorded as a
*metric* so a training run still has a dense signal to inspect. `--env.taskset.task.graded true`
folds the error curve into the reward when a run explicitly wants shaping.

**Why tolerance is physical.** Errors are compared in metres, so "1 metre" means the same
thing in every CRS. A tolerance on the printed decimals would be far stricter for a
7-digit easting than for a 3-digit longitude.

**Why procedural.** Coordinates are drawn to six decimal places inside each CRS's area of
use, so the stream never repeats and there is no answer key to leak. `INFINITE = True`;
bound every run with `-n`.

## Run

```bash
uv run validate geo-reprojection-v1 -n 40 --runtime.type subprocess   # no model needed
uv run eval geo-reprojection-v1 -n 20 --model <model-id>
uv run eval geo-reprojection-v1 -n 20 --env.taskset.families '["reproject"]' \
  --env.taskset.position-tolerance-m 0.5
```

## Verified

`uv run validate` round-trips every generated coordinate through PROJ and rejects any row
that does not return to its own input within 1 cm — this catches a coordinate drawn
outside its CRS's valid domain before a model ever sees it. 40/40 valid.

`tests/test_geo_reprojection.py` asserts the gold answer scores 1.0 in every family, and
that echoing the input, omitting the `<answer>` block, scattering candidate numbers in
prose, or supplying the wrong number of values all score 0.0.
