# geo-reprojection-v1

Coordinate reprojection and geodesy on the WGS84 / GDA2020 ellipsoid, graded against
[PROJ](https://proj.org/) rather than a stored answer key.

| | |
|---|---|
| **Archetype** | Deterministic reward, procedurally generated, `INFINITE` |
| **Harness** | Any. Tool-less (`null`) makes it a numeric-reasoning task; `bash` makes it a tool-use task |
| **Runtime** | No container (`NEEDS_CONTAINER` unset); run on a host `subprocess` runtime |
| **Reward** | `within_tolerance` (weight 1.0): pass/fail against a physical tolerance; graded mode opt-in |

## Task families

| Family | Ask | Answer | Accepted within |
|---|---|---|---|
| `reproject` | WGS84 lon/lat → projected easting/northing | `easting, northing` in metres | 1 m, Euclidean in the destination grid |
| `inverse` | Projected easting/northing → WGS84 lon/lat | `longitude, latitude` in decimal degrees | 1 m, measured as geodesic distance on the ellipsoid |
| `distance` | Geodesic distance between A and B | metres | 10 m |
| `bearing` | Initial forward azimuth from A to B | degrees clockwise from true north | 0.05°, modulo 360 |

Nine projected CRSs, each sampled only inside its true area of use: GDA2020 MGA zones
54–56, WGS84 UTM zones 10N/33N/56S, British National Grid, NZTM2000, and Web Mercator.
Coordinates are printed to six decimal places (degrees) or three (metres).

Each prompt states the task, the units and order of the answer, the tolerance it is
graded against (interpolated from the configured value), and the parsing rule below.
Nothing in the prompt prescribes a method. British National Grid prompts additionally
name the OSGB36 datum shift the key uses (see *Datum transformations*).

## Answer format and parsing

The reply must contain an `<answer>...</answer>` block. The parser, as implemented:

- grades the **last** `<answer>` block in the reply and ignores text outside it;
- reads numbers separated by commas or whitespace; because commas are separators,
  `3,941,601.087` is three numbers and a reply using thousands separators is
  *unparseable* (`answered` 0, reward 0), not merely wrong;
- requires exactly the number of values the family asks for — two for `reproject` and
  `inverse`, one for `distance` and `bearing` — and treats any other count as unparseable;
- accepts plain decimals and scientific notation, with any number of decimal places;
- compares bearings circularly, so 359.99 and −0.01 are the same heading and any
  equivalent angle modulo 360 is accepted.

## Rewards and metrics

| Name | Kind (weight) | Definition |
|---|---|---|
| `within_tolerance` | reward (1.0) | 1.0 iff the error is at most the row's tolerance, else 0.0; 0.0 when unparseable. With `task.graded = true`: `1 / (1 + (error / (tolerance × graded_halflife))²)`, a smooth decay still anchored on the tolerance |
| `error` | metric | Physical error: metres for `reproject` (Euclidean in the destination grid), `inverse` (geodesic distance on the WGS84 ellipsoid) and `distance`; degrees for `bearing`. −1.0 when unparseable. A reply PROJ cannot evaluate — a latitude outside ±90°, which has no point on the ellipsoid, or an overflowing literal — reads as the finite sentinel `20003931.5` m (half the meridional circumference; `180.0`° for bearings), never NaN |
| `error_ratio` | metric | `error / tolerance`, so 1.0 is the pass boundary in every family and the value is comparable across families where `error` is not. −1.0 when unparseable |
| `answered` | metric | 1.0 when the reply parsed. Separates a wrong answer from no answer — they need different fixes |

`error` mixes metres and degrees when families are mixed, so its run-level mean is not
meaningful across families; read `error_ratio`, or `error` per family.

## Arguments

Set on the taskset with `--env.taskset.<flag>`, or under `[env.taskset]` in a config toml.
Task-level flags take `--env.taskset.task.<flag>` / `[env.taskset.task]`.

| Flag | Default | Meaning |
|---|---|---|
| `--env.taskset.families` | `["reproject", "inverse", "distance", "bearing"]` | Families to draw from; each generated row picks one uniformly |
| `--env.taskset.position-tolerance-m` | `1.0` | Pass threshold for `reproject` and `inverse` rows, in metres |
| `--env.taskset.distance-tolerance-m` | `10.0` | Pass threshold for `distance` rows, in metres |
| `--env.taskset.bearing-tolerance-deg` | `0.05` | Pass threshold for `bearing` rows, in degrees |
| `--env.taskset.seed` | `20260826` | Seeds the coordinate stream; the same seed regenerates the same tasks |
| `--env.taskset.task.graded` | `false` | Fold the error curve into the reward instead of a hard pass/fail |
| `--env.taskset.task.graded-halflife` | `1.0` | Error, in tolerance units, at which the graded reward decays to 0.5 |

The tolerances are quoted in the prompt text, so changing one also changes what the
model is told. The taskset is infinite; bound every run with `-n`.

## Dependencies

`verifiers>=0.3.1` and `pyproj>=3.6`, which bundles PROJ and its EPSG database. No API
keys or environment variables, no network access, no PROJ grid files, and no container
(`NEEDS_CONTAINER` unset).

## Datum transformations

"WGS 84" in the answer key means one named EPSG operation per grid, pinned in the
taskset (`PINNED_DATUM_SHIFT`) rather than left to PROJ's best-available choice, so the
key for a given seed does not depend on which grids the grading host has installed or
on `PROJ_NETWORK`:

| Grid | Datum shift to WGS 84 | EPSG-stated accuracy |
|---|---|---|
| GDA2020 MGA zones 54–56 | `GDA2020 to WGS 84 (2)` — null transformation; the two datums are treated as coincident | 3 m |
| NZTM2000 | `NZGD2000 to WGS 84 (1)` — null transformation | 1 m |
| British National Grid | `OSGB36 to WGS 84 (6)` — the EPSG 7-parameter Helmert, **not** the OSTN15 grid; the prompt says so on these rows | 2 m |
| WGS 84 UTM zones, Web Mercator | none — the projection is exact | — |

The reference answer is therefore "what PROJ returns under that named operation", and the
1 m position tolerance is a tolerance against that reference, not against ground truth:
the EPSG operations themselves are only accurate to 1–3 m, and a solver that applies
OSTN15 to a British National Grid row can land metres from the key.

## Design notes

**Why binary reward.** Partial credit on a numeric answer is a reward-hacking surface: a
model that echoes the input, or emits a number of roughly the right magnitude, would bank
a fraction for free. The reward is pass/fail; the continuous error is recorded as a
*metric* so a training run still has a dense signal to inspect. `--env.taskset.task.graded true`
folds the error curve into the reward when a run explicitly wants shaping.

**Why tolerance is physical.** Errors are compared in the destination grid's own
projected metres: ground metres to within the 0.9996 scale factor on the transverse
Mercator and UTM grids, and up to about 2.9× ground metres for Web Mercator at the ±70°
edge of its sampling box (one grid metre there is 1/cos 70° ground metres). Inverse rows
are compared as geodesic distance on the WGS84 ellipsoid, so a longitude error costs the
same near the pole as at the equator. A tolerance on the printed decimals would instead
be far stricter for a 7-digit easting than for a 3-digit longitude.

**Why procedural.** Coordinates are drawn to six decimal places inside each CRS's area of
use, so the stream never repeats and there is no answer key to leak. `INFINITE = True`;
bound every run with `-n`.

**Why the key is re-derived in `validate()`.** Every row's expected answer is recomputed
from the inputs the model reads — the same PROJ calls, applied to the shipped numbers —
and every coordinate is round-tripped forward and back through PROJ. A generator bug
surfaces as a key/input disagreement, and a coordinate drawn outside its CRS's usable
domain fails the round-trip, before a model ever sees the row.

## Run

```bash
# Model-free: re-derive every key from its shipped inputs and round-trip through PROJ
uv run validate geo-reprojection-v1 -n 40 --runtime.type subprocess

# The measured run, with every flag spelled out (the repository's
# configs/geo-reprojection.toml sets the model, 40 tasks, temperature 0 and the null
# harness, but not the subprocess runtime or the episode timeout)
uv run eval geo-reprojection-v1 -n 40 --model openai/gpt-5-mini \
  --env.agent.harness.id null --env.agent.runtime.type subprocess \
  --env.timeout.episode 480 --sampling.temperature 0

# One family at a tighter tolerance
uv run eval geo-reprojection-v1 -n 20 --model <model-id> \
  --env.agent.harness.id null --env.agent.runtime.type subprocess \
  --env.taskset.families '["reproject"]' --env.taskset.position-tolerance-m 0.5
```

The flags matter for a tool-less task: without `--env.agent.harness.id null` verifiers
resolves the agent to the `bash` harness, and without `--env.agent.runtime.type
subprocess` even the null harness provisions a Prime sandbox for every rollout although
nothing here needs a container. `--env.timeout.episode 480` bounds a rollout that gets
stuck in provider retries; the measured run below had no episode timeout and hung.

## Status

| Check | Result |
|---|---|
| `uv run validate geo-reprojection-v1 -n 40 --runtime.type subprocess` | 40/40 valid — every key re-derived from its shipped inputs, every coordinate round-tripped within 1 cm |
| Offline test suite (pytest, in the repository) | Passes. Asserts that PROJ's own answer scores 1.0 in every family, and that echoing the input, omitting the `<answer>` block, scattering candidate numbers in prose, using thousands separators, supplying the wrong number of values, or answering with a latitude off the ellipsoid all score 0.0 (in graded mode too); that `validate()` rejects a tampered key in every family; and that the datum shift is pinned |
| Eval — `openai/gpt-5-mini` via Prime Inference, 2026-09-11 | 40 tasks requested, all four families, temperature 0.0, `null` harness, `subprocess` runtime: 36 traces recorded (the run was cut after 4 rollouts hung in a provider-500 retry loop with no timeout), 34 `agent_completed`, 2 provider errors excluded. `within_tolerance` 0.0882 (3/34); `answered` 1.000; `error` median 92.2, min 0.0, max 4,538,768.7, deciles [0.1, 0.7, 9.9, 52.4, 92.2, 114.4, 957.9, 2686.4, 7265.4] — `error` mixes metres and degrees across families, so the deciles are the summary, not a mean |
| Eval — `openai/gpt-5-mini`, 2026-09-12, revised prompt | Same 40 tasks and settings plus `--env.timeout.episode 480`: 37/40 completed, 3 hit the episode timeout (recorded as errors, no trace). `within_tolerance` **0.135** (5/37); `answered` 0.946; `error` median 39.1 |
| Eval — `openai/gpt-5-nano`, 2026-09-12 | 40/40 completed. `within_tolerance` **0.100** (4/40); `answered` 0.950; `error` median 122.9 |
| Eval — `deepseek/deepseek-v4-flash`, 2026-09-12 | 37/40 completed, 3 episode timeouts. `within_tolerance` **0.081** (3/37); `answered` 1.000; `error` median 4,778.7 |
| Eval — `google/gemini-2.5-flash-lite`, 2026-09-12 | 40/40 completed. `within_tolerance` **0.000** (0/40); `answered` 0.975; `error` median 208,230.9 |
| Reading across models | Every model answers in the required format almost every time, and every model is far outside the tolerances almost every time — the ordering (gpt-5-mini > gpt-5-nano > deepseek-v4-flash > gemini-2.5-flash-lite) is visible in the `error` medians long before it shows in the binary reward, which is what the graded metric is for. The `error` medians mix metres and degrees across families |
| `bash` harness (tool use) | Not yet run against a model |
| `task.graded = true` | Not yet run against a model |

Reading of the measured run: gpt-5-mini produces a well-formed answer on every task but
lands far outside the 1 m / 10 m / 0.05° tolerances; the reward separates "wrong" from
"unparseable" exactly as designed.

Caveats:

- Prime Inference returned bursts of upstream 500s on gpt-5-mini during that session and
  the run had no episode timeout, which is why it was cut at 36 traces;
  `--env.timeout.episode 480` was adopted for the later runs in the same session and is
  part of the command above.
- The prompt wording was revised after that run: the graded tolerance, the "last block
  is graded" rule and "no thousands separators" are now stated, the bearing prompt reads
  "from A to B", and British National Grid prompts name the OSGB36 datum shift. The
  reward, the parser, the tolerances and the answer keys are unchanged, so the numbers
  above are a record of the earlier prompt.
- `error_ratio` was added after the 2026-09-11 run; the 2026-09-12 runs record it (see the uploaded evaluations), and it is not summarised above because its mean is dominated by a few catastrophic answers.
