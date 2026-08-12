# Video analysis pipeline — debug findings

Evidence: all 29 `replay_payload.json` files under `outputs/video_analysis/`, produced by real runs
between 2026-08-03 and 2026-08-05. No speculation below is unmarked.

---

## Headline

The pipeline works. 11 of 29 runs reached `CALIBRATED` and produced speeds. The gap to
FullTrack-quality output is **five data defects and four missing render features**, not a missing
algorithm.

One run is already good: `132543_4be4a8` — 1.21 px reprojection, 101.14 km/h, 3.81 m length,
bounce at y=16.31 on a 20.12 m pitch. That is a coherent good-length delivery. The pipeline can
produce correct results today; it does not do so reliably.

---

## Defect 1 — world coordinates are published in two different orderings (critical)

Four of eleven calibrated runs have **x and y swapped**.

| analysis | y range (should be 0–20.12) | x range (should be ±1.5) | geometry_validity |
|---|---|---|---|
| `132522_b2d939` | -1.00 .. 2.67 | 2.50 .. 11.67 | `None` |
| `140407_73bdd3` | -1.00 .. 2.67 | 2.50 .. 11.67 | `None` |
| `143147_6119c8` | 0.17 .. 3.84 | -6.67 .. 2.50 | `None` |
| `232612_d45813` | -1.00 .. 2.00 | 2.50 .. 10.01 | `None` |
| `132543_4be4a8` | 10.96 .. 16.89 | -0.10 .. 0.56 | `VALID_METRIC_3D` |
| `000907_55c42b` | 2.45 .. 17.22 | -0.81 .. -0.55 | `VALID_METRIC_3D` |

The down-pitch extent is sitting in the lateral slot. This is exactly the hazard called out in
`cricket_pitch_geometry.py`:

> Legacy Calibration V2 helpers below use longitudinal-X / lateral-Y ordering; adapt through
> `calibration_to_canonical_world()` before publishing replay data.

**Every swapped run has `geometry_validity: None` and `in_pitch_fraction: None`. Every correct run
has `VALID_METRIC_3D` and `in_pitch_fraction: 1.0`.** So there is a code path that publishes
trajectory world positions without passing them through `calibration_to_canonical_world()`, and
that same path skips geometry validation — which is why nothing caught it.

It is not chronological: `143147` (swapped) → `143942` (correct) → `232612` (swapped). Two live
paths, one adapted and one not.

Consequence: `delivery_length_m` is `None` on all four, and their speeds cluster at 38–43 km/h
versus 90–138 km/h on the correct runs. A 3D replay fed these would draw the ball travelling
sideways across the pitch.

**Fix:** find the publisher that bypasses the adapter; make `calibration_to_canonical_world()` the
only way trajectory points reach `ReplayPayloadV1`. Then make geometry validation mandatory rather
than optional — a payload with `geometry_validity: None` should not be `CALIBRATED`.

---

## Defect 2 — the ball is only tracked over part of its flight

Down-pitch coverage on correct runs:

| analysis | tracked y span | fraction of 20.12 m |
|---|---|---|
| `143942_dbf1f2` | 1.52 – 16.53 | 75% |
| `000510_7b62b4` | 2.05 – 18.70 | 83% |
| `000907_55c42b` | 2.45 – 17.22 | 73% |
| `125835_702019` | 4.81 – 16.47 | 58% |
| `011106_b5d8d4` | 7.15 – 19.41 | 61% |
| `132543_4be4a8` | 10.96 – 16.89 | **29%** |

`release_speed_kmh` on a track that begins 11 m down the pitch is not release speed. This is the
main reason speeds range 90–138 km/h across similar footage.

**Fix:** treat release speed as unavailable unless the track starts within a defined distance of the
bowling end. The schema already supports this — return `UNAVAILABLE` with a reason rather than a
number derived from mid-flight. Report `average_pre_bounce_speed_kmh` instead when coverage is poor.

---

## Defect 3 — a quarter of every trajectory is invented

Across all runs: **757 `OBSERVED`, 73 `RECOVERED`, 270 `PHYSICS_FITTED`** samples.

Worst cases:

| analysis | observed | fitted |
|---|---|---|
| `121739_bc7cf0` | 8 | 14 |
| `122114_ed8a6f` | 7 | 11 |
| `124238_c3074e` | 11 | 12 |

More than half the "trajectory" is model output in those runs. The provenance field records this
honestly, but nothing downstream acts on it and the UI does not show it.

**Fix:** gate metrics on observed-sample count, not total sample count. Render fitted segments
differently in the 3D replay — dashed or faded — so a viewer can see which part of the arc was
measured.

---

## Defect 4 — browser-recorded clips have variable frame rate

Older runs are a clean 60 or 59.94 fps. Every recent run is not:

```
45.07  32.65  33.21  39.07  49.54  34.12  fps
```

These are the `MediaRecorder` clips. Variable frame rate means `timestamp_seconds` spacing is
irregular, and every speed is a distance ÷ time calculation.

**Fix:** transcode to constant frame rate at ingest. `_ensure_readable_video()` already calls ffmpeg
for browser clips — add `-r` and `-vsync cfr` to that call so the normalised file has fixed spacing.
Cheap, and it lands where the work already happens.

---

## Defect 5 — calibration quality is not gated or surfaced

| analysis | reprojection mean | speed | length | plausible? |
|---|---|---|---|---|
| `132543_4be4a8` | **1.21 px** | 101.14 | 3.81 m | yes |
| `000907_55c42b` | 8.52 px | 102.64 | 3.37 m | yes |
| `143942_dbf1f2` | 8.63 px | 103.68 | 4.30 m | yes |
| `125835_702019` | **22.13 px** | 90.29 | 6.08 m | doubtful |
| `011106_b5d8d4` | **22.66 px** | 138.06 | 1.58 m | no |

Reprojection error predicts plausibility almost perfectly. The 22 px runs produce the outliers.
Nothing in the UI distinguishes them from the 1.21 px run.

Separately: 6 of the 16 `IMAGE_SPACE_ONLY` runs had `camera.source: CALIBRATED` but carried
*"Tracking completed without accepted calibration."* The pose was computed and usable; nobody
pressed Accept, so it was discarded.

Three runs also had a mean detection confidence of **0.08** and still produced numbers.

**Fix:** a quality band on the review screen (green <5 px, amber <15 px, red beyond); block tracking
until calibration is accepted; refuse to publish metrics below a confidence floor.

---

## What is missing for the FullTrack 3D output

The data side is nearly there — `world_position` trajectories, `bounce.world_position`, and a full
camera pose all exist on good runs. What is absent is rendering.

| Missing | Where it goes | Notes |
|---|---|---|
| Trajectory line + bounce marker in 3D | `replayContent` slot on `VirtualPitchScene` | Slot already exists and is empty. drei `<Line>` over `trajectory[].world_position`. |
| Outfield | New mesh under the pitch | Beyond the pitch polygon is currently flat background colour. |
| Sky | `<Sky/>` from drei | Already a dependency. |
| Crowd + sponsor boards | Cylinder with texture at the horizon | This is most of what makes their replay read as a stadium. |
| Camera fly-through | `VirtualPitchCameraController` | Controller exists; needs keyframed presets. |
| Metric chips over the canvas | DOM, not 3D | Straightforward. |
| Spin | Not in `ReplayMetrics` | Would need adding to the schema and computing. |
| Shot type / footwork | `Models/CricShot10k/` | Models present, unwired. |

---

## Order of work

1. **Defect 1** — coordinate ordering. Everything else is built on these numbers.
2. **Defect 4** — constant frame rate at ingest. One ffmpeg flag, fixes all future clips.
3. **Defect 5** — accept gate and quality band. Recovers 6 of 16 failed runs and stops bad ones being trusted.
4. **Defect 2 and 3** — honest metric gating on coverage and observed-sample count.
5. **Rendering** — trajectory into `replayContent`, then sky and outfield, then crowd and camera moves.

Steps 1–4 are correctness and are worth more than any visual work. Step 5 is what makes it look
like the reference, and it is mostly scenery over data that already exists.
