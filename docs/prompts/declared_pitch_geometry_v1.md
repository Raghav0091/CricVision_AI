# Claude Code Prompt — Let the operator declare the pitch geometry

> Paste at the repo root. Ponytail rules apply: reuse before adding, deletion over addition,
> fewest files, no new dependencies. This touches shared geometry used in 28 places — grep every
> caller before changing the signature.

---

## The problem, with evidence

`/live` calibration succeeds (the operator places six points per wicket by hand), then recording a
delivery fails with:

```
Registration failed for all pose candidates.
```

Cause: `build_virtual_pitch_specification()` in `services/api/services/virtual_pitch_service.py`
takes no parameters and hardcodes regulation constants:

```python
for end, y in (("bowler", 0.0), ("striker", PITCH_LENGTH_M)):   # 20.12 m, always
```

So the PnP solver asks "where must the camera be, for a **20.12 m** pitch to look like this?" When
the operator is testing indoors with two improvised wickets three metres apart, no pose satisfies
that, and every candidate fails the plausibility gates in `real_pitch_registration_service`
(`camera_distance`, `near_far_projected_size_order`, `projected_wickets_reasonable`).

The refusal is correct. What is missing is the ability to tell it the truth about the pitch.

**The constants are already parameterised.** `CricketPitchDimensions` in
`packages/cricket_vision/calibration/cricket_pitch_geometry.py` is a frozen dataclass with a
`validate()` method, and `CricketPitchGeometry` in `schemas/video_analysis.py` already accepts
`pitch_length_m: float = Field(default=PITCH_LENGTH_M, gt=0, le=40)`. Only the specification
builder ignores them.

**Why this is worth doing:** it unblocks all indoor testing. PnP conditioning actually *improves*
at short range — regulation is a 20.12 / 0.2286 ≈ 88 baseline-to-feature ratio, while a 4 m pitch
with full-size wickets is ≈ 17, so the far wicket subtends a larger angle and is easier to solve.

---

## 1. Parameterise the specification builder

```python
def build_virtual_pitch_specification(
    dimensions: CricketPitchDimensions | None = None,
) -> VirtualPitchSpecification:
```

- `None` means regulation — **every existing caller must keep working untouched.**
- Call `dimensions.validate()` on entry and raise a 422-mapped error on failure.
- Replace the module-level constants inside the body with fields from `dimensions`:
  `pitch_length_m`, `pitch_width_m`, `wicket_width_m`, `wicket_height_m` (→ `stump_height_m`),
  `stump_diameter_m`, `popping_crease_distance_m`, `bowling_crease_length_m`,
  `return_crease_offset_m`.
- `VirtualPitchSpecification.dimensions` must report what was actually used, not the defaults.

**Grep first.** 28 call sites across 9 files:

```
camera_preset_parameterization.py   pitch_space_metrics_service.py
preset_auto_registration.py         pitch_space_track_service.py
real_pitch_registration_service.py  virtual_pitch_service.py
two_wicket_pitch_fit_service.py     routes/video_analysis.py
```

Do not change their behaviour in this task. They keep calling it with no argument and keep getting
regulation geometry.

**Caching:** if the builder memoises, key the cache on the dimensions, not on nothing. A cache that
ignores the parameter will silently return regulation geometry for a declared 3.2 m pitch — a bug
that would look exactly like this task never happened.

## 2. Thread geometry into registration

`real_pitch_registration_service.py` calls the builder at lines ~230, ~408, ~523, ~853, ~1604.
Accept a `CricketPitchDimensions | None` down that path and pass it to every one of those calls.
Missing a single site produces a solver mixing a 3.2 m model with a 20.12 m model — which will
still *return* a pose, just a meaningless one. That is the main risk in this task.

Same for `wicket_box_calibration_service.py`, which drives registration.

## 3. Carry it on the request

`schemas/wicket_box_calibration.py`, `WicketBoxCalibrationRegisterRequest` — add:

```python
pitch_geometry: CricketPitchGeometry | None = None
```

Reuse the existing `CricketPitchGeometry` from `schemas/video_analysis.py`; do not define a second
geometry model. `None` means regulation.

Persist the declared geometry into the accepted snapshot
(`accepted_wicket_box_calibration_v1.json`) so tracking, physics and replay all use the same pitch
the pose was solved against. A pose solved on 3.2 m and physics run on 20.12 m would report speeds
roughly 6× too high.

`load_active_accepted_wicket_box_calibration()` must return it, and
`delivery_physics_service` / `replay_payload_service` must consume it rather than defaulting.

## 4. Frontend

**Types** — mirror `CricketPitchGeometry` in `apps/web/lib/wicketCalibration/types.ts`.

**Live setup** — a "Pitch setup" control on `/live` before calibration:

- Default **22 yards (20.12 m)**, presented as the normal case.
- A "Custom pitch" toggle revealing a length input in metres, `0.5`–`40`.
- Alongside it, wicket width and stump height inputs defaulting to regulation, since improvised
  indoor wickets are rarely full size. Getting these wrong scales every distance.
- Persist to `localStorage` so a test rig survives a reload.
- Show the active geometry on the capture screen and on `DeliveryReview` — a 3.2 m declared pitch
  reporting 14 km/h must be obviously a rig reading, never mistaken for a net reading.

**Pipeline** — `apps/web/lib/deliveryPipeline.ts`, `runDeliveryAnalysis()` takes the geometry and
puts it on the register request. It already accepts `placedKeypoints`; add geometry beside it.

**`/video-analysis`** — same control on the calibration stage, same default.

## 5. Tests

Backend (`tests/`, pytest, mirror existing style):

1. `test_specification_defaults_to_regulation` — no argument gives `pitch_length_m == 20.12` and
   landmark positions byte-identical to today. This is the regression guard for all 28 callers.
2. `test_specification_honours_declared_length` — 4.0 m puts the striker wicket at y = 4.0.
3. `test_invalid_dimensions_rejected` — `pitch_length_m=0` and a popping crease beyond half the
   pitch length both raise.
4. `test_registration_solves_short_pitch` — synthesise a camera viewing a 4 m pitch, project the 12
   stump points, register with declared 4 m geometry, assert a candidate is selected and the
   recovered camera position is within 5 cm. **Then assert the same points with regulation geometry
   fail** — that is the exact bug this task fixes.
5. `test_declared_geometry_persists_to_accepted_snapshot` — declared 4 m survives accept and reload.
6. `test_physics_uses_declared_geometry` — a track over a declared 4 m pitch does not report speeds
   computed against 20.12 m.
7. `test_cache_keyed_on_dimensions` — if a cache exists, requesting 4 m then 20.12 m returns
   different specifications.

Frontend:

8. Custom geometry round-trips through `localStorage`.
9. `DeliveryReview` renders the declared pitch length whenever it differs from regulation.

## 6. Order

1. Parameterise the builder + test 1 (the regression guard). Run the full suite before going on.
2. Thread through registration + test 4.
3. Request field and accepted-snapshot persistence + test 5.
4. Physics and replay consumption + test 6.
5. Frontend controls and pipeline wiring.

Step 1 first, and confirm the existing suite is unchanged before touching anything else. If
regulation behaviour shifts by even a millimetre, every downstream comparison in this repo moves
with it.

**Known pre-existing failures** — 5 tests plus a collection error in
`test_video_ball_pipeline_integration.py` (`VideoBallTrackingStartRequest` does not exist) already
fail on a clean baseline. Confirm against `git stash` before attributing any of them to this work.
