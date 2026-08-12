# Plan — Getting FullTrack-quality results without a pitch

Status: proposed, not started. Written against the code as of this session.

---

## 1. What is actually wrong today

Three separate causes, often mistaken for one "it's not accurate" problem.

### 1.1 No metric baseline

`ReplayMetrics` speeds and `delivery_length_m` are derived from world positions, which come from
a PnP solve against two wickets a known distance apart. With no stumps there is no solve, so the
payload correctly reports `INSUFFICIENT_EVIDENCE` or `IMAGE_SPACE_ONLY` and every metric stays
`null`. The schema is behaving properly. Nothing downstream can fix this.

### 1.2 Frame rate, not megapixels

`CameraPreview` requests:

```ts
video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } }
```

No `frameRate` constraint, so the browser picks — typically 30fps.

A delivery at 65 km/h is 18.05 m/s. Per-frame travel:

| fps | distance between samples |
|-----|--------------------------|
| 30  | 0.60 m |
| 60  | 0.30 m |
| 120 | 0.15 m |
| 240 | 0.075 m |

Over a 20.12 m pitch, 30fps yields roughly 33 samples of the whole flight and far fewer pre-bounce.
Bounce-point estimation and release speed both degrade sharply at that sampling rate. Doubling the
frame rate buys more accuracy than any resolution change.

Resolution matters second-order: the ball is ~72 mm across, so at 1080p from 4 m behind the stumps
it spans a healthy number of pixels. It is not the limiting factor.

### 1.3 Calibration frames are downscaled

`CameraPreview.captureFrame()` caps at 1280 px wide, JPEG quality 0.62, with a `# ponytail` note
that this keeps Cloudflare quick-tunnel uploads small. That is a sensible trade for the live
alignment overlay, but stump landmark precision drives PnP quality, and PnP quality drives every
metric. Calibration frames should not be the compressed ones.

---

## 2. The home rig

The system never requires a regulation pitch. It requires dimensions that are **declared truthfully**.
`CricketPitchGeometry` accepts any `pitch_length_m` in `(0, 40]`, any `wicket_width_m` in `(0, 1]`,
and any `wicket_height_m` in `(0, 2]`, and `CricketPitchDimensions.validate()` enforces internal
consistency rather than regulation values.

### 2.1 Build

Two wicket substitutes and a tape measure. In order of preference:

1. **Full-size paper stumps.** Print three 711 mm × 38 mm strips per wicket, mount on cardboard,
   outer stump centres 95.25 mm either side of middle. Two of these. Declare regulation
   `wicket_width_m` and `wicket_height_m`, and the measured distance as `pitch_length_m`.
2. **Dowels or bamboo skewers.** Any three vertical rods of equal, measured height at measured
   spacing. Declare what you measured, not what a real stump is.
3. **Half-scale.** If space is tight, halve the wicket dimensions too — but keep the ratio of
   wicket width to pitch length as close to regulation as the room allows. See §2.3.

### 2.2 Placement

- Measure wicket-to-wicket distance to the nearest centimetre. This number is the single largest
  determinant of metric accuracy — a 5% error here is a 5% error in every speed reading.
- Camera setback should scale with pitch length. Regulation is ~4 m behind for 20.12 m, so keep
  roughly `pitch_length / 5`.
- Camera height above the ground plane matters for the pose solve. Higher is better, as the
  reference app's own guidance says.

### 2.3 What the scale costs you

PnP conditioning depends on the ratio between the baseline (pitch length) and the feature size
(wicket width). Regulation is 20.12 / 0.2286 ≈ 88. A 4 m indoor pitch with regulation-width wickets
gives ≈ 17, which is *better* conditioned, not worse — the far wicket subtends a larger angle.

So a short indoor pitch with full-size paper wickets is a genuinely favourable test case. What it
cannot reproduce is the perspective compression of a real 20 m view, so it validates the maths, not
the field performance.

**Expect from the rig:** correct pose, correct trajectory shape, correct bounce detection, and
speeds that are real for that rig. **Do not expect** them to predict how the system behaves at a
real net.

---

## 3. Phases

### Phase 1 — Capture quality (no rig needed)

1. Add `frameRate: { ideal: 60, min: 30 }` to the `getUserMedia` constraints in `CameraPreview`.
   Log `track.getSettings()` so the actual negotiated fps is visible rather than assumed.
2. Surface the negotiated fps in the capture UI. If a device gives 30fps, the operator should know
   before bowling, not after.
3. Raise the calibration-frame cap from 1280 to the native track width, keeping JPEG quality but
   applying it only to the *calibration* frame, not to the recorded clip. Measure the upload size
   over the tunnel before committing — the existing 1280 cap exists for a reason.
4. Record the negotiated `width`, `height` and `frameRate` into the analysis record so a bad result
   can later be attributed to capture rather than to the pipeline.

Acceptance: a recorded clip reports its true fps, and `prepare` stores it.

### Phase 2 — Declare geometry through the pipeline

Today the live flow relies on defaults. The rig needs the declared numbers to reach PnP.

1. Add pitch length, wicket width and wicket height to the live session setup (FullTrack has a
   pitch-length slider on its create-session screen; this is the same control).
2. Thread them into `runDeliveryAnalysis` and on into the wicket-box calibration request, so
   `CricketPitchGeometry` carries the rig's real dimensions.
3. Default to regulation. A user at a real net changes nothing.
4. Show the active geometry on the review screen. A 3.5 m declared pitch producing 12 km/h should
   be obviously a rig reading, not mistaken for a net reading.

Acceptance: declaring 3.50 m and measuring 3.50 m produces `measurement_validity: CALIBRATED`.

### Phase 3 — Validate against ground truth

The rig's value is that truth is knowable.

1. Roll the ball along a measured distance and time it — over a 3 m rig at walking pace, a phone
   stopwatch is accurate to a few percent. Compare against `release_speed_kmh`.
2. Drop the ball from a measured height onto a marked point. Compare `bounce.world_position`
   against the mark, in centimetres.
3. Place the ball at rest at a known point and confirm the tracked world position matches.
   This isolates calibration error from tracking error — if a stationary ball reports the wrong
   position, the pose is wrong and nothing else is worth debugging.
4. Record each result in `docs/qa/`. These become the regression baseline.

Acceptance: static ball position within 5 cm; rolled-ball speed within 10% of stopwatch.

### Phase 4 — Real footage through video-analysis

Independent of the rig, and the only thing that predicts field behaviour.

1. Capture one real net session on a phone: tripod, both wickets in frame, 6 balls, 60fps.
   One visit produces a permanent regression fixture.
2. Run it through `/video-analysis` end to end and store the resulting `ReplayPayloadV1` as a
   golden file.
3. Any pipeline change re-runs it and diffs the metrics. This is what stops accuracy regressing
   silently.

Acceptance: the same clip produces the same numbers across runs.

---

## 4. Sequencing and honesty

Phase 1 is worth doing regardless — it is small, it needs nothing, and low frame rate will limit
every later result. Phase 2 is what makes home testing meaningful. Phase 3 is what turns "looks
about right" into a number. Phase 4 is the only phase that tells you how it behaves at a net.

Two things this plan deliberately does not promise:

- **The rig will not make home results match FullTrack's net results.** It validates that the maths
  is right. Field accuracy needs field data.
- **No amount of tuning substitutes for the baseline.** Until two wickets of declared, measured
  separation are visible in frame, `null` metrics are the correct output and should be left alone.
