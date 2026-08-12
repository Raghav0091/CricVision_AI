# Claude Code Prompt — Finish per-device lens calibration

> Paste at the repo root. Ponytail rules apply: reuse before adding, deletion over addition,
> fewest files, no new dependencies. Two files already exist and are verified working — do not
> rewrite them.

---

## Why this exists

`wicket_box_calibration_service` currently solves camera pose with an **unknown focal length**. It
calls `build_intrinsics_candidates()` (in `real_pitch_registration_service.py`), which sweeps
`FOCAL_FOV_HYPOTHESES = (45.0, 60.0, 75.0)` between bounds of `image_width * 0.35` and
`image_width * 3.5`, assumes zero distortion, and marks the result `confidence="LOW"`.

Because focal length and camera distance trade against each other, that sweep can settle on a
low-error fit describing a lens the phone does not have. Measured on real runs in
`outputs/video_analysis/`:

```
132543_4be4a8   fx =  997 px   diagonal FOV 72.7°   plausible   -> 101 km/h, 3.81 m, coherent
121528_4a8a03   fx = 1852 px   diagonal FOV 43.3°   impossible  -> INVALID_REPROJECTION
```

Solving the lens once per phone removes the dominant unknown from every later pose solve.

---

## Already built — do not recreate

### `services/api/schemas/device_calibration.py`

- `CheckerboardSpec(columns=9, rows=6, square_size_mm)` — **inner corners**, rejects square grids
- `DeviceLensProfile` — intrinsics, distortion, `image_width`/`image_height`, `checkerboard`,
  `quality`; methods `camera_matrix()` and `scaled_to(width, height)`
- `CalibrationQuality` — `rms_reprojection_px`, `band`, `views_used`, `diagonal_fov_degrees`,
  `fov_plausible`, `advice`
- `DeviceCalibrationRequest`, `DeviceCalibrationResponse`, `quality_band()`
- Constants: `MIN_VIEWS = 8`, `RECOMMENDED_VIEWS = 20`, `GOOD_RMS_PX = 0.5`,
  `MIN/MAX_PLAUSIBLE_DIAGONAL_FOV_DEG = 55/95`

### `services/api/services/device_calibration_service.py`

- `calibrate_device_from_video(video_path, device_id, device_label, spec) -> DeviceLensProfile`
- `save_device_profile(profile) -> Path`
- `load_device_profile(device_id) -> DeviceLensProfile | None`
- `load_device_profile_for_frame(device_id, width, height) -> DeviceLensProfile | None`
  — rescales to the frame in hand, returns `None` when the aspect ratio differs
- `DeviceCalibrationError(message, status_code)`
- Storage: `outputs/device_calibration/{device_id}.json`

**Verified against a synthetic camera**: true `fx` 1500 recovered as 1497.15 (0.19% error),
RMS 0.41 px, 18 views accepted from 26 sampled. Rescaling 1080×1920 → 4K is exact; a
cross-aspect rescale raises.

---

## 1. Routes — `services/api/routes/device_calibration.py`

```
POST /device-calibration/solve      multipart: video + device_id + device_label? + columns + rows + square_size_mm
                                    -> DeviceCalibrationResponse   (solves, saves, returns profile)
GET  /device-calibration/{device_id}    -> DeviceLensProfile | 404
DELETE /device-calibration/{device_id}  -> 204   (lets a user redo a bad calibration)
```

- Mirror the error handling in `routes/video_analysis.py`: catch `DeviceCalibrationError`, re-raise
  as `HTTPException(status_code=exc.status_code, detail=exc.message)`.
- Save the upload to a temp path, run the solve, delete the video afterwards. **Do not retain
  calibration footage** — it has no value once the profile exists and it is the user's camera roll.
- Register in `services/api/main.py` alongside the existing routers.

## 2. Device identity

Browsers give no stable camera identifier, and `MediaDeviceInfo.deviceId` rotates between origins
and sessions. Do not use it.

Instead: generate a UUID once in the browser, persist to `localStorage` under
`cricvision.deviceId`, and let the user set a friendly `device_label` ("Raghav's Pixel"). Send both.

Add to `apps/web/lib/deviceIdentity.ts`:

```ts
export function getDeviceId(): string        // creates and persists on first call
export function getDeviceLabel(): string | null
export function setDeviceLabel(label: string): void
```

## 3. Client — `apps/web/lib/api.ts`

Follow the existing conventions in that file exactly (`getApiBaseUrl()`, `videoAnalysisError`-style
error extraction, typed returns):

```ts
export async function solveDeviceCalibration(
  video: Blob, deviceId: string, deviceLabel: string | null, spec: CheckerboardSpecInput
): Promise<DeviceCalibrationResponse>

export async function getDeviceCalibration(deviceId: string): Promise<DeviceLensProfile | null>
export async function deleteDeviceCalibration(deviceId: string): Promise<void>
```

Mirror the Pydantic models as TS types in `apps/web/lib/deviceCalibration/types.ts`.

## 4. Page — `apps/web/app/calibrate-device/page.tsx`

Reuse `CameraPreview` (it already handles main-lens selection, 4K/60fps request, pinch zoom, and
exposes `getStream()`), plus the existing `Button` and `Card`.

Stages: `intro` → `recording` → `solving` → `result`.

**intro** — what to print, that inner corners are 9×6 for a 10×7 board, that the board must be flat,
and that the square must be measured with a ruler after printing because printers rescale. Inputs
for columns, rows and measured square size in mm, defaulting to 9 / 6 / 25.

**recording** — `MediaRecorder` on the camera stream, 30-second guide timer, and a live checklist of
the four things that make or break the solve:

- tilt the board 20–45°
- reach all four corners of the frame, not just the middle
- vary distance
- move slowly

**solving** — upload, spinner, then render `CalibrationQuality` honestly:

- `band` as a coloured chip: GOOD green, ACCEPTABLE amber, POOR red
- `rms_reprojection_px` to 2 decimals
- `views_used` of `views_submitted`
- `diagonal_fov_degrees`, and a clear warning when `fov_plausible` is false
- `advice` verbatim — the service already writes the right sentence for each failure

**result** — profile summary, a "Re-calibrate" action, and the capture resolution it was measured at.

Warn if the recording resolution differs from what the user will shoot cricket at. Rescaling works
within an aspect ratio but not across one.

## 5. Use the profile — the change that matters

In `wicket_box_calibration_service.py`, `_solve_registration` (~line 1412) calls:

```python
intrinsics_candidates = build_intrinsics_candidates(frame.shape[1], frame.shape[0])
```

Replace with: when a device profile exists for this analysis, build a **single** candidate from it
and skip the sweep entirely. Fall back to `build_intrinsics_candidates` otherwise.

- `WicketBoxCalibrationRegisterRequest` gains an optional `device_id`.
- The candidate built from a profile must carry `source="device_calibration"` and
  `confidence="HIGH"` so the distinction is visible downstream.
- Use the profile's real `distortion_coefficients` rather than zeros.
- `load_device_profile_for_frame()` already handles rescaling and returns `None` when it cannot —
  treat `None` as "sweep as before".

Also tighten the fallback: reject sweep candidates whose diagonal FOV falls outside
`MIN/MAX_PLAUSIBLE_DIAGONAL_FOV_DEG`. That alone would have rejected the 43.3° result above.

## 6. Badge

A small shared component showing lens provenance, placed on `/live` and `/video-analysis`:

- green "Lens calibrated" + the RMS
- amber "Lens estimated" + a link to `/calibrate-device`

This is what lets an operator know before bowling whether the numbers will be trustworthy.

---

## 7. Tests

Backend (`tests/`, pytest — mirror existing style):

1. `test_synthetic_focal_recovery` — render checkerboard views through a known `K` with
   `cv2.projectPoints`, solve, assert recovered `fx` within 1% and RMS < 1 px. The working
   generator is described above; regenerate it in the test rather than committing a video.
2. `test_insufficient_views_rejected` — 3 views raises `DeviceCalibrationError` with status 422.
3. `test_square_grid_rejected` — `CheckerboardSpec(columns=6, rows=6, ...)` raises.
4. `test_profile_roundtrip` — save then load returns an equal profile.
5. `test_rescale_within_aspect` — 1080×1920 → 2160×3840 doubles `fx`; 1920×1080 raises.
6. `test_registration_prefers_device_profile` — with a stored profile, only one intrinsics
   candidate is tried and it carries `source="device_calibration"`.
7. `test_implausible_fov_rejected_in_sweep` — a candidate at 43° diagonal FOV is excluded.

Frontend (`vitest`):

8. `getDeviceId()` is stable across calls and persists.
9. The result view renders `—` and the advice string when `fov_plausible` is false, and never
   renders a numeric quality claim for a POOR band.

---

## 8. Order

1. Routes + `main.py` registration, then confirm `/docs` shows the endpoints.
2. `deviceIdentity.ts` and the `api.ts` client.
3. Wire the profile into `_solve_registration` — **this is the only step that changes results**, so
   do it before the UI and verify against a real analysis in `outputs/video_analysis/`.
4. Tighten the fallback FOV bounds.
5. Build the page.
6. Badge.
7. Tests.

Do not skip step 3 in favour of finishing the UI. A beautiful calibration screen whose output is
never consumed changes nothing.
