# Claude Code Prompt — Live Stump Calibration + Virtual Overlay (FullTrack-parity visuals)

> Paste into Claude Code at the repo root. Ponytail rules apply: reuse before adding, deletion over
> addition, fewest files, no new dependencies. Everything below is measured from reference footage —
> treat the numbers as specification, not suggestion.

---

## Scope

`/live` **camera mode only**. Nothing in this task touches `/video-analysis`, `/pitch-space-analysis`,
`virtual-pitch-lab`, upload mode, or `ExperimentalDeliveryTest`. Do not modify the backend unless a
step explicitly says so.

The setup wizard (`components/live/SetupWizard.tsx`) already exists and runs before this flow. This
task covers everything from the moment the camera opens to the moment the locked overlay is drawn.

---

## Stage machine

Extend the existing `LiveStage` in `apps/web/lib/types.ts`. The camera path becomes:

```
orientation-gate  ->  align-stumps  ->  mapping  ->  solving-calibration  ->  setup-complete
                          ^                                  |
                          +-------------- redetect ----------+
```

`orientation-gate` and `mapping` are new. `align-stumps`, `solving-calibration` and `setup-complete`
already exist — extend them, do not replace them.

---

## 1. Shared chrome

### 1.1 InstructionCard

One component, used by every stage. Reference behaviour: a white card pinned near the top of the
camera view with a small notched flag on its **bottom-left** corner.

```tsx
// components/live/InstructionCard.tsx
export function InstructionCard({ tone = "info", children }: {
  tone?: "info" | "warning";
  children: React.ReactNode;
})
```

- Container: `absolute left-0 right-4 top-16`, white background, no rounding on the left edge,
  `rounded-r-lg`, `px-4 py-3`, subtle shadow.
- Notch: a ~14 px triangle at the bottom-left, same white, rendered with a CSS `clip-path` pseudo
  element or an inline SVG. Do not add a library for this.
- `tone="info"`: text `#111`, centred, `text-sm font-semibold leading-5`.
- `tone="warning"`: text `#D32029`, right-aligned, `text-sm font-bold leading-5`.
- Multi-line content is passed as children with `<br />` or separate `<p>` elements.

### 1.2 Close button

Grey circle, white ×, `absolute right-4 top-4`, 32 px, `bg-white/85 text-ink`. Returns to `stage="setup"`.

### 1.3 Bottom action button

White pill, centred, `absolute bottom-8 left-1/2 -translate-x-1/2`, `px-8 py-3 rounded-full`,
label in **uppercase blue** (`#1A73E8`) `text-sm font-bold tracking-wide`. This is deliberately not
the lime `Button` — the camera stages use a light-on-dark treatment. Add it as a `variant="camera"`
on the existing `Button` rather than a new component.

---

## 2. `orientation-gate`

Blocks entry to `align-stumps` until the phone is held upright.

- Read tilt from `DeviceOrientationEvent`. On iOS 13+ call
  `DeviceOrientationEvent.requestPermission()` behind a user gesture — the wizard's final **Start**
  press is that gesture, so request it there and pass the result down.
- Angle to display: deviation from vertical, i.e. `Math.round(Math.abs(90 - beta))` clamped to 0–180.
  Reference app shows values like 165°, 143°, 130°, 117°, 106° as the phone is raised from flat.
- Threshold: **10°**. Below it, advance automatically to `align-stumps`.
- While above threshold, render `InstructionCard` with `tone="warning"`:

  ```
  Please hold the device in portrait view
  Current Angle: {angle}°
  It should be less than 10°
  ```

- **The guide boxes are hidden while this warning is showing.** Camera preview stays live underneath.
- If the API is unavailable or permission is denied, skip straight to `align-stumps` and set a
  `orientationUnavailable` flag. Never trap the user behind a sensor that does not exist.

---

## 3. `align-stumps` — the red guide boxes

### 3.1 Geometry (measured, normalised to the preview box)

Replace `CAMERA_ALIGNMENT_BOXES` in `components/live/StumpAlignmentOverlay.tsx` with:

```ts
export const CAMERA_ALIGNMENT_BOXES: BoxLayout = {
  striker:     { x: 0.347, y: 0.339, width: 0.309, height: 0.202 },
  non_striker: { x: 0.199, y: 0.586, width: 0.605, height: 0.265 }
};
```

Both boxes are horizontally centred (centre = 0.5015 in both cases — keep the symmetry). The current
values are far too narrow; the reference striker box is **2.2× wider** and the non-striker box **2× wider**
than what is in the file today. Grep for both constants before changing them — `UPLOAD_ALIGNMENT_BOXES`
is a separate layout and must not be touched.

### 3.2 Appearance

- Border: `2px dashed`, colour `#E0201F`. Measured dash rhythm is ~7 px on / ~7 px off on a 392 px-wide
  preview, which plain CSS `border-style: dashed` reproduces closely enough. Do not hand-roll an SVG
  dash array for this.
- No fill, no rounding, no glow. Remove the existing `shadow-[0_0_18px_rgba(255,85,79,0.32)]` and the
  `rounded-md` from `AlignmentBox` — the reference boxes are hard-edged.
- Label: a small pill **straddling the top border**, horizontally centred on the box, vertically
  centred on the border line itself (`-translate-y-1/2`). Light grey background `#D9D9D9`, black text,
  `text-[10px] font-bold`, `px-2 py-0.5`, slight rounding. Text: `Striker Stumps`, `Non-Striker Stumps`.
  The current implementation puts a red pill above the box, left-aligned — change both.

### 3.3 Behaviour — boxes are fixed

This is the important behavioural change. In the reference app the boxes **never move**. The user
zooms and re-aims the camera to fit the stumps into them.

- Render the boxes as a static overlay. Do not mount `DraggableGuideBoxes` by default.
- Add pinch-to-zoom on the camera preview via `MediaStreamTrack.applyConstraints({ advanced: [{ zoom }] })`,
  reading the range from `track.getCapabilities().zoom`. If the track reports no `zoom` capability,
  fall back to a CSS `transform: scale()` on the `<video>` **and** record the scale factor, because
  calibration must know the frame was digitally cropped.
- Keep `DraggableGuideBoxes` and `DraggableStumpKeypoints` reachable behind a small
  **Adjust manually** text button. They are a better fallback than the reference app has when
  detection struggles — do not delete them.
- `InstructionCard` (`tone="info"`):

  ```
  Fit the stumps completely in the boxes then press Continue.
  Pinch to zoom in or out!
  ```

- Bottom button: `CONTINUE` → advances to `mapping`.

---

## 4. `mapping`

Runs while the backend detects. Reuses `solveCalibration` from `lib/api.ts` — no new endpoint.

- `InstructionCard` (`tone="info"`):

  ```
  Please wait while we map the stumps.
  Make sure the non-striker stumps are fully visible.
  ```

- Hide the guide boxes. Draw detected landmarks as filled circles, radius ~5 px:
  - **Red** `#E0201F` — raw detection from the current frame
  - **Blue** `#1A3FD6` — the smoothed/accepted set

  Both are drawn simultaneously so the user sees them converge. Render inside the existing
  `StumpAlignmentOverlay` SVG rather than as separate DOM nodes.

- If the response reports the striker (far) wicket missing, do not proceed. Return to `align-stumps`
  with the warning card and keep the boxes visible. A single-wicket solve is degenerate and must
  never reach `setup-complete`.

- **Tilt the Camera Up** modal — show when the response indicates the release area is clipped at the
  top of frame (top of the striker detection is within 8% of the frame top). White rounded card,
  centred, max-width ~20rem:

  ```
  Tilt the Camera Up
  The bowler's release point is cut off at the top. Please point the camera up slightly.
                                                                                    OK
  ```

  Title `text-lg font-bold`, body `text-sm leading-6 text-black/70`, `OK` right-aligned, blue,
  uppercase-free. Dismissing returns to `align-stumps`.

---

## 5. `setup-complete` — the virtual overlay

The overlay itself already exists in `StumpAlignmentOverlay.tsx` (corridor, length zones). This step
locks it and keeps it on screen.

### 5.1 Lock the pose

Do **not** re-solve per frame. Once `solveCalibration` succeeds, freeze that result in state and
render everything from it. The only thing that clears it is the Redetect button. A frozen,
slightly-imperfect anchor looks stable; a live one jitters and drifts off the stumps.

### 5.2 Chrome

- `REDETECT` pill, `absolute left-4 top-4`, white background, blue uppercase text, same treatment as
  the bottom action button. Clears the locked calibration and returns to `align-stumps`.
- `InstructionCard` (`tone="info"`):

  ```
  Setup Complete. Press "Redetect" if the pitch is not detected correctly.
  Start bowling to trigger the recording!
  ```

### 5.3 Virtual wicket rendering

Draw from the locked calibration, inside the existing overlay SVG:

- **Stumps** — three gold cylinders per wicket. Each is a 4-point polygon (silhouette) filled with a
  horizontal linear gradient `#E8B33A → #C98F24`, plus a lighter `#F2C75C` cap ellipse. Use the true
  regulation size from `packages/cricket_vision/calibration/cricket_pitch_geometry.py`
  (`STUMP_HEIGHT_M`, `WICKET_WIDTH_M`, `STUMP_DIAMETER_MAX_M`) — do **not** exaggerate the dimensions.
  The reference app draws them visibly thicker and taller than reality, which is exactly what makes
  its misalignment obvious.
- **Bails** — two small quads spanning adjacent stump tops. The reference app omits these; include
  them, they measurably improve how well-seated the wicket looks.
- **Corridor** — the existing `pitch_corridor` quad, `fill="#8E93C8" fill-opacity="0.42"`, no stroke.
- **Depth order** — SVG has no z-buffer. Emit far wicket → corridor → near wicket, in that document
  order.

### 5.4 Coordinate convention

All 3D geometry uses the canonical frame already defined in `cricket_pitch_geometry.py`:

```
+X lateral across the pitch, +Y bowler-end → striker-end, +Z upward
```

That file is the single source of truth. Do not introduce a second convention in TypeScript, and do
not swap Y and Z — the legacy Calibration V2 ordering already differs and adding a third variant
will produce silent geometry bugs.

---

## 6. Files

Expect to touch these and no others:

```
apps/web/components/live/InstructionCard.tsx      NEW  (~40 lines)
apps/web/components/live/StumpAlignmentOverlay.tsx EDIT box constants, AlignmentBox, label pill,
                                                        keypoint dots, virtual wicket
apps/web/components/live/CameraPreview.tsx        EDIT pinch-to-zoom via applyConstraints
apps/web/app/live/page.tsx                        EDIT new stages, locked calibration state, chrome
apps/web/components/ui/Button.tsx                 EDIT add variant="camera"
apps/web/lib/types.ts                             EDIT extend LiveStage
```

No new dependency. No new route. No backend change.

---

## 7. Acceptance

1. Wizard **Start** → orientation warning appears with a live angle while the phone is flat, and
   clears on its own when raised past 10°.
2. Guide boxes match the measured rectangles, are hard-edged red dashed, carry grey label pills
   straddling the top border, and **do not move** when dragged.
3. Pinch zooms the camera; boxes stay put.
4. `CONTINUE` → mapping card, red and blue dots visible over the stumps.
5. Far wicket not visible → returns to `align-stumps` with a warning, never reaches setup-complete.
6. Success → `REDETECT` pill top-left, setup-complete card, gold wicket drawn on the real stumps.
7. The overlay does not jitter frame to frame — confirm the locked pose is genuinely frozen by
   checking that the calibration object identity does not change while the camera shakes.
8. `npm run build` passes clean.

Verify on the phone through `scripts/start_phone_tunnel.ps1` — `DeviceOrientationEvent`,
`applyConstraints({ zoom })`, and the camera itself all behave differently on desktop.
