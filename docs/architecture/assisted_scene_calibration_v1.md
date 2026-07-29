# Assisted Scene Calibration and Metric Activation V1

## Ownership

`SceneCalibrationResult` is the backend-owned state for the production Video
Analysis calibration workflow. Wicket Observation V1 remains the only detector
and landmark pipeline called by an explicit Detect Wickets attempt. Real Pitch
Registration V1 consumes the persisted observation; it does not invoke the stump
detector.

The workflow stages are:

`NOT_STARTED -> DETECTING_WICKETS -> OBSERVING_WICKETS -> GENERATING_POSE`

The terminal candidate stages are `NEEDS_ADJUSTMENT`,
`GROUND_PLANE_READY`, `METRIC_3D_READY`, `INSUFFICIENT_EVIDENCE`, and
`FAILED`. Redetection starts a new attempt and accepted snapshots are revisioned.

## Manual evidence

The six required anchors are the left base, right base, and top center for each
wicket. Missing automatic anchors remain unavailable until explicitly added.
Coordinates are stored in native video pixels with automatic, manually adjusted,
or manually added provenance.

When all three anchors for one wicket are manual, they supersede conflicting
detector-derived point, line, center, and envelope constraints for that wicket.
Those constraints remain in diagnostics with a rejection reason. The solver then
uses a six-point assisted minimum and an OpenCV SQPnP fallback if RANSAC cannot
seed from exactly six points. All normal automatic registration defaults remain
unchanged.

Optional popping-crease endpoints are low-weight pitch-plane evidence. They are
used only when explicitly placed and may be marked refinement or validation
only.

## Acceptance thresholds

All assisted acceptance thresholds live in
`services/api/services/scene_calibration_service.py`.

| Check | Metric 3D | Ground plane | Direction |
| --- | ---: | ---: | --- |
| Reprojection RMSE | 5 px | 9 px | maximum |
| Median reprojection error | 6 px | 10 px | maximum |
| Maximum inlier error | 12 px | 18 px | maximum |
| Wicket-envelope score | 0.30 | 0.20 | minimum |
| Temporal-stability score | 0.50 | 0.25 | minimum |
| Independent-scene score | 0.40 | 0.25 | minimum |
| Optional crease support, when supplied | 0.30 | 0.20 | minimum |
| Assignment ambiguity | 0.25 | 0.35 | maximum |
| Camera-position perturbation spread | 0.50 m | 1.00 m | maximum |
| Rotation perturbation spread | 2 degrees | 4 degrees | maximum |
| Projected-overlay sensitivity | 8 px | 15 px | maximum |

Both levels also require all six valid wicket anchors, a compatible candidate
classification, stable deterministic perturbation, no optimized parameter at a
bound, and every named registration plausibility check to pass. The plausibility
checks expose finite pose, camera height from 0.2 to 20 m, positive anchor depth,
camera facing, camera distance from 0.5 to 150 m, bounded focal length, near/far
size order, and scene-sized projected wickets.

Manual confirmation cannot bypass these checks. `VISUAL_ONLY` can display the
overlay but unlocks no metric analytics.

## Metric activation

An accepted calibration is stored as
`reports/accepted_scene_calibration_v1.json`; later acceptances use `_r2`, `_r3`,
and so on. Physics V1 reads only the active accepted assisted snapshot.

`METRIC_3D_READY` permits validated airborne and ground metrics.
`GROUND_PLANE_READY` permits bounce position, line and length, and top-down
ground replay while keeping airborne speed, height, and metric swing locked.
An unaccepted, visual-only, missing, or invalid assisted calibration leaves
Physics V1 in image-space mode and preserves detections, tracking, and replay.
