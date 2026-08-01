# Calibration Ambiguity Resolution V1

This milestone resolves lateral mirror ambiguity without changing camera-pose
geometry thresholds. Symmetric wicket evidence remains useful for fitting and
validation, but it is explicitly insufficient for deciding pitch left versus
pitch right.

## Coordinate Semantics

Virtual Pitch V1 keeps its canonical coordinate system:

- origin: bowler-end middle-stump base
- `+x`: pitch-right when looking from bowler end toward striker end
- `+y`: bowler end toward striker end
- `+z`: upward

Orientation evidence stores the native-video relationship as either
`IMAGE_LEFT_IS_PITCH_LEFT` or `IMAGE_LEFT_IS_PITCH_RIGHT`. Off-side and leg-side
labels are not produced because batter handedness is not part of this feature.

## Evidence

`OrientationEvidence` records the source, semantic label, native pixel location
when applicable, confidence, uncertainty, supported candidate IDs, rejected
candidate IDs, explanation, timestamp, and whether the evidence was user
confirmed.

Supported evidence types are:

- `USER_CONFIRMED_LATERAL_ORIENTATION`
- `SAVED_CAMERA_ORIENTATION_PRESET`
- `SEMANTIC_PITCH_EDGE_POINT`
- `SEMANTIC_CREASE_ENDPOINT`
- `TRUSTED_CAMERA_END`
- `TRUSTED_SESSION_DIRECTION`
- `FUTURE_AUTOMATIC_ASYMMETRIC_EVIDENCE`

Symmetric sources are listed in diagnostics as insufficient by themselves:
two symmetric wickets, wicket centres, wicket widths, unlabelled outer wicket
anchors, centreline, symmetric pitch boundaries, generic crease lines, ball
trajectory without a semantic side reference, camera height, camera distance,
focal length, and near/far scale alone.

## Candidate Rescoring

Registration still generates the same candidate families. When explicit
orientation evidence exists, the registration call receives the required
lateral mapping. Candidates with the conflicting mirror are retained in the JSON
diagnostics, but are marked ineligible and receive a zero arbitration score.
Their reprojection, temporal, envelope, physical, and uncertainty data are not
improved by orientation evidence.

After candidate selection, the existing geometric validation gates are rerun.
`lateral_orientation_resolved` is an explicit acceptance check. A calibration can
reach `GROUND_PLANE_READY` only if orientation is resolved and all ground-plane
thresholds still pass. Orientation confirmation alone cannot unlock
`METRIC_3D_READY`.

## User Workflow

The existing Scene Calibration workspace shows an orientation step only when
mirror ambiguity is present. The user chooses:

- image left equals pitch left
- image left equals pitch right
- not sure

The UI also shows candidate A/B summaries and lets the user toggle which backend
projection overlay is visible. The choice is semantic, not a visual beauty
contest.

## Semantic Anchors

Optional anchors can be placed for:

- `near_popping_crease_left`
- `near_popping_crease_right`
- `far_popping_crease_left`
- `far_popping_crease_right`
- `pitch_left_edge_reference`
- `pitch_right_edge_reference`

A labelled left/right pair can resolve lateral orientation when internally
consistent. A single point is insufficient. Contradictory labelled pairs or
anchors that contradict the user-confirmed mapping are rejected.

## Presets

`CameraOrientationPreset` stores a reusable fixed-camera orientation only. It
does not prove the camera pose remains unchanged and does not accept a
calibration. Reuse requires compatible resolution/aspect/camera-end metadata and
explicit user confirmation of the same fixed setup.

For future live calibration, the preset can remove repeated left/right
questions. Wicket detection, current-video pose refinement, validation, and
explicit acceptance remain required.
