# Calibration Specification

## Input

`POST /calibration/solve` accepts JSON containing:

- `frame_data_url`: base64 browser image data URL;
- `frame_width` and `frame_height`: positive native video dimensions;
- `box_layout`: normalized `striker` and `non_striker` rectangles, each with `x`, `y`, `width`, and `height` in the range 0–1.

Normalized coordinates make the layout independent of CSS scaling and device pixel ratio. The stored image is local development evidence, not a solved calibration.

## Stump detection result

A future detector result must include candidate bounding boxes, class labels, confidence, which set each candidate belongs to, expected stump count, validation warnings, model identity/version, and frame dimensions. A solve succeeds only when both stump sets pass geometry and evidence thresholds.

## Environment context

On real success the worker returns:

- striker and non-striker stump centers;
- measured pitch axis and corridor polygon;
- source frame dimensions;
- calibration quality and reasons;
- detector/model identity;
- timestamp and evidence references;
- assumptions and warnings needed by downstream tracking.

The reusable `EnvironmentContext` dataclass establishes the geometry contract. It does not manufacture detector evidence.

## Quality levels

- `Unavailable`: the required detector or evidence does not exist.
- `Poor`: detections exist but geometry is unsafe for downstream use.
- `Partial`: enough evidence for limited overlays, not projection claims.
- `Good`: both stump sets and pitch geometry satisfy defined validation gates.

Quality is evidence-based. A frontend box match by itself is not calibration.

## Failure modes

- dedicated detector missing;
- invalid or empty frame payload;
- one stump set absent or occluded;
- incorrect stump count or ambiguous objects;
- detections outside expected alignment regions;
- implausible perspective or pitch axis;
- low confidence or inconsistent geometry;
- excessive blur, darkness, glare, or camera motion.

The current endpoint safely returns `Unavailable` with `stump_detector_missing` after saving a valid frame. No fake success, virtual pitch, DRS/LBW, or official accuracy claim is permitted.
