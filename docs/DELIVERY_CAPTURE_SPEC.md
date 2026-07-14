# Delivery Capture Specification

## Rolling buffer

The browser will keep a bounded in-memory buffer of timestamped encoded frames or media chunks while capture is active. Buffer size must be fixed to avoid unbounded mobile memory growth. Orientation, dimensions, frame rate, and monotonic timestamps are recorded with each clip.

## Trigger and clip window

The delivery trigger is a future measured motion signal, not a UI timer. Initial target parameters are configurable rather than hard-coded:

- pre-roll: retain enough context to include the bowler approach/release;
- recording window: continue through bounce, batter, and stumps;
- cooldown: reject duplicate triggers while the current clip is finalized;
- timeout: return to waiting if a trigger does not produce a valid clip.

Exact values require device and cricket field testing. Automatic delivery detection is not implemented in this scaffold.

## Saved clip

Use a collision-safe name such as `{session_id}_{delivery_number:04d}_{utc_timestamp}.webm`. Store source timing and camera metadata beside the clip. A delivery number increments only after the clip is finalized and registered successfully.

## Analysis handoff

1. Finalize the buffered clip.
2. Register it with `POST /deliveries` using the active session ID and local/uploaded path.
3. Queue `POST /analysis` with the returned delivery ID.
4. Subscribe to status updates and keep live capture responsive while the worker runs.
5. Present replay only when the worker returns a completed, confidence-labelled result.

Failed uploads or analysis jobs retain their delivery record and expose a retryable error. The frontend must not infer tracking, speed, trajectory, or outcome while a job is pending or unavailable.
