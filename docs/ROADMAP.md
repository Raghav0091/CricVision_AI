# CricVision Pro Roadmap

## Milestone 1 — Architecture scaffold and health checks

Next.js application shell, FastAPI contracts, process-local stores, worker boundaries, reusable CV package, safe calibration failure, and architecture specifications.

## Milestone 2 — Live camera and stump alignment

Field-test rear-camera selection, permission recovery, responsive red alignment guides, rotation, aspect ratios, and mobile/tablet performance.

## Milestone 3 — Calibration capture and endpoint

Harden image transport, evidence storage, validation errors, retries, and environment-context versioning.

## Milestone 4 — Dedicated stump detector

Integrate a measured six-stump detector, geometry gates, model provenance, confidence levels, and calibration regression fixtures. No box-only success.

## Milestone 5 — Live delivery clip capture

Build the bounded rolling buffer, motion trigger, pre-roll, recording window, cooldown, clip persistence, and delivery registration.

## Milestone 6 — Moving delivery ball tracker

Connect detector candidates to a motion-first tracker with static-object rejection, calibration-aware constraints, diagnostics, and honest quality gates.

## Milestone 7 — Replay overlay after each delivery

Render observed/fitted trajectories and measured event markers, return replay status over WebSocket, and preserve capture responsiveness.

## Milestone 8 — Estimated 3D replay

Add calibrated camera/pitch mapping and uncertainty-aware 3D visualization. Sparse evidence must produce partial or unavailable output, never a fabricated path.

## Milestone 9 — Estimated LBW/DRS-style projection

Explore non-official projection only after calibration and trajectory validation. Gate every output by confidence and label it as an estimate. Do not imply official Hawk-Eye or umpiring accuracy.
