# Live Bowling Workflow

## Stage 1: Setup Guide

Mount the phone on a tripod or fixed support behind the non-striker stumps. All six stumps must be visible, lighting must be sufficient for short exposures, and players or equipment must not block the camera. The user confirms these conditions before camera setup starts.

## Stage 2: Stump Alignment

The frontend opens the browser camera, preferring the rear camera on mobile. It draws two responsive red dashed regions: a smaller striker-stump region in the upper/middle frame and a larger, constrained non-striker region near the camera. The user fits both stump sets inside the regions and presses Continue.

Alignment regions are composition guides only. Before Continue there is no blue line, pitch patch, virtual stump, DRS/LBW claim, or inferred calibration.

## Stage 3: Calibration Solve

Continue captures the current video frame at its native video dimensions. The frontend posts the image, dimensions, and normalized alignment-box layout to `/calibration/solve`, then shows `Detecting stumps and building pitch context...`.

The backend stores the frame and invokes dedicated stump validation when available. A failed solve returns a machine-readable reason and user message; the frontend returns to alignment. Missing detection returns `stump_detector_missing` and `Dedicated stump detector not available yet.` It never advances as a real success.

## Stage 4: Setup Complete

Only a successful detector result enters setup complete. The camera remains open. A virtual pitch/stump overlay is shown only when the response contains measured environment context. Redetect discards current context and returns to alignment. Start Capture enters the live delivery state.

Developer mock calibration may later be enabled for UI preview only. It must be opt-in and visibly labelled `Mock calibration preview — not real stump detection`.

## Stage 5: Live Capture

The browser keeps the camera open and maintains a bounded rolling frame buffer. A measured delivery-motion trigger selects pre-roll plus a recording window, saves one clip, increments delivery count, and observes a cooldown before accepting another trigger. The initial scaffold displays waiting/recording state but does not claim that capture exists.

## Stage 6: Analysis and Replay

The saved clip is registered through the API and queued for the worker. The worker performs ball detection, moving-ball tracking, confidence-gated physics trajectory estimation, and replay rendering. Status is sent to the frontend, initially over WebSocket. The result view shows measured evidence and confidence; unavailable or weak results remain labelled as such.
