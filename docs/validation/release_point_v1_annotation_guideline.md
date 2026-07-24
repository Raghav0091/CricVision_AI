# Release Point V1 Annotation Guideline

## Target Event

Select the first frame where the ball has physically separated from the bowler's hand and begins independent free flight.

## Frame Choice Rule

- If the ball is visibly touching or still hidden in the hand, do not mark that frame as release.
- If the next frame shows clear separation and independent motion, select that separated frame.
- If separation likely occurred between two frames, select the most defensible frame and record `uncertainty_start_frame` and `uncertainty_end_frame`.
- If the hand or ball is too blurred, occluded, or outside frame, mark `annotation_status` as `uncertain` or `not_visible`.
- Do not use model confidence as ground truth. Inspect the clean frame package.

## Required Annotation Fields

- `annotation_status`: `labeled`, `uncertain`, or `not_visible`
- `human_release_frame`: integer frame number when `annotation_status` is `labeled`
- `human_annotation_confidence`: `high`, `medium`, or `low`
- `release_visibility`: `visible`, `partially_visible`, or `not_visible`
- `uncertainty_start_frame` and `uncertainty_end_frame`: optional inclusive interval
- `failure_categories`: optional list for incorrect/unresolved cases
- `notes`: short human note

## Failure Categories

- `wrong_bowler_selected`
- `pose_wrist_inaccurate`
- `ball_invisible_near_release`
- `ball_detector_late`
- `tracker_begins_too_late`
- `backward_trajectory_inaccurate`
- `wrong_bowling_arm`
- `calibration_bowling_end_issue`
- `low_fps_ambiguity`
- `other`
