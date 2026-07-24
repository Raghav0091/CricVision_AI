# Release Point V1 Validation

Generated: 2026-07-24T00:40:44.362620Z

## Scope

This report validates the current Release Point V1 baseline only. It does not tune thresholds, retrain models, change pose providers, or implement Milestone 2.

## Dataset Audit

- Available analysis folders: 69
- Usable Release V1 inputs: 14
- Selected validation clips: 14
- Requested initial target: 20-30 usable real deliveries
- Finding: existing complete inputs are below the requested validation set size.

## Annotation Methodology

Annotators inspect frame packages around the predicted release frame and label the first frame where the ball has physically separated from the hand and begins independent free flight.
If separation occurs between low-FPS frames, annotators choose the most defensible frame and may record an uncertainty interval.
Annotations are stored separately at `outputs/release_validation/release_annotations.json`; predictions are not overwritten.

## Baseline Prediction Freeze

- Baseline clip count: 14
- Ready predictions: 8
- Unresolved/failed/missing predictions: 6
- Baseline file: `outputs/release_validation/baseline_release_v1_results.json`
- Collection statuses: `{"failed": 6, "ran_current_algorithm": 7, "read_existing": 1}`
- Failed baseline cases:
  - `rv1_001` `analysis_20260718_005833_bbaf9d`: VideoReleasePointError: A ready primary ball track is required before Release Point V1.
  - `rv1_003` `analysis_20260718_085114_739b2f`: VideoReleasePointError: tracking_result.json is malformed.
  - `rv1_004` `analysis_20260718_235819_20a95d`: VideoReleasePointError: tracking_result.json is malformed.
  - `rv1_005` `analysis_20260719_132629_7400e0`: VideoReleasePointError: tracking_result.json is malformed.
  - `rv1_006` `analysis_20260720_131209_2c5403`: VideoReleasePointError: A ready primary ball track is required before Release Point V1.
  - `rv1_007` `analysis_20260721_011137_44c35d`: VideoReleasePointError: tracking_result.json is malformed.

## Objective Metrics

- Dataset/clip count: 14
- Human annotation count: 14
- Valid labelled count: 0
- Exact-frame accuracy: n/a
- Within +/-1 frame: n/a
- Within +/-2 frames: n/a
- MAE: n/a
- Median absolute error: n/a
- Catastrophic error rate: n/a
- Unresolved rate: 0.429
- Prediction coverage: 0.571

## Confidence And Method Analysis

Confidence bins are descriptive only; Release V1 confidence is not treated as calibrated probability.

```json
{
  "0.00-0.39": {
    "count": 0,
    "within_2_rate": null,
    "mae": null
  },
  "0.40-0.59": {
    "count": 0,
    "within_2_rate": null,
    "mae": null
  },
  "0.60-0.79": {
    "count": 0,
    "within_2_rate": null,
    "mae": null
  },
  "0.80-1.00": {
    "count": 0,
    "within_2_rate": null,
    "mae": null
  }
}
```

Method breakdown:

```json
{
  "fallback_trajectory_only": {
    "baseline_count": 5,
    "labelled_count": 0,
    "within_2_rate": null,
    "mae": null
  },
  "missing": {
    "baseline_count": 6,
    "labelled_count": 0,
    "within_2_rate": null,
    "mae": null
  },
  "trajectory_pose_inferred": {
    "baseline_count": 3,
    "labelled_count": 0,
    "within_2_rate": null,
    "mae": null
  }
}
```

## Pose-Quality Analysis

```json
{
  "bowling_end_assignment_uncertain": {
    "baseline_count": 8,
    "labelled_count": 0,
    "within_2_rate": null,
    "mae": null
  },
  "bowler_selection_uncertain": {
    "baseline_count": 3,
    "labelled_count": 0,
    "within_2_rate": null,
    "mae": null
  },
  "low_confidence_wrist": {
    "baseline_count": 1,
    "labelled_count": 0,
    "within_2_rate": null,
    "mae": null
  },
  "pose_unavailable_or_unreliable": {
    "baseline_count": 5,
    "labelled_count": 0,
    "within_2_rate": null,
    "mae": null
  },
  "trajectory_only_estimate": {
    "baseline_count": 5,
    "labelled_count": 0,
    "within_2_rate": null,
    "mae": null
  }
}
```

## Failure Categories

```json
{}
```

## Latency

Total baseline collection wall time recorded across clips: 374.565s. Successful current-algorithm runs: 7 clips, mean 52.279s per delivery, median 47.473s. Failed current-algorithm attempts: 6 clips, mean 1.436s before failure. Pose inference time per frame is not separately emitted by Release V1 yet; use these as delivery-level CPU baseline timings.

## Limitations

- No release accuracy can be claimed until manually labelled real deliveries exist.
- Existing complete Release V1 input set is smaller than the requested 20-30 clips.
- Failure-package categories require human review of incorrect/unresolved examples.
- RTMPose-m sufficiency cannot be judged until wrist/pose errors are correlated with labelled frame errors.

## Recommendations Before Tuning

1. Add enough complete processed deliveries to reach at least 20-30 validation clips.
2. Human-label the generated frame packages before changing any thresholds.
3. Repair or exclude malformed/non-ready tracking cases before interpreting model accuracy.
4. Use the weak flags analysis to decide whether bowler selection, wrist confidence, or trajectory recovery is the limiting factor.

## Completion Decision

MILESTONE 1 RELEASE POINT V1 = INSUFFICIENT DATA

Proceed to MILESTONE 2 - CANONICAL COMPLETE DELIVERY TRACK: not recommended until Release Point V1 has labelled baseline metrics.
