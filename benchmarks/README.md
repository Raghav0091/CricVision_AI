# CricVision AI Benchmark Clips

Local benchmark clips for comparing Video Analysis behavior across speed modes.

## Setup

1. Put local test clips in `benchmarks/clips/`.
2. Do **not** commit video files to git.
3. Record expected outcomes in `benchmarks/expected_results.csv`.
4. Add free-form notes per clip in `benchmarks/notes/` if helpful.

## What to compare

Run the same clip through Video Analysis using:

- Smart Balanced
- Smart Accurate
- Debug Full Frame

For each run, record:

- Ball detection rate
- Ball tracking rate
- Whether bounce was found
- Whether impact was found
- Processing time (from performance details when enabled)
- Notes on visual quality or failure type

## expected_results.csv

Columns:

| Column | Description |
|--------|-------------|
| `clip_name` | Filename in `benchmarks/clips/` |
| `view_type` | Camera view (umpire, batter, bowler, side) |
| `lighting` | Lighting conditions |
| `total_frames` | Frame count |
| `expected_ball_visible_start` | First frame ball should appear |
| `expected_ball_visible_end` | Last frame ball should appear |
| `expected_bounce_found` | `yes` / `no` / `unknown` |
| `expected_impact_found` | `yes` / `no` / `unknown` |
| `expected_tracking_quality` | Excellent / Good / Medium / Poor |
| `notes` | Free text |

Use Detection Health and Raw Detection Preview in Video Analysis to inspect detector vs tracker behavior without changing analysis output.

## Trajectory Replay v1

After analysis, Video Analysis also shows **CricVision Trajectory Replay** — an approximate synthetic pitch view built from tracked image-space ball points. It does not change detection or tracking behavior. Speed, swing, and spin are not calibrated in v1.
