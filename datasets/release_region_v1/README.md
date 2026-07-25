# CricVision Release-Region Dataset V1

This is an internal, offline annotation dataset. Human labels and automatic
suggestions are never production Release Point inputs.

## Commands

```powershell
python scripts/release_region_dataset_builder.py audit
python scripts/release_region_dataset_builder.py build
.venv\Scripts\python.exe scripts/release_region_dataset_builder.py suggestions --models e4c
python scripts/release_region_dataset_annotator.py
python scripts/release_region_dataset_builder.py summary
```

`manifest.json` records clean source provenance, SHA-256 duplicate groups,
release-window evidence, crop coordinates, temporal sequences, and sampling
mode. `annotations.json` is resume-safe manual ground truth.

Automatic release phases and detector candidates are suggestions only.
They must not be copied into ground truth without manual review.

Generated images under `full_frames/`, `bowler_rois/`, and `hand_rois/` are
gitignored. Crops preserve native source pixels and are not upscaled.

## Annotation

The local Tkinter annotator supports full-frame, bowler, and hand views; zoom;
mouse-drawn ball boxes; copy-forward within a sequence; no-ball and uncertain
states; visual-condition labels; and atomic save/resume.

Complete the frame as `yes`, `no`, or `uncertain`. A `yes` frame should have a
ball box whenever the ball can be localized. Detector and hard-negative
suggestions remain untrusted until manually confirmed.
