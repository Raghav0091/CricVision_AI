import json

import numpy as np

from scripts.release_region_dataset_builder import (
    annotation_template,
    atomic_write_json,
    clamp_box,
    crop_image,
    hand_crop_box,
    sample_window,
    sha256_file,
    validate_annotation,
)
from scripts.release_region_dataset_annotator import copy_previous_bbox


def test_sha256_deduplicates_identical_sources(tmp_path):
    left = tmp_path / "left.mp4"
    right = tmp_path / "right.mp4"
    left.write_bytes(b"same video bytes")
    right.write_bytes(b"same video bytes")
    assert sha256_file(left) == sha256_file(right)


def test_release_window_is_dense_near_center_and_sparse_outside():
    rows = sample_window(60, 60.0, 197, "high")
    by_frame = {row["frame_index"]: row["sampling_mode"] for row in rows}
    assert by_frame[60] == "dense_release_window"
    assert by_frame[50] == "dense_release_window"
    assert len(rows) < 60
    assert list(by_frame) == sorted(by_frame)


def test_crop_coordinates_are_clamped_and_correct():
    frame = np.arange(100 * 200).reshape(100, 200)
    box = clamp_box([-10, 20, 250, 80], 200, 100)
    assert box == [0, 20, 200, 80]
    assert crop_image(frame, box).shape == (60, 200)


def test_hand_roi_is_not_fabricated_without_pose():
    box, wrist = hand_crop_box(None, 10, 1280, 720, [0, 0, 1280, 720])
    assert box is None
    assert wrist is None


def test_sequence_id_and_suggestions_remain_non_ground_truth():
    meta = {
        "sample_id": "a_f000001",
        "sequence_id": "seq_a",
        "analysis_id": "a",
        "source_frame_index": 1,
        "wrist_evidence": None,
        "automatic_suggestions": {
            "hard_negative_candidates": [
                {"hard_negative_candidate": True, "suggestion_only": True}
            ]
        },
    }
    row = annotation_template(meta)
    assert row["sequence_id"] == "seq_a"
    assert row["hard_negative_label"] == "unlabeled"
    assert row["automatic_suggestions"]["hard_negative_candidates"][0]["suggestion_only"]


def test_annotation_schema_round_trip_and_atomic_resume(tmp_path):
    meta = {
        "sample_id": "a_f000001", "sequence_id": "seq_a", "analysis_id": "a",
        "source_frame_index": 1, "wrist_evidence": None,
        "automatic_suggestions": {},
    }
    row = annotation_template(meta)
    validate_annotation(row)
    path = tmp_path / "annotations.json"
    atomic_write_json(path, {"annotations": [row]})
    assert json.loads(path.read_text())["annotations"][0] == row
    assert not path.with_suffix(".json.tmp").exists()


def test_copy_previous_bbox_stays_within_sequence():
    rows = [
        {"sequence_id": "seq_a", "ball_bbox_xyxy": [1, 2, 3, 4]},
        {"sequence_id": "seq_b", "ball_bbox_xyxy": [9, 9, 12, 12]},
        {"sequence_id": "seq_a", "ball_bbox_xyxy": None},
    ]
    assert copy_previous_bbox(rows, 2) == [1, 2, 3, 4]
