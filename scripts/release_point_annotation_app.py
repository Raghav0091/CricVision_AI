"""Local Streamlit annotator for Release Point V1 validation.

Launch from the repository root:

    streamlit run scripts/release_point_annotation_app.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = PROJECT_ROOT / "outputs" / "release_validation"
BASELINE_PATH = VALIDATION_ROOT / "baseline_release_v1_results.json"
ANNOTATIONS_PATH = VALIDATION_ROOT / "release_annotations.json"
GUIDELINE_PATH = PROJECT_ROOT / "docs" / "validation" / "release_point_v1_annotation_guideline.md"
FAILURE_CATEGORIES = [
    "wrong_bowler_selected",
    "pose_wrist_inaccurate",
    "ball_invisible_near_release",
    "ball_detector_late",
    "tracker_begins_too_late",
    "backward_trajectory_inaccurate",
    "wrong_bowling_arm",
    "calibration_bowling_end_issue",
    "low_fps_ambiguity",
    "other",
]


def main() -> None:
    st.set_page_config(page_title="CricVision Release V1 Annotator", layout="wide")
    st.title("Release Point V1 Human Annotation")
    st.caption("Local validation utility. It does not change model predictions.")

    baseline = _read_json(BASELINE_PATH)
    annotations_doc = _load_or_create_annotations(baseline)
    records = baseline.get("records", [])
    annotations = {
        item["validation_id"]: item for item in annotations_doc.get("annotations", [])
    }
    if not records:
        st.error("No baseline records found. Run: python scripts/release_point_validation.py baseline")
        return

    if "delivery_index" not in st.session_state:
        st.session_state.delivery_index = _first_unlabeled_index(records, annotations)
    st.session_state.delivery_index = max(
        0, min(st.session_state.delivery_index, len(records) - 1)
    )

    annotated_count = sum(
        1
        for record in records
        if annotations.get(record["validation_id"], {}).get("annotation_status")
        in {"labeled", "uncertain", "not_visible"}
    )
    st.progress(annotated_count / len(records), text=f"Annotated {annotated_count} / {len(records)}")

    _navigation(records)
    record = records[st.session_state.delivery_index]
    annotation = annotations.setdefault(
        record["validation_id"], _annotation_template_from_record(record)
    )
    _sync_prediction_fields(annotation, record)

    left, right = st.columns([0.62, 0.38], gap="large")
    with right:
        _details_panel(record, annotation)
    with left:
        current_frame = _frame_viewer(record, annotation)

    st.divider()
    _annotation_form(record, annotation, annotations_doc, current_frame)


def _navigation(records: list[dict[str, Any]]) -> None:
    cols = st.columns([1, 1, 2, 1])
    with cols[0]:
        if st.button("Previous Delivery", use_container_width=True):
            st.session_state.delivery_index = max(0, st.session_state.delivery_index - 1)
            st.rerun()
    with cols[1]:
        if st.button("Next Delivery", use_container_width=True):
            st.session_state.delivery_index = min(
                len(records) - 1, st.session_state.delivery_index + 1
            )
            st.rerun()
    with cols[2]:
        labels = [
            f"{item['validation_id']} | {item['analysis_id']}" for item in records
        ]
        selected = st.selectbox(
            "Delivery",
            options=list(range(len(records))),
            format_func=lambda index: labels[index],
            index=st.session_state.delivery_index,
            label_visibility="collapsed",
        )
        if selected != st.session_state.delivery_index:
            st.session_state.delivery_index = selected
            st.rerun()
    with cols[3]:
        if st.button("Skip", use_container_width=True):
            st.session_state.delivery_index = min(
                len(records) - 1, st.session_state.delivery_index + 1
            )
            st.rerun()


def _details_panel(record: dict[str, Any], annotation: dict[str, Any]) -> None:
    st.subheader(record["validation_id"])
    st.write(f"Analysis: `{record['analysis_id']}`")
    st.write(f"Video: `{Path(record['video_path']).name}`")
    st.write(f"FPS: `{record.get('fps')}`")
    st.write(f"Prediction status: `{record.get('prediction_status')}`")
    st.write(f"Predicted frame: `{record.get('predicted_release_frame')}`")
    st.write(f"Method: `{record.get('evidence_mode')}`")
    st.write(f"Confidence: `{record.get('confidence')}`")
    flags = record.get("quality_flags") or []
    st.write("Quality flags:")
    st.code("\n".join(flags) if flags else "none")
    if record.get("baseline_collection_error"):
        st.warning(record["baseline_collection_error"])
    st.write(f"Current annotation: `{annotation.get('annotation_status')}`")
    if GUIDELINE_PATH.is_file():
        with st.expander("Guideline", expanded=False):
            st.markdown(GUIDELINE_PATH.read_text(encoding="utf-8"))


def _frame_viewer(record: dict[str, Any], annotation: dict[str, Any]) -> int:
    st.subheader("Frame Review")
    frame_count = int(record.get("frame_count") or 0)
    center = _default_center_frame(record, annotation)
    key = f"frame_{record['validation_id']}"
    if key not in st.session_state:
        st.session_state[key] = center

    nav = st.columns([1, 1, 1, 1, 3])
    with nav[0]:
        if st.button("-5 Frames", key=f"minus5_{record['validation_id']}"):
            st.session_state[key] = max(0, st.session_state[key] - 5)
    with nav[1]:
        if st.button("Previous Frame", key=f"prev_{record['validation_id']}"):
            st.session_state[key] = max(0, st.session_state[key] - 1)
    with nav[2]:
        if st.button("Next Frame", key=f"next_{record['validation_id']}"):
            st.session_state[key] = min(max(0, frame_count - 1), st.session_state[key] + 1)
    with nav[3]:
        if st.button("+5 Frames", key=f"plus5_{record['validation_id']}"):
            st.session_state[key] = min(max(0, frame_count - 1), st.session_state[key] + 5)

    max_frame = max(0, frame_count - 1)
    current_frame = st.slider(
        "Frame",
        min_value=0,
        max_value=max_frame,
        value=min(st.session_state[key], max_frame),
        key=f"slider_{record['validation_id']}",
    )
    st.session_state[key] = current_frame

    image = _read_frame(record["video_path"], current_frame)
    caption = f"Frame {current_frame}"
    predicted = record.get("predicted_release_frame")
    if predicted == current_frame:
        caption += " | model predicted release"
    if image is None:
        st.error("Could not read this frame from the clean original video.")
    else:
        st.image(image, caption=caption, use_container_width=True)
    return current_frame


def _annotation_form(
    record: dict[str, Any],
    annotation: dict[str, Any],
    annotations_doc: dict[str, Any],
    current_frame: int,
) -> None:
    st.subheader("Annotation")
    cols = st.columns([1, 1, 1])
    with cols[0]:
        if st.button("Select Current Frame as TRUE RELEASE", type="primary", use_container_width=True):
            annotation["annotation_status"] = "labeled"
            annotation["human_release_frame"] = current_frame
            annotation["cannot_determine"] = False
            _save_annotation(annotations_doc)
            st.success(f"Saved true release frame {current_frame}.")
    with cols[1]:
        if st.button("Mark Uncertain", use_container_width=True):
            annotation["annotation_status"] = "uncertain"
            annotation["cannot_determine"] = False
            _save_annotation(annotations_doc)
            st.warning("Saved as uncertain.")
    with cols[2]:
        if st.button("Mark Cannot Determine", use_container_width=True):
            annotation["annotation_status"] = "not_visible"
            annotation["cannot_determine"] = True
            annotation["human_release_frame"] = None
            _save_annotation(annotations_doc)
            st.warning("Saved as cannot determine.")

    with st.form(f"annotation_form_{record['validation_id']}"):
        human_frame = st.number_input(
            "Human release frame",
            min_value=0,
            max_value=max(0, int(record.get("frame_count") or 1) - 1),
            value=int(annotation.get("human_release_frame") or current_frame),
            step=1,
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            confidence = st.selectbox(
                "Annotation confidence",
                ["", "high", "medium", "low"],
                index=_option_index(["", "high", "medium", "low"], annotation.get("human_annotation_confidence")),
            )
        with c2:
            visibility = st.selectbox(
                "Release visibility",
                ["", "visible", "partially_visible", "occluded", "not_visible"],
                index=_option_index(
                    ["", "visible", "partially_visible", "occluded", "not_visible"],
                    annotation.get("release_visibility"),
                ),
            )
        with c3:
            uncertainty_start = st.number_input(
                "Uncertainty start",
                min_value=0,
                max_value=max(0, int(record.get("frame_count") or 1) - 1),
                value=int(annotation.get("human_uncertainty_start") or annotation.get("uncertainty_start_frame") or human_frame),
                step=1,
            )
        with c4:
            uncertainty_end = st.number_input(
                "Uncertainty end",
                min_value=0,
                max_value=max(0, int(record.get("frame_count") or 1) - 1),
                value=int(annotation.get("human_uncertainty_end") or annotation.get("uncertainty_end_frame") or human_frame),
                step=1,
            )
        categories = st.multiselect(
            "Failure categories",
            FAILURE_CATEGORIES,
            default=[item for item in annotation.get("failure_categories", []) if item in FAILURE_CATEGORIES],
        )
        notes = st.text_area("Notes", value=annotation.get("notes") or "")
        submitted = st.form_submit_button("Save Details")
        if submitted:
            if annotation.get("annotation_status") == "unlabeled":
                annotation["annotation_status"] = "labeled"
            if annotation["annotation_status"] == "labeled":
                annotation["human_release_frame"] = int(human_frame)
            annotation["human_annotation_confidence"] = confidence or None
            annotation["release_visibility"] = visibility or None
            annotation["human_uncertainty_start"] = int(uncertainty_start)
            annotation["human_uncertainty_end"] = int(uncertainty_end)
            annotation["uncertainty_start_frame"] = int(uncertainty_start)
            annotation["uncertainty_end_frame"] = int(uncertainty_end)
            annotation["failure_categories"] = categories
            annotation["notes"] = notes
            _save_annotation(annotations_doc)
            st.success("Annotation details saved.")


def _read_frame(video_path: str, frame_index: int) -> Any | None:
    from Backends.src.utils.cv2_loader import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = capture.read()
        if not ok:
            return None
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def _default_center_frame(record: dict[str, Any], annotation: dict[str, Any]) -> int:
    for value in (
        annotation.get("human_release_frame"),
        record.get("predicted_release_frame"),
        _first_primary_track_frame(record["analysis_id"]),
    ):
        if isinstance(value, int):
            return value
    frame_count = int(record.get("frame_count") or 1)
    return max(0, frame_count // 2)


def _first_primary_track_frame(analysis_id: str) -> int | None:
    path = PROJECT_ROOT / "outputs" / "video_analysis" / analysis_id / "tracking" / "tracking_result.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    track = data.get("primary_track") or []
    if not track:
        return None
    frame = track[0].get("frame_index")
    return int(frame) if isinstance(frame, int) else None


def _load_or_create_annotations(baseline: dict[str, Any]) -> dict[str, Any]:
    if ANNOTATIONS_PATH.is_file():
        return _read_json(ANNOTATIONS_PATH)
    document = {
        "schema_version": "1.0",
        "created_at": _utc_now(),
        "annotation_guideline": str(GUIDELINE_PATH),
        "annotations": [
            _annotation_template_from_record(record)
            for record in baseline.get("records", [])
        ],
    }
    _write_json(ANNOTATIONS_PATH, document)
    return document


def _annotation_template_from_record(record: dict[str, Any]) -> dict[str, Any]:
    uncertainty = record.get("frame_uncertainty") or {}
    return {
        "validation_id": record["validation_id"],
        "analysis_id": record["analysis_id"],
        "video_reference": record["video_path"],
        "fps": record.get("fps"),
        "predicted_release_frame": record.get("predicted_release_frame"),
        "predicted_release_time_seconds": record.get("predicted_release_time_seconds"),
        "prediction_confidence": record.get("confidence"),
        "prediction_method": record.get("evidence_mode"),
        "prediction_quality_flags": record.get("quality_flags", []),
        "model_uncertainty_start_frame": uncertainty.get("start"),
        "model_uncertainty_end_frame": uncertainty.get("end"),
        "annotation_status": "unlabeled",
        "human_release_frame": None,
        "human_uncertainty_start": None,
        "human_uncertainty_end": None,
        "human_annotation_confidence": None,
        "release_visibility": None,
        "uncertainty_start_frame": None,
        "uncertainty_end_frame": None,
        "cannot_determine": False,
        "true_release_point_px": None,
        "failure_categories": [],
        "notes": "",
        "annotated_at": None,
    }


def _sync_prediction_fields(annotation: dict[str, Any], record: dict[str, Any]) -> None:
    refreshed = _annotation_template_from_record(record)
    for key, value in refreshed.items():
        if key.startswith("prediction") or key in {
            "video_reference",
            "fps",
            "predicted_release_frame",
            "predicted_release_time_seconds",
            "model_uncertainty_start_frame",
            "model_uncertainty_end_frame",
        }:
            annotation[key] = value


def _save_annotation(document: dict[str, Any]) -> None:
    document["updated_at"] = _utc_now()
    for annotation in document.get("annotations", []):
        if annotation.get("annotation_status") in {"labeled", "uncertain", "not_visible"}:
            annotation["annotated_at"] = annotation.get("annotated_at") or _utc_now()
    _write_json(ANNOTATIONS_PATH, document)


def _first_unlabeled_index(
    records: list[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
) -> int:
    for index, record in enumerate(records):
        status = annotations.get(record["validation_id"], {}).get("annotation_status")
        if status not in {"labeled", "uncertain", "not_visible"}:
            return index
    return 0


def _option_index(options: list[str], value: Any) -> int:
    return options.index(value) if value in options else 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    main()
