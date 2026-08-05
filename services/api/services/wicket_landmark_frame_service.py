"""Native-frame selection and wicket ROI extraction for landmark evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Iterable, Literal, Sequence

import cv2
import numpy as np

from ..schemas.wicket_observation import (
    PixelBox,
    SetupFrameCandidate,
    WicketObservationResult,
)
from .video_analysis_service import ANALYSIS_ID_PATTERN, VIDEO_ANALYSIS_ROOT


WicketRole = Literal["near", "far"]
_VALID_ROTATIONS = (0, 90, 180, 270)


@dataclass(frozen=True)
class CropToNativeTransform:
    """Axis-aligned crop transform in the already-oriented native frame."""

    x: int
    y: int
    width: int
    height: int
    native_width: int
    native_height: int

    def crop_to_native(self, point: tuple[float, float]) -> tuple[float, float]:
        return self.x + float(point[0]), self.y + float(point[1])

    def native_to_crop(self, point: tuple[float, float]) -> tuple[float, float]:
        return float(point[0]) - self.x, float(point[1]) - self.y


@dataclass(frozen=True)
class NativeFrame:
    frame_index: int
    timestamp_seconds: float
    image: np.ndarray = field(repr=False, compare=False)
    candidate: SetupFrameCandidate
    quality_score: float
    quality_factors: dict[str, float]
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class WicketCrop:
    frame_index: int
    role: WicketRole
    image: np.ndarray = field(repr=False, compare=False)
    transform: CropToNativeTransform
    requested_box: tuple[int, int, int, int]
    clipping_fraction: float
    quality_score: float
    quality_factors: dict[str, float]
    accepted: bool
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class NativeFrameSelection:
    considered_frame_ids: tuple[int, ...]
    selected_frames: tuple[NativeFrame, ...]
    rejected_frame_ids: tuple[int, ...]
    native_width: int
    native_height: int
    rotation_applied_degrees: int


@dataclass(frozen=True)
class AlignedRoleFrames:
    role: WicketRole
    crops: tuple[WicketCrop, ...]
    aligned_stack: object

    @property
    def consensus_image(self) -> np.ndarray:
        return self.aligned_stack.temporal_median

    @property
    def native_origin(self) -> tuple[float, float]:
        transform = self.aligned_stack.reference_crop.transform
        return float(transform.x), float(transform.y)

    @property
    def frame_index(self) -> int:
        return int(self.aligned_stack.reference_frame_index)


@dataclass(frozen=True)
class WicketLandmarkFrameBundle:
    selection: NativeFrameSelection
    supporting_frames: tuple[dict[str, object], ...]
    near: AlignedRoleFrames | None
    far: AlignedRoleFrames | None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def apply_orientation_once(
    frame: np.ndarray,
    rotation_degrees: int,
    *,
    orientation_already_applied: bool = False,
) -> np.ndarray:
    """Apply explicit clockwise metadata rotation once.

    OpenCV commonly returns an already oriented frame. Callers must declare that
    state; a non-zero second rotation is rejected instead of silently rotating
    native coordinates twice.
    """
    if rotation_degrees not in _VALID_ROTATIONS:
        raise ValueError("rotation_degrees must be one of 0, 90, 180, 270")
    if orientation_already_applied:
        if rotation_degrees:
            raise ValueError("frame orientation has already been applied")
        return frame
    rotations = {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    code = rotations[rotation_degrees]
    return frame if code is None else cv2.rotate(frame, code)


def map_detector_box_to_native(
    bbox: PixelBox,
    *,
    detector_width: int,
    detector_height: int,
    native_width: int,
    native_height: int,
    rotation_degrees: int = 0,
) -> PixelBox:
    """Map detector pixels into the oriented native frame exactly once."""
    if min(detector_width, detector_height, native_width, native_height) <= 0:
        raise ValueError("image dimensions must be positive")
    if rotation_degrees not in _VALID_ROTATIONS:
        raise ValueError("rotation_degrees must be one of 0, 90, 180, 270")

    sx = native_width / detector_width
    sy = native_height / detector_height
    x1, y1 = bbox.x * sx, bbox.y * sy
    x2 = (bbox.x + bbox.width) * sx
    y2 = (bbox.y + bbox.height) * sy
    if rotation_degrees == 90:
        points = ((native_height - y2, x1), (native_height - y1, x2))
        output_width, output_height = native_height, native_width
    elif rotation_degrees == 180:
        points = (
            (native_width - x2, native_height - y2),
            (native_width - x1, native_height - y1),
        )
        output_width, output_height = native_width, native_height
    elif rotation_degrees == 270:
        points = ((y1, native_width - x2), (y2, native_width - x1))
        output_width, output_height = native_height, native_width
    else:
        points = ((x1, y1), (x2, y2))
        output_width, output_height = native_width, native_height
    left = _clamp(min(point[0] for point in points), 0, output_width)
    top = _clamp(min(point[1] for point in points), 0, output_height)
    right = _clamp(max(point[0] for point in points), 0, output_width)
    bottom = _clamp(max(point[1] for point in points), 0, output_height)
    if right <= left or bottom <= top:
        raise ValueError("detector box does not overlap the native frame")
    return PixelBox(x=left, y=top, width=right - left, height=bottom - top)


def _local_quality(frame: np.ndarray) -> dict[str, float]:
    if frame.size == 0:
        return {
            "sharpness": 0.0,
            "exposure": 0.0,
            "contrast": 0.0,
            "sharpness_variance": 0.0,
        }
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean = float(gray.mean())
    contrast = float(gray.std())
    return {
        "sharpness": _clamp(math.log1p(sharpness) / math.log1p(600.0)),
        "exposure": _clamp(1.0 - abs(mean - 128.0) / 114.0),
        "contrast": _clamp(contrast / 48.0),
        "sharpness_variance": sharpness,
    }


def score_native_frame(
    frame: np.ndarray,
    candidate: SetupFrameCandidate,
) -> tuple[float, dict[str, float], tuple[str, ...]]:
    """Score one persisted candidate deterministically without detector work."""
    local = _local_quality(frame)
    factors = {
        "detector_confidence": _clamp(candidate.mean_detector_confidence),
        "detection_stability": _clamp(candidate.detection_stability),
        "wicket_completeness": _clamp(candidate.wicket_detection_count / 2.0),
        "sharpness": local["sharpness"],
        "exposure": local["exposure"],
        "contrast": local["contrast"],
        "obstruction_clearance": 1.0 - _clamp(candidate.obstruction_score),
    }
    score = (
        0.22 * factors["detector_confidence"]
        + 0.18 * factors["detection_stability"]
        + 0.18 * factors["wicket_completeness"]
        + 0.18 * factors["sharpness"]
        + 0.08 * factors["exposure"]
        + 0.10 * factors["contrast"]
        + 0.06 * factors["obstruction_clearance"]
    )
    reasons = list(candidate.rejection_reasons)
    if local["sharpness_variance"] < 18.0:
        reasons.append("motion_blur_or_low_detail")
    if factors["exposure"] < 0.12:
        reasons.append("exposure_out_of_range")
    if candidate.obstruction_score > 0.72:
        reasons.append("occlusion_or_clutter_high")
    if candidate.wicket_detection_count <= 0:
        reasons.append("no_persisted_wicket_detection")
    return round(_clamp(score), 6), factors, tuple(sorted(set(reasons)))


def rank_native_frames(
    frames: Iterable[tuple[np.ndarray, SetupFrameCandidate]],
    *,
    maximum: int = 8,
) -> tuple[NativeFrame, ...]:
    """Rank decoded native frames by quality with a stable frame-ID tie break."""
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    ranked: list[NativeFrame] = []
    for image, candidate in frames:
        score, factors, reasons = score_native_frame(image, candidate)
        ranked.append(
            NativeFrame(
                frame_index=candidate.frame_index,
                timestamp_seconds=candidate.timestamp_seconds,
                image=image,
                candidate=candidate,
                quality_score=score,
                quality_factors=factors,
                rejection_reasons=reasons,
            )
        )
    usable = [
        item
        for item in ranked
        if "no_persisted_wicket_detection" not in item.rejection_reasons
        and "motion_blur_or_low_detail" not in item.rejection_reasons
        and "exposure_out_of_range" not in item.rejection_reasons
    ]
    return tuple(
        sorted(usable, key=lambda item: (-item.quality_score, item.frame_index))[
            :maximum
        ]
    )


def balance_role_support(
    ranked: Sequence[NativeFrame],
    *,
    near_support: Sequence[int] = (),
    far_support: Sequence[int] = (),
    maximum: int = 8,
    minimum_per_role: int = 3,
) -> tuple[NativeFrame, ...]:
    """Reserve useful support for each available role, then fill by quality."""
    if maximum <= 0 or minimum_per_role <= 0:
        raise ValueError("selection limits must be positive")
    ordered = sorted(ranked, key=lambda item: (-item.quality_score, item.frame_index))
    chosen: set[int] = set()
    for support in (set(near_support), set(far_support)):
        role_ranked = [item for item in ordered if item.frame_index in support]
        chosen.update(
            item.frame_index for item in role_ranked[: min(minimum_per_role, maximum)]
        )
    for item in ordered:
        if len(chosen) >= maximum:
            break
        chosen.add(item.frame_index)
    return tuple(item for item in ordered if item.frame_index in chosen)[:maximum]


def _candidate_ids(observation: WicketObservationResult) -> list[int]:
    supported = set()
    for wicket in (observation.near_wicket, observation.far_wicket):
        if wicket is not None:
            supported.update(wicket.region.supporting_frame_ids)
    supported.update(item.frame_index for item in observation.supporting_frames)
    if observation.setup_frame is not None:
        supported.add(observation.setup_frame.frame_index)
    return sorted(supported)


def select_native_wicket_frames(
    video_path: Path,
    observation: WicketObservationResult,
    *,
    maximum: int = 8,
    rotation_degrees: int = 0,
    orientation_already_applied: bool = True,
) -> NativeFrameSelection:
    """Decode and select bounded persisted evidence frames from the clean video."""
    candidate_by_id = {
        item.frame_index: item for item in observation.frame_candidates
    }
    ids = [item for item in _candidate_ids(observation) if item in candidate_by_id]
    ids = ids[: max(1, maximum * 3)]
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("could not open clean source video")
    decoded: list[tuple[np.ndarray, SetupFrameCandidate]] = []
    try:
        for frame_id in ids:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                continue
            frame = apply_orientation_once(
                frame,
                rotation_degrees,
                orientation_already_applied=orientation_already_applied,
            )
            candidate = candidate_by_id[frame_id]
            if (frame.shape[1], frame.shape[0]) != (
                candidate.image_width,
                candidate.image_height,
            ):
                continue
            decoded.append((frame, candidate))
    finally:
        capture.release()
    ranked = rank_native_frames(decoded, maximum=max(1, len(decoded)))
    near_support = (
        observation.near_wicket.region.supporting_frame_ids
        if observation.near_wicket is not None
        else ()
    )
    far_support = (
        observation.far_wicket.region.supporting_frame_ids
        if observation.far_wicket is not None
        else ()
    )
    selected = balance_role_support(
        ranked,
        near_support=near_support,
        far_support=far_support,
        maximum=maximum,
    )
    selected_ids = {item.frame_index for item in selected}
    width = int(decoded[0][0].shape[1]) if decoded else 0
    height = int(decoded[0][0].shape[0]) if decoded else 0
    return NativeFrameSelection(
        considered_frame_ids=tuple(ids),
        selected_frames=selected,
        rejected_frame_ids=tuple(item for item in ids if item not in selected_ids),
        native_width=width,
        native_height=height,
        rotation_applied_degrees=(0 if orientation_already_applied else rotation_degrees),
    )


def extract_wicket_roi(
    frame: NativeFrame,
    bbox: PixelBox,
    *,
    role: WicketRole,
    maximum_clipping_fraction: float = 0.22,
) -> WicketCrop:
    """Extract a role-padded native crop and retain its exact native transform."""
    if role not in ("near", "far"):
        raise ValueError("role must be 'near' or 'far'")
    height, width = frame.image.shape[:2]
    if role == "near":
        pad_left = pad_right = 0.22
        pad_top, pad_bottom = 0.24, 0.30
        minimum_width, minimum_height = 36, 54
        max_area_fraction = 0.24
    else:
        pad_left = pad_right = 0.16
        pad_top, pad_bottom = 0.20, 0.24
        minimum_width, minimum_height = 18, 28
        max_area_fraction = 0.10
    requested = (
        int(math.floor(bbox.x - bbox.width * pad_left)),
        int(math.floor(bbox.y - bbox.height * pad_top)),
        int(math.ceil(bbox.x + bbox.width * (1.0 + pad_right))),
        int(math.ceil(bbox.y + bbox.height * (1.0 + pad_bottom))),
    )
    x1, y1 = max(0, requested[0]), max(0, requested[1])
    x2, y2 = min(width, requested[2]), min(height, requested[3])
    requested_area = max(1, requested[2] - requested[0]) * max(
        1, requested[3] - requested[1]
    )
    retained_area = max(0, x2 - x1) * max(0, y2 - y1)
    clipping = 1.0 - retained_area / requested_area
    image = frame.image[y1:y2, x1:x2].copy()
    local = _local_quality(image)
    roi_resolution = _clamp(
        min((x2 - x1) / minimum_width, (y2 - y1) / minimum_height)
    )
    factors = {
        **local,
        "roi_resolution": roi_resolution,
        "crop_completeness": 1.0 - _clamp(clipping),
        "frame_quality": frame.quality_score,
    }
    quality = _clamp(
        0.30 * frame.quality_score
        + 0.22 * local["sharpness"]
        + 0.12 * local["exposure"]
        + 0.12 * local["contrast"]
        + 0.14 * roi_resolution
        + 0.10 * factors["crop_completeness"]
    )
    reasons: list[str] = []
    if x2 <= x1 or y2 <= y1:
        reasons.append("crop_outside_native_frame")
    if x2 - x1 < minimum_width or y2 - y1 < minimum_height:
        reasons.append("native_roi_resolution_insufficient")
    if clipping > maximum_clipping_fraction:
        reasons.append("severe_frame_bound_clipping")
    if bbox.width * bbox.height > width * height * max_area_fraction:
        reasons.append("implausibly_large_wicket_region")
    return WicketCrop(
        frame_index=frame.frame_index,
        role=role,
        image=image,
        transform=CropToNativeTransform(
            x=x1,
            y=y1,
            width=max(0, x2 - x1),
            height=max(0, y2 - y1),
            native_width=width,
            native_height=height,
        ),
        requested_box=requested,
        clipping_fraction=round(clipping, 6),
        quality_score=round(quality, 6),
        quality_factors=factors,
        accepted=not reasons,
        rejection_reasons=tuple(reasons),
    )


def write_analysis_debug_crops(
    analysis_id: str,
    crops: Sequence[WicketCrop],
) -> tuple[Path, ...]:
    """Optionally persist crops inside the validated analysis-owned debug tree."""
    if not ANALYSIS_ID_PATTERN.fullmatch(analysis_id):
        raise ValueError("invalid analysis ID")
    analysis_root = (VIDEO_ANALYSIS_ROOT / analysis_id).resolve()
    if not analysis_root.is_dir():
        raise ValueError("analysis directory does not exist")
    debug_dir = (analysis_root / "calibration" / "wicket_landmarks_v1").resolve()
    if analysis_root not in debug_dir.parents:
        raise ValueError("debug directory must remain analysis-owned")
    debug_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for crop in sorted(crops, key=lambda item: (item.role, item.frame_index)):
        if crop.image.size == 0:
            continue
        destination = debug_dir / f"{crop.role}_frame_{crop.frame_index:06d}.png"
        if not cv2.imwrite(str(destination), crop.image):
            raise OSError(f"could not write debug crop for frame {crop.frame_index}")
        written.append(destination)
    return tuple(written)


def prepare_wicket_landmark_frames(
    analysis_id: str,
    observation: WicketObservationResult,
    *,
    maximum: int = 8,
    write_debug: bool = False,
) -> WicketLandmarkFrameBundle:
    """Build in-memory role crops and bounded aligned stacks from persisted ROIs."""
    from .wicket_roi_alignment import align_wicket_crops

    if observation.analysis_id != analysis_id:
        raise ValueError("observation does not belong to analysis")
    from .video_analysis_service import load_video_analysis

    metadata = load_video_analysis(analysis_id)
    video_path = VIDEO_ANALYSIS_ROOT / analysis_id / "raw" / metadata.stored_filename
    selection = select_native_wicket_frames(
        video_path,
        observation,
        maximum=maximum,
    )
    role_outputs: dict[WicketRole, AlignedRoleFrames | None] = {
        "near": None,
        "far": None,
    }
    all_crops: list[WicketCrop] = []
    for role, wicket in (
        ("near", observation.near_wicket),
        ("far", observation.far_wicket),
    ):
        if wicket is None:
            continue
        support = set(wicket.region.supporting_frame_ids)
        crops = tuple(
            extract_wicket_roi(frame, wicket.region.bbox, role=role)
            for frame in selection.selected_frames
            if frame.frame_index in support
        )
        all_crops.extend(crops)
        accepted = [item for item in crops if item.accepted]
        if accepted:
            role_outputs[role] = AlignedRoleFrames(
                role=role,
                crops=crops,
                aligned_stack=align_wicket_crops(accepted),
            )
    if write_debug:
        write_analysis_debug_crops(analysis_id, all_crops)
    supporting = tuple(
        {
            "frame_index": item.frame_index,
            "timestamp_seconds": item.timestamp_seconds,
            "quality_score": item.quality_score,
            "selected": True,
            "rejection_reasons": list(item.rejection_reasons),
        }
        for item in selection.selected_frames
    )
    return WicketLandmarkFrameBundle(
        selection=selection,
        supporting_frames=supporting,
        near=role_outputs["near"],
        far=role_outputs["far"],
    )
