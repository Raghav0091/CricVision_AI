"""Pose provider contract for Release Point V1.

The provider layer is intentionally small: heavy pose frameworks must normalize
their outputs into these dataclasses before release logic consumes them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable


CORE_KEYPOINTS: tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True)
class Keypoint:
    """One normalized 2D pose keypoint in source-frame pixel coordinates."""

    x: float
    y: float
    confidence: float

    def to_dict(self) -> dict[str, float]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "confidence": float(self.confidence),
        }


@dataclass(frozen=True)
class PoseProviderInfo:
    """Provider provenance preserved on every normalized pose person."""

    name: str
    model: str
    schema: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "model": self.model, "schema": self.schema}


@dataclass(frozen=True)
class PosePerson:
    """One detected person pose in one frame."""

    person_id: str
    frame_index: int
    timestamp_seconds: float
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    keypoints: dict[str, Keypoint]
    provider: PoseProviderInfo

    def keypoint(self, name: str) -> Keypoint | None:
        return self.keypoints.get(name)

    def bbox_center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def foot_point(self) -> tuple[float, float]:
        left = self.keypoint("left_ankle")
        right = self.keypoint("right_ankle")
        ankles = [kp for kp in (left, right) if kp is not None and kp.confidence > 0.05]
        if ankles:
            return (
                sum(kp.x for kp in ankles) / len(ankles),
                sum(kp.y for kp in ankles) / len(ankles),
            )
        x1, _y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, y2)

    def with_person_id(self, person_id: str) -> "PosePerson":
        return replace(self, person_id=person_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "frame_index": int(self.frame_index),
            "timestamp_seconds": float(self.timestamp_seconds),
            "bbox_xyxy": [float(value) for value in self.bbox_xyxy],
            "confidence": float(self.confidence),
            "keypoints": {
                name: keypoint.to_dict() for name, keypoint in self.keypoints.items()
            },
            "provider": self.provider.to_dict(),
        }


@dataclass(frozen=True)
class PoseSequence:
    """Normalized pose candidates grouped by absolute frame index."""

    frames: dict[int, list[PosePerson]] = field(default_factory=dict)
    provider: PoseProviderInfo | None = None

    def persons_at(self, frame_index: int) -> list[PosePerson]:
        return list(self.frames.get(frame_index, []))

    def frame_indices(self) -> list[int]:
        return sorted(self.frames)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider.to_dict() if self.provider else None,
            "frames": {
                str(frame_index): [person.to_dict() for person in people]
                for frame_index, people in sorted(self.frames.items())
            },
        }


class PoseProvider(ABC):
    """Abstract pose provider normalized for cricket release analysis."""

    provider_name: str
    model_name: str
    keypoint_schema: str

    @property
    def provider_info(self) -> PoseProviderInfo:
        return PoseProviderInfo(
            name=self.provider_name,
            model=self.model_name,
            schema=self.keypoint_schema,
        )

    @abstractmethod
    def estimate_frame(
        self,
        frame_bgr: Any,
        frame_index: int,
        timestamp_seconds: float,
    ) -> list[PosePerson]:
        """Estimate all people in one clean original BGR frame."""

    def estimate_sequence(
        self,
        video_path: str | Path,
        frame_window: Iterable[int],
        fps: float,
    ) -> PoseSequence:
        """Estimate poses from clean original video frames.

        Release Point V1 CV input must be the raw uploaded video under:
        outputs/video_analysis/<analysis_id>/raw/<stored_filename>
        """

        _validate_clean_original_video_path(video_path)
        if fps <= 0:
            raise ValueError("fps must be positive.")

        from Backends.src.utils.cv2_loader import cv2

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video for pose estimation: {video_path}")

        frames: dict[int, list[PosePerson]] = {}
        try:
            for frame_index in sorted(set(int(index) for index in frame_window)):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame_bgr = capture.read()
                if not ok:
                    continue
                timestamp_seconds = frame_index / fps
                frames[frame_index] = self.estimate_frame(
                    frame_bgr,
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                )
        finally:
            capture.release()

        return PoseSequence(frames=frames, provider=self.provider_info)


class FakePoseProvider(PoseProvider):
    """Deterministic test provider with realistic cricket pose-shaped output."""

    provider_name = "fake_pose"
    model_name = "deterministic_cricket_pose_v1"
    keypoint_schema = "cricvision_core_2d"

    def __init__(
        self,
        persons_by_frame: dict[int, list[PosePerson]] | None = None,
        *,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self._persons_by_frame = persons_by_frame
        self.width = width
        self.height = height

    def estimate_frame(
        self,
        frame_bgr: Any,
        frame_index: int,
        timestamp_seconds: float,
    ) -> list[PosePerson]:
        if self._persons_by_frame is not None:
            return [
                _normalize_provider(person, self.provider_info)
                for person in self._persons_by_frame.get(frame_index, [])
            ]
        return [self._default_bowler_pose(frame_index, timestamp_seconds)]

    def estimate_sequence(
        self,
        video_path: str | Path,
        frame_window: Iterable[int],
        fps: float,
    ) -> PoseSequence:
        if fps <= 0:
            raise ValueError("fps must be positive.")

        frames: dict[int, list[PosePerson]] = {}
        for frame_index in sorted(set(int(index) for index in frame_window)):
            frames[frame_index] = self.estimate_frame(
                frame_bgr=None,
                frame_index=frame_index,
                timestamp_seconds=frame_index / fps,
            )
        return PoseSequence(frames=frames, provider=self.provider_info)

    def _default_bowler_pose(
        self, frame_index: int, timestamp_seconds: float
    ) -> PosePerson:
        stride = frame_index * 4.0
        cx = self.width * 0.28 + stride
        hip_y = self.height * 0.58
        shoulder_y = self.height * 0.39
        wrist_lift = max(0.0, 70.0 - abs(frame_index - 8) * 8.0)
        keypoints = {
            "left_shoulder": Keypoint(cx - 32, shoulder_y, 0.82),
            "right_shoulder": Keypoint(cx + 34, shoulder_y - 4, 0.88),
            "left_elbow": Keypoint(cx - 48, shoulder_y + 54, 0.78),
            "right_elbow": Keypoint(cx + 62, shoulder_y - 24 - wrist_lift * 0.35, 0.84),
            "left_wrist": Keypoint(cx - 52, shoulder_y + 110, 0.74),
            "right_wrist": Keypoint(cx + 48, shoulder_y - 74 - wrist_lift, 0.81),
            "left_hip": Keypoint(cx - 24, hip_y, 0.83),
            "right_hip": Keypoint(cx + 28, hip_y, 0.84),
            "left_knee": Keypoint(cx - 42, hip_y + 82, 0.79),
            "right_knee": Keypoint(cx + 46, hip_y + 78, 0.80),
            "left_ankle": Keypoint(cx - 54, hip_y + 172, 0.76),
            "right_ankle": Keypoint(cx + 58, hip_y + 164, 0.77),
        }
        bbox = (cx - 92, shoulder_y - 130, cx + 104, hip_y + 196)
        return PosePerson(
            person_id="fake_bowler",
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            bbox_xyxy=tuple(float(value) for value in bbox),
            confidence=0.91,
            keypoints=keypoints,
            provider=self.provider_info,
        )


def _normalize_provider(person: PosePerson, provider: PoseProviderInfo) -> PosePerson:
    if person.provider == provider:
        return person
    return PosePerson(
        person_id=person.person_id,
        frame_index=person.frame_index,
        timestamp_seconds=person.timestamp_seconds,
        bbox_xyxy=person.bbox_xyxy,
        confidence=person.confidence,
        keypoints=person.keypoints,
        provider=provider,
    )


def _validate_clean_original_video_path(video_path: str | Path) -> None:
    path = Path(video_path)
    lowered_parts = [part.lower() for part in path.parts]
    lowered_name = path.name.lower()
    overlay_tokens = ("overlay", "debug", "replay", "annotated", "tracking")

    if any(token in lowered_name for token in overlay_tokens):
        raise ValueError("Pose estimation must read the clean original video, not overlays.")

    if "video_analysis" in lowered_parts and "raw" not in lowered_parts:
        raise ValueError(
            "Pose estimation for uploaded analysis must read outputs/video_analysis/"
            "<analysis_id>/raw/<stored_filename>."
        )
