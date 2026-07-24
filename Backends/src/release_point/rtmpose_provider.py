"""Real RTMPose provider adapter for Release Point V1.

The OpenMMLab stack is optional and intentionally imported lazily. The main
API must keep starting when the pose runtime is absent or broken.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

from .pose_provider import Keypoint, PosePerson, PoseProvider, PoseProviderInfo


COCO_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
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
CORE_INDEX_BY_NAME = {
    name: index for index, name in enumerate(COCO_KEYPOINT_NAMES)
}


class RTMPoseProviderUnavailable(RuntimeError):
    """Raised when the optional real pose runtime cannot be used."""


@dataclass(frozen=True)
class RTMPoseProviderConfig:
    pose2d: str = "human"
    model_name: str = "rtmpose-m_8xb256-420e_body8-256x192"
    keypoint_schema: str = "coco17"
    device: str | None = None
    det_model: str | None = None
    det_cat_ids: tuple[int, ...] = (0,)


class RTMPoseProvider(PoseProvider):
    """MMPose-backed provider normalized to CricVision's core 2D pose schema."""

    provider_name = "rtmpose_mmpose"

    def __init__(self, config: RTMPoseProviderConfig | None = None) -> None:
        self.config = config or RTMPoseProviderConfig()
        self.model_name = self.config.model_name
        self.keypoint_schema = self.config.keypoint_schema
        self._inferencer: Any | None = None

    @classmethod
    def is_available(cls) -> bool:
        return all(
            find_spec(package) is not None
            for package in ("torch", "mmengine", "mmcv", "mmpose", "mmdet")
        )

    def estimate_frame(
        self,
        frame_bgr: Any,
        frame_index: int,
        timestamp_seconds: float,
    ) -> list[PosePerson]:
        if frame_bgr is None:
            raise ValueError("RTMPoseProvider requires a decoded clean frame.")
        inferencer = self._load_inferencer()
        try:
            result = next(
                inferencer(
                    frame_bgr,
                    return_vis=False,
                    show=False,
                )
            )
        except Exception as exc:
            raise RTMPoseProviderUnavailable(
                f"RTMPose inference failed: {type(exc).__name__}."
            ) from exc
        return self._parse_result(result, frame_index, timestamp_seconds)

    def _load_inferencer(self) -> Any:
        if self._inferencer is not None:
            return self._inferencer
        if not self.is_available():
            raise RTMPoseProviderUnavailable(
                "MMPose runtime is unavailable; install torch, mmengine, mmcv, mmpose, and mmdet in the isolated pose environment."
            )
        try:
            from mmpose.apis import MMPoseInferencer
        except Exception as exc:
            raise RTMPoseProviderUnavailable(
                f"MMPose could not be imported: {type(exc).__name__}."
            ) from exc

        kwargs: dict[str, Any] = {
            "pose2d": self.config.pose2d,
            "device": self.config.device,
        }
        if self.config.det_model:
            kwargs["det_model"] = self.config.det_model
            kwargs["det_cat_ids"] = list(self.config.det_cat_ids)
        try:
            self._inferencer = MMPoseInferencer(**kwargs)
        except Exception as exc:
            raise RTMPoseProviderUnavailable(
                f"MMPose inferencer could not be initialised: {type(exc).__name__}."
            ) from exc
        return self._inferencer

    def _parse_result(
        self,
        result: dict[str, Any],
        frame_index: int,
        timestamp_seconds: float,
    ) -> list[PosePerson]:
        raw_instances = result.get("predictions", [])
        if raw_instances and isinstance(raw_instances[0], list):
            raw_instances = raw_instances[0]
        people: list[PosePerson] = []
        for index, instance in enumerate(raw_instances or []):
            person = self._parse_instance(
                instance,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                person_index=index,
            )
            if person is not None:
                people.append(person)
        return people

    def _parse_instance(
        self,
        instance: dict[str, Any],
        *,
        frame_index: int,
        timestamp_seconds: float,
        person_index: int,
    ) -> PosePerson | None:
        keypoints = _as_list(instance.get("keypoints"))
        scores = _as_list(instance.get("keypoint_scores") or instance.get("keypoints_scores"))
        if not keypoints:
            return None

        normalized: dict[str, Keypoint] = {}
        for name, index in CORE_INDEX_BY_NAME.items():
            if index >= len(keypoints):
                continue
            point = keypoints[index]
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            confidence = _score_at(scores, index)
            normalized[name] = Keypoint(
                x=float(point[0]),
                y=float(point[1]),
                confidence=confidence,
            )

        if not normalized:
            return None
        bbox = _bbox(instance, normalized)
        confidence = _person_confidence(instance, normalized)
        return PosePerson(
            person_id=f"rtmpose_person_{frame_index}_{person_index}",
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            bbox_xyxy=bbox,
            confidence=confidence,
            keypoints=normalized,
            provider=PoseProviderInfo(
                name=self.provider_name,
                model=self.model_name,
                schema=self.keypoint_schema,
            ),
        )


def infer_bowling_arm(
    poses_by_frame: dict[int, PosePerson],
) -> dict[str, Any]:
    """Small temporal wrist-activity heuristic; reports uncertainty honestly."""

    scores: dict[str, float] = {"left": 0.0, "right": 0.0}
    frames = sorted(poses_by_frame)
    for side in ("left", "right"):
        key = f"{side}_wrist"
        previous: Keypoint | None = None
        for frame in frames:
            wrist = poses_by_frame[frame].keypoint(key)
            if wrist is None or wrist.confidence < 0.25:
                continue
            if previous is not None:
                dx = wrist.x - previous.x
                dy = wrist.y - previous.y
                scores[side] += (dx * dx + dy * dy) ** 0.5 * min(1.0, wrist.confidence)
            previous = wrist
    best_side = max(scores, key=scores.get)
    other_side = "left" if best_side == "right" else "right"
    margin = scores[best_side] - scores[other_side]
    total = max(1.0, scores[best_side] + scores[other_side])
    confidence = max(0.0, min(1.0, margin / total))
    if confidence < 0.2:
        return {
            "bowling_arm": "unknown",
            "confidence": round(confidence, 3),
            "scores": {key: round(value, 3) for key, value in scores.items()},
            "quality_flags": ["bowling_arm_ambiguous"],
        }
    return {
        "bowling_arm": best_side,
        "confidence": round(confidence, 3),
        "scores": {key: round(value, 3) for key, value in scores.items()},
        "quality_flags": [],
    }


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, (int, float)):
        return [value]
    return list(value)


def _score_at(scores: list[Any], index: int) -> float:
    if index >= len(scores):
        return 0.0
    value = scores[index]
    if isinstance(value, (list, tuple)):
        value = value[0] if value else 0.0
    return max(0.0, min(1.0, float(value)))


def _bbox(
    instance: dict[str, Any],
    keypoints: dict[str, Keypoint],
) -> tuple[float, float, float, float]:
    raw_bboxes = _as_list(instance.get("bboxes") or instance.get("bbox"))
    if raw_bboxes:
        box = raw_bboxes[0] if raw_bboxes and isinstance(raw_bboxes[0], list) else raw_bboxes
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            return tuple(float(value) for value in box[:4])  # type: ignore[return-value]
    xs = [point.x for point in keypoints.values()]
    ys = [point.y for point in keypoints.values()]
    pad = 24.0
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def _person_confidence(
    instance: dict[str, Any],
    keypoints: dict[str, Keypoint],
) -> float:
    raw_scores = _as_list(instance.get("bbox_scores") or instance.get("bbox_score"))
    if raw_scores:
        return max(0.0, min(1.0, float(raw_scores[0])))
    values = [point.confidence for point in keypoints.values()]
    return sum(values) / len(values) if values else 0.0
