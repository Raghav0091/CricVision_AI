"""Bowler identity selection and temporal continuity for Release Point V1."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Any

from Backends.src.release_point.pose_provider import CORE_KEYPOINTS, PosePerson, PoseSequence


LOW_WRIST_CONFIDENCE = 0.35


@dataclass(frozen=True)
class BowlerPoseSequence:
    """Selected bowler poses keyed by absolute frame index.

    ``bowler_id`` is ``"unknown"`` when evidence is too weak to select a person.
    """

    bowler_id: str
    selection_confidence: float
    poses_by_frame: dict[int, PosePerson] = field(default_factory=dict)
    quality_flags: list[str] = field(default_factory=list)

    @property
    def frames(self) -> list[int]:
        return sorted(self.poses_by_frame)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bowler_id": self.bowler_id,
            "selection_confidence": float(self.selection_confidence),
            "frames": self.frames,
            "poses_by_frame": {
                str(frame_index): self.poses_by_frame[frame_index].to_dict()
                for frame_index in self.frames
            },
            "quality_flags": list(self.quality_flags),
        }


@dataclass
class _CandidateTrack:
    track_id: str
    poses: dict[int, PosePerson] = field(default_factory=dict)
    predicted_center: tuple[float, float] | None = None

    @property
    def last_frame(self) -> int:
        return max(self.poses)

    @property
    def last_pose(self) -> PosePerson:
        return self.poses[self.last_frame]

    def add(self, pose: PosePerson, alpha: float) -> None:
        center = pose.bbox_center()
        if self.predicted_center is None:
            self.predicted_center = center
        else:
            px, py = self.predicted_center
            cx, cy = center
            self.predicted_center = (
                px * (1.0 - alpha) + cx * alpha,
                py * (1.0 - alpha) + cy * alpha,
            )
        self.poses[pose.frame_index] = pose


class BowlerTracker:
    """Select the likely bowler without rewriting pose, ball, or API systems."""

    def __init__(
        self,
        *,
        max_gap_frames: int = 3,
        continuity_alpha: float = 0.45,
        min_selection_confidence: float = 0.45,
    ) -> None:
        self.max_gap_frames = max_gap_frames
        self.continuity_alpha = continuity_alpha
        self.min_selection_confidence = min_selection_confidence

    def track(
        self,
        pose_sequence: PoseSequence,
        *,
        scene_calibration: dict[str, Any] | None = None,
        pitch_context: dict[str, Any] | None = None,
        ball_track: list[dict[str, Any]] | None = None,
    ) -> BowlerPoseSequence:
        flags: list[str] = []
        tracks = self._build_tracks(pose_sequence)
        if not tracks:
            return BowlerPoseSequence(
                bowler_id="unknown",
                selection_confidence=0.0,
                poses_by_frame={},
                quality_flags=["no_pose_candidates"],
            )

        roi = _resolve_bowling_end_roi(scene_calibration, pitch_context)
        flags.extend(roi.flags)
        scored = [
            (self._score_track(track, roi, ball_track), track)
            for track in tracks
            if track.poses
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        best_score, best_track = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        confidence = _clamp(best_score * 0.75 + max(0.0, best_score - second_score) * 0.25)

        if confidence < self.min_selection_confidence:
            return BowlerPoseSequence(
                bowler_id="unknown",
                selection_confidence=round(confidence, 3),
                poses_by_frame={},
                quality_flags=_unique(flags + ["insufficient_bowler_evidence"]),
            )

        selected_poses = {
            frame_index: pose.with_person_id("bowler_01")
            for frame_index, pose in sorted(best_track.poses.items())
        }
        quality_flags = self._quality_flags(pose_sequence, selected_poses, flags, confidence)
        return BowlerPoseSequence(
            bowler_id="bowler_01",
            selection_confidence=round(confidence, 3),
            poses_by_frame=selected_poses,
            quality_flags=quality_flags,
        )

    def _build_tracks(self, pose_sequence: PoseSequence) -> list[_CandidateTrack]:
        tracks: list[_CandidateTrack] = []
        next_id = 1
        for frame_index in pose_sequence.frame_indices():
            people = sorted(
                pose_sequence.persons_at(frame_index),
                key=lambda person: person.confidence,
                reverse=True,
            )
            used: set[int] = set()
            for pose in people:
                best_track_index: int | None = None
                best_distance = float("inf")
                for index, track in enumerate(tracks):
                    if index in used:
                        continue
                    gap = pose.frame_index - track.last_frame
                    if gap <= 0 or gap > self.max_gap_frames + 1:
                        continue
                    distance = _distance(pose.bbox_center(), track.predicted_center or track.last_pose.bbox_center())
                    if distance < best_distance:
                        best_distance = distance
                        best_track_index = index

                max_distance = _association_distance(pose)
                if best_track_index is not None and best_distance <= max_distance:
                    track = tracks[best_track_index]
                    track.add(pose, self.continuity_alpha)
                    used.add(best_track_index)
                    continue

                track = _CandidateTrack(track_id=f"person_track_{next_id:02d}")
                track.add(pose, self.continuity_alpha)
                tracks.append(track)
                used.add(len(tracks) - 1)
                next_id += 1
        return tracks

    def _score_track(
        self,
        track: _CandidateTrack,
        roi: "_BowlingEndROI",
        ball_track: list[dict[str, Any]] | None,
    ) -> float:
        poses = list(track.poses.values())
        if not poses:
            return 0.0

        frame_presence = min(1.0, len(poses) / 8.0)
        person_confidence = sum(person.confidence for person in poses) / len(poses)
        endpoint_score = sum(roi.endpoint_score(person.foot_point()) for person in poses) / len(poses)
        striker_penalty = sum(roi.striker_penalty(person.foot_point()) for person in poses) / len(poses)
        corridor_score = sum(roi.corridor_score(person.foot_point()) for person in poses) / len(poses)
        motion_score = _motion_score(poses)
        ball_score = _ball_start_proximity_score(poses, ball_track)

        score = (
            endpoint_score * 0.42
            + corridor_score * 0.16
            + person_confidence * 0.14
            + frame_presence * 0.10
            + motion_score * 0.10
            + ball_score * 0.08
        )
        return _clamp(score - striker_penalty * 0.35)

    def _quality_flags(
        self,
        pose_sequence: PoseSequence,
        selected_poses: dict[int, PosePerson],
        existing_flags: list[str],
        confidence: float,
    ) -> list[str]:
        flags = list(existing_flags)
        if confidence < 0.65:
            flags.append("bowler_selection_uncertain")

        all_frames = pose_sequence.frame_indices()
        selected_frames = sorted(selected_poses)
        if all_frames and selected_frames:
            missing_inside_window = [
                frame
                for frame in range(selected_frames[0], selected_frames[-1] + 1)
                if frame not in selected_poses
            ]
            if missing_inside_window:
                flags.append("missing_pose_frames")

        for pose in selected_poses.values():
            missing_core = [name for name in CORE_KEYPOINTS if name not in pose.keypoints]
            if missing_core:
                flags.append("missing_core_keypoints")
                break

        for pose in selected_poses.values():
            wrists = [
                kp
                for name, kp in pose.keypoints.items()
                if name in {"left_wrist", "right_wrist"}
            ]
            if wrists and max(kp.confidence for kp in wrists) < LOW_WRIST_CONFIDENCE:
                flags.append("low_confidence_wrist")
                break
        return _unique(flags)


@dataclass(frozen=True)
class _BowlingEndROI:
    bowler_point: tuple[float, float] | None
    striker_point: tuple[float, float] | None
    corridor: tuple[tuple[float, float], ...] | None
    reference_distance: float
    flags: list[str]

    def endpoint_score(self, point: tuple[float, float]) -> float:
        if self.bowler_point is None:
            return 0.45
        distance = _distance(point, self.bowler_point)
        return _clamp(1.0 - distance / self.reference_distance)

    def striker_penalty(self, point: tuple[float, float]) -> float:
        if self.striker_point is None:
            return 0.0
        distance = _distance(point, self.striker_point)
        return _clamp(1.0 - distance / self.reference_distance)

    def corridor_score(self, point: tuple[float, float]) -> float:
        if self.corridor is None:
            return 0.45
        if _point_in_polygon(point, self.corridor):
            return 1.0
        distances = [_distance(point, vertex) for vertex in self.corridor]
        return _clamp(1.0 - min(distances) / self.reference_distance)


def _resolve_bowling_end_roi(
    scene_calibration: dict[str, Any] | None,
    pitch_context: dict[str, Any] | None,
) -> _BowlingEndROI:
    flags: list[str] = []
    if not scene_calibration:
        return _BowlingEndROI(
            bowler_point=None,
            striker_point=None,
            corridor=None,
            reference_distance=300.0,
            flags=["calibration_missing", "bowling_end_assignment_uncertain"],
        )

    image_size = _image_size(scene_calibration) or _image_size(pitch_context or {})
    if image_size is None:
        flags.append("calibration_pixel_scale_unknown")

    bowler_point = _point_from_calibration(
        scene_calibration, ("bowler_wicket", "bowler"), image_size
    )
    striker_point = _point_from_calibration(
        scene_calibration, ("striker_wicket", "striker"), image_size
    )

    if bowler_point is None:
        # Architecture V1: legacy non-striker is the initial bowling-end proxy.
        bowler_point = _point_from_calibration(
            scene_calibration, ("non_striker_wicket", "non_striker"), image_size
        )
        if bowler_point is not None:
            flags.append("bowling_end_assignment_uncertain")

    if striker_point is None:
        striker_point = _point_from_calibration(scene_calibration, ("striker",), image_size)

    corridor = _corridor_from_calibration(scene_calibration, image_size)
    reference_distance = 300.0
    if bowler_point is not None and striker_point is not None:
        reference_distance = max(120.0, _distance(bowler_point, striker_point) * 0.45)
    elif corridor:
        xs = [point[0] for point in corridor]
        ys = [point[1] for point in corridor]
        reference_distance = max(max(xs) - min(xs), max(ys) - min(ys), 120.0) * 0.35
    else:
        flags.extend(["calibration_missing", "bowling_end_assignment_uncertain"])

    return _BowlingEndROI(
        bowler_point=bowler_point,
        striker_point=striker_point,
        corridor=corridor,
        reference_distance=reference_distance,
        flags=_unique(flags),
    )


def _point_from_calibration(
    calibration: dict[str, Any],
    keys: tuple[str, ...],
    image_size: tuple[float, float] | None,
) -> tuple[float, float] | None:
    for key in keys:
        value = calibration.get(key)
        point = _extract_point(value, image_size)
        if point is not None:
            return point

    landmarks = calibration.get("landmarks")
    if isinstance(landmarks, list):
        for landmark in landmarks:
            if not isinstance(landmark, dict):
                continue
            landmark_id = str(landmark.get("id") or landmark.get("landmark_id") or "")
            wicket_end = str(landmark.get("wicket_end") or "")
            if any(
                _matches_calibration_key(key, landmark_id, wicket_end) for key in keys
            ):
                point = _extract_point(landmark, image_size)
                if point is not None:
                    return point
    return None


def _matches_calibration_key(key: str, landmark_id: str, wicket_end: str) -> bool:
    normalized = key.replace("_wicket", "")
    return (
        wicket_end == normalized
        or landmark_id == normalized
        or landmark_id.startswith(f"{normalized}_")
    )


def _extract_point(
    value: Any,
    image_size: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _scale_if_normalized((float(value[0]), float(value[1])), image_size)
    if not isinstance(value, dict):
        return None

    for key in ("bottom_center", "center", "approximate_wicket_base_reference"):
        point = _extract_point(value.get(key), image_size)
        if point is not None:
            return point

    if "image_x" in value and "image_y" in value:
        return (float(value["image_x"]), float(value["image_y"]))
    if "x" in value and "y" in value:
        return _scale_if_normalized((float(value["x"]), float(value["y"])), image_size)

    box = value.get("box")
    if isinstance(box, dict) and {"x", "y", "width", "height"} <= set(box):
        return _scale_if_normalized((
            float(box["x"]) + float(box["width"]) / 2.0,
            float(box["y"]) + float(box["height"]),
        ), image_size)
    return None


def _corridor_from_calibration(
    calibration: dict[str, Any],
    image_size: tuple[float, float] | None,
) -> tuple[tuple[float, float], ...] | None:
    pitch_geometry = calibration.get("pitch_geometry")
    candidates = [
        calibration.get("pitch_corridor"),
        calibration.get("corridor"),
        pitch_geometry.get("corridor") if isinstance(pitch_geometry, dict) else None,
    ]
    for candidate in candidates:
        if not isinstance(candidate, list) or len(candidate) < 3:
            continue
        points = tuple(
            point for raw in candidate if (point := _extract_point(raw, image_size)) is not None
        )
        if len(points) >= 3:
            return points
    return None


def _image_size(data: dict[str, Any]) -> tuple[float, float] | None:
    width = data.get("image_width") or data.get("width")
    height = data.get("image_height") or data.get("height")
    if width and height:
        return (float(width), float(height))
    return None


def _scale_if_normalized(
    point: tuple[float, float],
    image_size: tuple[float, float] | None,
) -> tuple[float, float] | None:
    x, y = point
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        if image_size is None:
            return None
        width, height = image_size
        return (x * width, y * height)
    return point


def _association_distance(pose: PosePerson) -> float:
    x1, y1, x2, y2 = pose.bbox_xyxy
    diagonal = hypot(x2 - x1, y2 - y1)
    return max(60.0, diagonal * 0.65)


def _motion_score(poses: list[PosePerson]) -> float:
    if len(poses) < 2:
        return 0.0
    centers = [pose.bbox_center() for pose in sorted(poses, key=lambda item: item.frame_index)]
    travel = sum(_distance(a, b) for a, b in zip(centers, centers[1:]))
    return _clamp(travel / 120.0)


def _ball_start_proximity_score(
    poses: list[PosePerson],
    ball_track: list[dict[str, Any]] | None,
) -> float:
    if not ball_track:
        return 0.0
    ball_points = []
    for point in ball_track[:5]:
        if not isinstance(point, dict):
            continue
        center = _extract_point(point.get("center") or point.get("center_px") or point)
        if center is not None:
            ball_points.append(center)
    if not ball_points:
        return 0.0

    pose_points = [pose.foot_point() for pose in poses[:5]]
    min_distance = min(_distance(pose, ball) for pose in pose_points for ball in ball_points)
    return _clamp(1.0 - min_distance / 220.0)


def _point_in_polygon(
    point: tuple[float, float], polygon: tuple[tuple[float, float], ...]
) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, vertex in enumerate(polygon):
        xi, yi = vertex
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and x < (
            (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _unique(flags: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for flag in flags:
        if flag not in seen:
            result.append(flag)
            seen.add(flag)
    return result
