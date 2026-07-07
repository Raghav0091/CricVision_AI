"""Trajectory-aware ball candidate selection for multi-detection frames."""

from __future__ import annotations

from collections import Counter
from math import hypot

# ponytail: fixed pixel thresholds are enough for MVP delivery clips.
STATIC_CELL_PX = 10
STATIC_MIN_HITS = 3
MIN_ACCEPTED_POINTS_FOR_RELIABILITY = 5
MAX_CONSECUTIVE_REJECT_FRAMES = 4


def _distance(point_a, point_b) -> float:
    if point_a is None or point_b is None:
        return float("inf")
    ax, ay = point_a
    bx, by = point_b
    return hypot(ax - bx, ay - by)


def _grid_key(center) -> tuple[int, int]:
    x, y = center
    return (int(round(x / STATIC_CELL_PX)), int(round(y / STATIC_CELL_PX)))


class TrajectoryBallSelector:
    """Pick one ball detection per frame using path continuity, not confidence alone."""

    def __init__(self, frame_width=1280, frame_height=720):
        self.frame_width = max(int(frame_width or 0), 1)
        self.frame_height = max(int(frame_height or 0), 1)
        self.raw_candidate_count = 0
        self.accepted_point_count = 0
        self.rejected_candidate_count = 0
        self.rejection_reasons: Counter[str] = Counter()
        self.accepted_positions: list[tuple[int, int]] = []
        self.static_hits: Counter[tuple[int, int]] = Counter()
        self.consecutive_reject_frames = 0

    def _max_jump_px(self) -> float:
        return max(60.0, hypot(self.frame_width, self.frame_height) * 0.12)

    def predict_next(self, kalman_prediction=None):
        if kalman_prediction is not None:
            return kalman_prediction
        if len(self.accepted_positions) >= 2:
            x1, y1 = self.accepted_positions[-2]
            x2, y2 = self.accepted_positions[-1]
            return (x2 + (x2 - x1), y2 + (y2 - y1))
        if self.accepted_positions:
            return self.accepted_positions[-1]
        return None

    def _record_static_hits(self, ball_detections) -> None:
        for detection in ball_detections:
            self.static_hits[_grid_key(detection["center"])] += 1

    def _is_static_hotspot(self, center, reference_center) -> bool:
        hits = self.static_hits.get(_grid_key(center), 0)
        if hits < STATIC_MIN_HITS:
            return False
        return _distance(center, reference_center) > self._max_jump_px() * 0.35

    def _score_candidate(
        self,
        detection,
        reference_center,
        predicted_center,
    ) -> tuple[float, list[str]]:
        center = detection["center"]
        rejections: list[str] = []
        max_jump = self._max_jump_px()

        if reference_center is not None:
            previous_distance = _distance(center, reference_center)
            if previous_distance > max_jump:
                rejections.append("impossible_jump")
            predicted_distance = _distance(center, predicted_center)
            if predicted_center is not None and predicted_distance > max_jump * 1.15:
                if "impossible_jump" not in rejections:
                    rejections.append("far_from_track")
            if self._is_static_hotspot(center, reference_center):
                rejections.append("static_false_positive")

        if rejections:
            return float("-inf"), rejections

        score = float(detection.get("confidence", 0.0)) * 12.0
        if reference_center is not None:
            score -= _distance(center, reference_center) * 0.18
        if predicted_center is not None:
            score -= _distance(center, predicted_center) * 0.22
        return score, rejections

    def select(
        self,
        ball_detections,
        previous_center=None,
        kalman_prediction=None,
    ):
        """Return the best trajectory-consistent detection, or None."""
        if not ball_detections:
            return None

        self.raw_candidate_count += len(ball_detections)
        self._record_static_hits(ball_detections)

        if previous_center is None and not self.accepted_positions:
            best = max(ball_detections, key=lambda item: item["confidence"])
            self._accept(best, rejected=len(ball_detections) - 1)
            return best

        reference_center = previous_center
        if reference_center is None and self.accepted_positions:
            reference_center = self.accepted_positions[-1]
        predicted_center = self.predict_next(kalman_prediction)

        scored: list[tuple[float, dict]] = []
        for detection in ball_detections:
            score, rejections = self._score_candidate(
                detection,
                reference_center,
                predicted_center,
            )
            if rejections:
                self.rejected_candidate_count += 1
                for reason in rejections:
                    self.rejection_reasons[reason] += 1
                continue
            scored.append((score, detection))

        if not scored:
            self.consecutive_reject_frames += 1
            self.rejection_reasons["no_acceptable_candidate"] += 1
            return None

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best = scored[0]
        alternate_rejections = len(scored) - 1
        for score, detection in scored[1:]:
            if best_score - score > 1.0:
                alternate_rejections += 1
                self.rejection_reasons["lower_trajectory_score"] += 1
        self._accept(best, rejected=alternate_rejections)
        return best

    def _accept(self, detection, rejected=0) -> None:
        self.accepted_positions.append(tuple(detection["center"]))
        self.accepted_point_count += 1
        self.rejected_candidate_count += max(rejected, 0)
        self.consecutive_reject_frames = 0

    def has_reliable_track(self) -> bool:
        if self.accepted_point_count < MIN_ACCEPTED_POINTS_FOR_RELIABILITY:
            return False
        if self.consecutive_reject_frames >= MAX_CONSECUTIVE_REJECT_FRAMES:
            return False
        if len(self.accepted_positions) < 2:
            return False
        return _distance(
            self.accepted_positions[0],
            self.accepted_positions[-1],
        ) >= 25.0

    def debug_summary(self, tracking_quality_label: str) -> dict:
        return {
            "raw_ball_candidate_count": self.raw_candidate_count,
            "accepted_ball_point_count": self.accepted_point_count,
            "rejected_candidate_count": self.rejected_candidate_count,
            "main_rejection_reasons": dict(
                self.rejection_reasons.most_common(5)
            ),
            "tracking_quality": tracking_quality_label,
            "trajectory_reliable": self.has_reliable_track(),
        }
