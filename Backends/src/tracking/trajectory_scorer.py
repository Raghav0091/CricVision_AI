"""Trajectory-aware ball candidate selection for multi-detection frames."""

from __future__ import annotations

from collections import Counter
from math import hypot

# ponytail: fixed pixel thresholds are enough for MVP delivery clips.
STATIC_CELL_PX = 10
STATIC_MIN_HITS = 3
BOOTSTRAP_MIN_FRAMES = 3
BOOTSTRAP_MIN_MOVEMENT_PX = 15.0
BOOTSTRAP_STATIC_MOVEMENT_PX = 8.0
BOOTSTRAP_EDGE_MARGIN_RATIO = 0.015
BOOTSTRAP_EDGE_MARGIN_MIN_PX = 4
BOOTSTRAP_WINDOW_MAX_FRAMES = 60
BOOTSTRAP_MAX_FRAME_GAP = 4
BOOTSTRAP_MAX_EDGE_FRACTION = 0.5
BOOTSTRAP_MAX_DOMINANT_STEP_RATIO = 0.65
BOOTSTRAP_PARTIAL_CHAIN_LIMIT = 50
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
        self._track_confirmed = False
        self._bootstrap_window: list[list[dict]] = []
        self._bootstrap_provisional_point_count = 0
        self._rejected_bootstrap_cells: set[tuple[int, int]] = set()

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

    def _bootstrap_edge_margin(self) -> float:
        return max(
            BOOTSTRAP_EDGE_MARGIN_MIN_PX,
            min(self.frame_width, self.frame_height)
            * BOOTSTRAP_EDGE_MARGIN_RATIO,
        )

    def _touches_frame_edge(self, detection) -> bool:
        margin = self._bootstrap_edge_margin()
        box = detection.get("box") or detection.get("bbox")
        if box is not None and len(box) >= 4:
            x1, y1, x2, y2 = box[:4]
            return (
                x1 <= margin
                or y1 <= margin
                or x2 >= self.frame_width - margin
                or y2 >= self.frame_height - margin
            )

        center = detection.get("center")
        if center is None:
            return False
        x, y = center
        return (
            x <= margin
            or y <= margin
            or x >= self.frame_width - margin
            or y >= self.frame_height - margin
        )

    def _clear_bootstrap_window(self) -> None:
        self._bootstrap_window.clear()
        self._bootstrap_provisional_point_count = 0

    def _reject_static_bootstrap_window(self) -> None:
        if not self._bootstrap_window:
            return
        dominant_cell = Counter(
            _grid_key(detection["center"])
            for frame_candidates in self._bootstrap_window
            for detection in frame_candidates
        ).most_common(1)
        if dominant_cell:
            self._rejected_bootstrap_cells.add(dominant_cell[0][0])
        self._clear_bootstrap_window()
        self.rejection_reasons["static_bootstrap_rejected"] += 1

    def _confirm_bootstrap_chain(self, chain) -> None:
        for _, detection in chain:
            self._accept(detection, rejected=0)
        self._track_confirmed = True
        self._clear_bootstrap_window()

    def _bootstrap_rejection_reasons(self, detection) -> list[str]:
        reasons = []
        if _grid_key(detection["center"]) in self._rejected_bootstrap_cells:
            reasons.append("bootstrap_cell_previously_rejected")
        if self._touches_frame_edge(detection):
            reasons.append("bootstrap_edge_candidate")
        return reasons

    def _split_bootstrap_detections(self, ball_detections):
        eligible = []
        rejected = []
        for detection in ball_detections:
            reasons = self._bootstrap_rejection_reasons(detection)
            if reasons:
                rejected.append((detection, reasons))
            else:
                eligible.append(detection)
        return eligible, rejected

    def _chain_positions(self, chain) -> list[tuple[int, int]]:
        return [tuple(detection["center"]) for _, detection in chain]

    def _chain_score(self, chain) -> float:
        score = 0.0
        positions = self._chain_positions(chain)
        for index, (_, detection) in enumerate(chain):
            score += float(detection.get("confidence", 0.0)) * 12.0
            if index > 0:
                step = _distance(positions[index - 1], positions[index])
                score += min(step, self._max_jump_px()) * 0.08
        if len(positions) >= 3:
            direction_hits = 0
            for index in range(2, len(positions)):
                ax, ay = positions[index - 2]
                bx, by = positions[index - 1]
                cx, cy = positions[index]
                v1 = (bx - ax, by - ay)
                v2 = (cx - bx, cy - by)
                if v1[0] * v2[0] + v1[1] * v2[1] > 0:
                    direction_hits += 1
            score += direction_hits * 2.5
        return score

    def _is_valid_bootstrap_chain(self, chain) -> bool:
        if len(chain) < BOOTSTRAP_MIN_FRAMES:
            return False

        positions = self._chain_positions(chain)
        grid_cells = [_grid_key(position) for position in positions]
        if len(set(grid_cells)) < len(positions):
            return False

        max_jump = self._max_jump_px()
        path_length = 0.0
        max_step = 0.0
        for index in range(1, len(positions)):
            step = _distance(positions[index - 1], positions[index])
            if step > max_jump or step < 1.0:
                return False
            path_length += step
            max_step = max(max_step, step)

        if path_length < BOOTSTRAP_MIN_MOVEMENT_PX:
            return False
        if _distance(positions[0], positions[-1]) < BOOTSTRAP_MIN_MOVEMENT_PX:
            return False
        if max_step / path_length > BOOTSTRAP_MAX_DOMINANT_STEP_RATIO:
            return False

        frame_indices = [frame_index for frame_index, _ in chain]
        for index in range(1, len(frame_indices)):
            gap = frame_indices[index] - frame_indices[index - 1]
            if gap < 1 or gap > BOOTSTRAP_MAX_FRAME_GAP:
                return False

        edge_count = sum(
            1 for _, detection in chain if self._touches_frame_edge(detection)
        )
        if edge_count / len(chain) > BOOTSTRAP_MAX_EDGE_FRACTION:
            return False

        return True

    def _bootstrap_window_is_static(self) -> bool:
        if not self._bootstrap_window:
            return False
        cells = [
            _grid_key(detection["center"])
            for frame_candidates in self._bootstrap_window
            for detection in frame_candidates
        ]
        if not cells:
            return False
        dominant_count = Counter(cells).most_common(1)[0][1]
        return dominant_count >= STATIC_MIN_HITS and len(set(cells)) == 1

    def _find_best_bootstrap_chain(self):
        best_valid = None
        best_valid_score = float("-inf")
        partials: list[tuple[float, list]] = []

        for frame_index, frame_candidates in enumerate(self._bootstrap_window):
            next_partials: list[tuple[float, list]] = []
            for detection in frame_candidates:
                solo_chain = [(frame_index, detection)]
                solo_score = self._chain_score(solo_chain)
                next_partials.append((solo_score, solo_chain))

                for score, chain in partials:
                    last_frame_index, last_detection = chain[-1]
                    gap = frame_index - last_frame_index
                    if gap < 1 or gap > BOOTSTRAP_MAX_FRAME_GAP:
                        continue
                    step = _distance(
                        last_detection["center"],
                        detection["center"],
                    )
                    if step > self._max_jump_px():
                        continue
                    if _grid_key(last_detection["center"]) == _grid_key(
                        detection["center"]
                    ):
                        continue

                    extended = chain + [(frame_index, detection)]
                    extended_score = self._chain_score(extended)
                    next_partials.append((extended_score, extended))
                    if (
                        self._is_valid_bootstrap_chain(extended)
                        and extended_score > best_valid_score
                    ):
                        best_valid_score = extended_score
                        best_valid = extended

            partials = sorted(next_partials, key=lambda item: item[0], reverse=True)[
                :BOOTSTRAP_PARTIAL_CHAIN_LIMIT
            ]

        self._bootstrap_provisional_point_count = sum(
            1 for frame_candidates in self._bootstrap_window if frame_candidates
        )
        return best_valid

    def _record_diagnostic(
        self,
        diagnostics,
        detection,
        *,
        candidate_index=0,
        selected=False,
        rejected=False,
        reasons=None,
        score=None,
        reference_center=None,
        predicted_center=None,
    ) -> None:
        if diagnostics is None:
            return
        reasons = reasons or []
        center = detection.get("center")
        diagnostics.append(
            {
                "candidate_id": detection.get(
                    "_debug_candidate_id",
                    candidate_index,
                ),
                "selected": bool(selected),
                "rejected": bool(rejected),
                "rejection_reason": ";".join(reasons),
                "score": score,
                "reference_center": reference_center,
                "predicted_center": predicted_center,
                "center": center,
            }
        )

    def _select_bootstrap(
        self,
        ball_detections,
        previous_center=None,
        diagnostics=None,
    ):
        eligible, bootstrap_rejections = self._split_bootstrap_detections(
            ball_detections
        )
        for detection, reasons in bootstrap_rejections:
            self.rejected_candidate_count += 1
            for reason in reasons:
                self.rejection_reasons[reason] += 1
            self._record_diagnostic(
                diagnostics,
                detection,
                candidate_index=ball_detections.index(detection),
                rejected=True,
                reasons=reasons,
                reference_center=previous_center,
                predicted_center=None,
            )

        self._bootstrap_window.append(eligible)
        if len(self._bootstrap_window) > BOOTSTRAP_WINDOW_MAX_FRAMES:
            self._bootstrap_window.pop(0)
            self.rejection_reasons["bootstrap_window_exhausted"] += 1

        best_chain = self._find_best_bootstrap_chain()
        if (
            best_chain is None
            and len(self._bootstrap_window) >= STATIC_MIN_HITS
            and self._bootstrap_window_is_static()
        ):
            self._reject_static_bootstrap_window()
            return self._select_bootstrap(
                ball_detections,
                previous_center,
                diagnostics,
            )
        last_frame_index = len(self._bootstrap_window) - 1
        if (
            best_chain is not None
            and best_chain[-1][0] == last_frame_index
            and self._is_valid_bootstrap_chain(best_chain)
        ):
            selected_detection = best_chain[-1][1]
            for index, detection in enumerate(ball_detections):
                is_selected = detection is selected_detection
                self._record_diagnostic(
                    diagnostics,
                    detection,
                    candidate_index=index,
                    selected=is_selected,
                    rejected=not is_selected,
                    reasons=(
                        []
                        if is_selected
                        else ["lower_bootstrap_trajectory_score"]
                    ),
                    score=(
                        self._chain_score(best_chain)
                        if is_selected
                        else None
                    ),
                    reference_center=previous_center,
                    predicted_center=None,
                )
            self._confirm_bootstrap_chain(best_chain)
            return selected_detection

        pending_reason = "bootstrap_pending_no_valid_chain"
        if not eligible:
            self.consecutive_reject_frames += 1
            self.rejection_reasons["no_acceptable_candidate"] += 1
        else:
            self.rejected_candidate_count += len(eligible)
            self.rejection_reasons[pending_reason] += len(eligible)
            for index, detection in enumerate(ball_detections):
                if detection not in eligible:
                    continue
                self._record_diagnostic(
                    diagnostics,
                    detection,
                    candidate_index=index,
                    rejected=True,
                    reasons=[pending_reason],
                    reference_center=previous_center,
                    predicted_center=None,
                )
        return None

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
        diagnostics=None,
    ):
        """Return the best trajectory-consistent detection, or None."""
        if not ball_detections:
            return None

        self.raw_candidate_count += len(ball_detections)
        self._record_static_hits(ball_detections)

        if not self._track_confirmed:
            return self._select_bootstrap(
                ball_detections,
                previous_center,
                diagnostics,
            )

        reference_center = previous_center
        if reference_center is None and self.accepted_positions:
            reference_center = self.accepted_positions[-1]
        predicted_center = self.predict_next(kalman_prediction)

        scored: list[tuple[float, dict]] = []
        for index, detection in enumerate(ball_detections):
            score, rejections = self._score_candidate(
                detection,
                reference_center,
                predicted_center,
            )
            if rejections:
                self.rejected_candidate_count += 1
                for reason in rejections:
                    self.rejection_reasons[reason] += 1
                self._record_diagnostic(
                    diagnostics,
                    detection,
                    candidate_index=index,
                    rejected=True,
                    reasons=rejections,
                    score=score,
                    reference_center=reference_center,
                    predicted_center=predicted_center,
                )
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
        for score, detection in scored:
            self._record_diagnostic(
                diagnostics,
                detection,
                candidate_index=ball_detections.index(detection),
                selected=detection is best,
                rejected=detection is not best,
                reasons=[] if detection is best else ["lower_trajectory_score"],
                score=score,
                reference_center=reference_center,
                predicted_center=predicted_center,
            )
        self._accept(best, rejected=alternate_rejections)
        return best

    def _accept(self, detection, rejected=0) -> None:
        self.accepted_positions.append(tuple(detection["center"]))
        self.accepted_point_count += 1
        self.rejected_candidate_count += max(rejected, 0)
        self.consecutive_reject_frames = 0

    def has_reliable_track(self) -> bool:
        if not self._track_confirmed:
            return False
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
        trajectory_reliable = self.has_reliable_track()
        return {
            "raw_ball_candidate_count": self.raw_candidate_count,
            "accepted_ball_point_count": self.accepted_point_count,
            "bootstrap_established": self._track_confirmed
            and (
                trajectory_reliable
                or self._bootstrap_can_reach_reliability()
            ),
            "bootstrap_provisional_point_count": self._bootstrap_provisional_point_count,
            "bootstrap_min_frames": BOOTSTRAP_MIN_FRAMES,
            "bootstrap_min_movement_px": BOOTSTRAP_MIN_MOVEMENT_PX,
            "rejected_candidate_count": self.rejected_candidate_count,
            "main_rejection_reasons": dict(
                self.rejection_reasons.most_common(5)
            ),
            "tracking_quality": tracking_quality_label,
            "trajectory_reliable": trajectory_reliable,
        }

    def _bootstrap_can_reach_reliability(self) -> bool:
        if not self._track_confirmed:
            return False
        if len(self.accepted_positions) < BOOTSTRAP_MIN_FRAMES:
            return False
        prefix = self.accepted_positions[:BOOTSTRAP_MIN_FRAMES]
        chain = [(index, {"center": center}) for index, center in enumerate(prefix)]
        return self._is_valid_bootstrap_chain(chain)
