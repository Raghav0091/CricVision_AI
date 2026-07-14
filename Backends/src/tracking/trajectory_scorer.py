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
MIN_FULL_ANALYSIS_TRACK_POINTS = 12
MAX_CONSECUTIVE_REJECT_FRAMES = 4
DELIVERY_MAX_FRAME_GAP = 8
DELIVERY_MAX_CONSECUTIVE_MISSES = 5
DELIVERY_NEAR_STATIC_STEP_PX = BOOTSTRAP_STATIC_MOVEMENT_PX
DELIVERY_NEAR_STATIC_CONSECUTIVE = 4
DELIVERY_MAX_FAR_FROM_PREDICTION_STREAK = 3
DELIVERY_SEGMENT_JUMP_RATIO = 0.85


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
        self._active_delivery_track = False
        self._track_terminated = False
        self._last_accepted_frame_index = -1
        self._select_call_index = -1
        self._consecutive_near_static_acceptances = 0
        self._consecutive_far_from_prediction_frames = 0
        self._bootstrap_window: list[list[dict]] = []
        self._bootstrap_provisional_point_count = 0
        self._rejected_bootstrap_cells: set[tuple[int, int]] = set()
        self._accepted_frame_indices: list[int] = []

    def apply_ranked_tracklet(
        self,
        tracklet_points: list[dict],
        *,
        segment_end_frame: int | None = None,
    ) -> None:
        """Replace accepted track state with an offline-ranked best tracklet."""
        ordered = sorted(
            tracklet_points,
            key=lambda point: int(point["frame_index"]),
        )
        if not ordered:
            return
        self.accepted_positions = [
            (int(round(point["x"])), int(round(point["y"])))
            for point in ordered
        ]
        self._accepted_frame_indices = [
            int(point["frame_index"]) for point in ordered
        ]
        self.accepted_point_count = len(ordered)
        self._track_confirmed = len(ordered) >= BOOTSTRAP_MIN_FRAMES
        self._active_delivery_track = self._track_confirmed
        end_frame = (
            int(segment_end_frame)
            if segment_end_frame is not None
            else int(ordered[-1]["frame_index"])
        )
        self._last_accepted_frame_index = end_frame
        self._track_terminated = True
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

    def _confirm_bootstrap_chain(self, chain, current_frame: int | None = None) -> None:
        last_window_index = len(self._bootstrap_window) - 1
        for window_index, detection in chain:
            frame_index = None
            if current_frame is not None:
                frame_index = current_frame - (last_window_index - window_index)
            self._accept(detection, rejected=0, frame_index=frame_index)
        self._track_confirmed = True
        self._active_delivery_track = True
        if current_frame is not None:
            self._last_accepted_frame_index = current_frame
        self._clear_bootstrap_window()

    def _resolve_frame_index(self, frame_index) -> int:
        if frame_index is not None:
            self._select_call_index = int(frame_index)
        else:
            self._select_call_index += 1
        return self._select_call_index

    def _terminate_delivery_track(self, reason: str = "delivery_track_lost") -> None:
        if self._track_terminated:
            return
        self._track_terminated = True
        self._active_delivery_track = False
        self.rejection_reasons[reason] += 1

    def _reject_after_termination(
        self,
        ball_detections,
        *,
        previous_center=None,
        predicted_center=None,
        diagnostics=None,
    ):
        for index, detection in enumerate(ball_detections):
            self.rejected_candidate_count += 1
            self.rejection_reasons["after_track_terminated"] += 1
            self._record_diagnostic(
                diagnostics,
                detection,
                candidate_index=index,
                rejected=True,
                reasons=["after_track_terminated"],
                reference_center=previous_center,
                predicted_center=predicted_center,
            )

    def _delivery_frame_gap_exceeded(self, current_frame: int) -> bool:
        if self._last_accepted_frame_index < 0:
            return False
        return (
            current_frame - self._last_accepted_frame_index
            > DELIVERY_MAX_FRAME_GAP
        )

    def _near_static_acceptance_streak_after(self, center) -> int:
        if not self.accepted_positions:
            return 0
        step = _distance(self.accepted_positions[-1], center)
        if step < DELIVERY_NEAR_STATIC_STEP_PX:
            return self._consecutive_near_static_acceptances + 1
        return 0

    def _should_terminate_delivery_track(
        self,
        current_frame: int,
        *,
        pending_center=None,
        predicted_center=None,
    ) -> str | None:
        if not self._active_delivery_track or self._track_terminated:
            return None

        if self._delivery_frame_gap_exceeded(current_frame):
            return "delivery_track_lost"

        reliable = self.has_reliable_track()
        if (
            reliable
            and self.consecutive_reject_frames >= DELIVERY_MAX_CONSECUTIVE_MISSES
        ):
            return "delivery_track_lost"

        if pending_center is not None:
            if (
                self._near_static_acceptance_streak_after(pending_center)
                >= DELIVERY_NEAR_STATIC_CONSECUTIVE
            ):
                return "delivery_track_lost"

            if predicted_center is not None:
                max_jump = self._max_jump_px()
                if (
                    _distance(pending_center, predicted_center)
                    > max_jump * DELIVERY_SEGMENT_JUMP_RATIO
                    and self.consecutive_reject_frames > 0
                ):
                    return "delivery_track_lost"

        if (
            self._consecutive_far_from_prediction_frames
            >= DELIVERY_MAX_FAR_FROM_PREDICTION_STREAK
        ):
            return "delivery_track_lost"

        return None

    def _on_delivery_miss_frame(
        self,
        current_frame: int,
        *,
        predicted_center=None,
    ) -> None:
        self.consecutive_reject_frames += 1
        if predicted_center is not None:
            self._consecutive_far_from_prediction_frames += 1

        reason = self._should_terminate_delivery_track(
            current_frame,
            predicted_center=predicted_center,
        )
        if reason:
            self._terminate_delivery_track(reason)

    def _on_delivery_accept_frame(self, detection, current_frame: int) -> None:
        center = detection["center"]
        self._accept(detection, rejected=0, frame_index=current_frame)
        step = (
            _distance(self.accepted_positions[-2], center)
            if len(self.accepted_positions) >= 2
            else float("inf")
        )
        if step < DELIVERY_NEAR_STATIC_STEP_PX:
            self._consecutive_near_static_acceptances += 1
        else:
            self._consecutive_near_static_acceptances = 0
        self._last_accepted_frame_index = current_frame
        self._consecutive_far_from_prediction_frames = 0

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
        current_frame=0,
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
                current_frame=current_frame,
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
            self._confirm_bootstrap_chain(best_chain, current_frame)
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
        frame_index=None,
    ):
        """Return the best trajectory-consistent detection, or None."""
        current_frame = self._resolve_frame_index(frame_index)

        if self._track_terminated:
            if ball_detections:
                reference_center = previous_center
                if reference_center is None and self.accepted_positions:
                    reference_center = self.accepted_positions[-1]
                self._reject_after_termination(
                    ball_detections,
                    previous_center=reference_center,
                    predicted_center=self.predict_next(kalman_prediction),
                    diagnostics=diagnostics,
                )
            return None

        if not ball_detections:
            if self._active_delivery_track:
                self._on_delivery_miss_frame(
                    current_frame,
                    predicted_center=self.predict_next(kalman_prediction),
                )
            return None

        self.raw_candidate_count += len(ball_detections)
        self._record_static_hits(ball_detections)

        if not self._track_confirmed:
            return self._select_bootstrap(
                ball_detections,
                previous_center,
                diagnostics,
                current_frame=current_frame,
            )

        reference_center = previous_center
        if reference_center is None and self.accepted_positions:
            reference_center = self.accepted_positions[-1]
        predicted_center = self.predict_next(kalman_prediction)

        if self._active_delivery_track and self._delivery_frame_gap_exceeded(
            current_frame
        ):
            self._terminate_delivery_track("delivery_track_lost")
            self._reject_after_termination(
                ball_detections,
                previous_center=reference_center,
                predicted_center=predicted_center,
                diagnostics=diagnostics,
            )
            return None

        scored: list[tuple[float, dict]] = []
        all_far_from_prediction = bool(ball_detections) and predicted_center is not None
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
                if not any(
                    reason in ("far_from_track", "impossible_jump")
                    for reason in rejections
                ):
                    all_far_from_prediction = False
                continue
            all_far_from_prediction = False
            scored.append((score, detection))

        if not scored:
            if self._active_delivery_track:
                if all_far_from_prediction and predicted_center is not None:
                    self._consecutive_far_from_prediction_frames += 1
                self._on_delivery_miss_frame(
                    current_frame,
                    predicted_center=predicted_center,
                )
                if self._track_terminated and ball_detections:
                    self._reject_after_termination(
                        ball_detections,
                        previous_center=reference_center,
                        predicted_center=predicted_center,
                        diagnostics=diagnostics,
                    )
            else:
                self.consecutive_reject_frames += 1
                self.rejection_reasons["no_acceptable_candidate"] += 1
            return None

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best = scored[0]

        if self._active_delivery_track:
            accept_reason = self._should_terminate_delivery_track(
                current_frame,
                pending_center=best["center"],
                predicted_center=predicted_center,
            )
            if accept_reason:
                self._terminate_delivery_track(accept_reason)

        alternate_rejections = len(scored) - 1
        for score, detection in scored[1:]:
            if best_score - score > 1.0:
                alternate_rejections += 1
                self.rejection_reasons["lower_trajectory_score"] += 1

        if self._track_terminated:
            self.rejected_candidate_count += len(ball_detections)
            for index, detection in enumerate(ball_detections):
                self.rejection_reasons["after_track_terminated"] += 1
                self._record_diagnostic(
                    diagnostics,
                    detection,
                    candidate_index=index,
                    rejected=True,
                    reasons=["after_track_terminated"],
                    score=None,
                    reference_center=reference_center,
                    predicted_center=predicted_center,
                )
            return None

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

        if self._active_delivery_track:
            self._on_delivery_accept_frame(best, current_frame)
            self.rejected_candidate_count += max(alternate_rejections, 0)
        else:
            self._accept(best, rejected=alternate_rejections)
        return best

    def _accept(self, detection, rejected=0, frame_index=None) -> None:
        self.accepted_positions.append(tuple(detection["center"]))
        self.accepted_point_count += 1
        if frame_index is not None:
            self._accepted_frame_indices.append(int(frame_index))
        elif self._select_call_index >= 0:
            self._accepted_frame_indices.append(self._select_call_index)
        self.rejected_candidate_count += max(rejected, 0)
        self.consecutive_reject_frames = 0

    def delivery_track_found(self) -> bool:
        if not self._track_confirmed:
            return False
        if self.accepted_point_count < BOOTSTRAP_MIN_FRAMES:
            return False
        if len(self.accepted_positions) < 2:
            return False
        return (
            _distance(
                self.accepted_positions[0],
                self.accepted_positions[-1],
            )
            >= BOOTSTRAP_MIN_MOVEMENT_PX * 0.5
        )

    def selected_track_start_frame(self) -> int | None:
        if not self._accepted_frame_indices:
            return None
        return int(self._accepted_frame_indices[0])

    def selected_track_end_frame(self) -> int | None:
        if not self._accepted_frame_indices:
            return None
        return int(self._accepted_frame_indices[-1])

    def selected_track_frame_count(self) -> int:
        return len(self._accepted_frame_indices)

    def selected_track_span_frames(self) -> int:
        if len(self._accepted_frame_indices) < 2:
            return len(self._accepted_frame_indices)
        return (
            self._accepted_frame_indices[-1] - self._accepted_frame_indices[0] + 1
        )

    def selected_track_duration_sec(self, fps: float | None) -> float | None:
        if fps is None or fps <= 0:
            return None
        span = self.selected_track_span_frames()
        if span <= 0:
            return None
        return span / float(fps)

    def segment_tracking_rate(self) -> float:
        span = self.selected_track_span_frames()
        if span <= 0:
            return 0.0
        return (self.accepted_point_count / span) * 100.0

    def is_short_for_delivery_analysis(
        self,
        min_frames: int = 8,
        min_movement: float = 40.0,
    ) -> bool:
        if not self.has_reliable_track():
            return False
        if self.accepted_point_count < min_frames:
            return True
        if len(self.accepted_positions) < 2:
            return True
        if (
            _distance(
                self.accepted_positions[0],
                self.accepted_positions[-1],
            )
            < min_movement
        ):
            return True
        if self.accepted_point_count < MIN_FULL_ANALYSIS_TRACK_POINTS:
            return True
        return False

    def short_track_reason(
        self,
        min_frames: int = 8,
        min_movement: float = 40.0,
    ) -> str | None:
        if not self.delivery_track_found() or not self.is_short_for_delivery_analysis(
            min_frames=min_frames,
            min_movement=min_movement,
        ):
            return None
        reasons: list[str] = []
        if self.accepted_point_count < min_frames:
            reasons.append("too_few_track_points")
        if len(self.accepted_positions) >= 2 and (
            _distance(
                self.accepted_positions[0],
                self.accepted_positions[-1],
            )
            < min_movement
        ):
            reasons.append("insufficient_track_movement")
        if self.accepted_point_count < MIN_FULL_ANALYSIS_TRACK_POINTS:
            reasons.append("too_short_for_bounce_length")
        return ";".join(reasons) if reasons else "short_delivery_segment"

    def has_reliable_track(self) -> bool:
        if not self._track_confirmed:
            return False
        if self.accepted_point_count < MIN_ACCEPTED_POINTS_FOR_RELIABILITY:
            return False
        if (
            self._active_delivery_track
            and not self._track_terminated
            and self.consecutive_reject_frames >= MAX_CONSECUTIVE_REJECT_FRAMES
        ):
            return False
        if len(self.accepted_positions) < 2:
            return False
        return _distance(
            self.accepted_positions[0],
            self.accepted_positions[-1],
        ) >= 25.0

    def debug_summary(
        self,
        tracking_quality_label: str,
        *,
        fps: float | None = None,
        min_track_points_for_bounce: int = 8,
        min_movement_distance: float = 40.0,
    ) -> dict:
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
            "delivery_track_found": self.delivery_track_found(),
            "delivery_track_terminated": self._track_terminated,
            "selected_track_start_frame": self.selected_track_start_frame(),
            "selected_track_end_frame": self.selected_track_end_frame(),
            "selected_track_frame_count": self.selected_track_frame_count(),
            "selected_track_span_frames": self.selected_track_span_frames(),
            "selected_track_duration_sec": self.selected_track_duration_sec(fps),
            "short_track_reason": self.short_track_reason(
                min_frames=min_track_points_for_bounce,
                min_movement=min_movement_distance,
            ),
        }

    def _bootstrap_can_reach_reliability(self) -> bool:
        if not self._track_confirmed:
            return False
        if len(self.accepted_positions) < BOOTSTRAP_MIN_FRAMES:
            return False
        prefix = self.accepted_positions[:BOOTSTRAP_MIN_FRAMES]
        chain = [(index, {"center": center}) for index, center in enumerate(prefix)]
        return self._is_valid_bootstrap_chain(chain)


def resolve_delivery_tracking_quality(
    selector: TrajectoryBallSelector,
    *,
    interpolated_frames: int = 0,
    kalman_predicted_frames: int = 0,
    min_track_points_for_bounce: int = 8,
    min_movement_distance: float = 40.0,
) -> tuple[str, bool]:
    """Return overall quality label and whether line/length/bounce should stay unknown."""
    from Backends.src.tracking.ball_tracking_utils import get_tracking_quality_label

    if not selector.has_reliable_track():
        return "Poor", True

    if selector.is_short_for_delivery_analysis(
        min_frames=min_track_points_for_bounce,
        min_movement=min_movement_distance,
    ):
        return "Partial", True

    overall_tracking_quality = get_tracking_quality_label(
        selector.segment_tracking_rate(),
        interpolated_frames,
        kalman_predicted_frames,
    )
    if overall_tracking_quality == "Poor":
        # ponytail: a reliable segment should not be downgraded to Poor by span math alone.
        overall_tracking_quality = "Medium"
    return overall_tracking_quality, False


def default_max_link_distance_px(frame_width: int, frame_height: int) -> float:
    """Reasonable per-step link distance for offline replay."""
    return max(48.0, min(max(int(frame_width or 0), 1), max(int(frame_height or 0), 1)) * 0.12)


OFFLINE_MIN_TOTAL_MOVEMENT_PX = 12.0
OFFLINE_MIN_AVERAGE_MOVEMENT_PX = 0.5
OFFLINE_MIN_SPATIAL_SPREAD_PX = 8.0
OFFLINE_MIN_UNIQUE_CENTER_RATIO = 0.25
EXTENSION_VELOCITY_SAMPLE_POINTS = 3
EXTENSION_MAX_FRAME_GAP = 3
EXTENSION_MAX_CONSECUTIVE_MISSES = 2
EXTENSION_MAX_SPAN_FRAMES = 40
EXTENSION_DISTANCE_TOLERANCE_PX = 28.0
EXTENSION_MIN_SMOOTHNESS_RATIO = 0.9


def rank_candidate_tracklets(
    frame_candidates: dict[int, list[dict]],
    frame_width: int,
    frame_height: int,
    fps: float | None = None,
    *,
    top_n: int = 10,
    max_frame_gap: int = 5,
    max_link_distance_px: float | None = None,
    min_tracklet_points: int = 3,
    partial_chain_limit: int = 200,
    min_total_movement_px: float = OFFLINE_MIN_TOTAL_MOVEMENT_PX,
    min_average_movement_px: float = OFFLINE_MIN_AVERAGE_MOVEMENT_PX,
) -> dict:
    """Offline retrospective ranking of ball tracklets without online termination.

    Links raw per-frame candidates into tracklets using distance and frame-gap rules
    only. Does not use bootstrap confirmation or ``after_track_terminated``.
    """
    frame_width = max(int(frame_width or 0), 1)
    frame_height = max(int(frame_height or 0), 1)
    max_link = max_link_distance_px or default_max_link_distance_px(
        frame_width, frame_height
    )
    min_points = max(int(min_tracklet_points or 0), 2)
    max_gap = max(int(max_frame_gap or 0), 1)

    completed, failed_attempts = _build_offline_tracklets(
        frame_candidates,
        max_frame_gap=max_gap,
        max_link_distance_px=max_link,
        min_tracklet_points=min_points,
        partial_chain_limit=partial_chain_limit,
    )

    ranked: list[dict] = []
    rejected_static: list[dict] = []
    for chain in completed:
        metrics = _offline_tracklet_metrics(
            chain,
            frame_width=frame_width,
            frame_height=frame_height,
            fps=fps,
            include_tracklet_points=True,
        )
        rejection_reason = _offline_static_rejection_reason(
            metrics,
            chain=chain,
            min_total_movement_px=min_total_movement_px,
            min_average_movement_px=min_average_movement_px,
        )
        if rejection_reason:
            metrics["rejection_reason"] = rejection_reason
            rejected_static.append(metrics)
            continue
        ranked.append(metrics)

    ranked.sort(key=lambda item: item["final_segment_score"], reverse=True)
    rejected_static.sort(
        key=lambda item: (item.get("point_count", 0), item.get("frame_span", 0)),
        reverse=True,
    )

    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
        item["reason"] = _offline_segment_reason(item, rank=index, top_n=top_n)

    top_segments = ranked[:top_n]
    winner = top_segments[0] if top_segments else None
    overlap_40_76 = [
        item
        for item in ranked
        if item["start_frame"] <= 76 and item["end_frame"] >= 40
    ]

    why_no_segments = None
    nearest_failed = []
    if not ranked:
        why_no_segments = _why_no_offline_segments(
            frame_candidates,
            failed_attempts,
            rejected_static=rejected_static,
            min_tracklet_points=min_points,
            max_frame_gap=max_gap,
            max_link_distance_px=max_link,
            min_total_movement_px=min_total_movement_px,
            min_average_movement_px=min_average_movement_px,
        )
        nearest_failed = _nearest_failed_tracklet_attempts(
            failed_attempts,
            min_tracklet_points=min_points,
            limit=top_n,
        )

    return {
        "frame_width": frame_width,
        "frame_height": frame_height,
        "fps": fps,
        "max_frame_gap": max_gap,
        "max_link_distance_px": round(max_link, 2),
        "min_tracklet_points": min_points,
        "min_total_movement_px": min_total_movement_px,
        "min_average_movement_px": min_average_movement_px,
        "total_ball_frames": len(frame_candidates),
        "candidate_segment_count": len(completed),
        "rejected_static_segment_count": len(rejected_static),
        "total_valid_tracklets": len(ranked),
        "top_segments": top_segments,
        "winner": winner,
        "rejected_static_segments": rejected_static[:top_n],
        "frames_40_76_in_any_ranked_segment": bool(overlap_40_76),
        "ranked_segments_overlapping_frames_40_76": overlap_40_76[:top_n],
        "ranking_applies_after_track_terminated": False,
        "why_no_segments": why_no_segments,
        "nearest_failed_segments": nearest_failed,
    }


def _build_offline_tracklets(
    frame_candidates: dict[int, list[dict]],
    *,
    max_frame_gap: int,
    max_link_distance_px: float,
    min_tracklet_points: int,
    partial_chain_limit: int,
) -> tuple[list[list[tuple[int, dict]]], list[dict]]:
    ordered = sorted(frame_candidates.items())
    partials: list[list[tuple[int, dict]]] = []
    completed: list[list[tuple[int, dict]]] = []
    seen_completed: set[tuple] = set()
    failed_attempts: list[dict] = []

    def _record_completed(chain: list[tuple[int, dict]]) -> None:
        if len(chain) < min_tracklet_points:
            return
        key = tuple((frame_index, detection.get("center")) for frame_index, detection in chain)
        if key in seen_completed:
            return
        seen_completed.add(key)
        completed.append(chain)

    def _record_failure(chain: list[tuple[int, dict]], reason: str, **extra) -> None:
        if len(chain) < 2:
            return
        failed_attempts.append(
            {
                "start_frame": chain[0][0],
                "end_frame": chain[-1][0],
                "point_count": len(chain),
                "reason": reason,
                **extra,
            }
        )

    for frame_index, detections in ordered:
        if not detections:
            continue
        next_partials: list[list[tuple[int, dict]]] = []

        for detection in detections:
            center = detection.get("center")
            if center is None:
                continue
            solo = [(frame_index, detection)]
            next_partials.append(solo)

            for chain in partials:
                prev_frame, prev_detection = chain[-1]
                gap = frame_index - prev_frame
                if gap < 1:
                    continue
                if gap > max_frame_gap:
                    _record_failure(
                        chain,
                        "frame_gap_too_large",
                        gap=gap,
                        max_frame_gap=max_frame_gap,
                    )
                    continue
                step = _distance(prev_detection.get("center"), center)
                if step > max_link_distance_px:
                    _record_failure(
                        chain,
                        "link_distance_exceeded",
                        step_px=round(step, 2),
                        max_link_distance_px=round(max_link_distance_px, 2),
                    )
                    continue
                extended = chain + [(frame_index, detection)]
                next_partials.append(extended)
                if len(extended) >= min_tracklet_points:
                    _record_completed(extended)

        partials = sorted(
            next_partials,
            key=_offline_partial_priority,
            reverse=True,
        )[:partial_chain_limit]

    for chain in partials:
        if len(chain) >= min_tracklet_points:
            _record_completed(chain)
        elif len(chain) >= 2:
            _record_failure(chain, "too_few_points", min_tracklet_points=min_tracklet_points)

    return completed, failed_attempts


def _offline_partial_priority(chain: list[tuple[int, dict]]) -> float:
    if len(chain) < 2:
        return float(len(chain))
    positions = [detection.get("center") for _, detection in chain]
    movement = _distance(positions[0], positions[-1])
    return len(chain) * 1000.0 + movement


def _offline_tracklet_metrics(
    chain: list[tuple[int, dict]],
    *,
    frame_width: int,
    frame_height: int,
    fps: float | None,
    include_tracklet_points: bool = False,
) -> dict:
    points = [
        {
            "frame_index": frame_index,
            "x": float(detection["center"][0]),
            "y": float(detection["center"][1]),
            "confidence": float(detection.get("confidence", 0.0) or 0.0),
        }
        for frame_index, detection in chain
    ]
    start_frame = points[0]["frame_index"]
    end_frame = points[-1]["frame_index"]
    point_count = len(points)
    frame_span = max(1, end_frame - start_frame + 1)
    total_movement = _tracklet_displacement(points)
    path_length = _tracklet_path_length(points)
    avg_movement = path_length / max(1, point_count - 1)
    smoothness = _tracklet_smoothness(points)
    edge_fraction = _tracklet_edge_fraction(points, frame_width, frame_height)
    mean_confidence = sum(point["confidence"] for point in points) / point_count
    unique_center_count = _tracklet_unique_center_count(chain)
    spatial_spread = _tracklet_spatial_spread(points)

    static_penalty = 0.0
    if total_movement < BOOTSTRAP_STATIC_MOVEMENT_PX:
        static_penalty += 40.0
    if total_movement < BOOTSTRAP_MIN_MOVEMENT_PX:
        static_penalty += 25.0
    edge_penalty = edge_fraction * 50.0
    confidence_score = mean_confidence * 12.0

    final_segment_score = (
        point_count * 12.0
        + frame_span * 0.18
        + min(total_movement, 240.0) * 0.35
        + min(path_length, 360.0) * 0.08
        + smoothness * 8.0
        + confidence_score
        - static_penalty
        - edge_penalty
    )

    duration_sec = None
    if fps and fps > 0:
        duration_sec = round(frame_span / float(fps), 3)

    result = {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "point_count": point_count,
        "frame_span": frame_span,
        "duration_sec": duration_sec,
        "total_movement": round(total_movement, 2),
        "average_movement_per_frame": round(avg_movement, 2),
        "unique_center_count": unique_center_count,
        "spatial_spread_px": round(spatial_spread, 2),
        "static_penalty": round(static_penalty, 2),
        "edge_penalty": round(edge_penalty, 2),
        "smoothness_score": round(smoothness, 3),
        "confidence_score": round(confidence_score, 3),
        "final_segment_score": round(final_segment_score, 3),
    }
    if include_tracklet_points:
        result["tracklet_points"] = [
            {
                "frame_index": point["frame_index"],
                "x": point["x"],
                "y": point["y"],
                "confidence": point.get("confidence"),
            }
            for point in points
        ]
    return result


def _offline_static_rejection_reason(
    metrics: dict,
    *,
    chain: list[tuple[int, dict]],
    min_total_movement_px: float,
    min_average_movement_px: float,
) -> str | None:
    point_count = metrics["point_count"]
    total_movement = metrics["total_movement"]
    avg_movement = metrics["average_movement_per_frame"]
    unique_center_count = metrics.get("unique_center_count") or _tracklet_unique_center_count(chain)
    spatial_spread = metrics.get("spatial_spread_px") or 0.0

    if total_movement < min_total_movement_px:
        return "insufficient_total_movement"
    if avg_movement < min_average_movement_px:
        return "insufficient_average_movement"
    if spatial_spread < OFFLINE_MIN_SPATIAL_SPREAD_PX:
        return "near_static_segment"
    if point_count >= 5:
        unique_ratio = unique_center_count / point_count
        if unique_ratio < OFFLINE_MIN_UNIQUE_CENTER_RATIO:
            return "near_static_segment"
    return None


def _tracklet_unique_center_count(chain: list[tuple[int, dict]]) -> int:
    keys = set()
    for _, detection in chain:
        center = detection.get("center")
        if center is None:
            continue
        keys.add(_grid_key(center))
    return len(keys)


def _tracklet_spatial_spread(points) -> float:
    if len(points) < 2:
        return 0.0
    max_dist = 0.0
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            max_dist = max(
                max_dist,
                hypot(
                    points[right]["x"] - points[left]["x"],
                    points[right]["y"] - points[left]["y"],
                ),
            )
    return max_dist


def _offline_segment_reason(item: dict, *, rank: int, top_n: int) -> str:
    if rank == 1:
        parts = ["selected_winner", "meaningful_movement"]
        return ";".join(parts)
    if rank <= top_n:
        return "rejected_lower_score_than_winner"
    return "rejected_outside_top_k"


def _why_no_offline_segments(
    frame_candidates: dict[int, list[dict]],
    failed_attempts: list[dict],
    *,
    rejected_static: list[dict] | None = None,
    min_tracklet_points: int,
    max_frame_gap: int,
    max_link_distance_px: float,
    min_total_movement_px: float,
    min_average_movement_px: float,
) -> dict:
    total_candidates = sum(len(detections) for detections in frame_candidates.values())
    reason_counts: Counter[str] = Counter()
    for attempt in failed_attempts:
        reason_counts[attempt.get("reason", "unknown")] += 1
    static_reason_counts: Counter[str] = Counter()
    for segment in rejected_static or []:
        static_reason_counts[segment.get("rejection_reason", "near_static_segment")] += 1

    parts = []
    if total_candidates < min_tracklet_points:
        parts.append("too_few_total_candidates")
    if not frame_candidates:
        parts.append("no_ball_candidates_after_filter")
    if rejected_static and not static_reason_counts:
        parts.append("all_segments_rejected_as_static")
    elif static_reason_counts and not reason_counts:
        parts.append("all_rankable_segments_rejected_as_static")
    if reason_counts.get("link_distance_exceeded", 0) > reason_counts.get("frame_gap_too_large", 0):
        parts.append("links_mostly_blocked_by_distance")
    elif reason_counts.get("frame_gap_too_large", 0) > 0:
        parts.append("links_mostly_blocked_by_frame_gap")
    if reason_counts.get("too_few_points", 0) > 0:
        parts.append("chains_too_short")

    return {
        "summary": ";".join(parts) if parts else "no_tracklet_met_min_points",
        "total_candidates": total_candidates,
        "candidate_frames": len(frame_candidates),
        "min_tracklet_points": min_tracklet_points,
        "max_frame_gap": max_frame_gap,
        "max_link_distance_px": round(max_link_distance_px, 2),
        "min_total_movement_px": min_total_movement_px,
        "min_average_movement_px": min_average_movement_px,
        "failure_reason_counts": dict(reason_counts.most_common(8)),
        "static_rejection_reason_counts": dict(static_reason_counts.most_common(8)),
        "rejected_static_segment_count": len(rejected_static or []),
    }


def _nearest_failed_tracklet_attempts(
    failed_attempts: list[dict],
    *,
    min_tracklet_points: int,
    limit: int,
) -> list[dict]:
    if not failed_attempts:
        return []
    ranked = sorted(
        failed_attempts,
        key=lambda item: (item.get("point_count", 0), item.get("end_frame", 0)),
        reverse=True,
    )
    deduped: list[dict] = []
    seen: set[tuple] = set()
    for attempt in ranked:
        key = (
            attempt.get("start_frame"),
            attempt.get("end_frame"),
            attempt.get("point_count"),
            attempt.get("reason"),
        )
        if key in seen:
            continue
        seen.add(key)
        item = dict(attempt)
        item["needed_more_points"] = max(0, min_tracklet_points - item.get("point_count", 0))
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def _tracklet_displacement(points) -> float:
    if len(points) < 2:
        return 0.0
    return hypot(
        points[-1]["x"] - points[0]["x"],
        points[-1]["y"] - points[0]["y"],
    )


def _tracklet_path_length(points) -> float:
    total = 0.0
    for index in range(1, len(points)):
        total += hypot(
            points[index]["x"] - points[index - 1]["x"],
            points[index]["y"] - points[index - 1]["y"],
        )
    return total


def _tracklet_smoothness(points) -> float:
    if len(points) < 3:
        return 0.0
    aligned = 0
    possible = 0
    for index in range(2, len(points)):
        ax, ay = points[index - 2]["x"], points[index - 2]["y"]
        bx, by = points[index - 1]["x"], points[index - 1]["y"]
        cx, cy = points[index]["x"], points[index]["y"]
        v1 = (bx - ax, by - ay)
        v2 = (cx - bx, cy - by)
        if hypot(*v1) <= BOOTSTRAP_STATIC_MOVEMENT_PX or hypot(*v2) <= BOOTSTRAP_STATIC_MOVEMENT_PX:
            continue
        possible += 1
        if v1[0] * v2[0] + v1[1] * v2[1] > 0:
            aligned += 1
    return 0.0 if possible == 0 else aligned / possible


def _tracklet_edge_fraction(points, frame_width, frame_height) -> float:
    margin = max(
        BOOTSTRAP_EDGE_MARGIN_MIN_PX,
        min(frame_width, frame_height) * BOOTSTRAP_EDGE_MARGIN_RATIO,
    )
    edge_count = sum(
        1
        for point in points
        if (
            point["x"] <= margin
            or point["y"] <= margin
            or point["x"] >= frame_width - margin
            or point["y"] >= frame_height - margin
        )
    )
    return edge_count / len(points)


def should_enable_online_best_tracklet(
    *,
    ball_tracking_mode: str,
    speed_mode: str,
) -> bool:
    from Backends.src.analysis.smart_pipeline import resolve_smart_mode
    from Backends.src.tracking.ball_tracking_utils import normalize_ball_tracking_mode

    return (
        normalize_ball_tracking_mode(ball_tracking_mode) == "Accuracy / Small Ball"
        or resolve_smart_mode(speed_mode) == "Debug Full Frame"
    )


def _estimate_tracklet_velocity(
    points: list[dict],
    *,
    direction: str,
    sample_count: int = EXTENSION_VELOCITY_SAMPLE_POINTS,
) -> tuple[float, float]:
    """Estimate average per-frame motion from the start or end of a tracklet."""
    ordered = sorted(points, key=lambda point: int(point["frame_index"]))
    if direction == "backward":
        sample = ordered[: max(2, min(sample_count, len(ordered)))]
    else:
        sample = ordered[-max(2, min(sample_count, len(ordered))) :]
    if len(sample) < 2:
        return (0.0, 0.0)
    vx_total = 0.0
    vy_total = 0.0
    samples = 0
    for index in range(1, len(sample)):
        gap = int(sample[index]["frame_index"]) - int(sample[index - 1]["frame_index"])
        if gap <= 0:
            continue
        vx_total += (float(sample[index]["x"]) - float(sample[index - 1]["x"])) / gap
        vy_total += (float(sample[index]["y"]) - float(sample[index - 1]["y"])) / gap
        samples += 1
    if samples == 0:
        return (0.0, 0.0)
    return (vx_total / samples, vy_total / samples)


def _point_near_frame_edge(
    x: float,
    y: float,
    frame_width: int,
    frame_height: int,
) -> bool:
    margin = max(
        BOOTSTRAP_EDGE_MARGIN_MIN_PX,
        min(frame_width, frame_height) * BOOTSTRAP_EDGE_MARGIN_RATIO,
    )
    return (
        x <= margin
        or y <= margin
        or x >= frame_width - margin
        or y >= frame_height - margin
    )


def _extension_max_distance_px(
    vx: float,
    vy: float,
    frame_gap: int,
    frame_width: int,
    frame_height: int,
) -> float:
    speed = hypot(vx, vy)
    dynamic = EXTENSION_DISTANCE_TOLERANCE_PX + speed * max(frame_gap, 1) * 1.25
    cap = default_max_link_distance_px(frame_width, frame_height) * 0.75
    return min(max(dynamic, EXTENSION_DISTANCE_TOLERANCE_PX), cap)


def _candidate_to_tracklet_point(frame_index: int, detection: dict) -> dict:
    center = detection["center"]
    return {
        "frame_index": int(frame_index),
        "x": float(center[0]),
        "y": float(center[1]),
        "confidence": float(detection.get("confidence", 0.0) or 0.0),
    }


def _pick_extension_candidate(
    candidates: list[dict],
    predicted: tuple[float, float],
    max_distance_px: float,
) -> dict | None:
    best_detection = None
    best_distance = float("inf")
    for detection in candidates or []:
        center = detection.get("center")
        if center is None:
            continue
        distance_px = _distance(center, predicted)
        if distance_px <= max_distance_px and distance_px < best_distance:
            best_detection = detection
            best_distance = distance_px
    return best_detection


def _extension_candidate_rejection(
    detection: dict,
    *,
    predicted: tuple[float, float],
    previous_point: dict | None,
    frame_width: int,
    frame_height: int,
    max_distance_px: float,
) -> str | None:
    center = detection.get("center")
    if center is None:
        return "invalid_candidate"
    if _distance(center, predicted) > max_distance_px:
        return "too_far_from_prediction"
    if previous_point is not None:
        if _distance(
            center,
            (previous_point["x"], previous_point["y"]),
        ) <= BOOTSTRAP_STATIC_MOVEMENT_PX:
            return "static_candidate"
    if _point_near_frame_edge(center[0], center[1], frame_width, frame_height):
        return "edge_noise"
    return None


def _extend_tracklet_one_direction(
    tracklet_points: list[dict],
    frame_candidates: dict[int, list[dict]],
    frame_width: int,
    frame_height: int,
    *,
    direction: str,
    max_frame_gap: int = EXTENSION_MAX_FRAME_GAP,
    max_consecutive_misses: int = EXTENSION_MAX_CONSECUTIVE_MISSES,
    max_span_frames: int = EXTENSION_MAX_SPAN_FRAMES,
) -> tuple[list[dict], Counter[str]]:
    ordered = sorted(tracklet_points, key=lambda point: int(point["frame_index"]))
    if len(ordered) < 2:
        return [], Counter()

    rejection_reasons: Counter[str] = Counter()
    extended: list[dict] = []
    vx, vy = _estimate_tracklet_velocity(ordered, direction=direction)

    if direction == "backward":
        anchor = ordered[0]
        anchor_frame = int(anchor["frame_index"])
        last_point = anchor
        last_frame = anchor_frame
        candidate_frames = range(anchor_frame - 1, anchor_frame - max_span_frames - 1, -1)
    else:
        anchor = ordered[-1]
        anchor_frame = int(anchor["frame_index"])
        last_point = anchor
        last_frame = anchor_frame
        candidate_frames = range(anchor_frame + 1, anchor_frame + max_span_frames + 1)

    consecutive_misses = 0
    for target_frame in candidate_frames:
        frame_gap = abs(target_frame - last_frame)
        if frame_gap > max_frame_gap:
            consecutive_misses += 1
            rejection_reasons["frame_gap_too_large"] += 1
            if consecutive_misses > max_consecutive_misses:
                rejection_reasons["missing_frames_exceeded"] += 1
                break
            continue

        predicted = (
            float(last_point["x"]) + vx * (target_frame - last_frame),
            float(last_point["y"]) + vy * (target_frame - last_frame),
        )
        max_distance_px = _extension_max_distance_px(
            vx,
            vy,
            frame_gap,
            frame_width,
            frame_height,
        )
        candidates = frame_candidates.get(target_frame, [])
        if not candidates:
            consecutive_misses += 1
            if consecutive_misses > max_consecutive_misses:
                rejection_reasons["missing_frames_exceeded"] += 1
                break
            continue

        detection = _pick_extension_candidate(candidates, predicted, max_distance_px)
        if detection is None:
            consecutive_misses += 1
            rejection_reasons["too_far_from_prediction"] += 1
            if consecutive_misses > max_consecutive_misses:
                rejection_reasons["missing_frames_exceeded"] += 1
                break
            continue

        rejection = _extension_candidate_rejection(
            detection,
            predicted=predicted,
            previous_point=last_point,
            frame_width=frame_width,
            frame_height=frame_height,
            max_distance_px=max_distance_px,
        )
        if rejection:
            consecutive_misses += 1
            rejection_reasons[rejection] += 1
            if consecutive_misses > max_consecutive_misses:
                rejection_reasons["missing_frames_exceeded"] += 1
                break
            continue

        point = _candidate_to_tracklet_point(target_frame, detection)
        if direction == "backward":
            extended.insert(0, point)
        else:
            extended.append(point)
        last_point = point
        last_frame = target_frame
        consecutive_misses = 0

    return extended, rejection_reasons


def _fit_quality_rank(quality: str | None) -> int:
    return {
        "Good": 3,
        "Partial": 2,
        "Medium": 2,
        "Poor": 1,
        None: 0,
    }.get(quality, 0)


def _extension_fit_quality(
    tracklet_points: list[dict],
    frame_width: int,
    frame_height: int,
    fps: float | None,
) -> str | None:
    if len(tracklet_points) < 3:
        return None
    from Backends.src.tracking.trajectory_fit import fit_delivery_trajectory

    end_frame = int(tracklet_points[-1]["frame_index"])
    fit_result = fit_delivery_trajectory(
        tracklet_points,
        frame_size=(frame_width, frame_height),
        fps=fps,
        delivery_track_terminated_frame=end_frame,
    )
    return fit_result.get("trajectory_fit_quality")


def _extension_base_result(
    ordered: list[dict],
    *,
    enabled: bool,
    original_fit_quality: str | None = None,
) -> dict:
    return {
        "extension_enabled": bool(enabled),
        "tracklet_points": ordered,
        "backward_extension_points": 0,
        "forward_extension_points": 0,
        "extended_segment_start_frame": ordered[0]["frame_index"] if ordered else None,
        "extended_segment_end_frame": ordered[-1]["frame_index"] if ordered else None,
        "extended_segment_point_count": len(ordered),
        "extension_applied": False,
        "extension_rejection_reasons": {},
        "extension_fallback_reason": None,
        "extension_preserved_original_segment": False,
        "extension_fit_delta": 0,
        "trajectory_fit_quality_after_extension": original_fit_quality,
        "original_smoothness_score": round(_tracklet_smoothness(ordered), 3),
        "extended_smoothness_score": round(_tracklet_smoothness(ordered), 3),
    }


def extend_ranked_tracklet(
    tracklet_points: list[dict],
    frame_candidates: dict[int, list[dict]],
    frame_width: int,
    frame_height: int,
    fps: float | None = None,
    *,
    enabled: bool = True,
) -> dict:
    """Conservatively extend a ranked tracklet backward and forward."""
    ordered = sorted(tracklet_points, key=lambda point: int(point["frame_index"]))
    frame_width = max(int(frame_width or 0), 1)
    frame_height = max(int(frame_height or 0), 1)
    original_fit_quality = _extension_fit_quality(
        ordered,
        frame_width,
        frame_height,
        fps,
    )
    base_result = _extension_base_result(
        ordered,
        enabled=enabled,
        original_fit_quality=original_fit_quality,
    )
    if not enabled or len(ordered) < 3:
        if not enabled:
            base_result["extension_fallback_reason"] = "disabled"
        else:
            base_result["extension_fallback_reason"] = "segment_too_short"
        base_result["extension_preserved_original_segment"] = True
        return base_result

    original_smoothness = _tracklet_smoothness(ordered)

    backward, backward_rejections = _extend_tracklet_one_direction(
        ordered,
        frame_candidates,
        frame_width,
        frame_height,
        direction="backward",
    )
    forward, forward_rejections = _extend_tracklet_one_direction(
        ordered,
        frame_candidates,
        frame_width,
        frame_height,
        direction="forward",
    )
    combined = backward + ordered + forward
    combined_smoothness = _tracklet_smoothness(combined)
    rejection_reasons = backward_rejections + forward_rejections

    if not backward and not forward:
        return {
            **base_result,
            "extension_rejection_reasons": dict(rejection_reasons.most_common()),
            "extension_fallback_reason": "no_plausible_extension",
            "extension_preserved_original_segment": True,
        }

    extended_fit_quality = _extension_fit_quality(
        combined,
        frame_width,
        frame_height,
        fps,
    )
    extension_fit_delta = (
        _fit_quality_rank(extended_fit_quality)
        - _fit_quality_rank(original_fit_quality)
    )
    smoothness_degraded = (
        original_smoothness > 0
        and combined_smoothness < original_smoothness * EXTENSION_MIN_SMOOTHNESS_RATIO
    )
    fit_degraded = extension_fit_delta < 0

    if smoothness_degraded or fit_degraded:
        fallback_reason = (
            "fit_quality_degraded"
            if fit_degraded
            else "smoothness_degraded"
        )
        return {
            **base_result,
            "extension_rejection_reasons": dict(rejection_reasons.most_common()),
            "extension_fallback_reason": fallback_reason,
            "extension_preserved_original_segment": True,
            "extension_fit_delta": extension_fit_delta,
            "trajectory_fit_quality_after_extension": original_fit_quality,
            "extended_smoothness_score": round(combined_smoothness, 3),
        }

    duration_sec = None
    start_frame = int(combined[0]["frame_index"])
    end_frame = int(combined[-1]["frame_index"])
    if fps and fps > 0:
        duration_sec = round((end_frame - start_frame + 1) / float(fps), 3)

    return {
        "extension_enabled": True,
        "tracklet_points": combined,
        "backward_extension_points": len(backward),
        "forward_extension_points": len(forward),
        "extended_segment_start_frame": start_frame,
        "extended_segment_end_frame": end_frame,
        "extended_segment_point_count": len(combined),
        "extension_applied": True,
        "extension_rejection_reasons": dict(rejection_reasons.most_common()),
        "extension_fallback_reason": None,
        "extension_preserved_original_segment": False,
        "extension_fit_delta": extension_fit_delta,
        "trajectory_fit_quality_after_extension": extended_fit_quality,
        "original_smoothness_score": round(original_smoothness, 3),
        "extended_smoothness_score": round(combined_smoothness, 3),
        "best_segment_duration_sec": duration_sec,
    }


def build_ball_positions_from_tracklet(
    tracklet_points: list[dict],
    total_frames: int,
) -> list:
    """Build per-frame ball positions with None outside the ranked tracklet."""
    positions: list = [None] * max(int(total_frames or 0), 0)
    for point in tracklet_points:
        frame_no = int(point["frame_index"])
        if 0 <= frame_no < len(positions):
            positions[frame_no] = (
                int(round(float(point["x"]))),
                int(round(float(point["y"]))),
            )
    return positions


def select_online_best_tracklet(
    frame_candidates: dict[int, list[dict]],
    frame_width: int,
    frame_height: int,
    fps: float | None = None,
    *,
    top_n: int = 10,
    max_frame_gap: int = 5,
    max_link_distance_px: float | None = None,
    min_tracklet_points: int = 3,
    min_total_movement_px: float = OFFLINE_MIN_TOTAL_MOVEMENT_PX,
    min_average_movement_px: float = OFFLINE_MIN_AVERAGE_MOVEMENT_PX,
    enable_extension: bool = True,
) -> dict:
    """Rank raw frame candidates and return the best moving tracklet, if any."""
    ranking = rank_candidate_tracklets(
        frame_candidates,
        frame_width,
        frame_height,
        fps,
        top_n=top_n,
        max_frame_gap=max_frame_gap,
        max_link_distance_px=max_link_distance_px,
        min_tracklet_points=min_tracklet_points,
        min_total_movement_px=min_total_movement_px,
        min_average_movement_px=min_average_movement_px,
    )
    winner = ranking.get("winner")
    if not winner:
        return {
            "applied": False,
            "fallback_reason": "no_valid_ranked_segment",
            "ranking": ranking,
            "tracklet_points": [],
        }

    tracklet_points = winner.get("tracklet_points") or []
    if len(tracklet_points) < min_tracklet_points:
        return {
            "applied": False,
            "fallback_reason": "ranked_segment_too_short",
            "ranking": ranking,
            "tracklet_points": tracklet_points,
        }

    extension = extend_ranked_tracklet(
        tracklet_points,
        frame_candidates,
        frame_width,
        frame_height,
        fps,
        enabled=enable_extension,
    )
    tracklet_points = extension["tracklet_points"]
    segment_end_frame = extension.get("extended_segment_end_frame")
    segment_start_frame = extension.get("extended_segment_start_frame")

    return {
        "applied": True,
        "fallback_reason": None,
        "ranking": ranking,
        "winner": winner,
        "tracklet_points": tracklet_points,
        "best_segment_start_frame": winner.get("start_frame"),
        "best_segment_end_frame": winner.get("end_frame"),
        "best_segment_point_count": winner.get("point_count"),
        "best_segment_duration_sec": winner.get("duration_sec"),
        "best_segment_score": winner.get("final_segment_score"),
        "best_segment_reason": winner.get("reason"),
        "rejected_static_segment_count": ranking.get("rejected_static_segment_count", 0),
        "candidate_segment_count": ranking.get("candidate_segment_count", 0),
        "extension_enabled": extension.get("extension_enabled", False),
        "extension_applied": extension.get("extension_applied", False),
        "backward_extension_points": extension.get("backward_extension_points", 0),
        "forward_extension_points": extension.get("forward_extension_points", 0),
        "extended_segment_start_frame": segment_start_frame,
        "extended_segment_end_frame": segment_end_frame,
        "extended_segment_point_count": extension.get(
            "extended_segment_point_count",
            len(tracklet_points),
        ),
        "extension_rejection_reasons": extension.get("extension_rejection_reasons", {}),
        "extension_fallback_reason": extension.get("extension_fallback_reason"),
        "extension_preserved_original_segment": extension.get(
            "extension_preserved_original_segment",
            not extension.get("extension_applied", False),
        ),
        "extension_fit_delta": extension.get("extension_fit_delta", 0),
        "trajectory_fit_quality_after_extension": extension.get(
            "trajectory_fit_quality_after_extension"
        ),
    }


def count_after_track_terminated_from_frame(
    frame_candidates: dict[int, list[dict]],
    tracklet_points: list[dict],
    *,
    frame_width: int,
    frame_height: int,
    segment_end_frame: int,
) -> int:
    """Count how many later candidates would hit after_track_terminated."""
    selector = TrajectoryBallSelector(frame_width, frame_height)
    selector.apply_ranked_tracklet(
        tracklet_points,
        segment_end_frame=segment_end_frame,
    )
    for frame_index in sorted(frame_candidates):
        if frame_index <= segment_end_frame:
            continue
        detections = frame_candidates.get(frame_index, [])
        if detections:
            selector.select(detections, frame_index=frame_index)
    return selector.rejection_reasons.get("after_track_terminated", 0)
