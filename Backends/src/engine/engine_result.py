"""Stable result contract for delivery analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from Backends.src.calibration.calibration_context import (
    normalize_calibration_context,
)


@dataclass(slots=True)
class EngineResult:
    """Structured result that retains the legacy flat result for the UI."""

    success: bool = False
    report_result: dict = field(default_factory=dict)
    processed_video_path: str | None = None
    processed_video_validation: dict = field(default_factory=dict)
    calibration_context: dict = field(
        default_factory=normalize_calibration_context
    )
    visual_observer_summary: dict = field(default_factory=dict)
    delivery_report: dict = field(default_factory=dict)
    impact_report: dict = field(default_factory=dict)
    shot_report: dict = field(default_factory=dict)
    outcome_prediction: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    pipeline_result: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_pipeline_result(
        cls,
        result=None,
        *,
        calibration_context=None,
        warnings=None,
        errors=None,
    ) -> "EngineResult":
        raw = dict(result) if isinstance(result, Mapping) else {}
        success = bool(raw.get("success", False))

        normalized_calibration = normalize_calibration_context(
            raw.get("calibration_context", calibration_context)
        )
        processed_path = raw.get("processed_video_path") or raw.get(
            "output_path"
        )
        if processed_path:
            processed_path = str(processed_path)

        delivery_report = _mapping(raw.get("delivery_report"))
        if not delivery_report:
            delivery_report = {
                key: raw.get(key)
                for key in (
                    "estimated_line",
                    "estimated_length",
                    "ball_detection_rate",
                    "ball_tracking_rate",
                    "overall_tracking_quality",
                    "calibration_status",
                    "calibration_warning",
                )
                if key in raw
            }

        impact_report = _mapping(
            raw.get("impact_report") or raw.get("impact_info")
        )
        shot_report = _mapping(raw.get("shot_report") or raw.get("shot_info"))
        outcome_prediction = _mapping(
            raw.get("outcome_prediction") or raw.get("outcome_info")
        )
        visual_observer = _mapping(
            raw.get("visual_observer_summary")
            or raw.get("visual_observer_repair")
        )
        timings = _mapping(raw.get("timings") or raw.get("performance_profile"))

        result_warnings = _string_list(raw.get("warnings"))
        result_warnings.extend(_string_list(warnings))
        result_errors = _string_list(raw.get("errors"))
        result_errors.extend(_string_list(errors))
        if not success and raw.get("error"):
            result_errors.insert(0, str(raw["error"]))

        report_result = _mapping(raw.get("report_result"))
        if not report_result:
            report_result = {
                "delivery_report": delivery_report,
                "impact_report": impact_report,
                "shot_report": shot_report,
                "direction_report": _mapping(raw.get("direction_info")),
                "outcome_prediction": outcome_prediction,
                "agent_review": _mapping(raw.get("agent_info")),
                "observer_timeline": _mapping(raw.get("observer_timeline")),
                "visual_observer_summary": visual_observer,
            }

        return cls(
            success=success,
            report_result=report_result,
            processed_video_path=processed_path,
            processed_video_validation=_mapping(
                raw.get("processed_video_validation")
            ),
            calibration_context=normalized_calibration,
            visual_observer_summary=visual_observer,
            delivery_report=delivery_report,
            impact_report=impact_report,
            shot_report=shot_report,
            outcome_prediction=outcome_prediction,
            timings=timings,
            warnings=_unique(result_warnings),
            errors=_unique(result_errors),
            pipeline_result=raw,
        )

    @classmethod
    def failure(
        cls,
        message,
        *,
        calibration_context=None,
        warnings=None,
    ) -> "EngineResult":
        message = str(message or "Delivery analysis failed.")
        return cls.from_pipeline_result(
            {"success": False, "error": message},
            calibration_context=calibration_context,
            warnings=warnings,
            errors=[message],
        )

    def to_dict(self) -> dict:
        result = dict(self.pipeline_result)
        result.update(
            {
                "success": self.success,
                "report_result": self.report_result,
                "processed_video_path": self.processed_video_path,
                "processed_video_validation": self.processed_video_validation,
                "calibration_context": self.calibration_context,
                "visual_observer_summary": self.visual_observer_summary,
                "delivery_report": self.delivery_report,
                "impact_report": self.impact_report,
                "shot_report": self.shot_report,
                "outcome_prediction": self.outcome_prediction,
                "timings": self.timings,
                "warnings": list(self.warnings),
                "errors": list(self.errors),
            }
        )
        if self.errors:
            result.setdefault("error", self.errors[0])
        return result


def _mapping(value) -> dict:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, (list, tuple, set)):
        return [str(value)]
    return [str(item) for item in value if str(item).strip()]


def _unique(items) -> list[str]:
    return list(dict.fromkeys(items))
