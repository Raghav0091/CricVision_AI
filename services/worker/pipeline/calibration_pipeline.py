class CalibrationPipeline:
    def solve(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "success": False,
            "quality": "Unavailable",
            "reason": "stump_detector_missing",
            "message": "Dedicated stump detector not available yet.",
        }
