class DeliveryCapturePipeline:
    def process(self, clip_path: str) -> dict[str, object]:
        return {
            "clip_path": clip_path,
            "status": "unavailable",
            "reason": "analysis_pipeline_not_connected",
        }
