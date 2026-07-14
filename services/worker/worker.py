from dataclasses import dataclass

from .pipeline.delivery_capture_pipeline import DeliveryCapturePipeline


@dataclass
class Worker:
    """Future queue consumer; synchronous entrypoint is enough for the scaffold."""

    capture_pipeline: DeliveryCapturePipeline

    def handle_delivery(self, clip_path: str) -> dict[str, object]:
        return self.capture_pipeline.process(clip_path)
