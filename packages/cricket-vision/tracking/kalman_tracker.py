class KalmanTracker:
    """Reserved adapter for a future measured-state tracker."""

    def update(self, *_args: object, **_kwargs: object) -> None:
        raise NotImplementedError("Kalman tracking is not implemented yet.")
