class ReplayPipeline:
    def render(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        return {"status": "unavailable", "replay_path": None}
