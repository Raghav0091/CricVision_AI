# CricVision AI — Development Rules

## General

- Do not rewrite the app unless explicitly requested.
- Inspect the real flow first; keep each change small and focused.
- Do not add unrelated features during cleanup.
- Preserve user-facing behavior unless the task explicitly changes UI.
- Keep old session results backward-compatible.

## Models

- Do not load YOLO when importing the Dashboard.
- Do not load `shot_classifier.keras` at startup.
- Keep model loading lazy and cached.
- Tests must not require `HF_TOKEN` or download Hugging Face models.
- Do not commit model files.

## Git

- Never use `git add .`.
- Before staging, run `git status --short`.
- Before committing, run `git diff --cached --name-only`.
- Do not commit `outputs/`, processed videos, reports, `data/session_results.json`, model files, `Models/remote/`, `.env`, `.streamlit/secrets.toml`, `__pycache__/`, `.pytest_cache/`, large videos, or `AGENTS.md` unless explicitly requested.

## Validation

Run:

```text
python -m compileall -q Backends
python -m pytest -q
python scripts/smoke_check.py
python scripts/performance_check.py
git diff --check
```

Tests use synthetic or dummy data and require no real YOLO files, GPU, camera, `HF_TOKEN`, or internet.

## UI

- Prefer summary-first UI; put technical detail in expanders or tabs.
- Do not show pitch maps, field maps, wagon wheels, or shot-placement maps in production reports.
- Keep debug overlays optional.
- State low confidence honestly.

## Agentic AI

- Computer vision and deterministic rules produce tracking, line, length, shot, and outcome facts.
- LLMs may explain structured outputs and recommend drills; they must not invent observations.
- A Coaching Agent may use structured session data only.

## Product Order

1. CricVision Engine.
2. Continuous Session Engine.
3. Fast Mode.
4. Coaching Agent.
5. Club Dashboard.

Phone, AR, and glasses apps are future clients, not the current priority.
