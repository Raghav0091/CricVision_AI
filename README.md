# CricVision AI

CricVision's current development application is one Next.js frontend connected
to one FastAPI backend. Physics Engine V1 and Virtual Pitch V1 are modules in
that backend, not separate services.

## Windows local development

CricVision needs two Command Prompt windows.

**Command Prompt 1 - backend**

```bat
cd /d C:\CricVision_AI
.venv\Scripts\activate
python -m uvicorn services.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Backend checks:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

**Command Prompt 2 - frontend**

```bat
cd /d C:\CricVision_AI\apps\web
set NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
npm run dev
```

Open `http://localhost:3000/video-analysis`. Activate only `.venv`;
`.venv_pose` is not used. Press `Ctrl+C` in each terminal to stop its server.
If port 8000 is occupied, close the old backend terminal before starting
another backend.

## Project structure

| Path | Role |
|---|---|
| `apps/web/` | Next.js frontend (video analysis, virtual pitch replay) |
| `services/api/` | FastAPI backend |
| `packages/cricket_vision/` | Shared pitch geometry and CV contracts |
| `Models/` | Local model weights (ball detector, stump detector) |
| `outputs/` | Generated analysis artifacts (gitignored) |
| `tests/` | Regression tests for API and video analysis |