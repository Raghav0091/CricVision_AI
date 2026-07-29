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

| Path | Role | Maturity |
|---|---|---|
| `main.py` | Historical Streamlit entry point | Legacy |
| `Backends/` | Historical Streamlit implementation | Legacy |
| `apps/web/` | Normal Next.js frontend | Current |
| `services/api/` | Normal FastAPI backend | Current |
| `services/worker/` | Background processing boundary | Future scaffold, unconnected |
| `packages/cricket_vision/` | Framework-independent shared CV contracts/geometry | Future scaffold |
| `Models/` | Local model weights | Active assets |
| `scripts/` | Diagnostics, replay, smoke, and performance tools | Active development |
| `tests/` | Lightweight regression tests for current code | Active development |
| `docs/` | Current architecture, specifications, audits, and roadmap | Documentation |

See `docs/CLEANUP_AUDIT.md` for the import evidence behind this structure and `docs/ARCHITECTURE_V2.md` for the migration direction.
