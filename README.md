# CricVision AI

CricVision is migrating from a working Streamlit prototype to a separated web/API/worker architecture. The migration is incremental: the Streamlit application remains the current feature-complete development surface.

## Current application: Streamlit

```powershell
cd C:\CricVision_AI
streamlit run main.py
```

`main.py` and `Backends/` are active. They currently provide Dashboard, Live Session, Video Analysis, Session Results, model loading, calibration, tracking, reports, and processed-video support.

## Future frontend: Next.js

```powershell
cd C:\CricVision_AI\apps\web
npm install
npm run dev
```

Open `http://localhost:3000`. This frontend is a runnable architecture foundation; live delivery capture and analysis are not complete.

## Future API: FastAPI

From the repository root:

```powershell
uv pip install -r services\api\requirements.txt
python -m uvicorn services.api.main:app --reload
```

Open `http://localhost:8000/health` or `http://localhost:8000/docs`. The API uses local/process-only storage and returns an honest unavailable response while a dedicated stump detector is missing.

## Project structure

| Path | Role | Maturity |
|---|---|---|
| `main.py` | Streamlit entry point | Current |
| `Backends/` | Streamlit UI and active CV/analysis implementation | Legacy active |
| `apps/web/` | Next.js browser frontend | Future scaffold |
| `services/api/` | FastAPI control plane | Future scaffold |
| `services/worker/` | Background processing boundary | Future scaffold, unconnected |
| `packages/cricket_vision/` | Framework-independent shared CV contracts/geometry | Future scaffold |
| `Models/` | Local model weights | Active assets |
| `scripts/` | Diagnostics, replay, smoke, and performance tools | Active development |
| `tests/` | Lightweight regression tests for current code | Active development |
| `docs/` | Current architecture, specifications, audits, and roadmap | Documentation |

See `docs/CLEANUP_AUDIT.md` for the import evidence behind this structure and `docs/ARCHITECTURE_V2.md` for the migration direction.
