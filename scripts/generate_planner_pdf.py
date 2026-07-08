#!/usr/bin/env python3
"""One-shot generator for docs/CricVision_App_Working_Planner.pdf."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "CricVision_App_Working_Planner.pdf"


def _build_sections() -> list[tuple[str, list[str]]]:
    return [
        (
            "1. App Overview and Purpose",
            [
                "CricVision AI is a Streamlit-based cricket performance analysis application.",
                "It helps coaches and players analyze individual deliveries from uploaded video clips",
                "or short live webcam recordings. The app detects ball, bat, and stump objects using",
                "YOLO models, builds a per-frame detection timeline, repairs short tracking gaps,",
                "and produces coach-oriented reports: impact, shot type, direction, outcome, and",
                "agent-style coaching notes.",
                "",
                "Public MVP scope (branch stable-clean-ui-8ca838e):",
                "  - Dashboard: system status and quick navigation",
                "  - Live Session: webcam delivery capture and analysis",
                "  - Video Analysis: upload a clip, run smart pipeline, review reports",
                "  - Session Results: browse and filter saved delivery history",
                "",
                "Out of scope on this branch: auto exercise detection, multiplayer, FastAPI backend,",
                "database-backed sessions, model training UI (dev pages gated), and map rendering",
                "in production reports.",
            ],
        ),
        (
            "2. How to Run",
            [
                "Prerequisites:",
                "  - Python 3.11+ (tested on 3.11 in CI; local 3.13 works)",
                "  - Webcam (optional, for Live Session only)",
                "  - Model weights under Models/ (see Section 6)",
                "",
                "Setup:",
                "  cd CricVision_AI",
                "  python -m venv .venv",
                "  .venv\\Scripts\\activate          (Windows)",
                "  source .venv/bin/activate         (macOS/Linux)",
                "  pip install -r requirements.txt",
                "",
                "Run the app:",
                "  streamlit run main.py",
                "",
                "Open the URL shown (default http://localhost:8501). Dashboard loads without",
                "loading heavy models; weights load lazily on first analysis.",
                "",
                "Optional Hugging Face fallback (Streamlit Cloud / missing local weights):",
                "  - Set HF_TOKEN in .streamlit/secrets.toml or environment",
                "  - Repo: RAGHAV0049/cricvision-models",
                "  - Downloads go to Models/remote/ on first use",
                "",
                "Smoke / dev checks (no models required):",
                "  python -m compileall -q Backends",
                "  python scripts/smoke_check.py",
                "  pytest",
            ],
        ),
        (
            "3. User-Facing Features",
            [
                "3.1 Dashboard (Backends/src/ui/dashboard.py)",
                "  - Quick actions: Start Live Session, Analyze Video",
                "  - Metrics: reports count, last analysis, active model, system readiness",
                "  - Model readiness from model_registry.validate_model_paths()",
                "",
                "3.2 Live Session (Backends/src/ui/live_session.py)",
                "  - Webcam via streamlit-webrtc",
                "  - Record a short delivery clip (up to ~450 frames)",
                "  - Ball + Stump Detector or Ensemble (if multiple weights exist)",
                "  - Detection presets: Fast Bowling / Balanced / High Precision",
                "  - Field setup card: batter handedness, fielders context",
                "  - Kalman smoothing and bounce heuristics on live trajectory",
                "  - Saves delivery report to session JSON on success",
                "",
                "3.3 Video Analysis (Backends/src/ui/video_analysis.py)",
                "  - Upload mp4/mov/avi/mkv delivery clip",
                "  - Practice Environment Calibration (optional 2D stump/pitch context)",
                "  - Analysis modes: Smart Balanced (default), Smart Accurate, Debug Full Frame",
                "  - Processed video preview with Clean or Debug overlay",
                "  - Advanced: Bowling / Batting / Full Delivery analysis, model picker,",
                "    detection presets, pitch ROI, manual corner calibration",
                "  - Results UI: Processed Video, Quick Summary, tabbed detail cards",
                "    (Summary, Tracking Quality, Impact & Shot, Calibration, Technical)",
                "",
                "3.4 Session Results (Backends/src/ui/results_page.py)",
                "  - Loads data/session_results.json",
                "  - Summary analytics and filters",
                "  - Expandable per-delivery cards with full report fields",
                "  - Backward compatible with older saved records",
            ],
        ),
        (
            "4. Architecture and Data Flow",
            [
                "Layered architecture:",
                "",
                "  [Streamlit UI]  main.py -> dashboard | live_session | video_analysis | results",
                "        |",
                "  [Orchestration]  video_analysis.py / live_session.py frame loops",
                "        |",
                "  [Video pipeline]  video_reader -> detection_pipeline -> annotation_writer",
                "        |                              |",
                "        |                    YOLO models (lazy load)",
                "        v",
                "  [Reports]  report_pipeline -> tracking repair -> observer timeline",
                "               -> impact -> shot -> direction -> outcome -> agent enrichment",
                "        |",
                "  [Storage]  session_store.py -> data/session_results.json",
                "",
                "Text data-flow (uploaded video):",
                "",
                "  User uploads clip",
                "       -> video_reader opens and iterates frames",
                "       -> detection_pipeline runs YOLO per smart_pipeline stride/ROI rules",
                "       -> frame_detections timeline built (ball, bat, stump per frame)",
                "       -> calibration_context refined from stump detections",
                "       -> report_pipeline: Visual Observer 2D repair",
                "       -> impact_detection, shot_classification, shot_direction,",
                "          outcome_prediction, delivery_enrichment, vision_agent",
                "       -> annotation_writer optional processed MP4 + review frames",
                "       -> UI renders summary + tabs; user may save to session JSON",
                "",
                "Key design rule: one shared detection timeline feeds all reports.",
                "No duplicate YOLO passes for separate report modules.",
            ],
        ),
        (
            "5. File and Folder Map",
            [
                "Entry and UI:",
                "  main.py                    Streamlit entry, page routing, SHOW_DEV_PAGES gate",
                "  Backends/src/ui/dashboard.py",
                "  Backends/src/ui/live_session.py",
                "  Backends/src/ui/video_analysis.py   (largest page; frame loops)",
                "  Backends/src/ui/results_page.py",
                "  Backends/src/ui/components.py     shared cards, summary, filters",
                "  Backends/src/ui/analysis_helpers.py session persist helpers",
                "  Backends/src/ui/theme.py            sidebar nav, global styling",
                "  Backends/src/ui/interactive_field_map.py  field setup card",
                "",
                "Video pipeline:",
                "  video_pipeline/video_reader.py      open, iterate, write frames",
                "  video_pipeline/detection_pipeline.py models, ROI, ensemble, calibration",
                "  video_pipeline/report_pipeline.py   report orchestration",
                "  video_pipeline/annotation_writer.py overlays, MP4 conversion",
                "  video_pipeline/performance_timer.py timing schema",
                "",
                "Analysis and agents:",
                "  analysis/smart_pipeline.py          Smart Balanced/Accurate/Debug settings",
                "  analysis/frame_detection_utils.py   timeline normalization",
                "  analysis/impact_detection.py        bat-ball impact frame",
                "  analysis/shot_classification.py     shot type heuristics",
                "  analysis/shot_direction.py          direction zone",
                "  analysis/outcome_prediction.py      predicted outcome",
                "  analysis/delivery_enrichment.py       post-shot pipeline glue",
                "  analysis/cricket_agent.py             coaching feedback",
                "  agents/tracking_repair_agent.py       2D gap repair",
                "  agents/visual_observer_agent.py       repair confidence/decision",
                "  agents/observer_timeline.py           coverage summary",
                "  agents/vision_agent.py                consistency review",
                "",
                "Calibration (model-free, deterministic):",
                "  calibration/calibration_context.py",
                "  calibration/stump_calibration.py",
                "  calibration/pitch_calibration.py",
                "",
                "Models and config:",
                "  models/model_registry.py            central weight registry",
                "  models/model_loader.py              cached lazy YOLO/Keras",
                "  models/remote_model_loader.py       Hugging Face fallback",
                "  config/paths.py                     project-root paths",
                "  config/constants.py                 detection presets, thresholds",
                "",
                "Storage and outputs:",
                "  storage/session_store.py            JSON session persistence",
                "  data/session_results.json             saved deliveries",
                "  data/clips/                         optional saved clips",
                "  outputs/video_analysis/             analysis artifacts",
                "  outputs/processed_videos/           annotated MP4 previews",
                "  outputs/reports/                    text/JSON report files",
                "  outputs/review_frames/              low-confidence frame exports",
                "  Models/                             YOLO/Keras weight files",
            ],
        ),
        (
            "6. Models Used Where",
            [
                "Registry keys (models/model_registry.py):",
                "",
                "  current_best (default)",
                "    Path: Models/cricket_objects/best.pt",
                "    Classes: ball, stump | Used: Video Analysis, Live Session default",
                "",
                "  cricshot_ball",
                "    Path: Models/CricShot10k/ball_detector.pt",
                "    Used: Batting Analysis ball model; ensemble member",
                "",
                "  cricshot_bat",
                "    Path: Models/CricShot10k/bat_detector.pt",
                "    Used: Batting Analysis, Full Delivery Analysis",
                "",
                "  player_type (experimental, not wired)",
                "  striker_segmentation (experimental, not wired)",
                "  shot_classifier.keras (lazy_only, not wired to UI reports)",
                "",
                "Legacy weights kept on disk (ensemble/advanced only):",
                "  Models/ball_detector/best.pt",
                "  Models/cricket_objects/best_external.pt",
                "",
                "Loading behavior:",
                "  - No YOLO/TensorFlow at app startup",
                "  - get_cached_yolo_model() on first analysis action",
                "  - Local file first; HF download if remote_key set and token present",
                "  - shot_classifier.keras never loads unless explicitly wired (future)",
            ],
        ),
        (
            "7. Analysis Modes and Settings",
            [
                "7.1 Smart speed modes (video upload, default path)",
                "  Smart Balanced (default): ball every frame, bat stride 2, ROI on,",
                "    resize 854px, yolo 640, light annotation, single-pass same model",
                "  Smart Accurate: all models every frame, resize 960, ROI on, full annotation",
                "  Debug Full Frame: no ROI, full frame 960, all strides 1, debug overlays",
                "",
                "7.2 Analysis type (Advanced expander)",
                "  Bowling Analysis: ball + stump detection focus",
                "  Batting Analysis: separate ball + bat models (CricShot10k)",
                "  Full Delivery Analysis: ball/stump + bat model combined reports",
                "",
                "7.3 Detection presets (config/constants.py)",
                "  Fast Bowling Mode:   imgsz 960, confidence 0.15",
                "  Balanced Mode:       imgsz 768, confidence 0.25  (default)",
                "  High Precision Mode: imgsz 960, confidence 0.35",
                "",
                "7.4 Overlay and output toggles",
                "  Generate processed video preview (on by default)",
                "  Overlay detail: Clean (default) vs Debug (ROI, bounce, labels)",
                "  Practice calibration: camera view, handedness, auto-estimate stumps",
                "  Pitch calibration: auto from stumps or manual 4-corner points",
                "",
                "7.5 Live Session presets",
                "  Same DETECTION_PRESETS as video; model Ball+Stump or Ensemble",
            ],
        ),
        (
            "8. Tracking Approach on This Branch",
            [
                "This branch (stable-clean-ui-8ca838e) uses deterministic 2D tracking repair,",
                "not true 3D multi-camera tracking.",
                "",
                "Video Analysis pipeline:",
                "  1. YOLO produces per-frame ball/bat/stump detections",
                "  2. tracking_repair_agent extracts ball path across frames",
                "  3. Flags missing points, low confidence, implausible jumps",
                "  4. Repairs only short bounded gaps; marks source=observer_repair",
                "  5. visual_observer_agent summarizes repair confidence and decision",
                "  6. Reports consume repaired timeline; raw kept for comparison",
                "  7. Failure falls back to raw detections (analysis still completes)",
                "",
                "Live Session (separate path):",
                "  - BallKalmanTracker + interpolate_missing_positions + smooth_trajectory",
                "  - Bounce detection via direction change heuristic",
                "  - Visual Observer repair NOT yet integrated in Live Session",
                "    (planned after real-video validation per refactor_notes.md)",
                "",
                "Constants (config/constants.py):",
                "  LOW_CONFIDENCE_REVIEW_THRESHOLD = 0.35",
                "  MAX_REASONABLE_BALL_JUMP_PX = 180",
                "",
                "Explicit non-goals on this branch:",
                "  - No metric 3D ball tracking",
                "  - No AR Nets hardware integration",
                "  - No production wagon-wheel map rendering",
            ],
        ),
        (
            "9. Data Outputs and Storage Paths",
            [
                "Session persistence:",
                "  data/session_results.json     all saved delivery records (JSON array)",
                "  data/clips/                   optional clip copies linked to sessions",
                "",
                "Generated artifacts (per analysis run):",
                "  outputs/processed_videos/     annotated MP4 (browser-safe conversion)",
                "  outputs/reports/              exported report files",
                "  outputs/review_frames/        frames flagged for low confidence",
                "  outputs/video_analysis/       misc analysis output, field history CSV",
                "  outputs/field_setups/         latest_field_setup.json",
                "",
                "Remote model cache:",
                "  Models/remote/                HF-downloaded weights",
                "",
                "Session record fields (normalized by session_store.py):",
                "  id, created_at, analysis_mode, model info, delivery_report,",
                "  impact_result, shot_result, direction_result, outcome_result,",
                "  agent_result, calibration_context, visual_observer_repair summary,",
                "  processed_video_path, performance profile",
                "",
                "Note: no automatic retention policy; outputs accumulate until manual cleanup.",
                "Do not delete files still referenced by saved session records.",
            ],
        ),
        (
            "10. Branch Context: stable-clean-ui vs main",
            [
                "Current branch: stable-clean-ui-8ca838e (commit 8ca838e)",
                "Theme: Clean Video Analysis results UI + stable public MVP",
                "",
                "Recent focus on this branch:",
                "  - Tabbed result cards (Summary, Tracking, Impact, Calibration, Technical)",
                "  - render_analysis_summary_card() coach-oriented quick summary",
                "  - Processed video validation before st.video() preview",
                "  - Clean overlay default; Debug overlay optional",
                "  - Calibration quality capped so estimates cannot show as High",
                "  - Practice Environment Calibration Part 1 (2D context, no map)",
                "  - Visual Observer 2D ball tracking repair in Video Analysis",
                "  - Video pipeline extraction (reader, detection, report, annotation)",
                "",
                "Compared to experimental/main tracking work:",
                "  - This branch prioritizes stable UI, lazy models, and text/card reports",
                "  - Map rendering intentionally removed from production reports",
                "  - Live Session still uses legacy Kalman path; observer repair is video-only",
                "  - Large frame loops remain in UI modules (known debt; safe for MVP)",
                "  - Dev pages (Field Map, Datasets, Training) hidden via SHOW_DEV_PAGES=False",
            ],
        ),
        (
            "11. Limitations and Known Gaps",
            [
                "Accuracy and environment:",
                "  - 2D pixel tracking; line/length are approximate heuristics",
                "  - Poor lighting, occlusion, or non-standard camera angles reduce quality",
                "  - Practice calibration is estimated 2D context, not true 3D pitch model",
                "",
                "Architecture / maintenance:",
                "  - video_analysis.py (~1900 lines) and live_session.py (~1300 lines) are monolithic",
                "  - Frame loops not yet moved fully into video_pipeline package",
                "  - No output retention or disk quota management",
                "  - Local JSON sessions: no multi-user locking or cloud sync",
                "",
                "Models:",
                "  - shot_classifier.keras, player_type, striker_segmentation registered but unwired",
                "  - Ensemble requires multiple legacy weight files on disk",
                "  - Real YOLO accuracy validation is manual/local only",
                "",
                "Features not in production:",
                "  - Wagon wheel / field map in reports (history CSV only, internal)",
                "  - Visual Observer repair in Live Session (pending validation)",
                "  - True 3D tracking, AR nets mode, virtual fielders",
                "  - Auto exercise / fitness MVP (separate deprecated path in AGENTS.md)",
            ],
        ),
        (
            "12. Suggested User Workflow (Step-by-Step Planner)",
            [
                "Phase A - Setup (one time)",
                "  1. Clone repo and install requirements.txt",
                "  2. Verify Models/cricket_objects/best.pt exists (Dashboard: System Ready)",
                "  3. Optional: place CricShot10k weights for batting/full delivery modes",
                "  4. Optional: set HF_TOKEN for cloud deploy without local weights",
                "  5. Run: streamlit run main.py",
                "",
                "Phase B - Calibrate practice context (per session)",
                "  1. Open Video Analysis",
                "  2. Set field setup: batter handedness and fielders if relevant",
                "  3. Enable Practice Environment Calibration",
                "  4. Choose camera view (Umpire End default) and confirm calibration",
                "  5. Leave auto-estimate stumps enabled unless manual corners needed",
                "",
                "Phase C - Analyze a delivery",
                "  1. Upload a short single-delivery clip (side-on or umpire view works best)",
                "  2. Keep Smart Balanced unless you need maximum accuracy (Smart Accurate)",
                "  3. Click Analyze Delivery; wait for pipeline (models load on first run)",
                "  4. Review Quick Result Summary and Processed Video Preview",
                "  5. Open tabs: Tracking Quality, Impact & Shot, Calibration, Technical Details",
                "",
                "Phase D - Review and save",
                "  1. Check impact frame, shot type, direction, outcome, agent notes",
                "  2. Note Visual Observer repair confidence in Tracking Quality tab",
                "  3. Save delivery to session history when satisfied",
                "  4. Open Session Results to compare deliveries over time",
                "",
                "Phase E - Live practice (optional)",
                "  1. Open Live Session from Dashboard",
                "  2. Allow webcam permissions; select model and preset",
                "  3. Record one delivery; review inline report",
                "  4. Saved results appear in Session Results alongside video analyses",
                "",
                "Phase F - Maintenance",
                "  1. Periodically archive or prune outputs/ if disk space is low",
                "  2. Keep data/session_results.json backed up",
                "  3. Re-run python scripts/smoke_check.py after upgrades",
            ],
        ),
        (
            "Document Info",
            [
                "Title: CricVision AI - App Working Planner",
                "Branch: stable-clean-ui-8ca838e (commit 8ca838e)",
                "Generated for: CricVision AI cricket Streamlit MVP",
                "Entrypoint: streamlit run main.py",
            ],
        ),
    ]


def generate_pdf(output_path: Path) -> int:
    from fpdf import FPDF

    class PlannerPDF(FPDF):
        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 8, f"Page {self.page_no()}", align="C")

    pdf = PlannerPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 18, 18)

    ew = pdf.w - pdf.l_margin - pdf.r_margin

    def write_title(text: str):
        pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(ew, 9, text)
        pdf.ln(2)

    def write_heading(text: str):
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(ew, 7, text)
        pdf.ln(1)

    def write_body(lines: list[str]):
        pdf.set_font("Helvetica", "", 10)
        for line in lines:
            if line == "":
                pdf.ln(3)
            else:
                pdf.multi_cell(ew, 5, line)
        pdf.ln(1)

    # Cover page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.ln(25)
    pdf.multi_cell(ew, 12, "CricVision AI")
    pdf.set_font("Helvetica", "", 14)
    pdf.multi_cell(ew, 8, "App Working Planner")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(ew, 6, "Complete guide: setup, architecture, models, tracking,")
    pdf.multi_cell(ew, 6, "analysis modes, outputs, and recommended workflow.")
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(ew, 5, "Branch: stable-clean-ui-8ca838e")
    pdf.multi_cell(ew, 5, "Commit: 8ca838e - Clean video analysis results UI")

    # Table of contents
    pdf.add_page()
    write_title("Table of Contents")
    pdf.set_font("Helvetica", "", 10)
    for heading, _ in _build_sections():
        pdf.multi_cell(ew, 6, heading)
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(ew, 5, "Tip: Use your PDF reader outline/bookmarks if available.")

    # Sections
    for heading, lines in _build_sections():
        pdf.add_page()
        write_heading(heading)
        write_body(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    return pdf.page_no()


def main() -> int:
    try:
        pages = generate_pdf(OUTPUT)
    except Exception as exc:
        print(f"PDF generation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {OUTPUT} ({pages} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
