"""Project-root paths that do not depend on the process working directory."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
SESSION_RESULTS_FILE = DATA_DIR / "session_results.json"
SESSION_CLIPS_DIR = DATA_DIR / "clips"

MODELS_DIR = PROJECT_ROOT / "Models"
BALL_MODEL_PATH = MODELS_DIR / "ball_detector" / "best.pt"
CRICKET_OBJECTS_MODEL_PATH = MODELS_DIR / "cricket_objects" / "best.pt"
EXTERNAL_BALL_MODEL_PATH = MODELS_DIR / "cricket_objects" / "best_external.pt"
REMOTE_MODEL_DIR = MODELS_DIR / "remote"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VIDEO_ANALYSIS_OUTPUT_DIR = OUTPUTS_DIR / "video_analysis"
PROCESSED_VIDEO_DIR = OUTPUTS_DIR / "processed_videos"
REPORTS_DIR = OUTPUTS_DIR / "reports"
REVIEW_FRAMES_DIR = OUTPUTS_DIR / "review_frames"
FIELD_SETUPS_DIR = OUTPUTS_DIR / "field_setups"
FIELD_SETUP_PATH = FIELD_SETUPS_DIR / "latest_field_setup.json"
FIELD_ANALYSIS_HISTORY_PATH = VIDEO_ANALYSIS_OUTPUT_DIR / "field_analysis_history.csv"
