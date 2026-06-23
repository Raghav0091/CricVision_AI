"""Hugging Face model download helpers for Streamlit Cloud deployments.

Remote weights are downloaded lazily into ``Models/remote`` only when a caller
tries to load a model whose local weight file is missing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


HF_REPO_ID = "RAGHAV0049/cricvision-models"
REMOTE_MODEL_DIR = Path("Models/remote")
REMOTE_MODEL_FILES = {
    "cricshot_ball": "ball_detector.pt",
    "cricshot_bat": "bat_detector.pt",
    "player_type": "player_type_detector.pt",
    "striker_segmentation": "striker_bat_segmentation.pt",
    "shot_classifier": "shot_classifier.keras",
}

try:
    import streamlit as st
except ImportError:  # pragma: no cover - Streamlit is optional for CLI imports.
    st = None


def _is_running_in_streamlit() -> bool:
    if st is None:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _warn(message: str) -> None:
    if _is_running_in_streamlit():
        try:
            st.warning(message)
        except Exception:
            pass


def load_env_file() -> bool:
    """Load ``.env`` if python-dotenv is installed; never crash if it is not."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False

    try:
        return bool(load_dotenv())
    except Exception:
        return False


def get_hf_token() -> Optional[str]:
    """Read HF_TOKEN from Streamlit secrets first, then environment variables."""
    if st is not None:
        try:
            token = st.secrets.get("HF_TOKEN")
            if token:
                return str(token)
        except Exception:
            pass

    load_env_file()
    token = os.getenv("HF_TOKEN")
    return token or None


def is_remote_model_key(model_key: str) -> bool:
    return model_key in REMOTE_MODEL_FILES


def get_remote_filename(model_key: str) -> Optional[str]:
    return REMOTE_MODEL_FILES.get(model_key)


def download_remote_model(model_key: str, force_download: bool = False) -> Optional[str]:
    """Download a configured remote model and return its local path.

    Failures are intentionally soft so one unavailable remote model does not
    crash the whole Streamlit app.
    """
    filename = get_remote_filename(model_key)
    if filename is None:
        message = f"Remote model key is not configured: {model_key}"
        print(message)
        _warn(message)
        return None

    REMOTE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    local_path = REMOTE_MODEL_DIR / filename
    if local_path.is_file() and not force_download:
        return str(local_path)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        message = (
            "huggingface_hub is not installed. Add it to requirements.txt "
            "to enable remote model downloads."
        )
        print(f"{message} ({error})")
        _warn(message)
        return None

    try:
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type="model",
            token=get_hf_token(),
            local_dir=str(REMOTE_MODEL_DIR),
            force_download=force_download,
        )
        return str(downloaded_path)
    except Exception as error:
        message = (
            f"Could not download {filename} from Hugging Face. "
            "If the model repo is private, add HF_TOKEN to Streamlit secrets "
            "or your local .env file."
        )
        print(f"{message} Error: {error}")
        _warn(message)
        return None


def get_remote_model_status() -> dict:
    """Report configured remote files and local cache state without downloading."""
    return {
        model_key: {
            "remote_key": model_key,
            "filename": filename,
            "repo_id": HF_REPO_ID,
            "configured": True,
            "local_cache_path": str(REMOTE_MODEL_DIR / filename),
            "local_cache_exists": (REMOTE_MODEL_DIR / filename).is_file(),
        }
        for model_key, filename in REMOTE_MODEL_FILES.items()
    }
