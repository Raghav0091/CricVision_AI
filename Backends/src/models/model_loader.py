"""Lazy model loading with local-first, Hugging Face-backed resolution."""

from functools import lru_cache
from pathlib import Path

from Backends.src.models.model_registry import get_model_info, get_model_path
from Backends.src.models.remote_model_loader import download_remote_model

try:
    import streamlit as st
except ImportError:  # Allows registry/analysis utilities to work outside Streamlit.
    st = None


def _warn(message: str) -> None:
    if st is not None:
        try:
            st.warning(message)
            return
        except Exception:
            pass
    print(message)


def resolve_model_path(model_key: str, allow_remote: bool = True):
    """Resolve a registered model path, downloading the remote fallback on use.

    Local weights always win. Remote downloads are attempted only when the local
    file is missing and the registry entry declares a ``remote_key``.
    """
    info = get_model_info(model_key)
    if info is None:
        _warn(f"Unknown model key: {model_key}")
        return None

    local_path = get_model_path(model_key)
    if local_path is not None and local_path.is_file():
        return local_path

    remote_key = info.get("remote_key")
    if allow_remote and remote_key:
        downloaded_path = download_remote_model(remote_key)
        if downloaded_path:
            path = Path(downloaded_path)
            if path.is_file():
                return path

    return None


def load_yolo_model(model_key: str):
    """Load one registered YOLO model, returning ``None`` on a safe failure."""
    info = get_model_info(model_key)
    if info is None:
        _warn(f"Unknown model key: {model_key}")
        return None
    if info.get("type") not in {"yolo_detection", "yolo_segmentation"}:
        _warn(f"{info['name']} is not a YOLO model and cannot be loaded here.")
        return None

    model_path = resolve_model_path(model_key, allow_remote=True)
    if model_path is None:
        _warn(
            f"{info['name']} is unavailable. Add the local model file or configure "
            "HF_TOKEN if the Hugging Face repo is private."
        )
        return None

    try:
        from ultralytics import YOLO

        return YOLO(str(model_path))
    except Exception as error:
        _warn(f"Could not load {info['name']}: {error}")
        return None


@lru_cache(maxsize=None)
def _cached_yolo_model(model_key: str):
    return load_yolo_model(model_key)


if st is not None:
    _cached_yolo_model = st.cache_resource(show_spinner=False)(_cached_yolo_model)


def get_cached_yolo_model(model_key: str):
    """Load a model on first use and reuse it for later analyses."""
    return _cached_yolo_model(model_key)


def load_keras_model(model_key: str):
    """Load a Keras model lazily; TensorFlow is imported only inside this call."""
    info = get_model_info(model_key)
    if info is None:
        _warn(f"Unknown model key: {model_key}")
        return None
    if info.get("type") != "keras_sequence_classifier":
        _warn(f"{info['name']} is not a Keras model and cannot be loaded here.")
        return None

    model_path = resolve_model_path(model_key, allow_remote=True)
    if model_path is None:
        _warn(
            f"{info['name']} is unavailable. Add the local model file or configure "
            "HF_TOKEN if the Hugging Face repo is private."
        )
        return None

    try:
        from tensorflow import keras
    except Exception as error:
        _warn(f"TensorFlow/Keras is not installed, so {info['name']} cannot load: {error}")
        return None

    try:
        return keras.models.load_model(str(model_path))
    except Exception as error:
        _warn(f"Could not load {info['name']}: {error}")
        return None


@lru_cache(maxsize=None)
def _cached_keras_model(model_key: str):
    return load_keras_model(model_key)


if st is not None:
    _cached_keras_model = st.cache_resource(show_spinner=False)(_cached_keras_model)


def get_cached_keras_model(model_key: str):
    """Load a Keras model on first explicit use; never called at app startup."""
    return _cached_keras_model(model_key)


def clear_model_cache() -> None:
    """Clear only the multi-model loader caches."""
    for cached_loader in (_cached_yolo_model, _cached_keras_model):
        clear = getattr(cached_loader, "clear", None)
        if callable(clear):
            clear()
        cache_clear = getattr(cached_loader, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()
