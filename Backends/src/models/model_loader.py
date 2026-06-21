"""Lazy YOLO model loading with a Streamlit-aware resource cache."""

from functools import lru_cache

from Backends.src.models.model_registry import get_model_info, get_model_path

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
    # A return-to-None API is intentional; CLI callers can decide how to log.


def load_yolo_model(model_key: str):
    """Load one registered YOLO model, returning ``None`` on a safe failure."""
    info = get_model_info(model_key)
    if info is None:
        _warn(f"Unknown model key: {model_key}")
        return None
    if info.get("type") not in {"yolo_detection", "yolo_segmentation"}:
        _warn(f"{info['name']} is not a YOLO model and cannot be loaded here.")
        return None

    model_path = get_model_path(model_key)
    if model_path is None or not model_path.is_file():
        _warn(f"Model file not found for {info['name']}: {model_path}")
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


def clear_model_cache() -> None:
    """Clear only the multi-model loader cache."""
    clear = getattr(_cached_yolo_model, "clear", None)
    if callable(clear):
        clear()
    cache_clear = getattr(_cached_yolo_model, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()

