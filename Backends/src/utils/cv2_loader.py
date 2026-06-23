"""Lazy OpenCV loader for Streamlit UI modules."""

_cv2 = None


def get_cv2():
    """Import cv2 on demand; show a Streamlit error instead of crashing app import."""
    global _cv2
    if _cv2 is not None:
        return _cv2
    try:
        import cv2 as cv2_module

        _cv2 = cv2_module
        return _cv2
    except ImportError as exc:
        try:
            import streamlit as st

            st.error(
                "OpenCV (cv2) could not be loaded. "
                "Ensure opencv-python-headless is installed and packages.txt system libraries are available."
            )
        except Exception:
            pass
        raise ImportError("OpenCV (cv2) is not available") from exc


class _LazyCv2:
    def __getattr__(self, name):
        return getattr(get_cv2(), name)


cv2 = _LazyCv2()
