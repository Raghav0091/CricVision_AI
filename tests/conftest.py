"""Shared pytest fixtures for CricVision AI."""

from __future__ import annotations

import pytest


@pytest.fixture
def temp_session_store(tmp_path, monkeypatch):
    """Redirect session JSON storage to a temporary directory."""
    session_file = tmp_path / "session_results.json"
    monkeypatch.setattr(
        "Backends.src.storage.session_store.SESSION_RESULTS_FILE",
        session_file,
    )
    monkeypatch.setattr(
        "Backends.src.storage.session_store.SESSION_DATA_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        "Backends.src.storage.session_store.SESSION_CLIPS_DIR",
        tmp_path / "clips",
    )
    return session_file


def ball_detection(center, confidence=0.9, box_size=10, box_key="box"):
    """Build a minimal ball detection dict."""
    cx, cy = center
    half = box_size // 2
    detection = {
        box_key: [cx - half, cy - half, cx + half, cy + half],
        "confidence": confidence,
        "center": [cx, cy],
    }
    return detection


def bat_detection(center, confidence=0.85, box_size=40):
    """Build a minimal bat detection dict."""
    cx, cy = center
    half = box_size // 2
    return {
        "bbox": [cx - half, cy - half, cx + half, cy + half],
        "confidence": confidence,
        "center": [cx, cy],
    }


def frame_record(frame_index, ball=None, bat=None, stumps=None):
    """Build one normalized-style frame record."""
    return {
        "frame_index": frame_index,
        "ball_detections": [ball] if ball else [],
        "bat_detections": [bat] if bat else [],
        "stump_detections": stumps or [],
    }
