"""Per-device lens calibration: solving, storage, and its effect on registration."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from services.api.schemas.device_calibration import (
    MAX_PLAUSIBLE_DIAGONAL_FOV_DEG,
    MIN_PLAUSIBLE_DIAGONAL_FOV_DEG,
    CalibrationQuality,
    CheckerboardSpec,
    DeviceLensProfile,
)
from services.api.services import device_calibration_service
from services.api.services.device_calibration_service import (
    DeviceCalibrationError,
    calibrate_device_from_video,
    load_device_profile,
    save_device_profile,
)
from services.api.services.real_pitch_registration_service import (
    build_intrinsics_candidates,
    diagonal_fov_degrees,
)
from services.api.services.wicket_box_calibration_service import (
    _intrinsics_candidates_for_frame,
)


COLUMNS = 9
ROWS = 6
SQUARE_MM = 25.0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TRUE_FOCAL_PX = 1000.0

# device_calibration_service samples one frame in every FRAME_SAMPLE_STRIDE, so
# each synthetic pose is held for that many frames to become exactly one view.
FRAMES_PER_POSE = device_calibration_service.FRAME_SAMPLE_STRIDE

SPEC = CheckerboardSpec(columns=COLUMNS, rows=ROWS, square_size_mm=SQUARE_MM)


@pytest.fixture(autouse=True)
def isolated_profile_store(tmp_path, monkeypatch):
    """Keep tests off the real outputs/device_calibration directory."""
    monkeypatch.setattr(
        device_calibration_service,
        "DEVICE_CALIBRATION_ROOT",
        tmp_path / "device_calibration",
    )


def _board_render() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A flat picture of the printed board, plus where its corners sit in mm.

    The board is drawn with a white quiet zone, because findChessboardCorners
    cannot locate the outermost inner corners on a board that bleeds off its
    own edge.
    """
    px_per_mm = 4.0
    margin_mm = 25.0
    squares_x, squares_y = COLUMNS + 1, ROWS + 1
    width = int((squares_x * SQUARE_MM + 2 * margin_mm) * px_per_mm)
    height = int((squares_y * SQUARE_MM + 2 * margin_mm) * px_per_mm)

    image = np.full((height, width), 255, np.uint8)
    margin_px = int(margin_mm * px_per_mm)
    square_px = int(SQUARE_MM * px_per_mm)
    for row in range(squares_y):
        for column in range(squares_x):
            if (column + row) % 2 == 0:
                image[
                    margin_px + row * square_px: margin_px + (row + 1) * square_px,
                    margin_px + column * square_px: margin_px + (column + 1) * square_px,
                ] = 0

    # Board-plane millimetres, with the origin on the first inner corner so the
    # geometry matches the object points the service builds, then recentred so
    # the board sits on the optical axis rather than off in a corner.
    x0 = -(margin_mm + SQUARE_MM) - (COLUMNS - 1) * SQUARE_MM / 2
    y0 = -(margin_mm + SQUARE_MM) - (ROWS - 1) * SQUARE_MM / 2
    x1 = x0 + width / px_per_mm
    y1 = y0 + height / px_per_mm

    source_px = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    corners_mm = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return image, source_px, corners_mm


def _camera_matrix() -> np.ndarray:
    # An off-centre principal point, because a calibration that only ever sees
    # a perfectly centred one cannot show that it recovers the real thing.
    return np.float32(
        [
            [TRUE_FOCAL_PX, 0.0, FRAME_WIDTH / 2 - 6.0],
            [0.0, TRUE_FOCAL_PX, FRAME_HEIGHT / 2 + 4.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _poses(count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Tilted, distance-varied views — the ones that separate focal from range."""
    rng = np.random.default_rng(7)
    poses = []
    for _ in range(count):
        rotation, _ = cv2.Rodrigues(
            np.float32(
                [
                    math.radians(rng.uniform(-32, 32)),
                    math.radians(rng.uniform(-32, 32)),
                    math.radians(rng.uniform(-18, 18)),
                ]
            )
        )
        rotation_vector, _ = cv2.Rodrigues(rotation)
        translation = np.float32(
            [rng.uniform(-40, 40), rng.uniform(-30, 30), rng.uniform(420, 720)]
        ).reshape(3, 1)
        poses.append((rotation_vector, translation))
    return poses


def _write_calibration_video(path: Path, view_count: int) -> None:
    camera_matrix = _camera_matrix()
    board, source_px, corners_mm = _board_render()
    corners_3d = np.hstack(
        [corners_mm, np.zeros((4, 1), np.float32)]
    ).astype(np.float32)

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (FRAME_WIDTH, FRAME_HEIGHT),
    )
    try:
        for rotation_vector, translation in _poses(view_count):
            projected, _ = cv2.projectPoints(
                corners_3d,
                rotation_vector,
                translation,
                camera_matrix,
                np.zeros(5),
            )
            homography = cv2.getPerspectiveTransform(
                source_px,
                projected.reshape(-1, 2).astype(np.float32),
            )
            warped = cv2.warpPerspective(
                board,
                homography,
                (FRAME_WIDTH, FRAME_HEIGHT),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
            frame = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
            for _ in range(FRAMES_PER_POSE):
                writer.write(frame)
    finally:
        writer.release()


def _profile(**overrides) -> DeviceLensProfile:
    values = {
        "device_id": "test-device",
        "device_label": "Test phone",
        "calibrated_at": datetime(2026, 8, 1, tzinfo=timezone.utc),
        "image_width": 1080,
        "image_height": 1920,
        "focal_length_x_px": 1400.0,
        "focal_length_y_px": 1402.0,
        "principal_point_x_px": 538.0,
        "principal_point_y_px": 962.0,
        "distortion_coefficients": [0.04, -0.15, 0.001, -0.002, 0.06],
        "checkerboard": SPEC,
        "quality": CalibrationQuality(
            rms_reprojection_px=0.41,
            band="GOOD",
            views_used=18,
            views_submitted=26,
            diagonal_fov_degrees=72.7,
            fov_plausible=True,
            advice="Calibration is good.",
        ),
    }
    values.update(overrides)
    return DeviceLensProfile(**values)


def test_synthetic_focal_recovery(tmp_path) -> None:
    video = tmp_path / "calibration.mp4"
    _write_calibration_video(video, view_count=14)

    profile = calibrate_device_from_video(video, "synthetic", "Synthetic", SPEC)

    assert profile.focal_length_x_px == pytest.approx(TRUE_FOCAL_PX, rel=0.01)
    assert profile.focal_length_y_px == pytest.approx(TRUE_FOCAL_PX, rel=0.01)
    assert profile.quality.rms_reprojection_px < 1.0
    assert profile.image_width == FRAME_WIDTH
    assert profile.image_height == FRAME_HEIGHT
    # The principal point was deliberately off centre; recovering the centre of
    # the frame instead would mean the solve ignored it.
    assert profile.principal_point_x_px == pytest.approx(FRAME_WIDTH / 2 - 6.0, abs=25.0)


def test_insufficient_views_rejected(tmp_path) -> None:
    video = tmp_path / "too_short.mp4"
    _write_calibration_video(video, view_count=3)

    with pytest.raises(DeviceCalibrationError) as excinfo:
        calibrate_device_from_video(video, "synthetic", None, SPEC)

    assert excinfo.value.status_code == 422


def test_square_grid_rejected() -> None:
    with pytest.raises(ValueError):
        CheckerboardSpec(columns=6, rows=6, square_size_mm=SQUARE_MM)


def test_profile_roundtrip() -> None:
    profile = _profile(device_id="roundtrip")
    save_device_profile(profile)

    restored = load_device_profile("roundtrip")

    assert restored == profile


def test_rescale_within_aspect() -> None:
    profile = _profile(image_width=1080, image_height=1920, focal_length_x_px=1400.0)

    scaled = profile.scaled_to(2160, 3840)

    assert scaled.focal_length_x_px == pytest.approx(2800.0)
    assert scaled.principal_point_x_px == pytest.approx(1076.0)
    assert scaled.image_width == 2160

    with pytest.raises(ValueError):
        profile.scaled_to(1920, 1080)


def test_registration_prefers_device_profile() -> None:
    profile = _profile(device_id="registration", image_width=1080, image_height=1920)
    save_device_profile(profile)

    candidates = _intrinsics_candidates_for_frame("registration", 1080, 1920)

    assert len(candidates) == 1
    assert candidates[0].source == "device_calibration"
    assert candidates[0].confidence == "HIGH"
    assert candidates[0].focal_length_x_px == pytest.approx(profile.focal_length_x_px)
    # Measured distortion must survive; zeroing it would throw away half of
    # what the calibration bought.
    assert candidates[0].distortion_coefficients == profile.distortion_coefficients
    assert candidates[0].principal_point_x_px == pytest.approx(538.0)

    # An uncalibrated device falls back to the sweep rather than failing.
    fallback = _intrinsics_candidates_for_frame("no-such-device", 1080, 1920)
    assert all(item.source == "bounded_image_hypothesis" for item in fallback)


def test_implausible_fov_rejected_in_sweep() -> None:
    for width, height in ((1280, 720), (1920, 1080), (720, 1280), (1080, 1920)):
        candidates = build_intrinsics_candidates(width, height)
        assert candidates
        for candidate in candidates:
            fov = diagonal_fov_degrees(candidate.focal_length_x_px, width, height)
            assert MIN_PLAUSIBLE_DIAGONAL_FOV_DEG <= fov <= MAX_PLAUSIBLE_DIAGONAL_FOV_DEG

    # The 43.3-degree failure came out of focal refinement, not the seeds, so
    # the refinement bounds have to carry the same limits the seeds do.
    portrait = build_intrinsics_candidates(720, 1280)[0]
    assert diagonal_fov_degrees(portrait.upper_focal_bound_px, 720, 1280) == pytest.approx(
        MIN_PLAUSIBLE_DIAGONAL_FOV_DEG
    )
    assert diagonal_fov_degrees(portrait.lower_focal_bound_px, 720, 1280) == pytest.approx(
        MAX_PLAUSIBLE_DIAGONAL_FOV_DEG
    )
    assert 43.3 < diagonal_fov_degrees(portrait.upper_focal_bound_px, 720, 1280)
