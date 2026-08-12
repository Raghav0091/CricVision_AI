"""Per-device lens calibration contracts.

A phone's focal length and distortion are fixed physical properties, so they
are solved once from a checkerboard and reused by every later pose solve. This
removes the dominant unknown from wicket-box PnP: without it the solver trades
focal length against camera distance and can settle on a low-error fit that
describes a lens the phone does not have.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DEVICE_CALIBRATION_SCHEMA_VERSION = "device_calibration_v1"

# Diagonal field of view bounds for a plausible phone main camera. Ultrawides
# sit near 100-120 and nobody films cricket on one; anything under 55 implies a
# telephoto that phones do not use for video.
MIN_PLAUSIBLE_DIAGONAL_FOV_DEG = 55.0
MAX_PLAUSIBLE_DIAGONAL_FOV_DEG = 95.0

# cv2.calibrateCamera returns RMS reprojection error in pixels.
GOOD_RMS_PX = 0.5
ACCEPTABLE_RMS_PX = 1.0

MIN_VIEWS = 8
RECOMMENDED_VIEWS = 20


class DeviceCalibrationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CheckerboardSpec(DeviceCalibrationModel):
    """Inner-corner counts, not square counts — cv2 wants the corners."""

    columns: int = Field(default=9, ge=3, le=20)
    rows: int = Field(default=6, ge=3, le=20)
    square_size_mm: float = Field(gt=1.0, le=200.0)

    @model_validator(mode="after")
    def reject_square_grid(self) -> "CheckerboardSpec":
        # A square grid is rotationally ambiguous, so corner ordering can flip
        # between views and the solve degrades silently.
        if self.columns == self.rows:
            raise ValueError(
                "Use a checkerboard with different column and row counts; "
                "a square grid has ambiguous orientation."
            )
        return self


class CalibrationQuality(DeviceCalibrationModel):
    rms_reprojection_px: float = Field(ge=0)
    band: Literal["GOOD", "ACCEPTABLE", "POOR"]
    views_used: int = Field(ge=0)
    views_submitted: int = Field(ge=0)
    diagonal_fov_degrees: float = Field(gt=0)
    fov_plausible: bool
    advice: str


class DeviceLensProfile(DeviceCalibrationModel):
    schema_version: Literal["device_calibration_v1"] = DEVICE_CALIBRATION_SCHEMA_VERSION
    device_id: str = Field(min_length=1, max_length=128)
    device_label: str | None = Field(default=None, max_length=200)
    calibrated_at: datetime

    # Intrinsics are resolution-dependent: focal length in pixels scales with
    # frame width, so the resolution they were measured at travels with them.
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)

    focal_length_x_px: float = Field(gt=0)
    focal_length_y_px: float = Field(gt=0)
    principal_point_x_px: float = Field(gt=0)
    principal_point_y_px: float = Field(gt=0)
    distortion_coefficients: list[float] = Field(min_length=4, max_length=8)

    checkerboard: CheckerboardSpec
    quality: CalibrationQuality

    def camera_matrix(self) -> list[list[float]]:
        return [
            [self.focal_length_x_px, 0.0, self.principal_point_x_px],
            [0.0, self.focal_length_y_px, self.principal_point_y_px],
            [0.0, 0.0, 1.0],
        ]

    def transposed(self) -> "DeviceLensProfile":
        """Same profile with the axes swapped, for a 90-degree rotation.

        Turning the phone does not change the lens. The sensor reads out the
        same pixels and the frame is simply rotated, so the focal length in
        pixels is unchanged along each physical axis — only which axis is called
        width swaps. Refusing to reuse the profile here would send the solver
        back to guessing the focal length purely because the operator held the
        phone sideways.

        The principal point is mirrored about the centre, which is exact only
        for one of the two rotation directions. Phone principal points sit
        within a few pixels of centre, so the residual is far smaller than the
        error from having no profile at all.
        """
        return self.model_copy(
            update={
                "image_width": self.image_height,
                "image_height": self.image_width,
                "focal_length_x_px": self.focal_length_y_px,
                "focal_length_y_px": self.focal_length_x_px,
                "principal_point_x_px": self.principal_point_y_px,
                "principal_point_y_px": self.principal_point_x_px,
            }
        )

    def scaled_to(self, width: int, height: int) -> "DeviceLensProfile":
        """Rescale intrinsics to a different capture resolution.

        Handles a 90-degree rotation by transposing first. Beyond that, a
        differing aspect ratio means a different crop of the sensor, which is
        not a pure scale and cannot be recovered.
        """
        if width <= 0 or height <= 0:
            raise ValueError("Target resolution must be positive.")
        rotated = width / height
        if abs(rotated - self.image_height / self.image_width) < 0.02:
            return self.transposed()._scaled_within_aspect(width, height)
        return self._scaled_within_aspect(width, height)

    def _scaled_within_aspect(self, width: int, height: int) -> "DeviceLensProfile":
        source_aspect = self.image_width / self.image_height
        target_aspect = width / height
        if abs(source_aspect - target_aspect) > 0.02:
            raise ValueError(
                "Calibration cannot be rescaled across aspect ratios; "
                f"profile is {self.image_width}x{self.image_height}, "
                f"requested {width}x{height}."
            )
        scale = width / self.image_width
        return self.model_copy(
            update={
                "image_width": width,
                "image_height": height,
                "focal_length_x_px": self.focal_length_x_px * scale,
                "focal_length_y_px": self.focal_length_y_px * scale,
                "principal_point_x_px": self.principal_point_x_px * scale,
                "principal_point_y_px": self.principal_point_y_px * scale,
            }
        )


class DeviceCalibrationRequest(DeviceCalibrationModel):
    device_id: str = Field(min_length=1, max_length=128)
    device_label: str | None = Field(default=None, max_length=200)
    checkerboard: CheckerboardSpec


class DeviceCalibrationResponse(DeviceCalibrationModel):
    success: bool
    status: Literal["CALIBRATED", "INSUFFICIENT_VIEWS", "FAILED"]
    profile: DeviceLensProfile | None = None
    message: str


def quality_band(rms_px: float) -> Literal["GOOD", "ACCEPTABLE", "POOR"]:
    if rms_px <= GOOD_RMS_PX:
        return "GOOD"
    if rms_px <= ACCEPTABLE_RMS_PX:
        return "ACCEPTABLE"
    return "POOR"
