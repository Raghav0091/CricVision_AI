"""Pydantic API contracts."""

from .preset_auto_registration import (
    CAMERA_SETUP_PRESETS_BY_ID,
    CAMERA_SETUP_PRESETS,
    STANDARD_REAR_WICKET_NET_V1,
    AutoRegistrationStatus,
    CameraSetupPresetListResponse,
    CameraSetupPreset,
    PresetAutoRegistrationResult,
    PresetAutoRegistrationRunRequest,
    PresetCompatibilityInput,
    PresetCompatibilityStatus,
    get_camera_setup_preset,
    list_camera_setup_presets,
)

__all__ = [
    "AutoRegistrationStatus",
    "CAMERA_SETUP_PRESETS_BY_ID",
    "CAMERA_SETUP_PRESETS",
    "CameraSetupPresetListResponse",
    "CameraSetupPreset",
    "PresetAutoRegistrationResult",
    "PresetAutoRegistrationRunRequest",
    "PresetCompatibilityInput",
    "PresetCompatibilityStatus",
    "STANDARD_REAR_WICKET_NET_V1",
    "get_camera_setup_preset",
    "list_camera_setup_presets",
]
