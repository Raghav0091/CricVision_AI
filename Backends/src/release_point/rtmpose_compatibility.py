"""Isolated RTMPose/RTMW compatibility probe.

This module intentionally imports optional pose dependencies lazily so the main
CricVision app can start without MMPose installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
from importlib.util import find_spec
import json
import platform
import subprocess
import sys
from typing import Any


@dataclass(frozen=True)
class RTMPoseCompatibilityReport:
    platform: str
    python_version: str
    packages: dict[str, str | None]
    torch_cuda_available: bool | None
    torch_cuda_version: str | None
    torch_device_count: int | None
    cpu_fallback_available: bool
    installation_weight: str
    inference_api: str
    licensing: str
    model_availability: str
    graceful_startup: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "python_version": self.python_version,
            "packages": dict(self.packages),
            "torch_cuda_available": self.torch_cuda_available,
            "torch_cuda_version": self.torch_cuda_version,
            "torch_device_count": self.torch_device_count,
            "cpu_fallback_available": self.cpu_fallback_available,
            "installation_weight": self.installation_weight,
            "inference_api": self.inference_api,
            "licensing": self.licensing,
            "model_availability": self.model_availability,
            "graceful_startup": self.graceful_startup,
            "notes": list(self.notes),
        }


def assess_rtmpose_compatibility() -> RTMPoseCompatibilityReport:
    packages = {
        package: _package_version(package)
        for package in ("torch", "mmengine", "mmcv", "mmpose", "mmdet")
    }
    notes: list[str] = []

    torch_probe = _probe_torch_runtime() if packages["torch"] is not None else {}
    torch_cuda_available = torch_probe.get("cuda_available")
    torch_cuda_version = torch_probe.get("cuda_version")
    torch_device_count = torch_probe.get("device_count")
    if packages["torch"] is None:
        notes.append("PyTorch is not installed in the active Python environment.")
    elif torch_probe.get("error"):
        notes.append(f"PyTorch runtime probe failed: {torch_probe['error']}")
    else:
        if not torch_cuda_available:
            notes.append("CUDA is not available through PyTorch; CPU inference is the fallback.")

    if packages["mmpose"] is None:
        notes.append("MMPose is not installed; RTMPose provider must remain optional.")
    if packages["mmcv"] is None:
        notes.append("MMCV is not installed; Windows wheels may need careful version pinning.")

    return RTMPoseCompatibilityReport(
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        python_version=sys.version.split()[0],
        packages=packages,
        torch_cuda_available=torch_cuda_available,
        torch_cuda_version=torch_cuda_version,
        torch_device_count=torch_device_count,
        cpu_fallback_available=packages["torch"] is not None,
        installation_weight=(
            "Heavy optional stack: PyTorch + MMEngine/MMCV + MMPose model configs/checkpoints. "
            "Do not add to default app requirements until validated on the target deployment."
        ),
        inference_api=(
            "Expected wrapper path is MMPose inferencer APIs normalized behind PoseProvider; "
            "no release engine code should import MMPose directly."
        ),
        licensing=(
            "MMPose is Apache-2.0; individual model checkpoints/configs still need model-card "
            "review before bundling or commercial deployment."
        ),
        model_availability=(
            "RTMPose/RTMW whole-body models are available in the OpenMMLab ecosystem, but no "
            "checkpoint is bundled in this repository."
        ),
        graceful_startup=True,
        notes=notes,
    )


def _package_version(package: str) -> str | None:
    if find_spec(package) is None:
        return None
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "installed"


def _probe_torch_runtime() -> dict[str, Any]:
    code = (
        "import json, torch; "
        "print(json.dumps({"
        "'cuda_available': bool(torch.cuda.is_available()), "
        "'cuda_version': torch.version.cuda, "
        "'device_count': int(torch.cuda.device_count())"
        "}))"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception as exc:  # ponytail: compatibility probe should never block app startup.
        return {"error": str(exc)}
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "unknown error").strip()
        return {"error": message.splitlines()[0] if message else "unknown error"}
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"error": "PyTorch probe returned non-JSON output."}
