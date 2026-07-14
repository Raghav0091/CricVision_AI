import base64
import binascii
from datetime import datetime, timezone
from pathlib import Path


CALIBRATION_DIR = Path("outputs") / "pro_v2" / "calibration"


def save_calibration_frame(frame_data_url: str) -> Path:
    if "," not in frame_data_url:
        raise ValueError("Calibration frame must be a browser image data URL.")
    header, encoded = frame_data_url.split(",", 1)
    if not header.startswith("data:image/") or ";base64" not in header:
        raise ValueError("Calibration frame must be a base64 image data URL.")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Calibration frame data is invalid.") from exc
    if not content:
        raise ValueError("Calibration frame is empty.")

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = CALIBRATION_DIR / f"calibration_{timestamp}.jpg"
    path.write_bytes(content)
    return path
