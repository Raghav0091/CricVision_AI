import base64
import binascii
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_DIR = PROJECT_ROOT / "outputs" / "calibration_frames"


def save_calibration_frame(frame_data_url: str) -> tuple[Path, Image.Image]:
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

    try:
        image = Image.open(BytesIO(content)).convert("RGB")
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Calibration frame is not a valid image.") from exc

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = CALIBRATION_DIR / f"calibration_{timestamp}.jpg"
    image.save(path, format="JPEG", quality=92)
    return path, image
