"""Print the isolated RTMPose/RTMW compatibility report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Backends.src.release_point.rtmpose_compatibility import (
    assess_rtmpose_compatibility,
)


if __name__ == "__main__":
    print(json.dumps(assess_rtmpose_compatibility().to_dict(), indent=2))
