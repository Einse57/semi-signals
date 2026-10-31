"""Optional local test assets (not in this repository). Tests that need them are skipped when missing."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# MediaPipe's sample image, see README "Tests" for the download link
TEST_IMAGE = Path(os.environ.get("FACE_TEST_IMAGE") or ROOT / "data" / "facetest" / "business-person.png")
