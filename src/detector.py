"""Person detector using YOLOX via rtmlib (ONNX Runtime / OpenVINO)."""

from __future__ import annotations

import numpy as np

from rtmlib import YOLOX


class PersonDetector:
    """Detect persons in a frame and return bounding boxes.

    Uses YOLOX-m (HumanArt+COCO) by default.  ONNX models are
    auto-downloaded on first use.
    """

    def __init__(
        self,
        model_url: str | None = None,
        model_input_size: tuple[int, int] | None = None,
        device: str | None = None,
        backend: str | None = None,
        score_thr: float = 0.5,
    ) -> None:
        from src.config import (
            DET_ONNX_URL, DET_INPUT_SIZE,
            INFERENCE_DEVICE, INFERENCE_BACKEND,
        )

        if device is None:
            device = INFERENCE_DEVICE
        if backend is None:
            backend = INFERENCE_BACKEND

        url = model_url or DET_ONNX_URL
        input_size = model_input_size or DET_INPUT_SIZE

        self._model = YOLOX(
            onnx_model=url,
            model_input_size=input_size,
            backend=backend,
            device=device,
            score_thr=score_thr,
        )

    # ── public API ────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> list[np.ndarray]:
        """Return a list of [x1, y1, x2, y2, score] arrays for each person."""
        bboxes = self._model(frame)  # (N, 4) xyxy, already filtered

        if bboxes is None or len(bboxes) == 0:
            return []

        # Append confidence=1.0 (bboxes already passed score threshold)
        return [
            np.append(bb, 1.0).astype(np.float32) for bb in bboxes
        ]
