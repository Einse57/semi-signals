"""Skeleton pose estimator using RTMPose via rtmlib (ONNX Runtime / OpenVINO)."""

from __future__ import annotations

import numpy as np

from rtmlib import RTMPose


class PoseEstimator:
    """Extract COCO-17 skeleton keypoints for detected persons.

    Uses RTMPose-m by default.  ONNX models are auto-downloaded on
    first use.
    """

    def __init__(
        self,
        model_url: str | None = None,
        model_input_size: tuple[int, int] | None = None,
        device: str | None = None,
        backend: str | None = None,
    ) -> None:
        from src.config import (
            POSE_ONNX_URL, POSE_INPUT_SIZE,
            INFERENCE_DEVICE, INFERENCE_BACKEND,
        )

        if device is None:
            device = INFERENCE_DEVICE
        if backend is None:
            backend = INFERENCE_BACKEND

        url = model_url or POSE_ONNX_URL
        input_size = model_input_size or POSE_INPUT_SIZE

        self._model = RTMPose(
            onnx_model=url,
            model_input_size=input_size,
            backend=backend,
            device=device,
        )

    # ── public API ────────────────────────────────────────────
    def estimate(
        self,
        frame: np.ndarray,
        bboxes: list[np.ndarray],
    ) -> list[dict]:
        """Run top-down pose estimation for each bbox.

        Returns a list of dicts, one per person:
            {
                "keypoints": np.ndarray (17, 2),   # x, y
                "scores":    np.ndarray (17,),      # per-joint confidence
                "bbox":      np.ndarray (5,),       # x1,y1,x2,y2,score
            }
        """
        if len(bboxes) == 0:
            return []

        # rtmlib RTMPose expects list of [x1,y1,x2,y2]
        bbox_xyxy = [bb[:4].tolist() for bb in bboxes]

        keypoints, scores = self._model(frame, bboxes=bbox_xyxy)
        # keypoints: (N, 17, 2), scores: (N, 17)

        output: list[dict] = []
        for i, bb in enumerate(bboxes):
            output.append(
                {
                    "keypoints": keypoints[i].astype(np.float32),
                    "scores": scores[i].astype(np.float32),
                    "bbox": bb,
                }
            )
        return output
