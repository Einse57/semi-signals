"""Hand pose estimator: 21-keypoint per hand via RTMPose-hand.

Crops around body wrist positions and runs RTMPose-hand on each crop,
avoiding a separate hand detection pass.
"""

from __future__ import annotations

import numpy as np
from rtmlib import RTMPose


class HandEstimator:
    """Extract 21-keypoint hand poses by cropping around wrist positions.

    Uses the body-pose wrist keypoints (COCO indices 9=L, 10=R) and
    the forearm vector (elbow→wrist) to compute a crop box, then runs
    RTMPose-hand on each crop.
    """

    # COCO-17 indices
    _L_ELBOW, _L_WRIST = 7, 9
    _R_ELBOW, _R_WRIST = 8, 10

    def __init__(
        self,
        model_url: str | None = None,
        model_input_size: tuple[int, int] | None = None,
        device: str | None = None,
        backend: str | None = None,
        wrist_conf_thr: float = 0.4,
        crop_scale: float = 2.5,
    ) -> None:
        from src.config import (
            HAND_ONNX_URL, HAND_INPUT_SIZE,
            INFERENCE_DEVICE, INFERENCE_BACKEND,
        )

        url = model_url or HAND_ONNX_URL
        input_size = model_input_size or HAND_INPUT_SIZE

        self._model = RTMPose(
            onnx_model=url,
            model_input_size=input_size,
            backend=backend or INFERENCE_BACKEND,
            device=device or INFERENCE_DEVICE,
        )
        self._wrist_conf_thr = wrist_conf_thr
        self._crop_scale = crop_scale

    def _wrist_crop_bbox(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        elbow_idx: int,
        wrist_idx: int,
        img_h: int,
        img_w: int,
    ) -> np.ndarray | None:
        """Compute a square crop box centred on the wrist.

        The side length is ``crop_scale * forearm_length``.  If the
        elbow is not visible, a fixed 120 px box is used instead.
        """
        if scores[wrist_idx] < self._wrist_conf_thr:
            return None

        wx, wy = keypoints[wrist_idx]

        if scores[elbow_idx] >= self._wrist_conf_thr:
            ex, ey = keypoints[elbow_idx]
            forearm = np.linalg.norm([wx - ex, wy - ey])
            half = max(forearm * self._crop_scale / 2.0, 40.0)
        else:
            half = 60.0  # fallback

        x1 = max(0, int(wx - half))
        y1 = max(0, int(wy - half))
        x2 = min(img_w, int(wx + half))
        y2 = min(img_h, int(wy + half))

        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return None

        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def estimate(
        self,
        frame: np.ndarray,
        body_keypoints: np.ndarray,
        body_scores: np.ndarray,
    ) -> dict[str, dict | None]:
        """Run hand pose for left and right hands.

        Returns dict with keys ``"left"`` and ``"right"``, each either
        ``None`` (wrist not visible) or a dict::

            {
                "keypoints": np.ndarray (21, 2),  # in original frame coords
                "scores":    np.ndarray (21,),
                "bbox":      np.ndarray (4,),      # crop box used
            }
        """
        h, w = frame.shape[:2]
        result: dict[str, dict | None] = {"left": None, "right": None}

        sides = [
            ("left",  self._L_ELBOW, self._L_WRIST),
            ("right", self._R_ELBOW, self._R_WRIST),
        ]

        for side, elbow_idx, wrist_idx in sides:
            bbox = self._wrist_crop_bbox(
                body_keypoints, body_scores,
                elbow_idx, wrist_idx, h, w,
            )
            if bbox is None:
                continue

            bbox_list = [bbox[:4].tolist()]
            kpts, scores = self._model(frame, bboxes=bbox_list)
            # kpts: (1, 21, 2), scores: (1, 21)

            result[side] = {
                "keypoints": kpts[0].astype(np.float32),
                "scores": scores[0].astype(np.float32),
                "bbox": bbox,
            }

        return result
