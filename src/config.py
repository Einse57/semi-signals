"""Shared configuration constants."""

from pathlib import Path
import torch

# ── Project root ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Device selection (XPU > CUDA > CPU) ───────────────────────
def get_device() -> torch.device:
    if torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()


# ── Patch rtmlib to support Intel GPU via OpenVINO ────────────
def _patch_rtmlib_openvino_gpu() -> None:
    """Monkey-patch rtmlib's BaseTool to support GPU with OpenVINO.

    rtmlib's built-in openvino backend hardcodes CPU and assumes
    exactly 2 outputs.  This patch:
      1. Adds a 'gpu' device string to the onnxruntime backend mapping
         (fallback if ORT-OpenVINO EP works).
      2. Overrides BaseTool.__init__ so that backend='openvino' with
         device='gpu' compiles to the Intel iGPU.
      3. Fixes the inference method to handle any number of outputs.
    """
    try:
        from rtmlib.tools import base as _base
        import os
        import numpy as np

        _orig_init = _base.BaseTool.__init__

        def _patched_init(self, onnx_model=None, model_input_size=None,
                          mean=None, std=None, backend='opencv',
                          device='cpu'):
            if backend == 'openvino':
                if not os.path.exists(onnx_model):
                    from rtmlib.tools.base import download_checkpoint
                    onnx_model = download_checkpoint(onnx_model)

                from openvino import Core
                core = Core()
                model_ov = core.read_model(model=onnx_model)

                # Map device string → OpenVINO device name
                ov_device = 'GPU' if device == 'gpu' else 'CPU'
                self.compiled_model = core.compile_model(
                    model=model_ov,
                    device_name=ov_device,
                    config={'PERFORMANCE_HINT': 'LATENCY'},
                )
                # Store all output layers (not just 2)
                self._ov_outputs = [
                    self.compiled_model.output(i)
                    for i in range(len(model_ov.outputs))
                ]
                # Keep legacy attrs for any code that uses them
                self.input_layer = self.compiled_model.input(0)
                if len(self._ov_outputs) >= 1:
                    self.output_layer0 = self._ov_outputs[0]
                if len(self._ov_outputs) >= 2:
                    self.output_layer1 = self._ov_outputs[1]

                print(f'load {onnx_model} with openvino backend ({ov_device})')

                self.onnx_model = onnx_model
                self.model_input_size = model_input_size
                self.mean = mean
                self.std = std
                self.backend = backend
                self.device = device
            else:
                _orig_init(self, onnx_model, model_input_size,
                           mean, std, backend, device)

        _base.BaseTool.__init__ = _patched_init

        # Patch inference to handle variable output count
        _orig_inference = _base.BaseTool.inference

        def _patched_inference(self, img: np.ndarray):
            if self.backend == 'openvino':
                img = img.transpose(2, 0, 1)
                img = np.ascontiguousarray(img, dtype=np.float32)
                input_tensor = img[None, :, :, :]
                results = self.compiled_model(input_tensor)
                return [results[out] for out in self._ov_outputs]
            return _orig_inference(self, img)

        _base.BaseTool.inference = _patched_inference

    except Exception:
        pass  # rtmlib not installed or structure changed

_patch_rtmlib_openvino_gpu()


def get_inference_device() -> str:
    """Return the best rtmlib device string for ONNX inference.

    Returns 'gpu' if OpenVINO + Intel GPU are available, else 'cpu'.
    """
    try:
        import openvino as ov
        core = ov.Core()
        if "GPU" in core.available_devices:
            return "gpu"
    except Exception:
        pass
    return "cpu"


def get_inference_backend() -> str:
    """Return 'openvino' if GPU available, else 'onnxruntime'."""
    if get_inference_device() == "gpu":
        return "openvino"
    return "onnxruntime"

INFERENCE_DEVICE = get_inference_device()
INFERENCE_BACKEND = get_inference_backend()

# ── ONNX model URLs (rtmlib auto-downloads & caches) ─────────
# YOLOX-tiny person detector (HumanArt+COCO, 416×416) – fast on CPU
DET_ONNX_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip"
)
DET_INPUT_SIZE = (416, 416)

# RTMPose-m 17-keypoint body (Body7, 192×256)
POSE_ONNX_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
)
POSE_INPUT_SIZE = (192, 256)

# RTMPose-m hand (21 keypoints, 256×256)
HAND_ONNX_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip"
)
HAND_INPUT_SIZE = (256, 256)

# ── Skeleton topology (COCO-17) ──────────────────────────────
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (5, 6),                                  # shoulders
    (5, 7), (7, 9),                          # left arm
    (6, 8), (8, 10),                         # right arm
    (5, 11), (6, 12),                        # torso
    (11, 12),                                # hips
    (11, 13), (13, 15),                      # left leg
    (12, 14), (14, 16),                      # right leg
]

# COCO-17 keypoint names
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# ── Hand keypoints (21-point standard) ───────────────────────
HAND_KEYPOINT_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

# Hand skeleton edges (21-keypoint connectivity)
HAND_SKELETON_EDGES = [
    # thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# ── Body-region definitions for motion intensity ──────────────
# Each region maps to COCO-17 joint indices.
# Per-side so each chart can overlay left vs right.
# A region is "visible" when at least one constituent joint passes
# the keypoint confidence threshold.
BODY_REGIONS: dict[str, list[int]] = {
    "head":    [0, 1, 2],              # nose, eyes (VSViG drops ears)
    "l_arm":   [5, 7],                 # l_shoulder, l_elbow
    "r_arm":   [6, 8],                 # r_shoulder, r_elbow
    "l_hand":  [9],                    # l_wrist (overridden by hand model)
    "r_hand":  [10],                   # r_wrist (overridden by hand model)
    "l_leg":   [11, 13, 15],           # l_hip, l_knee, l_ankle
    "r_leg":   [12, 14, 16],           # r_hip, r_knee, r_ankle
    "overall": list(range(17)),        # all joints
}

# Display colours (BGR) for each region line
REGION_COLORS: dict[str, tuple[int, int, int]] = {
    "head":    (0, 255, 255),    # cyan
    "l_arm":   (0, 200, 0),      # green
    "r_arm":   (200, 0, 0),      # blue
    "l_hand":  (0, 255, 128),    # light green
    "r_hand":  (255, 100, 100),  # light blue
    "l_leg":   (0, 200, 0),      # green
    "r_leg":   (200, 0, 0),      # blue
    "overall": (180, 180, 180),  # dim grey
}

# ── Chart layout: which regions appear together on each mini-chart ──
# Each entry: title → list of region keys to overlay.
CHART_GROUPS: list[tuple[str, list[str]]] = [
    ("Hands",  ["l_hand", "r_hand"]),
    ("Arms",   ["l_arm", "r_arm"]),
    ("Head",   ["head"]),
    ("Legs",   ["l_leg", "r_leg"]),
]

# ── Display ───────────────────────────────────────────────────
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540
CHART_TIME_WINDOW = 60.0      # seconds of history shown on charts (1 min)
MOTION_HISTORY_LEN = 2_400    # data-points to keep (~1 min at 40 FPS)
