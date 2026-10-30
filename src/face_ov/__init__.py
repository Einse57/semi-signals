"""OpenVINO port of the MediaPipe Face Landmarker graph (detector, landmarks, blendshapes).

Self-contained: depends on numpy, OpenCV and OpenVINO only. The geometry follows MediaPipe's
calculators (google-ai-edge/mediapipe, Apache-2.0); see geometry.py for per-function file references.
"""
from .bundle import unpack_task
from .face_geometry import FaceGeometry
from .geometry import BLENDSHAPE_SUBSET, Rect
from .landmarker import DEVICES, FaceLandmarkerOV, FaceResult

__all__ = ["BLENDSHAPE_SUBSET", "DEVICES", "FaceGeometry", "FaceLandmarkerOV", "FaceResult", "Rect", "unpack_task"]
