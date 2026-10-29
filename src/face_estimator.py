"""Optional face stage: MediaPipe Face Landmarker (478 3D landmarks + 52 blendshapes).

Where the landmarker looks is set by --face-preprocess (src/face_preprocess.py): by default the frame is
taken to be the face region already ("none"); "detect" finds, crops and aligns the face; "coco-head"
crops from the COCO-17 head keypoints of the body pose (nose, eyes, ears), the same way the hand crops
come from the wrists (src/hand_estimator.py).

Where it runs (backend="mediapipe", the default): MediaPipe Tasks runs the bundled TFLite models
(BlazeFace short-range, Face Mesh V2, Blendshape V2) on its own CPU runtime (TFLite + XNNPACK), so it
stays on the CPU when the body and hand models run on the Intel iGPU.
backend="openvino": the same three models from the same bundle run with OpenVINO on CPU, GPU, NPU or
AUTO (src/face_ov, a re-implementation of the MediaPipe graph; docs/FACE_OPENVINO.md).

The model bundle is not in this repository: scripts/fetch_models.py --face downloads it and checks
its SHA-256 (FACE_TASK_URL / FACE_TASK_SHA256 below). Google, Apache-2.0; see NOTICE and
docs/FACE_LANDMARKS.md for the model card notes.

Coordinates returned are in original-frame pixels: x, y in px; z in px on the same scale as x
(MediaPipe's z is relative to the face centre and scaled with the face width).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

FACE_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                 "face_landmarker/float16/1/face_landmarker.task")
FACE_TASK_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"  # checked by fetch_models.py
FACE_TASK_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

FACE_BACKENDS = ("mediapipe", "openvino")
FACE_DEVICES = ("CPU", "GPU", "NPU", "AUTO")

N_FACE = 478                     # 468 mesh + 10 iris landmarks
N_BLEND = 52
IRIS_IDX = list(range(468, 478))

# COCO-17 head keypoints
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO = 5, 6
HEAD_IDX = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)

# Typical distance between two head keypoints as a fraction of the face width (ear to ear).
# Used to estimate the face width from whichever pair is visible.
_PAIR_RATIO = {
    (L_EAR, R_EAR): 1.0, (L_EYE, R_EYE): 0.45, (L_EYE, L_EAR): 0.35, (R_EYE, R_EAR): 0.35,
    (L_EYE, R_EAR): 0.75, (R_EYE, L_EAR): 0.75, (NOSE, L_EYE): 0.28, (NOSE, R_EYE): 0.28,
    (NOSE, L_EAR): 0.55, (NOSE, R_EAR): 0.55,
}

# Face regions (MediaPipe face mesh indices). "l_" and "r_" are the subject's left and right, as in
# COCO-17 and the MediaPipe *Left/*Right blendshapes (not the viewer's left/right).
_L_EYE_LIDS = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
_R_EYE_LIDS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
_L_IRIS = [473, 474, 475, 476, 477]
_R_IRIS = [468, 469, 470, 471, 472]
_L_BROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
_R_BROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
         78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415]
_L_OVAL = [338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377]
_R_OVAL = [109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148]
_L_MOUTH = [291, 375, 321, 405, 314, 409, 270, 269, 267, 308, 324, 318, 402, 317, 415, 310, 311, 312]
_R_MOUTH = [61, 146, 91, 181, 84, 185, 40, 39, 37, 78, 95, 88, 178, 87, 191, 80, 81, 82]
_L_CHEEK = [425, 411, 280, 352, 346, 347, 330, 266, 423, 426]
_R_CHEEK = [205, 187, 50, 123, 117, 118, 101, 36, 203, 206]

FACE_REGIONS: dict[str, list[int]] = {
    "mouth": _LIPS,
    "l_eye": _L_EYE_LIDS + _L_IRIS,
    "r_eye": _R_EYE_LIDS + _R_IRIS,
    "l_face": _L_OVAL + _L_BROW + _L_EYE_LIDS + _L_CHEEK + _L_MOUTH,
    "r_face": _R_OVAL + _R_BROW + _R_EYE_LIDS + _R_CHEEK + _R_MOUTH,
}
FACE_WIDTH_PAIR = (234, 454)      # right / left cheek extremes on the face oval

BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]
assert len(BLENDSHAPE_NAMES) == N_BLEND


def head_crop_bbox(keypoints: np.ndarray, scores: np.ndarray, img_h: int, img_w: int,
                   body_bbox=None, conf_thr: float = 0.3, scale: float = 2.0,
                   min_side: float = 48.0) -> np.ndarray | None:
    """Square head crop [x1, y1, x2, y2] (float32, may extend past the frame; the caller pads).

    Centre: mean of the visible head keypoints (nose, eyes, ears with score >= conf_thr).
    Side: scale * estimated face width. The face width comes from the visible head keypoint pairs
    (_PAIR_RATIO; the largest estimate wins). Fallback when fewer than two head keypoints are visible:
    face width = 0.4 * shoulder width if both shoulders are visible, else 0.12 * the longer side of
    body_bbox (lying or sitting subjects: the longer side is roughly the body length).
    Returns None when no head keypoint is visible, when only one is visible and there is no fallback
    size, or when the crop lies entirely outside the frame.
    """
    kp = np.asarray(keypoints, np.float32)
    sc = np.asarray(scores, np.float32)
    vis = [i for i in HEAD_IDX if sc[i] >= conf_thr and np.all(np.isfinite(kp[i]))]
    if not vis:
        return None
    cx, cy = kp[vis].mean(axis=0)
    width = 0.0
    for (a, b), r in _PAIR_RATIO.items():
        if a in vis and b in vis:
            width = max(width, float(np.linalg.norm(kp[a] - kp[b])) / r)
    if width <= 0:
        if sc[L_SHO] >= conf_thr and sc[R_SHO] >= conf_thr:
            width = 0.4 * float(np.linalg.norm(kp[L_SHO] - kp[R_SHO]))
        elif body_bbox is not None and np.all(np.isfinite(np.asarray(body_bbox[:4], float))):
            bw, bh = float(body_bbox[2] - body_bbox[0]), float(body_bbox[3] - body_bbox[1])
            width = 0.12 * max(bw, bh)
        else:
            return None
    side = max(min_side, scale * width)
    half = side / 2.0
    x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
    if x2 <= 0 or y2 <= 0 or x1 >= img_w or y1 >= img_h:
        return None
    return np.array([x1, y1, x2, y2], np.float32)


def square_crop(frame: np.ndarray, box) -> np.ndarray:
    """Crop box from frame, zero-padding where the box extends past the frame edges."""
    x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
    h, w = frame.shape[:2]
    out = np.zeros((y2 - y1, x2 - x1, 3), frame.dtype)
    sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if sx2 > sx1 and sy2 > sy1:
        out[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = frame[sy1:sy2, sx1:sx2]
    return out


def empty_face(n: int | None = None) -> dict:
    """NaN placeholders, shape (478, 3), (52,), scalar; with n: a leading persons axis."""
    lead = () if n is None else (n,)
    return dict(landmarks=np.full(lead + (N_FACE, 3), np.nan, np.float32),
                blendshapes=np.full(lead + (N_BLEND,), np.nan, np.float32),
                score=np.full(lead, np.nan, np.float32) if n is not None else np.float32(np.nan),
                bbox=np.full(lead + (4,), np.nan, np.float32),
                transform=np.full(lead + (4, 4), np.nan, np.float32))


class FaceEstimator:
    """Face Landmarker (IMAGE mode) on a whole image or a crop.

    backend: "mediapipe" (MediaPipe Tasks, CPU; default) or "openvino" (src/face_ov on `device`)."""

    def __init__(self, model_path: str | os.PathLike | None = None, crop_scale: float = 2.0,
                 conf_thr: float = 0.3, min_det_conf: float = 0.5, min_presence: float = 0.5,
                 transform: bool = False, num_threads: int | None = None, num_faces: int = 1,
                 blendshapes: bool = True, backend: str = "mediapipe", device: str = "CPU") -> None:
        path = Path(model_path) if model_path else FACE_TASK_PATH
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run python scripts/fetch_models.py --face")
        if backend not in FACE_BACKENDS:
            raise ValueError(f"face backend must be one of {FACE_BACKENDS}, not {backend!r}")
        self.backend, self.device = backend, (str(device).upper() if backend == "openvino" else "CPU")
        self._crop_scale, self._conf_thr, self._transform = crop_scale, conf_thr, transform
        self.num_faces, self.blendshapes = int(num_faces), bool(blendshapes)
        self.model_path = str(path)
        self._lm = self._ov = None
        if backend == "openvino":
            from src.face_ov import FaceLandmarkerOV
            expect = FACE_TASK_SHA256 if model_path is None else None
            self._ov = FaceLandmarkerOV(path, device=self.device, video=False,
                                        min_detection_confidence=min_det_conf,
                                        min_presence_confidence=min_presence, expected_sha256=expect)
            return
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.IMAGE, num_faces=self.num_faces,
            min_face_detection_confidence=min_det_conf, min_face_presence_confidence=min_presence,
            output_face_blendshapes=self.blendshapes, output_facial_transformation_matrixes=transform)
        self._lm = vision.FaceLandmarker.create_from_options(opts)

    @property
    def runtime(self) -> str:
        if self.backend == "openvino":
            return f"openvino ({self.device})"
        return "mediapipe (TFLite/XNNPACK, CPU)"

    def close(self):
        """Call before the program exits (MediaPipe cannot close its graph during interpreter shutdown)."""
        if self._lm is not None:
            self._lm.close()
            self._lm = None
        if self._ov is not None:
            self._ov.close()
            self._ov = None

    def _faces(self, img: np.ndarray, offset=(0.0, 0.0)) -> list[dict]:
        """Every face found (up to num_faces): landmarks (478, 3) in img px plus offset (z on the x
        scale), blendshapes (52,) (NaN when off), score, transform (4, 4) (NaN when off)."""
        if img.size == 0:
            return []
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        w, h = img.shape[1], img.shape[0]
        scale = np.array([w, h, w], np.float32)
        off = np.array([offset[0], offset[1], 0.0], np.float32)
        out = []
        if self._ov is not None:
            r = self._ov.process(rgb)
            for f in ([] if r is None else [r]):
                bs = f.blendshapes.astype(np.float32) if self.blendshapes else np.full(N_BLEND, np.nan, np.float32)
                out.append(dict(landmarks=(f.landmarks * scale + off).astype(np.float32), blendshapes=bs,
                                score=np.float32(1.0), transform=np.full((4, 4), np.nan, np.float32)))
            return out
        res = self._lm.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        for i, f in enumerate(res.face_landmarks):
            lm = np.array([[p.x, p.y, p.z] for p in f], np.float32) * scale + off
            if lm.shape[0] != N_FACE:
                continue
            bs = np.full(N_BLEND, np.nan, np.float32)
            if self.blendshapes and res.face_blendshapes and i < len(res.face_blendshapes):
                for c in res.face_blendshapes[i]:
                    if 0 <= c.index < N_BLEND:
                        bs[c.index] = c.score
            # The Tasks API reports no per-face confidence; a returned face passed both the detection
            # (min_det_conf) and the presence (min_presence) thresholds. Landmark presence is used when
            # the runtime fills it, else 1.0.
            pres = [p.presence for p in f if getattr(p, "presence", None)]
            tf = np.full((4, 4), np.nan, np.float32)
            if self._transform and res.facial_transformation_matrixes and i < len(res.facial_transformation_matrixes):
                tf = np.asarray(res.facial_transformation_matrixes[i], np.float32).reshape(4, 4)
            out.append(dict(landmarks=lm, blendshapes=bs, score=np.float32(np.mean(pres) if pres else 1.0),
                            transform=tf))
        return out

    def detect_image(self, img: np.ndarray, offset=(0.0, 0.0)) -> dict | None:
        """First face on a whole BGR image (see _faces); None if no face was found."""
        faces = self._faces(img, offset)
        return faces[0] if faces else None

    def all_faces(self, img: np.ndarray, offset=(0.0, 0.0)) -> list[np.ndarray]:
        """Landmarks (478, 3) of every face found (up to num_faces), in img px plus offset. Used to blur
        bystanders in de-identified video (src/segment_export.py)."""
        return [f["landmarks"] for f in self._faces(img, offset)]

    def estimate_crop(self, frame: np.ndarray, box) -> dict | None:
        """Run on one crop box; returns landmarks (478,3) in frame px, blendshapes (52,), score, bbox,
        transform (4,4) or None if no face was found in the crop."""
        crop = square_crop(frame, box)
        r = self.detect_image(crop, offset=(box[0], box[1]))
        if r is None:
            return None
        r["bbox"] = np.asarray(box[:4], np.float32)
        return r

    def estimate(self, frame: np.ndarray, body_keypoints: np.ndarray, body_scores: np.ndarray,
                 body_bbox=None) -> dict | None:
        """Face for one person: crop from the head keypoints, then the landmarker. None if no face."""
        h, w = frame.shape[:2]
        box = head_crop_bbox(body_keypoints, body_scores, h, w, body_bbox=body_bbox,
                             conf_thr=self._conf_thr, scale=self._crop_scale)
        if box is None:
            return None
        return self.estimate_crop(frame, box)


def add_face_backend_args(ap) -> None:
    """--face-backend / --face-device for the scripts that take --face."""
    ap.add_argument("--face-backend", default="mediapipe", choices=FACE_BACKENDS,
                    help="face models on MediaPipe Tasks (CPU, default) or OpenVINO (src/face_ov)")
    ap.add_argument("--face-device", default="CPU", type=str.upper, choices=FACE_DEVICES,
                    help="with --face-backend openvino: OpenVINO device (default CPU)")


class FaceStage:
    """Run-time face tracking switch. The face model loads the first time it is switched on."""

    def __init__(self, model_path=None, on=False, preprocess=None, backend="mediapipe", device="CPU"):
        from src.face_preprocess import DEFAULT_MODE
        self.model_path, self.est, self.error, self.pre = model_path, None, None, None
        self.preprocess = preprocess or DEFAULT_MODE
        self.backend, self.device = backend, device
        self.on = False
        if on:
            self.toggle()

    def toggle(self) -> bool:
        if self.on:
            self.on = False
            return False
        if self.est is None:
            try:
                from src.face_preprocess import FacePreprocessor
                self.est = FaceEstimator(model_path=self.model_path, backend=self.backend, device=self.device)
                self.pre = FacePreprocessor(self.est, self.preprocess)
                self.error = None
                print(f"[viewer] face model loaded: {self.est.model_path} ({self.est.runtime})")
            except Exception as e:      # missing runtime or model file: stay off, say why
                self.error = str(e).splitlines()[0][:120]
                print(f"[viewer] face tracking unavailable: {self.error}")
                return False
        self.on = True
        return True

    def run(self, frame, body_keypoints=None, body_scores=None, body_bbox=None) -> dict | None:
        """Face for the subject with the chosen preprocessing (src/face_preprocess.py); None if off or no face."""
        if not self.on or self.pre is None:
            return None
        return self.pre.run(frame, body_keypoints, body_scores, body_bbox)

    def close(self):
        if self.est is not None:
            self.est.close()


def _align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2D similarity (Umeyama) that maps src onto dst; returns src aligned to dst. src, dst: (N, 2)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    va = (a ** 2).sum()
    if va <= 0:
        return src
    u, s, vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(u @ vt)) or 1.0
    D = np.diag([1.0, d])
    R = u @ D @ vt
    c = (s * np.diag(D)).sum() / va
    return (c * (R @ a.T)).T + md


def face_region_speeds(prev: np.ndarray | None, cur: np.ndarray | None, dt: float) -> dict[str, float]:
    """Per-region face motion between two frames, head motion removed, in face widths per second.

    Both frames' 2D landmarks are aligned with a similarity transform (translation, rotation, scale)
    fitted on all 478 points, so rigid head movement is removed and what remains is movement of the
    face itself (mouth, eyelids, one side of the face). The speed is the mean landmark displacement in
    the region divided by the face width (landmarks 234-454) and by dt. NaN when either frame has no face.
    """
    out = {r: float("nan") for r in FACE_REGIONS}
    if prev is None or cur is None or dt <= 0:
        return out
    p, c = np.asarray(prev, float)[:, :2], np.asarray(cur, float)[:, :2]
    if not (np.isfinite(p).all() and np.isfinite(c).all()):
        return out
    fw = float(np.linalg.norm(c[FACE_WIDTH_PAIR[0]] - c[FACE_WIDTH_PAIR[1]]))
    if fw <= 1e-6:
        return out
    pa = _align(p, c)
    disp = np.linalg.norm(c - pa, axis=1)
    for r, idx in FACE_REGIONS.items():
        out[r] = float(disp[idx].mean() / fw / dt)
    return out


class FaceMotion:
    """Scrolling per-region face motion for one tracked subject (mouth, l_eye, r_eye, l_face, r_face)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._prev, self._t = None, None
        self.t: list[float] = []
        self.speeds: dict[str, list[float]] = {r: [] for r in FACE_REGIONS}

    def update(self, t: float, landmarks: np.ndarray | None) -> dict[str, float]:
        lm = None if landmarks is None or not np.isfinite(landmarks).all() else landmarks
        sp = face_region_speeds(self._prev, lm, (t - self._t) if self._t is not None else 0.0)
        self._prev, self._t = lm, t
        self.t.append(float(t))
        for r in FACE_REGIONS:
            self.speeds[r].append(sp[r])
        return sp
