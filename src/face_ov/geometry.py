# Derived from MediaPipe (https://github.com/google-ai-edge/mediapipe), Apache-2.0, modified:
#   mediapipe/calculators/tflite/ssd_anchors_calculator.cc, tensors_to_detections_calculator.cc,
#   mediapipe/calculators/util/{non_max_suppression,detection_projection,detections_to_rects,
#   rect_transformation,landmark_projection,landmarks_smoothing}_calculator.cc, one_euro_filter.cc,
#   mediapipe/calculators/tensor/image_to_tensor_converter_opencv.cc, tensors_to_landmarks_calculator.cc,
#   mediapipe/tasks/cc/vision/face_landmarker/face_blendshapes_graph.cc (146-landmark subset),
#   face_landmarks_detector_graph.cc, mediapipe/tasks/cc/vision/face_detector/face_detector_graph.cc
# Copyright The MediaPipe Authors. Re-implemented in Python/numpy; see LICENSES/Apache-2.0.txt and NOTICE.
"""Pure-numpy geometry of the MediaPipe Face Landmarker pipeline (no OpenVINO, no MediaPipe needed).

Every function names the MediaPipe file whose behaviour it follows (google-ai-edge/mediapipe,
Apache-2.0, Copyright The MediaPipe Authors). This is an independent re-implementation in Python of
the documented calculator behaviour; constants (anchor options, rect scales, keypoint indices) are taken
from those files. See NOTICE.

Conventions: a "rect" is MediaPipe's NormalizedRect, (x_center, y_center, width, height, rotation),
with centre and size normalised by the image width/height and rotation in radians (clockwise on screen,
y down). Detections are (score, box[xmin, ymin, xmax, ymax], keypoints (K, 2)) in normalised coords.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --- mediapipe/tasks/cc/vision/face_detector/face_detector_graph.cc (short-range defaults) ---------
SHORT_RANGE_ANCHORS = dict(num_layers=4, min_scale=0.1484375, max_scale=0.75, input_size=128,
                           anchor_offset=0.5, strides=(8, 16, 16, 16), aspect_ratios=(1.0,),
                           fixed_anchor_size=True, interpolated_scale_aspect_ratio=1.0)
DET_NUM_BOXES, DET_NUM_COORDS, DET_NUM_KEYPOINTS = 896, 16, 6
DET_SCALE = 128.0                    # x/y/w/h_scale = kShortRangeImageSize
DET_SCORE_CLIP = 100.0               # score_clipping_thresh
DET_RECT_SCALE = 1.5                 # ConfigureRectTransformationCalculator (no square_long)
DET_ROT_KP = (0, 1)                  # rotation_vector_start/end_keypoint_index (eyes)

# --- mediapipe/tasks/cc/vision/face_landmarker/face_landmarks_detector_graph.cc --------------------
LMK_ROT_IDX = (33, 263)              # ConfigureFaceDetectionsToRectsCalculator
LMK_RECT_SCALE = 1.5                 # ConfigureFaceRectTransformationCalculator, square_long = true
ONE_EURO = dict(min_cutoff=0.05, beta=80.0, derivate_cutoff=1.0, frequency=30.0)

# --- mediapipe/tasks/cc/vision/face_landmarker/face_blendshapes_graph.cc (kLandmarksSubsetIdxs) ----
BLENDSHAPE_SUBSET = np.array([
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66,
    67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145,
    146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 172, 173, 176,
    178, 181, 185, 191, 195, 197, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285,
    288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334,
    336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387,
    388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466, 468, 469, 470, 471, 472, 473, 474,
    475, 476, 477], np.int64)
assert BLENDSHAPE_SUBSET.size == 146


@dataclass
class Rect:
    x_center: float
    y_center: float
    width: float
    height: float
    rotation: float = 0.0


def _calc_scale(min_scale, max_scale, idx, n):
    # ssd_anchors_calculator.cc CalculateScale
    return min_scale + (max_scale - min_scale) * idx / (n - 1.0) if n > 1 else (min_scale + max_scale) * 0.5


def ssd_anchors(num_layers=4, min_scale=0.1484375, max_scale=0.75, input_size=128, anchor_offset=0.5,
                strides=(8, 16, 16, 16), aspect_ratios=(1.0,), fixed_anchor_size=True,
                interpolated_scale_aspect_ratio=1.0) -> np.ndarray:
    """Anchors (N, 4) = x_center, y_center, w, h. mediapipe/calculators/tflite/ssd_anchors_calculator.cc
    GenerateAnchors (strides path, no reduce_boxes_in_lowest_layer)."""
    out = []
    layer = 0
    while layer < num_layers:
        ws, hs = [], []
        last = layer
        while last < len(strides) and strides[last] == strides[layer]:
            scale = _calc_scale(min_scale, max_scale, last, len(strides))
            ars, scs = list(aspect_ratios), [scale] * len(aspect_ratios)
            if interpolated_scale_aspect_ratio > 0:
                nxt = 1.0 if last == len(strides) - 1 else _calc_scale(min_scale, max_scale, last + 1, len(strides))
                scs.append(math.sqrt(scale * nxt))
                ars.append(interpolated_scale_aspect_ratio)
            for ar, sc in zip(ars, scs):
                hs.append(sc / math.sqrt(ar))
                ws.append(sc * math.sqrt(ar))
            last += 1
        fm = int(math.ceil(input_size / strides[layer]))
        for y in range(fm):
            for x in range(fm):
                for w, h in zip(ws, hs):
                    out.append(((x + anchor_offset) / fm, (y + anchor_offset) / fm,
                                1.0 if fixed_anchor_size else w, 1.0 if fixed_anchor_size else h))
        layer = last
    return np.asarray(out, np.float32)


def decode_detections(raw_boxes: np.ndarray, raw_scores: np.ndarray, anchors: np.ndarray,
                      min_score: float = 0.5, scale: float = DET_SCALE, clip: float = DET_SCORE_CLIP):
    """mediapipe/calculators/tensor/tensors_to_detections_calculator.cc DecodeBoxes + score path,
    reverse_output_order = true (XYWH), sigmoid_score, score_clipping_thresh. Boxes and keypoints are in
    the normalised coords of the detector input tensor. Returns scores (M,), boxes (M,4), kps (M,K,2)."""
    rb = np.asarray(raw_boxes, np.float32).reshape(-1, DET_NUM_COORDS)
    s = np.clip(np.asarray(raw_scores, np.float64).reshape(-1), -clip, clip)
    s = (1.0 / (1.0 + np.exp(-s))).astype(np.float32)
    keep = s >= min_score
    rb, s, a = rb[keep], s[keep], anchors[keep]
    xc = rb[:, 0] / scale * a[:, 2] + a[:, 0]
    yc = rb[:, 1] / scale * a[:, 3] + a[:, 1]
    w = rb[:, 2] / scale * a[:, 2]
    h = rb[:, 3] / scale * a[:, 3]
    boxes = np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], 1)
    kp = rb[:, 4:4 + 2 * DET_NUM_KEYPOINTS].reshape(-1, DET_NUM_KEYPOINTS, 2)
    kps = np.empty_like(kp)
    kps[..., 0] = kp[..., 0] / scale * a[:, None, 2] + a[:, None, 0]
    kps[..., 1] = kp[..., 1] / scale * a[:, None, 3] + a[:, None, 1]
    return s, boxes, kps


def _iou(box, boxes):
    ix1 = np.maximum(box[0], boxes[:, 0]); iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2]); iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = a + b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def weighted_nms(scores, boxes, kps, iou_thr: float = 0.5, max_out: int | None = None):
    """mediapipe/calculators/util/non_max_suppression_calculator.cc WeightedNonMaxSuppression
    (INTERSECTION_OVER_UNION). Each output keeps the top score of its cluster and the score-weighted
    mean box and keypoints of every candidate with IoU > iou_thr (the top one included)."""
    order = np.argsort(-scores, kind="stable")
    remained = list(order)
    out = []
    while remained:
        top = remained[0]
        idx = np.asarray(remained)
        sim = _iou(boxes[top], boxes[idx])
        cand, rest = idx[sim > iou_thr], idx[~(sim > iou_thr)]
        box, kp = boxes[top].copy(), kps[top].copy()
        if cand.size:
            w = scores[cand][:, None]
            tot = float(w.sum())
            box = (boxes[cand] * w).sum(0) / tot
            kp = (kps[cand] * w[:, :, None]).sum(0) / tot
        out.append((float(scores[top]), box.astype(np.float32), kp.astype(np.float32)))
        if max_out and len(out) >= max_out:
            break
        if rest.size == idx.size:
            break
        remained = list(rest)
    return out


def letterbox_params(img_w: int, img_h: int, size: int = 128):
    """Full-image ROI with keep_aspect_ratio (image_to_tensor_utils.cc PadRoi): the ROI becomes a square
    of side max(W, H) centred on the image. Returns (side_px, pad_x, pad_y) where pad_* are the
    normalised letterbox paddings on each side of the tensor."""
    side = float(max(img_w, img_h))
    return side, (1.0 - img_w / side) / 2.0, (1.0 - img_h / side) / 2.0


def project_detection(box, kps, img_w, img_h):
    """DetectionProjectionCalculator for the letterboxed full-image ROI: tensor-normalised -> image-normalised."""
    side, px, py = letterbox_params(img_w, img_h)
    sx, sy = side / img_w, side / img_h
    b = np.array([(box[0] - px) * sx, (box[1] - py) * sy, (box[2] - px) * sx, (box[3] - py) * sy], np.float32)
    k = np.stack([(kps[:, 0] - px) * sx, (kps[:, 1] - py) * sy], 1).astype(np.float32)
    return b, k


def normalize_radians(a: float) -> float:
    return a - 2 * math.pi * math.floor((a - (-math.pi)) / (2 * math.pi))


def rotation_from_points(p0, p1, img_w, img_h, target=0.0) -> float:
    """detections_to_rects_calculator.cc ComputeRotation (points normalised; computed in pixels)."""
    x0, y0 = p0[0] * img_w, p0[1] * img_h
    x1, y1 = p1[0] * img_w, p1[1] * img_h
    return normalize_radians(target - math.atan2(-(y1 - y0), x1 - x0))


def rect_from_box(box, rotation=0.0) -> Rect:
    """detections_to_rects_calculator.cc RectFromBox (normalised)."""
    w, h = float(box[2] - box[0]), float(box[3] - box[1])
    return Rect(float(box[0]) + w / 2, float(box[1]) + h / 2, w, h, rotation)


def transform_rect(r: Rect, img_w, img_h, scale_x=1.5, scale_y=1.5, shift_x=0.0, shift_y=0.0,
                   square_long=False) -> Rect:
    """rect_transformation_calculator.cc TransformNormalizedRect."""
    w, h, rot = r.width, r.height, r.rotation
    xc, yc = r.x_center, r.y_center
    if rot == 0.0:
        xc += w * shift_x
        yc += h * shift_y
    else:
        xc += (img_w * w * shift_x * math.cos(rot) - img_h * h * shift_y * math.sin(rot)) / img_w
        yc += (img_w * w * shift_x * math.sin(rot) + img_h * h * shift_y * math.cos(rot)) / img_h
    if square_long:
        long_side = max(w * img_w, h * img_h)
        w, h = long_side / img_w, long_side / img_h
    return Rect(xc, yc, w * scale_x, h * scale_y, rot)


def rect_iou(a: Rect, b: Rect) -> float:
    """association_norm_rect_calculator.cc: IoU of two rects as axis-aligned boxes (rotation ignored)."""
    ax0, ay0, bx0, by0 = a.x_center - a.width / 2, a.y_center - a.height / 2, b.x_center - b.width / 2, b.y_center - b.height / 2
    iw = min(ax0 + a.width, bx0 + b.width) - max(ax0, bx0)
    ih = min(ay0 + a.height, by0 + b.height) - max(ay0, by0)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a.width * a.height + b.width * b.height - inter
    return float(inter / union) if union > 0 else 0.0


def detection_to_roi(box, kps, img_w, img_h) -> Rect:
    """Face detector -> expanded face rect (face_detector_graph.cc: DetectionsToRects with eye rotation,
    then RectTransformation scale 1.5)."""
    rot = rotation_from_points(kps[DET_ROT_KP[0]], kps[DET_ROT_KP[1]], img_w, img_h)
    return transform_rect(rect_from_box(box, rot), img_w, img_h, DET_RECT_SCALE, DET_RECT_SCALE)


def landmarks_to_roi(lm_norm: np.ndarray, img_w, img_h) -> Rect:
    """Landmarks -> next-frame ROI (face_landmarks_detector_graph.cc: LandmarksToDetection, then
    DetectionsToRects with rotation 33 -> 263, then RectTransformation scale 1.5, square_long)."""
    x, y = lm_norm[:, 0], lm_norm[:, 1]
    box = np.array([x.min(), y.min(), x.max(), y.max()], np.float32)
    rot = rotation_from_points(lm_norm[LMK_ROT_IDX[0]], lm_norm[LMK_ROT_IDX[1]], img_w, img_h)
    return transform_rect(rect_from_box(box, rot), img_w, img_h, LMK_RECT_SCALE, LMK_RECT_SCALE,
                          square_long=True)


def roi_corners_px(r: Rect, img_w, img_h) -> np.ndarray:
    """cv::boxPoints order (bottom-left, top-left, top-right, bottom-right) of the rotated ROI in pixels,
    as image_to_tensor_converter_opencv.cc builds it (angle in degrees, clockwise with y down)."""
    cx, cy, w, h = r.x_center * img_w, r.y_center * img_h, r.width * img_w, r.height * img_h
    c, s = math.cos(r.rotation), math.sin(r.rotation)
    # corners relative to centre before rotation, in boxPoints order
    rel = np.array([[-w / 2, h / 2], [-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2]], np.float64)
    rot = np.array([[c, -s], [s, c]])
    return (rel @ rot.T + [cx, cy]).astype(np.float32)


def roi_warp_matrix(r: Rect, img_w, img_h, out_w=256, out_h=256) -> np.ndarray:
    """3x3 perspective (affine in practice) from image px to the out_w x out_h tensor (corner mapping of
    image_to_tensor_converter_opencv.cc: boxPoints -> (0,h), (0,0), (w,0), (w,h))."""
    import cv2
    src = roi_corners_px(r, img_w, img_h)
    dst = np.array([[0, out_h], [0, 0], [out_w, 0], [out_w, out_h]], np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def project_landmarks(raw: np.ndarray, r: Rect, img_w, img_h, in_w=256, in_h=256) -> np.ndarray:
    """Raw model landmarks (478, 3) in tensor px -> normalised image landmarks.
    tensors_to_landmarks_calculator.cc (x/in_w, y/in_h, z/in_w), landmark_letterbox_removal (no padding
    here: keep_aspect_ratio is off for the landmark model), landmark_projection_calculator.cc
    (NORM_RECT + IMAGE_DIMENSIONS path: rotated sub-rect matrix, z scaled by the projected unit length,
    see CalculateZScale)."""
    x = raw[:, 0] / in_w - 0.5
    y = raw[:, 1] / in_h - 0.5
    z = raw[:, 2] / in_w
    W, H = r.width * img_w, r.height * img_h
    c, s = math.cos(r.rotation), math.sin(r.rotation)
    px = (c * x * W - s * y * H) + r.x_center * img_w
    py = (s * x * W + c * y * H) + r.y_center * img_h
    # CalculateZScale: length of the projected (0,0)->(1,0) segment in normalised image coords
    z_scale = math.hypot(W * c / img_w, W * s / img_h)
    return np.stack([px / img_w, py / img_h, z * z_scale], 1).astype(np.float32)


def blendshape_input(lm_norm: np.ndarray, img_w, img_h) -> np.ndarray:
    """face_blendshapes_graph.cc: 146-landmark subset, x*W and y*H (LandmarksToTensorCalculator with
    IMAGE_SIZE, attributes X, Y, not flattened) -> (1, 146, 2)."""
    sub = lm_norm[BLENDSHAPE_SUBSET, :2] * np.array([img_w, img_h], np.float32)
    return sub[None].astype(np.float32)


class _LowPass:
    # mediapipe/util/filtering/low_pass_filter.cc
    def __init__(self):
        self.raw = None
        self.stored = None

    def apply(self, v, alpha):
        self.stored = v if self.stored is None else alpha * v + (1 - alpha) * self.stored
        self.raw = v
        return self.stored


class OneEuroLandmarks:
    """mediapipe/calculators/util/landmarks_smoothing_calculator(.cc, _utils.cc) OneEuroFilterImpl with
    mediapipe/util/filtering/one_euro_filter.cc, as configured by face_landmarks_detector_graph.cc
    (min_cutoff 0.05, beta 80, derivate_cutoff 1). Filters landmarks in pixels (z scaled by image width),
    value_scale = 1 / object scale (mean of the landmark bbox width and height). Vectorised over points."""

    def __init__(self, min_cutoff=0.05, beta=80.0, derivate_cutoff=1.0, frequency=30.0):
        self.p = (min_cutoff, beta, derivate_cutoff, frequency)
        self.reset()

    def reset(self):
        self.freq = self.p[3]
        self.last_t = None
        self.x = None       # last filtered
        self.x_raw = None
        self.dx = None

    @staticmethod
    def _alpha(cutoff, freq):
        te = 1.0 / freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def apply(self, lm_norm: np.ndarray, t_us: int, img_w: int, img_h: int) -> np.ndarray:
        min_cutoff, beta, dcut, _ = self.p
        pts = lm_norm.astype(np.float64) * np.array([img_w, img_h, img_w])
        xy = pts[:, :2]
        scale = ((xy[:, 0].max() - xy[:, 0].min()) + (xy[:, 1].max() - xy[:, 1].min())) / 2.0
        if scale < 1e-6:
            return lm_norm
        vscale = 1.0 / scale
        if self.last_t is not None:
            if t_us <= self.last_t:
                return lm_norm
            self.freq = 1.0 / ((t_us - self.last_t) * 1e-6)
        self.last_t = t_us
        if self.x_raw is None:
            dval = np.zeros_like(pts)
        else:
            dval = (pts - self.x_raw) * vscale * self.freq
        a_d = self._alpha(dcut, self.freq)
        self.dx = dval if self.dx is None else a_d * dval + (1 - a_d) * self.dx
        cutoff = min_cutoff + beta * np.abs(self.dx)
        te = 1.0 / self.freq
        a = 1.0 / (1.0 + (1.0 / (2 * np.pi * cutoff)) / te)
        self.x = pts if self.x is None else a * pts + (1 - a) * self.x
        self.x_raw = pts
        return (self.x / np.array([img_w, img_h, img_w])).astype(np.float32)
