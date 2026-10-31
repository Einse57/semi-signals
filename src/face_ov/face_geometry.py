# Derived from MediaPipe (https://github.com/google-ai-edge/mediapipe), Apache-2.0, modified:
#   mediapipe/tasks/cc/vision/face_geometry/libs/geometry_pipeline.cc
#   mediapipe/tasks/cc/vision/face_geometry/libs/procrustes_solver.cc
#   mediapipe/tasks/cc/vision/face_geometry/face_geometry_from_landmarks_graph.cc
#   mediapipe/tasks/cc/vision/face_geometry/proto/{geometry_pipeline_metadata,mesh_3d}.proto
# Copyright The MediaPipe Authors. Re-implemented in Python/numpy; see LICENSES/Apache-2.0.txt and NOTICE.
"""Facial transformation matrix (MediaPipe output_facial_transformation_matrixes) from 478 landmarks.

The canonical face mesh and the Procrustes landmark weights are read at run time from
geometry_pipeline_metadata_landmarks.binarypb inside the user-fetched face_landmarker.task; they are not
stored in this repository.
"""
from __future__ import annotations

import math
import struct

import numpy as np

# face_geometry_from_landmarks_graph.cc: environment used by the Face Landmarker task.
VERTICAL_FOV_DEG = 63.0
NEAR_PLANE = 1.0
FAR_PLANE = 10000.0
N_MESH = 468                 # iris landmarks (468..477) are dropped before the geometry pipeline
_VERTEX_SIZE = 5             # Mesh3d VERTEX_PT: x, y, z, u, v
_COMPACT_EPS = 1e-3          # IsScreenLandmarkListTooCompact


# ---- minimal protobuf wire-format reader (GeometryPipelineMetadata / Mesh3d / WeightedLandmarkRef) ----
def _varint(b: bytes, i: int):
    v = shift = 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 0x7F) << shift
        if c < 0x80:
            return v, i
        shift += 7


def _fields(b: bytes):
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        num, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        elif wt == 2:
            n, i = _varint(b, i)
            v, i = b[i:i + n], i + n
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wt}")
        yield num, wt, v


def parse_metadata(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """GeometryPipelineMetadata -> canonical landmarks (3, N) and weights (N,)."""
    verts: list[float] = []
    basis: list[tuple[int, float]] = []
    for num, wt, v in _fields(data):
        if num == 1 and wt == 2:                          # canonical_mesh (Mesh3d)
            for n2, w2, v2 in _fields(v):
                if n2 == 3 and w2 == 2:                   # packed vertex_buffer
                    verts += struct.unpack(f"<{len(v2) // 4}f", v2)
                elif n2 == 3 and w2 == 5:
                    verts.append(struct.unpack("<f", v2)[0])
        elif num == 2 and wt == 2:                        # procrustes_landmark_basis
            lid, w = 0, 0.0
            for n2, w2, v2 in _fields(v):
                if n2 == 1:
                    lid = v2
                elif n2 == 2:
                    w = struct.unpack("<f", v2)[0]
            basis.append((lid, w))
    vb = np.asarray(verts, np.float32)
    if vb.size == 0 or vb.size % _VERTEX_SIZE:
        raise ValueError("geometry metadata: bad canonical mesh vertex buffer")
    canon = vb.reshape(-1, _VERTEX_SIZE)[:, :3].T.copy()
    weights = np.zeros(canon.shape[1], np.float32)
    for lid, w in basis:
        weights[lid] = w
    if not weights.sum() > 0:
        raise ValueError("geometry metadata: no Procrustes landmark weights")
    return canon, weights


# ---- procrustes_solver.cc: weighted orthogonal Procrustes with uniform scale --------------------------
def solve_weighted_orthogonal(src: np.ndarray, dst: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """4x4 [s*R | t] minimising sum_i w_i |s R src_i + t - dst_i|^2 (src, dst: (3, N))."""
    sw = np.sqrt(weights.astype(np.float64))
    ws, wt = src * sw, dst * sw
    total = float((sw * sw).sum())
    com = (ws * sw).sum(1) / total
    cws = ws - np.outer(com, sw)
    u, _, vt = np.linalg.svd(wt @ cws.T)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        u[:, 2] *= -1
    rot = u @ vt
    scale = float(((rot @ cws) * wt).sum() / (cws * ws).sum())
    rs = scale * rot
    t = ((wt - rs @ ws) * sw).sum(1) / total
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = rs, t
    return m


class FaceGeometry:
    """geometry_pipeline.cc ScreenToMetricSpaceConverter (FACE_LANDMARK_PIPELINE, TOP_LEFT_CORNER origin)."""

    def __init__(self, metadata, vertical_fov_deg: float = VERTICAL_FOV_DEG,
                 near: float = NEAR_PLANE):
        if not isinstance(metadata, bytes):
            with open(metadata, "rb") as f:
                metadata = f.read()
        self.canonical, self.weights = parse_metadata(metadata)
        self.fov, self.near = float(vertical_fov_deg), float(near)

    def _frustum(self, w: int, h: int):
        hn = 2.0 * self.near * math.tan(0.5 * math.radians(self.fov))
        wn = w * hn / h
        return -0.5 * wn, 0.5 * wn, -0.5 * hn, 0.5 * hn

    def _scale(self, pts):
        return float(np.linalg.norm(solve_weighted_orthogonal(self.canonical, pts, self.weights)[:3, 0]))

    def transform(self, lm_norm: np.ndarray, img_w: int, img_h: int) -> np.ndarray | None:
        """Normalised landmarks (>= 468, 3) -> 4x4 pose transform (float32), None if too compact."""
        lm = np.asarray(lm_norm, np.float64)[:N_MESH]
        d = lm[:, :2] - lm[:, :2].mean(0)
        if math.sqrt(float((d * d).sum(1).max())) <= _COMPACT_EPS:
            return None
        left, right, bottom, top = self._frustum(img_w, img_h)
        s = lm.T.copy()                                         # ProjectXY
        s[1] = 1.0 - s[1]
        s *= np.array([right - left, top - bottom, right - left])[:, None]
        s += np.array([left, bottom, 0.0])[:, None]
        depth_offset = float(s[2].mean())

        inter = s.copy()
        inter[2] *= -1                                          # ChangeHandedness
        k1 = self._scale(inter)
        inter = s.copy()
        inter[2] = (inter[2] - depth_offset + self.near) / k1   # MoveAndRescaleZ
        inter[0] *= inter[2] / self.near                        # UnprojectXY
        inter[1] *= inter[2] / self.near
        inter[2] *= -1
        k2 = self._scale(inter)

        m = s.copy()
        m[2] = (m[2] - depth_offset + self.near) / (k1 * k2)
        m[0] *= m[2] / self.near
        m[1] *= m[2] / self.near
        m[2] *= -1
        return solve_weighted_orthogonal(self.canonical, m, self.weights).astype(np.float32)
