"""Landmark-driven face swap (`warp` backend of src/deid_swap.py): OpenCV and numpy only, no learned weights.

Per face and frame, a synthetic donor face is warped onto the subject's MediaPipe landmarks: the 898
triangles of the MediaPipe canonical face mesh (Apache-2.0) are mapped piecewise-affinely from the donor
to the subject (back-facing triangles skipped, far triangles drawn first), the warped donor is matched to
the subject's face colour in Lab (mean and standard deviation) and blended in with Poisson (seamless)
cloning. The donor's irises are redrawn at the subject's iris landmarks, so gaze follows the subject.

Geometry (cfg "geometry"): "donor" (default) places the donor's eyes, brows, nose and mouth in the donor's
proportions and adds the subject's expression (each landmark's deviation from the subject's running mean
shape, head pose removed), keeping the subject's face outline; "subject" warps onto the subject's own
landmark positions.

What carries over from the subject: face outline, expression, head pose, gaze, skin colour (mean and
spread), everything outside the face (hair, ears, neck, body) and, with geometry "subject", the face's
proportions.

`render_donor` makes a synthetic donor from the canonical mesh alone (shaded mesh, painted eyes, brows and
lips): no photograph and no image model.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

N_MESH = 468
R_IRIS, L_IRIS = (468, 469, 470, 471, 472), (473, 474, 475, 476, 477)   # centre first


def load_mesh(path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (468, 3) and triangles (898, 3) of mediapipe/modules/face_geometry/data/canonical_face_model.obj."""
    v, f = [], []
    for line in Path(path).read_text().splitlines():
        t = line.split()
        if t and t[0] == "v":
            v.append([float(x) for x in t[1:4]])
        elif t and t[0] == "f":
            f.append([int(x.split("/")[0]) - 1 for x in t[1:4]])
    v, f = np.asarray(v, np.float32), np.asarray(f, np.int32)
    if v.shape != (N_MESH, 3) or f.ndim != 2 or f.max() >= N_MESH:
        raise ValueError(f"{path}: not the MediaPipe canonical face mesh ({v.shape}, {f.shape})")
    return v, f


def _loops(name: str) -> list[list[int]]:
    """Ordered index cycles of a MediaPipe FaceLandmarksConnections contour."""
    from mediapipe.tasks.python.vision import FaceLandmarksConnections as C
    nbr: dict[int, list[int]] = {}
    for e in getattr(C, name):
        nbr.setdefault(e.start, []).append(e.end)
        nbr.setdefault(e.end, []).append(e.start)
    seen, loops = set(), []
    for s in nbr:
        if s in seen:
            continue
        loop, prev, cur = [s], None, s
        seen.add(s)
        while True:
            nxt = [n for n in nbr[cur] if n != prev and (n not in seen or (n == s and len(loop) > 2))]
            if not nxt or nxt[0] == s:
                break
            prev, cur = cur, nxt[0]
            loop.append(cur)
            seen.add(cur)
        loops.append(loop)
    return loops


def _inner_outer(loops, pts):
    """Split two loops into (outer, inner) by enclosed area."""
    a = sorted(loops, key=lambda lp: -abs(cv2.contourArea(pts[lp].astype(np.float32))))
    return a[0], (a[1] if len(a) > 1 else None)


def render_donor(mesh_path, seed: int = 0, size: int = 512, skin=None, iris=None, brow=None, lips=None):
    """Synthetic donor face: (BGR image, landmarks (468, 2) in image px). Deterministic for a seed."""
    rng = np.random.default_rng(seed)
    v, f = load_mesh(mesh_path)
    skin = np.array(skin if skin is not None else rng.choice(
        [(120, 150, 200), (140, 170, 215), (90, 120, 170), (70, 95, 140), (160, 190, 230)]), float)
    iris = np.array(iris if iris is not None else rng.choice([(60, 80, 110), (110, 90, 50), (70, 110, 80)]), float)
    brow = np.array(brow if brow is not None else skin * rng.uniform(0.25, 0.45), float)
    lips = np.array(lips if lips is not None else (skin * np.array([0.75, 0.7, 0.95])).clip(0, 255), float)
    # mild random proportions so donors differ from each other and from the mean shape
    v = v * np.array([rng.uniform(0.92, 1.08), rng.uniform(0.94, 1.06), 1.0], np.float32)
    s = size * 0.8 / (v[:, 0].max() - v[:, 0].min())
    pts = np.stack([size / 2 + s * v[:, 0], size * 0.5 - s * (v[:, 1] - (v[:, 1].max() + v[:, 1].min()) / 2)], 1)
    img = np.full((size, size, 3), 128, np.uint8)
    light = np.array([0.3, 0.45, 1.0]) / np.linalg.norm([0.3, 0.45, 1.0])
    for t in f[np.argsort(v[f, 2].mean(1))]:                     # far to near
        n = np.cross(v[t[1]] - v[t[0]], v[t[2]] - v[t[0]])
        nn = np.linalg.norm(n)
        if nn == 0:
            continue
        n = n / nn
        if n[2] < 0:
            n = -n
        shade = 0.55 + 0.45 * max(0.0, float(n @ light))
        cv2.fillConvexPoly(img, pts[t].round().astype(np.int32), (skin * shade).clip(0, 255).tolist(), cv2.LINE_AA)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), 2.5)
    img = (img.astype(np.float32) * (1 + 0.04 * noise[..., None] / (noise.std() + 1e-6))).clip(0, 255).astype(np.uint8)
    P = lambda lp: pts[lp].round().astype(np.int32)
    for name in ("FACE_LANDMARKS_LEFT_EYEBROW", "FACE_LANDMARKS_RIGHT_EYEBROW"):
        for lp in _loops(name):
            cv2.fillPoly(img, [P(lp)], brow.tolist(), cv2.LINE_AA)
    outer, inner = _inner_outer(_loops("FACE_LANDMARKS_LIPS"), pts)
    cv2.fillPoly(img, [P(outer)], lips.tolist(), cv2.LINE_AA)
    if inner is not None:
        cv2.fillPoly(img, [P(inner)], (lips * 0.35).tolist(), cv2.LINE_AA)
    for name in ("FACE_LANDMARKS_LEFT_EYE", "FACE_LANDMARKS_RIGHT_EYE"):
        for lp in _loops(name)[:1]:
            cv2.fillPoly(img, [P(lp)], (225, 230, 235), cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    return img, pts.astype(np.float32), dict(iris_bgr=[int(x) for x in iris])


def _eye_loops():
    return [_loops("FACE_LANDMARKS_RIGHT_EYE")[0], _loops("FACE_LANDMARKS_LEFT_EYE")[0]]


class WarpSwapper:
    """FaceSwapper for backend "warp"; cfg as in src/deid_swap.py with models.face_mesh, donor and
    donor_landmarks (.npy, (468, 2) donor px; detected with the Face Landmarker if absent)."""
    name = "warp"

    def __init__(self, cfg: dict):
        _, self.tri = load_mesh(cfg["models"]["face_mesh"])
        self.donor = cv2.imread(str(cfg["donor"]))
        if self.donor is None:
            raise ValueError(f"cannot read donor image {cfg['donor']}")
        lmp = cfg.get("donor_landmarks")
        if lmp:
            self.dlm = np.load(lmp).astype(np.float32)[:N_MESH, :2]
        else:
            from src.face_estimator import FaceEstimator
            est = FaceEstimator(num_faces=1)
            try:
                faces = est.all_faces(self.donor)
            finally:
                est.close()
            if not faces:
                raise ValueError("no face found in the donor image; give donor_landmarks")
            self.dlm = faces[0][:N_MESH, :2].astype(np.float32)
        self.iris_bgr = tuple(cfg.get("iris_bgr", (70, 85, 110)))
        self.donor_licence = cfg.get("donor_licence")
        self.eyes = _eye_loops()
        # geometry "donor" (default): the donor's face proportions with the subject's outline, pose and
        # expression; "subject": the subject's own landmark positions (keeps the subject's face shape)
        self.geometry = cfg.get("geometry", "donor")
        if self.geometry not in ("donor", "subject"):
            raise ValueError(f"geometry must be donor or subject, not {self.geometry!r}")
        self.mix = float(cfg.get("geometry_mix", 1.0))
        self.oval = sorted({i for lp in _loops("FACE_LANDMARKS_FACE_OVAL") for i in lp})
        self.tracks: list[dict] = []
        d = self.dlm[self.tri]                                   # (T, 3, 2) donor triangles
        self.donor_sign = np.sign(_area2(d))
        self.dsrc = np.concatenate([d, np.ones((len(d), 3, 1), np.float32)], 2)   # for the affine solve

    def close(self):
        pass

    def reset(self):
        self.tracks = []

    def swap(self, frame, faces):
        out = frame.copy()
        for lm in faces:
            if lm is not None and np.isfinite(np.asarray(lm)[:N_MESH, :2]).all():
                lm = np.asarray(lm, np.float32)
                tgt = self._target(lm) if self.geometry == "donor" else lm[:, :2]
                out = self._one(out, lm, tgt)
        return out

    def _track(self, p):
        """Running mean of this face's shape in the donor frame (its neutral estimate), matched by position."""
        c, w = p.mean(0), float(np.ptp(p[:, 0]))
        for tr in self.tracks:
            if np.linalg.norm(tr["c"] - c) < 0.5 * max(w, tr["w"]):
                tr["c"], tr["w"] = c, w
                return tr
        tr = dict(c=c, w=w, sum=np.zeros((N_MESH, 2), np.float64), n=0)
        self.tracks.append(tr)
        return tr

    def _target(self, lm):
        """Target positions (478, 2): donor proportions + the subject's expression (deviation from its running
        mean shape), in the subject's pose; face-oval points stay on the subject's outline."""
        p = lm[:N_MESH, :2].astype(np.float64)
        A, t = _sim(p, self.dlm.astype(np.float64))              # subject -> donor frame
        q = p @ A.T + t
        tr = self._track(p)
        tr["sum"] += q
        tr["n"] += 1
        neutral = tr["sum"] / tr["n"]
        tq = (1 - self.mix) * q + self.mix * (self.dlm + (q - neutral))
        tq[self.oval] = q[self.oval]
        Ai = np.linalg.inv(A)
        tp = (tq - t) @ Ai.T
        out = lm[:, :2].astype(np.float64).copy()
        out[:N_MESH] = tp
        if lm.shape[0] >= 478:                                   # irises: same place relative to the eye corners
            for (a, b), ir in (((33, 133), R_IRIS), ((362, 263), L_IRIS)):
                e, e2 = p[b] - p[a], tp[b] - tp[a]
                n2 = float(e @ e)
                if n2 <= 0:
                    continue
                for k in ir:
                    d = lm[k, :2] - p[a]
                    u, v = (d @ e) / n2, (d[0] * e[1] - d[1] * e[0]) / n2
                    out[k] = tp[a] + u * e2 + v * np.array([e2[1], -e2[0]])
        return out.astype(np.float32)

    def _one(self, frame, lm, tgt):
        """Warp the donor onto target positions tgt (478 or 468, 2); lm gives depth order (z)."""
        H, W = frame.shape[:2]
        p = np.concatenate([tgt[:N_MESH], lm[:N_MESH, :2]])      # bbox covers both target and subject
        x0, y0 = np.floor(p.min(0)).astype(int) - 2
        x1, y1 = np.ceil(p.max(0)).astype(int) + 3
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return frame
        q = tgt[:N_MESH] - np.array([x0, y0], np.float32)
        t = q[self.tri]                                          # (T, 3, 2) subject triangles
        vis = (np.sign(_area2(t)) == self.donor_sign) & (np.abs(_area2(t)) > 1e-3)
        z = lm[:N_MESH, 2][self.tri].mean(1) if lm.shape[1] > 2 else np.zeros(len(self.tri))
        order = np.where(vis)[0][np.argsort(-z[vis])]            # far first, near overwrite
        ids = np.full((y1 - y0, x1 - x0), -1, np.int32)
        for i in order:
            cv2.fillConvexPoly(ids, t[i].round().astype(np.int32), int(i))
        m = ids >= 0
        if m.sum() < 16:
            return frame
        # affine subject -> donor per triangle: A (T, 2, 3) solving [x y 1] A^T = donor
        ts = np.concatenate([t, np.ones((len(t), 3, 1), np.float32)], 2)
        A = np.zeros((len(t), 3, 2), np.float32)
        good = np.abs(np.linalg.det(ts)) > 1e-6
        A[good] = np.linalg.solve(ts[good], self.dsrc[good][..., :2])
        yy, xx = np.nonzero(m)
        k = ids[yy, xx]
        Ak = A[k]
        mx = np.full(ids.shape, -1, np.float32)
        my = np.full(ids.shape, -1, np.float32)
        mx[yy, xx] = xx * Ak[:, 0, 0] + yy * Ak[:, 1, 0] + Ak[:, 2, 0]
        my[yy, xx] = xx * Ak[:, 0, 1] + yy * Ak[:, 1, 1] + Ak[:, 2, 1]
        warped = cv2.remap(self.donor, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        roi = frame[y0:y1, x0:x1]
        mask = (m * 255).astype(np.uint8)
        # gaze: the donor's iris at the subject's iris landmarks, inside the subject's eye opening
        if lm.shape[0] >= 478 and tgt.shape[0] >= 478:
            for loop, ir in zip(self.eyes, (R_IRIS, L_IRIS)):
                c = tgt[ir[0]] - (x0, y0)
                r = float(np.linalg.norm(tgt[list(ir[1:])] - tgt[ir[0]], axis=1).mean())
                if r >= 0.5:
                    em = np.zeros_like(mask)
                    cv2.fillPoly(em, [q[loop].round().astype(np.int32)], 255)
                    disc = np.zeros_like(mask)
                    cv2.circle(disc, tuple(np.round(c).astype(int)), max(1, int(round(r))), 255, -1, cv2.LINE_AA)
                    sel = (em > 0) & (disc > 0)
                    warped[sel] = self.iris_bgr
                    pupil = np.zeros_like(mask)
                    cv2.circle(pupil, tuple(np.round(c).astype(int)), max(1, int(round(r * 0.45))), 255, -1)
                    warped[(em > 0) & (pupil > 0)] = (20, 20, 20)
        warped = _match_colour(warped, roi, cv2.erode(mask, np.ones((5, 5), np.uint8)))
        inner = cv2.erode(mask, np.ones((3, 3), np.uint8))
        out = frame.copy()
        pad = 4                                                  # seamlessClone needs the mask off the border
        dst = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        src = cv2.copyMakeBorder(warped, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        msk = cv2.copyMakeBorder(inner, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        ys, xs = np.nonzero(msk)
        if len(xs) < 16:
            return frame
        bx, by, bw, bh = cv2.boundingRect(np.stack([xs, ys], 1).astype(np.int32))
        try:
            blended = cv2.seamlessClone(src, dst, msk, (bx + bw // 2, by + bh // 2), cv2.NORMAL_CLONE)
        except cv2.error:
            blended = dst.copy()
            blended[msk > 0] = src[msk > 0]
        out[y0:y1, x0:x1] = blended[pad:-pad, pad:-pad]
        return out


def _sim(src, dst):
    """Similarity (A = s R, t) with dst ~ src @ A.T + t (Umeyama, no reflection)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    va = (a ** 2).sum()
    u, sv, vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(u @ vt)) or 1.0
    D = np.diag([1.0, d])
    R = u @ D @ vt
    c = (sv * np.diag(D)).sum() / va if va > 0 else 1.0
    A = c * R
    return A, md - ms @ A.T


def _area2(t):
    return (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1])


def _match_colour(src, ref, mask):
    """Match src's Lab mean and standard deviation inside mask to ref's."""
    sel = mask > 0
    if sel.sum() < 16:
        return src
    a = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    ma, sa = a[sel].mean(0), a[sel].std(0) + 1e-3
    mb, sb = b[sel].mean(0), b[sel].std(0)
    a = (a - ma) * (sb / sa) + mb
    return cv2.cvtColor(a.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
