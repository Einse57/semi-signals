"""LivePortrait reenactment (`liveportrait` backend of src/deid_swap.py), CPU. RESEARCH USE ONLY.

LivePortrait (github.com/KlingTeam/LivePortrait) animates a still source portrait with the motion of a
driving face. Here the donor image is the source and each subject face drives it (per face track, motion
relative to the track's first frame, LivePortrait's default "pose-friendly" image-to-video settings with
stitching). The animated donor crop (512 px) is aligned to the subject's face with a similarity transform
fitted between its MediaPipe landmarks and the subject's, and alpha-blended inside the feathered subject
face hull, so hair, ears, body and background stay the subject's.

LivePortrait's code and weights are NOT part of this repository: the configuration points at a local,
git-ignored checkout and at weights fetched with pinned sha256 (docs/FACE_SWAP_LICENCES.md). Its
detection/landmark models are InsightFace non-commercial research models, so output made with this
backend is for research only and is marked "reduced risk, not de-identified".

Differences from upstream inference.py: float32 on CPU; crops come from the MediaPipe landmarks handed to
`swap` (5 points through LivePortrait's own crop_image, crop sizes and ratios from its CropConfig) instead
of InsightFace detection and the 203-point landmark model; lip normalisation (which needs the 203 points)
is off; frames are processed one at a time, so the driving motion is not smoothed.

Configuration:

    {"backend": "liveportrait", "licence_reviewed": true, "research_use_acknowledged": true,
     "models": {"code": "/path/third_party_local/LivePortrait",
                "weights": "/path/weights/liveportrait"},      # holds liveportrait/base_models/...
     "donor": "/path/donor.png", "donor_landmarks": "/path/donor_landmarks.npy",
     "donor_licence": "...", "device": "cpu"}
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from src.deid_ghost import five_points, _umeyama


def _import_liveportrait(code: str) -> dict:
    """Import LivePortrait's `src` package under another name (this repository's package is also `src`)."""
    saved = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
    for k in saved:
        del sys.modules[k]
    n_threads = cv2.getNumThreads()
    import importlib.machinery
    import importlib.util
    import types
    pkg = types.ModuleType("src")                      # LivePortrait's src/ is a namespace package
    pkg.__path__ = [str(Path(code) / "src")]
    pkg.__spec__ = importlib.machinery.ModuleSpec("src", None, is_package=True)
    pkg.__spec__.submodule_search_locations = pkg.__path__
    sys.modules["src"] = pkg
    sys.path.insert(0, code)
    try:
        import src.live_portrait_wrapper as w        # noqa: F401  (LivePortrait's package)
        import src.config.inference_config as ic
        import src.utils.crop as crop
        import src.utils.camera as cam
        mods = dict(wrapper=w, config=ic, crop=crop, camera=cam)
    finally:
        sys.path.remove(code)
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            sys.modules["liveportrait_" + k] = sys.modules.pop(k)
        sys.modules.update(saved)
        cv2.setNumThreads(n_threads)                  # LivePortrait's crop.py forces OpenCV to one thread
        cv2.ocl.setUseOpenCL(False)
    return mods


class LivePortraitSwapper:
    name = "liveportrait"

    def __init__(self, cfg: dict):
        import torch
        self.torch = torch
        m = cfg["models"]
        code = str(Path(m["code"]).resolve())
        wd = Path(m["weights"])
        self.lp = lp = _import_liveportrait(code)
        b = wd / "liveportrait"
        icfg = lp["config"].InferenceConfig(
            checkpoint_F=str(b / "base_models/appearance_feature_extractor.pth"),
            checkpoint_M=str(b / "base_models/motion_extractor.pth"),
            checkpoint_G=str(b / "base_models/spade_generator.pth"),
            checkpoint_W=str(b / "base_models/warping_module.pth"),
            checkpoint_S=str(b / "retargeting_models/stitching_retargeting_module.pth"),
            flag_force_cpu=cfg.get("device", "cpu") == "cpu", flag_use_half_precision=False)
        self.w = lp["wrapper"].LivePortraitWrapper(inference_cfg=icfg)
        self.donor_licence = cfg.get("donor_licence")
        donor = cv2.imread(str(cfg["donor"]))
        if donor is None:
            raise ValueError(f"cannot read donor image {cfg['donor']}")
        dlm = np.load(cfg["donor_landmarks"]) if cfg.get("donor_landmarks") else self._detect(donor)
        # source crop: CropConfig defaults dsize 512, scale 2.3, vy_ratio -0.125, rotation on
        c = lp["crop"].crop_image(donor[:, :, ::-1], five_points(dlm), dsize=512, scale=2.3, vy_ratio=-0.125,
                                  flag_do_rot=True)
        crop256 = cv2.resize(c["img_crop"], (256, 256))
        I_s = self.w.prepare_source(crop256)
        self.x_s_info = self.w.get_kp_info(I_s)
        self.R_s = lp["camera"].get_rotation_matrix(self.x_s_info["pitch"], self.x_s_info["yaw"],
                                                    self.x_s_info["roll"])
        self.f_s = self.w.extract_feature_3d(I_s)
        self.x_s = self.w.transform_keypoint(self.x_s_info)
        from src.face_estimator import FaceEstimator
        self.est = FaceEstimator(num_faces=1)
        self.tracks: list[dict] = []

    @staticmethod
    def _detect(img):
        from src.face_estimator import FaceEstimator
        est = FaceEstimator(num_faces=1)
        try:
            faces = est.all_faces(img)
        finally:
            est.close()
        if not faces:
            raise ValueError("no face found in the donor image; give donor_landmarks")
        return faces[0]

    def close(self):
        self.est.close()

    def reset(self):
        self.tracks = []

    def _track(self, p):
        c, w = p.mean(0), float(np.ptp(p[:, 0]))
        for tr in self.tracks:
            if np.linalg.norm(tr["c"] - c) < 0.5 * max(w, tr["w"]):
                tr["c"], tr["w"] = c, w
                return tr
        tr = dict(c=c, w=w, d0=None, M=None)
        self.tracks.append(tr)
        return tr

    def swap(self, frame, faces):
        out = frame.copy()
        for lm in faces:
            if lm is None:
                continue
            lm = np.asarray(lm, np.float32)
            if not np.isfinite(lm[:, :2]).all():
                continue
            out = self._one(out, lm)
        return out

    def _animate(self, frame, lm, tr):
        torch, lp, w = self.torch, self.lp, self.w
        # driving crop: CropConfig defaults dsize 512, scale_crop_driving_video 2.2, vy_ratio -0.1, rotation on
        d = lp["crop"].crop_image(frame[:, :, ::-1], five_points(lm), dsize=512, scale=2.2, vy_ratio=-0.1,
                                  flag_do_rot=True)
        I_d = w.prepare_source(cv2.resize(d["img_crop"], (256, 256)))
        info = w.get_kp_info(I_d)
        R_d = lp["camera"].get_rotation_matrix(info["pitch"], info["yaw"], info["roll"])
        if tr["d0"] is None:
            tr["d0"] = (info, R_d)
        info0, R_d0 = tr["d0"]
        s = self.x_s_info
        R_new = (R_d @ R_d0.permute(0, 2, 1)) @ self.R_s
        delta = s["exp"] + (info["exp"] - info0["exp"])
        scale = s["scale"] * (info["scale"] / info0["scale"])
        t = s["t"] + (info["t"] - info0["t"])
        t[..., 2] = 0
        x_d = scale * (s["kp"] @ R_new + delta) + t
        x_d = w.stitching(self.x_s, x_d)
        out = w.warp_decode(self.f_s, self.x_s, x_d)
        return w.parse_output(out["out"])[0][:, :, ::-1]          # 512x512 BGR, donor crop space

    def _one(self, frame, lm):
        tr = self._track(lm[:468, :2])
        anim = self._animate(frame, lm, tr)
        faces = self.est.all_faces(anim)
        if faces:                                                   # animated face -> subject face
            tr["M"] = _umeyama(np.asarray(faces[0], np.float64)[:468, :2], lm[:468, :2].astype(np.float64))
        elif tr["M"] is None:
            return frame
        h, wd = frame.shape[:2]
        sw = cv2.warpAffine(anim, tr["M"], (wd, h), borderMode=cv2.BORDER_REPLICATE)
        mask = np.zeros((h, wd), np.uint8)
        hull = cv2.convexHull(lm[:468, :2].astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        r = max(3, int(0.06 * np.sqrt(cv2.contourArea(hull) + 1)))
        mask = cv2.erode(mask, np.ones((r, r), np.uint8))
        k = 2 * r + 1
        a = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32)[..., None] / 255.0
        return (a * sw + (1 - a) * frame).astype(np.uint8)
