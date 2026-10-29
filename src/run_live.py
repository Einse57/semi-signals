"""Stage 1 entry-point: live webcam skeleton tracking + motion chart."""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from src.detector import PersonDetector
from src.pose_estimator import PoseEstimator
from src.hand_estimator import HandEstimator
from src.motion import MotionQuantifier
from src.display import (
    compose_display,
    draw_angles_overlay,
    draw_hand_skeleton,
    draw_region_chart,
    draw_face_mesh,
    draw_skeleton,
)
from src.textdraw import put_block


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1 – live skeleton tracking")
    p.add_argument("--cam", type=int, default=0,
                   help="Webcam index (default: 0)")
    p.add_argument("--input", type=str, default=None,
                   help="Path to video file (overrides --cam)")
    p.add_argument("--list-cams", action="store_true",
                   help="List available webcams and exit")
    p.add_argument("--backend", type=str, default="onnxruntime",
                   choices=["onnxruntime", "openvino", "opencv"],
                   help="ONNX inference backend")
    p.add_argument("--det-interval", type=int, default=5,
                   help="Run detection every N frames (reuse bboxes otherwise)")
    p.add_argument("--det-score", type=float, default=0.5)
    p.add_argument("--kpt-score", type=float, default=0.3)
    p.add_argument("--no-hands", action="store_true",
                   help="Disable hand/finger tracking (faster, wrist-only)")
    p.add_argument("--face", action="store_true",
                   help="Start with face tracking on (MediaPipe Face Landmarker on a head crop; key 'm')")
    p.add_argument("--no-face-mesh", action="store_true",
                   help="With face tracking: draw contours only, no mesh")
    p.add_argument("--face-preprocess", default="none", choices=["none", "detect", "coco-head"],
                   help="none = the frame is the face region (default); detect = find, crop and align; "
                        "coco-head = crop from the body head keypoints")
    p.add_argument("--face-model", default=None,
                   help="face_landmarker.task (default models/face_landmarker.task)")
    from src.face_estimator import add_face_backend_args
    add_face_backend_args(p)
    return p.parse_args()


def _list_cameras(max_index: int = 10) -> list[int]:
    """Probe Windows camera indices using DirectShow."""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            available.append(i)
            print(f"  Camera {i}: {w}x{h}")
            cap.release()
    return available


def main() -> None:
    args = parse_args()

    if args.list_cams:
        print("[INFO] Scanning for cameras...")
        cams = _list_cameras()
        if not cams:
            print("  No cameras found.")
        sys.exit(0)

    is_video_file = args.input is not None
    if is_video_file:
        source = args.input
        cap = cv2.VideoCapture(source)
    else:
        source = args.cam
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if is_video_file:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = total / src_fps if src_fps else 0
        print(f"[INFO] Video: {source} ({src_w}x{src_h}, {src_fps:.0f} fps, {dur:.1f}s)")
    else:
        print(f"[INFO] Webcam {source}: {src_w}x{src_h}")

    print("[INFO] Loading models...")
    from src.config import INFERENCE_DEVICE, INFERENCE_BACKEND
    print(f"[INFO] Inference: backend={INFERENCE_BACKEND}, device={INFERENCE_DEVICE}")
    detector = PersonDetector()
    pose_est = PoseEstimator()
    hand_est = HandEstimator() if not args.no_hands else None
    hands_enabled = not args.no_hands
    motion_q = MotionQuantifier()

    # Warmup: run one dummy inference so first real frame isn't slow
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    detector.detect(dummy)
    pose_est.estimate(dummy, [np.array([0, 0, 64, 64, 1.0], dtype=np.float32)])
    if hand_est is not None:
        dummy_kpts = np.zeros((17, 2), dtype=np.float32)
        dummy_scores = np.zeros(17, dtype=np.float32)
        hand_est.estimate(dummy, dummy_kpts, dummy_scores)
    status = "ON" if hands_enabled else "OFF"
    print(f"[INFO] Models warmed up. Hands: {status}.")
    print("[INFO] Keys: 'q' quit | 'h' toggle hands | 'm' toggle face | 'c' cycle camera")
    from src.face_estimator import FaceMotion, FaceStage
    from src.config import FACE_CHART_GROUPS
    face = FaceStage(args.face_model, on=args.face, preprocess=args.face_preprocess,
                     backend=args.face_backend, device=args.face_device)    # model loads on first enable
    face_mot = FaceMotion()

    frame_idx = 0
    cached_bboxes: list[np.ndarray] = []
    fps_t0 = time.perf_counter()
    fps_val = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detect persons (skip-frame: reuse cached bboxes)
        if frame_idx % args.det_interval == 0:
            cached_bboxes = detector.detect(frame)

        # 2. Estimate pose for each person (every frame for smooth tracking)
        pose_results = pose_est.estimate(frame, cached_bboxes)

        # 3. Process best person for motion quantification
        if pose_results:
            best = max(pose_results, key=lambda r: r["bbox"][4])
            kpts = best["keypoints"]
            kpt_scores = best["scores"]

            # 3a. Hand pose (21 keypoints per hand) — toggleable
            hand_results = None
            if hands_enabled and hand_est is not None:
                hand_results = hand_est.estimate(frame, kpts, kpt_scores)
            metrics = motion_q.update(kpts, scores=kpt_scores,
                                       hand_results=hand_results)
            if face.on:
                fr = face.run(frame, kpts, kpt_scores, best["bbox"])
                face_lm = fr["landmarks"] if fr is not None else None
                face_mot.update(time.monotonic(), face_lm)
                draw_face_mesh(frame, face_lm, mesh=not args.no_face_mesh)

            draw_skeleton(frame, kpts, kpt_scores, score_thr=args.kpt_score)
            if hand_results is not None:
                draw_hand_skeleton(frame, hand_results.get("left"),
                                   score_thr=args.kpt_score)
                draw_hand_skeleton(frame, hand_results.get("right"),
                                   score_thr=args.kpt_score)

        # 4. FPS counter
        now = time.perf_counter()
        dt = now - fps_t0
        fps_t0 = now
        if dt > 0:
            fps_val = 0.8 * fps_val + 0.2 * (1.0 / dt)  # smoothed
        mode = "Body" + (" + Hands" if hands_enabled else "") + (" + Face" if face.on else "")
        lines = [(f"FPS: {fps_val:.1f}", (0, 255, 0)), (f"Mode: {mode}", (0, 220, 255))]
        if face.error and not face.on:
            lines.append((f"Face unavailable: {face.error}", (0, 0, 255)))
        put_block(frame, lines, (12, 10), max(16, frame.shape[0] // 30))

        # 5. Compose and show
        chart_img = draw_region_chart(motion_q.region_intensity_history,
                                       chart_h=350,
                                       timestamps=motion_q.timestamps)
        if face.on:
            chart_img = np.vstack([chart_img, draw_region_chart(
                face_mot.speeds, chart_h=200, timestamps=face_mot.t, groups=FACE_CHART_GROUPS)])
        display = compose_display(frame, chart_img)
        cv2.imshow("Semi-Signals | Stage 1", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("h"):
            hands_enabled = not hands_enabled
            if hands_enabled and hand_est is None:
                print("[INFO] Loading hand model...")
                hand_est = HandEstimator()
                dummy_kpts = np.zeros((17, 2), dtype=np.float32)
                dummy_scores = np.zeros(17, dtype=np.float32)
                hand_est.estimate(np.zeros((64,64,3), dtype=np.uint8),
                                  dummy_kpts, dummy_scores)
            motion_q.reset_hands()
            print(f"[INFO] Hand tracking: {'ON' if hands_enabled else 'OFF'}")
        if key == ord("m"):
            face.toggle()
            face_mot._prev = None
            print(f"[INFO] Face tracking: {'ON' if face.on else 'OFF'}")
        if key == ord("c") and not is_video_file:
            # Cycle to next available camera
            cap.release()
            next_cam = (source + 1) % 10
            for _ in range(10):
                test = cv2.VideoCapture(next_cam, cv2.CAP_DSHOW)
                if test.isOpened():
                    cap = test
                    source = next_cam
                    motion_q.reset()
                    cached_bboxes = []
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"[INFO] Switched to camera {source} ({w}x{h})")
                    break
                test.release()
                next_cam = (next_cam + 1) % 10
            else:
                # No other camera found, reopen original
                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                print("[WARN] No other camera found")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    face.close()


if __name__ == "__main__":
    main()
