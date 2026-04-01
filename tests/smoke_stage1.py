"""Quick smoke test for Stage 1 components."""
import numpy as np

print("[1] Loading detector (YOLOX-m via rtmlib)...")
from src.detector import PersonDetector
det = PersonDetector()
print(f"    model loaded: {det._model is not None}")

frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
bboxes = det.detect(frame)
print(f"    detected {len(bboxes)} persons (random noise → may be 0)")

print("[2] Loading pose estimator (RTMPose-m via rtmlib)...")
from src.pose_estimator import PoseEstimator
pose = PoseEstimator()
print(f"    model loaded: {pose._model is not None}")

# If no detections on random noise, use full-frame bbox
if not bboxes:
    h, w = frame.shape[:2]
    bboxes = [np.array([0, 0, w, h, 1.0], dtype=np.float32)]
    print("    (using full-frame fallback bbox for testing)")

results = pose.estimate(frame, bboxes)
print(f"    pose results: {len(results)} persons")
if results:
    kp = results[0]["keypoints"]
    sc = results[0]["scores"]
    print(f"    keypoints shape: {kp.shape}")
    print(f"    scores shape: {sc.shape}")

print("[3] Motion quantifier...")
from src.motion import MotionQuantifier
mq = MotionQuantifier()
if results:
    metrics = mq.update(results[0]["keypoints"])
    print(f"    total_intensity: {metrics['total_intensity']:.2f}")
    print(f"    angles: {metrics['joint_angles']}")

print("[OK] Stage 1 smoke test passed!")
