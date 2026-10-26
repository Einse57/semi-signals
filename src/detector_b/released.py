"""Leaky reference: released VSViG-base weights (trained on all 14 patients) on the same per-second windows."""
import argparse, numpy as np, torch
from src.detector_b.train_lopo import load, predict
from src.vsvig_runner import load_torch_model
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("slim", "patches", "labels", "weights", "out"): ap.add_argument("--" + k, required=True)
    a = ap.parse_args(); dev = torch.device("cuda")
    m = load_torch_model(a.weights).to(dev)
    recs = load(a.slim, a.patches, a.labels)
    np.savez(a.out, **{r["name"]: predict(m, r, dev) for r in recs}); print("done", len(recs))
