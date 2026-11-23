"""Exact rest-pose clearance check of a rest-mesh .npz (rest_mannequin, rest_G0_jacket, rest_G0_pants): winding number on EVERY garment vertex.
  python verify_rest.py <rest.npz> <report.json> [--margin-mm 3]       exit 0 = zero inside, 1 = still inside
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "prototypes/P2_clearance_field"))
import clearance_field as CF
ap = argparse.ArgumentParser(); ap.add_argument("rest", type=Path); ap.add_argument("report", type=Path); ap.add_argument("--margin-mm", type=float, default=3.0); a = ap.parse_args()
Z = np.load(HERE.parents[1] / "baseline/capture_current_walk.npz"); R = np.load(a.rest)
V, T, _ = CF.weld(R["rest_mannequin"].astype(float), Z["tri"]); S = CF.TriSurface(V, T, "rest_body")
rep = dict(rest=str(a.rest), body_closed=S.facts["closed"], garments={})
for g in ["G0_jacket", "G0_pants"]:
    P = R[f"rest_{g}"].astype(float)
    w = np.concatenate([S.winding_number(P[i:i + 200], chunk_pairs=600_000) for i in range(0, len(P), 200)])
    d = np.concatenate([S.closest_point(P[i:i + 2000])["distance"] for i in range(0, len(P), 2000)])
    ins = w > 0.5; sd = np.where(ins, -d, d)
    rep["garments"][g] = dict(points=int(len(P)), inside=int(ins.sum()), min_signed_distance_mm=float(sd.min() * 1000),
                              below_margin=int((sd < a.margin_mm / 1000 - 1e-6).sum()), max_abs_winding_dev=float(np.abs(w - np.round(w)).max()))
    print(g, rep["garments"][g], flush=True)
rep["status"] = "PASS" if all(v["inside"] == 0 for v in rep["garments"].values()) else "FAIL"
a.report.write_text(json.dumps(rep, indent=1)); print("REST CHECK", rep["status"]); sys.exit(0 if rep["status"] == "PASS" else 1)
