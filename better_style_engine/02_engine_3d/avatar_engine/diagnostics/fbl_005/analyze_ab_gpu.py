"""FBL-005 step 2, GPU twin of analyze_ab.py (same rule, same outputs; Colab L4). Added 2026-11-19; analyze_ab.py is unchanged.

Inside = generalised winding number > 0.5 on the position-welded body of the SAME variant: weld() and TriSurface from
prototypes/P2_clearance_field/clearance_field.py (unchanged), the per-row solid-angle sum of fbl_004/gpu_clearance.py (torch, float64,
exact summation, no hierarchy, no distance pre-filter). Two parity checks guard the GPU path:
  1. the packet invariant: variant A must reproduce 424/600 (frame 61) and 425/587 (frame 76), else STOP (as analyze_ab.py);
  2. per variant and garment, the 2048 garment points nearest the body plus every GPU-inside point are recomputed with NumPy; flags must match.
Displacement of a vertex = Euclidean distance (mm) between variant X and variant A at the same frame and index (max, p99, median).
Report only. Recommend nothing (packet FBL-005).
Run:  python -B analyze_ab_gpu.py [--device cuda|cpu]     (needs evals.npz from eval_variants.py in this folder)
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
AE = HERE.parents[1]
sys.path.insert(0, str(AE.parent / "prototypes/P2_clearance_field")); sys.path.insert(0, str(HERE.parent / "fbl_004"))
import clearance_field as CF                           # noqa: E402  unchanged library
import gpu_clearance as GC                             # noqa: E402  winding_numpy / winding_torch (float64)
G = ["G0_jacket", "G0_pants"]
EXPECT_A = {61: {"G0_jacket": 424, "G0_pants": 600}, 76: {"G0_jacket": 425, "G0_pants": 587}}


def disp(x, a):
    d = np.linalg.norm(x.astype(float) - a.astype(float), axis=1) * 1e3
    return dict(max_mm=round(float(d.max()), 4), p99_mm=round(float(np.percentile(d, 99)), 4), median_mm=round(float(np.median(d)), 4),
                vertices_moved_over_0p01mm=int((d > 0.01).sum()), vertices=int(len(d)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"]); ap.add_argument("--rows", type=int, default=1024)
    a = ap.parse_args()
    if a.device == "cuda":
        import torch
        assert torch.cuda.is_available(), "no CUDA device"
    E = np.load(HERE / "evals.npz", allow_pickle=False); tri = np.load(AE / "baseline/capture_current_walk.npz")["tri"]
    res = {}; t_all = time.time()
    for fr in (61, 76):
        for v in "ABC":
            V, T, _ = CF.weld(E[f"{v}_{fr}_body"].astype(float), tri); surf = CF.TriSurface(V, T, "body")
            if not surf.facts["closed"]:
                raise SystemExit(f"STOP: body not closed, variant {v} frame {fr}")
            r = dict(body_closed=surf.facts["closed"], body_boundary_edges=surf.facts["boundary_edges"], inside={}, garment_displacement_vs_A={},
                     seconds={}, numpy_spot_check={})
            for g in G:
                P = E[f"{v}_{fr}_{g}"].astype(float); t0 = time.time()
                wn = GC.winding_torch(surf, P, a.rows, a.device)
                r["inside"][g] = int((wn > 0.5).sum()); r["seconds"][g] = round(time.time() - t0, 2)
                d = np.concatenate([surf.closest_point(P[s:s + 4000])["distance"] for s in range(0, len(P), 4000)])
                sel = np.union1d(np.argsort(d)[:2048], np.nonzero(wn > 0.5)[0][:4096])
                w_ref = GC.winding_numpy(surf, P[sel], max(1, 250_000 // len(surf.T)))
                r["numpy_spot_check"][g] = dict(points=int(len(sel)), flag_mismatches=int(((w_ref > 0.5) != (wn[sel] > 0.5)).sum()),
                                                max_abs_winding_diff=float(np.abs(w_ref - wn[sel]).max()))
                r["garment_displacement_vs_A"][g] = disp(P, E[f"A_{fr}_{g}"])
                print(fr, v, g, r["inside"][g], f"{r['seconds'][g]}s", "spot-check mismatches", r["numpy_spot_check"][g]["flag_mismatches"], flush=True)
            r["body_displacement_vs_A"] = disp(E[f"{v}_{fr}_body"], E[f"A_{fr}_body"]); res[f"{v}_{fr}"] = r
            if v == "A" and r["inside"] != EXPECT_A[fr]:
                (HERE / "ab.json").write_text(json.dumps(res, indent=1)); raise SystemExit(f"STOP: variant A frame {fr} gives {r['inside']}, expected {EXPECT_A[fr]}")
    spot_ok = all(c["flag_mismatches"] == 0 for r in res.values() for c in r["numpy_spot_check"].values())
    out = dict(variants={"A": "body off, garments on (as built)", "B": "all off", "C": "all on"}, frames=[61, 76], expected_A=EXPECT_A, results=res,
               capture_crosscheck=json.loads(str(E["capture_crosscheck"])), as_built_flags=json.loads(str(E["as_built_flags"])),
               scene_sha256_before_after=json.loads(str(E["scene_sha256_before_after"])), method="analyze_ab_gpu.py (torch float64, " + a.device + ")",
               numpy_spot_check_all_match=spot_ok, seconds=round(time.time() - t_all, 1))
    (HERE / "ab.json").write_text(json.dumps(out, indent=1))
    L = ["# FBL-005 preserve-volume A/B", "", "Report only. Recommend nothing (packet FBL-005).", "",
         f"Scene sha256 before/after: `{out['scene_sha256_before_after'][0][:16]}...` / `{out['scene_sha256_before_after'][1][:16]}...` (unchanged: {out['scene_sha256_before_after'][0] == out['scene_sha256_before_after'][1]}).",
         f"Variant A reproduces the packet invariant: frame 61 = {res['A_61']['inside']}, frame 76 = {res['A_76']['inside']}.",
         f"GPU winding numbers cross-checked with NumPy on the points nearest the body: all flags match = {spot_ok}.", "",
         "| Frame | Variant | Jacket inside | Pants inside | Jacket move vs A max / p99 / median (mm) | Pants move vs A max / p99 / median (mm) | Body move vs A max / p99 (mm) |",
         "|---|---|---|---|---|---|---|"]
    for fr in (61, 76):
        for v in "ABC":
            r = res[f"{v}_{fr}"]; j, p, b = r["garment_displacement_vs_A"]["G0_jacket"], r["garment_displacement_vs_A"]["G0_pants"], r["body_displacement_vs_A"]
            L.append(f"| {fr} | {v} | {r['inside']['G0_jacket']} | {r['inside']['G0_pants']} | {j['max_mm']} / {j['p99_mm']} / {j['median_mm']} | "
                     f"{p['max_mm']} / {p['p99_mm']} / {p['median_mm']} | {b['max_mm']} / {b['p99_mm']} |")
    L += ["", "Variants: A = as built (body linear, garments preserve-volume), B = all linear, C = all preserve-volume. Only the Armature modifier "
          "flag `use_deform_preserve_volume` was changed, in memory (eval_variants.py); the scene file was never saved."]
    (HERE / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({k: v["inside"] for k, v in res.items()}, indent=1)); print("numpy spot check all match:", spot_ok)


if __name__ == "__main__":
    main()
