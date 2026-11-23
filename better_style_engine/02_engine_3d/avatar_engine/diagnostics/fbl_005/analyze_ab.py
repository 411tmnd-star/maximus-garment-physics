"""FBL-005 step 2 (numpy only): inside counts and displacements per variant, from evals.npz.

Inside = generalised winding number > 0.5 on the position-welded body of the SAME variant (fbl_004/fast_clearance.py, exact).
Displacement of a vertex = Euclidean distance (mm) between variant X and variant A at the same frame and index; reported as max, p99, median over all vertices.
Run:  python -I -B analyze_ab.py [--workers 3]
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fbl_004"))
import fast_clearance as FC                          # noqa: E402
CF = FC.CF
AE = HERE.parents[1]
G = ["G0_jacket", "G0_pants"]
EXPECT_A = {61: {"G0_jacket": 424, "G0_pants": 600}, 76: {"G0_jacket": 425, "G0_pants": 587}}


def disp(x, a):
    d = np.linalg.norm(x.astype(float) - a.astype(float), axis=1) * 1e3
    return dict(max_mm=round(float(d.max()), 4), p99_mm=round(float(np.percentile(d, 99)), 4), median_mm=round(float(np.median(d)), 4), vertices_moved_over_0p01mm=int((d > 0.01).sum()), vertices=int(len(d)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=3); a = ap.parse_args()
    E = np.load(HERE / "evals.npz", allow_pickle=False); tri = np.load(AE / "baseline/capture_current_walk.npz")["tri"]
    import multiprocessing as mp
    res = {}
    for fr in (61, 76):
        for v in "ABC":
            V, T, _ = CF.weld(E[f"{v}_{fr}_body"].astype(float), tri); surf = CF.TriSurface(V, T, "body")
            if not surf.facts["closed"]: raise SystemExit(f"STOP: body not closed, variant {v} frame {fr}")
            pool = mp.get_context("spawn").Pool(a.workers, initializer=FC._wk_init, initargs=(V, T)) if a.workers > 1 else None
            r = dict(body_closed=surf.facts["closed"], body_boundary_edges=surf.facts["boundary_edges"], inside={}, garment_displacement_vs_A={}, seconds={})
            for g in G:
                P = E[f"{v}_{fr}_{g}"].astype(float); ck = HERE / f"winding_{v}_{fr}_{g}.npz"; t0 = time.time()
                if ck.exists(): wn = np.load(ck)["wn"]
                else:
                    wn = FC.winding_parallel(pool, a.workers, P, FC.DEFAULT_CHUNK_PAIRS, len(surf.T))[0] if pool else FC.winding_chunked(surf, P, FC.DEFAULT_CHUNK_PAIRS, time.time() + 1e9, [])[0]
                    np.savez(ck, wn=wn)
                r["inside"][g] = int((wn > 0.5).sum()); r["seconds"][g] = round(time.time() - t0, 1)
                r["garment_displacement_vs_A"][g] = disp(P, E[f"A_{fr}_{g}"])
                print(fr, v, g, r["inside"][g], f"{r['seconds'][g]}s", flush=True)
            r["body_displacement_vs_A"] = disp(E[f"{v}_{fr}_body"], E[f"A_{fr}_body"])
            if pool: pool.close(); pool.join()
            res[f"{v}_{fr}"] = r
            if v == "A" and r["inside"] != EXPECT_A[fr]:
                (HERE / "ab.json").write_text(json.dumps(res, indent=1)); raise SystemExit(f"STOP: variant A frame {fr} gives {r['inside']}, expected {EXPECT_A[fr]}")
    out = dict(variants={"A": "body off, garments on (as built)", "B": "all off", "C": "all on"}, frames=[61, 76], expected_A=EXPECT_A, results=res,
               capture_crosscheck=json.loads(str(E["capture_crosscheck"])), as_built_flags=json.loads(str(E["as_built_flags"])))
    (HERE / "ab.json").write_text(json.dumps(out, indent=1)); print(json.dumps({k: v["inside"] for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
