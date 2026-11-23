"""FBL-004 gpu_clearance: the base.clearance rule of harness/gates.py on a GPU (Colab L4), resumable.

Rule (unchanged): a garment vertex is inside when the generalised winding number (exact summation,
float64, van Oosterom-Strackee solid angle) of the position-welded body is > 0.5.
weld() and TriSurface come from prototypes/P2_clearance_field/clearance_field.py, imported unchanged.
The per-chunk arithmetic is TriSurface.winding_number written in torch, still float64, no approximation,
no distance pre-filter, no hierarchy. Only the summation order inside a row may differ from numpy, so
winding values can differ in the last bits; the inside flags are checked against every CPU checkpoint.

Backends: --device cuda (Colab), cpu (torch on CPU), numpy (no torch; slow, for local smoke tests).

Run on Colab (from this folder):
    python gpu_clearance.py --out /content/drive/MyDrive/COLAB_RESULTS/clearance
Local smoke test (numpy, first 300 points of frame 61 only):
    python -I -B gpu_clearance.py --device numpy --frames 61 --limit 300 --out <scratch dir>
"""
import sys, os, time, json, hashlib, argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
AE = HERE.parents[1]
SOLVERS = AE.parent
sys.path.insert(0, str(SOLVERS / "prototypes/P2_clearance_field"))
import clearance_field as CF                           # noqa: E402  unchanged library

CAP = AE / "baseline/capture_current_walk.npz"
GARMENTS = ["G0_jacket", "G0_pants"]
GATE_LIMIT = 0                                         # gate_limits.json: 0 inside vertices


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def winding_numpy(surf, P, rows):
    out = np.empty(len(P))
    for s in range(0, len(P), rows):
        p = P[s:s + rows][:, None, :]; a = surf.A[None] - p; b = surf.B[None] - p; c = surf.C[None] - p
        la = np.linalg.norm(a, axis=2); lb = np.linalg.norm(b, axis=2); lc = np.linalg.norm(c, axis=2)
        num = np.einsum("qti,qti->qt", a, np.cross(b, c)); den = la * lb * lc + (a * b).sum(2) * lc + (b * c).sum(2) * la + (c * a).sum(2) * lb
        out[s:s + rows] = np.arctan2(num, den).sum(1) / (2.0 * np.pi)
    return out


def winding_torch(surf, P, rows, device):
    import torch
    A = torch.as_tensor(surf.A, dtype=torch.float64, device=device)[None]
    B = torch.as_tensor(surf.B, dtype=torch.float64, device=device)[None]
    C = torch.as_tensor(surf.C, dtype=torch.float64, device=device)[None]
    Pt = torch.as_tensor(P, dtype=torch.float64, device=device)
    out = torch.empty(len(P), dtype=torch.float64, device=device)
    with torch.no_grad():
        for s in range(0, len(P), rows):
            p = Pt[s:s + rows][:, None, :]; a = A - p; b = B - p; c = C - p
            la = torch.linalg.norm(a, dim=2); lb = torch.linalg.norm(b, dim=2); lc = torch.linalg.norm(c, dim=2)
            num = (a * torch.linalg.cross(b, c, dim=2)).sum(2)
            den = la * lb * lc + (a * b).sum(2) * lc + (b * c).sum(2) * la + (c * a).sum(2) * lb
            out[s:s + rows] = torch.atan2(num, den).sum(1) / (2.0 * np.pi)
    if device == "cuda":
        torch.cuda.synchronize()
    return out.cpu().numpy()


def cpu_reference(fr, ref_dir):
    """Existing CPU results for this frame, as {garment: (winding, inside)}; None if there are none."""
    f = ref_dir / f"inside_f{fr:03d}.npz"
    if f.exists():
        z = np.load(f); return {g: (z[g + "_winding"], z[g + "_inside"]) for g in GARMENTS}, f.name
    if fr == 61 and (ref_dir / "ref_inside_f061.npz").exists():
        z = np.load(ref_dir / "ref_inside_f061.npz")
        return {"G0_jacket": (z["jacket_winding"], z["jacket_inside"]), "G0_pants": (z["pants_winding"], z["pants_inside"])}, "ref_inside_f061.npz"
    return None, None


def run_frame(Z, fr, a, cf):
    t0 = time.time()
    frames = Z["frames"]; k = int(np.nonzero(frames == fr)[0][0]); i = cf.index(fr)
    V, T, _ = CF.weld(Z["body"][k].astype(float), Z["tri"]); surf = CF.TriSurface(V, T, "body")
    arrays = {}; per = {}
    for g in GARMENTS:
        P = Z["garment__" + g][i].astype(float)
        if a.limit:
            P = P[:a.limit]
        t1 = time.time()
        wn = winding_numpy(surf, P, max(1, 250_000 // len(surf.T))) if a.device == "numpy" else winding_torch(surf, P, a.rows, a.device)
        inside = wn > 0.5
        arrays[g + "_winding"] = wn; arrays[g + "_inside"] = inside; arrays[g + "_index"] = np.nonzero(inside)[0]
        per[g] = dict(points=int(len(P)), inside=int(inside.sum()), seconds=round(time.time() - t1, 2))
    meta = dict(frame=fr, device=a.device, rows=a.rows, limit=a.limit, body_triangles=int(len(surf.T)), body_vertices_welded=int(len(V)),
                body_closed=surf.facts["closed"], garments=per, wall_seconds=round(time.time() - t0, 2))
    ref, ref_name = cpu_reference(fr, a.ref_dir) if a.ref_dir else (None, None)
    if ref is None and a.spot_check:
        # no CPU checkpoint for this capture: re-compute with numpy on the points that matter (nearest the body + every GPU-inside point)
        chk = {}
        for g in GARMENTS:
            P = Z["garment__" + g][i].astype(float)[:len(arrays[g + "_winding"])]
            d = np.concatenate([surf.closest_point(P[s:s + 4000])["distance"] for s in range(0, len(P), 4000)])
            sel = np.union1d(np.argsort(d)[:a.spot_check], arrays[g + "_index"][:4096])
            w_ref = winding_numpy(surf, P[sel], max(1, 250_000 // len(surf.T)))
            chk[g] = dict(points_checked=int(len(sel)), inside_flags_identical=bool(((w_ref > 0.5) == arrays[g + "_inside"][sel]).all()),
                          flag_mismatches=int(((w_ref > 0.5) != arrays[g + "_inside"][sel]).sum()), max_abs_winding_diff=float(np.abs(w_ref - arrays[g + "_winding"][sel]).max()))
        meta["cpu_check"] = dict(reference=f"numpy spot check ({a.spot_check} nearest-to-body points + GPU-inside points)", **chk)
    if ref is not None:
        chk = {}
        for g in GARMENTS:
            w_ref, in_ref = ref[g][0][:len(arrays[g + "_winding"])], ref[g][1][:len(arrays[g + "_inside"])]
            chk[g] = dict(inside_flags_identical=bool((in_ref == arrays[g + "_inside"]).all()),
                          flag_mismatches=int((in_ref != arrays[g + "_inside"]).sum()),
                          max_abs_winding_diff=float(np.abs(w_ref - arrays[g + "_winding"]).max()))
        meta["cpu_check"] = dict(reference=ref_name, **chk)
    name = f"inside_f{fr:03d}_gpu.npz" if not a.limit else f"inside_f{fr:03d}_gpu_limit{a.limit}.npz"
    tmp = a.out / (name + ".tmp.npz")
    np.savez(tmp, meta_json=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, a.out / name)                       # checkpoint appears only when complete
    return meta, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="*", default=None, help="default: every clearance frame in the capture")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "numpy"])
    ap.add_argument("--rows", type=int, default=1024, help="garment points per GPU chunk (1024 x 26756 pairs ~ 3 GB float64)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N points per garment (smoke test)")
    ap.add_argument("--capture", type=Path, default=None, help="capture .npz (default: avatar_engine/baseline/capture_current_walk.npz)")
    ap.add_argument("--ref-dir", default=str(HERE), help="folder with CPU checkpoints inside_fNNN.npz; 'none' for a new capture")
    ap.add_argument("--spot-check", type=int, default=1024, help="frames without a checkpoint: numpy re-check of the N points nearest the body (0 = off)")
    a = ap.parse_args()
    a.ref_dir = None if a.ref_dir.lower() == "none" else Path(a.ref_dir)
    global CAP
    if a.capture: CAP = a.capture.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            sys.exit("No CUDA GPU. In Colab: Runtime > Change runtime type > L4 GPU.")
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    cap_sha = sha(CAP)
    Z = np.load(CAP); meta = json.loads(str(Z["meta"])); cf = [int(x) for x in meta["clearance_frames"]]
    frames = a.frames or cf
    summary = dict(capture=CAP.name, capture_sha256=cap_sha, device=a.device, frames={})
    for fr in frames:
        name = f"inside_f{fr:03d}_gpu.npz" if not a.limit else f"inside_f{fr:03d}_gpu_limit{a.limit}.npz"
        if (a.out / name).exists():
            m = json.loads(str(np.load(a.out / name)["meta_json"])); print(f"frame {fr}: checkpoint exists, skipped", flush=True)
        else:
            m, _ = run_frame(Z, fr, a, cf)
        summary["frames"][fr] = m
        g = m["garments"]; chk = m.get("cpu_check")
        line = f"frame {fr:3d}: jacket inside {g['G0_jacket']['inside']:5d}  pants inside {g['G0_pants']['inside']:5d}  {m['wall_seconds']:7.1f}s"
        if chk:
            ok = all(chk[x]["inside_flags_identical"] for x in GARMENTS)
            line += f"  CPU check vs {chk['reference']}: {'MATCH' if ok else 'MISMATCH'} (max |dw| {max(chk[x]['max_abs_winding_diff'] for x in GARMENTS):.1e})"
        print(line, flush=True)
    assert sha(CAP) == cap_sha, "capture changed during the run"
    checks = [m["cpu_check"] for m in summary["frames"].values() if "cpu_check" in m]
    summary["cpu_checks_all_match"] = all(c[g]["inside_flags_identical"] for c in checks for g in GARMENTS) if checks else None
    if not a.limit:
        worst = max(m["garments"][g]["inside"] for m in summary["frames"].values() for g in GARMENTS)
        summary["gate"] = dict(name="base.clearance", limit=GATE_LIMIT, worst_inside=worst, frames_measured=len(summary["frames"]),
                               frames_planned=len(cf), status=("PASS" if worst <= GATE_LIMIT else "FAIL") if set(cf) <= set(summary["frames"]) else "INCOMPLETE")
        print("GATE base.clearance:", summary["gate"]["status"], f"(worst {worst} inside vertices, limit {GATE_LIMIT})", flush=True)
    print("CPU cross-check:", {True: "all frames MATCH", False: "MISMATCH - do not trust these results", None: "no CPU reference for these frames"}[summary["cpu_checks_all_match"]], flush=True)
    (a.out / ("summary.json" if not a.limit else f"summary_limit{a.limit}.json")).write_text(json.dumps(summary, indent=1, default=str))
    if summary["cpu_checks_all_match"] is False:
        sys.exit(2)


if __name__ == "__main__":
    main()
