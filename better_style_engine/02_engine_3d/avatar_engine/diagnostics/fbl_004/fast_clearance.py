"""FBL-004 fast_clearance: the same clearance rule as harness/gates.py, with fixed small chunks.

Rule (unchanged): garment vertex is inside when the generalised winding number (exact summation,
float64, van Oosterom-Strackee solid angle) of the position-welded body is > 0.5.
The library (prototypes/P2_clearance_field/clearance_field.py) is imported unchanged: weld(), TriSurface.
The per-chunk arithmetic below is the loop body of TriSurface.winding_number copied token for token;
only the chunk size is a parameter and each chunk is timed.

What differs from the verbatim path, and nothing else:
  1. chunk_pairs is small and fixed (default 250_000 pairs, about 6 MB per temporary instead of ~144 MB)
  2. one TriSurface per frame, shared by both garments (gates.py already does this; kept)
  3. a checkpoint file per frame (inside_f0NN.npz); an existing checkpoint is not recomputed
  4. each frame runs in its own process (restart between frames)
  5. stop conditions of the packet: peak working set > 6 GB, or a frame > 30 min -> exit 3

Optional, NOT in the packet's list of permitted means (reported as such in REPORT.md): --workers N splits the
points of a garment into contiguous slices of whole chunks and runs them in N processes. Every row of the
solid-angle sum is computed independently, so the result is bit-identical; this is checked against the reference.

Not used: hierarchical or sampled winding numbers, any distance pre-filter, float32, approximation.

Run (orchestrator, restarts a worker process per frame):
    python -I -B fast_clearance.py --frames 61 69 76 84 91
Single frame in this process:
    python -I -B fast_clearance.py --worker 61 [--chunk-pairs N] [--out DIR] [--tag NAME]
"""
import sys, os, time, json, hashlib, argparse, subprocess
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
AE = HERE.parents[1]                                   # solvers/avatar_engine
SOLVERS = AE.parent
sys.path.insert(0, str(SOLVERS / "prototypes/P2_clearance_field"))
import clearance_field as CF                           # noqa: E402  unchanged library

CAP = AE / "baseline/capture_current_walk.npz"
GARMENTS = ["G0_jacket", "G0_pants"]                   # order of Z.files in gates.py
DEFAULT_CHUNK_PAIRS = 250_000
MEM_LIMIT_BYTES = 6 * 2**30
FRAME_LIMIT_SECONDS = 30 * 60


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def peak_ws():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return int(getattr(mi, "peak_wset", mi.rss))
    except Exception:  # noqa: BLE001
        return 0


def winding_chunked(surf, P, chunk_pairs, t_deadline, timings):
    """Loop body identical to TriSurface.winding_number; chunk_pairs is explicit and each chunk is timed."""
    P = np.atleast_2d(np.asarray(P, float)); m = len(surf.T); out = np.empty(len(P)); step = max(1, chunk_pairs // max(m, 1))
    for s in range(0, len(P), step):
        t0 = time.perf_counter()
        p = P[s:s + step][:, None, :]; a = surf.A[None] - p; b = surf.B[None] - p; c = surf.C[None] - p
        la = np.linalg.norm(a, axis=2); lb = np.linalg.norm(b, axis=2); lc = np.linalg.norm(c, axis=2)
        num = np.einsum("qti,qti->qt", a, np.cross(b, c)); den = la * lb * lc + (a * b).sum(2) * lc + (b * c).sum(2) * la + (c * a).sum(2) * lb
        out[s:s + step] = np.arctan2(num, den).sum(1) / (2.0 * np.pi)
        timings.append(time.perf_counter() - t0)
        if time.time() > t_deadline:
            raise TimeoutError("frame exceeded the 30 minute limit")
        if peak_ws() > MEM_LIMIT_BYTES:
            raise MemoryError("peak working set exceeded 6 GB")
    return out, step


_W = {}


def _wk_init(V, T):
    _W["surf"] = CF.TriSurface(V, T, "body")


def _wk_task(args):
    """One slice of whole chunks in a worker process. Same loop body as winding_chunked."""
    P, chunk_pairs, deadline = args
    tm = []
    wn, step = winding_chunked(_W["surf"], P, chunk_pairs, deadline, tm)
    return wn, tm, peak_ws()


def cap_workers(n):
    """Packet cap: max(1, min(cpu_count - 1, 4)); a larger request is reduced, never raised."""
    return max(1, min(n, max(1, min((os.cpu_count() or 2) - 1, 4))))


def clock_mhz():
    try:
        import psutil
        f = psutil.cpu_freq()
        return round(f.current) if f else None
    except Exception:  # noqa: BLE001
        return None


def winding_parallel(pool, workers, P, chunk_pairs, m, deadline):
    step = max(1, chunk_pairs // max(m, 1)); nchunks = -(-len(P) // step)
    per = -(-nchunks // (workers * 4)) * step              # about 4 tasks per worker, all slices whole chunks
    parts = [P[s:s + per] for s in range(0, len(P), per)]
    import multiprocessing as mp
    job = pool.map_async(_wk_task, [(x, chunk_pairs, deadline) for x in parts])
    try:
        res = job.get(timeout=max(1.0, deadline - time.time()))
    except mp.TimeoutError:
        pool.terminate(); raise TimeoutError("frame exceeded the 30 minute limit")
    return np.concatenate([r[0] for r in res]), [t for r in res for t in r[1]], step, [r[2] for r in res]


def run_frame(fr, chunk_pairs, outdir, tag, workers=1):
    t_start = time.time(); deadline = t_start + FRAME_LIMIT_SECONDS
    cap_before = sha(CAP)
    Z = np.load(CAP)
    frames = Z["frames"]; meta = json.loads(str(Z["meta"])); cf = [int(x) for x in meta["clearance_frames"]]
    k = int(np.nonzero(frames == fr)[0][0]); i = cf.index(fr)
    t0 = time.perf_counter()
    V, T, _ = CF.weld(Z["body"][k].astype(float), Z["tri"]); surf = CF.TriSurface(V, T, "body")
    t_surface = time.perf_counter() - t0
    res = {}; arrays = {}; per = {}
    pool = None; workers = cap_workers(workers); child_peaks = []; clocks = [clock_mhz()]
    if workers > 1:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(workers, initializer=_wk_init, initargs=(V, T))
    for g in GARMENTS:
        P = Z["garment__" + g][i].astype(float); tm = []
        t1 = time.perf_counter()
        if pool is None:
            wn, step = winding_chunked(surf, P, chunk_pairs, deadline, tm)
        else:
            wn, tm, step, pk = winding_parallel(pool, workers, P, chunk_pairs, len(surf.T), deadline); child_peaks += pk
        dt = time.perf_counter() - t1; clocks.append(clock_mhz())
        inside = wn > 0.5
        arrays[g + "_winding"] = wn; arrays[g + "_inside"] = inside; arrays[g + "_index"] = np.nonzero(inside)[0]
        arrays[g + "_chunk_seconds"] = np.asarray(tm)
        q = np.array_split(np.asarray(tm), 4) if len(tm) >= 4 else [np.asarray(tm)]
        per[g] = dict(points=int(len(P)), inside=int(inside.sum()), seconds=round(dt, 2), rows_per_chunk=int(step), chunks=len(tm),
                      pairs_per_second=round(len(P) * len(surf.T) / dt), cpu_mhz_after=clocks[-1],
                      chunk_seconds_mean=round(float(np.mean(tm)), 5), chunk_seconds_max=round(float(np.max(tm)), 5),
                      chunk_seconds_mean_by_quarter=[round(float(x.mean()), 5) for x in q])
        print(f"frame {fr} {g} inside {int(inside.sum())} of {len(P)}  {dt:.1f}s  ({len(tm)} chunks of {step} rows)", flush=True)
    if pool is not None:
        pool.close(); pool.join()
    meta_out = dict(frame=fr, tag=tag, chunk_pairs=int(chunk_pairs), workers=int(workers), body_triangles=int(len(surf.T)), body_vertices_welded=int(len(V)),
                    body_facts={key: surf.facts[key] for key in ("closed", "manifold", "boundary_edges", "non_manifold_edges", "consistently_oriented")},
                    surface_build_seconds=round(t_surface, 3), garments=per, wall_seconds=round(time.time() - t_start, 2),
                    peak_working_set_mb=round(peak_ws() / 2**20, 1), worker_peak_working_set_mb_sum_upper_bound=round(sum(child_peaks[-workers:] if workers > 1 else []) / 2**20, 1),
                    cpu_mhz_samples=clocks, logical_cpus=os.cpu_count(), numpy=np.__version__,
                    capture_sha256_before=cap_before, capture_sha256_after=sha(CAP))
    assert meta_out["capture_sha256_before"] == meta_out["capture_sha256_after"]
    name = f"inside_f{fr:03d}.npz" if tag == "fast" else f"inside_f{fr:03d}_{tag}.npz"
    tmp = outdir / (name + ".tmp.npz")
    np.savez(tmp, meta_json=np.array(json.dumps(meta_out)), **arrays)
    os.replace(tmp, outdir / name)                       # checkpoint appears only when complete
    print(json.dumps(meta_out), flush=True)
    return meta_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="*", default=[61, 69, 76, 84, 91])
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--chunk-pairs", type=int, default=DEFAULT_CHUNK_PAIRS)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--tag", default="fast")
    ap.add_argument("--workers", type=int, default=1)       # capped to max(1, min(cpu_count - 1, 4)) in run_frame
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.worker is not None:
        try:
            run_frame(a.worker, a.chunk_pairs, a.out, a.tag, a.workers)
        except (TimeoutError, MemoryError) as e:
            print("STOP:", e, flush=True); sys.exit(3)
        return
    for fr in a.frames:
        name = f"inside_f{fr:03d}.npz" if a.tag == "fast" else f"inside_f{fr:03d}_{a.tag}.npz"
        if (a.out / name).exists():
            print(f"frame {fr}: checkpoint exists, skipped", flush=True); continue
        cmd = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--worker", str(fr), "--chunk-pairs", str(a.chunk_pairs), "--out", str(a.out), "--tag", a.tag, "--workers", str(a.workers)]
        r = subprocess.run(cmd)                          # a fresh process per frame
        if r.returncode != 0:
            print(f"frame {fr}: worker exit {r.returncode}; stopping", flush=True); sys.exit(r.returncode)


if __name__ == "__main__":
    main()
