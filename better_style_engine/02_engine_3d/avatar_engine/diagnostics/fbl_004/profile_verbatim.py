"""FBL-004 step 1 and 2: frame 61 on the verbatim clearance path.

The path is the one in harness/gates.py (lines 246-259):
    V, T, _ = CF.weld(BODY[k], TRI); surf = CF.TriSurface(V, T, "body")
    for g in garments: P = Z[g][i].astype(float); wn = surf.winding_number(P); inside = wn > 0.5
The library itself is imported unchanged from prototypes/P2_clearance_field/clearance_field.py.

Writes profile_verbatim.json and ref_inside_f061.npz. Read-only on the capture.
Run:  python -I -B profile_verbatim.py
"""
import sys, time, json, hashlib, platform, subprocess
from pathlib import Path
import numpy as np
import psutil

HERE = Path(__file__).resolve().parent
AE = HERE.parents[1]                       # solvers/avatar_engine
SOLVERS = AE.parents[0]                    # solvers/
sys.path.insert(0, str(SOLVERS / "prototypes/P2_clearance_field"))
import clearance_field as CF               # noqa: E402  unchanged library

CAP = AE / "baseline/capture_current_walk.npz"
FR = 61
GARMENTS = ["garment__G0_jacket", "garment__G0_pants"]   # same order as Z.files in gates.py

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def cpu_info():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-CimInstance Win32_Processor | Select-Object Name,MaxClockSpeed,CurrentClockSpeed | ConvertTo-Json"],
                             capture_output=True, text=True, timeout=60).stdout
        return json.loads(out)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}

if __name__ == "__main__":
    cap_hash_before = sha(CAP)
    proc = psutil.Process()
    Z = np.load(CAP)
    frames = Z["frames"]; BODY = Z["body"].astype(float); TRI = Z["tri"]
    k = int(np.nonzero(frames == FR)[0][0]); i = 0          # i: index into clearance_frames (61 is first)
    assert int(json.loads(str(Z["meta"]))["clearance_frames"][i]) == FR

    cpu_before = cpu_info()
    rss0 = proc.memory_info().rss
    t_all = time.time()
    t0 = time.time()
    V, T, _ = CF.weld(BODY[k], TRI); surf = CF.TriSurface(V, T, "body")
    t_surface = time.time() - t0
    per = {}; inside = {}; wind = {}
    for g in GARMENTS:
        P = Z[g][i].astype(float)
        t1 = time.time(); wn = surf.winding_number(P); dt = time.time() - t1
        ins = wn > 0.5
        short = g[len("garment__"):]
        per[short] = dict(seconds=round(dt, 2), points=int(len(P)), inside=int(ins.sum()))
        inside[short] = ins; wind[short] = wn
        print(short, "inside", int(ins.sum()), "of", len(P), f"{dt:.1f}s", flush=True)
    wall = time.time() - t_all
    peak = proc.memory_info().peak_wset if hasattr(proc.memory_info(), "peak_wset") else None
    cpu_after = cpu_info()

    np.savez(HERE / "ref_inside_f061.npz",
             jacket_inside=inside["G0_jacket"], pants_inside=inside["G0_pants"],
             jacket_index=np.nonzero(inside["G0_jacket"])[0], pants_index=np.nonzero(inside["G0_pants"])[0],
             jacket_winding=wind["G0_jacket"], pants_winding=wind["G0_pants"])

    result = dict(
        frame=FR, body_frame_index=k, clearance_index=i,
        path="harness/gates.py lines 246-259 (CF.weld, CF.TriSurface, winding_number > 0.5), library unchanged",
        body_vertices_welded=int(len(V)), body_triangles=int(len(surf.T)),
        body_facts={key: surf.facts[key] for key in ("closed", "manifold", "boundary_edges", "non_manifold_edges", "consistently_oriented", "degenerate_triangles_dropped")},
        surface_build_seconds=round(t_surface, 2),
        garments=per,
        wall_seconds_total=round(wall, 2),
        peak_working_set_mb=round(peak / 2**20, 1) if peak else None,
        rss_start_mb=round(rss0 / 2**20, 1),
        cpu_before=cpu_before, cpu_after=cpu_after,
        python=platform.python_version(), numpy=np.__version__,
        chunk_pairs_default=6_000_000,
        capture_sha256_before=cap_hash_before, capture_sha256_after=sha(CAP),
    )
    (HERE / "profile_verbatim.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k2: v for k2, v in result.items() if k2 not in ("cpu_before", "cpu_after")}, indent=1))
