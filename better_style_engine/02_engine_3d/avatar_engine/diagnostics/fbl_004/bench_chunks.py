"""FBL-004 chunk-size micro-benchmark on 600 jacket points of frame 61 (read-only). Compares with the verbatim reference winding values bit for bit."""
import sys, time, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fast_clearance as FC
CF = FC.CF
Z = np.load(FC.CAP); k = int(np.nonzero(Z["frames"] == 61)[0][0])
V, T, _ = CF.weld(Z["body"][k].astype(float), Z["tri"]); surf = CF.TriSurface(V, T, "body")
ref = np.load(HERE / "ref_inside_f061.npz")["jacket_winding"]
P = Z["garment__G0_jacket"][0].astype(float)
sel = np.concatenate([np.arange(0, 300), np.nonzero(np.load(HERE / "ref_inside_f061.npz")["jacket_inside"])[0][:300]])
rows = []
for cp in [100_000, 250_000, 500_000, 1_000_000, 2_000_000, 6_000_000]:
    tm = []; t = time.perf_counter(); w, step = FC.winding_chunked(surf, P[sel], cp, time.time() + 1e9, tm); dt = time.perf_counter() - t
    rows.append(dict(chunk_pairs=cp, rows_per_chunk=step, seconds_600_points=round(dt, 2), seconds_per_1e9_pairs=round(dt / (len(sel) * len(surf.T)) * 1e9, 1),
                     bit_identical_to_reference=bool(np.array_equal(w, ref[sel])), max_abs_diff=float(np.abs(w - ref[sel]).max())))
    print(rows[-1], flush=True)
(HERE / "bench_chunks.json").write_text(json.dumps(rows, indent=1))
