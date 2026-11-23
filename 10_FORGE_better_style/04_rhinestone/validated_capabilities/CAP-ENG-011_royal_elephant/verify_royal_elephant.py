"""verify_royal_elephant.py - CAP-ENG-011 evidence script.

Scope: 2D stone-layout generation for the Royal Elephant & Crown insignia (royal_elephant.elephant()).
Checks (acceptance criteria):
  A1 the function returns a Design with at least 1000 stones;
  A2 every stone belongs to a named structure (non-empty tag), no untagged stones;
  A3 minimum edge-to-edge gap reported by the engine >= design gap (0.22 mm) - 1e-6;
  A4 an independent brute-force (KD-tree) clearance over the stone cover discs agrees: min gap >= design gap - 1e-6.
Not checked: 3D projection onto the garment, GLB/production export, preview rendering (see CAP-ENG-011 limitations).
Run: python -I -B verify_royal_elephant.py   (prints one JSON line, exit 0 on pass)
"""
import hashlib, json, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
WS = Path(__file__).resolve().parents[2]
T = WS / "03_ENGINE" / "rhinestone_engine" / "garments" / "tools"
sys.path.insert(0, str(T))
import numpy as np
from scipy.spatial import cKDTree
import royal_elephant


def fh(n):
    return hashlib.sha256((T / n).read_bytes()).hexdigest()


t = time.time()
D, info = royal_elephant.elephant()
stones = len(D.S)
rep = float(D.min_gap())
disc = D.discs()
xy, r, own = disc[:, :2], disc[:, 2], disc[:, 3].astype(int)
tree = cKDTree(xy)
mn = 1e9
for i in range(len(xy)):
    for j in tree.query_ball_point(xy[i], r[i] + r.max() + 0.5):
        if own[j] > own[i]:
            mn = min(mn, float(np.hypot(*(xy[i] - xy[j])) - r[i] - r[j]))
untagged = sum(1 for s in D.S if not s["tag"])
res = dict(stones=stones, untagged=untagged, structures=len({s["tag"] for s in D.S}), design_gap_mm=D.gap, min_gap_engine_mm=round(rep, 6),
           min_gap_independent_mm=round(mn, 6), rejected_attempts=D.rej, seconds=round(time.time() - t, 2),
           royal_elephant_sha256=fh("royal_elephant.py"), royal_core_sha256=fh("royal_core.py"))
res["pass"] = bool(stones >= 1000 and untagged == 0 and rep >= D.gap - 1e-6 and mn >= D.gap - 1e-6)
print(json.dumps(res))
sys.exit(0 if res["pass"] else 1)
