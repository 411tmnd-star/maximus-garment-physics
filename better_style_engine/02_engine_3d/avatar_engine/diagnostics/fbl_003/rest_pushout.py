"""FBL-003 rest-pose push-out (numpy, local). Reads rest_meshes.npz, writes rest_fixed.npz + rest_pushout_report.json.

Method (monotone, cannot oscillate): every garment vertex v has a signed distance sd(v) to the rest body (+ outside, - inside; sign from the
exact winding-number rule) and an outward unit direction dir(v) (away from its nearest body point, or toward it if inside).
  need(v)  = max(0, MARGIN - sd(v))                         # push that clears v by MARGIN
  push(v)  = max over u of need(u) * (1 - |v-u|/R_u)^2     # R_u = R0 + K*need(u): neighbours are lifted too, so no cliff/spike
  v'       = v + push(v) * dir(v)                           # each vertex moves ONLY along its own outward direction
then sd/dir are recomputed for the moved vertices and the pass repeats until nothing is inside and every vertex is >= MARGIN away.
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "prototypes/P2_clearance_field"))
import clearance_field as CF
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--rest", type=Path, default=HERE / "rest_meshes.npz"); ap.add_argument("--out-dir", type=Path, default=HERE)
ap.add_argument("--margin-mm", type=float, default=3.0); ap.add_argument("--max-iter", type=int, default=6)
ap.add_argument("--only", default=None, help="comma list of garments (default both)")
A = ap.parse_args(); A.out_dir.mkdir(parents=True, exist_ok=True)
MARGIN = A.margin_mm / 1000; R0 = 0.020; K = 0.75; MAX_ITER = A.max_iter; NEAR = 0.15; CREASE_R = 0.010
Z = np.load(HERE.parents[1] / "baseline/capture_current_walk.npz")      # only its triangle list (body topology) is used
RM = np.load(A.rest)
V, T, _ = CF.weld(RM["rest_mannequin"], Z["tri"]); S = CF.TriSurface(V, T, "rest_body")
# The body self-intersects (arm/torso, thighs): some of its triangles are buried inside other body parts. Pushing toward a buried
# triangle leaves the vertex inside. So directions/distances use only the OUTER skin (triangles whose outward-offset centroid is
# outside the body); inside/outside is still the exact winding number of the FULL body (the gate rule).
OUTER_CACHE = A.out_dir / "body_outer_mask.npy"
if OUTER_CACHE.exists() and len(np.load(OUTER_CACHE)) == len(S.T):
    outer = np.load(OUTER_CACHE)
else:
    probe = S.cen + 0.0005 * S.FN
    wp = np.concatenate([S.winding_number(probe[i:i + 200], chunk_pairs=600_000) for i in range(0, len(probe), 200)])
    outer = wp < 0.5; np.save(OUTER_CACHE, outer)
S_out = CF.TriSurface(V, S.T[outer], "rest_body_outer")
print(f"body triangles {len(S.T)}: outer {int(outer.sum())}, buried {int((~outer).sum())}", flush=True)

def signed(P, idx):
    """sd and outward direction for vertices idx of P (exact winding sign for everything within NEAR)."""
    sd = np.empty(len(idx)); dr = np.zeros((len(idx), 3))
    for s in range(0, len(idx), 2000):
        ii = idx[s:s + 2000]; c = S_out.closest_point(P[ii]); d = c["distance"]; vec = P[ii] - c["point"]
        near = np.where(d < NEAR)[0]; w = np.concatenate([S.winding_number(P[ii[near[j:j + 200]]], chunk_pairs=600_000) for j in range(0, len(near), 200)]) if len(near) else np.empty(0)
        inside = np.zeros(len(ii), bool); inside[near[w > 0.5]] = True
        out = np.where(inside, -d, d); n = np.linalg.norm(vec, axis=1, keepdims=True)
        # outward direction from local geometry, NOT from the winding sign: a vertex on/near the skin can be "inside" by winding while lying
        # a hair in front of the face; flipping on the winding sign then pushed it MARGIN into the body (seen as min sd stuck at -MARGIN).
        # side of the skin from angle-weighted pseudonormals (correct at edges and vertices too, Baerentzen & Aanaes 2005)
        fn = S_out.FN[c["tri"]]; u = vec / np.maximum(n, 1e-12)
        behind = S_out.signed_distance(P[ii], rule="pseudonormal", cp=c)["inside"]
        u = np.where(behind[:, None], -u, u)                                            # behind the skin -> toward/through it
        # nearest point on an edge/vertex: typically the crease where two self-intersecting parts' skins meet (arm/torso). There the
        # only sensible outward direction is the bisector = area-weighted normal of the skin within CREASE_R of that point.
        fe = np.where(c["feature"] != 0)[0]
        for k, lst in zip(fe, S_out.tree.query_ball_point(c["point"][fe], CREASE_R)):
            lst = np.asarray(lst, int)
            if len(lst):
                m = (S_out.FN[lst] * S_out.area[lst, None]).sum(0); nm = np.linalg.norm(m)
                if nm > 1e-12: u[k] = m / nm
        u = np.where(n > 1e-7, u, fn)                                                   # on the surface: the face normal
        sd[s:s + 2000] = out; dr[s:s + 2000] = u
    return sd, dr

result = {"margin_m": MARGIN, "R0_m": R0, "K": K, "garments": {}}; fixed = {}
for g in [x for x in ["G0_jacket", "G0_pants"] if not A.only or x in A.only.split(",")]:
    P0 = RM[f"rest_{g}"].astype(float); P = P0.copy(); n = len(P); allv = np.arange(n)
    sd, dr = signed(P, allv); n_in0 = int((sd < 0).sum()); log = [n_in0]; worst = [float(sd.min() * 1000)]; reverts = []
    for it in range(MAX_ITER):
        need = np.maximum(0.0, MARGIN - sd)
        if need.max() <= 1e-6: break
        src = np.where(need > 1e-9)[0]; tree = cKDTree(P); push = np.zeros(n)
        for u in src:
            R = R0 + K * need[u]; nb = np.array(tree.query_ball_point(P[u], R), int)
            val = need[u] * (1 - np.linalg.norm(P[nb] - P[u], axis=1) / R) ** 2
            push[nb] = np.maximum(push[nb], val)
        push = np.maximum(push, need)
        mv = np.where(push > 0)[0]; Pprev, sdprev, drprev = P[mv].copy(), sd[mv].copy(), dr[mv].copy()
        P[mv] += push[mv, None] * dr[mv]; sd[mv], dr[mv] = signed(P, mv)
        # monotone guard: undo any move that took a vertex from outside to inside, or deeper inside
        worse = ((sdprev >= 0) & (sd[mv] < 0)) | ((sdprev < 0) & (sd[mv] < sdprev - 1e-9))
        newly = int(((sdprev >= 0) & (sd[mv] < 0)).sum()); deeper = int(((sdprev < 0) & (sd[mv] < sdprev - 1e-9)).sum())
        b = mv[worse]; P[b], sd[b], dr[b] = Pprev[worse], sdprev[worse], drprev[worse]
        log.append(int((sd < 0).sum())); worst.append(float(sd.min() * 1000)); reverts.append(dict(newly_inside=newly, deeper=deeper))
        print(g, "pass", it + 1, "inside", log[-1], "min sd mm", round(worst[-1], 3), "moved", len(mv), "reverted: newly-inside", newly, "deeper", deeper, flush=True)
        if it >= 2 and log[-1] >= log[-3]: print(g, "no progress in 2 passes, stopping", flush=True); break
    disp = np.linalg.norm(P - P0, axis=1); moved = disp > 1e-9
    result["garments"][g] = dict(inside_before=n_in0, inside_per_pass=log, min_signed_distance_mm_per_pass=worst, inside_after=int((sd < 0).sum()),
        verts_moved=int(moved.sum()), max_disp_mm=float(disp.max() * 1000), median_disp_moved_mm=float(np.median(disp[moved]) * 1000) if moved.any() else 0.0,
        min_signed_distance_mm_after=float(sd.min() * 1000), reverted_per_pass=reverts)
    ri = np.where(sd < 0)[0]; np.savez(A.out_dir / f"residual_{g}.npz", index=ri, pos=P[ri], sd=sd[ri], pos_orig=P0[ri])   # where the leftovers are
    fixed[f"rest_{g}"] = P
    print(g, result["garments"][g], flush=True)
np.savez(A.out_dir / "rest_fixed.npz", rest_mannequin=RM["rest_mannequin"], **fixed)
(A.out_dir / "rest_pushout_report.json").write_text(json.dumps(result, indent=1))
print("DONE", flush=True)
