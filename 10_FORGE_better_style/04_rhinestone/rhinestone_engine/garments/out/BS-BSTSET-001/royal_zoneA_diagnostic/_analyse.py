import sys, json, math, numpy as np
sys.path.insert(0, sys.argv[1]+'/tools'); sys.dont_write_bytecode=True
import bpy
from mathutils import Vector
import stones_core as C, royal_core as K
from scipy.spatial import cKDTree
W, DG = sys.argv[1], sys.argv[2]; MM = C.MM
sys.path.insert(0, W+'/../../prototypes/P1_bstset_exact_hull'); import convex_gap as CG
out = {}
# ---------------- 1. keystone probe on the original collar (read-only)
bpy.ops.wm.open_mainfile(filepath=W+'/work/camp_rebuilt.blend', load_ui=False)
col = C.Mesh('collar'); tr = json.load(open(DG+'/keystone_trace.json')); q = np.array(tr['design_point']); n = np.array(tr['n'])
probe = {}
for dx in (0.0, 1e-6, 1e-4, 1e-3, 5e-3):
    for sgn in (1, -1):
        qq = q + np.array([sgn * dx, 0, 0]); loc, nn, idx, d = col.bvh.ray_cast(Vector(qq + 0.03 * n), Vector(-n), 0.06)
        probe[f'dx={sgn*dx:+.0e}'] = None if loc is None else dict(hit_dist_mm=round(d / MM, 3), dot=round(float(np.array(nn) @ n), 3), y=round(loc[1], 5))
loc, nn, idx, d = col.bvh.find_nearest(Vector(q)); probe['nearest_surface'] = dict(dist_mm=round(d / MM, 3), P=[round(v, 5) for v in loc], face_normal=[round(v, 3) for v in nn], dot_with_chart_n=round(float(np.array(nn) @ n), 3))
for L in (0.06, 0.12, 0.3):
    r = col.bvh.ray_cast(Vector(q + 0.03 * n), Vector(-n), L); probe[f'ray_len_{L}'] = None if r[0] is None else round(r[3] / MM, 2)
r = col.bvh.ray_cast(Vector(q - 0.03 * n), Vector(n), 0.06); probe['ray_from_inside'] = None if r[0] is None else round(r[3] / MM, 2)
# vertices at x≈0 (seam) near the chart point
V = col.V; near = V[(np.abs(V[:, 0]) < 1e-7) & (np.linalg.norm(V - q, axis=1) < 0.01)]; probe['collar_verts_exactly_on_x=0_within_10mm'] = int(len(near))
out['keystone_probe'] = probe
# ---------------- 2. exact gaps of all candidate pairs, pre and post clearance
pre = np.load(DG+'/pre_clearance.npz'); keep = np.load(DG+'/keep_mask.npy'); fin = np.load(W+'/out/BS-BSTSET-001/royal_zoneA_staging/zoneA_stones.npz')
tag = pre['tag']; Pp = pre['P']
assert keep.sum() == len(fin['P']) and np.allclose(Pp[keep], fin['P']), 'reproduction differs from staging'
out['reproduction_matches_staging'] = True
def solids(Pm, Nm, Tm, shape, Lm, Wm, bez):
    Pv, Nv, Rv, Hv, own, plain = [], [], [], [], [], []
    for i in range(len(Pm)):
        B = np.cross(Tm[i], Nm[i]); simple = shape[i] == 'round' and not bez[i]
        for a, b, r in K.cover_discs(str(shape[i]), Lm[i] / MM, Wm[i] / MM, K.BEZEL if bez[i] else 0.0):
            Pv.append(Pm[i] + a * MM * Tm[i] + b * MM * B); Nv.append(Nm[i]); Rv.append(r * MM); Hv.append(0.30 * Wm[i]); own.append(i); plain.append(simple)
    return [np.array(x) for x in (Pv, Nv, Rv, Hv, own, plain)]
def pairgaps(Pm, Nm, Tm, shape, Lm, Wm, bez, wide):
    Pv, Nv, Rv, Hv, own, plain = solids(Pm, Nm, Tm, shape, Lm, Wm, bez)
    pr = cKDTree(Pv).query_pairs(2 * Rv.max() + 0.4 * MM, output_type='ndarray'); pr = pr[own[pr[:, 0]] != own[pr[:, 1]]]
    i, j = pr[:, 0], pr[:, 1]; dist = np.linalg.norm(Pv[i] - Pv[j], axis=1); d = dist - Rv[i] - Rv[j]; cosn = np.einsum('ij,ij->i', Nv[i], Nv[j]); bound = np.hypot(Rv, Hv)
    sure = ((d > 0.12 * MM) & (cosn > 0.94)) | (dist - bound[i] - bound[j] > wide)
    sup = {}
    def S(k):
        if k not in sup: sup[k] = CG.round_support(C.CHATON, Pv[k], Nv[k], 2 * Rv[k]) if plain[k] else CG.CirclesSupport(np.array([Pv[k], Pv[k] + Hv[k] * Nv[k]]), Nv[k], np.array([Rv[k], Rv[k]]))
        return sup[k]
    rows = []
    for k in np.nonzero(~sure)[0]:
        up, lo, it, hit = CG.gjk_distance(S(int(i[k])), S(int(j[k]))); rows.append((int(own[i[k]]), int(own[j[k]]), 0.0 if hit else lo, up))
    return rows, float(d[sure].min() / MM) if sure.any() else None, int(sure.sum())
G = 0.079 * MM
rows, dmin_sure, nsure = pairgaps(fin['P'], fin['N'], fin['T'], fin['shape'], fin['L'], fin['W'], fin['bezel'], G)   # bound threshold = 0.079 so nothing under it is skipped
below = [r for r in rows if r[2] < G]
ftag = fin['tag']
out['final_pairs'] = dict(pairs_needing_exact=len(rows), pairs_decided_by_bound_clear_of_0p079=nsure, min_gap_of_bound_pairs_mm=round(dmin_sure, 4),
    exact_gaps_mm=sorted(round(r[2] / MM, 4) for r in rows), pairs_below_0p079=len(below),
    below_0p079=[dict(a=r[0], b=r[1], tag_a=str(ftag[r[0]]), tag_b=str(ftag[r[1]]), gap_mm=round(r[2] / MM, 4), upper_mm=round(r[3] / MM, 4), P_a=[round(float(v), 4) for v in fin['P'][r[0]]]) for r in below])
# ---------------- 3. removed stones and what they collided with
rows0, dm0, ns0 = pairgaps(Pp, pre['N'], pre['T'], pre['shape'], pre['L'], pre['W'], pre['bezel'], 0.05 * MM)
bad = [r for r in rows0 if r[2] < 0.05 * MM]; rem = np.nonzero(~keep)[0]; kd = cKDTree(Pp); recs = []
for g in rem:
    partners = [(r[1] if r[0] == g else r[0], r[2]) for r in bad if g in (r[0], r[1])]
    near = sorted(set(str(tag[k]) for k in kd.query_ball_point(Pp[g], 4.5 * MM) if k != g))
    recs.append(dict(index=int(g), tag=str(tag[g]), P=[round(float(v), 4) for v in Pp[g]], collided_with=[dict(index=int(k), tag=str(tag[k]), kept=bool(keep[k]), gap_mm=round(gg / MM, 4)) for k, gg in partners], structures_within_4p5mm=near))
out['removed'] = recs
cnt = {}
for r in recs:
    key = ' <-> '.join(sorted([r['tag']] + [c['tag'] for c in r['collided_with']][:1])); cnt[key] = cnt.get(key, 0) + 1
out['removed_by_pair_type'] = cnt; out['pairs_below_0p05_pre_clearance'] = len(bad); out['pairs_overlapping_pre'] = sum(1 for r in bad if r[2] <= 0)
json.dump(out, open(DG+'/analysis.json', 'w'), indent=1, default=str)
print(json.dumps({k: out[k] for k in ('keystone_probe', 'final_pairs', 'removed_by_pair_type', 'pairs_below_0p05_pre_clearance')}, indent=0, default=str)[:6000])
