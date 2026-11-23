"""PHASE 3a: merge zones, audit against the master (independent of the solver's own samples), write the stride-7 Float32 buffer
and a BSRS v1.2 file that the UNMODIFIED Phase-2 exporter (bsrs_to_instanced_glb.py) can read.

Buffer b_i = [px, py, pz, nx, ny, nz, scale] in glTF axes (Y-up, metres): (x, y, z)_blender -> (x, z, -y).
scale = stone DIAMETER in metres (chaton_v1 is unit-diameter, rhs_format DIAMETERS_M). The whitepaper's "2.80 / 4.00" assumes a 1 mm
unit mesh in mm space; with metre positions that would make 2.8 m stones, so it is not used (documented deviation).
usage: python p3_assemble.py <master.blend> <out_dir>"""
import os
import sys, json, math, hashlib
import numpy as np, bpy
bpy.ops.wm.open_mainfile(filepath=sys.argv[1], load_ui=False); OUT = sys.argv[2]
from scipy.spatial import cKDTree
import bst_core as C
sys.path.insert(0, os.path.normpath(os.path.expanduser('~/Desktop/better_style_execution_workspace/08_KNOWLEDGE/stage3_engineering/opus_requests/REQ-03_bst_factor_isolation/scratch/rl/py')))  # REQ-03: redirected from the live rhinestone_lab\py
import rhs_format as RF

ab, c = np.load(OUT + '/zone_ab_stones.npz'), np.load(OUT + '/zone_c_stones.npz')
P = np.vstack([ab['P'], c['P']]); N = np.vstack([ab['N'], c['N']]); T = np.vstack([ab['T'], c['T']]); R = np.concatenate([ab['R'], c['R']])
zone = np.concatenate([ab['zone'], c['zone']]); curve = np.concatenate([ab['curve'], c['curve']]); s = np.concatenate([ab['s'], c['s']])
kind = np.concatenate([np.where(np.char.find(ab['curve'].astype(str), 'track1') >= 0, 'track1', 'track2'), c['kind']])
sub = np.array(['crown' if str(k).startswith('crown') else ('script' if z == 'C' else z) for k, z in zip(kind, zone)])
n = len(P); rep = dict(total=int(n))
rep['by_zone'] = {'A_collar': int((zone == 'A').sum()), 'B_cuffs': int((zone == 'B').sum()), 'C_back': int((zone == 'C').sum()),
                  'C_script': int((sub == 'script').sum()), 'C_crown': int((sub == 'crown').sum())}
rep['by_size'] = {'SS10': int(np.isclose(R, C.R_SS10).sum()), 'SS16': int(np.isclose(R, C.R_SS16).sum())}
rep['by_curve'] = {str(k): int(v) for k, v in zip(*np.unique(curve.astype(str), return_counts=True))} if False else None
rep['crown_kinds'] = {str(k): int(v) for k, v in zip(*np.unique(kind[sub == 'crown'].astype(str), return_counts=True))}

# ---------------- penetration (all garment layers)
from mathutils.bvhtree import BVHTree
def obj_bvh(name):                                            # audit obstacle BVH straight from the object's polygons (n-gons allowed)
    me = bpy.data.objects[name].data; return BVHTree.FromPolygons([v.co[:] for v in me.vertices], [p.vertices[:] for p in me.polygons])
bvhs = [obj_bvh(o) for o in ('Tee', 'collar', 'placket_overlap', 'placket_buttons')]
d8, dd, m8 = C.penetration_check(P, N, T, R, bvhs)
rep['penetration'] = dict(spec8_plus_centre=dict(stones_penetrating=int((d8 > 0).sum()), max_depth_mm=float(d8.max() / C.MM), samples=int(n * 9), missed_rays=int(m8.sum())),
                          dense_97=dict(stones_penetrating=int((dd > 0).sum()), max_depth_mm=float(dd.max() / C.MM), by_zone={z: int(((dd > 0) & (zone == z)).sum()) for z in 'ABC'}))
np.save(OUT + '/pen_dense.npy', dd)

# ---------------- FINISHING (documented deviations from the spec-literal solver; spec-only numbers are kept above)
FINISH = '--spec-only' not in sys.argv
rep['penetration_spec_literal'] = rep['penetration']
if FINISH:
    # (1) dense-envelope re-lift: rays of the 97-point grid in the final frame; lift by depth + glue if depth <= 0.2 mm
    lift = (dd > -C.GLUE_OFFSET) & (dd <= 0.2 * C.MM)
    if '--no-relift' in sys.argv: lift[:] = False                 # REQ-03 F_f
    P[lift] +=(dd[lift] + C.GLUE_OFFSET)[:, None] * N[lift]   # every stone ends >= glue clear on the dense grid
    # (2) reject stones still penetrating > 0.2 mm (fabric folds over the footprint: OCCLUDED_BY_OTHER_LAYER)
    occl = dd > 0.2 * C.MM
    # (3) priority prune so every pair clears eps_min
    def prio(i):
        z, k = zone[i], str(kind[i])
        if z == 'A': return 0 if k == 'track1' else 1
        if z == 'B': return 2 if k == 'track1' else 3
        return {'crown_boundary16': 4, 'shared_anchor': 5, 'track': 6, 'brent_start': 6, 'crown_lattice16': 7, 'crown_inter10': 8, 'crown_spire10': 8}.get(k, 9)
    order0 = sorted(range(n), key=lambda i: (prio(i), str(zone[i]), str(curve[i]), s[i]))
    acc = []; kd_pts = []; rejected = dict(occluded=int(occl.sum()), clearance=0); keepm = np.zeros(n, bool)
    from scipy.spatial import cKDTree as _K
    tree = None; buf_pts = []; buf_idx = []; _blk = np.full(n, -1); _bm = np.full(n, np.inf)
    for i in order0:
        if occl[i]: continue
        ok = True
        for j in acc:
            pass
        if acc:
            Pa = P[acc]; dmin = np.linalg.norm(Pa - P[i], axis=1) - R[acc] - R[i]
            ok = dmin.min() >= C.EPS_MIN - 1e-9
            _blk[i] = acc[int(np.argmin(dmin))]; _bm[i] = dmin.min()
        if ok: acc.append(i); keepm[i] = True
        else: rejected['clearance'] += 1
    np.savez(OUT + '/req03_prune.npz', P=P, N=N, T=T, R=R, zone=zone, curve=curve.astype(str), kind=kind.astype(str), s=s, occl=occl, keep=keepm,
             blocker=_blk, blocker_metric=_bm, dd=dd, lift=lift, order=np.array(order0))
    rep['finishing'] = dict(relifted=int(lift.sum()), rejected=rejected, rejected_clearance_by_zone={z: int(((~keepm) & (~occl) & (zone == z)).sum()) for z in 'ABC'},
                            rejected_occluded_by_zone={z: int((occl & (zone == z)).sum()) for z in 'ABC'})
    sel = keepm
    P, N, T, R, zone, curve, s, kind, sub = P[sel], N[sel], T[sel], R[sel], zone[sel], curve[sel], s[sel], kind[sel], sub[sel]; n = int(sel.sum())
    rep['total'] = n
    rep['by_zone'] = {'A_collar': int((zone == 'A').sum()), 'B_cuffs': int((zone == 'B').sum()), 'C_back': int((zone == 'C').sum()), 'C_script': int((sub == 'script').sum()), 'C_crown': int((sub == 'crown').sum())}
    rep['by_size'] = {'SS10': int(np.isclose(R, C.R_SS10).sum()), 'SS16': int(np.isclose(R, C.R_SS16).sum())}
    rep['crown_kinds'] = {str(k): int(v) for k, v in zip(*np.unique(kind[sub == 'crown'].astype(str), return_counts=True))}
    d8, dd, m8 = C.penetration_check(P, N, T, R, bvhs)
    rep['penetration'] = dict(spec8_plus_centre=dict(stones_penetrating=int((d8 > 0).sum()), max_depth_mm=float(d8.max() / C.MM), samples=int(n * 9), missed_rays=int(m8.sum())),
                              dense_97=dict(stones_penetrating=int((dd > 0).sum()), max_depth_mm=float(dd.max() / C.MM)))

# ---------------- E5 along-track gaps (consecutive seated centres per curve, s-ordered; closed tracks wrap)
closed_curves = {k for k, v in json.load(open(OUT + '/zone_ab_report.json')).items() if isinstance(v, dict) and 'opened' in v and v['opened'] is None}
track_gaps = []
for cv in sorted(set(curve.astype(str))):
    if cv == 'C_crown_pave': continue
    idx = np.nonzero(curve.astype(str) == cv)[0]; idx = idx[np.argsort(s[idx], kind='stable')]
    if len(idx) < 2: continue
    pairs = list(zip(idx[:-1], idx[1:])) + ([(idx[-1], idx[0])] if (cv in closed_curves or cv.startswith('C_crown_c')) else [])
    for i, j in pairs: track_gaps.append((cv, np.linalg.norm(P[i] - P[j]) - R[i] - R[j]))
g = np.array([x[1] for x in track_gaps]); zc = np.array([x[0][0] for x in track_gaps])
def st(a): return dict(n=int(len(a)), min_mm=float(a.min() / C.MM), max_mm=float(a.max() / C.MM), in_window=int(((a >= C.EPS_MIN - 1e-9) & (a <= C.EPS_MAX + 1e-9)).sum())) if len(a) else None
rep['E5_along_track'] = {'all': st(g), 'A': st(g[zc == 'A']), 'B': st(g[zc == 'B']), 'C_script+crown_contours': st(g[zc == 'C'])}
rep['E5_out_of_window_pairs'] = sorted([(cv, round(x / C.MM, 4)) for cv, x in track_gaps if not (C.EPS_MIN - 1e-9 <= x <= C.EPS_MAX + 1e-9)], key=lambda t: t[1])[:40]

# ---------------- all-pairs clearance (any two stones), nearest-neighbour gaps
kd = cKDTree(P); pr = kd.query_pairs(2 * C.R_SS16 + 0.5 * C.MM, output_type='ndarray')
pg = np.linalg.norm(P[pr[:, 0]] - P[pr[:, 1]], axis=1) - R[pr[:, 0]] - R[pr[:, 1]]
nn = np.full(n, np.inf)
np.minimum.at(nn, pr[:, 0], pg); np.minimum.at(nn, pr[:, 1], pg)
viol = pr[pg < C.EPS_MIN - 1e-9]
rep['all_pairs'] = dict(pairs_checked=int(len(pr)), min_gap_mm=float(pg.min() / C.MM), overlapping_pairs=int((pg < 0).sum()), below_eps_min_pairs=int(len(viol)),
                        below_eps_min_by_zone_pair={k: int(v) for k, v in zip(*np.unique([''.join(sorted(zone[a] + zone[b])) + ':' + ('same' if curve[a] == curve[b] else 'cross') for a, b in viol], return_counts=True))} if len(viol) else {})
fin = np.isfinite(nn)
rep['nearest_neighbour_gap_mm'] = dict(min=float(nn[fin].min() / C.MM), p50=float(np.median(nn[fin]) / C.MM), max=float(nn[fin].max() / C.MM),
                                       crown_pave=dict(min=float(nn[fin & (sub == 'crown')].min() / C.MM), p50=float(np.median(nn[fin & (sub == 'crown')]) / C.MM), max=float(nn[fin & (sub == 'crown')].max() / C.MM)))
rep['max_inter_crystal_clearance_mm'] = dict(along_track=float(g.max() / C.MM), nearest_neighbour_any_stone=float(nn[fin].max() / C.MM))

# ---------------- 7-float buffer (glTF axes)
pg_ = np.stack([P[:, 0], P[:, 2], -P[:, 1]], 1); ng = np.stack([N[:, 0], N[:, 2], -N[:, 1]], 1); ng /= np.linalg.norm(ng, axis=1, keepdims=True)
order = np.lexsort((s, curve.astype(str), zone))                                # deterministic record order: zone, curve, s
buf = np.concatenate([pg_, ng, (2 * R)[:, None]], 1)[order].astype('<f4')
buf.tofile(OUT + '/BS-BSTSET-001_stones.f32'); rep['buffer'] = dict(file='BS-BSTSET-001_stones.f32', stride=7, records=int(len(buf)), bytes=int(buf.nbytes), sha256=hashlib.sha256(buf.tobytes()).hexdigest())

# ---------------- BSRS via the closed-form quaternion q = [nz/k, 0, -nx/k, k/2], k = sqrt(2(1+ny))
def qmat(q):
    x, y, z, w = q; return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
B = buf.astype(np.float64); M = []; Q = []
for b in B:
    nx, ny, nz = b[3:6]
    if ny > -0.99999: k = math.sqrt(2 * (1 + ny)); q = np.array([nz / k, 0.0, -nx / k, 0.5 * k])
    else: q = np.array([1.0, 0, 0, 0])
    Rm = qmat(q) * b[6]; Q.append(q)
    M.append([*Rm[:, 0], 0, *Rm[:, 1], 0, *Rm[:, 2], 0, *b[:3], 1])
M = np.array(M, dtype='<f4'); Q = np.array(Q)
rep['closed_form_check_python'] = dict(max_abs_Rq_y_minus_n=float(max(np.abs(qmat(q) @ [0, 1, 0] - b[3:6]).max() for q, b in zip(Q, B))), max_quat_norm_err=float(np.abs(np.linalg.norm(Q, axis=1) - 1).max()))
zid = {'A': 1, 'B': 2, 'C': 3}; zo, cu, ko, Ro = zone[order], curve[order].astype(str), kind[order].astype(str), R[order]
size = ['SS16' if np.isclose(r, C.R_SS16) else 'SS10' for r in Ro]
pl = dict(intended_size=size, outcome=['ACCEPTED'] * n, reason=['NONE'] * n, panel=[{'A': 'collar_leaf_outer', 'B': 'sleeve_outer', 'C': 'body_back_outer'}[z] for z in zo],
          surface=[{'A': 'collar_shell', 'B': 'tee_shell', 'C': 'tee_shell'}[z] for z in zo], layer=['shell_outer'] * n, curve=list(cu))
raw = RF.write_bsrs(M, size, bezel_id=0, zone_id=[zid[z] for z in zo], placement=pl)
open(OUT + '/BS-BSTSET-001_stones.bsrs', 'wb').write(raw); chk = RF.read_bsrs(raw); assert chk['ok'], chk['errors']
rep['bsrs'] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), read_back_ok=bool(chk['ok']))
np.savez(OUT + '/assembled_meta.npz', zone=zo, curve=cu, kind=ko, R=Ro, P_blender=P[order], N_blender=N[order], T_blender=T[order])
json.dump(rep, open(OUT + '/phase3_audit.json', 'w'), indent=1)
print(json.dumps(rep, indent=1))
