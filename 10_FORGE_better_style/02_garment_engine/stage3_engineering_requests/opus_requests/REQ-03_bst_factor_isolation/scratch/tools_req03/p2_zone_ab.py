"""PHASE 2 Zones A (collar perimeter, double track) and B (sleeve hems, double track).
Perimeter = crest where the visible outer surface turns under (zero set of f = N.out). Offsets = iso-lines of the fast-marching
geodesic distance from that crest at d1 = R10 + eps_hem (eps_hem = 0.05 mm) and d2 = d1 + D10 + eps* (spec 1.3.4). An iso-line of the
geodesic distance is the trimmed offset by construction, so no curve-curve clipping step is needed.
usage: python p2_zone_ab.py <master.blend> <out_dir>"""
import sys, json, time, math
import numpy as np, bpy
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
bpy.ops.wm.open_mainfile(filepath=sys.argv[1], load_ui=False); OUT = sys.argv[2]
import bst_core as C, bst_tracks as K

D1 = C.R_SS10 + 0.05 * C.MM; D2 = D1 + 2 * C.R_SS10 + C.EPS_NOM
# ---- REQ-03 factor flags (defaults reproduce the original)
def _flag(name, default):
    return next((a.split('=', 1)[1] for a in sys.argv if a.startswith(name + '=')), default)
D2 += float(_flag('--d2-add-mm', 0.0)) * C.MM
K.CFG['trim_tol'] = float(_flag('--trim', 0.97))
K.CFG['walk_step'] = float(_flag('--walk-step-mm', 0.0)) * C.MM or None
K.CFG['despike'] = '--no-despike' not in sys.argv
K.CFG['spline_smooth'] = '--no-spline-smooth' not in sys.argv
K.CFG['start_full_mesh'] = '--start-full-mesh' in sys.argv
ANY_SHEET = '--any-vertex-sheet' in sys.argv
ZONES = _flag('--zones', 'A,B_L,B_R').split(',')
print('REQ-03 CFG', K.CFG, 'D2_mm', D2 / C.MM, 'ANY_SHEET', ANY_SHEET, 'ZONES', ZONES)
print('d1 = %.4f mm, d2 = %.4f mm' % (D1 / C.MM, D2 / C.MM))
man = C.MeshData('Mannequin')
CREST_LEVEL = float(next((a.split('=')[1] for a in sys.argv if a.startswith('--crest=')), 0.5))   # f = N.o iso-level used as the perimeter (0.5 = surface turned 60 deg from outward)


def group_mask(obj, md, gname):
    o = bpy.data.objects[obj]; gi = o.vertex_groups[gname].index; w = np.zeros(len(md.V))
    for v in o.data.vertices:
        for g in v.groups:
            if g.group == gi: w[v.index] = g.weight
    return w > 0.5


REGION_LEVEL = float(next((a.split('=')[1] for a in sys.argv if a.startswith('--region=')), 0.0))   # f-level bounding the FMM propagation region

def outer_component(md, f, vmask, prefer):
    """Largest-area connected component (shared vertices) of triangles with all f > 0 and all vertices in vmask."""
    ok = np.all(f[md.T] > REGION_LEVEL, 1) & np.all(vmask[md.T], 1); tri = np.nonzero(ok)[0]
    rows = np.repeat(np.arange(len(tri)), 3); cols = md.T[tri].ravel()
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(tri), len(md.V))).tocsr(); G = A @ A.T
    nc, lab = connected_components(G, directed=False)
    P = md.V[md.T[tri]]; area = np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1) / 2
    score = np.bincount(lab, weights=area * prefer(P.mean(1)))
    k = int(np.argmax(score)); return tri[lab == k], dict(components=int(nc), chosen_area_cm2=float(area[lab == k].sum() * 1e4))


def tracks_for(md, f, comp, label, select_loop, keep_for=None):
    f = f - CREST_LEVEL
    kc = keep_for(0.0) if keep_for else None
    compv = np.zeros(len(md.V), bool); compv[md.T[comp].ravel()] = True
    mixed = np.nonzero(np.any(compv[md.T], 1) & np.any(f[md.T] > 0, 1) & np.any(f[md.T] <= 0, 1))[0]
    if kc is not None:
        cen = md.V[md.T[mixed]].mean(1); mixed = mixed[kc(cen)]
    seeds = K.seed_from_crest(md, mixed, f)
    t0 = time.time(); dist = K.fmm_geodesic(md, np.concatenate([comp, mixed]), seeds, d_max=D2 + 2 * C.MM)
    crest = [K.close_if_near(P, c) for P, c in K.marching(md, mixed, f, 0.0)]
    info = dict(crest_triangles=len(mixed), seeds=len(seeds), fmm_s=round(time.time() - t0, 1),
                crest_loops=[(len(P), round(K.poly_len(P, c) / C.MM, 1), c) for P, c in crest][:8])
    out = []
    for k, d in ((1, D1), (2, D2)):
        loops = [K.close_if_near(P, c) for P, c in K.marching(md, comp, dist, d, keep=keep_for(d) if keep_for else None)]
        P, closed = select_loop(loops)
        info[f'iso_d{k}'] = dict(candidates=[(round(K.poly_len(Q, c) / C.MM, 1), c) for Q, c in loops][:10], chosen_len_mm=round(K.poly_len(P, closed) / C.MM, 2), closed=closed)
        out.append((f'{label}_track{k}', P, closed, d, dist))
    return out, info


def check_offset(curve, dist, md, d):
    """Geodesic distance (FMM field, barycentric-interpolated) at the dense curve samples vs the target d_k."""
    vals = []
    for q in curve.Q[::10]:
        t, b = md.nearest(q); vals.append(b @ dist[md.T[t]])
    v = np.array(vals); return dict(target_mm=d / C.MM, min_mm=float(v.min() / C.MM), max_mm=float(v.max() / C.MM), mean_mm=float(v.mean() / C.MM))


report = {}; stones = []; SHEETS = {}
def crest_curve(md, f, tri_ids, label, keep=None, angular=None):
    pieces = [K.close_if_near(P, c) for P, c in K.marching(md, tri_ids, f - CREST_LEVEL, 0.0, keep=keep)]
    info = dict(crest_pieces=sorted([(round(K.poly_len(P, c) * 1e3, 1), c) for P, c in pieces], reverse=True)[:6])
    if angular is None:
        P, c = max(((P, c) for P, c in pieces if c), key=lambda x: K.poly_len(*x))
    else:                                   # one crest point per angle bin (most distal) -> closed by construction across seams
        th_of, tt_of, bins = angular; allp = np.vstack([P for P, c in pieces]); b = ((th_of(allp) + np.pi) / (2 * np.pi) * bins).astype(int) % bins
        P = []
        for k in range(bins):
            m = b == k
            if m.any(): P.append(allp[m][np.argmax(tt_of(allp[m]))])
        P = np.array(P); c = True; info['angular_bins_filled'] = len(P)
    cur = K.SurfaceCurve(label + '_crest', P, c, md, smooth_mm=0.3); info['crest_len_mm'] = round(cur.L * 1e3, 2); return cur, info

def tracks_for(md, f, tri_mask, crest, label, seat_mask=None):
    sheet = K.outer_sheet_bvh(md, tri_mask); out = []; info = {}
    SHEETS[label] = K.outer_sheet_bvh(md, seat_mask) if seat_mask is not None else sheet
    for k, d in ((1, D1), (2, D2)):
        Q, st = K.offset_walk(crest, d, md, sheet, f); info[f'offset_d{k}'] = st
        out.append((f'{label}_track{k}', Q, True, d)); 
    return out, info

# ---------------- Zone A: collar
col = C.MeshData('collar'); col.outwardness(man.bvh); o_s = K.smooth_field(col, col.o_dir, 10); o_s /= np.linalg.norm(o_s, axis=1, keepdims=True)
fcol = np.einsum('ij,ij->i', col.VN, o_s)
crestA, infoA = crest_curve(col, fcol, np.arange(len(col.T)), 'A_collar')
_selA = np.all(fcol[col.T] > CREST_LEVEL, 1)
_walkA = np.any(fcol[col.T] > CREST_LEVEL, 1) if ANY_SHEET else _selA                 # F_g1: walk sheet only; seating sheet kept identical
trA, i2 = tracks_for(col, fcol, _walkA, crestA, 'A_collar', seat_mask=_selA if ANY_SHEET else None); infoA.update(i2); report['zone_A'] = infoA
CRESTS = {'A_collar': (crestA, col)}
# ---------------- Zone B: sleeves
tee = C.MeshData('Tee'); trB = []
for side in ('L', 'R'):
    if f'B_{side}' not in ZONES: continue
    am = group_mask('Mannequin', man, f'upperarm01.{side}') | group_mask('Mannequin', man, f'upperarm02.{side}')
    Pa = man.V[am]; c0 = Pa.mean(0); a = np.linalg.svd(Pa - c0)[2][0]
    sm = group_mask('Tee', tee, f'set_in_sleeve_{side}')
    if (tee.V[sm].mean(0) - c0) @ a < 0: a = -a
    tt = (tee.V - c0) @ a
    rad = tee.V - c0 - np.outer(tt, a); rad /= np.linalg.norm(rad, axis=1, keepdims=True)
    fax = np.einsum('ij,ij->i', tee.VN, rad)
    e1 = np.cross(a, [0, 1, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(a, e1)
    th_of = lambda P, c0=c0, e1=e1, e2=e2: np.arctan2((P - c0) @ e2, (P - c0) @ e1); tt_of = lambda P, c0=c0, a=a: (P - c0) @ a
    bins = 720; bi = ((th_of(tee.V[sm]) + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    tmax = np.full(bins, -np.inf); np.maximum.at(tmax, bi, tt[sm])
    keep = lambda P, tmax=tmax: tt_of(P) >= tmax[((th_of(P) + np.pi) / (2 * np.pi) * bins).astype(int) % bins] - 4e-3
    tri_s = np.nonzero(np.any(sm[tee.T], 1))[0]
    crestB, info = crest_curve(tee, fax, tri_s, f'B_cuff_{side}', keep=keep, angular=(th_of, tt_of, bins))
    info.update(arm_axis=a.round(4).tolist())
    CRESTS[f'B_cuff_{side}'] = (crestB, tee)
    _fx = np.any(fax[tee.T] > CREST_LEVEL, 1) if ANY_SHEET else np.all(fax[tee.T] > CREST_LEVEL, 1)
    tr, i2 = tracks_for(tee, fax, np.all(sm[tee.T], 1) & _fx, crestB, f'B_cuff_{side}', seat_mask=np.all(sm[tee.T], 1) & np.all(fax[tee.T] > 0.0, 1)); info.update(i2); report[f'zone_B_{side}'] = info; trB += tr

# ---------------- E5 spacing on every closed track (SS10)
np.savez(OUT + '/tracks_raw.npz', **{cid: P for cid, P, c, d in trA + trB})
json.dump(report, open(OUT + '/zone_ab_extract_report.json', 'w'), indent=1, default=str)
# ---- REQ-03 diagnostics dump
_dd = {}
for lab, (cr, md_) in CRESTS.items():
    _sv = getattr(md_, 'sheet', None); md_.sheet = None
    _dd[lab + '__crestQ'] = cr.Q; _dd[lab + '__crest_closed'] = np.array(cr.closed)
    _dd[lab + '__crestN'] = np.array([md_.point_normal(*md_.nearest(q))[1] for q in cr.Q]); md_.sheet = _sv
for (cid, dum), dg in K.DIAG.items():
    tk = 'd1' if abs(dg['d'] - D1) < 1e-9 else 'd2'; lab = cid.replace('_crest', '')
    for kk, v in dg.items(): _dd[f'{lab}__{tk}__{kk}'] = np.asarray(v)
np.savez(OUT + '/req03_walk_diag.npz', **_dd)
if '--extract-only' in sys.argv: sys.exit(0)
DENSE = {}
for zone, md, trs in (('A', col, trA), ('B', tee, trB)):
    for cid, P, closed, d in trs:
        assert closed, cid
        md.sheet = SHEETS[cid.rsplit('_track', 1)[0]]
        cur = K.SurfaceCurve(cid, P, True, md)
        seat = (lambda md_, cur_: (lambda s: C.seat_at_surface_point(cur_.point(s), cur_.tangent(s), C.R_SS10, md_)))(md, cur)
        t0 = time.time(); nd0 = len(K.DISCONTINUITIES); sol = K.solve_closed_loop(cur, C.R_SS10, seat); opened = None
        if len(K.DISCONTINUITIES) > nd0:            # seated jump on the bridged part -> do not bridge: open the track at its largest raw gap
            Po, gap = K.split_at_largest_gap(P); cur = K.SurfaceCurve(cid, Po, False, md)
            seat = (lambda md_, cur_: (lambda s: C.seat_at_surface_point(cur_.point(s), cur_.tangent(s), C.R_SS10, md_)))(md, cur)
            nd1 = len(K.DISCONTINUITIES); sol = K.solve_open_free(cur, C.R_SS10, seat)
            opened = dict(raw_gap_mm=gap * 1e3, closed_attempt_discontinuities=nd1 - nd0, open_discontinuities=len(K.DISCONTINUITIES) - nd1, end_trim_mm=sol['open_end_trim_mm'])
        DENSE[cid + '__Q'] = cur.Q; DENSE[cid + '__s'] = cur.s; DENSE[cid + '__closed'] = np.array(cur.closed)
        Qd = cur.Q; v = np.diff(Qd, axis=0); v /= np.linalg.norm(v, axis=1, keepdims=True); turn = np.degrees(np.arccos(np.clip((v[:-1] * v[1:]).sum(1), -1, 1)))
        rep = dict(opened=opened, L_mm=round(cur.L / C.MM, 3), despiked=cur.despiked, max_turn_per_0p1mm_deg=float(np.nanmax(turn)), n=sol['n'], eps_mm=sol['eps'] / C.MM, in_window=bool(sol['in_window']), window_n=sol['window'],
                   solve_s=round(time.time() - t0, 1),
                   misses=int(sum(st['misses'] for _, st in sol['stones'])))
        report[cid] = rep; print(cid, json.dumps(rep))
        for s, st in sol['stones']:
            stones.append(dict(zone=zone, curve=cid, tier=1 if cid.endswith('track1') else 2, R=C.R_SS10, s=s, P=st['P'], N=st['N'], T=st['T'], obj=md.name, misses=st['misses']))

np.savez(OUT + '/zone_ab_stones.npz', P=np.array([x['P'] for x in stones]), N=np.array([x['N'] for x in stones]), T=np.array([x['T'] for x in stones]),
         R=np.array([x['R'] for x in stones]), zone=np.array([x['zone'] for x in stones]), curve=np.array([x['curve'] for x in stones]),
         tier=np.array([x['tier'] for x in stones]), s=np.array([x['s'] for x in stones]), obj=np.array([x['obj'] for x in stones]))
np.savez(OUT + '/req03_track_curves.npz', **DENSE)
report['seat_discontinuities'] = K.DISCONTINUITIES
json.dump(report, open(OUT + '/zone_ab_report.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
print(json.dumps({k: v for k, v in report.items() if k.startswith('zone')}, indent=1, default=str)[:4000])
