"""REQ-03 Part 3: prune metric (|Pi-Pj| - Ri - Rj on seated base centres) vs exact chaton_v1 hull gap.
Final layout = recorded assembled_meta.npz (scratch copy). Rejected stones' frames = req03_prune.npz of the verified-identical rerun.
usage: python -I -B part3_metric.py <run_with_prune>"""
import sys, os, json
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from req03_lib import *
sys.path.insert(0, os.path.join(BASE, 'scratch', 'rl', 'py'))
import exact_hull as EH, chaton_geometry as CG

run = sys.argv[1] if len(sys.argv) > 1 else 'F0_instrumented'
TRI = CG.TRI; EI, EJ = EH._EI, EH._EJ
def stone(P, N, T, R): return dict(base=P, normal=N / np.linalg.norm(N), tangent=T / np.linalg.norm(T), diameter=2 * R)
def exact(a, b):
    sep = EH.separation_exact(a, b); inter = EH.intersects_edges(a, b)
    if inter: return dict(inter=True, sat_gap=sep['gap_m'], exact_gap=sep['gap_m'])     # penetration depth (SAT MTD, negative)
    HA = EH.hull_world(a['base'], a['normal'], a['tangent'], a['diameter']); HB = EH.hull_world(b['base'], b['normal'], b['tangent'], b['diameter'])
    return dict(inter=False, sat_gap=sep['gap_m'], exact_gap=hull_distance(HA, HB, TRI, EI, EJ))

m = np.load(os.path.join(BASE, 'scratch', 'out_orig', 'assembled_meta.npz'))
P, N, T, R, zone = m['P_blender'], m['N_blender'], m['T_blender'], m['R'], m['zone'].astype(str)
# frame sanity: T orthogonal to N?
ortho = np.abs((N * T).sum(1) / np.linalg.norm(N, axis=1) / np.linalg.norm(T, axis=1))
res = dict(chaton=dict(params=CG.P, SS10_diameter_mm=2.8, SS10_girdle_radius_mm=1.40, SS10_height_mm=0.30 * 2.8, SS16_girdle_radius_mm=2.0, SS16_height_mm=0.30 * 4.0,
                       octagon_apothem_SS10_mm=1.40 * np.cos(np.pi / 8), note='octagon vertices lie ON the girdle circle (radius = R used by the prune)'),
           frame_T_dot_N_max=float(ortho.max()))
pr = cKDTree(P).query_pairs(2 * 2.0e-3 + 0.5e-3, output_type='ndarray')
metric = np.linalg.norm(P[pr[:, 0]] - P[pr[:, 1]], axis=1) - R[pr[:, 0]] - R[pr[:, 1]]
ex = [exact(stone(P[i], N[i], T[i], R[i]), stone(P[j], N[j], T[j], R[j])) for i, j in pr]
eg = np.array([e['exact_gap'] for e in ex]); sg = np.array([e['sat_gap'] for e in ex]); it = np.array([e['inter'] for e in ex])
rng = np.random.default_rng(20261110); samp = rng.choice(len(pr), size=min(500, len(pr)), replace=False)
band = np.nonzero((metric >= 0.05e-3) & (metric <= 0.15e-3))[0]; sel = np.union1d(samp, band)
def block(ix):
    dlt = (eg[ix] - metric[ix]) / MM
    return dict(n=int(len(ix)), exact_minus_metric_mm=q(dlt, (0, 1, 5, 50, 95, 100)), exact_lt_metric=int((dlt < 0).sum()),
                exact_overlap=int(it[ix].sum()), exact_below_eps_min=int((eg[ix] < EPS_MIN).sum()), metric_accepts=int((metric[ix] >= EPS_MIN - 1e-9).sum()),
                accepted_by_metric_but_exact_overlap_or_lt_0p05=int(((metric[ix] >= EPS_MIN - 1e-9) & ((eg[ix] < EPS_MIN) | it[ix])).sum()),
                sat_minus_exact_max_mm=float((sg[ix] - eg[ix])[~it[ix]].max() / MM) if (~it[ix]).any() else None,
                min_exact_gap_mm=float(eg[ix].min() / MM), min_metric_gap_mm=float(metric[ix].min() / MM))
res['final_pairs_random500_plus_band'] = block(sel); res['final_random500_only'] = block(samp); res['final_band_0p05_0p15_only'] = block(band); res['final_all_neighbour_pairs'] = block(np.arange(len(pr)))
res['seed'] = 20261110
# normal-angle dependence
ang = np.degrees(np.arccos(np.clip((N[pr[:, 0]] * N[pr[:, 1]]).sum(1) / np.linalg.norm(N[pr[:, 0]], axis=1) / np.linalg.norm(N[pr[:, 1]], axis=1), -1, 1)))
res['final_all_pairs_corr_exact_minus_metric_vs_normal_angle'] = corr(eg - metric, ang)
w = np.argsort(eg - metric)[:5]
res['final_worst5_exact_minus_metric'] = [dict(pair=[int(pr[k, 0]), int(pr[k, 1])], zones=[zone[pr[k, 0]], zone[pr[k, 1]]], metric_mm=float(metric[k] / MM), exact_mm=float(eg[k] / MM), normal_angle_deg=float(ang[k])) for k in w]

# ---------------- rejected stones (frames from the rerun's prune dump)
pz = dict(np.load(os.path.join(RUNS, run, 'req03_prune.npz')))
Pp, Np, Tp, Rp, keep, occl, order = pz['P'], pz['N'], pz['T'], pz['R'], pz['keep'], pz['occl'], pz['order']
rank = np.empty(len(order), int); rank[order] = np.arange(len(order))
rej = np.nonzero((~keep) & (~occl))[0]; kd = cKDTree(Pp); rows = []
for i in rej:
    nb = [j for j in kd.query_ball_point(Pp[i], 4.5e-3) if j != i and keep[j] and rank[j] < rank[i]]
    best = None
    for j in nb:
        mt = np.linalg.norm(Pp[i] - Pp[j]) - Rp[i] - Rp[j]; e = exact(stone(Pp[i], Np[i], Tp[i], Rp[i]), stone(Pp[j], Np[j], Tp[j], Rp[j]))
        if best is None or e['exact_gap'] < best['exact_gap']: best = dict(j=int(j), metric=float(mt), **e)
    mmin = min(np.linalg.norm(Pp[i] - Pp[j]) - Rp[i] - Rp[j] for j in nb)
    rows.append(dict(i=int(i), zone=str(pz['zone'][i]), curve=str(pz['curve'][i]), metric_min_mm=float(mmin / MM), exact_min_mm=float(best['exact_gap'] / MM), exact_inter=bool(best['inter'])))
ok = [r for r in rows if (not r['exact_inter']) and r['exact_min_mm'] >= 0.05]
res['rejected'] = dict(n=len(rows), exact_fine_ge_0p05=len(ok), exact_fine_by_zone={z: sum(r['zone'] == z for r in ok) for z in 'ABC'},
                       exact_overlap=sum(r['exact_inter'] for r in rows), exact_0_to_0p05=sum((not r['exact_inter']) and r['exact_min_mm'] < 0.05 for r in rows),
                       metric_min_mm=q([r['metric_min_mm'] for r in rows]), exact_min_mm=q([r['exact_min_mm'] for r in rows]),
                       exact_minus_metric_mm=q([r['exact_min_mm'] - r['metric_min_mm'] for r in rows]))
res['rejected_rows'] = rows
json.dump(res, open(os.path.join(BASE, 'part3_metric.json'), 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
print(json.dumps({k: v for k, v in res.items() if k != 'rejected_rows'}, indent=1, default=str))
