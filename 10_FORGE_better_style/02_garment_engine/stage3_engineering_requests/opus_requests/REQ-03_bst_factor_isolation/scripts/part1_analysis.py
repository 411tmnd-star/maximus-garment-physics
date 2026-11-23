"""REQ-03 Part 1 (+ architect addendum: start snap). Reads the instrumented default run (verified identical to the recorded outputs).
usage: python -I -B part1_analysis.py <run_name>"""
import sys, json, os
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from req03_lib import *

run = sys.argv[1] if len(sys.argv) > 1 else 'F0_instrumented'
R = load_run(run); wd = R['req03_walk_diag']; tc = R['req03_track_curves']; pr = R['req03_prune']
D1 = 1.45e-3; D2 = float(wd['A_collar__d2__d'])
LABS = {'A': 'A_collar', 'B_L': 'B_cuff_L', 'B_R': 'B_cuff_R'}
res = dict(run=run, d1_mm=D1 / MM, d2_mm=D2 / MM, nominal_separation_mm=(D2 - D1) / MM)

# ---------------- 1a + addendum (1),(2)
p1a = {}; per_zone_cache = {}
for z, lab in LABS.items():
    Qc = wd[lab + '__crestQ']; closed = bool(wd[lab + '__crest_closed']); Nc = wd[lab + '__crestN']
    sp = spacing(Qc, closed); zr = dict(crest_closed=closed, crest_vertices=int(len(Qc)), crest_spacing_mm=q(sp / MM))
    sgn = float(wd[f'{lab}__d1__sgn']); kg, sc = crest_kappa_g(Qc, Nc, closed, sgn, 1.0e-3); kg05, _ = crest_kappa_g(Qc, Nc, closed, sgn, 0.5e-3)
    kdc = cKDTree(Qc)
    for tk, d in (('d1', D1), ('d2', D2)):
        g = lambda k: wd[f'{lab}__{tk}__{k}']
        raw = g('raw'); keep = g('keep'); dv = g('dist_vertex'); snap = g('snap'); path = g('path'); start = g('start')
        dp, _, _ = dist_to_polyline(raw, Qc, closed)
        # discretisation bound of the vertex-only query: max over samples of (vertex dist - polyline dist); analytic: dmax^2/(8 d)
        bound = float(sp.max() ** 2 / (8 * d))
        assert np.allclose(raw[keep], R['tracks_raw'][f'{lab}_track{tk[1]}']), 'tracks_raw mismatch'
        e = (dp - d)[keep]
        zr[tk] = dict(samples=int(len(raw)), kept=int(keep.sum()), trimmed=int((~keep).sum()),
                      euclid_vertex_kept_mm=[float(dv[keep].min() / MM), float(dv[keep].max() / MM)],
                      euclid_polyline_kept_mm=[float(dp[keep].min() / MM), float(dp[keep].max() / MM)],
                      vertex_minus_polyline_max_mm=float((dv - dp).max() / MM), analytic_vertex_bound_mm=bound / MM,
                      dist_minus_d_kept_mm=q(e / MM), kept_exceeding_d_plus_bound=int((e > bound).sum()),
                      kept_exceeding_d_by_0p05mm=int((e > 0.05e-3).sum()), kept_exceeding_d_by_0p25mm=int((e > 0.25e-3).sum()),
                      start_snap_mm=q(snap / MM, (0, 50, 95, 100)), start_edge_len_mm=q(g('snap_edge') / MM, (0, 50, 95, 100)),
                      walk_path_mm=q(path / MM), snap_plus_path_minus_d_mm=q((snap + path - d) / MM),
                      corr_distmd_vs_snap_kept=corr(e, snap[keep]), corr_distmd_vs_snap_plus_path_kept=corr(e, (snap + path - d)[keep]),
                      corr_distmd_vs_path_kept=corr(e, path[keep]))
        # scatter summary: (dist-d) binned by snap
        bins = np.array([0, 0.1, 0.2, 0.4, 0.8, 1.6, 100]) * MM; bi = np.digitize(snap[keep], bins) - 1
        zr[tk]['scatter_by_snap_bin_mm'] = [dict(snap_bin=[float(bins[b] / MM), float(bins[b + 1] / MM)], n=int((bi == b).sum()),
                                                 dist_minus_d_median=float(np.median(e[bi == b]) / MM) if (bi == b).any() else None,
                                                 dist_minus_d_max=float(e[bi == b].max() / MM) if (bi == b).any() else None) for b in range(len(bins) - 1)]
        # crest curvature at each start sample
        ci = kdc.query(start)[1]; per_zone_cache[(z, tk)] = dict(raw=raw, keep=keep, snap=snap, path=path, dp=dp, kg=kg[ci], kg05=kg05[ci],
                                                                    turn=g('max_turn_deg'), chord=g('max_chord_ratio'), isl=g('island_changes'), pc=g('max_proj_corr'), S=g('S'))
    zr['rho_g_le_d2_concave_crest_vertices'] = int(((kg > 0) & (1 / np.maximum(np.abs(kg), 1e-12) <= D2)).sum())
    zr['crest_kappa_g_per_mm'] = q(kg * MM)
    p1a[z] = zr
res['part1a_and_snap'] = p1a

# ---------------- 1c: separation dips along the tracks, per start sample
p1c = {}
for z, lab in LABS.items():
    a = per_zone_cache[(z, 'd1')]; b = per_zone_cache[(z, 'd2')]
    Q1 = tc[f'{lab}_track1__Q']; cl1 = bool(tc[f'{lab}_track1__closed'])
    sep, _, _ = dist_to_polyline(b['raw'], Q1, cl1)                 # raw track-2 sample to the dense track-1 curve
    pair = np.linalg.norm(b['raw'] - a['raw'], axis=1)              # same crest start sample, d2 walk end vs d1 walk end
    m = b['keep']
    dip = (sep < 2.85e-3) & m
    ex = dict(n_kept_track2=int(m.sum()), sep_mm=q(sep[m] / MM), pair_sep_mm=q(pair[m] / MM), dips_lt_2p850=int(dip.sum()), dips_lt_2p800=int(((sep < 2.80e-3) & m).sum()))
    feats = dict(kappa_g_signed=b['kg'], kappa_g_signed_delta0p5=b['kg05'], abs_kappa_g=np.abs(b['kg']), max_normal_turn_deg_walk2=b['turn'],
                 max_chord_over_h_walk2=b['chord'], island_changes_walk2=b['isl'].astype(float), max_proj_correction_walk2=b['pc'], start_snap=b['snap'],
                 walk2_path=b['path'], walk1_path=a['path'])
    ex['corr_sep_vs'] = {k: corr(sep[m], v[m]) for k, v in feats.items()}
    ex['dip_rate_given'] = {
        'kappa_g>0 (crest concave toward tracks)': [int((dip & (b['kg'] > 0)).sum()), int((m & (b['kg'] > 0)).sum())],
        'kappa_g<=0': [int((dip & (b['kg'] <= 0)).sum()), int((m & (b['kg'] <= 0)).sum())],
        'rho_g<=d2 & concave': [int((dip & (b['kg'] > 0) & (1 / np.maximum(np.abs(b['kg']), 1e-12) <= D2)).sum()), int((m & (b['kg'] > 0) & (1 / np.maximum(np.abs(b['kg']), 1e-12) <= D2)).sum())],
        'max_normal_turn>10deg': [int((dip & (b['turn'] > 10)).sum()), int((m & (b['turn'] > 10)).sum())],
        'max_chord/h>1.05 (jump)': [int((dip & (b['chord'] > 1.05)).sum()), int((m & (b['chord'] > 1.05)).sum())],
        'island_change>0': [int((dip & (b['isl'] > 0)).sum()), int((m & (b['isl'] > 0)).sum())],
        'start_snap>0.2mm': [int((dip & (b['snap'] > 0.2e-3)).sum()), int((m & (b['snap'] > 0.2e-3)).sum())]}
    ex['dip_feature_medians_vs_nondip'] = {k: [float(np.median(v[dip])) if dip.any() else None, float(np.median(v[m & ~dip]))] for k, v in feats.items()}
    p1c[z] = ex
res['part1c'] = p1c

# ---------------- 1b: every clearance-rejected stone in zones A and B
zone = pr['zone']; curve = pr['curve']; keep = pr['keep']; occl = pr['occl']; P = pr['P']; Rr = pr['R']; s = pr['s']; blk = pr['blocker']; bm = pr['blocker_metric']
rej = np.nonzero((~keep) & (~occl) & np.isin(zone, ['A', 'B']))[0]
rows = []
for i in rej:
    cv = str(curve[i]); lab = cv.rsplit('_track', 1)[0]; tk = int(cv[-1]); z = {'A_collar': 'A', 'B_cuff_L': 'B_L', 'B_cuff_R': 'B_R'}[lab]
    Qs, ss = tc[cv + '__Q'], tc[cv + '__s']; xp = np.array([np.interp(s[i], ss, Qs[:, k]) for k in range(3)])
    other = f'{lab}_track{3 - tk}'
    sep_i = float(dist_to_polyline(xp[None], tc[other + '__Q'], bool(tc[other + '__closed']))[0][0])
    c = per_zone_cache[(z, f'd{tk}')]; kk = np.nonzero(c['keep'])[0]; j = kk[cKDTree(c['raw'][kk]).query(xp)[1]]
    kg_i = float(c['kg'][j]); rho = 1 / max(abs(kg_i), 1e-12)
    b = int(blk[i])
    rows.append(dict(idx=int(i), curve=cv, track=tk, s_mm=float(s[i] / MM), blocker_curve=str(curve[b]), blocker_track=int(str(curve[b])[-1]) if 'track' in str(curve[b]) else None,
                     centre_dist_mm=float(np.linalg.norm(P[i] - P[b]) / MM), metric_gap_mm=float(bm[i] / MM), local_sep_mm=sep_i * 1e3,
                     kappa_g_per_mm=kg_i * MM, rho_g_mm=rho * 1e3, concave_toward_tracks=bool(kg_i > 0), rho_g_le_d2=bool(rho <= D2),
                     start_snap_mm=float(c['snap'][j] / MM), max_turn_deg=float(c['turn'][j]), max_chord_ratio=float(c['chord'][j])))
res['part1b_rows'] = rows
bands = [(-1, 2.80), (2.80, 2.85), (2.85, 2.875), (2.875, 99)]
tab = {}
for zname, zsel in (('A', lambda r: r['curve'].startswith('A_')), ('B', lambda r: r['curve'].startswith('B_'))):
    for lo, hi in bands:
        sel = [r for r in rows if zsel(r) and lo <= r['local_sep_mm'] < hi]
        key = f'{zname} sep[{lo if lo > 0 else "<"},{hi if hi < 99 else ">"})'
        tab[key] = dict(n=len(sel), track1=sum(r['track'] == 1 for r in sel), track2=sum(r['track'] == 2 for r in sel),
                        rho_g_le_d2=sum(r['rho_g_le_d2'] for r in sel), rho_g_le_d2_and_concave=sum(r['rho_g_le_d2'] and r['concave_toward_tracks'] for r in sel))
res['part1b_table'] = tab
res['part1b_summary'] = dict(n=len(rows), by_track={t: sum(r['track'] == t for r in rows) for t in (1, 2)},
                             blocker_same_track=sum(r['blocker_track'] == r['track'] for r in rows), blocker_other_track=sum(r['blocker_track'] is not None and r['blocker_track'] != r['track'] for r in rows),
                             blocker_zone_C=sum(r['blocker_track'] is None for r in rows),
                             centre_dist_mm=q([r['centre_dist_mm'] for r in rows]), local_sep_mm=q([r['local_sep_mm'] for r in rows]), rho_g_mm=q([r['rho_g_mm'] for r in rows]),
                             by_curve={c: sum(r['curve'] == c for r in rows) for c in sorted({r['curve'] for r in rows})})
json.dump(res, open(os.path.join(BASE, f'part1_{run}.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'part1b_rows'}, indent=1)[:12000])
