"""REQ-03 Part 2: summarise every finished variant run in scratch/runs. usage: python -I -B part2_summary.py"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from req03_lib import *

out = {}
for name in sorted(os.listdir(RUNS)):
    R = load_run(name)
    if 'req03_run' not in R: continue
    row = dict(run=R['req03_run'])
    if 'phase3_audit' in R:
        a = R['phase3_audit']; row['audit'] = dict(total=a['total'], finishing=a.get('finishing'), buffer_sha256=a['buffer']['sha256'], bsrs_sha256=a['bsrs']['sha256'],
                                                    E5_A_max_mm=a['E5_along_track']['A']['max_mm'] if a['E5_along_track']['A'] else None,
                                                    E5_B_max_mm=a['E5_along_track']['B']['max_mm'] if a['E5_along_track']['B'] else None)
    if 'zone_ab_extract_report' in R:
        e = R['zone_ab_extract_report']
        row['offset_euclid_to_crest_vertex_mm'] = {z: {k: e[z][k]['euclid_to_crest_mm'] + [e[z][k]['trimmed'], e[z][k]['samples']] for k in ('offset_d1', 'offset_d2') if k in e[z]} for z in e}
    if 'zone_ab_report' in R:
        zr = R['zone_ab_report']; row['tracks'] = {k: dict(n=v['n'], L_mm=v['L_mm'], eps_mm=v['eps_mm'], despiked=v['despiked'], opened=v['opened'] is not None) for k, v in zr.items() if isinstance(v, dict) and 'n' in v}
    if 'req03_prune' in R:
        closed = {k for k, v in R['zone_ab_report'].items() if isinstance(v, dict) and 'opened' in v and v['opened'] is None}
        row['prune'] = prune_summary(R['req03_prune'], closed)
    if 'req03_track_curves' in R:
        row['separation_mm'] = {}
        for z in TRACK_CURVES:
            d = track_separation(R['req03_track_curves'], z)
            if d is not None: row['separation_mm'][z] = dict(min=float(d.min() / MM), p5=float(np.percentile(d, 5) / MM), median=float(np.median(d) / MM), frac_lt_2p850=float((d < 2.85e-3).mean()))
    if 'req03_walk_diag' in R:                                       # polyline-exact distance of kept track samples to the crest
        wd = R['req03_walk_diag']; row['euclid_to_crest_polyline_kept_mm'] = {}
        for lab in ('A_collar', 'B_cuff_L', 'B_cuff_R'):
            if lab + '__crestQ' not in wd: continue
            Qc = wd[lab + '__crestQ']; cl = bool(wd[lab + '__crest_closed']); rr = {}
            for tk in ('d1', 'd2'):
                raw = wd[f'{lab}__{tk}__raw']; keep = wd[f'{lab}__{tk}__keep']; d = float(wd[f'{lab}__{tk}__d']); dp, _, _ = dist_to_polyline(raw[keep], Qc, cl)
                rr[tk] = dict(d_mm=d / MM, min=float(dp.min() / MM), max=float(dp.max() / MM), p95=float(np.percentile(dp, 95) / MM),
                              snap_median_mm=float(np.median(wd[f'{lab}__{tk}__snap']) / MM), snap_max_mm=float(np.max(wd[f'{lab}__{tk}__snap']) / MM))
            row['euclid_to_crest_polyline_kept_mm'][lab] = rr
    out[name] = row
json.dump(out, open(os.path.join(BASE, 'part2_runs.json'), 'w'), indent=1)
for k, v in out.items():
    p = v.get('prune', {})
    print(k, v['run']['wall_s'], 'final', p.get('final_total'), 'pre', p.get('pre_total'), 'rej', p.get('zone_rejects'),
          {z: (p[z].get('track1_rejected_clearance'), p[z].get('track2_rejected_clearance'), round(p[z].get('track2_max_void_mm', -1), 2)) for z in ('A_collar', 'B_cuff_L', 'B_cuff_R') if z in p},
          {z: (round(s['min'], 3), round(s['p5'], 3), round(s['median'], 3)) for z, s in v.get('separation_mm', {}).items()})
