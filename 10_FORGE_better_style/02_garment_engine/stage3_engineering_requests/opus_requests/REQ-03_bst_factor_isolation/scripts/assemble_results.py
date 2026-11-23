"""REQ-03: merge part1 / part2 / part3 JSON into REQ-03_results.json and add raw-vs-dense separation for every run."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from req03_lib import *

p1 = json.load(open(os.path.join(BASE, 'part1_F0_instrumented.json')))
p2 = json.load(open(os.path.join(BASE, 'part2_runs.json')))
p3 = json.load(open(os.path.join(BASE, 'part3_metric.json')))
rawdense = {}
for name in p2:
    R = load_run(name)
    if 'req03_walk_diag' not in R or 'req03_track_curves' not in R: continue
    wd = R['req03_walk_diag']; tc = R['req03_track_curves']; rr = {}
    for z, (c1, c2) in TRACK_CURVES.items():
        lab = c1.rsplit('_track', 1)[0]
        if lab + '__d1__raw' not in wd or c1 + '__Q' not in tc: continue
        r1, k1 = wd[lab + '__d1__raw'], wd[lab + '__d1__keep']; r2, k2 = wd[lab + '__d2__raw'], wd[lab + '__d2__keep']
        cl = bool(wd[lab + '__crest_closed'])
        sr, _, _ = dist_to_polyline(r2[k2], r1[k1], cl)
        sd, _, _ = dist_to_polyline(tc[c2 + '__Q'], tc[c1 + '__Q'], bool(tc[c1 + '__closed']))
        pair = np.linalg.norm(r2 - r1, axis=1)[k1 & k2]
        rr[z] = dict(pair_same_start=q(pair / MM, (0, 5, 50)), raw2_to_raw1_polyline=q(sr / MM, (0, 5, 50)) | dict(frac_lt_2p850=float((sr < 2.85e-3).mean())),
                     dense2_to_dense1=q(sd / MM, (0, 5, 50)) | dict(frac_lt_2p850=float((sd < 2.85e-3).mean())))
    rawdense[name] = rr
out = dict(request='REQ-03', part1=p1, part2_runs=p2, part2_raw_vs_dense_separation=rawdense, part3=p3)
json.dump(out, open(os.path.join(BASE, 'REQ-03_results.json'), 'w'), indent=1)
for k, v in rawdense.items():
    print(k, {z: (round(x['raw2_to_raw1_polyline']['p5'], 3), round(x['raw2_to_raw1_polyline']['frac_lt_2p850'], 3), round(x['dense2_to_dense1']['p5'], 3), round(x['dense2_to_dense1']['frac_lt_2p850'], 3)) for z, x in v.items()})
