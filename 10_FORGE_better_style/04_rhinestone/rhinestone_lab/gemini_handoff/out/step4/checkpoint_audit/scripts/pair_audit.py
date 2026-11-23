# READ-ONLY audit: independent exact SAT (face axes + all edge-edge axes) on the chaton hull for every collision pair of the exported review bundle.
import json, sys, math, itertools, collections
import numpy as np
sys.path.insert(0, '/home/user/human-engine/better_style_catalog/rhinestone_lab/py')
import chaton_geometry as CG
B = json.load(open(sys.argv[1] if len(sys.argv) > 1 else '/home/user/human-engine/better_style_catalog/rhinestone_lab/out/step4/real_authoring/out/review.review.json'))
St = {s['id']: s for s in B['stones']}
# unique edge directions of the unit hull (in stone frame)
E = set(); V = CG.V
for a, b, c in CG.TRI:
    for u, v in ((a, b), (b, c), (c, a)):
        d = V[v] - V[u]; d = d / np.linalg.norm(d)
        if d[np.argmax(np.abs(d))] < 0: d = -d
        E.add(tuple(np.round(d, 9)))
E = np.array(sorted(E)); U = CG.UNIQ_NORMALS
def frame(s):
    t = s['transform']; n = np.array(t['normal'], float); tg = np.array(t['tangent'], float); z = np.cross(tg, n); return np.array(t['base'], float), n, tg, z, t['diameter']
def hullw(s):
    b, n, tg, z, d = frame(s); return b + d * (V[:, 0:1] * tg + V[:, 1:2] * n + V[:, 2:3] * z), (tg, n, z)
def axes_of(fr):
    tg, n, z = fr; Ax = lambda v: v[0] * tg + v[1] * n + v[2] * z
    return np.array([Ax(u) for u in U]), np.array([Ax(e) for e in E])
def exact(a, b):
    HA, fa = hullw(a); HB, fb = hullw(b); UA, EA = axes_of(fa); UB, EB = axes_of(fb)
    ax = [UA, UB]; X = np.cross(EA[:, None, :], EB[None, :, :]).reshape(-1, 3); nn = np.linalg.norm(X, axis=1); X = X[nn > 1e-9] / nn[nn > 1e-9][:, None]; ax.append(X)
    cA, cB = HA.mean(0), HB.mean(0); ax.append(((cB - cA) / np.linalg.norm(cB - cA))[None])
    ax = np.vstack(ax); pa, pb = HA @ ax.T, HB @ ax.T; s = np.maximum(pb.min(0) - pa.max(0), pa.min(0) - pb.max(0)); k = int(s.argmax())
    face = k < len(UA) + len(UB)
    return float(s.max()), ('face' if face else 'edge-edge/centre'), float(np.linalg.norm(cB - cA))
def circ_radius(P):
    a, b, c = P; ab, bc, ca = np.linalg.norm(b-a), np.linalg.norm(c-b), np.linalg.norm(a-c); cr = np.linalg.norm(np.cross(b-a, c-a))
    return float('inf') if cr < 1e-12 else ab*bc*ca/(2*cr)
rowstones = collections.defaultdict(list)
for s in B['stones']:
    if s['transform']: rowstones[(s['curve_id'], s['row'])].append(s)
for v in rowstones.values(): v.sort(key=lambda s: s['sample'])
curves = {c['curve_id']: c for c in B['curves']}
out = []
for p in B['collisions']:
    a, b = St[p['stone_a']], St[p['stone_b']]
    sep, axis_kind, dc = exact(a, b)
    na, nb = np.array(a['transform']['normal']), np.array(b['transform']['normal'])
    ba, bb = np.array(a['transform']['base']), np.array(b['transform']['base'])
    # local curvature radius from three consecutive same-row stones around the pair
    R = None
    if (a['curve_id'], a['row']) == (b['curve_id'], b['row']):
        rs = rowstones[(a['curve_id'], a['row'])]; ia = [i for i, s_ in enumerate(rs) if s_['id'] == a['id']][0]; ib = [i for i, s_ in enumerate(rs) if s_['id'] == b['id']][0]; lo, hi = min(ia, ib), max(ia, ib); cand = []
        bs = lambda i: np.array(rs[i]['transform']['base'])
        if lo >= 1: cand.append(circ_radius([bs(lo-1), bs(lo), bs(hi)]))
        if hi + 1 < len(rs): cand.append(circ_radius([bs(lo), bs(hi), bs(hi+1)]))
        cand = [c for c in cand if np.isfinite(c)]; R = float(np.mean(cand)) if cand else None
    sm = curves[a['curve_id']]['sampler']; arc = float(np.mean(sm['arc_step_m'])) if sm.get('arc_step_m') else None
    row_arc = None
    out.append(dict(a=a['id'], b=b['id'], category=p['category'], curve=a['curve_id'], curve_b=b['curve_id'], row_a=a['row'], row_b=b['row'], sample_a=a['sample'], sample_b=b['sample'],
        size_a=a['realized_size'], size_b=b['realized_size'], outcome_a=a['outcome'], outcome_b=b['outcome'], centre_dist_mm=dc*1e3, base_dist_mm=float(np.linalg.norm(ba-bb))*1e3,
        arc_step_mm=arc*1e3 if arc else None, touch_mm=(a['transform']['diameter']+b['transform']['diameter'])/2*1e3, engine_sev=p['severity'], engine_depth_mm=p['depth_mm'], engine_sep_mm=p['separation_mm'],
        exact_gap_mm=sep*1e3, exact_axis=axis_kind, normal_angle_deg=float(np.degrees(np.arccos(np.clip(na@nb, -1, 1)))), tangent_angle_deg=float(np.degrees(np.arccos(np.clip(np.array(a['transform']['tangent'])@np.array(b['transform']['tangent']), -1, 1)))),
        radius_mm=R*1e3 if R else None, snap_a=a['snap_mm'], snap_b=b['snap_mm'], lift_a=a['lift_mm'], lift_b=b['lift_mm'], nd_a=a['measurement']['normal_dev_local_deg'] if a['measurement'] else None, nd_b=b['measurement']['normal_dev_local_deg'] if b['measurement'] else None))
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else '/tmp/claude-0/-home-user-human-engine/fe403360-08ed-5551-930f-a67b8d68b9ee/scratchpad/pairs.json', 'w'), indent=1)
print(len(out), 'pairs')
