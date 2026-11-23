"""exact_hull.py  INDEPENDENT exact verification of crystal-crystal intersection (pure numpy, no Blender).

This module does NOT replace or modify py/collisions.py. The collision engine remains the authoritative detector. This module exists to verify it:
the engine's narrow phase tests face axes and the centre axis only (conservative: it can over-report, never under-report). Here the same chaton_v1 hull (24 vertices, 44 triangles)
is tested with two independent exact methods:
  separation_exact   full separating-axis test: face normals of both hulls + all edge-edge cross products + centre axis. gap_m > 0: separated by that distance along the best axis
                     (a lower bound of the true distance); gap_m <= 0: intersecting, -gap_m is the minimum translation distance (penetration depth).
  intersects_edges   exact edge-triangle intersection of the two triangulated hulls (no axes involved).
Both must agree; verify_pairs() checks every engine pair and reports false positives (engine OVERLAP, hulls separated) and, with find_all_intersections(), false negatives.
All lengths are metres; fields that carry millimetres say so (`_mm`)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chaton_geometry as CG

V = CG.V; TRI = CG.TRI; UNIQ = CG.UNIQ_NORMALS
_EDGES = sorted({tuple(sorted(e)) for a, b, c in TRI for e in ((a, b), (b, c), (c, a))})
_EDIR = []
for _i, _j in _EDGES:
    _d = V[_j] - V[_i]; _d = _d / np.linalg.norm(_d)
    if _d[int(np.argmax(np.abs(_d)))] < 0: _d = -_d
    _EDIR.append(tuple(np.round(_d, 9)))
EDGE_DIRS = np.array(sorted(set(_EDIR)))                       # unique edge directions of the unit hull, in the stone frame
CIRCUM_RADIUS = float(np.linalg.norm(V - V.mean(0), axis=1).max())     # of the unit-diameter hull


def frame(normal, tangent):
    n = np.asarray(normal, float); t = np.asarray(tangent, float); return t, n, np.cross(t, n)


def hull_world(base, normal, tangent, diameter):
    """World vertices (24, 3) of a stone: X = tangent, Y = normal (crown), Z = X x Y; scale = diameter; origin = base centre (the frame the placement code writes)."""
    t, n, z = frame(normal, tangent); b = np.asarray(base, float)
    return b + diameter * (V[:, 0:1] * t + V[:, 1:2] * n + V[:, 2:3] * z)


def _axes(t, n, z):
    face = UNIQ[:, 0:1] * t + UNIQ[:, 1:2] * n + UNIQ[:, 2:3] * z; edge = EDGE_DIRS[:, 0:1] * t + EDGE_DIRS[:, 1:2] * n + EDGE_DIRS[:, 2:3] * z; return face, edge


def separation_exact(a, b):
    """a, b: dicts with base, normal, tangent (3-vectors) and diameter (m). Returns {'gap_m', 'axis' ('face' | 'edge_edge' | 'centre'), 'centre_distance_m'}."""
    HA = hull_world(a['base'], a['normal'], a['tangent'], a['diameter']); HB = hull_world(b['base'], b['normal'], b['tangent'], b['diameter'])
    ta, na, za = frame(a['normal'], a['tangent']); tb, nb, zb = frame(b['normal'], b['tangent'])
    fa, ea = _axes(ta, na, za); fb, eb = _axes(tb, nb, zb)
    X = np.cross(ea[:, None, :], eb[None, :, :]).reshape(-1, 3); nn = np.linalg.norm(X, axis=1); X = X[nn > 1e-9] / nn[nn > 1e-9][:, None]
    cA, cB = HA.mean(0), HB.mean(0); dc = float(np.linalg.norm(cB - cA)); cen = ((cB - cA) / dc)[None] if dc > 1e-12 else np.zeros((0, 3))
    ax = np.vstack([fa, fb, X, cen]); pa, pb = HA @ ax.T, HB @ ax.T; s = np.maximum(pb.min(0) - pa.max(0), pa.min(0) - pb.max(0)); k = int(s.argmax())
    kind = 'face' if k < len(fa) + len(fb) else ('edge_edge' if k < len(fa) + len(fb) + len(X) else 'centre')
    return {'gap_m': float(s.max()), 'axis': kind, 'centre_distance_m': dc}


_EI = np.array([e[0] for e in _EDGES]); _EJ = np.array([e[1] for e in _EDGES]); _T0, _T1, _T2 = TRI[:, 0], TRI[:, 1], TRI[:, 2]


def _edges_vs_tris(H_edges, H_tris, eps=1e-12, tol=1e-9):
    """Moller-Trumbore for ALL (edge, triangle) combinations at once: edges of hull X against triangles of hull Y. True if any segment crosses any triangle."""
    p = H_edges[_EI][:, None, :]; q = H_edges[_EJ][:, None, :]; a = H_tris[_T0][None]; b = H_tris[_T1][None]; c = H_tris[_T2][None]
    d = q - p; e1 = b - a; e2 = c - a; h = np.cross(d, e2); det = (e1 * h).sum(-1); ok = np.abs(det) > eps; f = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = p - a; u = f * (s * h).sum(-1); qv = np.cross(s, e1); v = f * (d * qv).sum(-1); t = f * (e2 * qv).sum(-1)
    return bool((ok & (u >= -tol) & (u <= 1 + tol) & (v >= -tol) & (u + v <= 1 + tol) & (t >= -tol) & (t <= 1 + tol)).any())


def intersects_edges(a, b):
    """Exact edge-triangle test (each hull's edges against the other's triangles). Does not use separating axes, so it is an independent check of separation_exact."""
    HA = hull_world(a['base'], a['normal'], a['tangent'], a['diameter']); HB = hull_world(b['base'], b['normal'], b['tangent'], b['diameter'])
    return _edges_vs_tris(HA, HB) or _edges_vs_tris(HB, HA)


def stone_from_record(rec):
    t = rec['transform']; return {'base': np.asarray(t['base'], float), 'normal': np.asarray(t['normal'], float), 'tangent': np.asarray(t['tangent'], float), 'diameter': float(t['diameter'])}


def verify_pairs(stones, pairs, near_clearance_m=1e-4):
    """stones: {stone_id: stone dict}; pairs: engine collision pairs ({'stone_a','stone_b','severity',...}).
    Returns one dict per pair with both exact verdicts and the comparison with the engine's severity. `false_positive` = the engine called it OVERLAP_* and the hulls are separated.
    `method_disagreement` = the two exact methods disagree (should never happen)."""
    out = []
    for p in pairs:
        a, b = stones[p['stone_a']], stones[p['stone_b']]; sep = separation_exact(a, b); inter = intersects_edges(a, b); sat_inter = sep['gap_m'] <= -1e-9
        eng_over = str(p['severity']).startswith('OVERLAP')
        out.append({'stone_a': p['stone_a'], 'stone_b': p['stone_b'], 'engine_severity': p['severity'], 'engine_depth_mm': p.get('depth_mm'), 'exact_gap_mm': sep['gap_m'] * 1e3, 'exact_separating_axis': sep['axis'], 'centre_distance_mm': sep['centre_distance_m'] * 1e3,
                    'intersects_exact': bool(inter), 'method_disagreement': bool(inter != sat_inter), 'false_positive': bool(eng_over and not inter), 'near_clearance_exact': bool((not inter) and sep['gap_m'] < near_clearance_m),
                    'exact_depth_minus_engine_depth_mm': (-sep['gap_m'] * 1e3 - p['depth_mm']) if (inter and p.get('depth_mm') is not None) else None})
    return out


def find_all_intersections(stones, ids=None):
    """Brute-force (broad phase by circumscribed radii) exact scan of ALL pairs, independent of the engine's spatial hash. Returns the set of frozenset({id_a, id_b}) that intersect."""
    ids = list(stones) if ids is None else list(ids); C = np.array([hull_world(**{k: stones[i][k] for k in ('base', 'normal', 'tangent', 'diameter')}).mean(0) for i in ids]); R = np.array([CIRCUM_RADIUS * stones[i]['diameter'] for i in ids]); found = set()
    for i in range(len(ids)):
        d = np.linalg.norm(C[i + 1:] - C[i], axis=1); for_j = np.flatnonzero(d < R[i] + R[i + 1:] + 1e-9)
        for jj in for_j:
            j = i + 1 + int(jj)
            if intersects_edges(stones[ids[i]], stones[ids[j]]): found.add(frozenset((ids[i], ids[j])))
    return found
