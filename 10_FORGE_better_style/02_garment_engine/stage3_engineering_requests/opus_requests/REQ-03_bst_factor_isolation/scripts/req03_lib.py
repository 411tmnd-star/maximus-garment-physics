"""REQ-03 analysis helpers (pure numpy/scipy; reads scratch run folders only)."""
import json, os
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

MM = 1e-3; R10 = 1.40e-3; EPS_MIN = 0.05e-3
BASE = os.path.normpath(os.path.expanduser('~/Desktop/better_style_execution_workspace/08_KNOWLEDGE/stage3_engineering/opus_requests/REQ-03_bst_factor_isolation'))
RUNS = os.path.join(BASE, 'scratch', 'runs')
TRACK_CURVES = {'A': ('A_collar_track1', 'A_collar_track2'), 'B_L': ('B_cuff_L_track1', 'B_cuff_L_track2'), 'B_R': ('B_cuff_R_track1', 'B_cuff_R_track2')}


def q(a, ps=(0, 5, 50, 95, 100)):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a): return None
    return {f'p{p}' if p not in (0, 100) else ('min' if p == 0 else 'max'): float(np.percentile(a, p)) for p in ps} | {'n': int(len(a))}


def corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0: return dict(n=int(m.sum()), pearson=None, spearman=None)
    return dict(n=int(m.sum()), pearson=float(pearsonr(x[m], y[m])[0]), spearman=float(spearmanr(x[m], y[m])[0]))


def poly_segments(Q, closed):
    A = Q; B = np.roll(Q, -1, 0)
    if not closed: A, B = Q[:-1], Q[1:]
    else:
        if np.allclose(Q[0], Q[-1]): A, B = Q[:-1], Q[1:]
    return A, B


def dist_to_polyline(X, Q, closed, k=16):
    """Exact point-to-polyline distance (segments), candidate segments = those adjacent to the k nearest vertices."""
    A, B = poly_segments(Q, closed); nseg = len(A)
    kd = cKDTree(A); _, idx = kd.query(X, k=min(k, nseg)); idx = np.atleast_2d(idx)
    cand = np.concatenate([idx, (idx - 1) % nseg], 1)
    a = A[cand]; b = B[cand]; e = b - a; w = X[:, None, :] - a
    t = np.clip((w * e).sum(-1) / np.maximum((e * e).sum(-1), 1e-30), 0, 1)
    d = np.linalg.norm(w - t[..., None] * e, axis=-1); j = d.argmin(1)
    return d[np.arange(len(X)), j], cand[np.arange(len(X)), j], t[np.arange(len(X)), j]


def spacing(Q, closed):
    A, B = poly_segments(Q, closed); return np.linalg.norm(B - A, axis=1)


def crest_kappa_g(Qc, Nc, closed, sgn, delta=1.0e-3):
    """Signed geodesic curvature of the crest at every crest vertex. Estimator: arc-length resample, second difference
    k = (p(s-D) - 2p(s) + p(s+D)) / D^2 with D = delta, normal component (crest Phong normal N) removed, then projected on the
    offset direction Bdir = sgn * (N x T). kappa_g > 0: the crest bends TOWARD the track side (offsets compress there)."""
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Qc, axis=0), axis=1))]); L = s[-1]
    def P(x):
        x = np.mod(x, L) if closed else np.clip(x, 0, L)
        return np.stack([np.interp(x, s, Qc[:, i]) for i in range(3)], 1)
    a, b, c = P(s - delta), P(s), P(s + delta)
    kv = (a - 2 * b + c) / delta ** 2; T = (c - a); T /= np.linalg.norm(T, axis=1, keepdims=True)
    N = Nc - (Nc * T).sum(1, keepdims=True) * T; N /= np.linalg.norm(N, axis=1, keepdims=True)
    kt = kv - (kv * N).sum(1, keepdims=True) * N
    Bd = sgn * np.cross(N, T)
    return (kt * Bd).sum(1), s


def load_run(name):
    d = os.path.join(RUNS, name); r = {}
    for f in ('req03_walk_diag', 'req03_track_curves', 'req03_prune', 'tracks_raw', 'zone_ab_stones'):
        p = os.path.join(d, f + '.npz')
        if os.path.exists(p): r[f] = dict(np.load(p, allow_pickle=False))
    for f in ('phase3_audit', 'zone_ab_report', 'zone_ab_extract_report', 'req03_run'):
        p = os.path.join(d, f + '.json')
        if os.path.exists(p): r[f] = json.load(open(p))
    return r


def track_separation(tc, z):
    """Distance from every dense track-2 curve sample (0.1 mm table) to the dense track-1 polyline."""
    c1, c2 = TRACK_CURVES[z]
    if c1 + '__Q' not in tc: return None
    d, _, _ = dist_to_polyline(tc[c2 + '__Q'], tc[c1 + '__Q'], bool(tc[c1 + '__closed']))
    return d


def prune_summary(pr, curves_closed):
    zone = pr['zone']; curve = pr['curve']; keep = pr['keep']; occl = pr['occl']; P = pr['P']; R = pr['R']; s = pr['s']
    rej = (~keep) & (~occl)
    out = {}
    for z, lab in (('A', 'A_collar'), ('B', 'B_cuff_L'), ('B', 'B_cuff_R')):
        row = {}
        for tk in (1, 2):
            cv = f'{lab}_track{tk}'; m = curve == cv
            row[f'track{tk}_pre'] = int(m.sum()); row[f'track{tk}_rejected_clearance'] = int((m & rej).sum()); row[f'track{tk}_occluded'] = int((m & occl).sum()); row[f'track{tk}_final'] = int((m & keep).sum())
            idx = np.nonzero(m & keep)[0]; idx = idx[np.argsort(s[idx], kind='stable')]
            if len(idx) > 1:
                pairs = list(zip(idx[:-1], idx[1:])) + ([(idx[-1], idx[0])] if cv in curves_closed else [])
                g = np.array([np.linalg.norm(P[i] - P[j]) - R[i] - R[j] for i, j in pairs]); row[f'track{tk}_max_void_mm'] = float(g.max() / MM)
        out[lab] = row
    out['zone_rejects'] = {zz: int((rej & (zone == zz)).sum()) for zz in 'ABC'}
    out['zone_occluded'] = {zz: int((occl & (zone == zz)).sum()) for zz in 'ABC'}
    out['final_total'] = int(keep.sum()); out['pre_total'] = int(len(keep))
    return out


# ------------------------------------------------------------------------------- exact convex-hull distance (chaton_v1)
def _closest_pt_tri(p, a, b, c):
    """Ericson closest point on triangle, vectorised over broadcast shapes (...,3)."""
    ab = b - a; ac = c - a; ap = p - a
    d1 = (ab * ap).sum(-1); d2 = (ac * ap).sum(-1)
    bp = p - b; d3 = (ab * bp).sum(-1); d4 = (ac * bp).sum(-1)
    cp = p - c; d5 = (ab * cp).sum(-1); d6 = (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
    den = va + vb + vc; den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    v = vb / den; w = vc / den; res = a + ab * v[..., None] + ac * w[..., None]
    # edge / vertex regions
    with np.errstate(divide='ignore', invalid='ignore'):
        cAB = (vc <= 0) & (d1 >= 0) & (d3 <= 0); tAB = d1 / (d1 - d3)
        cAC = (vb <= 0) & (d2 >= 0) & (d6 <= 0); tAC = d2 / (d2 - d6)
        cBC = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0); tBC = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    res = np.where(cBC[..., None], b + (c - b) * np.nan_to_num(tBC)[..., None], res)
    res = np.where(cAC[..., None], a + ac * np.nan_to_num(tAC)[..., None], res)
    res = np.where(cAB[..., None], a + ab * np.nan_to_num(tAB)[..., None], res)
    res = np.where(((d6 >= 0) & (d5 <= d6))[..., None], c, res)
    res = np.where(((d3 >= 0) & (d4 <= d3))[..., None], b, res)
    res = np.where(((d1 <= 0) & (d2 <= 0))[..., None], a, res)
    return res


def _seg_seg(p1, q1, p2, q2):
    d1 = q1 - p1; d2 = q2 - p2; r = p1 - p2; a = (d1 * d1).sum(-1); e = (d2 * d2).sum(-1); f = (d2 * r).sum(-1)
    c = (d1 * r).sum(-1); b = (d1 * d2).sum(-1); den = a * e - b * b
    s = np.where(den > 1e-30, np.clip((b * f - c * e) / np.where(den > 1e-30, den, 1), 0, 1), 0.0)
    t = (b * s + f) / e
    s = np.where(t < 0, np.clip(-c / a, 0, 1), np.where(t > 1, np.clip((b - c) / a, 0, 1), s)); t = np.clip(t, 0, 1)
    return np.linalg.norm((p1 + d1 * s[..., None]) - (p2 + d2 * t[..., None]), axis=-1)


def hull_distance(HA, HB, TRI, EI, EJ):
    """Exact Euclidean distance between two DISJOINT convex polytopes given by triangulated hulls: min of vertex-triangle (both ways)
    and edge-edge distances."""
    def vt(H1, H2):
        p = H1[:, None, :]; a = H2[TRI[:, 0]][None]; b = H2[TRI[:, 1]][None]; c = H2[TRI[:, 2]][None]
        return np.linalg.norm(p - _closest_pt_tri(p, a, b, c), axis=-1).min()
    ee = _seg_seg(HA[EI][:, None], HA[EJ][:, None], HB[EI][None], HB[EJ][None]).min()
    return float(min(vt(HA, HB), vt(HB, HA), ee))
