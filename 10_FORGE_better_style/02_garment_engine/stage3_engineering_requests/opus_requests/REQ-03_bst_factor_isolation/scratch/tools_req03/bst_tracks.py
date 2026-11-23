"""bst_tracks.py - PHASE 2 track machinery: crest (perimeter) extraction, fast-marching geodesic offsets, arc-length curves,
and the E5 spacing solver (closed loops and open segments, seated-chord form).

Why "seated chord": the E5 invariant (spec section 0.2) is on the Euclidean distance between SEATED 3D basal centres, while the
pseudocode spaces stones by arc length and then clips eps_exact. On a curved track chord < arc, and the footing lift moves every
centre along its own N_final, so the pseudocode cannot guarantee the window. Here each next stone is solved (Brent) so that the
seated centre-to-centre distance is exactly R_i + R_j + eps, and eps is solved (Brent) so that the run closes on the end anchor.
"""
import heapq, math
import numpy as np
from scipy.interpolate import splprep, splev, PchipInterpolator
from scipy.optimize import brentq
import bst_core as C


# ---------------------------------------------------------------- level sets on a triangle mesh
def marching(md, tri_ids, field, level, keep=None):
    """Iso-line field == level over the given triangles -> list of (points (k,3), closed). Crossing points are keyed by mesh edge,
    so segments chain exactly; deterministic order (sorted keys)."""
    segs = []; pt = {}
    for t in tri_ids:
        vs = md.T[t]; f = field[vs] - level
        if not np.all(np.isfinite(f)): continue
        cross = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            fa, fb = f[a], f[b]
            if (fa < 0) != (fb < 0):
                key = (min(vs[a], vs[b]), max(vs[a], vs[b]))
                if key not in pt:
                    va, vb = (vs[a], vs[b]) if vs[a] < vs[b] else (vs[b], vs[a]); ga, gb = field[va] - level, field[vb] - level
                    w = ga / (ga - gb); pt[key] = (1 - w) * md.V[va] + w * md.V[vb]
                cross.append(key)
        if len(cross) == 2: segs.append((cross[0], cross[1]))
    if keep is not None and segs:
        mid = np.array([(pt[a] + pt[b]) / 2 for a, b in segs]); m = keep(mid); segs = [sg for sg, k in zip(segs, m) if k]
    adj = {}
    for i, (a, b) in enumerate(segs): adj.setdefault(a, []).append(i); adj.setdefault(b, []).append(i)
    used = np.zeros(len(segs), bool); out = []
    starts = sorted(adj, key=lambda k: (len(adj[k]) != 1, k))       # open chains first (degree-1 keys), then loops
    for k0 in starts:
        for si in adj[k0]:
            if used[si]: continue
            chain = [k0]; cur = k0; s = si
            while s is not None and not used[s]:
                used[s] = True; a, b = segs[s]; nxt = b if a == cur else a; chain.append(nxt); cur = nxt
                s = next((x for x in adj[cur] if not used[x]), None)
            closed = chain[0] == chain[-1] and len(chain) > 3
            P = np.array([pt[k] for k in (chain[:-1] if closed else chain)]); out.append((P, closed))
    return out


def close_if_near(P, closed, gap=8e-3):
    """An open chain whose two ends are within `gap` is a loop interrupted at a seam (the crossing points on the two sides of a
    seam with split vertices do not share an edge key) -> close it."""
    if not closed and len(P) > 10 and np.linalg.norm(P[0] - P[-1]) < gap: return P, True
    return P, closed


def poly_len(P, closed):
    d = np.linalg.norm(np.diff(np.vstack([P, P[:1]]) if closed else P, axis=0), axis=1); return float(d.sum())


def smooth_field(md, f, iters=10, mask=None):
    """Uniform-Laplacian smoothing of a per-vertex field (scalar or vector rows) over mesh edges. Used on the OUTWARD REFERENCE
    DIRECTION o only - never on f = N.o, because across a thin shell rim the outer (+1) and inner (-1) sheets are one edge apart
    and smoothing f destroys the crest (measured: sleeve crest fragmented into <26 mm pieces)."""
    from scipy.sparse import coo_matrix
    e = np.concatenate([md.T[:, [0, 1]], md.T[:, [1, 2]], md.T[:, [2, 0]]]); e = np.concatenate([e, e[:, ::-1]])
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(md.V),) * 2).tocsr(); A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel(); g = f.copy(); dd = np.maximum(deg, 1) if f.ndim == 1 else np.maximum(deg, 1)[:, None]
    for _ in range(iters):
        g = 0.5 * g + 0.5 * (A @ g) / dd
    return g


# ---------------------------------------------------------------- fast marching (virtual-source triangle update)
def fmm_geodesic(md, tri_ids, seeds, d_max=math.inf):
    """Geodesic distance over the triangles tri_ids from seeds {vertex: initial distance}. Triangle update: unfold the triangle,
    put a virtual point source consistent with the two known distances on the far side of the known edge; accept it if the
    straight ray to the unknown vertex crosses the known edge, else fall back to the two edge updates (Dijkstra-like)."""
    v2t = {}
    for t in tri_ids:
        for v in md.T[t]: v2t.setdefault(int(v), []).append(int(t))
    D = {v: math.inf for v in v2t}; alive = set(); heap = []
    for v, d in sorted(seeds.items()):
        if v in D and d < D[v]: D[v] = d; heapq.heappush(heap, (d, v))
    V = md.V
    def upd(c, a, b):
        da, db = D[a], D[b]; A, B, Cc = V[a], V[b], V[c]
        best = min(da + np.linalg.norm(Cc - A) if a in alive else math.inf, db + np.linalg.norm(Cc - B) if b in alive else math.inf)
        if a in alive and b in alive:
            e = B - A; cl = np.linalg.norm(e); ex = e / cl; w = Cc - A; xc = w @ ex; yc = np.linalg.norm(w - xc * ex)
            xs = (da * da - db * db + cl * cl) / (2 * cl); ys2 = da * da - xs * xs
            if ys2 >= 0:
                ys = -math.sqrt(ys2); dc = math.hypot(xc - xs, yc - ys)
                xcross = xs + (xc - xs) * (0 - ys) / (yc - ys) if yc != ys else -1
                if 0 <= xcross <= cl: best = min(best, dc)
        return best
    while heap:
        d, v = heapq.heappop(heap)
        if v in alive or d > D[v]: continue
        if d > d_max: break
        alive.add(v)
        for t in v2t[v]:
            tv = [int(x) for x in md.T[t]]
            for c in tv:
                if c in alive: continue
                a, b = [x for x in tv if x != c]
                nd = upd(c, a, b)
                if nd < D[c]: D[c] = nd; heapq.heappush(heap, (nd, c))
    out = np.full(len(md.V), np.inf)
    for v, d in D.items(): out[v] = d
    return out


def seed_from_crest(md, tri_ids, f):
    """Exact initial distances for vertices of crest-crossing triangles: distance (3D) to the f = 0 segment in that triangle,
    only for vertices on the f > 0 side."""
    seeds = {}
    for t in tri_ids:
        vs = md.T[t]; fv = f[vs]
        if np.all(fv > 0) or np.all(fv <= 0): continue
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (fv[a] > 0) != (fv[b] > 0):
                w = fv[a] / (fv[a] - fv[b]); pts.append((1 - w) * md.V[vs[a]] + w * md.V[vs[b]])
        if len(pts) != 2: continue
        p, q = pts; e = q - p
        for k in range(3):
            if fv[k] <= 0: continue
            x = md.V[vs[k]]; s = np.clip((x - p) @ e / (e @ e), 0, 1); d = np.linalg.norm(x - (p + s * e))
            seeds[int(vs[k])] = min(seeds.get(int(vs[k]), math.inf), d)
    return seeds


# ---------------------------------------------------------------- arc-length curve on the surface
class SurfaceCurve:
    """Closed or open on-surface curve with a dense arc-length table. Closed loops are cut at the canonical lexicographic minimum
    (spec 1.2.3: argmin(x + 1e-4 z) in Blender world for 3D loops) and fitted with a periodic cubic B-spline (C2), then every dense
    sample is projected back onto the surface (closest point). Arc length = cumulative chord of the 0.1 mm dense table."""

    def __init__(self, curve_id, P, closed, md, smooth_mm=0.3, step=0.1 * C.MM):
        self.id, self.closed, self.md = curve_id, closed, md
        if closed:
            k = int(np.argmin(P[:, 0] + 1e-4 * P[:, 2])); P = np.roll(P, -k, 0)
        d = np.linalg.norm(np.diff(np.vstack([P, P[:1]]) if closed else P, axis=0), axis=1); keep = np.concatenate([[True], d[:-1] > 1e-6]) if closed else np.concatenate([[True], d > 1e-6])
        P = P[keep]
        self.despiked = 0
        is_track = '_track' in str(curve_id)                      # REQ-03: F_e knobs act on the post-trim TRACK curves only, never the crest
        if is_track and not CFG['spline_smooth']: smooth_mm = 0.0
        for _ in range(200 if (CFG['despike'] or not is_track) else 0):                                 # iterative despike: drop vertices that turn back by > 60 deg
            Pc = np.vstack([P[-1:], P, P[:1]]) if closed else P
            v1 = Pc[1:-1] - Pc[:-2]; v2 = Pc[2:] - Pc[1:-1]
            c = np.einsum('ij,ij->i', v1, v2) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-30)
            bad = np.nonzero(c < 0.5)[0]
            if not len(bad): break
            if not closed: bad = bad + 1
            bad = bad[np.concatenate([[True], np.diff(bad) > 1])]   # non-adjacent first
            P = np.delete(P, bad, 0); self.despiked += len(bad)
        Pf = np.vstack([P, P[:1]]) if closed else P
        u = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Pf, axis=0), axis=1))]); L0 = u[-1]
        tck, _ = splprep(Pf.T, u=u / L0, s=len(Pf) * (smooth_mm * C.MM) ** 2, per=1 if closed else 0, k=3)
        n = max(64, int(L0 / step)); tt = np.linspace(0, 1, n + 1)
        Q = np.array(splev(tt, tck)).T
        Q = np.array([md.V[md.T[t]].T @ b for t, b in (md.nearest(q) for q in Q)])      # project back onto M
        if closed: Q[-1] = Q[0]
        self.Q = Q; self.s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]); self.L = float(self.s[-1])

    def point(self, s):
        if self.closed: s = s % self.L
        s = min(max(s, 0.0), self.L); i = min(int(np.searchsorted(self.s, s, 'right')) - 1, len(self.s) - 2)
        w = (s - self.s[i]) / max(self.s[i + 1] - self.s[i], 1e-15); return (1 - w) * self.Q[i] + w * self.Q[i + 1]

    def tangent(self, s, h=0.2 * C.MM):
        if self.closed: a, b = self.point(s - h), self.point(s + h)
        else:
            lo = min(max(s - h, 0.0), self.L - 2 * h); a, b = self.point(lo), self.point(lo + 2 * h)
        return (b - a) / np.linalg.norm(b - a)


# ---------------------------------------------------------------- E5 spacing
def window_counts(arc_avail, D):
    """Spec 4.3 Case A integer bounds on span count n for an available length (arc form, used to pick candidates)."""
    n_min = math.ceil(arc_avail / (D + C.EPS_MAX)); n_max = math.floor(arc_avail / (D + C.EPS_MIN))
    n_star = int(round(arc_avail / (D + C.EPS_NOM))); return n_min, n_max, int(min(max(n_star, n_min), max(n_max, n_min)))


def _next_stone(curve, s_prev, P_prev, R_prev, R, eps, seat, s_cap):
    """Smallest s > s_prev with |seat(s).P - P_prev| = R_prev + R + eps (Brent on the seated distance)."""
    target = R_prev + R + eps
    g = lambda s: np.linalg.norm(seat(s)['P'] - P_prev) - target
    lo = s_prev + 0.5 * target; hi = s_prev + 1.05 * target
    while g(hi) < 0:
        hi = s_prev + (hi - s_prev) * 1.5
        if hi > s_cap + 3 * target: return None
    while g(lo) > 0 and lo > s_prev + 1e-6: lo = s_prev + (lo - s_prev) * 0.5
    if g(lo) > 0:                                   # seated position is discontinuous right after s_prev: scan for the first crossing
        xs = np.arange(s_prev + 2e-5, hi, 5e-5); gs = [g(x) for x in xs]
        k = next(i for i, v in enumerate(gs) if v >= 0)
        if k == 0: DISCONTINUITIES.append((curve.id, float(s_prev), float(gs[0]))); return float(xs[0])
        lo, hi = xs[k - 1], xs[k]
    s = brentq(g, lo, hi, xtol=1e-9, rtol=1e-12); return s


DISCONTINUITIES = []


def run_from(curve, s0, st0, R0, R, eps, seat, s_end, max_n=100000):
    """Step stones (radius R) from a placed stone (st0 at s0, radius R0) toward s_end. Returns [(s, seat dict)]."""
    out = []; s, P, Rp = s0, st0['P'], R0
    for _ in range(max_n):
        sn = _next_stone(curve, s, P, Rp, R, eps, seat, s_end)
        if sn is None or sn > s_end: break
        st = seat(sn); out.append((sn, st)); s, P, Rp = sn, st['P'], R
    return out


def solve_closed_loop(curve, R, seat):
    """Closed track of one stone size. n from the spec window on L, eps solved so that the seated closing gap equals eps
    (every consecutive seated gap is then eps to Brent tolerance). Tries n*, then neighbours, keeps the eps closest to eps*
    inside the window."""
    L = curve.L; D = 2 * R; n_min, n_max, n_star = window_counts(L, D)
    st0 = seat(0.0); best = None; attempts = []
    trial = run_from(curve, 0.0, st0, R, R, C.EPS_NOM, seat, L)            # seated chord count at eps* (arc-length n over-counts on wiggly/cornered tracks)
    if not trial: raise RuntimeError(f'{curve.id}: loop shorter than two pitches (L={L/C.MM:.2f} mm)')
    n_star = len(trial) + 1; last = np.linalg.norm(trial[-1][1]['P'] - st0['P']) - 2 * R
    if last > C.EPS_NOM + R: n_star += 1
    for n in sorted(range(n_star - 2, n_star + 3), key=lambda k: (abs(k - n_star), k)):
        def h(eps):
            run = run_from(curve, 0.0, st0, R, R, eps, seat, L * 1.5, max_n=n - 1)
            if len(run) < n - 1: return -1.0
            gap = np.linalg.norm(run[-1][1]['P'] - st0['P']) - 2 * R
            if run[-1][0] >= L: gap = -2 * R - np.linalg.norm(run[-1][1]['P'] - st0['P'])   # wrapped past stone 0: signed (monotone in eps)
            return gap - eps
        e0 = L / n - D; lo, hi = e0 - 0.15 * C.MM, e0 + 0.15 * C.MM
        hlo, hhi = h(lo), h(hi); attempts.append((n, lo / C.MM, hi / C.MM, hlo / C.MM, hhi / C.MM))
        if hlo * hhi > 0: continue
        eps = brentq(h, lo, hi, xtol=1e-10)
        cand = (0 if C.EPS_MIN <= eps <= C.EPS_MAX else 1, abs(eps - C.EPS_NOM), n, eps)
        if best is None or cand < best: best = cand
        if cand[0] == 0: break
    if best is None: raise RuntimeError(f'{curve.id}: no closing eps; L={L/C.MM:.2f}mm attempts={attempts}')
    _, _, n, eps = best
    run = [(0.0, st0)] + run_from(curve, 0.0, st0, R, R, eps, seat, L * 1.5, max_n=n - 1)
    return dict(stones=run, eps=eps, n=n, window=(n_min, n_max, n_star), in_window=C.EPS_MIN <= eps <= C.EPS_MAX)


# ---------------------------------------------------------------- spec 1.3.4 offset: walk along B = N x T on the outer sheet
def outer_sheet_bvh(md, tri_mask):
    from mathutils.bvhtree import BVHTree
    ids = np.nonzero(tri_mask)[0]
    return BVHTree.FromPolygons(md.V.tolist(), md.T[ids].tolist(), all_triangles=True), ids


CFG = dict(trim_tol=0.97, walk_step=None, despike=True, spline_smooth=True, start_full_mesh=False)   # REQ-03 factor knobs (defaults = original)
DIAG = {}                                                                                              # REQ-03 per-walk diagnostics, keyed by (crest id, d)


def offset_walk(crest, d, md, sheet, f, step=0.2e-3, trim_tol=0.97):
    trim_tol = CFG['trim_tol']
    return _offset_walk_req03(crest, d, md, sheet, f, step, trim_tol)


def _offset_walk_req03(crest, d, md, sheet, f, step, trim_tol):
    """REQ-03 instrumented copy of the original offset_walk. Identical arithmetic when CFG is at defaults; additionally records per-start
    diagnostics (start snap, per-step chord / h, normal turn, triangle island changes)."""
    from mathutils import Vector
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    bvh, ids = sheet
    def proj(p):
        loc, _, idx, _ = bvh.find_nearest(Vector(p)); t = int(ids[idx]); P = md.V[md.T[t]]; b = C.bary_of(P, np.array(loc))
        n = b @ md.VN[md.T[t]]; return np.array(loc), n / np.linalg.norm(n), t, b
    def proj_full(p):
        loc, _, idx, _ = md.bvh.find_nearest(Vector(p)); t = int(idx); P = md.V[md.T[t]]; b = C.bary_of(P, np.array(loc))
        n = b @ md.VN[md.T[t]]; return np.array(loc), n / np.linalg.norm(n), t, b
    # face islands of the sheet (shared-vertex connectivity)
    rows = np.repeat(np.arange(len(ids)), 3); cols = md.T[ids].ravel()
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ids), len(md.V))).tocsr(); _, lab = connected_components(A @ A.T, directed=False)
    island = {int(t): int(l) for t, l in zip(ids, lab)}
    S = np.arange(0, crest.L, step * 2)
    starts = [(crest.point(s), crest.tangent(s)) for s in S]
    votes = 0
    for p, t in starts[::max(1, len(starts) // 50)]:
        q, n, tri, b = proj(p); B = np.cross(n, t); B /= np.linalg.norm(B)
        q2, _, t2, b2 = proj(q + 0.5e-3 * B); votes += 1 if b2 @ f[md.T[t2]] > b @ f[md.T[tri]] else -1
    sgn = 1.0 if votes >= 0 else -1.0
    wst = CFG['walk_step'] or step
    k = max(1, int(math.ceil(d / wst))); h = d / k; out = []
    dg = dict(S=S, start=np.array([p for p, _ in starts]), snap=[], snap_edge=[], q0=[], path=[], max_chord_ratio=[], max_turn_deg=[], island_changes=[],
              max_proj_corr=[], n0=[], B0=[], first_tri=[], last_tri=[])
    for p, t in starts:
        q, n, t0, _ = proj_full(p) if CFG['start_full_mesh'] else proj(p); T = t
        dg['snap'].append(float(np.linalg.norm(q - p))); dg['q0'].append(q); dg['n0'].append(n)
        tv = md.V[md.T[t0]]; dg['snap_edge'].append(float(np.mean(np.linalg.norm(tv - np.roll(tv, 1, 0), axis=1))))
        path = 0.0; mcr = 0.0; mturn = 0.0; ich = 0; mpc = 0.0; tri_prev = t0; B0 = None
        for _ in range(k):
            T = T - (T @ n) * n; T /= np.linalg.norm(T); B = sgn * np.cross(n, T)
            if B0 is None: B0 = B
            tgt = q + h * B; qn, nn, tn, _ = proj(tgt)
            ch = float(np.linalg.norm(qn - q)); path += ch; mcr = max(mcr, ch / h); mpc = max(mpc, float(np.linalg.norm(qn - tgt)))
            mturn = max(mturn, float(np.degrees(np.arccos(np.clip(n @ nn, -1, 1)))))
            if island.get(tri_prev, -1) != island.get(tn, -2) and tri_prev in island: ich += 1
            q, n, tri_prev = qn, nn, tn
        out.append(q)
        dg['path'].append(path); dg['max_chord_ratio'].append(mcr); dg['max_turn_deg'].append(mturn); dg['island_changes'].append(ich); dg['max_proj_corr'].append(mpc)
        dg['B0'].append(B0); dg['first_tri'].append(t0); dg['last_tri'].append(tri_prev)
    out = np.array(out)
    kd = cKDTree(crest.Q); dist, _ = kd.query(out)
    keep = dist >= trim_tol * d
    dg = {kk: np.asarray(v) for kk, v in dg.items()}; dg.update(raw=out, dist_vertex=dist, keep=keep, sgn=sgn, h=h, k=k, d=d)
    DIAG[(crest.id, round(d * 1e6))] = dg
    return out[keep], dict(samples=len(out), trimmed=int((~keep).sum()), sign=sgn, euclid_to_crest_mm=[float(dist[keep].min() * 1e3), float(dist[keep].max() * 1e3)])


def _offset_walk_original_unused(crest, d, md, sheet, f, step=0.2e-3, trim_tol=0.97):
    """Gamma_off(s) = Pi_M(Gamma(s) + d B(s)) evaluated as a surface walk: k = ceil(d/step) steps of length d/k along the local
    B = N x T (T parallel-transported), each followed by projection onto the outer sheet (BVH of outer triangles only, so a step
    can never snap to the inner shell). Inward sign of B fixed per curve by the majority of f increase. Concave-corner
    swallowtails are trimmed: an offset point closer than trim_tol*d (Euclidean) to the crest came from the other side of a
    corner. Returns (points (m,3), stats)."""
    from mathutils import Vector
    from scipy.spatial import cKDTree
    bvh, ids = sheet
    def proj(p):
        loc, _, idx, _ = bvh.find_nearest(Vector(p)); t = int(ids[idx]); P = md.V[md.T[t]]; b = C.bary_of(P, np.array(loc))
        n = b @ md.VN[md.T[t]]; return np.array(loc), n / np.linalg.norm(n), t, b
    S = np.arange(0, crest.L, step * 2)
    starts = [(crest.point(s), crest.tangent(s)) for s in S]
    votes = 0
    for p, t in starts[::max(1, len(starts) // 50)]:
        q, n, tri, b = proj(p); B = np.cross(n, t); B /= np.linalg.norm(B)
        q2, _, t2, b2 = proj(q + 0.5e-3 * B); votes += 1 if b2 @ f[md.T[t2]] > b @ f[md.T[tri]] else -1
    sgn = 1.0 if votes >= 0 else -1.0
    k = max(1, int(math.ceil(d / step))); h = d / k; out = []
    for p, t in starts:
        q, n, _, _ = proj(p); T = t
        for _ in range(k):
            T = T - (T @ n) * n; T /= np.linalg.norm(T); B = sgn * np.cross(n, T); q, n, _, _ = proj(q + h * B)
        out.append(q)
    out = np.array(out)
    kd = cKDTree(crest.Q); dist, _ = kd.query(out)
    keep = dist >= trim_tol * d
    return out[keep], dict(samples=len(out), trimmed=int((~keep).sum()), sign=sgn, euclid_to_crest_mm=[float(dist[keep].min() * 1e3), float(dist[keep].max() * 1e3)])


def solve_open_free(curve, R, seat):
    """Open track with both ends free (no locked junction): every gap = eps* exactly (seated chord); the leftover arc is split
    equally between the two ends (run centred)."""
    st = seat(0.0); run = [(0.0, st)] + run_from(curve, 0.0, st, R, R, C.EPS_NOM, seat, curve.L)
    off = 0.5 * (curve.L - run[-1][0])
    st = seat(off); run = [(off, st)] + run_from(curve, off, st, R, R, C.EPS_NOM, seat, curve.L)
    return dict(stones=run, eps=C.EPS_NOM, n=len(run), window=None, in_window=True, open_end_trim_mm=(off * 1e3, (curve.L - run[-1][0]) * 1e3))


def split_at_largest_gap(P):
    d = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1); k = int(np.argmax(d))
    return np.roll(P, -(k + 1), 0), float(d[k])


def solve_closed_loop_modulated(curve, R, seat_r, R_sub=None, k_max=6):
    """Spec 4.3 Case A2 (quantisation deadband) on a closed loop: if no span count n puts the uniform seated gap inside
    [eps_min, eps_max], substitute k stones R -> R_sub (SS16 -> SS10, -1.2 mm each) at evenly spaced indices
    (j = floor(i n / k)), k = 1..k_max, until it does. seat_r(s, radius)."""
    R_sub = R_sub or C.R_SS10
    base = solve_closed_loop(curve, R, lambda s: seat_r(s, R))
    if base['in_window'] or R_sub >= R: base['radii'] = [R] * base['n']; base['substituted'] = 0; return base
    L = curve.L; tries = []
    for k in range(1, k_max + 1):
        for n in (base['n'] - 1, base['n'], base['n'] + 1):
            if n <= k: continue
            radii = [R] * n
            for i in range(k): radii[(i * n) // k] = R_sub
            st0 = seat_r(0.0, radii[0])
            def run_r(eps):
                out = [(0.0, st0)]; s, P, Rp = 0.0, st0['P'], radii[0]
                for j in range(1, n):
                    Rj = radii[j]; sn = _next_stone(curve, s, P, Rp, Rj, eps, (lambda Rj_: (lambda x: seat_r(x, Rj_)))(Rj), L * 1.5)
                    if sn is None: return out
                    st = seat_r(sn, Rj); out.append((sn, st)); s, P, Rp = sn, st['P'], Rj
                return out
            def h(eps):
                r = run_r(eps)
                if len(r) < n: return -1.0
                gap = np.linalg.norm(r[-1][1]['P'] - st0['P']) - radii[-1] - radii[0]
                if r[-1][0] >= L: gap = -radii[-1] - radii[0] - np.linalg.norm(r[-1][1]['P'] - st0['P'])
                return gap - eps
            e0 = (L - 2 * sum(radii)) / n; lo, hi = e0 - 0.3 * C.MM, e0 + 0.3 * C.MM
            hl, hh = h(lo), h(hi); tries.append((k, n, hl / C.MM, hh / C.MM))
            if hl * hh > 0: continue
            eps = brentq(h, lo, hi, xtol=1e-10)
            if C.EPS_MIN <= eps <= C.EPS_MAX:
                return dict(stones=run_r(eps), eps=eps, n=n, window=base['window'], in_window=True, radii=radii, substituted=k)
    base['radii'] = [R] * base['n']; base['substituted'] = 0; base['modulation_tries'] = tries; return base
