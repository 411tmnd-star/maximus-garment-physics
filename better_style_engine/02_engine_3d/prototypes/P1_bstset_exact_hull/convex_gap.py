"""convex_gap.py  PROTOTYPE independent distance between convex solids (numpy only). Not production code.

CONTRACT
  Units / frame   Any consistent length unit, any right-handed frame; nothing is converted. Better Style stones: metres, Blender world (Z-up).
  Solids          A convex solid is given by a SUPPORT FUNCTION  s(d) -> the point of the solid furthest along direction d.
                    PolytopeSupport(vertices)            convex hull of a point set (the octagonal chaton_v1 hull: 24 vertices)
                    CirclesSupport(centres, axis, radii)  convex hull of coaxial circles (a ROUND stone: base, girdle-top and table circles); exact, no faceting
  Methods         gjk_distance(A, B)       Gilbert-Johnson-Keerthi on the Minkowski difference. Returns (upper, lower, iterations, hit):
                                           upper = |v| of the best point found (a true upper bound of the distance),
                                           lower = max over iterations of v.w/|v| (a true lower bound; a separating-plane certificate),
                                           hit   = True when the solids intersect or touch (origin inside the simplex, distance 0).
                                           The pair (lower, upper) brackets the exact distance; the bracket width is reported, never assumed.
                  polytope_distance(HA, HB, TRI, EDGES)   independent second method for two DISJOINT triangulated convex polytopes:
                                           min over vertex-triangle (both ways) and edge-edge distances. Exact for the piecewise-linear hulls.
  Not provided    Penetration depth. For intersecting pairs this module only reports hit = True; depth is taken from a separating-axis test.
  Degenerate      Coincident solids, zero direction, flat simplices: handled by the sub-simplex enumeration (least squares, rank-deficient safe)."""
import itertools
import numpy as np


class PolytopeSupport:
    def __init__(self, vertices): self.V = np.asarray(vertices, float); self.centre = self.V.mean(0)
    def __call__(self, d): return self.V[int(np.argmax(self.V @ d))]


class CirclesSupport:
    """Convex hull of circles that share one axis. centres (k,3), axis unit (3,), radii (k,)."""
    def __init__(self, centres, axis, radii):
        self.C = np.asarray(centres, float); self.n = np.asarray(axis, float) / np.linalg.norm(axis); self.r = np.asarray(radii, float); self.centre = self.C.mean(0)
    def __call__(self, d):
        dp = d - (d @ self.n) * self.n; L = np.linalg.norm(dp); k = int(np.argmax(self.C @ d + self.r * L))
        return self.C[k] if L < 1e-300 else self.C[k] + self.r[k] * dp / L


_SUBSETS = {k: [c for m in range(1, k + 1) for c in itertools.combinations(range(k), m)] for k in (1, 2, 3, 4)}


def _closest_on_simplex(W):
    """Closest point to the origin on the simplex with vertices W (k,3), k <= 4. Enumerates every sub-simplex, keeps those whose affine
    minimiser has non-negative barycentrics, returns the smallest. Returns (point, vertex subset, barycentrics)."""
    best = None
    for sub in _SUBSETS[len(W)]:
        S = W[list(sub)]
        if len(sub) == 1: lam = np.ones(1)
        else:
            E = (S[1:] - S[0]).T                                                 # 3 x (m-1)
            t, *_ = np.linalg.lstsq(E, -S[0], rcond=None); lam = np.concatenate([[1 - t.sum()], t])
        if (lam < -1e-14).any(): continue
        p = lam @ S; d2 = float(p @ p)
        if best is None or d2 < best[0] - 1e-300: best = (d2, p, sub, lam)
    return best[1], best[2], best[3]


def gjk_distance(A, B, rel_tol=1e-12, max_iter=200):
    d = A.centre - B.centre
    if not d.any(): d = np.array([1.0, 0, 0])
    w = A(-d) - B(d); W = w[None]; v = w; lower = 0.0; it = 0; scale = max(np.linalg.norm(A.centre), np.linalg.norm(B.centre), 1e-30)
    for it in range(1, max_iter + 1):
        nv = np.linalg.norm(v)
        if nv <= 1e-13 * max(scale, np.abs(W).max()): return 0.0, 0.0, it, True      # origin on the simplex: touching or intersecting
        w = A(-v) - B(v); lb = float(v @ w) / nv; lower = max(lower, lb)
        if nv - lower <= rel_tol * nv: return float(nv), float(lower), it, False
        if any(np.array_equal(w, x) for x in W): return float(nv), float(lower), it, False   # no new support point: converged for polytopes
        W = np.vstack([W, w]); v, sub, lam = _closest_on_simplex(W); W = W[list(sub)]
        if len(sub) == 4: return 0.0, 0.0, it, True                                  # origin inside a tetrahedron of the difference
    return float(np.linalg.norm(v)), float(lower), it, False


# ------------------------------------------------------------------ independent polytope method (feature pairs)
def _pt_tri(P, A, B, C):
    """Closest point on triangle (Ericson 5.1.5), broadcast shapes (...,3). Returns distance."""
    ab = B - A; ac = C - A; ap = P - A; d1 = (ab * ap).sum(-1); d2 = (ac * ap).sum(-1); bp = P - B; d3 = (ab * bp).sum(-1); d4 = (ac * bp).sum(-1)
    cp = P - C; d5 = (ab * cp).sum(-1); d6 = (ac * cp).sum(-1); va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        den = va + vb + vc; q = A + ab * (vb / den)[..., None] + ac * (vc / den)[..., None]
        q = np.where(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[..., None], B + (C - B) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[..., None], q)
        q = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None], A + ac * (d2 / (d2 - d6))[..., None], q)
        q = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None], A + ab * (d1 / (d1 - d3))[..., None], q)
    q = np.where(((d6 >= 0) & (d5 <= d6))[..., None], C, q); q = np.where(((d3 >= 0) & (d4 <= d3))[..., None], B, q); q = np.where(((d1 <= 0) & (d2 <= 0))[..., None], A, q)
    return np.linalg.norm(P - q, axis=-1)


def _seg_seg(p1, q1, p2, q2):
    """Segment-segment distance (Ericson 5.1.9), broadcast."""
    d1 = q1 - p1; d2 = q2 - p2; r = p1 - p2; a = (d1 * d1).sum(-1); e = (d2 * d2).sum(-1); f = (d2 * r).sum(-1); c = (d1 * r).sum(-1); b = (d1 * d2).sum(-1); den = a * e - b * b
    s = np.where(den > 1e-300, np.clip((b * f - c * e) / np.where(den > 1e-300, den, 1.0), 0, 1), 0.0); t = (b * s + f) / e
    s = np.where(t < 0, np.clip(-c / a, 0, 1), np.where(t > 1, np.clip((b - c) / a, 0, 1), s)); t = np.clip(t, 0, 1)
    return np.linalg.norm((p1 + d1 * s[..., None]) - (p2 + d2 * t[..., None]), axis=-1)


def polytope_distance(HA, HB, TRI, EDGES):
    def vt(H1, H2): return _pt_tri(H1[:, None, :], H2[TRI[:, 0]][None], H2[TRI[:, 1]][None], H2[TRI[:, 2]][None]).min()
    ee = _seg_seg(HA[EDGES[:, 0]][:, None], HA[EDGES[:, 1]][:, None], HB[EDGES[:, 0]][None], HB[EDGES[:, 1]][None]).min()
    return float(min(vt(HA, HB), vt(HB, HA), ee))


# ------------------------------------------------------------------ the stone models (parameters are INPUTS, read from chaton_v1.params.json by the caller)
def chaton_unit(params):
    """Unit-diameter octagonal chaton_v1: vertices (24,3) in the stone frame (x tangent, y normal/crown, z = x cross y), triangles (44,3), edges (66,2).
    Built here from the parameter file alone (independent of the lab's chaton_geometry.py)."""
    n, R, rt, hr, H, ph = params["sides"], params["girdle_radius"], params["table_radius"], params["rim_height"], params["total_height"], np.radians(params["table_phase_deg"])
    def ring(r, y, off=0.0):
        a = off + np.arange(n) * 2 * np.pi / n; return np.stack([r * np.cos(a), np.full(n, y), -r * np.sin(a)], 1)
    V = np.vstack([ring(R, 0.0), ring(R, hr), ring(rt, H, ph)])
    from scipy.spatial import ConvexHull
    hull = ConvexHull(V); T = hull.simplices.copy(); c = V.mean(0)
    for k, (a, b, cc) in enumerate(T):
        if np.cross(V[b] - V[a], V[cc] - V[a]) @ (V[a] - c) < 0: T[k] = (a, cc, b)
    E = np.array(sorted({tuple(sorted(e)) for a, b, cc in T for e in ((a, b), (b, cc), (cc, a))}))
    return V, T, E


def stone_frame(N, T):
    n = N / np.linalg.norm(N); t = T - (T @ n) * n; t = t / np.linalg.norm(t); return t, n, np.cross(t, n)


def hull_world(Vunit, P, N, T, diameter):
    t, n, z = stone_frame(N, T); return P + diameter * (Vunit[:, 0:1] * t + Vunit[:, 1:2] * n + Vunit[:, 2:3] * z)


def round_support(params, P, N, diameter):
    """Round stone with the same profile as chaton_v1 (base circle, girdle-top circle, table circle). The octagon is INSCRIBED in the girdle circle,
    so this solid contains the octagonal hull."""
    n = N / np.linalg.norm(N); ys = np.array([0.0, params["rim_height"], params["total_height"]]) * diameter
    rs = np.array([params["girdle_radius"], params["girdle_radius"], params["table_radius"]]) * diameter
    return CirclesSupport(P + ys[:, None] * n, n, rs)
