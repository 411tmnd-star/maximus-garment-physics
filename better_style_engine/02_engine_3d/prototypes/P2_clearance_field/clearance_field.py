"""clearance_field.py  PROTOTYPE reusable clearance field (numpy + scipy only; no Blender). Not production code.

CONTRACT
  Units / frame   Any consistent length unit and any right-handed Cartesian frame; the library never converts. Callers state both (Better Style: metres, Blender Z-up).
                  Results are invariant under rigid motion applied to mesh and queries together.
  Surface input   Triangle mesh: V (n,3) float, T (m,3) int. Polygons must be triangulated by the caller (declare how).
                  `TriSurface.facts` reports: closed (no boundary edges), manifold (every edge has exactly 2 faces), boundary / non-manifold edge counts,
                  degenerate triangles (excluded from all queries), consistent orientation.
  Queries         closest_point(P)      exact unsigned distance to the triangle set, closest point, triangle id, barycentric coords, feature (face / edge / vertex).
                  winding_number(P)     generalised winding number (sum of signed solid angles / 4 pi), exact summation.
                  signed_distance(P)    unsigned distance with a sign from a NAMED rule:
                      'winding'       inside iff w > 0.5.  Defined for closed surfaces, also self-intersecting ones (w can be 2). Refused on open surfaces unless allow_open.
                      'pseudonormal'  sign of (p - c) . n_feature with the angle-weighted pseudonormal of the closest feature. Exact for closed, manifold, non-self-intersecting surfaces.
                      'face_normal'   sign of (p - c) . n_face of the closest triangle. LEGACY rule (what the engine's builders and probes use). Ambiguous at edges and
                                      vertices and wrong in concavities; provided only to reproduce existing numbers.
                  sign_ambiguity(P, tol) for each query: how many faces tie for nearest (within tol) and whether their face-normal signs disagree.
                                        Where they disagree the 'face_normal' rule has no defined answer: the result depends on the search's tie-break.
                  self_contact(eps)     vertices of the surface that lie within eps of a non-neighbouring, opposing part of the same surface.
                  clearance_report      percentiles and counts of signed clearance for a point set, optionally by label.
  Accuracy        closest_point is exact for the piecewise-linear surface up to floating point (the acceleration is a provably complete candidate search,
                  verified against brute force in the tests). winding_number is exact summation; its value is not meaningful within ~1e-9 of the surface.
                  Distances are to the TRIANGULATED surface: against a smooth original the error is bounded by the triangulation's sag.
  Performance     closest_point: KD-tree on triangle centroids; per query all triangles whose centroid lies within (upper bound + largest triangle radius).
                  winding_number: O(queries x triangles), chunked; 1e9 pair evaluations is about a minute on one core. No hierarchical approximation in this prototype."""
import numpy as np
from scipy.spatial import cKDTree

_TOL_BARY = 1e-12


def weld(V, T, tol=0.0):
    """Merge coincident vertices (exactly equal when tol == 0, else on a grid of size tol). Returns V2, T2, index map old->new. Triangles that collapse are dropped."""
    V = np.asarray(V, float); key = V if tol == 0 else np.rint(V / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True); inv = inv.reshape(-1)
    T2 = inv[np.asarray(T)]; ok = (T2[:, 0] != T2[:, 1]) & (T2[:, 1] != T2[:, 2]) & (T2[:, 0] != T2[:, 2])
    return V[first], T2[ok], inv


def _closest_on_triangles(P, A, B, C):
    """Vectorised closest point on triangle (Ericson, Real-Time Collision Detection 5.1.5). P, A, B, C: (k,3). Returns closest (k,3), bary (k,3)."""
    ab = B - A; ac = C - A; ap = P - A
    d1 = (ab * ap).sum(1); d2 = (ac * ap).sum(1); bp = P - B; d3 = (ab * bp).sum(1); d4 = (ac * bp).sum(1); cp = P - C; d5 = (ab * cp).sum(1); d6 = (ac * cp).sum(1)
    vc = d1 * d4 - d3 * d2; vb = d5 * d2 - d1 * d6; va = d3 * d6 - d5 * d4
    k = len(P); u = np.zeros(k); v = np.zeros(k); w = np.zeros(k); done = np.zeros(k, bool)
    def put(m, uu, vv, ww):
        m = m & ~done; u[m] = uu[m] if np.ndim(uu) else uu; v[m] = vv[m] if np.ndim(vv) else vv; w[m] = ww[m] if np.ndim(ww) else ww; done[m] = True
    put((d1 <= 0) & (d2 <= 0), 1.0, 0.0, 0.0)                                                   # vertex A
    put((d3 >= 0) & (d4 <= d3), 0.0, 1.0, 0.0)                                                  # vertex B
    with np.errstate(divide="ignore", invalid="ignore"):
        t = d1 / (d1 - d3); put((vc <= 0) & (d1 >= 0) & (d3 <= 0), 1 - t, t, np.zeros(k))       # edge AB
        put((d6 >= 0) & (d5 <= d6), 0.0, 0.0, 1.0)                                              # vertex C
        t = d2 / (d2 - d6); put((vb <= 0) & (d2 >= 0) & (d6 <= 0), 1 - t, np.zeros(k), t)       # edge AC
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6)); put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), np.zeros(k), 1 - t, t)   # edge BC
        den = va + vb + vc; put(np.ones(k, bool), va / den, vb / den, vc / den)                 # interior
    bary = np.stack([u, v, w], 1); bary = np.where(np.isfinite(bary), bary, 1.0 / 3.0)
    return bary[:, :1] * A + bary[:, 1:2] * B + bary[:, 2:] * C, bary


class TriSurface:
    def __init__(self, V, T, name=None):
        self.V = np.ascontiguousarray(V, float); T = np.asarray(T, np.int64); self.name = name
        if self.V.ndim != 2 or self.V.shape[1] != 3 or T.ndim != 2 or T.shape[1] != 3: raise ValueError("V must be (n,3), T (m,3)")
        if not np.isfinite(self.V).all(): raise ValueError("non-finite vertex")
        A, B, C = self.V[T[:, 0]], self.V[T[:, 1]], self.V[T[:, 2]]; n = np.cross(B - A, C - A); a2 = np.linalg.norm(n, axis=1)
        good = a2 > 1e-30; self.n_degenerate = int((~good).sum()); self.T = T[good]; A, B, C, n, a2 = A[good], B[good], C[good], n[good], a2[good]
        self.A, self.B, self.C = A, B, C; self.FN = n / a2[:, None]; self.area = 0.5 * a2
        self.cen = (A + B + C) / 3.0; self.rmax = float(np.sqrt(np.max([((X - self.cen) ** 2).sum(1) for X in (A, B, C)]))) if len(A) else 0.0
        self.tree = cKDTree(self.cen) if len(A) else None
        e = np.sort(np.vstack([self.T[:, [0, 1]], self.T[:, [1, 2]], self.T[:, [2, 0]]]), 1); ue, cnt = np.unique(e, axis=0, return_counts=True)
        d = np.vstack([self.T[:, [0, 1]], self.T[:, [1, 2]], self.T[:, [2, 0]]]); _, dcnt = np.unique(d, axis=0, return_counts=True)       # a directed edge used twice = inconsistent orientation
        self.facts = dict(name=name, vertices=int(len(self.V)), triangles=int(len(self.T)), degenerate_triangles_dropped=self.n_degenerate, boundary_edges=int((cnt == 1).sum()),
                          non_manifold_edges=int((cnt > 2).sum()), closed=bool((cnt == 1).sum() == 0), manifold=bool((cnt == 2).all()), consistently_oriented=bool((dcnt == 1).all()),
                          area=float(self.area.sum()), largest_triangle_radius=self.rmax)
        self._pn = None

    # ---------------------------------------------------------------- closest point
    def closest_point(self, P, k0=4, chunk=20000):
        """Exact closest point for each query. Returns dict(distance, point, tri, bary, feature) ; feature: 0 face, 1 edge, 2 vertex."""
        P = np.atleast_2d(np.asarray(P, float)); n = len(P); D = np.empty(n); X = np.empty((n, 3)); TI = np.empty(n, np.int64); BA = np.empty((n, 3))
        for s in range(0, n, chunk):
            p = P[s:s + chunk]; k = min(k0, len(self.cen)); _, idx = self.tree.query(p, k=k); idx = idx.reshape(len(p), -1)
            qi = np.repeat(np.arange(len(p)), idx.shape[1]); ti = idx.ravel(); c, _ = _closest_on_triangles(p[qi], self.A[ti], self.B[ti], self.C[ti])
            ub = np.sqrt(((c - p[qi]) ** 2).sum(1)).reshape(len(p), -1).min(1)                     # a true upper bound on the distance
            lists = self.tree.query_ball_point(p, ub + self.rmax + 1e-12)                          # complete: a triangle at distance <= ub has its centroid within ub + rmax
            cnt = np.fromiter((len(l) for l in lists), np.int64, len(p)); qi = np.repeat(np.arange(len(p)), cnt); ti = np.fromiter((t for l in lists for t in l), np.int64, int(cnt.sum()))
            c, b = _closest_on_triangles(p[qi], self.A[ti], self.B[ti], self.C[ti]); d2 = ((c - p[qi]) ** 2).sum(1)
            order = np.lexsort((ti, d2, qi)); first = order[np.concatenate([[0], np.cumsum(cnt)[:-1]])]      # smallest distance, ties to the lowest triangle id: deterministic
            D[s:s + chunk] = np.sqrt(d2[first]); X[s:s + chunk] = c[first]; TI[s:s + chunk] = ti[first]; BA[s:s + chunk] = b[first]
        feat = (BA < _TOL_BARY).sum(1)                                                             # zeros among barycentrics: 0 -> face, 1 -> edge, 2 -> vertex
        return dict(distance=D, point=X, tri=TI, bary=BA, feature=feat)

    def closest_point_bruteforce(self, P, chunk=64):
        """Reference implementation: every query against every triangle. For tests only."""
        P = np.atleast_2d(np.asarray(P, float)); D = np.empty(len(P)); TI = np.empty(len(P), np.int64); m = len(self.T)
        for s in range(0, len(P), chunk):
            p = P[s:s + chunk]; qi = np.repeat(np.arange(len(p)), m); ti = np.tile(np.arange(m), len(p)); c, _ = _closest_on_triangles(p[qi], self.A[ti], self.B[ti], self.C[ti])
            d = np.sqrt(((c - p[qi]) ** 2).sum(1)).reshape(len(p), m); TI[s:s + chunk] = d.argmin(1); D[s:s + chunk] = d.min(1)
        return dict(distance=D, tri=TI)

    # ---------------------------------------------------------------- winding number
    def winding_number(self, P, chunk_pairs=6_000_000):
        """Generalised winding number, exact summation (van Oosterom & Strackee solid angle). 1 inside a closed outward-oriented surface, 0 outside."""
        P = np.atleast_2d(np.asarray(P, float)); m = len(self.T); out = np.empty(len(P)); step = max(1, chunk_pairs // max(m, 1))
        for s in range(0, len(P), step):
            p = P[s:s + step][:, None, :]; a = self.A[None] - p; b = self.B[None] - p; c = self.C[None] - p
            la = np.linalg.norm(a, axis=2); lb = np.linalg.norm(b, axis=2); lc = np.linalg.norm(c, axis=2)
            num = np.einsum("qti,qti->qt", a, np.cross(b, c)); den = la * lb * lc + (a * b).sum(2) * lc + (b * c).sum(2) * la + (c * a).sum(2) * lb
            out[s:s + step] = np.arctan2(num, den).sum(1) / (2.0 * np.pi)
        return out

    # ---------------------------------------------------------------- pseudonormals
    def _pseudonormals(self):
        if self._pn is None:
            T = self.T; VN = np.zeros_like(self.V)
            for i in range(3):                                                                    # angle-weighted vertex pseudonormal (Baerentzen & Aanaes 2005)
                p0, p1, p2 = self.V[T[:, i]], self.V[T[:, (i + 1) % 3]], self.V[T[:, (i + 2) % 3]]; e1 = p1 - p0; e2 = p2 - p0
                cosang = (e1 * e2).sum(1) / (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)); np.add.at(VN, T[:, i], np.arccos(np.clip(cosang, -1, 1))[:, None] * self.FN)
            emap = {}
            for i in range(3):
                for t, (u, v) in enumerate(np.sort(T[:, [i, (i + 1) % 3]], 1)): emap.setdefault((int(u), int(v)), []).append(t)
            self._pn = (VN, emap)
        return self._pn

    # ---------------------------------------------------------------- signed distance
    def signed_distance(self, P, rule="winding", allow_open=False, cp=None):
        """Returns dict(signed, inside, + closest_point fields [+ winding]). Positive = outside."""
        P = np.atleast_2d(np.asarray(P, float)); cp = cp or self.closest_point(P); d = P - cp["point"]
        if rule == "face_normal": inside = (d * self.FN[cp["tri"]]).sum(1) < 0
        elif rule == "pseudonormal":
            VN, emap = self._pseudonormals(); N = self.FN[cp["tri"]].copy(); T = self.T[cp["tri"]]; z = cp["bary"] < _TOL_BARY
            for i in np.nonzero(cp["feature"] == 2)[0]: N[i] = VN[T[i][~z[i]][0]]
            for i in np.nonzero(cp["feature"] == 1)[0]:
                u, v = sorted(int(x) for x in T[i][~z[i]]); N[i] = self.FN[emap[(u, v)]].sum(0)
            inside = (d * N).sum(1) < 0
        elif rule == "winding":
            if not self.facts["closed"] and not allow_open: raise ValueError(f"surface {self.name!r} is open ({self.facts['boundary_edges']} boundary edges): winding rule refused (pass allow_open=True to accept a fractional field)")
            w = self.winding_number(P); inside = w > 0.5; cp = dict(cp, winding=w)
        else: raise ValueError(f"unknown rule {rule!r}")
        return dict(cp, signed=np.where(inside, -cp["distance"], cp["distance"]), inside=inside, rule=rule)

    # ---------------------------------------------------------------- sign ambiguity of the legacy rule
    def sign_ambiguity(self, P, tol, cp=None):
        """Per query: n_tied (faces within tol of the nearest distance), inside_any / inside_all under the face-normal rule, ambiguous = inside_any and not inside_all.
        tol is a length in the mesh unit; the caller chooses it from the coordinate precision of the consumer being audited (float32 at 1 m: about 1e-7 m)."""
        P = np.atleast_2d(np.asarray(P, float)); cp = cp or self.closest_point(P); n = len(P); lists = self.tree.query_ball_point(P, cp["distance"] + tol + self.rmax)
        nt = np.zeros(n, np.int64); anyi = np.zeros(n, bool); alli = np.zeros(n, bool)
        for k in range(n):
            c = np.asarray(lists[k], np.int64); q, _ = _closest_on_triangles(np.repeat(P[k][None], len(c), 0), self.A[c], self.B[c], self.C[c]); tie = np.linalg.norm(q - P[k], axis=1) <= cp["distance"][k] + tol
            sg = ((P[k] - q[tie]) * self.FN[c[tie]]).sum(1) < 0; nt[k] = int(tie.sum()); anyi[k] = sg.any(); alli[k] = sg.all()
        return dict(n_tied=nt, inside_any=anyi, inside_all=alli, ambiguous=anyi & ~alli)

    # ---------------------------------------------------------------- self contact
    def self_contact(self, eps):
        """Boolean per vertex: within eps of a triangle that is not in its 2-ring and whose normal opposes the vertex normal (surface pressed against itself or passing through itself)."""
        T = self.T; nv = len(self.V); inc = [[] for _ in range(nv)]
        for t, tri in enumerate(T):
            for v in tri: inc[int(v)].append(t)
        VN = np.zeros_like(self.V); np.add.at(VN, T.ravel(), np.repeat(self.FN * self.area[:, None], 3, 0)); ln = np.linalg.norm(VN, axis=1); VN = VN / np.where(ln > 0, ln, 1)[:, None]
        used = np.unique(T); lists = self.tree.query_ball_point(self.V[used], eps + self.rmax); flag = np.zeros(nv, bool); dist = np.full(nv, np.inf)
        for v, cand in zip(used, lists):
            if not cand: continue
            ring = set(int(x) for t in inc[v] for x in T[t]); cand = np.array([t for t in cand if ring.isdisjoint(T[t].tolist())], np.int64)
            if not len(cand): continue
            cand = cand[self.FN[cand] @ VN[v] < 0]
            if not len(cand): continue
            c, _ = _closest_on_triangles(np.repeat(self.V[v][None], len(cand), 0), self.A[cand], self.B[cand], self.C[cand]); d = np.sqrt(((c - self.V[v]) ** 2).sum(1)).min()
            dist[v] = d; flag[v] = d <= eps
        return dict(flag=flag, distance=dist, n_flagged=int(flag.sum()), eps=eps)


def clearance_report(signed, labels=None, limits=()):
    """Summary of signed clearance values (same unit as the mesh). `limits`: thresholds to count below (callers pass owner-approved limits; none are built in)."""
    signed = np.asarray(signed, float)
    def one(x):
        if not len(x): return dict(n=0)
        r = dict(n=int(len(x)), inside=int((x < 0).sum()), min=float(x.min()), p1=float(np.percentile(x, 1)), p5=float(np.percentile(x, 5)), p50=float(np.percentile(x, 50)), max=float(x.max()))
        for L in limits: r[f"below_{L:g}"] = int((x < L).sum())
        return r
    out = dict(all=one(signed))
    if labels is not None:
        labels = np.asarray(labels); out["by_label"] = {str(l): one(signed[labels == l]) for l in np.unique(labels)}
    return out
