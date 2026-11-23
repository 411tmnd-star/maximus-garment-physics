"""stones_core.py - shared machinery for the garment stone layouts.
Applies the Stage 3 findings:
  * tracks of one band are stations of the SAME surface sections (shared parameter), never independently smoothed curves,
    so their mutual separation is fixed by construction (RS-1 formulation A, RS-2 separation budget);
  * clearance is certified by the exact convex-hull distance (P1 convex_gap, ROUND stone model = the stricter of the two shapes),
    not by the sphere metric; nothing is "pruned to fit" inside a band;
  * every stone is seated by a ring ray-cast tilt fit and re-checked on a dense footprint grid.
Units: metres, Blender world (Z up). Stone record: P (base centre), N (outward), T (tangent), R (girdle radius), zone, cls."""
import sys, math, json
import numpy as np
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from scipy.spatial import cKDTree

MM = 1e-3
SS = {'SS6': 1.00 * MM, 'SS10': 1.40 * MM, 'SS16': 2.00 * MM, 'SS20': 2.35 * MM, 'SS30': 3.20 * MM}        # girdle radius
G_MIN = 0.05 * MM            # minimum exact gap (engine value; not an Owner-approved threshold)
GLUE = 0.010 * MM
H_CAST = 2.5 * MM
CHATON = dict(girdle_radius=0.5, rim_height=0.06, table_radius=0.29, total_height=0.30, sides=8, table_phase_deg=22.5)


def unit(v): return v / max(np.linalg.norm(v), 1e-30)


class Mesh:
    """Triangulated object with vertex normals, a 3D BVH and optional UV-space BVHs."""

    def __init__(self, name):
        o = bpy.data.objects[name]; me = o.data; self.name = name; self.me = me
        me.calc_loop_triangles()
        nv = len(me.vertices); V = np.zeros(nv * 3); me.vertices.foreach_get('co', V); self.V = V.reshape(-1, 3)
        nt = len(me.loop_triangles); T = np.zeros(nt * 3, np.int64); me.loop_triangles.foreach_get('vertices', T); self.T = T.reshape(-1, 3)
        L = np.zeros(nt * 3, np.int64); me.loop_triangles.foreach_get('loops', L); self.TL = L.reshape(-1, 3)
        P = self.V[self.T]; fn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]); self.area2 = np.linalg.norm(fn, axis=1)
        vn = np.zeros_like(self.V)
        for k in range(3): np.add.at(vn, self.T[:, k], fn)
        self.VN = vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-30)
        self.FN = fn / np.maximum(self.area2[:, None], 1e-30); self.FC = P.mean(1)
        self.bvh = BVHTree.FromPolygons(self.V.tolist(), self.T.tolist(), all_triangles=True)
        self.uvb = {}

    def outer_mask(self, ref_bvh):
        """triangle faces away from the reference body (mannequin): the visible outer shell"""
        out = np.zeros(len(self.T), bool)
        for i, c in enumerate(self.FC):
            loc, _, _, _ = ref_bvh.find_nearest(Vector(c)); out[i] = self.FN[i] @ (c - np.array(loc)) > 0
        self.outer = out; return out

    def loop_uv(self, layer):
        uv = np.zeros(len(self.me.loops) * 2); self.me.uv_layers[layer].data.foreach_get('uv', uv); return uv.reshape(-1, 2)

    def build_uv(self, key, uv_tri, mask):
        """uv_tri (nt,3,2). UV BVH over triangles passing mask and non-degenerate in UV."""
        J = (uv_tri[:, 1, 0] - uv_tri[:, 0, 0]) * (uv_tri[:, 2, 1] - uv_tri[:, 0, 1]) - (uv_tri[:, 2, 0] - uv_tri[:, 0, 0]) * (uv_tri[:, 1, 1] - uv_tri[:, 0, 1])
        ok = mask & (np.abs(J) > 1e-14); ids = np.nonzero(ok)[0]
        pts = np.concatenate([uv_tri[ids], np.zeros((len(ids), 3, 1))], 2).reshape(-1, 3)
        self.uvb[key] = (BVHTree.FromPolygons(pts.tolist(), np.arange(len(pts)).reshape(-1, 3).tolist(), all_triangles=True), ids, uv_tri)
        return len(ids)

    def uv_to_surface(self, key, uv):
        """-> (P0, N, dP/du, dP/dv, tri) or None"""
        bvh, ids, UVT = self.uvb[key]
        loc, _, idx, _ = bvh.ray_cast(Vector((uv[0], uv[1], 1.0)), Vector((0, 0, -1.0)), 2.0)
        if idx is None: return None
        t = int(ids[idx]); U = UVT[t]
        Juv = np.array([[U[1, 0] - U[0, 0], U[2, 0] - U[0, 0]], [U[1, 1] - U[0, 1], U[2, 1] - U[0, 1]]])
        l12 = np.linalg.solve(Juv, np.array([uv[0] - U[0, 0], uv[1] - U[0, 1]])); b = np.array([1 - l12[0] - l12[1], l12[0], l12[1]])
        P = self.V[self.T[t]]; n = unit(b @ self.VN[self.T[t]])
        if n @ self.FN[t] < 0: n = -n
        du1, dv1, du2, dv2 = U[1, 0] - U[0, 0], U[1, 1] - U[0, 1], U[2, 0] - U[0, 0], U[2, 1] - U[0, 1]; det = du1 * dv2 - du2 * dv1; e1, e2 = P[1] - P[0], P[2] - P[0]
        return b @ P, n, (dv2 * e1 - dv1 * e2) / det, (-du2 * e1 + du1 * e2) / det, t


def print_uv_tri(mesh, layer='UVMap_Fabric'):
    """UVMap_Print as render_sku.add_print_uv builds it: faces whose UV frame is mirrored relative to the normal get u -> -u."""
    uv = mesh.loop_uv(layer)[mesh.TL].copy(); P = mesh.V[mesh.T]
    e1, e2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]; d1, d2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]; det = np.where(np.abs(det) < 1e-14, 1e-14, det)
    Tg = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / det[:, None]; Bt = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / det[:, None]
    mir = np.einsum('ij,ij->i', np.cross(Tg, Bt), mesh.FN) < 0
    uv[mir, :, 0] *= -1.0; return uv, mir


class Seater:
    """Ring ray-cast tilt fit + lift + dense footprint check against every garment surface."""

    def __init__(self, bvhs): self.bvhs = bvhs

    def cast(self, q, n):
        best = None; bl = None
        for bvh in self.bvhs:
            loc, _, _, d = bvh.ray_cast(Vector(q + H_CAST * n), Vector(-n), 2 * H_CAST)
            if loc is not None and (best is None or d < best): best = d; bl = np.array(loc)
        return bl

    def seat(self, P0, N0, t_raw, R, max_lift=0.6 * MM, max_tilt_deg=40.0, occl=0.8 * MM):
        """-> dict(P, N, T) or (None, reason)"""
        N0 = unit(N0); T = t_raw - (t_raw @ N0) * N0
        if np.linalg.norm(T) < 1e-9: T = np.cross(N0, [0, 0, 1.0])
        T = unit(T); B = np.cross(N0, T)
        c = self.cast(P0, N0)
        if c is None: return None, 'no_surface'
        if (c - P0) @ N0 > occl: return None, 'occluded'
        P0 = c; ang = 2 * math.pi * np.arange(8) / 8; el = []; miss = 0
        for a in ang:
            x = self.cast(P0 + R * (math.cos(a) * T + math.sin(a) * B), N0)
            if x is None: miss += 1; el.append(0.0)
            else: el.append((x - P0) @ N0)
        if miss > 1: return None, 'edge'
        el = np.array(el); a_ = 2.0 / (8 * R) * np.sum(el * np.cos(ang)); b_ = 2.0 / (8 * R) * np.sum(el * np.sin(ang))
        if math.degrees(math.atan(math.hypot(a_, b_))) > max_tilt_deg: return None, 'tilt'
        Nf = unit(N0 - a_ * T - b_ * B); Tf = unit(T - (T @ Nf) * Nf); Bf = np.cross(Nf, Tf)
        # dense footprint (centre + 3 rings x 12) in the final frame: lift so that nothing is above the base plane
        top = 0.0
        for r in (0.0, 0.4, 0.75, 1.0):
            for k in range(1 if r == 0 else 12):
                th = 2 * math.pi * k / 12; x = self.cast(P0 + r * R * (math.cos(th) * Tf + math.sin(th) * Bf), Nf)
                if x is not None: top = max(top, (x - P0) @ Nf)
        if top > max_lift: return None, 'fold'
        return dict(P=P0 + (top + GLUE) * Nf, N=Nf, T=Tf), None


# ------------------------------------------------------------------ exact clearance (P1 prototype, round stone model)
def _load_gap():
    import importlib.util, pathlib
    p = pathlib.Path(__file__).resolve().parents[3] / 'prototypes' / 'P1_bstset_exact_hull' / 'convex_gap.py'
    sp = importlib.util.spec_from_file_location('convex_gap', p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def verify_exact(P, N, R, prio, g_min=G_MIN, log=print):
    """All neighbouring pairs: sphere-style lower bound first, exact GJK (round stone) where it is not decisive.
    Stones are removed only where the exact gap < g_min; the lower-priority (higher prio value) stone goes. Returns keep mask + report."""
    CG = _load_gap(); n = len(P); keep = np.ones(n, bool); kd = cKDTree(P)
    pairs = kd.query_pairs(2 * R.max() + 0.4 * MM, output_type='ndarray')
    d = np.linalg.norm(P[pairs[:, 0]] - P[pairs[:, 1]], axis=1) - R[pairs[:, 0]] - R[pairs[:, 1]]
    cosn = np.einsum('ij,ij->i', N[pairs[:, 0]], N[pairs[:, 1]])
    sure = (d > 0.12 * MM) & (cosn > 0.94)              # coplanar-ish and clear by more than the measured sphere-metric error bound
    cand = pairs[~sure]; exact = np.full(len(cand), np.inf)
    sup = {}
    def S(i):
        if i not in sup: sup[i] = CG.round_support(CHATON, P[i], N[i], 2 * R[i])
        return sup[i]
    for k, (i, j) in enumerate(cand):
        up, lo, it, hit = CG.gjk_distance(S(int(i)), S(int(j))); exact[k] = 0.0 if hit else lo
    bad = cand[exact < g_min]; order = np.argsort(exact[exact < g_min])
    removed = 0
    for k in order:
        i, j = bad[k]
        if keep[i] and keep[j]:
            keep[i if prio[i] > prio[j] else j] = False; removed += 1
    ok_exact = exact[(exact >= g_min)]
    rep = dict(stones=int(n), neighbour_pairs=int(len(pairs)), pairs_decided_by_bound=int(sure.sum()), pairs_exact=int(len(cand)),
               pairs_below_min=int((exact < g_min).sum()), pairs_overlapping=int((exact <= 0).sum()), removed=int(removed),
               min_exact_gap_kept_mm=float(ok_exact.min() / MM) if len(ok_exact) else None, model='round (CirclesSupport), g_min %.3f mm' % (g_min / MM))
    return keep, rep


# ------------------------------------------------------------------ packing along curves
def resample(Pl, step, closed=False):
    Pl = np.asarray(Pl, float)
    if closed: Pl = np.vstack([Pl, Pl[:1]])
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Pl, axis=0), axis=1))]); L = d[-1]
    if L < step: return Pl[:1], L
    s = np.arange(0, L, step) if closed else np.linspace(0, L, max(2, int(round(L / step)) + 1))
    return np.stack([np.interp(s, d, Pl[:, k]) for k in range(Pl.shape[1])], 1), L


def even_positions(L, pitch, closed, phase=0.0):
    """arc positions with an integer count. Closed: pitch stretched (never below `pitch`) so the loop closes exactly.
    Open: run centred; phase 0.5 gives the brick offset against a phase-0 neighbour row."""
    n = int(math.floor(L / pitch))
    if n < 1: return np.array([])
    if closed: p = L / n; return (phase * p + p * np.arange(n)) % L
    off = 0.5 * (L - n * pitch)
    return off + pitch * (np.arange(n) + 0.5) if phase else off + pitch * np.arange(n + 1)


class Grid2:
    """incremental minimum-distance acceptance in a plane (mm)"""

    def __init__(self, cell): self.c = cell; self.g = {}

    def ok(self, p, r, gap):
        cx, cy = int(p[0] // self.c), int(p[1] // self.c)
        for i in range(cx - 1, cx + 2):
            for j in range(cy - 1, cy + 2):
                for q, rq in self.g.get((i, j), ()):
                    if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < (r + rq + gap) ** 2: return False
        return True

    def add(self, p, r): self.g.setdefault((int(p[0] // self.c), int(p[1] // self.c)), []).append(((p[0], p[1]), r))


def write_outputs(prefix, stones, report):
    """npz (Blender frame), f32 stride 7 [px,py,pz,nx,ny,nz,scale] in glTF axes (x, z, -y) as the BSTSET buffer, and the report."""
    P = np.array([s['P'] for s in stones]); N = np.array([s['N'] for s in stones]); T = np.array([s['T'] for s in stones])
    R = np.array([s['R'] for s in stones]); zone = np.array([s['zone'] for s in stones]); cls = np.array([s.get('cls', 0) for s in stones])
    np.savez(prefix + '_stones.npz', P=P, N=N, T=T, R=R, zone=zone, cls=cls)
    g = lambda A: np.stack([A[:, 0], A[:, 2], -A[:, 1]], 1)
    np.concatenate([g(P), g(N), (2 * R)[:, None]], 1).astype('<f4').tofile(prefix + '_stones.f32')
    json.dump(report, open(prefix + '_stones.report.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
