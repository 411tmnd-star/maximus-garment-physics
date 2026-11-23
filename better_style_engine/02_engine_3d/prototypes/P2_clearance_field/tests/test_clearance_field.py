"""Deterministic tests and independent reference checks for clearance_field (pytest)."""
import sys, numpy as np, pytest, trimesh
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clearance_field as CF
R = np.random.default_rng(20261110)

def sphere(sub=4, r=1.0, c=(0, 0, 0)):
    m = trimesh.creation.icosphere(subdivisions=sub, radius=r); return np.asarray(m.vertices) + np.array(c), np.asarray(m.faces)
def box(ext=(1.0, 0.6, 0.4)):
    m = trimesh.creation.box(extents=ext); return np.asarray(m.vertices), np.asarray(m.faces)
def torus(R0=1.0, r0=0.3, nu=96, nv=48):
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False); v = np.linspace(0, 2 * np.pi, nv, endpoint=False); U, Vv = np.meshgrid(u, v, indexing="ij")
    P = np.stack([(R0 + r0 * np.cos(Vv)) * np.cos(U), (R0 + r0 * np.cos(Vv)) * np.sin(U), r0 * np.sin(Vv)], -1).reshape(-1, 3); idx = lambda i, j: (i % nu) * nv + (j % nv)
    T = [t for i in range(nu) for j in range(nv) for t in ((idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)), (idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)))]
    return P, np.array(T)
def l_prism():                                   # concave L-shaped prism, outward oriented
    xy = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float); n = 6; V = np.vstack([np.c_[xy, np.zeros(n)], np.c_[xy, np.ones(n)]])
    cap = [(0, 1, 2), (0, 2, 3), (0, 3, 5), (3, 4, 5)]; T = [(a, c, b) for a, b, c in cap] + [(a + n, b + n, c + n) for a, b, c in cap]
    for i in range(n): j = (i + 1) % n; T += [(i, j, j + n), (i, j + n, i + n)]
    return V, np.array(T)
def in_l(P): x, y, z = P.T; return (z > 0) & (z < 1) & (x > 0) & (y > 0) & (((x < 2) & (y < 1)) | ((x < 1) & (y < 2)))
def ray_parity(V, T, P, dirs):
    """Independent inside test: odd number of ray-triangle crossings (Moller-Trumbore), majority over several directions."""
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]; votes = np.zeros(len(P))
    for d in dirs:
        e1, e2 = B - A, C - A; h = np.cross(d, e2); a = (e1 * h).sum(1); f = 1.0 / a; cnt = np.zeros(len(P), int)
        for i, p in enumerate(P):
            s = p - A; u = f * (s * h).sum(1); q = np.cross(s, e1); v = f * (q @ d); t = f * (e2 * q).sum(1); cnt[i] = int(((u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-12)).sum())
        votes += cnt % 2
    return votes > len(dirs) / 2
DIRS = [np.array(d) / np.linalg.norm(d) for d in ([0.31, 0.77, 0.55], [-0.62, 0.23, 0.75], [0.13, -0.91, 0.39])]

def test_facts_closed_and_open():
    assert CF.TriSurface(*box()).facts["closed"] and CF.TriSurface(*box()).facts["manifold"] and CF.TriSurface(*box()).facts["consistently_oriented"]
    V, T = sphere(2); s = CF.TriSurface(V, T[V[T].mean(1)[:, 2] > 0]); assert not s.facts["closed"] and s.facts["boundary_edges"] > 0
def test_accelerated_closest_point_equals_bruteforce():
    V, T = torus(nu=40, nv=20); s = CF.TriSurface(V, T); P = R.uniform(-1.6, 1.6, (400, 3)); a = s.closest_point(P); b = s.closest_point_bruteforce(P)
    assert np.abs(a["distance"] - b["distance"]).max() < 1e-12
def test_box_signed_distance_analytic():
    ext = np.array([1.0, 0.6, 0.4]); s = CF.TriSurface(*box(ext)); P = R.uniform(-1, 1, (3000, 3)); q = np.abs(P) - ext / 2
    ref = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
    for rule in ("winding", "pseudonormal"):
        assert np.abs(s.signed_distance(P, rule)["signed"] - ref).max() < 1e-12, rule
def test_sphere_error_bounded_by_sag():
    V, T = sphere(4); s = CF.TriSurface(V, T); P = R.normal(size=(2000, 3)); P *= R.uniform(0.2, 2.0, (2000, 1)) / np.linalg.norm(P, axis=1, keepdims=True)
    ref = np.linalg.norm(P, axis=1) - 1.0; got = s.signed_distance(P, "winding")["signed"]; sag = 1.0 - np.linalg.norm(V[T].mean(1), axis=1).min()
    assert (np.abs(got - ref) <= sag + 1e-12).all() and sag < 2e-3
def test_torus_inside_matches_analytic_and_ray_parity():
    V, T = torus(); s = CF.TriSurface(V, T); P = R.uniform(-1.5, 1.5, (1500, 3)); rho = np.sqrt((np.sqrt(P[:, 0] ** 2 + P[:, 1] ** 2) - 1.0) ** 2 + P[:, 2] ** 2); keep = np.abs(rho - 0.3) > 0.01
    P, rho = P[keep], rho[keep]; w = s.winding_number(P); assert ((w > 0.5) == (rho < 0.3)).all() and np.abs(w - np.rint(w)).max() < 1e-9
    assert (ray_parity(V, T, P[:300], DIRS) == (w[:300] > 0.5)).all()
def wedge(half_deg=12.0, L=1.0, h=1.0):             # triangular prism with a sharp (2*half_deg) edge along z at the origin, outward oriented
    t = np.radians(half_deg); xy = np.array([[0, 0], [L * np.cos(t), -L * np.sin(t)], [L * np.cos(t), L * np.sin(t)]]); V = np.vstack([np.c_[xy, np.zeros(3)], np.c_[xy, np.full(3, h)]])
    T = [(0, 2, 1), (3, 4, 5)] + [x for i in range(3) for j in [(i + 1) % 3] for x in ((i, j, j + 3), (i, j + 3, i + 3))]
    return V, np.array(T), t
def test_concave_L_all_rules_agree_with_analytic():
    """Right-angled concave solid: all three rules agree with the analytic answer (the legacy rule does NOT fail here; kept as a recorded negative result)."""
    V, T = l_prism(); s = CF.TriSurface(V, T); assert s.facts["closed"] and s.facts["consistently_oriented"]
    P = R.uniform(-0.5, 2.5, (6000, 3)) * [1, 1, 0.6] + [0, 0, 0.2]; truth = in_l(P); far = s.closest_point(P)["distance"] > 1e-6; P, truth = P[far], truth[far]
    for rule in ("winding", "pseudonormal", "face_normal"): assert (s.signed_distance(P, rule)["inside"] == truth).all(), rule
def test_sharp_edge_legacy_rule_fails_where_others_do_not():
    """Acute dihedral angle: for points whose closest feature is the sharp edge, the closest-triangle normal can point away from an outside point."""
    V, T, t = wedge(); s = CF.TriSurface(V, T); assert s.facts["closed"] and s.facts["consistently_oriented"]
    P = np.c_[R.uniform(-0.3, 0.3, 4000), R.uniform(-0.3, 0.3, 4000), R.uniform(0.2, 0.8, 4000)]
    truth = (P[:, 0] > 0) & (np.abs(P[:, 1]) < P[:, 0] * np.tan(t)) & (P[:, 0] < np.cos(t)); far = s.closest_point(P)["distance"] > 1e-6; P, truth = P[far], truth[far]
    win = s.signed_distance(P, "winding")["inside"]; pn = s.signed_distance(P, "pseudonormal")["inside"]; leg = s.signed_distance(P, "face_normal")["inside"]
    assert (win == truth).all() and (pn == truth).all()
    assert (leg != truth).sum() > 0 and (leg & ~truth).sum() > 0                 # legacy rule reports outside points as inside
def test_two_overlapping_spheres_winding_two_in_the_lens():
    Va, Ta = sphere(3); Vb, Tb = sphere(3, c=(1.2, 0, 0)); V = np.vstack([Va, Vb]); T = np.vstack([Ta, Tb + len(Va)]); s = CF.TriSurface(V, T)
    w = s.winding_number(np.array([[0.6, 0, 0], [0.0, 0, 0], [3.0, 0, 0]])); assert np.allclose(w, [2, 1, 0], atol=1e-9)
    assert s.signed_distance(np.array([[0.6, 0, 0]]), "winding")["inside"][0]
def test_open_surface_refused_by_winding_rule():
    V, T = sphere(2); s = CF.TriSurface(V, T[V[T].mean(1)[:, 2] > 0])
    with pytest.raises(ValueError): s.signed_distance(np.zeros((1, 3)), "winding")
    assert 0.0 < s.signed_distance(np.array([[0, 0, 0.2]]), "winding", allow_open=True)["winding"][0] < 1.0
def test_rigid_motion_invariance_and_determinism():
    V, T = torus(nu=48, nv=24); P = R.uniform(-1.5, 1.5, (500, 3)); a = CF.TriSurface(V, T).signed_distance(P, "winding")["signed"]
    Q, _ = np.linalg.qr(R.normal(size=(3, 3))); Q *= np.sign(np.linalg.det(Q)); t = np.array([0.3, -1.1, 2.2])
    b = CF.TriSurface(V @ Q.T + t, T).signed_distance(P @ Q.T + t, "winding")["signed"]; assert np.abs(a - b).max() < 1e-9
    assert (CF.TriSurface(V, T).signed_distance(P, "winding")["signed"] == a).all()
def test_weld_restores_closedness():
    V, T = box(); V2 = V[T].reshape(-1, 3); T2 = np.arange(len(V2)).reshape(-1, 3); assert not CF.TriSurface(V2, T2).facts["closed"]
    Vw, Tw, _ = CF.weld(V2, T2); assert CF.TriSurface(Vw, Tw).facts["closed"] and len(Vw) == 8
def test_self_contact_detected_only_where_surfaces_meet():
    Va, Ta = sphere(3); Vb, Tb = sphere(3, c=(2.02, 0, 0)); V = np.vstack([Va, Vb]); T = np.vstack([Ta, Tb + len(Va)]); sc = CF.TriSurface(V, T).self_contact(0.05)
    assert sc["n_flagged"] > 0 and np.abs(V[sc["flag"]][:, 0] - 1.01).max() < 0.35
    assert CF.TriSurface(Va, Ta).self_contact(0.05)["n_flagged"] == 0
def test_degenerate_triangles_dropped_and_feature_codes():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]], float); s = CF.TriSurface(V, np.array([[0, 1, 2], [0, 1, 3]])); assert s.facts["degenerate_triangles_dropped"] == 1
    cp = s.closest_point(np.array([[0.2, 0.2, 1.0], [0.5, -1.0, 0.0], [-1.0, -1.0, 0.0]])); assert list(cp["feature"]) == [0, 1, 2] and np.allclose(cp["distance"], [1, 1, np.sqrt(2)])
def test_clearance_report_counts():
    r = CF.clearance_report(np.array([-1.0, 0.5, 2.0, 3.0]), labels=np.array(["a", "a", "b", "b"]), limits=(1.0,)); assert r["all"]["inside"] == 1 and r["all"]["below_1"] == 2 and r["by_label"]["b"]["inside"] == 0


def test_sign_ambiguity_on_a_reflex_edge():
    """Two faces meeting at a concave (reflex) edge of a closed solid: a point inside the solid, nearest to that edge, lies in front of one face and behind the other.
    The face-normal rule then has two answers; the pseudonormal and the winding number have one. A point nearest to a face interior is never ambiguous."""
    import numpy as np, clearance_field as CF
    # L-shaped prism: outline in xy, extruded in z. The reflex edge is the vertical line x = 1, y = 1.
    out = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float); n = len(out); V = np.vstack([np.c_[out, np.zeros(n)], np.c_[out, np.ones(n)]])
    T = []
    for i in range(n): j = (i + 1) % n; T += [(i, j, n + j), (i, n + j, n + i)]
    for a, b, c in [(0, 1, 2), (0, 2, 3), (0, 3, 5), (3, 4, 5)]: T += [(c, b, a), (n + a, n + b, n + c)]
    S = CF.TriSurface(V, np.array(T), "L prism"); assert S.facts["closed"] and S.facts["consistently_oriented"]
    P = np.array([[1.3, 1.1, 0.5],        # outside, in the notch, nearest point on the face y = 1 (interior of a face)
                  [0.9, 0.8, 0.5],        # inside, nearest feature is the reflex edge (distance 0.2236), tied between the faces x = 1 and y = 1
                  [0.5, 0.5, 0.5]])       # deep inside
    a = S.sign_ambiguity(P, 1e-9); w = S.signed_distance(P, "winding")["inside"]; pn = S.signed_distance(P, "pseudonormal")["inside"]
    assert list(w) == [False, True, True] and list(pn) == [False, True, True]
    assert a["n_tied"][1] >= 2 and not a["ambiguous"][0] and not a["ambiguous"][2]
    # at the reflex edge both tied faces see the point behind them, so here the legacy rule happens to agree; the ambiguity arises at a CONVEX edge seen from outside its wedge:
    Q = np.array([[2.2, -0.1, 0.5]]); b = S.sign_ambiguity(Q, 1e-9); assert b["n_tied"][0] >= 2 and not b["ambiguous"][0] and not S.signed_distance(Q, "winding")["inside"][0]


def test_sign_ambiguity_sharp_wedge():
    """A thin wedge (dihedral angle 20 deg). A point outside, beyond the sharp edge but off its bisector, is nearest to the edge and lies in front of one face and BEHIND the plane of the other:
    the face-normal rule is ambiguous there; winding and pseudonormal both say outside."""
    import numpy as np, clearance_field as CF
    a = np.radians(10); V = np.array([[0, 0, 0], [0, 0, 1], [-1, np.tan(a), 0], [-1, np.tan(a), 1], [-1, -np.tan(a), 0], [-1, -np.tan(a), 1]], float)
    T = np.array([(0, 2, 3), (0, 3, 1), (0, 1, 5), (0, 5, 4), (2, 4, 5), (2, 5, 3), (0, 4, 2), (1, 3, 5)]); S = CF.TriSurface(V, T, "wedge")
    if not S.facts["consistently_oriented"] or S.signed_distance(np.array([[-0.5, 0, 0.5]]), "winding")["winding"][0] < 0: T = T[:, ::-1]; S = CF.TriSurface(V, T, "wedge")
    P = np.array([[0.2, 0.15, 0.5]]); r = S.sign_ambiguity(P, 1e-9)
    assert r["n_tied"][0] == 2 and r["ambiguous"][0] and not S.signed_distance(P, "winding")["inside"][0] and not S.signed_distance(P, "pseudonormal")["inside"][0]
