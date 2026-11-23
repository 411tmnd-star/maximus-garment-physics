"""Tests for convex_gap.py: analytic fixtures and cross-method agreement.  python -I -m pytest -q -p no:cacheprovider tests"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); import convex_gap as G

PARAMS = dict(sides=8, girdle_radius=0.5, rim_height=0.06, table_radius=0.29, total_height=0.30, table_phase_deg=22.5)
CUBE = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
rng = np.random.default_rng(7)


def rot(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def test_cube_face_face():
    u, l, _, hit = G.gjk_distance(G.PolytopeSupport(CUBE), G.PolytopeSupport(CUBE + [1.25, 0.1, -0.2])); assert not hit and abs(u - 0.25) < 1e-12 and abs(l - 0.25) < 1e-12


def test_cube_edge_edge():
    u, l, _, hit = G.gjk_distance(G.PolytopeSupport(CUBE), G.PolytopeSupport(CUBE + [1.3, 1.4, 0.0])); assert not hit and abs(u - np.hypot(0.3, 0.4)) < 1e-12


def test_cube_vertex_vertex():
    u, l, _, hit = G.gjk_distance(G.PolytopeSupport(CUBE), G.PolytopeSupport(CUBE + [1.1, 1.2, 1.3])); assert abs(u - np.linalg.norm([0.1, 0.2, 0.3])) < 1e-12 and u - l < 1e-10


def test_overlap_and_touch():
    assert G.gjk_distance(G.PolytopeSupport(CUBE), G.PolytopeSupport(CUBE + [0.5, 0.5, 0.5]))[3]
    u, l, _, hit = G.gjk_distance(G.PolytopeSupport(CUBE), G.PolytopeSupport(CUBE + [1.0, 0.0, 0.0])); assert hit or u < 1e-12


def test_coincident():
    assert G.gjk_distance(G.PolytopeSupport(CUBE), G.PolytopeSupport(CUBE))[3]


def test_bracket_is_valid_and_tight_random_polytopes():
    V, T, E = G.chaton_unit(PARAMS)
    for _ in range(300):
        A = V @ rot(rng).T; B = V @ rot(rng).T + rng.normal(size=3) * 0.9
        u, l, it, hit = G.gjk_distance(G.PolytopeSupport(A), G.PolytopeSupport(B))
        if hit: continue
        fp = G.polytope_distance(A, B, T, E); assert l <= fp + 1e-12 and fp <= u + 1e-12 and u - l < 1e-9, (u, l, fp)


def test_rigid_invariance():
    V, T, E = G.chaton_unit(PARAMS); A = V; B = V @ rot(rng).T + [1.4, 0.2, 0.1]; R = rot(rng); t = rng.normal(size=3) * 5
    a = G.gjk_distance(G.PolytopeSupport(A), G.PolytopeSupport(B))[0]; b = G.gjk_distance(G.PolytopeSupport(A @ R.T + t), G.PolytopeSupport(B @ R.T + t))[0]; assert abs(a - b) < 1e-12


def test_chaton_unit_matches_parameters():
    V, T, E = G.chaton_unit(PARAMS); assert V.shape == (24, 3) and len(T) == 44 and len(E) == 66
    assert abs(np.linalg.norm(V[:16, [0, 2]], axis=1).max() - 0.5) < 1e-15 and abs(V[:, 1].max() - 0.30) < 1e-15
    vol = sum(np.dot(V[a], np.cross(V[b], V[c])) for a, b, c in T) / 6; assert vol > 0                      # outward orientation


def test_round_coplanar_discs_analytic():
    """Two round stones on one plane, parallel axes: the gap is centre distance minus the two girdle radii."""
    n = np.array([0, 0, 1.0]); d = 2.0; A = G.round_support(PARAMS, np.zeros(3), n, d); B = G.round_support(PARAMS, np.array([2.5, 0, 0]), n, d)
    u, l, _, hit = G.gjk_distance(A, B, rel_tol=1e-13); assert not hit and abs(u - 0.5) < 1e-9 and l <= 0.5 + 1e-12


def test_round_contains_octagon():
    """The round solid contains the inscribed octagonal hull, so its gap can never exceed the hull gap."""
    V, T, E = G.chaton_unit(PARAMS)
    for _ in range(200):
        Na, Nb = rng.normal(size=3), rng.normal(size=3); Ta, Tb = rng.normal(size=3), rng.normal(size=3); Pb = rng.normal(size=3) * 1.2
        hu = G.gjk_distance(G.PolytopeSupport(G.hull_world(V, np.zeros(3), Na, Ta, 1.0)), G.PolytopeSupport(G.hull_world(V, Pb, Nb, Tb, 1.0)))
        ro = G.gjk_distance(G.round_support(PARAMS, np.zeros(3), Na, 1.0), G.round_support(PARAMS, Pb, Nb, 1.0))
        if ro[3]: continue
        assert not hu[3] and ro[0] <= hu[0] + 1e-9


def test_round_tilted_matches_fine_polygon():
    """Round solid against a 720-gon inscribed approximation: agreement within the polygon's sagitta."""
    P2 = dict(PARAMS, sides=720, table_phase_deg=0.0); Vf, _, _ = G.chaton_unit(P2); sag = 0.5 * (1 - np.cos(np.pi / 720))
    for _ in range(40):
        Na, Nb = rng.normal(size=3), rng.normal(size=3); Ta, Tb = rng.normal(size=3), rng.normal(size=3); Pb = rng.normal(size=3); Pb = Pb / np.linalg.norm(Pb) * 1.6
        ro = G.gjk_distance(G.round_support(PARAMS, np.zeros(3), Na, 1.0), G.round_support(PARAMS, Pb, Nb, 1.0))
        po = G.gjk_distance(G.PolytopeSupport(G.hull_world(Vf, np.zeros(3), Na, Ta, 1.0)), G.PolytopeSupport(G.hull_world(Vf, Pb, Nb, Tb, 1.0)))
        if ro[3] or po[3]: continue
        assert -1e-9 <= po[0] - ro[0] <= 2 * sag + 1e-9, (po[0], ro[0])
