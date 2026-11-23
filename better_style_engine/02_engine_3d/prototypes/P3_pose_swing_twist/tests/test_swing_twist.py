"""Tests for swing_twist.py.  python -I -m pytest -q -p no:cacheprovider tests"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); import swing_twist as ST
rng = np.random.default_rng(11)
def rrot(): return ST.axis_angle(rng.normal(size=3), rng.uniform(-np.pi, np.pi))


def test_decomposition_recomposes_and_axes():
    for _ in range(500):
        D = rrot(); ax = ST.unit(rng.normal(size=3)); tw, sw, S, T, ill = ST.swing_twist(D, ax)
        assert np.abs(S @ T - D).max() < 1e-12 and np.abs(T @ ax - ax).max() < 1e-12
        if sw > 1e-6 and not ill:
            w, x, y, z = ST.quat(S); assert abs(np.dot([x, y, z], ax)) < 1e-9          # swing axis is perpendicular to the twist axis


def test_known_values():
    y = np.array([0, 1.0, 0]); D = ST.axis_angle([1, 0, 0], 0.7) @ ST.axis_angle(y, -1.1); tw, sw, *_ = ST.swing_twist(D, y); assert abs(tw + 1.1) < 1e-12 and abs(sw - 0.7) < 1e-12


def test_min_arc_has_no_twist_about_source_direction():
    for _ in range(300):
        a = ST.unit(rng.normal(size=3)); b = ST.unit(rng.normal(size=3)); R = ST.min_arc(a, b); assert np.abs(R @ a - b).max() < 1e-12
        tw, sw, *_ = ST.swing_twist(R, a); assert abs(tw) < 1e-9 and abs(sw - np.arccos(np.clip(a @ b, -1, 1))) < 1e-9


def test_composed_min_arcs_create_twist():
    """The mechanism REQ-04 measured: two successive minimal-arc aims are not a minimal arc; relative to the start they carry twist."""
    a = np.array([0, 1.0, 0]); b = np.array([1.0, 0, 0]); c = np.array([0, 0, 1.0]); R = ST.min_arc(b, c) @ ST.min_arc(a, b); assert np.abs(R @ a - c).max() < 1e-12
    tw, _, *_ = ST.swing_twist(R, a); assert abs(abs(tw) - np.pi / 2) < 1e-9
    tw2, _, *_ = ST.swing_twist(ST.min_arc(a, c), a); assert abs(tw2) < 1e-12


def test_min_arc_opposite_raises():
    try: ST.min_arc([0, 1, 0], [0, -1, 0]); assert False
    except ValueError: pass


def test_ik_lengths_and_pole():
    for _ in range(300):
        S = rng.normal(size=3); a, b = rng.uniform(0.2, 0.4, 2); T = S + ST.unit(rng.normal(size=3)) * rng.uniform(abs(a - b) + 0.01, a + b - 0.01); pole = rng.normal(size=3)
        E, W, cl = ST.two_bone_ik(S, T, a, b, pole); assert not cl and abs(np.linalg.norm(E - S) - a) < 1e-12 and abs(np.linalg.norm(W - E) - b) < 1e-9 and np.abs(W - T).max() < 1e-12
        dn = ST.unit(T - S); pp = pole - dn * (pole @ dn); assert (E - S) @ pp >= -1e-12


def test_ik_clamps_out_of_reach():
    E, W, cl = ST.two_bone_ik(np.zeros(3), [5, 0, 0], 0.3, 0.25, [0, 0, 1]); assert cl and abs(np.linalg.norm(W) - (0.55 - 1e-3)) < 1e-12


def test_frame_from_two():
    for _ in range(200):
        R = rrot(); e = ST.unit(rng.normal(size=3)); h = rng.normal(size=3); Rr = ST.frame_from_two(e, h, R @ e, R @ h); assert np.abs(Rr - R).max() < 1e-9 and abs(np.linalg.det(Rr) - 1) < 1e-12
