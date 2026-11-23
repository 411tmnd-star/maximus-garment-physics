"""swing_twist.py  PROTOTYPE rotation utilities for pose construction (numpy only; no Blender). Not production code.

CONTRACT
  Rotations       3x3 proper orthogonal matrices acting on column vectors, any right-handed frame. Angles in radians unless a name ends in _deg.
  swing_twist(D, axis)   D = Swing @ Twist, Twist about the unit `axis` (applied first), Swing about an axis perpendicular to `axis`.
                         Returns (twist angle in (-pi, pi], swing angle in [0, pi], Swing, Twist, ill_conditioned).
                         Ill-conditioned when the swing is within ~6 deg of pi (the twist is then undefined); flagged, never hidden.
  min_arc(a, b)          the rotation of smallest angle taking direction a to direction b. Undefined for a = -b: raises.
  two_bone_ik(S, T, a, b, pole)   elbow position for a two-segment limb (lengths a, b) from shoulder S to target T, bent towards `pole`.
                         The reach is clamped to [|a-b| + margin, a + b - margin]; returns (E, W, clamped) with W the wrist actually reached.
  frame_from_two(e_loc, h_loc, e_w, h_w)   the rotation R with R e_loc = e_w and R h_loc = h_w (after h is made perpendicular to e on both sides)."""
import numpy as np


def unit(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    if n < 1e-300: raise ValueError("zero vector")
    return v / n


def quat(R):
    """(w, x, y, z), w >= 0, from a rotation matrix (Shepperd)."""
    t = np.trace(R)
    if t > 0: s = np.sqrt(t + 1) * 2; q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3; s = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2; q = np.empty(4)
        q[0] = (R[k, j] - R[j, k]) / s; q[1 + i] = 0.25 * s; q[1 + j] = (R[j, i] + R[i, j]) / s; q[1 + k] = (R[k, i] + R[i, k]) / s
    q = q / np.linalg.norm(q); return q if q[0] >= 0 else -q


def axis_angle(axis, ang):
    k = unit(axis); K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]); return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def angle_of(R): return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def swing_twist(D, axis=(0.0, 1.0, 0.0)):
    a = unit(axis); w, x, y, z = quat(D); p = float(np.dot([x, y, z], a)); n = np.hypot(w, p); ill = bool(n < 0.05)
    twist = 2 * np.arctan2(p, w); twist = (twist + np.pi) % (2 * np.pi) - np.pi
    Tw = axis_angle(a, twist) if abs(twist) > 0 else np.eye(3); Sw = D @ Tw.T
    return float(twist), float(2 * np.arccos(min(1.0, n))), Sw, Tw, ill


def min_arc(a, b):
    a = unit(a); b = unit(b); c = float(a @ b); ax = np.cross(a, b); s = np.linalg.norm(ax)
    if s < 1e-12:
        if c > 0: return np.eye(3)
        raise ValueError("min_arc undefined for opposite directions")
    return axis_angle(ax / s, np.arctan2(s, c))


def two_bone_ik(S, T, a, b, pole, margin=1e-3):
    S = np.asarray(S, float); d = np.asarray(T, float) - S; L0 = np.linalg.norm(d); L = min(max(L0, abs(a - b) + margin), a + b - margin); dn = d / L0
    p = np.asarray(pole, float); p = p - dn * (p @ dn); p = p / max(np.linalg.norm(p), 1e-9); x = (a * a - b * b + L * L) / (2 * L); h = np.sqrt(max(a * a - x * x, 0.0))
    return S + dn * x + p * h, S + dn * L, bool(abs(L - L0) > 1e-12)


def frame_from_two(e_loc, h_loc, e_w, h_w):
    def basis(e, h):
        e = unit(e); h = unit(np.asarray(h, float) - (np.asarray(h, float) @ e) * e); return np.stack([h, e, np.cross(h, e)], 1)
    return basis(e_w, h_w) @ basis(e_loc, h_loc).T
