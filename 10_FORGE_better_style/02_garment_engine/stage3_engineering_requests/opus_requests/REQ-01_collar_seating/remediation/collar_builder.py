"""collar_builder.py  Fold-over polo / rugby collar (GF-010, GF-011, GF-012, GF-014): COLLAR STAND + FOLDED COLLAR LEAF with points and a front opening.

Why this exists: simulating a wide flat collar ring (the first BS-POLO-001 master) gives a flat closed O-ring with a wavy free edge, because a free
cloth edge with low bending stiffness flutters and a closed ring has no front opening.  Real polo collars are a rib-knit band folded at a roll line:
the inner half (the stand) is sewn to the neckline and stands up; the outer half (the leaf) rolls over the top and lies down over the stand and
onto the shirt, ending in two points at the placket.

Construction, in the bake stage (after the drape, before polish):
  1. The SIMULATED closed rib ring is the stand (its sewn row follows the real draped neckline).  `cut_stand_front` deletes the stand faces inside
     the front opening (the placket gap), so the collar is open at centre front.
  2. `foldover_leaf` builds the leaf as a separate shell: for each column (an angle around the neck, outside the front gap) a cross-section path
       roll  : half circle of radius r over the stand's top edge (the roll line / fold crease),
       drop  : down the outside of the stand, 2r off it,
       flare : from the stand base outward/down over the shirt, length `extra(angle)`,
     with `extra`, the flare angle and a sideways spread all blended toward the front ends, which turns the two end columns into collar points.
  3. `lift_over_shell` pushes the drop/flare rows to `over_shirt_gap_m` outside the shirt surface (BVH of the body faces) and smooths along rows, so
     the leaf rests ON the shirt with no intersection, and its outer edge is a smooth curve (measured by `rim_waviness`).

Pure numpy except `lift_over_shell` (mathutils BVHTree).  All lengths in metres.
"""
import math
import numpy as np


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0); return x * x * (3 - 2 * x)


def ring_frame(stand_rows, cf_dir=(0.0, -1.0, 0.0)):
    """stand_rows: (rows, n, 3), row 0 = sewn neckline row, last row = free top edge, columns in ring order.
    Returns centre, axis (pointing from the sewn row to the top), orthonormal f (centre-front) and g = axis x f, and per-column angle theta
    (0 at centre front, increasing toward g)."""
    B = stand_rows[0]; T = stand_rows[-1]; ctr = B.mean(0)
    _, _, vt = np.linalg.svd(B - ctr); ax = vt[2]
    if np.dot((T - B).mean(0), ax) < 0: ax = -ax
    f = np.asarray(cf_dir, float); f = f - f.dot(ax) * ax; f /= np.linalg.norm(f); g = np.cross(ax, f)
    r = B - ctr; theta = np.arctan2(r @ g, r @ f)
    return ctr, ax, f, g, theta


def cut_stand_front(theta_of_face, gap_half_angle):
    """Boolean mask of stand faces to delete: those whose centroid lies inside the front opening."""
    return np.abs(theta_of_face) < gap_half_angle


def gap_half_angle(stand_rows, front_gap_m):
    ctr, ax, f, g, _ = ring_frame(stand_rows)
    r = stand_rows[0] - ctr; r = r - np.outer(r @ ax, ax); rho = float(np.linalg.norm(r, axis=1).mean())
    return math.asin(min(0.95, 0.5 * front_gap_m / rho)), rho


def _periodic_interp(theta, values, phi):
    """values (n, 3) sampled at angles theta (any order); evaluate at phi (m,) with 2*pi periodicity."""
    o = np.argsort(np.mod(theta, 2 * math.pi)); th = np.mod(theta, 2 * math.pi)[o]; V = values[o]
    th = np.r_[th[-1] - 2 * math.pi, th, th[0] + 2 * math.pi]; V = np.vstack([V[-1:], V, V[:1]])
    p = np.mod(phi, 2 * math.pi)
    return np.column_stack([np.interp(p, th, V[:, d]) for d in range(3)])


def foldover_leaf(stand_rows, cfg):
    """Build the collar leaf grid.  Returns dict(P=(rows, cols, 3), phi=(cols,), arc=(cols,), ell=(rows, cols), kind=(rows,) in {'roll','drop','flare'},
    gap_half=float, rho=float).  cfg keys (archetype spec construction.foldover_collar):
      leaf_extra_back_m, leaf_extra_front_m, roll_radius_m, front_gap_m, point_spread_deg, flare_deg_back, flare_deg_front, columns, rows."""
    ctr, ax, f, g, theta = ring_frame(stand_rows)
    gh, rho = gap_half_angle(stand_rows, cfg['front_gap_m'])
    nc = int(cfg['columns'])

    def fair_ring(R, n=40):                                                       # stand rows are closed rings in column order: periodic 1-2-1
        for _ in range(n): R = 0.25 * np.roll(R, 1, 0) + 0.5 * R + 0.25 * np.roll(R, -1, 0)
        return R
    # Columns are sampled by ARC LENGTH along the faired ring, in ring order (sorting by angle is wrong: the draped neckline is not monotone in angle,
    # and an angle sort interleaves columns, which made 7 cm steps in the first version).  Start at centre front and walk the ring.
    def fair_keep_radius(R):                                                       # 40 passes of 1-2-1 on a closed ring SHRINK it (the stand sank
        S = fair_ring(R); c2 = R[:, :2].mean(0)                                      # 28,164 verts into the neck); keep the shape's smoothness but
        r0 = np.linalg.norm(R[:, :2] - c2, axis=1); r_s = r0.copy()                  # restore each point's distance from the neck axis (lightly smoothed)
        for _ in range(6): r_s = 0.25 * np.roll(r_s, 1) + 0.5 * r_s + 0.25 * np.roll(r_s, -1)
        d = S[:, :2] - c2; d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9); S[:, :2] = c2 + d * r_s[:, None]; return S
    Bf = fair_keep_radius(stand_rows[0]); Tf = fair_keep_radius(stand_rows[-1])
    r_ = Bf - ctr; th = np.arctan2(r_ @ g, r_ @ f); j0 = int(np.argmin(np.abs(th))); n = len(th)
    order = [(j0 + k) % n for k in range(n + 1)]
    thu = np.unwrap(th[order]); thu -= thu[0]
    if thu[-1] < 0: order = [(j0 - k) % n for k in range(n + 1)]; thu = np.unwrap(th[order]); thu -= thu[0]
    seg = np.linalg.norm(np.diff(Bf[order], axis=0), axis=1); sarc = np.r_[0, np.cumsum(seg)]
    s0 = float(np.interp(gh, np.maximum.accumulate(thu), sarc)); s1 = float(np.interp(2 * math.pi - gh, np.maximum.accumulate(thu), sarc))
    sc = np.linspace(s0, s1, nc)
    T = np.column_stack([np.interp(sc, sarc, Tf[order][:, d]) for d in range(3)]); B = np.column_stack([np.interp(sc, sarc, Bf[order][:, d]) for d in range(3)])
    phi = np.interp(sc, sarc, np.maximum.accumulate(thu))
    Tt = np.gradient(T, axis=0); tang = Tt / np.maximum(np.linalg.norm(Tt, axis=1, keepdims=True), 1e-12)
    sgn_end = 1.0                                                                  # tangent points from the first collar end toward the second
    o = B - ctr; o[:, 2] = 0.0; o /= np.linalg.norm(o, axis=1, keepdims=True)    # horizontal outward: the best-fit axis of a saddle-shaped neckline
                                                                                     # is tilted, and projecting on it turned 'outward' inward at the sides
    s = T - B; h = np.linalg.norm(s, axis=1); sh = s / h[:, None]
    d_end = np.minimum(sc - s0, s1 - sc) / max(rho, 1e-6); fe = _smoothstep(1.0 - d_end / math.radians(70.0))   # 1 at the collar points, 0 from ~70 deg of arc inward
    extra = cfg['leaf_extra_back_m'] + (cfg['leaf_extra_front_m'] - cfg['leaf_extra_back_m']) * fe
    alpha = np.radians(cfg['flare_deg_back'] + (cfg['flare_deg_front'] - cfg['flare_deg_back']) * fe)
    sgn = np.where(sc < 0.5 * (s0 + s1), 1.0, -1.0)                                # away from centre front along the neckline
    fe_tip = _smoothstep(1.0 - d_end / math.radians(cfg.get('spread_ramp_deg', 25.0)))   # the sideways spread belongs at the tips: a wide ramp shears
    spread = math.tan(math.radians(cfg['point_spread_deg'])) * fe_tip                     # the grid where the neckline turns (collapsed quads)
    D = -sh * np.cos(alpha)[:, None] + o * np.sin(alpha)[:, None] + tang * (sgn * spread)[:, None]
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    r = cfg['roll_radius_m']; n_roll, n_drop = 6, 4; n_flare = max(3, int(cfg['rows']) - n_roll - n_drop)
    rows, ell, kind = [], [], []
    n_stand = int(cfg.get('stand_rows', 5))
    for i in range(n_stand - 1):                                                   # stand: from the sewn neckline up to the roll line (same columns, so the
        lam = i / (n_stand - 1); rows.append(B + (T - B) * lam); ell.append(np.full(nc, -(1 - lam) * float(np.mean(h)))); kind.append('stand')   # opening is shared)
    for i in range(n_roll):                                                        # roll: psi 0..pi over the top edge
        psi = math.pi * i / (n_roll - 1)
        rows.append(T + o * (r * (1 - math.cos(psi))) + sh * (r * math.sin(psi))); ell.append(np.full(nc, r * psi)); kind.append('roll')
    dd = float(cfg.get('drop_depth_frac', 1.0))                                    # the leaf covers the stand but its fold-down stops short of the
    for i in range(1, n_drop + 1):                                                 # neckline seam (dd < 1): at full depth its lower edge lands on the
        lam = dd * i / n_drop; rows.append(T + 2 * r * o - sh * (lam * h)[:, None]); ell.append(math.pi * r + lam * h); kind.append('drop')   # shirt at the seam
    Q0 = T + 2 * r * o - sh * (dd * h)[:, None]
    for i in range(1, n_flare + 1):                                                # flare: over the shirt; collar points at the ends
        mu = i / n_flare; rows.append(Q0 + D * (mu * extra)[:, None]); ell.append(math.pi * r + dd * h + mu * extra); kind.append('flare')
    P = np.stack(rows); ell = np.stack([np.broadcast_to(e, (nc,)) for e in ell])
    arc = sc - s0
    return dict(P=P, phi=phi, arc=arc, ell=ell, kind=np.array(kind), gap_half=gh, rho=rho, extra=extra, ctr=ctr, axis=ax, frontness=fe)


def _signed(shell_bvh, p):
    """Signed distance of p to the shell along the face normal, or None when the nearest point lies on the shell's BOUNDARY (the neck opening),
    where the face normal says nothing about inside/outside (pushing along it throws vertices sideways)."""
    from mathutils import Vector
    hit = shell_bvh.find_nearest(Vector(p))
    if hit[0] is None: return None
    d = Vector(p) - hit[0]; sd = d.dot(hit[1])
    if d.length > 1e-6 and abs(sd) < 0.9 * d.length: return None
    return sd, np.array(hit[1])


def _fair(row, passes):
    for _ in range(passes):
        sm = row.copy(); sm[1:-1] = 0.25 * row[:-2] + 0.5 * row[1:-1] + 0.25 * row[2:]; row = sm
    return row


def lift_over_shell(P, kind, shell_bvh, gap, iters=6, envelope_cols=9, sigma_cols=3.0, **_):
    """Lift the drop/flare rows to >= gap outside the shirt (shell_bvh = BVHTree of the shirt body faces, normals outward).

    A rib collar is stiff: it BRIDGES the shirt's small wrinkles.  So the lift is not per vertex (that copies every wrinkle into the collar edge, the
    first version's 6 mm scallops) but a smooth upper envelope per row: the needed lift along each column is max-filtered over `envelope_cols` and
    Gaussian-smoothed over `sigma_cols`, and applied along a column-smoothed normal; repeated `iters` times.  The roll above the first drop row
    gets that row's lift ramped in, so the fold never stretches into one long edge.  Returns (P, n_lift_samples, min_signed_distance_m)."""
    from scipy.ndimage import maximum_filter1d, gaussian_filter1d
    P = P.copy(); P0 = P.copy(); movable = [k for k, kd in enumerate(kind) if kd in ('drop', 'flare')]; rolls = [k for k, kd in enumerate(kind) if kd == 'roll']
    nc = P.shape[1]; pushed = 0
    flare = [k for k, kd in enumerate(kind) if kd == 'flare']
    for _ in range(3):                                                              # gravity: the flare rests ON the shirt (a lift alone never lets a
        for i, k in enumerate(flare):                                              # hovering leaf come down; from behind one then sees under it)
            w = (i + 1) / len(flare)
            for c in range(nc):
                r_ = _signed(shell_bvh, P[k, c])
                if r_ is not None and r_[0] > gap: P[k, c] = P[k, c] - r_[1] * (r_[0] - gap) * w
        for k in flare: P[k] = _fair(P[k], 2)
    for _ in range(iters):
        any_need = False
        for k in movable:
            need = np.zeros(nc); nrm = np.zeros((nc, 3))
            for c in range(nc):
                r_ = _signed(shell_bvh, P[k, c])
                if r_ is not None:
                    nrm[c] = r_[1]
                    if r_[0] < gap: need[c] = gap - r_[0]; pushed += 1
            if not need.any(): continue
            any_need = True
            env = gaussian_filter1d(maximum_filter1d(need, size=envelope_cols, mode='nearest'), sigma_cols, mode='nearest')
            env = np.maximum(env, need)                                                     # never below the requirement
            nrm = gaussian_filter1d(nrm, sigma_cols, axis=0, mode='nearest'); nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
            P[k] += nrm * env[:, None]
        for k in movable[:-1]: P[k] = 0.5 * P[k] + 0.25 * (P[k - 1] + P[k + 1])       # keep each column a smooth curve
        if not any_need: break
    if rolls and movable:
        disp = P[movable[0]] - P0[movable[0]]
        for i, k in enumerate(rolls[1:], start=1): P[k] = P0[k] + disp * (i / len(rolls))
    P[-1] = _fair(P[-1], 4)
    P = relax_collapsed(P)
    for _ in range(4):                                                             # guarantee: per-vertex push for whatever the envelope left inside
        for k in movable:
            for c in range(nc):
                r_ = _signed(shell_bvh, P[k, c])
                if r_ is not None and r_[0] < gap: P[k, c] = P[k, c] + r_[1] * (gap - r_[0]); pushed += 1
    P = respace_columns(P, P0, movable)                                            # pushes can pull neighbouring rows together: restore the spacing
    for k in movable:                                                              # last one-sided check (re-spacing moves points along the column only)
        for c in range(nc):
            r_ = _signed(shell_bvh, P[k, c])
            if r_ is not None and r_[0] < gap: P[k, c] = P[k, c] + r_[1] * (gap - r_[0]); pushed += 1
    sdmin = 1e9
    for k in movable:
        for c in range(nc):
            r_ = _signed(shell_bvh, P[k, c])
            if r_ is not None: sdmin = min(sdmin, r_[0])
    return P, pushed, float(sdmin)


def respace_rows(P, P0, rows):
    """Re-distribute the columns of each lifted row along that row's own polyline at the relative arc positions they had before lifting."""
    P = P.copy()
    for k in rows:
        Q = P[k]; s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]; s0 = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P0[k], axis=0), axis=1))]
        if s[-1] > 0 and s0[-1] > 0: P[k] = np.column_stack([np.interp(s0 / s0[-1] * s[-1], s, Q[:, d]) for d in range(3)])
    return P


def respace_columns(P, P0, rows):
    """For each column, re-distribute the lifted drop/flare rows along that column's own polyline (from the last fixed row to the rim) at the
    relative arc positions they had before lifting.  Shape kept, spacing restored: no rows can collapse onto each other."""
    P = P.copy(); r0 = rows[0] - 1; idx = [r0] + list(rows)
    for c in range(P.shape[1]):
        Q = P[idx, c]; s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
        Q0 = P0[idx, c]; s0 = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q0, axis=0), axis=1))]
        if s[-1] <= 0 or s0[-1] <= 0: continue
        t = s0 / s0[-1] * s[-1]
        P[idx[1:], c] = np.column_stack([np.interp(t[1:], s, Q[:, d]) for d in range(3)])
    return P


def quad_areas(P):
    a = P[:-1, :-1]; b = P[:-1, 1:]; c = P[1:, 1:]; d = P[1:, :-1]
    return 0.5 * np.linalg.norm(np.cross(c - a, d - b), axis=2)


def relax_collapsed(P, frac=0.2, iters=12):
    """Where a large lift folds rows of the roll together, quads collapse (the gate counts the resulting sub-1e-10 m2 triangles after Subsurf).
    Laplacian-relax only the vertices of quads below `frac` x the median quad area, holding the first row (on the stand) and the rim."""
    P = P.copy()
    for _ in range(iters):
        A = quad_areas(P); bad = A < frac * np.median(A)
        if not bad.any(): break
        mask = np.zeros(P.shape[:2], bool); r, c = np.where(bad)
        for dr in (0, 1):
            for dc in (0, 1): mask[r + dr, c + dc] = True
        mask[0] = False; mask[-1] = False
        avg = P.copy(); avg[1:-1, 1:-1] = 0.25 * (P[:-2, 1:-1] + P[2:, 1:-1] + P[1:-1, :-2] + P[1:-1, 2:])
        avg[1:-1, 0] = 0.5 * (P[:-2, 0] + P[2:, 0]); avg[1:-1, -1] = 0.5 * (P[:-2, -1] + P[2:, -1])
        P[mask] = 0.5 * P[mask] + 0.5 * avg[mask]
    return P


def rim_waviness(rim, mask=None, step_m=0.008):
    """Std-dev of the turning angle (rad) along an open polyline: a smooth collar edge scores low, a scalloped one high.
    Same metric as the aesthetic auditor's `collar_waviness_rad_std`.  `mask` (bool per rim point) restricts it to a section: the gate measures the
    band between the collar points (frontness == 0), because the points curve on purpose and would count as 'waviness'."""
    if mask is not None: rim = rim[np.asarray(mask, bool)]
    a = np.r_[0, np.cumsum(np.linalg.norm(np.diff(rim, axis=0), axis=1))]           # resample at a fixed 8 mm arc step: the statistic needs a scale, otherwise
    st = np.arange(0, a[-1], step_m)                                                 # sub-mm jitter on closely spaced columns dominates it
    rim = np.column_stack([np.interp(st, a, rim[:, k]) for k in range(3)])
    d = np.diff(rim, axis=0); d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    ang = np.arccos(np.clip(np.einsum('ij,ij->i', d[:-1], d[1:]), -1, 1))
    return float(ang.std())


def scallop_amplitude(rim, mask=None, window_m=0.06, step_m=0.004, closed=False):
    """Scallop / waviness amplitude of an edge (metres): the edge resampled at `step_m`, minus its own moving average over `window_m`; returns
    (max, rms) of that high-pass deviation.  A smooth edge -- however its overall shape curves -- scores ~0; a scalloped or fluttering edge scores
    its scallop height.  Low-pass = cubic Savitzky-Golay over `window_m` (a moving average would cut the corners of a curved edge and
    report curvature as scallops).  This is the GATED collar metric.  (The turning-angle std in `rim_waviness` also rises for a legitimately shaped collar that
    turns at the back corners -- the raw, unlifted leaf of a clean collar scores 0.126 rad -- so it is reported but not gated.)"""
    if mask is not None: rim = rim[np.asarray(mask, bool)]
    if closed: rim = np.vstack([rim, rim[:1]])
    a = np.r_[0, np.cumsum(np.linalg.norm(np.diff(rim, axis=0), axis=1))]; st = np.arange(0, a[-1], step_m)
    from scipy.signal import savgol_filter
    R = np.column_stack([np.interp(st, a, rim[:, k]) for k in range(3)]); w = max(5, int(round(window_m / step_m)) | 1)
    L = savgol_filter(R, w, 3, axis=0, mode='wrap' if closed else 'interp')   # cubic local fit: follows real curvature, removes wiggles shorter than the window
    dev = np.linalg.norm(R - L, axis=1)
    return float(dev.max()), float(np.sqrt((dev ** 2).mean()))


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# REQ-01 neckline seating.  The stand base used to be the simulated rib row captured BEFORE the shell was refined / solidified / smoothed, and it was
# never re-bound to the final shell, so it floated 8-18 mm above the neckline rim at centre back / centre front.  The two functions below bind it:
#   seat_stand_rows      control level: rigid per-column translation of the stand rows so row 0 lands SEAT_CLEARANCE_M above the shell's neckline rim
#   enforce_base_clearance  final mesh: every vertex of the stand's bottom ring is forced into SEAT_BAND_M of the shell surface (outside it)
# ---------------------------------------------------------------------------------------------------------------------------------------------------
SEAT_CLEARANCE_M = 0.0005                 # target distance stand base -> shell surface
SEAT_BAND_M = (0.0001, 0.0010)            # accepted band: 0.1 mm (no intersection) .. 1.0 mm (seated)
SEAT_RING_COL_DEG = 2.0                   # bottom ring = per 2-degree column, vertices within SEAT_RING_BAND_M of that column's lowest z
SEAT_RING_BAND_M = 0.001


def neckline_rim_bvh(Vg, tris_all, body_bvh, base_pts, radius=0.03, edge_cos=0.3):
    """BVHTree of the shell's neckline RIM faces near the stand base.  Rim = faces whose normal is ~tangent to the body: |n . (centroid - nearest body
    point)| / |centroid - nearest body point| <= edge_cos (the solidify rim band / neckline edge; the outer and inner sheets are excluded).  Only the faces
    within `radius` of a stand-base point are classified (a few thousand, not the whole shirt)."""
    from scipy.spatial import cKDTree
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    Va = np.asarray(Vg, float); Tb = np.asarray(tris_all, int)
    C = Va[Tb].mean(1); cand = np.unique(np.concatenate([x for x in cKDTree(C).query_ball_point(np.asarray(base_pts), radius) if len(x)] or [np.zeros(0, int)])).astype(int)
    N = np.cross(Va[Tb[cand, 1]] - Va[Tb[cand, 0]], Va[Tb[cand, 2]] - Va[Tb[cand, 0]]); N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-18)
    keep = []
    for k, i in enumerate(cand):
        hb = body_bvh.find_nearest(Vector(C[i]))
        if hb[0] is None: continue
        d = C[i] - np.array(hb[0]); L = np.linalg.norm(d)
        if L > 1e-9 and abs(float(N[k] @ d)) / L <= edge_cos: keep.append(i)
    if not keep: return None, 0
    return BVHTree.FromPolygons([tuple(v) for v in Va], [tuple(Tb[i]) for i in keep]), len(keep)


def _periodic_smooth(D, passes):
    for _ in range(passes): D = 0.25 * np.roll(D, 1, 0) + 0.5 * D + 0.25 * np.roll(D, -1, 0)
    return D


def seat_stand_rows(stand_rows, rim_bvh, clearance=SEAT_CLEARANCE_M, smooth_passes=3):
    """Rigidly translate every column of the stand rows so that row 0 (the sewn base) lies `clearance` above the nearest point of the shell's
    neckline rim, on the side of the rim that faces the stand top.  Column height, lean and ring order are preserved (only a translation per column,
    lightly smoothed round the ring so neighbouring columns cannot shear).  Returns (rows, per-column displacement (n, 3), report)."""
    from mathutils import Vector
    R = stand_rows.copy(); B = R[0]; T = R[-1]; n = B.shape[0]; up = (T - B).mean(0); up /= np.linalg.norm(up)
    D = np.zeros((n, 3)); ok = np.zeros(n, bool)
    for j in range(n):
        hit = rim_bvh.find_nearest(Vector(B[j]))
        if hit[0] is None: continue
        q = np.array(hit[0]); nrm = np.array(hit[1])
        if nrm @ up < 0: nrm = -nrm
        D[j] = q + nrm * clearance - B[j]; ok[j] = True
    if not ok.all():                                                               # columns the rim BVH could not answer: interpolate round the ring
        j_ok = np.flatnonzero(ok); jj = np.arange(n)
        for d in range(3): D[:, d] = np.interp(jj, j_ok, D[j_ok, d], period=n)
    Ds = _periodic_smooth(D, smooth_passes)
    R = R + Ds[None, :, :]
    mag = 1000 * np.linalg.norm(Ds, axis=1)
    return R, Ds, dict(seat_columns=n, seat_columns_without_rim_hit=int((~ok).sum()), seat_shift_mm_min=round(float(mag.min()), 2), seat_shift_mm_median=round(float(np.median(mag)), 2), seat_shift_mm_max=round(float(mag.max()), 2))


def stand_bottom_ring(P, w_stand, w_leaf, col_deg=SEAT_RING_COL_DEG, band_m=SEAT_RING_BAND_M):
    """Indices of the stand's bottom ring: pure-stand vertices (stand weight > 0, leaf weight == 0), binned in `col_deg` columns around the neck
    (theta = atan2(x - cx, -(y - cy)), 0 = centre front), keeping in each column the vertices within `band_m` of that column's lowest z."""
    idx = np.flatnonzero((w_stand > 0) & (w_leaf == 0))
    if idx.size == 0: return idx, np.zeros(0)
    Q = P[idx]; c = Q[:, :2].mean(0); th = np.degrees(np.arctan2(Q[:, 0] - c[0], -(Q[:, 1] - c[1]))); col = np.floor((th + 180.0) / col_deg).astype(int)
    keep = np.zeros(len(idx), bool)
    for k in np.unique(col):
        m = np.flatnonzero(col == k); keep[m[Q[m, 2] <= Q[m, 2].min() + band_m]] = True
    return idx[keep], th[keep]


def enforce_base_clearance(P, w_stand, w_leaf, shell_bvh, clearance=SEAT_CLEARANCE_M, band=SEAT_BAND_M, falloff_m=0.004, max_iter=8):
    """Final-mesh seating.  P (n, 3) final collar vertex positions (edited in place).  Every bottom-ring vertex whose distance to the shell surface
    is outside `band`, or which lies INSIDE the shell (negative side of the nearest face normal), is moved to exactly `clearance` outside the nearest
    shell point (along the line shell point -> vertex, or along the face normal if it was inside).  Pure-stand vertices within `falloff_m` above the
    ring follow with a smoothstep falloff so the base does not kink.  Repeats until the ring (recomputed each pass) is entirely inside the band.
    Returns a report dict."""
    from mathutils import Vector
    from scipy.spatial import cKDTree
    moved_total = 0; it = 0; last_bad = None
    for it in range(1, max_iter + 1):
        ring, _ = stand_bottom_ring(P, w_stand, w_leaf)
        if ring.size == 0: break
        delta = np.zeros((ring.size, 3)); bad = 0
        for k, i in enumerate(ring):
            p = P[i]; hit = shell_bvh.find_nearest(Vector(p))
            if hit[0] is None: continue
            q = np.array(hit[0]); nrm = np.array(hit[1]); d = p - q; L = float(np.linalg.norm(d)); inside = float(d @ nrm) < 0.0
            if (not inside) and band[0] <= L <= band[1]: continue
            u = nrm if (inside or L < 1e-9) else d / L
            delta[k] = (q + u * clearance) - p; bad += 1
        last_bad = bad
        if bad == 0: break
        ph = delta.any(1)
        P[ring] += delta; moved_total += int(ph.sum())
        rest = np.flatnonzero((w_stand > 0) & (w_leaf == 0)); rest = np.setdiff1d(rest, ring)
        if rest.size and ph.any():
            kd = cKDTree(P[ring[ph]] - delta[ph]); dist, nn = kd.query(P[rest], distance_upper_bound=falloff_m)
            m = np.isfinite(dist)
            if m.any():
                t = np.clip(dist[m] / falloff_m, 0, 1); w = 1 - t * t * (3 - 2 * t); P[rest[m]] += delta[ph][nn[m]] * w[:, None]
    return dict(base_ring_iterations=it, base_ring_vertices_moved=moved_total, base_ring_unresolved_last_pass=last_bad)


def leaf_faces(nr, nc):
    return [(r * nc + c, r * nc + c + 1, (r + 1) * nc + c + 1, (r + 1) * nc + c) for r in range(nr - 1) for c in range(nc - 1)]


def build_leaf_object(bpy, garment, stand_rows, cfg, work_dir=None, name='collar', body_bvh=None, seat_stand=True):
    """Build the leaf as its own object on the FINAL shirt surface (garment = the baked shell with modifiers applied): construct, lift over the
    shirt's body faces, give it Solidify (outward, so its underside keeps the lifted gap) and Subsurf, apply them, and measure the FINAL mesh:
    min gap to the shirt, degenerate triangles, rim scallop.  Returns (object, report)."""
    import bmesh
    from mathutils.bvhtree import BVHTree
    from mathutils import Vector
    me_g = garment.data; me_g.calc_loop_triangles(); Vg = [tuple(v.co) for v in me_g.vertices]
    body_tris = [tuple(t.vertices) for t in me_g.loop_triangles if me_g.polygons[t.polygon_index].material_index == 0]
    body_tris_all = list(body_tris)
    shell_all = BVHTree.FromPolygons(Vg, body_tris)                                # both layers of the solidified shirt (for the under-shirt ray test)
    if body_bvh is not None:                                                       # clearance is measured to the OUTER layer only: near the inner layer the
        Va = np.asarray(Vg); Tb = np.asarray(body_tris)                             # face normal points AT the body, so a push along it drove leaf vertices
        C = Va[Tb].mean(1); N = np.cross(Va[Tb[:, 1]] - Va[Tb[:, 0]], Va[Tb[:, 2]] - Va[Tb[:, 0]])   # through the shirt to the body floor (377 verts under
        keep = np.zeros(len(Tb), bool)                                              # the shirt in the failed build)
        for k in range(len(Tb)):
            hb = body_bvh.find_nearest(Vector(C[k]))
            keep[k] = hb[0] is None or float(np.dot(N[k], C[k] - np.array(hb[0]))) > 0.0
        body_tris = [body_tris[k] for k in np.flatnonzero(keep)]
    shell = BVHTree.FromPolygons(Vg, body_tris)
    if body_bvh is not None:                                                       # the stand hugs the neck: keep every stand row >= 3 mm off the body
        stand_rows = stand_rows.copy()
        for _ in range(4):
            for k in range(stand_rows.shape[0]):
                for j in range(stand_rows.shape[1]):
                    hit = body_bvh.find_nearest(Vector(stand_rows[k, j]))
                    if hit[0] is not None:
                        sd = (Vector(stand_rows[k, j]) - hit[0]).dot(hit[1])
                        if sd < 0.003: stand_rows[k, j] = stand_rows[k, j] + np.array(hit[1]) * (0.003 - sd)
    seat_rep = {}
    if seat_stand and body_bvh is not None:                                        # REQ-01: bind the stand base to the FINAL shell's neckline rim (the captured
        rim_bvh, n_rim = neckline_rim_bvh(Vg, body_tris_all, body_bvh, stand_rows[0])   # simulated row floats 8-18 mm above it at centre back / front)
        seat_rep['seat_rim_faces'] = n_rim
        if rim_bvh is not None: stand_rows, _, r_ = seat_stand_rows(stand_rows, rim_bvh); seat_rep.update(r_)
    LF = foldover_leaf(stand_rows, cfg); P0 = LF['P'].copy(); body_pushes = 0
    if body_bvh is not None:                                                       # body first (raw grid), then the smooth shirt lift + re-spacing: near the
        for _ in range(4):                                                         # shirt's neckline boundary the shirt-lift skips vertices, so the BODY is
            for k in range(P0.shape[0]):                                           # the surface that keeps them out
                for j in range(P0.shape[1]):
                    hit = body_bvh.find_nearest(Vector(P0[k, j]))
                    if hit[0] is not None:
                        sd = (Vector(P0[k, j]) - hit[0]).dot(hit[1])
                        if sd < 0.002: P0[k, j] = P0[k, j] + np.array(hit[1]) * (0.002 - sd); body_pushes += 1
    P, npush, sdmin = lift_over_shell(P0, LF['kind'], shell, cfg['over_shirt_gap_m'])
    if work_dir: np.savez(work_dir + '/collar_leaf_debug.npz', stand_rows=stand_rows, leaf_raw=LF['P'], leaf=P, phi=LF['phi'], frontness=LF['frontness'])
    nr, nc = P.shape[:2]; bm = bmesh.new(); vs = [bm.verts.new(tuple(P[i, j])) for i in range(nr) for j in range(nc)]
    uvl = bm.loops.layers.uv.new('UVMap_Fabric'); faces = []
    for (a, b, c, d) in leaf_faces(nr, nc):
        f = bm.faces.new((vs[a], vs[b], vs[c], vs[d])); f.smooth = True; faces.append(f)
    idx = {v: k for k, v in enumerate(vs)}
    for f in faces:
        for lp in f.loops:
            i, j = divmod(idx[lp.vert], nc); lp[uvl].uv = (1.14 + LF['arc'][j] / 0.30, 0.80 + LF['ell'][i, j] / 0.30)
    mid = faces[(nr - 2) * (nc - 1) + (nc - 1) // 2]; out = np.array(mid.calc_center_median()[:]) - LF['ctr']; out -= (out @ LF['axis']) * LF['axis']
    if np.dot(np.array(mid.normal[:]), out) < 0: bmesh.ops.reverse_faces(bm, faces=faces)           # normals face away from the neck
    me = bpy.data.meshes.new(name + '_Mesh'); bm.to_mesh(me); bm.free()
    me.materials.append(bpy.data.materials.get('Collar_Rib'))
    ob = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(ob)
    kinds = LF['kind']; gs = ob.vertex_groups.new(name='collar_stand'); gl = ob.vertex_groups.new(name='collar_leaf')
    gs.add([i * nc + j for i in range(nr) if kinds[i] == 'stand' for j in range(nc)], 1.0, 'REPLACE')
    gl.add([i * nc + j for i in range(nr) if kinds[i] != 'stand' for j in range(nc)], 1.0, 'REPLACE')
    so = ob.modifiers.new('Solidify', 'SOLIDIFY'); so.thickness = cfg.get('thickness_m', 0.002); so.offset = 0.0; so.use_even_offset = False   # centred: the folded sheet's 'outward' normal points INTO the neck on the stand side; so.use_rim = True
    ss = ob.modifiers.new('Subsurf', 'SUBSURF'); ss.levels = ss.render_levels = 2
    dg = bpy.context.evaluated_depsgraph_get(); me2 = bpy.data.meshes.new_from_object(ob.evaluated_get(dg)); old = ob.data
    ob.modifiers.clear(); ob.data = me2; bpy.data.meshes.remove(old); me2.name = name + '_Mesh'
    for p in me2.polygons: p.use_smooth = True
    gl_idx = ob.vertex_groups['collar_leaf'].index                                 # shirt clearance applies to the LEAF only: the stand's base is sewn
    is_leaf = np.array([any(g.group == gl_idx and g.weight > 0.5 for g in v.groups) for v in me2.vertices])   # onto the shirt's neckline by design
    final_pushes = 0                                                              # Subsurf pulls the tight fold in between control vertices: final
    for _ in range(3):                                                             # clearance pass on the final vertices, as the shell gets after smoothing
        for v in me2.vertices:
            if is_leaf[v.index]:
                hit = shell.find_nearest(v.co)
                if hit[0] is not None:
                    d = v.co - hit[0]
                    if not (d.length > 1e-6 and abs(d.dot(hit[1])) < 0.9 * d.length) and d.dot(hit[1]) < 0.0008:
                        v.co = v.co + hit[1] * (0.0008 - d.dot(hit[1])); final_pushes += 1
            if body_bvh is not None:
                hb = body_bvh.find_nearest(v.co)
                if hb[0] is not None and (v.co - hb[0]).dot(hb[1]) < 0.0005: v.co = v.co + hb[1] * (0.0005 - (v.co - hb[0]).dot(hb[1])); final_pushes += 1
    if seat_stand:                                                                 # REQ-01 final pass: Solidify + Subsurf moved the base edge; force the bottom ring
        gs_idx = ob.vertex_groups['collar_stand'].index                            # into 0.1-1.0 mm of the shell surface (outside it), whatever the earlier stages did
        Pf = np.array([v.co[:] for v in me2.vertices]); w_s = np.array([next((g.weight for g in v.groups if g.group == gs_idx), 0.0) for v in me2.vertices])
        w_l = np.array([next((g.weight for g in v.groups if g.group == gl_idx), 0.0) for v in me2.vertices])
        seat_rep.update(enforce_base_clearance(Pf, w_s, w_l, shell_all)); me2.vertices.foreach_set('co', Pf.ravel()); me2.update()
    Vf = np.array([v.co[:] for v in me2.vertices]); sd_f = []
    for p in Vf[is_leaf]:
        hit = shell.find_nearest(Vector(p))
        if hit[0] is None: continue
        d = Vector(p) - hit[0]
        if d.length > 1e-6 and abs(d.dot(hit[1])) < 0.9 * d.length: continue                  # nearest point on the shirt's boundary: no inside/outside
        sd_f.append(d.dot(hit[1]))
    under = 0                                                                      # independent check: leaf vertices with shirt ABOVE them (ray along the
    if body_bvh is not None:                                                         # body normal, outward, through both shirt layers) are under the shirt
        for p in Vf[is_leaf]:
            hb = body_bvh.find_nearest(Vector(p))
            if hb[0] is not None and shell_all.ray_cast(Vector(p) + hb[1] * 1e-5, hb[1], 0.05)[0] is not None: under += 1
    me2.calc_loop_triangles(); T = np.array([t.vertices[:] for t in me2.loop_triangles])
    ar = np.linalg.norm(np.cross(Vf[T[:, 1]] - Vf[T[:, 0]], Vf[T[:, 2]] - Vf[T[:, 0]]), axis=1) / 2
    rim = P[-1]; A = quad_areas(P[[k for k in range(nr) if kinds[k] != 'stand']])
    inside_body = 0
    if body_bvh is not None:                                                         # the stand sits against the neck: check it against the BODY too
        for p in Vf:
            hit = body_bvh.find_nearest(Vector(p))
            if hit[0] is not None and (Vector(p) - hit[0]).dot(hit[1]) < 0: inside_body += 1
    rep = dict(leaf_rows=nr, leaf_cols=nc, leaf_lift_pushes=npush, leaf_body_pushes=body_pushes, leaf_min_gap_over_shirt_mm=round(1000 * sdmin, 2), final_leaf_min_gap_over_shirt_mm=round(1000 * float(min(sd_f)), 2),
               final_leaf_verts=len(Vf), final_leaf_clearance_pushes=final_pushes, final_leaf_degenerate_triangles=int((ar < 1e-10).sum()), final_collar_verts_inside_body=inside_body, final_leaf_verts_under_shirt=under, leaf_min_quad_area_ratio=round(float(A.min() / np.median(A)), 3),
               rim_scallop_max_mm=round(1000 * scallop_amplitude(rim)[0], 2), rim_scallop_rms_mm=round(1000 * scallop_amplitude(rim)[1], 2),
               rim_turning_angle_std_rad=round(rim_waviness(rim, LF['frontness'] < 1e-3), 4), collar_points=[np.round(P[-1, 0], 4).tolist(), np.round(P[-1, -1], 4).tolist()])
    rep.update(seat_rep)
    return ob, rep
