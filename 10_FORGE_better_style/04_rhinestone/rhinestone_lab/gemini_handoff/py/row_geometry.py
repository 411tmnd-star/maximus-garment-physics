"""row_geometry.py  geometry diagnostics for lateral stone rows on a curve (pure numpy, no Blender). All lengths are METRES (fields ending `_m`); angles degrees (`_deg`); curvature 1/m (`_per_m`).

Why this exists (checkpoint audit): stones of a row are placed at the SAME sample index as the centre curve, displaced sideways by `offset_m` along the surface (`PlacementEngine._offset_on_surface`).
On a bend, a row on the inner side advances less than the centre curve per sample (and one on the outer side more). With 0.4 mm of slack between SS10 stones (2.8 mm stones, 3.2 mm pitch) an inner row
overlaps its own neighbours. These functions measure that from the ACTUAL generated offset paths; they do not assume the planar formula.

Two evidence classes (kept apart in every report):
  early_warning  a geometric condition that can cause a physical collision (chord below the touching distance, fold, near-cusp, self-intersection of the offset path, ...).
                 It is evidence about the PATH. It does not prove two crystals intersect.
  confirmed      a physical intersection of two crystal hulls. That is decided ONLY by the collision engine (exact hull SAT) and verified by py/exact_hull.py; this module never decides it.

Geodesic vs 3D curvature: a circle around a cylinder has 3D curvature 1/R but zero geodesic curvature; an offset row along the axis keeps the circumference, so there is no compression.
The predicted shortening therefore uses the curvature component IN the surface (kappa_g = dT/ds . (N x T)), never the 3D curvature. `predicted_speed_ratio` is a diagnostic model that is compared with the
measurement (`model_error`); the measurement is what findings are based on."""
import math, sys
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chaton_geometry as CG

UNIT_INRADIUS = float(CG.P['girdle_radius'] * math.cos(math.pi / CG.P['sides']))      # of the unit-DIAMETER hull: inscribed circle radius of the girdle polygon (and of the base ring)


@dataclass
class RowThresholds:
    """Every value is explicit, configurable and carries its provenance. Nothing here is a production rule."""
    min_speed_ratio: float = 0.25          # offset-path speed / centre-path speed below which the path is a NEAR CUSP. PROVISIONAL heuristic; the physical criterion is the chord vs touching distance, evaluated separately
    self_intersection_eps_m: float = 2e-5  # two segments of one row path that are not neighbours ALONG THE PATH (arc separation > 4 eps) closer than this = self-intersection (20 um: mesh and ray-cast noise scale)
    dense_step_m: float = 5e-4             # target spacing of the dense offset path (0.5 mm = the step of `_offset_on_surface`; 1/5.6 of an SS10 diameter)
    resolution_rel_tol: float = 0.10       # the metric must change less than this when the dense step is halved, otherwise the path is reported UNDER-RESOLVED
    provenance: dict = field(default_factory=lambda: {
        'min_speed_ratio': 'provisional heuristic chosen for Stage 4A; a cusp has ratio 0, a straight row has 1; no production meaning',
        'self_intersection_eps_m': 'one fifth of Stage 3 EngineConfig.clearance_m (1e-4 m): a CROSSING, not an approach; approaches closer than the clearance are left to the collision engine; provisional',
        'dense_step_m': 'equals the 0.5 mm re-projection step of PlacementEngine._offset_on_surface; provisional',
        'resolution_rel_tol': 'provisional convergence tolerance for the halve-the-step check'})

    def to_dict(self): return {'values': {k: getattr(self, k) for k in ('min_speed_ratio', 'self_intersection_eps_m', 'dense_step_m', 'resolution_rel_tol')}, 'units': {'min_speed_ratio': '1', 'self_intersection_eps_m': 'm', 'dense_step_m': 'm', 'resolution_rel_tol': '1'}, 'provenance': dict(self.provenance), 'status': 'provisional'}


def touch_distance_m(d_a, d_b): return 0.5 * (d_a + d_b)                         # equal-size girdles touch at one diameter (corner to corner)
def guaranteed_overlap_distance_m(d_a, d_b): return 2.0 * UNIT_INRADIUS * 0.5 * (d_a + d_b)   # centres closer than this: the inscribed discs of coplanar stones overlap, so the hulls intersect whatever their yaw


def chords_m(P, closed=False):
    P = np.asarray(P, float); d = P[1:] - P[:-1]
    if closed and len(P) > 2: d = np.vstack([d, P[:1] - P[-1:]])
    return np.linalg.norm(d, axis=1)


def spacing_report(P, diameters_m, centre_chords_m=None, closed=False):
    """Spacing of consecutive stones of ONE row. P: (N,3) stone centres on the surface; diameters_m: length-N. Returns per-pair lists and the pairs below the touching / guaranteed-overlap distance.
    `ratio_to_centre` = row chord / centre-curve chord at the same sample pair (the compression the row suffers relative to the curve it is offset from)."""
    P = np.asarray(P, float); n = len(P); dm = np.asarray(diameters_m, float)
    if n < 2: return {'pairs': 0, 'chord_m': [], 'touch_m': [], 'guaranteed_overlap_m': [], 'ratio_to_touch': [], 'below_touch': [], 'below_guaranteed_overlap': [], 'ratio_to_centre': None, 'min_chord_m': None, 'min_ratio_to_touch': None}
    ch = chords_m(P, closed); idx = [(i, (i + 1) % n) for i in range(len(ch))]
    touch = np.array([touch_distance_m(dm[i], dm[j]) for i, j in idx]); guar = np.array([guaranteed_overlap_distance_m(dm[i], dm[j]) for i, j in idx])
    rep = {'pairs': len(ch), 'index_pairs': idx, 'chord_m': ch.tolist(), 'touch_m': touch.tolist(), 'guaranteed_overlap_m': guar.tolist(), 'ratio_to_touch': (ch / touch).tolist(),
           'below_touch': [k for k in range(len(ch)) if ch[k] < touch[k] - 1e-12], 'below_guaranteed_overlap': [k for k in range(len(ch)) if ch[k] < guar[k] - 1e-12], 'min_chord_m': float(ch.min()), 'min_ratio_to_touch': float((ch / touch).min())}
    if centre_chords_m is not None:
        cc = np.asarray(centre_chords_m, float); rep['ratio_to_centre'] = (ch / np.maximum(cc, 1e-12)).tolist(); rep['centre_chord_m'] = cc.tolist()
    else: rep['ratio_to_centre'] = None
    return rep


def paired_row_asymmetry(chords_inner_m, chords_outer_m):
    """For the two rows on opposite sides of the centre curve at the same sample pairs: mean (should equal the centre chord to first order) and half-difference (the bend signal). Reported, not assumed."""
    a = np.asarray(chords_inner_m, float); b = np.asarray(chords_outer_m, float); k = min(len(a), len(b)); a = a[:k]; b = b[:k]
    return {'mean_chord_m': (0.5 * (a + b)).tolist(), 'half_difference_m': (0.5 * (b - a)).tolist(), 'min_inner_m': float(np.minimum(a, b).min()) if k else None, 'max_outer_m': float(np.maximum(a, b).max()) if k else None}


# ------------------------------------------------------------------------------------------------ curvature and the prediction model (diagnostic only)
def surface_curvature(C, N):
    """Curvature of the centre path split into the part IN the surface and the part normal to it. C: (M,3) points ordered along the path; N: (M,3) unit outward surface normals.
    Returns kappa_g_per_m (signed: positive = the path bends to the LEFT, left = N x T, the side of positive row offsets), kappa_n_per_m, kappa_3d_per_m (magnitude) at interior points (end points copy the neighbours)."""
    C = np.asarray(C, float); N = np.asarray(N, float); m = len(C)
    if m < 3: z = np.zeros(m); return {'kappa_g_per_m': z, 'kappa_n_per_m': z.copy(), 'kappa_3d_per_m': z.copy()}
    seg = C[1:] - C[:-1]; ln = np.linalg.norm(seg, axis=1); T = seg / np.maximum(ln, 1e-15)[:, None]
    dT = T[1:] - T[:-1]; ds = 0.5 * (ln[1:] + ln[:-1]); k = dT / np.maximum(ds, 1e-15)[:, None]; Nm = N[1:-1]; Tm = 0.5 * (T[1:] + T[:-1]); Tm = Tm / np.maximum(np.linalg.norm(Tm, axis=1), 1e-15)[:, None]
    left = np.cross(Nm, Tm); kg = (k * left).sum(1); kn = (k * Nm).sum(1); k3 = np.linalg.norm(k, axis=1)
    pad = lambda a: np.concatenate([a[:1], a, a[-1:]]); return {'kappa_g_per_m': pad(kg), 'kappa_n_per_m': pad(kn), 'kappa_3d_per_m': pad(k3)}


def predicted_speed_ratio(kappa_g_per_m, offset_m):
    """Planar-in-the-surface model: a row at signed offset d (positive = left) advances 1 - d * kappa_g per unit of centre arc length. Valid for small d*kappa and gentle surface torsion; it is
    compared with the measured ratio (model_error) and never used to decide anything. 1 - d*kappa <= 0 means the offset exceeds the local radius of curvature (fold)."""
    return 1.0 - float(offset_m) * np.asarray(kappa_g_per_m, float)


# ------------------------------------------------------------------------------------------------ dense offset path analysis (folds, cusps, self-intersection)
def segment_distances(P, i, j):
    """Minimum distance between segment (P[i],P[i+1]) and each segment (P[j_k],P[j_k+1]); vectorised closest-point-of-two-segments."""
    p1 = P[i]; q1 = P[i + 1]; p2 = P[j]; q2 = P[j + 1]; d1 = q1 - p1; d2 = q2 - p2; r = p1 - p2
    a = float(d1 @ d1); e = (d2 * d2).sum(1); f = (d2 * r).sum(1); b = d2 @ d1; c = r @ d1
    denom = a * e - b * b; s = np.where(denom > 1e-24, np.clip((b * f - c * e) / np.where(denom > 1e-24, denom, 1.0), 0, 1), 0.0)
    t = np.where(e > 1e-24, (b * s + f) / np.where(e > 1e-24, e, 1.0), 0.0)
    s = np.where(t < 0, np.clip(-c / max(a, 1e-24), 0, 1), np.where(t > 1, np.clip((b - c) / max(a, 1e-24), 0, 1), s)); t = np.clip(t, 0, 1)
    return np.linalg.norm((p1 + s[:, None] * d1) - (p2 + t[:, None] * d2), axis=1)


def self_intersections(P, closed=False, eps_m=2e-5, skip=2):
    """Pairs of segments of the polyline that touch (distance < eps_m) although they are NOT neighbours along the path: index difference > skip AND arc-length separation > 4 eps_m (so the tiny
    steps of a contracted offset path are not mistaken for crossings). Returns [(i, j, distance_m)] with i < j. O(N^2) vectorised; fine for N up to a few thousand."""
    P = np.asarray(P, float)
    if closed and len(P) > 2: P = np.vstack([P, P[:1]])
    n = len(P) - 1; out = []
    if n < skip + 2: return out
    ln = np.linalg.norm(P[1:] - P[:-1], axis=1); cum = np.concatenate([[0.0], np.cumsum(ln)]); total = cum[-1]
    for i in range(n - 1):
        j = np.arange(i + skip + 1, n)
        if closed: j = j[np.minimum(j - i, n - (j - i)) > skip]            # cyclic adjacency
        sep = cum[j] - cum[i + 1]
        if closed: sep = np.minimum(sep, total - (cum[j + 1] - cum[i]))
        j = j[sep > 4 * eps_m]
        if len(j) == 0: continue
        dm = segment_distances(P, i, j)
        for jj, dd in zip(j[dm < eps_m], dm[dm < eps_m]): out.append((int(i), int(jj), float(dd)))
    return out


def dense_path_metrics(C, R, closed=False, thr=None):
    """C: dense centre path (M,3); R: the offset path generated from it, point for point (M,3) (entries may be NaN where the surface ended -> counted as `off_panel`).
    speed_ratio[k] = |R[k+1]-R[k]| / |C[k+1]-C[k]| : 1 on a straight row, 0 at a cusp; reversal[k] = the offset advances AGAINST the centre path (dot < 0).
    A fold is confirmed by either a reversal or a self-intersection of the offset path; a near cusp by a small speed ratio. None of these uses the stone pitch, so a narrow defect between two stone
    positions is still found (the resolution check below shows when the dense step is itself too coarse)."""
    thr = thr or RowThresholds(); C = np.asarray(C, float); R = np.asarray(R, float); valid = ~np.isnan(R).any(1)
    off = int((~valid).sum()); ok = valid[:-1] & valid[1:]
    dC = C[1:] - C[:-1]; dR = R[1:] - R[:-1]; lc = np.linalg.norm(dC, axis=1); lr = np.linalg.norm(dR, axis=1)
    ratio = np.where(ok, lr / np.maximum(lc, 1e-15), np.nan); dot = np.where(ok, (dC * dR).sum(1), np.nan)
    rev = ok & (dot < 0); near = ok & (ratio < thr.min_speed_ratio)
    Rv = R[valid]; si = self_intersections(Rv, closed, thr.self_intersection_eps_m) if len(Rv) > 4 else []
    fin = ratio[ok]; rev_len = float(lc[rev].sum()); cusp_len = float(lc[near].sum())
    return {'points': int(len(C)), 'off_panel_points': off, 'min_speed_ratio': float(np.nanmin(fin)) if len(fin) else None, 'max_speed_ratio': float(np.nanmax(fin)) if len(fin) else None,
            'reversal_segments': np.flatnonzero(rev).tolist(), 'near_cusp_segments': np.flatnonzero(near).tolist(), 'self_intersections': [{'segment_a': a, 'segment_b': b, 'distance_m': d} for a, b, d in si[:50]], 'self_intersection_count': len(si),
            'reversed_centre_arc_length_m': rev_len, 'near_cusp_centre_arc_length_m': cusp_len, 'fold_confirmed': bool(rev.any() or len(si) > 0), 'near_cusp': bool(near.any()), 'speed_ratio': [None if np.isnan(x) else float(x) for x in ratio]}


def resolution_check(make_dense, step_m, thr=None, closed=False):
    """make_dense(step_m) -> (C, R) dense centre and offset paths. Evaluates the metrics at `step_m` and `step_m/2`. The dense path is UNDER-RESOLVED (a narrower defect may still exist and the
    step must be reduced) when the fold / near-cusp VERDICTS differ, or when the centre-path arc length over which the row runs backwards (or is near a cusp) changes by more than
    thr.resolution_rel_tol plus one half-step of discretisation. The minimum speed ratio is reported but NOT used: at an exact cusp a chord that straddles the reversal takes any value in [0, 1]
    depending on where the sample falls, so it does not converge. Passing this check shows the verdict is stable under halving; it cannot prove that no defect narrower than the half step exists."""
    thr = thr or RowThresholds(); C1, R1 = make_dense(step_m); C2, R2 = make_dense(step_m / 2); m1 = dense_path_metrics(C1, R1, closed, thr); m2 = dense_path_metrics(C2, R2, closed, thr)
    flags_differ = (m1['fold_confirmed'] != m2['fold_confirmed']) or (m1['near_cusp'] != m2['near_cusp'])
    def stable(a, b): return abs(a - b) <= thr.resolution_rel_tol * max(a, b) + step_m / 2
    lens_stable = stable(m1['reversed_centre_arc_length_m'], m2['reversed_centre_arc_length_m']) and stable(m1['near_cusp_centre_arc_length_m'], m2['near_cusp_centre_arc_length_m'])
    converged = (not flags_differ) and lens_stable
    return {'step_m': step_m, 'half_step_m': step_m / 2, 'verdict_flags_differ': bool(flags_differ), 'reversed_arc_length_m_step': m1['reversed_centre_arc_length_m'], 'reversed_arc_length_m_half_step': m2['reversed_centre_arc_length_m'],
            'near_cusp_arc_length_m_step': m1['near_cusp_centre_arc_length_m'], 'near_cusp_arc_length_m_half_step': m2['near_cusp_centre_arc_length_m'], 'min_speed_ratio_step_reported_only': m1['min_speed_ratio'], 'min_speed_ratio_half_step_reported_only': m2['min_speed_ratio'],
            'converged': bool(converged), 'metrics_half_step': m2, 'metrics_step': m1}


def model_comparison(C, N, R, offset_m):
    """Compare the measured speed ratio of the offset path with the in-surface curvature model, per segment. model_error = measured - predicted (dimensionless)."""
    C = np.asarray(C, float); R = np.asarray(R, float); k = surface_curvature(C, N); kg = 0.5 * (k['kappa_g_per_m'][1:] + k['kappa_g_per_m'][:-1]); pred = predicted_speed_ratio(kg, offset_m)
    lc = np.linalg.norm(C[1:] - C[:-1], axis=1); lr = np.linalg.norm(R[1:] - R[:-1], axis=1); meas = lr / np.maximum(lc, 1e-15); ok = np.isfinite(meas)
    err = np.where(ok, meas - pred, np.nan); fin = err[np.isfinite(err)]
    return {'offset_m': float(offset_m), 'predicted_speed_ratio': pred.tolist(), 'measured_speed_ratio': meas.tolist(), 'model_error': [None if np.isnan(x) else float(x) for x in err], 'median_abs_model_error': float(np.median(np.abs(fin))) if len(fin) else None,
            'max_abs_model_error': float(np.abs(fin).max()) if len(fin) else None, 'offset_exceeds_local_radius_segments': np.flatnonzero(pred <= 0).tolist(), 'kappa_g_per_m': k['kappa_g_per_m'].tolist(), 'kappa_3d_per_m': k['kappa_3d_per_m'].tolist()}


# ------------------------------------------------------------------------------------------------ cross-row / non-consecutive proximity
def proximity_pairs(rows, closed_rows=()):
    """rows: {row_key: {'P': (N,3) centres, 'd': (N,) diameters_m}}. Lists pairs of stones whose centres are closer than the touching distance, EXCLUDING consecutive stones of the same row (those are in
    spacing_report). category: 'same_row_nonconsecutive' or 'cross_row'. These are early warnings; the collision engine decides physical intersection."""
    keys = list(rows); P = np.vstack([np.asarray(rows[k]['P'], float) for k in keys]); D = np.concatenate([np.asarray(rows[k]['d'], float) for k in keys])
    owner = np.concatenate([[ki] * len(rows[k]['P']) for ki, k in enumerate(keys)]).astype(int); local = np.concatenate([np.arange(len(rows[k]['P'])) for k in keys])
    sizes = {ki: len(rows[k]['P']) for ki, k in enumerate(keys)}; closed = {keys.index(k) for k in closed_rows if k in keys}; out = []
    for a in range(len(P) - 1):
        d = np.linalg.norm(P[a + 1:] - P[a], axis=1); tm = 0.5 * (D[a] + D[a + 1:])
        for b in np.flatnonzero(d < tm - 1e-12) + a + 1:
            if owner[a] == owner[b]:
                gap = abs(int(local[a]) - int(local[b])); n = sizes[int(owner[a])]
                if gap == 1 or (int(owner[a]) in closed and gap == n - 1): continue
                cat = 'same_row_nonconsecutive'
            else: cat = 'cross_row'
            out.append({'category': cat, 'row_a': keys[int(owner[a])], 'index_a': int(local[a]), 'row_b': keys[int(owner[b])], 'index_b': int(local[b]), 'centre_distance_m': float(d[b - a - 1]), 'touch_m': float(tm[b - a - 1])})
    return out
