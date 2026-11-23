"""placement_v2.py  contract-driven placement engine with full-footprint support analysis, explicit outcomes and collision detection (Blender layer).
Step 2's py/placement.py is untouched and still tested; this module is the stage-3 engine.

What is different from Step 2
  * The outward direction is DECLARED by the panel contract (never taken from the nearest triangle). The evaluated local surface normal (footprint plane fit) is corrected against it:
    flipped when it points against the declaration (noted), reported as ambiguous when it is nearly perpendicular to it, reported as inconsistent when the faces under one stone disagree.
  * A curve names its surface, panel and zone. Rays are cast on THAT panel's faces only. Anything else above it (another layer, the inner shell, an overlay object) is detected and reported as occlusion.
  * Each stone gets a 49-point footprint analysis: centre / maximum / minimum clearance, contact coverage, penetration, normal deviation, and whether every supported point lies on the intended layer.
  * Outcome (ACCEPTED, ACCEPTED_WITH_CLEARANCE, FLOATING_REVIEW, SIZE_SUBSTITUTION_CANDIDATE, REJECTED) comes from placement_policy with configurable thresholds. Geometry is never deformed and no filler is created.
  * The curve tangent only sets the yaw about the outward axis; it never influences the outward direction.
  * Every stone is accounted for: with a transform (any outcome, written to BSRS) or without (reported with a reason). Collisions are reported, never resolved by removal.
"""
import math, sys
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arc_sampler as AS, rhs_format as RF, stone_matrix as SM, placement_policy as PP, collisions as CO, contracts as CT, surfaces as SF
import placement as PL1                      # Step-2 module: curve -> ArcPath conversion is reused unchanged
import mathutils

DIAM = AS.DIAMETERS_M


def _octagon(r, off=0.0): return [(r * math.cos(math.radians(off + 45 * k)), -r * math.sin(math.radians(off + 45 * k))) for k in range(8)]
FIT_UNIT = [(0.0, 0.0)] + _octagon(0.5) + _octagon(0.5 * math.cos(math.radians(22.5)), 22.5)             # 17 points used for the plane fit
def _eval_unit():
    c = _octagon(0.5); pts = [(0.0, 0.0)] + c; rim = list(range(1, 9))
    for k in range(8):
        a, b = np.array(c[k]), np.array(c[(k + 1) % 8]); rim.append(len(pts)); pts.append(tuple((a + b) / 2)); pts += [tuple(a * 0.75 + b * 0.25), tuple(a * 0.25 + b * 0.75)]
    for f in (0.5, 0.75): pts += [(f * x, f * z) for x, z in c]
    return pts, rim
EVAL_UNIT, RIM_IDX = _eval_unit()                                                                          # 49 points; indices of the 16 rim points (8 corners + 8 edge midpoints)
assert len(EVAL_UNIT) == 49 and len(RIM_IDX) == 16


@dataclass
class EngineConfig:
    thresholds: PP.Thresholds = field(default_factory=PP.Thresholds)
    clearance_m: float = 1e-4; max_snap_m: float = 0.010; cast_dist_m: float = 0.030; probe_height_m: float = 0.004; fit_iterations: int = 2
    other_panel_margin_m: float = 0.0005; lift_flag_m: float = 5e-5; collision: CO.CollisionConfig = field(default_factory=CO.CollisionConfig)
    projection_mode: str = 'outward'        # where the curve point lands on the panel. 'outward' (Stage 3): along the declared outward direction. 'normal' (Stage 4): closest point on the panel (along the surface normal), which keeps the spacing along the curve when the declared outward direction is oblique to the surface
    row_offset_mode: str = 'plane'          # 'plane' (Stage 3, unchanged): offset in the tangent plane at the curve point, sign from the raw mesh winding. 'surface' (Stage 4): small steps ALONG the panel, re-projected each step, sign from the DECLARED outward direction
    keep_footprint: bool = False            # Stage 4: store the 49 footprint gaps and support flags in each record (engineering review). Off by default: Stage-3 behaviour is unchanged


def _frame(n, tan):
    t = tan - (tan @ n) * n; degenerate = np.linalg.norm(t) < 0.05            # tangent within ~3 degrees of the normal: no usable yaw, fall back to a stable frame (flagged)
    if degenerate: t = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0]); t = t - (t @ n) * n
    t = t / np.linalg.norm(t); return t, np.cross(t, n), bool(degenerate)


def _angle(a, b): return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b)))))


class PlacementEngine:
    def __init__(self, surfaces, bundle, cfg=None):
        self.s = surfaces; self.b = bundle; self.cfg = cfg or EngineConfig()
        errs = bundle.validate()
        if errs: raise SF.ContractGeometryError('invalid contract bundle: ' + '; '.join(f"{e['code']} {e['path']}" for e in errs[:6]))

    # ------------------------------------------------------------------ one stone, one size
    def _core(self, panel, p, tan, size, th):
        """Fit + footprint analysis. Returns dict with 'error' (reason, notes) or the full result."""
        cfg = self.cfg; d = DIAM[size]; pid = panel.panel_id; H = cfg.probe_height_m; res = {'ambiguities': [], 'notes': []}
        o = panel.outward(np.asarray(p, float)[None])[0]; res['outward'] = o
        first = self.s.cast_panel(pid, p + o * cfg.cast_dist_m, -o, cfg.cast_dist_m * 2)
        if first is None:
            anyh = self.s.cast_everything(p + o * cfg.cast_dist_m, -o, cfg.cast_dist_m * 2)
            res['error'] = ('OFF_PANEL', f'no face of panel {pid} below the curve point; the surface found there is {anyh[1]}') if anyh else ('NO_SURFACE', 'no geometry below the curve point along the declared outward direction'); return res
        p0, nraw, dist_along, _ = first
        if cfg.projection_mode == 'normal':
            q = panel.bvh.find_nearest(mathutils.Vector(p))
            if q[0] is not None and np.linalg.norm(np.array(q[0]) - p0) < max(0.004, 3.0 * float(np.linalg.norm(p0 - p))): p0 = np.array(q[0], float); nraw = np.array(q[1], float)        # closest point on THIS panel, accepted only if it is near the outward-cast hit (same local sheet; the allowance scales with the sag of the curve above the surface)
        snap = float(np.linalg.norm(p0 - p)); res['snap_m'] = snap; res['first_hit'] = p0
        if snap > cfg.max_snap_m: res['error'] = ('SNAP_TOO_FAR', f'{snap * 1e3:.2f} mm from the curve point to panel {pid}'); return res
        ev = self.s.cast_everything(p + o * cfg.cast_dist_m, -o, cfg.cast_dist_m * 2)
        if ev and pid not in ev[3] and ev[0] < dist_along - th.occlusion_tol_m: res['ambiguities'].append({'kind': 'OCCLUDED_BY_OTHER_LAYER', 'occluder': ev[1], 'height_mm': (dist_along - ev[0]) * 1e3})
        dpan = snap
        for opid, dist in self.s.nearest_panel_distances(p).items():
            if opid != pid and dist is not None and dist < dpan - cfg.other_panel_margin_m:
                res['ambiguities'].append({'kind': 'CURVE_CLOSER_TO_OTHER_PANEL', 'panel': opid, 'distance_mm': dist * 1e3, 'declared_panel_distance_mm': dpan * 1e3})
        dot0 = float(nraw @ o); res['normal_flipped'] = dot0 < 0; n = nraw if dot0 >= 0 else -nraw
        if abs(dot0) < th.ambiguity_dot: res['ambiguities'].append({'kind': 'AMBIGUOUS_OUTWARD', 'dot': dot0}); res['error'] = ('AMBIGUOUS_OUTWARD', f'face normal and declared outward are nearly perpendicular (dot {dot0:.3f})')
        misses = 0; raws = []
        for _ in range(cfg.fit_iterations):
            t, z, dg = _frame(n, tan); pts = []; ok = []; raws = []
            for lx, lz in FIT_UNIT:
                q = p0 + t * lx * d + z * lz * d; h = self.s.cast_panel(pid, q + n * H, -n, 2 * H)
                if h is None: ok.append(False); continue
                ok.append(True); pts.append(h[0]); raws.append(h[1])
            misses = ok.count(False)
            if not ok[0]: res['error'] = ('NO_SURFACE', 'centre footprint ray found no panel face'); return res
            P = np.array(pts); c = P.mean(0)
            if len(P) >= 4: nn = np.linalg.svd(P - c)[2][-1]; n = nn if nn @ o >= 0 else -nn; n = n / np.linalg.norm(n)
        fr = float(np.mean([(r @ o) > 0 for r in raws])) if raws else 1.0
        if th.mixed_normal_band < fr < 1 - th.mixed_normal_band: res['ambiguities'].append({'kind': 'INCONSISTENT_NORMALS', 'fraction_facing_outward': fr}); res.setdefault('error', ('INCONSISTENT_NORMALS', f'faces under the stone disagree about orientation ({fr:.2f} face outward)'))
        t, z, dg = _frame(n, tan); res['tangent_degenerate'] = dg; P = np.array(pts); p0c = P[0]; lift = float(((P - p0c) @ n).max())
        base = p0c + n * (lift + cfg.clearance_m); res.update(base=base, normal=n, tangent=t, z=z, lift_m=lift, size=size, diameter=d)
        # ---- full footprint analysis on a DIFFERENT, denser point set than the fit (49 points)
        gaps = np.full(49, np.nan); supported = np.zeros(49, bool); had_panel = np.zeros(49, bool); foreign = 0; hitn = None
        for k, (lx, lz) in enumerate(EVAL_UNIT):
            q = base + t * lx * d + z * lz * d; org = q + n * H; hp = self.s.cast_panel(pid, org, -n, 2 * H)
            if hp is None: continue
            had_panel[k] = True; gaps[k] = hp[2] - H; he = self.s.cast_everything(org, -n, 2 * H)
            supported[k] = he is not None and pid in he[3] and abs(he[0] - hp[2]) < 1e-6
            if not supported[k]: foreign += 1
            if k == 0: hitn = hp[1] if hp[1] @ o >= 0 else -hp[1]
        if not had_panel[0]: res['error'] = ('NO_SURFACE', 'centre of the footprint has no panel face below it'); return res
        hv = ~np.isnan(gaps); cl = cfg.clearance_m; contact = ((gaps[hv] - cl) <= th.contact_tol_m); rim = [gaps[i] for i in RIM_IDX if hv[i]]
        meas = PP.Measurement(float(gaps[0]), float(np.nanmax(gaps)), float(np.nanmin(gaps)), float(contact.mean()), float(np.mean([(g - cl) <= th.contact_tol_m for g in rim])) if rim else 0.0, float(max(0.0, -np.nanmin(gaps))),
                              _angle(n, hitn), _angle(n, o), float(supported[had_panel].mean()) if had_panel.any() else 0.0, int(foreign), int(49 - had_panel.sum()))
        res['measurement'] = meas; res['gaps'] = gaps; res['supported'] = supported; res['had_panel'] = had_panel; return res

    # ------------------------------------------------------------------ one sample -> record
    def place_sample(self, panel, p, tan, size, zone_cfg, th):
        res = self._core(panel, np.asarray(p, float), np.asarray(tan, float), size, th); rec = {'intended_size': size, 'realized_size': size, 'notes': [], 'ambiguities': res['ambiguities'], 'transform': None, 'measurement': None}
        if 'snap_m' in res: rec['snap_mm'] = res['snap_m'] * 1e3
        if 'error' in res and 'base' not in res: rec.update(outcome='REJECTED', reason=res['error'][0], notes=[res['error'][1]]); return rec
        meas = res['measurement']; rec['normal_flipped'] = res['normal_flipped']; rec['lift_mm'] = res['lift_m'] * 1e3; rec['tangent_degenerate'] = res.get('tangent_degenerate', False)
        def at_size(s2):
            r2 = self._core(panel, np.asarray(p, float), np.asarray(tan, float), s2, th); return r2.get('measurement') if 'error' not in r2 else None
        dec = PP.decide(meas, th, size, zone_cfg, at_size)
        order = [a['kind'] for a in res['ambiguities']]
        if order or 'error' in res:                                                                       # ambiguity beats everything: the intended surface or direction is not established
            kind = order[0] if order else res['error'][0]; dec = PP.Decision('REJECTED', kind if kind in PP.REASON_ID else 'CONTRACT_ERROR', size, size, [f'ambiguity: {a}' for a in res['ambiguities']] or [res['error'][1]])
        final = res
        if dec.outcome == 'SIZE_SUBSTITUTION_CANDIDATE':
            r2 = self._core(panel, np.asarray(p, float), np.asarray(tan, float), dec.realized_size, th); final = r2; meas = r2['measurement']; rec['lift_mm'] = r2['lift_m'] * 1e3
        rec.update(outcome=dec.outcome, reason=dec.reason, realized_size=dec.realized_size, notes=list(dec.notes), substitution_would_pass=list(dec.substitution_would_pass))
        rec['measurement'] = {'centre_clearance_mm': meas.centre_clearance_m * 1e3, 'max_footprint_clearance_mm': meas.max_footprint_clearance_m * 1e3, 'min_clearance_mm': meas.min_clearance_m * 1e3, 'contact_coverage': meas.contact_coverage, 'rim_coverage': meas.rim_coverage,
                              'penetration_mm': meas.penetration_m * 1e3, 'normal_dev_local_deg': meas.normal_dev_local_deg, 'normal_dev_outward_deg': meas.normal_dev_outward_deg, 'layer_support': meas.layer_support, 'foreign_points': meas.foreign_points, 'footprint_misses': meas.misses}
        rec['transform'] = {'base': final['base'], 'normal': final['normal'], 'tangent': final['tangent'], 'diameter': final['diameter']}
        fl = 0
        if final['lift_m'] > self.cfg.lift_flag_m: fl |= RF.FLAG_LIFTED
        if meas.centre_clearance_m > th.bridge_max_gap_m: fl |= RF.FLAG_FLOATING
        if dec.outcome != 'ACCEPTED' and dec.reason in ('EDGE_OVERHANG', 'NORMAL_DEVIATION', 'UNSUPPORTED_BY_INTENDED_LAYER'): fl |= RF.FLAG_LOW_CONFIDENCE
        rec['flags'] = fl
        if self.cfg.keep_footprint: rec['footprint'] = {'gaps_mm': [None if np.isnan(x) else float(x) * 1e3 for x in final['gaps']], 'supported': [bool(x) for x in final['supported']], 'had_panel': [bool(x) for x in final['had_panel']], 'rim_index': list(RIM_IDX)}
        return rec

    # ------------------------------------------------------------------ curves
    def run(self, curve_paths):
        """curve_paths: {curve_id: ArcPath}. Returns PlacementRun. All stones of all curves are placed first, then collisions are evaluated across everything."""
        recs = []; curve_reports = {}
        for c in self.b.curves['curves']:
            cid = c['curve_id']; path = curve_paths[cid]; panel = self.s.panels[c['panel_id']]; zone = c['placement_zone']; zc = self.b.curves['zones'][zone]; th = self.cfg.thresholds.merged(zc.get('thresholds', {}))
            sm = c['sampler']; sc = AS.SamplerConfig(size_pattern=tuple(sm.get('size_pattern', ['SS16'])), gap_m=sm.get('gap_m') or {'SS10': 0.0004, 'SS16': 0.0004, 'SS20': 0.0004}, inset_start_m=sm.get('inset_start_m', 0.0), inset_end_m=sm.get('inset_end_m', 0.0),
                                                      mode=sm.get('mode', 'fit'), phase_m=sm.get('phase_m', 0.0), lateral_offsets_m=tuple(sm.get('lateral_offsets_m', (0.0,))))
            ss = AS.sample_path(path, sc); curve_reports[cid] = {'sampler': ss.report, 'panel_id': panel.panel_id, 'zone': zone, 'attempted': 0}
            for row, off in enumerate(sc.lateral_offsets_m):
                for i in range(len(ss.s)):
                    p = ss.pos[i]; tan = ss.tan[i]; sid = f'{cid}#r{row}:{i}'; curve_reports[cid]['attempted'] += 1
                    if off != 0.0 and self.cfg.row_offset_mode == 'surface':
                        q = self._offset_on_surface(panel, p, tan, off)
                        if q is None: recs.append(self._fail(c, panel, zc, cid, sid, i, row, ss, 'OFF_PANEL', f'row offset {off * 1e3:.1f} mm leaves the panel (no face below the stepped point)', p)); continue
                        p = q
                    elif off != 0.0:
                        h0 = self.s.cast_panel(panel.panel_id, p + panel.outward(p[None])[0] * self.cfg.cast_dist_m, -panel.outward(p[None])[0], self.cfg.cast_dist_m * 2)
                        if h0 is None: recs.append(self._fail(c, panel, zc, cid, sid, i, row, ss, 'NO_SURFACE', 'row-offset pre-pass found no panel face', p)); continue
                        side = np.cross(h0[1], tan); side /= np.linalg.norm(side); p = h0[0] + side * off
                    rec = self.place_sample(panel, p, tan, ss.size_class[i], zc, th); rec.update(stone_id=sid, curve_id=cid, panel_id=panel.panel_id, surface_id=panel.surface_id, layer_id=panel.layer_id, zone=zone, zone_id=zc.get('zone_id', 0), row=row, sample=i,
                                                                                                    sample_index=(row << 20) | i, arc_s=float(ss.s[i]), n_in_curve=len(ss.s), bezel_id=c.get('bezel_id', 0), endpoint=bool(ss.endpoint[i]), position_world=[float(x) for x in p])
                    if rec['transform'] is not None and ss.endpoint[i]: rec['flags'] = rec.get('flags', 0) | RF.FLAG_ENDPOINT
                    recs.append(rec)
        return PlacementRun(self, recs, curve_reports)

    def _offset_on_surface(self, panel, p0, tan, off):
        """Move `off` metres sideways ALONG the panel: steps of at most 0.5 mm, each re-projected onto the panel; the direction is cross(outward-oriented normal, tangent), so +off is always to the left of the
        curve direction seen from outside. Returns the surface point or None if the panel ends."""
        cfg = self.cfg; pid = panel.panel_id; k = max(1, int(math.ceil(abs(off) / 0.0005))); d = abs(off) / k; sgn = 1.0 if off > 0 else -1.0; cur = np.asarray(p0, float)
        for _ in range(k + 1):
            o = panel.outward(cur[None])[0]; h = self.s.cast_panel(pid, cur + o * cfg.cast_dist_m, -o, cfg.cast_dist_m * 2)
            if h is None: return None
            n = h[1] if h[1] @ o >= 0 else -h[1]; cur = h[0]
            if _ == k: break
            t = tan - (tan @ n) * n; ln = np.linalg.norm(t)
            if ln < 0.05: return None
            cur = cur + np.cross(n, t / ln) * sgn * d
        return cur

    def _fail(self, c, panel, zc, cid, sid, i, row, ss, reason, note, p):
        return {'intended_size': ss.size_class[i], 'realized_size': ss.size_class[i], 'outcome': 'REJECTED', 'reason': reason, 'notes': [note], 'ambiguities': [], 'transform': None, 'measurement': None, 'stone_id': sid, 'curve_id': cid, 'panel_id': panel.panel_id,
                'surface_id': panel.surface_id, 'layer_id': panel.layer_id, 'zone': c['placement_zone'], 'zone_id': zc.get('zone_id', 0), 'row': row, 'sample': i, 'sample_index': (row << 20) | i, 'arc_s': float(ss.s[i]), 'n_in_curve': len(ss.s), 'bezel_id': c.get('bezel_id', 0), 'endpoint': False, 'position_world': [float(x) for x in p]}


class PlacementRun:
    def __init__(self, engine, records, curve_reports):
        self.engine = engine; self.records = records; self.curve_reports = curve_reports; self.collisions = None
        placed = [r for r in records if r['transform'] is not None]
        if placed:
            self.collisions = CO.detect(dict(base=[r['transform']['base'] for r in placed], normal=[r['transform']['normal'] for r in placed], tangent=[r['transform']['tangent'] for r in placed], size=[r['realized_size'] for r in placed], id=[r['stone_id'] for r in placed],
                                             curve_id=[r['curve_id'] for r in placed], row=[r['row'] for r in placed], sample=[r['sample'] for r in placed], panel_id=[r['panel_id'] for r in placed], layer_id=[r['layer_id'] for r in placed], zone=[r['zone'] for r in placed], n_in_curve=[r['n_in_curve'] for r in placed]), engine.cfg.collision)
            for r in placed:
                pc = self.collisions['per_stone'][r['stone_id']]; r['collisions'] = pc['count']; r['collision_severity'] = pc['severity']
        assert len(records) == sum(cr['attempted'] for cr in curve_reports.values()), 'conservation violated'

    def counts(self):
        out = {'attempted': len(self.records), 'with_transform': sum(1 for r in self.records if r['transform'] is not None), 'without_transform': sum(1 for r in self.records if r['transform'] is None), 'outcomes': {}, 'reasons': {}}
        for r in self.records: out['outcomes'][r['outcome']] = out['outcomes'].get(r['outcome'], 0) + 1; out['reasons'][r['reason']] = out['reasons'].get(r['reason'], 0) + 1
        return out

    def raise_if(self, outcomes=('REJECTED',)):
        bad = [r for r in self.records if r['outcome'] in outcomes]
        if bad: raise SF.ContractGeometryError(f'{len(bad)} stones ended as {outcomes}: ' + '; '.join(sorted({r['reason'] for r in bad})))

    def to_bsrs(self):
        """BSRS v1.1: every stone WITH a transform (all outcomes), full placement metadata. Matrix scale = realized diameter; ext keeps the intended size."""
        P = [r for r in self.records if r['transform'] is not None]
        M = np.array([SM.pack_frame(tuple(r['transform']['base']), tuple(r['transform']['normal']), tuple(r['transform']['tangent']), r['transform']['diameter']) for r in P], 'f4').reshape(-1, 16)
        pl = dict(intended_size=[r['intended_size'] for r in P], outcome=[r['outcome'] for r in P], reason=[r['reason'] for r in P], panel=[r['panel_id'] for r in P], surface=[r['surface_id'] for r in P], layer=[r['layer_id'] for r in P], curve=[r['curve_id'] for r in P],
                  collisions=[min(255, r.get('collisions', 0)) for r in P], collision_severity=[r.get('collision_severity', 0) for r in P])
        blob = RF.write_bsrs(M, [r['realized_size'] for r in P], bezel_id=[r['bezel_id'] for r in P], zone_id=[r['zone_id'] for r in P], flags=[r.get('flags', 0) for r in P], sample_index=[r['sample_index'] for r in P], placement=pl)
        return blob, M, P

    def report(self):
        def J(o):
            if isinstance(o, dict): return {k: J(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)): return [J(v) for v in o]
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, (np.floating, np.integer)): return o.item()
            return o
        return J({'counts': self.counts(), 'curves': self.curve_reports, 'ambiguities': [{'stone_id': r['stone_id'], 'panel_id': r['panel_id'], 'ambiguities': r['ambiguities']} for r in self.records if r['ambiguities']],
                  'without_transform': [{'stone_id': r['stone_id'], 'reason': r['reason'], 'notes': r['notes'], 'position_world': r['position_world']} for r in self.records if r['transform'] is None],
                  'not_accepted': [{'stone_id': r['stone_id'], 'curve_id': r['curve_id'], 'panel_id': r['panel_id'], 'zone': r['zone'], 'outcome': r['outcome'], 'reason': r['reason'], 'intended_size': r['intended_size'], 'realized_size': r['realized_size'], 'has_transform': r['transform'] is not None,
                                   'centre_clearance_mm': (r['measurement'] or {}).get('centre_clearance_mm'), 'position_world': r['position_world'], 'notes': r['notes'][:2]} for r in self.records if r['outcome'] not in ('ACCEPTED', 'ACCEPTED_WITH_CLEARANCE')],
                  'collisions': {'summary': self.collisions['summary'] if self.collisions else {}, 'pairs': self.collisions['pairs'] if self.collisions else [], 'pair_count': len(self.collisions['pairs']) if self.collisions else 0}})
