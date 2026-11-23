"""engineering_validation.py  Stage 4A: engineering validity and approval, kept SEPARATE from the Stage 3 placement workflow outcome. Pure python/numpy (no Blender).

Three independent dimensions (never merged into one enum):
  A. WORKFLOW OUTCOME   the five Stage 3 outcomes (ACCEPTED, ACCEPTED_WITH_CLEARANCE, FLOATING_REVIEW, SIZE_SUBSTITUTION_CANDIDATE, REJECTED). Unchanged. Describes what the placement workflow did.
  B. ENGINEERING FINDINGS  derived from MEASUREMENTS (collision pairs, projection displacement, orientation angles, support clearances, row geometry, surface relations, panel authorization), never from the
                           outcome label. Each finding has a severity (ADVISORY < REVIEW < BLOCKING), an evidence class and the threshold + unit + provenance that produced it. A stone's engineering severity is the worst
                           of its findings, or CLEAR. An ACCEPTED stone can carry a BLOCKING collision finding; a FLOATING_REVIEW stone is not engineering-valid unless its own measurements say so.
  C. APPROVAL           derived from A, B and the panel authorization: APPROVED | APPROVED_WITH_ADVISORY | HELD_FOR_REVIEW | NOT_APPROVED. Only APPROVED and APPROVED_WITH_ADVISORY are eligible for the Design preview.
                        The Engineering preview always shows every placement, rejected and unresolved ones included. Nothing is deleted, moved or shrunk to obtain a clean layout.

Evidence classes on a finding: `confirmed` (physical: an exact-hull SAT collision from py/collisions.py, or a measured penetration), `measured` (a geometric measurement against a threshold),
`early_warning` (a path/geometry condition that can lead to a collision but is not one; see py/row_geometry.py), `unknown` (the data needed was not available).
Units: metres in the engine; every field that is not metres says so in its name (`_mm`, `_deg`, `_diameters`). Thresholds are explicit, configurable, PROVISIONAL, and carry their provenance; none is a production rule."""
import copy, hashlib, json, math, sys
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import placement_policy as PP, collisions as CO, row_geometry as RG

SEVERITIES = ('ADVISORY', 'REVIEW', 'BLOCKING'); SEV_RANK = {'CLEAR': 0, 'ADVISORY': 1, 'REVIEW': 2, 'BLOCKING': 3}
APPROVALS = ('APPROVED', 'APPROVED_WITH_ADVISORY', 'HELD_FOR_REVIEW', 'NOT_APPROVED'); APPROVAL_RANK = {a: i for i, a in enumerate(APPROVALS)}
DESIGN_PREVIEW_ELIGIBLE = ('APPROVED', 'APPROVED_WITH_ADVISORY')
WORKFLOW_APPROVABLE = ('ACCEPTED', 'ACCEPTED_WITH_CLEARANCE')            # identical to the Stage 4 `design_preview_outcomes`
EVIDENCE = ('confirmed', 'measured', 'early_warning', 'unknown')
COLLISION_CATEGORY = {'same_curve_consecutive': 'same_row_consecutive', 'same_curve_nonconsecutive': 'same_row_nonconsecutive', 'row_to_row': 'cross_row', 'adjacent_curves_same_panel': 'cross_curve',
                      'overlapping_regions_same_layer': 'cross_panel', 'overlapping_layers': 'cross_panel'}
_DEF_PP = PP.Thresholds(); _DEF_CO = CO.CollisionConfig()


def T(value, unit, provenance): return {'value': value, 'unit': unit, 'provenance': provenance, 'status': 'provisional'}


@dataclass
class EngineeringPolicy:
    """Explicit thresholds with units and provenance. `zone_overrides` = {zone: {name: value}}. Everything is PROVISIONAL; none is a production rule."""
    thresholds: dict = field(default_factory=lambda: {
        'projection_displacement_review_diameters': T(0.5, 'stone diameters (realized size)', 'provisional: a stone moved by more than half a diameter from the authored point is no longer where the design placed it relative to its neighbours (stone pitch is about 1.14 diameters); chosen from the checkpoint distribution (91 of 108 accepted chest_probe stones < 0.5 mm, 15 between 2.2 and 9.6 mm); no production meaning'),
        'projection_displacement_blocking_diameters': T(1.0, 'stone diameters (realized size)', 'provisional: displaced by a full diameter or more, the stone can occupy its neighbour\'s authored position; chosen with the review value above; no production meaning'),
        'orientation_review_deg': T(30.0, 'deg', 'provisional engineering REVIEW trigger, not a production threshold: checkpoint audit distribution of angle(declared outward, evaluated surface normal): median 9-14 deg, p95 27-78 deg per panel. The panel stays unresolved until an operator resolves it'),
        'support_centre_clearance_review_mm': T(_DEF_PP.bridge_max_gap_m * 1e3, 'mm', 'copied from placement_policy.Thresholds.bridge_max_gap_m (Stage 3 provisional bridging limit); evaluated here from the raw measurement, independently of the workflow outcome'),
        'support_penetration_blocking_mm': T(_DEF_PP.penetration_tol_m * 1e3, 'mm', 'copied from placement_policy.Thresholds.penetration_tol_m; provisional'),
        'support_min_layer_support': T(_DEF_PP.min_layer_support, 'fraction of footprint points', 'copied from placement_policy.Thresholds.min_layer_support; provisional'),
        'support_max_footprint_misses': T(_DEF_PP.max_footprint_misses, 'footprint rays', 'copied from placement_policy.Thresholds.max_footprint_misses; provisional'),
        'collision_near_clearance_mm': T(_DEF_CO.min_gap_m * 1e3, 'mm', 'copied from collisions.CollisionConfig.min_gap_m; separated hulls closer than this are near-clearance'),
        'surface_relation_other_panel_review_mm': T(0.0, 'mm', 'any CURVE_CLOSER_TO_OTHER_PANEL / OCCLUDED_BY_OTHER_LAYER / AMBIGUOUS_OUTWARD / INCONSISTENT_NORMALS ambiguity recorded by the evaluator is a finding'),
    })
    collision_severity: dict = field(default_factory=lambda: {'OVERLAP_CRITICAL': 'BLOCKING', 'OVERLAP_MAJOR': 'BLOCKING', 'OVERLAP_MINOR': 'REVIEW', 'TOO_CLOSE': 'ADVISORY'})
    collision_severity_provenance: str = 'provisional mapping of collisions.py severities (depth fractions 0.10 / 0.30 of the mean diameter and centre distance 0.5 diameters) to approval: a major or critical interpenetration is never approvable; a minor one needs review; near-clearance is advisory'
    row_severity: dict = field(default_factory=lambda: {'ROW_SPACING_BELOW_TOUCH_PRE_GENERATION': 'REVIEW', 'ROW_SPACING_BELOW_TOUCH': 'ADVISORY', 'ROW_FOLD': 'REVIEW', 'ROW_SELF_INTERSECTION': 'REVIEW', 'ROW_NEAR_CUSP': 'ADVISORY',
                                                       'ROW_NONCONSECUTIVE_PROXIMITY': 'ADVISORY', 'ROW_OFF_PANEL': 'REVIEW', 'ROW_RESOLUTION_INSUFFICIENT': 'REVIEW'})
    row_severity_provenance: str = 'provisional: path-level conditions are early warnings, they cannot BLOCK on their own because only the exact-hull collision engine decides physical intersection; a row spacing warning after generation is advisory when the hulls are separated'
    zone_overrides: dict = field(default_factory=dict)

    def value(self, name, zone=None):
        if zone is not None and name in self.zone_overrides.get(zone, {}): return self.zone_overrides[zone][name]
        return self.thresholds[name]['value']

    def threshold_ref(self, name, zone=None):
        t = self.thresholds[name]; v = self.value(name, zone); return {'name': name, 'value': v, 'unit': t['unit'], 'provenance': t['provenance'] + (' (zone override)' if v != t['value'] else ''), 'status': 'provisional'}

    def to_dict(self): return {'thresholds': copy.deepcopy(self.thresholds), 'collision_severity': dict(self.collision_severity), 'collision_severity_provenance': self.collision_severity_provenance, 'row_severity': dict(self.row_severity),
                               'row_severity_provenance': self.row_severity_provenance, 'zone_overrides': copy.deepcopy(self.zone_overrides), 'status': 'provisional, not a production rule'}


def _fid(*parts): return 'F-' + hashlib.sha1('|'.join(str(p) for p in parts).encode()).hexdigest()[:10]


def finding(code, severity, evidence, category, stones, explanation, measurements=None, threshold=None, source='', curve_ids=(), rows=(), panel_id=None, extra=None, key=()):
    assert severity in SEVERITIES and evidence in EVIDENCE
    f = {'id': _fid(code, *sorted(stones), *key), 'code': code, 'severity': severity, 'evidence': evidence, 'category': category, 'stones': sorted(stones), 'curve_ids': sorted(set(curve_ids)), 'rows': sorted(set(rows)), 'panel_id': panel_id,
         'measurements': measurements or {}, 'threshold': threshold, 'source': source, 'explanation': explanation, 'unresolved': True}
    if extra: f.update(extra)
    return f


# ------------------------------------------------------------------------------------------------ collisions (the engine is authoritative; exact verification is attached when supplied)
def collision_findings(pairs, stones, policy, verification=None):
    """pairs: collision engine pairs. stones: {stone_id: record}. verification: {frozenset(ids): verify_pairs() entry} from py/exact_hull.py. Severity is mapped from the ENGINE severity;
    the independent exact verdict is attached as evidence, and a pair the exact verifier shows to be separated is reported as `exact_verification: false_positive` (kept, not dropped)."""
    out = []
    for p in pairs:
        a, b = stones[p['stone_a']], stones[p['stone_b']]; sev = policy.collision_severity[p['severity']]; cat = COLLISION_CATEGORY[p['category']]; over = p['severity'].startswith('OVERLAP')
        v = (verification or {}).get(frozenset((p['stone_a'], p['stone_b'])))
        meas = {'engine_severity': p['severity'], 'engine_depth_mm': p['depth_mm'], 'engine_separation_mm': p['separation_mm'], 'centre_distance_mm': float(np.linalg.norm(np.array(a['transform']['base']) - np.array(b['transform']['base']))) * 1e3}
        extra = {'collision_category': cat, 'engine_category': p['category'], 'location_m': [float(x) for x in p['location']]}
        if v: extra['exact_verification'] = {'intersects_exact': v['intersects_exact'], 'exact_gap_mm': v['exact_gap_mm'], 'separating_axis': v['exact_separating_axis'], 'false_positive': v['false_positive'], 'methods_agree': not v['method_disagreement'], 'verified_by': 'py/exact_hull.py (SAT with edge-edge axes; edge-triangle intersection)'}
        if over: expl = f"crystals {p['stone_a']} and {p['stone_b']} interpenetrate ({p['depth_mm']:.3f} mm deep, {cat.replace('_', ' ')}); the exact collision engine reports {p['severity']}"
        else: expl = f"crystals {p['stone_a']} and {p['stone_b']} are separated by {p['separation_mm']:.3f} mm (< {policy.value('collision_near_clearance_mm'):.2f} mm near-clearance limit, {cat.replace('_', ' ')})"
        out.append(finding('COLLISION_OVERLAP' if over else 'COLLISION_NEAR_CLEARANCE', sev, 'confirmed' if over else 'measured', cat, [p['stone_a'], p['stone_b']], expl, meas, {'name': 'collision_severity_map', 'value': policy.collision_severity[p['severity']],
                          'unit': 'severity', 'provenance': policy.collision_severity_provenance, 'status': 'provisional'}, 'collisions.py (authoritative exact-hull SAT)', [a['curve_id'], b['curve_id']], [a['row'], b['row']], a['panel_id'] if a['panel_id'] == b['panel_id'] else None, extra, key=(p['severity'],)))
    return out


# ------------------------------------------------------------------------------------------------ projection displacement (metres internally, millimetres explicitly for reporting)
def displacement_mm(authored_m, projected_m): return float(np.linalg.norm(np.asarray(projected_m, float) - np.asarray(authored_m, float))) * 1000.0


def projection_findings(records, policy):
    """Compares the AUTHORED point (the point handed to the projection) with where the placement landed. Uses `record['projection']` (authored/projected/final coordinates, direction, settings) when the engine kept it
    (EngineConfig.keep_projection), otherwise the Stage 3 `snap_mm` (authored -> projected distance) with `detail_available: false`. A missing measurement is an `unknown` finding, never zero."""
    out = []
    for r in records:
        if r['transform'] is None: continue
        d = r['transform']['diameter']; z = r.get('zone'); rev = policy.value('projection_displacement_review_diameters', z) * d * 1e3; blk = policy.value('projection_displacement_blocking_diameters', z) * d * 1e3; pr = r.get('projection')
        if pr is not None: disp = displacement_mm(pr['authored_m'], pr['projected_m']); total = displacement_mm(pr['authored_m'], r['transform']['base']); src = 'record.projection (authored_m -> projected_m, metres converted to mm)'
        elif r.get('snap_mm') is not None: disp = float(r['snap_mm']); total = None; src = 'record.snap_mm (Stage 3: distance from the curve point to the panel hit); coordinates not recorded'
        else:
            out.append(finding('PROJECTION_RECORD_MISSING', 'REVIEW', 'unknown', 'projection', [r['stone_id']], 'no authored/projected coordinates and no snap distance were recorded for this placement; displacement cannot be assessed', {}, None, 'engine record', [r['curve_id']], [r['row']], r['panel_id'])); continue
        if disp < rev: continue
        sev = 'BLOCKING' if disp >= blk else 'REVIEW'; thr = policy.threshold_ref('projection_displacement_blocking_diameters' if sev == 'BLOCKING' else 'projection_displacement_review_diameters', z)
        meas = {'projection_displacement_mm': disp, 'stone_diameter_mm': d * 1e3, 'review_limit_mm': rev, 'blocking_limit_mm': blk, 'displacement_in_diameters': disp / (d * 1e3), 'detail_available': pr is not None}
        extra = {}
        if pr is not None:
            vec = (np.asarray(pr['projected_m'], float) - np.asarray(pr['authored_m'], float)) * 1e3; meas.update(authored_to_final_mm=total, displacement_vector_mm=[float(x) for x in vec]); extra['projection'] = {k: pr[k] for k in pr if k not in ('authored_m', 'projected_m', 'final_m')} | {'authored_m': pr['authored_m'], 'projected_m': pr['projected_m'], 'final_m': pr.get('final_m')}
        out.append(finding('PROJECTION_DISPLACEMENT', sev, 'measured', 'projection', [r['stone_id']], f"the placement landed {disp:.2f} mm ({disp / (d * 1e3):.2f} stone diameters) from the authored point on panel {r['panel_id']}; {sev} limit {(blk if sev == 'BLOCKING' else rev):.2f} mm",
                          meas, thr, src, [r['curve_id']], [r['row']], r['panel_id'], extra))
    return out


# ------------------------------------------------------------------------------------------------ support and surface relations (from raw measurements, independent of the outcome label)
def support_findings(records, policy):
    out = []
    for r in records:
        m = r.get('measurement')
        if not m: continue
        z = r.get('zone'); sid = [r['stone_id']]; base = dict(curve_ids=[r['curve_id']], rows=[r['row']], panel_id=r['panel_id'])
        if m['penetration_mm'] > policy.value('support_penetration_blocking_mm', z): out.append(finding('SUPPORT_PENETRATION', 'BLOCKING', 'confirmed', 'support', sid, f"{m['penetration_mm']:.3f} mm of the footprint lies below the base plane", {'penetration_mm': m['penetration_mm']}, policy.threshold_ref('support_penetration_blocking_mm', z), 'footprint analysis (49 rays)', **base))
        if m['centre_clearance_mm'] > policy.value('support_centre_clearance_review_mm', z): out.append(finding('SUPPORT_CENTRE_CLEARANCE', 'REVIEW', 'measured', 'support', sid, f"the stone centre floats {m['centre_clearance_mm']:.3f} mm above the cloth", {'centre_clearance_mm': m['centre_clearance_mm'], 'contact_coverage': m['contact_coverage'], 'rim_coverage': m['rim_coverage']}, policy.threshold_ref('support_centre_clearance_review_mm', z), 'footprint analysis (49 rays)', **base))
        if m['layer_support'] < policy.value('support_min_layer_support', z) or m['foreign_points'] > 0: out.append(finding('SUPPORT_LAYER', 'REVIEW', 'measured', 'surface_relation', sid, f"only {m['layer_support']:.2f} of the footprint rests on the intended layer ({m['foreign_points']} points on another surface)", {'layer_support': m['layer_support'], 'foreign_points': m['foreign_points']}, policy.threshold_ref('support_min_layer_support', z), 'footprint analysis (49 rays)', **base))
        if m['footprint_misses'] > policy.value('support_max_footprint_misses', z): out.append(finding('SUPPORT_EDGE_OVERHANG', 'REVIEW', 'measured', 'surface_relation', sid, f"{m['footprint_misses']} footprint rays found no panel surface (stone overhangs the panel edge)", {'footprint_misses': m['footprint_misses']}, policy.threshold_ref('support_max_footprint_misses', z), 'footprint analysis (49 rays)', **base))
    return out


def surface_relation_findings(records, policy):
    """Ambiguities the evaluator recorded (occlusion by another layer, curve closer to another panel, perpendicular outward, mixed face orientation). Present even on REJECTED stones: they are facts about the surface relationship."""
    out = []
    for r in records:
        for a in r.get('ambiguities', []):
            k = a['kind']; meas = {kk: vv for kk, vv in a.items() if kk != 'kind'}
            out.append(finding('SURFACE_' + k, 'REVIEW', 'measured', 'surface_relation', [r['stone_id']], f"surface relationship not established for {r['stone_id']}: {k}" + (f" ({', '.join(f'{kk}={vv:.3g}' if isinstance(vv, float) else f'{kk}={vv}' for kk, vv in meas.items())})" if meas else ''), meas, policy.threshold_ref('surface_relation_other_panel_review_mm', r.get('zone')),
                               'evaluator ambiguity record', [r['curve_id']], [r['row']], r['panel_id'], key=(k,)))
    return out


# ------------------------------------------------------------------------------------------------ orientation and authorization
def angle_deg(a, b): a = np.asarray(a, float); b = np.asarray(b, float); return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))))))


def orientation_angles(r):
    """The three directions kept apart: declared panel outward, evaluated local surface normal (outward-oriented), final crystal axis. Returns the pairwise angles in degrees and where they came from."""
    pr = r.get('projection'); m = r.get('measurement')
    if pr is not None and pr.get('declared_outward') is not None and pr.get('surface_normal') is not None and r['transform'] is not None:
        c = r['transform']['normal']; d = pr['declared_outward']; s = pr['surface_normal']
        return {'declared_vs_surface_deg': angle_deg(d, s), 'surface_vs_crystal_deg': angle_deg(s, c), 'declared_vs_crystal_deg': angle_deg(d, c), 'source': 'record.projection vectors'}
    if m: return {'declared_vs_surface_deg': None, 'surface_vs_crystal_deg': m['normal_dev_local_deg'], 'declared_vs_crystal_deg': m['normal_dev_outward_deg'], 'source': 'measurement.normal_dev_* (vectors not recorded)'}
    return None


def orientation_summary(records):
    """Per panel: distribution of angle(declared, surface) (falls back to angle(declared, crystal)). Pure statistics for the report; the review trigger is applied in orientation_findings."""
    by = {}
    for r in records:
        a = orientation_angles(r)
        if a is None: continue
        v = a['declared_vs_surface_deg'] if a['declared_vs_surface_deg'] is not None else a['declared_vs_crystal_deg']; by.setdefault(r['panel_id'], []).append(v)
    return {p: {'n': len(v), 'median_deg': float(np.median(v)), 'p95_deg': float(np.percentile(v, 95)), 'max_deg': float(np.max(v))} for p, v in by.items()}


def orientation_findings(records, panel_auth, policy):
    """A stone whose declared outward differs from the evaluated surface normal by more than the (provisional) review angle is a REVIEW finding while the panel's discrepancy is unresolved.
    A panel is resolved only by an explicit operator record (panel_auth[panel]['orientation_review'] with status 'resolved', an operator and a cause); the authored outward direction is never changed here."""
    out = []; panels = {}
    for r in records:
        if r['transform'] is None: continue
        a = orientation_angles(r)
        if a is None: continue
        z = r.get('zone'); lim = policy.value('orientation_review_deg', z); v = a['declared_vs_surface_deg'] if a['declared_vs_surface_deg'] is not None else a['declared_vs_crystal_deg']
        if v is None or v <= lim: continue
        rv = (panel_auth.get(r['panel_id']) or {}).get('orientation_review') or {}; resolved = rv.get('status') == 'resolved' and rv.get('operator') and rv.get('cause') and not rv.get('stale')
        panels.setdefault(r['panel_id'], []).append(r['stone_id'])
        meas = {'declared_vs_surface_deg': a['declared_vs_surface_deg'], 'surface_vs_crystal_deg': a['surface_vs_crystal_deg'], 'declared_vs_crystal_deg': a['declared_vs_crystal_deg'], 'angle_source': a['source'], 'review_limit_deg': lim}
        out.append(finding('ORIENTATION_DISCREPANCY' if not resolved else 'ORIENTATION_DISCREPANCY_RESOLVED', 'ADVISORY' if resolved else 'REVIEW', 'measured', 'orientation', [r['stone_id']],
                           f"declared panel outward and the evaluated surface differ by {v:.1f} deg (> {lim:.0f} deg review trigger); " + (f"resolved by {rv.get('operator')}: {rv.get('cause')}" if resolved else 'the discrepancy is unresolved: sheet selection, winding, steep geometry and a true outward error have not been separated'),
                           meas, policy.threshold_ref('orientation_review_deg', z), 'orientation comparison', [r['curve_id']], [r['row']], r['panel_id'], {'orientation_review': {k: rv.get(k) for k in ('status', 'operator', 'cause')} if rv else None}))
    return out


def authorization_findings(records, panel_auth, policy):
    """Suggested != authorized. Curves on non-authorized panels are blocked upstream; any stone that nevertheless exists on such a panel is NOT approvable. An authorization whose operator kind is not 'human'
    (scripted test operator, unspecified) is visible as an ADVISORY finding: the layout is approved on a recorded decision, not on a human review."""
    out = []; seen = set()
    for r in records:
        pid = r['panel_id']; au = panel_auth.get(pid) or {'state': 'missing'}; st = au.get('state', 'missing')
        if st != 'authorized':
            out.append(finding('PANEL_NOT_AUTHORIZED', 'BLOCKING', 'confirmed', 'authorization', [r['stone_id']], f"the outward reference of panel {pid} is '{st}', not authorized by an operator", {'authorization_state': st}, None, 'surface contract outward_authorization', [r['curve_id']], [r['row']], pid))
        elif au.get('operator_kind') != 'human':
            out.append(finding('AUTHORIZATION_NOT_HUMAN', 'ADVISORY', 'measured', 'authorization', [r['stone_id']], f"panel {pid} was authorized by '{au.get('authorized_by')}' (operator kind: {au.get('operator_kind') or 'unspecified'}), basis: inspection verdict {au.get('verdict')}; this is a recorded decision, not a human review", {'authorized_by': au.get('authorized_by'), 'operator_kind': au.get('operator_kind'), 'verdict': au.get('verdict')}, None, 'surface contract outward_authorization', [r['curve_id']], [r['row']], pid))
    return out


# ------------------------------------------------------------------------------------------------ row geometry (post-generation, from the placements actually produced)
def row_diagnostics_from_records(records):
    """Groups placed stones by (curve, row) and measures the ACTUAL resulting spacing (stone centres = transform base) and proximity. Pure; works on any run."""
    rows = {}
    for r in records:
        if r['transform'] is None: continue
        rows.setdefault((r['curve_id'], r['row']), []).append(r)
    out = {}
    for (cid, row), rs in rows.items():
        rs.sort(key=lambda x: x['sample']); P = np.array([x['transform']['base'] for x in rs]); dm = [x['transform']['diameter'] for x in rs]; closed = False
        sp = RG.spacing_report(P, dm, closed=closed)
        # only truly consecutive samples (sample index differs by 1) are pairs; a missing stone (no transform) breaks adjacency
        idx = [k for k in range(len(rs) - 1) if rs[k + 1]['sample'] - rs[k]['sample'] == 1]
        out[(cid, row)] = {'stone_ids': [x['stone_id'] for x in rs], 'samples': [x['sample'] for x in rs], 'spacing': sp, 'adjacent_pair_indices': idx, 'diameters_m': dm}
    prox = RG.proximity_pairs({f'{c}#r{r}': {'P': np.array([x['transform']['base'] for x in sorted(rs, key=lambda x: x['sample'])]), 'd': [x['transform']['diameter'] for x in sorted(rs, key=lambda x: x['sample'])]} for (c, r), rs in rows.items()})
    return out, prox


def row_findings_post_generation(records, collision_pair_set, policy, centre_chords=None):
    """Early warnings from the generated placements. A consecutive pair whose centres are closer than the touching distance is reported as ROW_SPACING_BELOW_TOUCH; if the collision engine has also flagged that pair as
    OVERLAP it is linked (`confirmed_by_collision`), otherwise it stays an early warning (the hulls may be separated depending on their yaw)."""
    diag, prox = row_diagnostics_from_records(records); out = []; sev = policy.row_severity
    for (cid, row), d in diag.items():
        sp = d['spacing']
        for k in sp.get('below_touch', []):
            if k not in d['adjacent_pair_indices']: continue
            a, b = d['stone_ids'][k], d['stone_ids'][k + 1]; conf = frozenset((a, b)) in collision_pair_set
            ratio = sp['ratio_to_touch'][k]
            out.append(finding('ROW_SPACING_BELOW_TOUCH', sev['ROW_SPACING_BELOW_TOUCH'], 'early_warning', 'row_geometry', [a, b], f"consecutive stones {a} and {b} are {sp['chord_m'][k] * 1e3:.2f} mm apart, below the touching distance {sp['touch_m'][k] * 1e3:.2f} mm (ratio {ratio:.2f}); " + ('the collision engine confirms an overlap' if conf else 'the hulls are not reported as overlapping'),
                               {'chord_m': sp['chord_m'][k], 'chord_mm': sp['chord_m'][k] * 1e3, 'touch_distance_mm': sp['touch_m'][k] * 1e3, 'guaranteed_overlap_distance_mm': sp['guaranteed_overlap_m'][k] * 1e3, 'ratio_to_touch': ratio, 'below_guaranteed_overlap': k in sp['below_guaranteed_overlap']}, None, 'row_geometry.spacing_report (generated stone centres)', [cid], [row], None, {'confirmed_by_collision': bool(conf)}))
    for p in prox:
        ra, rb = p['row_a'], p['row_b']; ca, wa = ra.rsplit('#r', 1); cb, wb = rb.rsplit('#r', 1); ida = _stone_id_by_index(diag, (ca, int(wa)), p['index_a']); idb = _stone_id_by_index(diag, (cb, int(wb)), p['index_b'])
        if ida and idb: out.append(finding('ROW_NONCONSECUTIVE_PROXIMITY', sev['ROW_NONCONSECUTIVE_PROXIMITY'], 'early_warning', 'row_geometry', [ida, idb], f"{p['category'].replace('_', ' ')}: stones {ida} and {idb} are {p['centre_distance_m'] * 1e3:.2f} mm apart (touching distance {p['touch_m'] * 1e3:.2f} mm); the collision engine decides whether the hulls intersect",
                                           {'centre_distance_mm': p['centre_distance_m'] * 1e3, 'touch_distance_mm': p['touch_m'] * 1e3, 'proximity_category': p['category']}, None, 'row_geometry.proximity_pairs', [ca, cb], [int(wa), int(wb)], None, {'confirmed_by_collision': frozenset((ida, idb)) in collision_pair_set}))
    return out


def _stone_id_by_index(diag, key, i):
    d = diag.get(key); return d['stone_ids'][i] if d and i < len(d['stone_ids']) else None


def findings_from_pre_generation(rowdiag, policy):
    """rowdiag: the dict produced by authoring.row_diagnostics (bpy side): {curve_id: {'rows': {row: {...dense metrics, spacing, resolution, off_panel, sample_stone_ids}}}}. Turns it into findings with evidence 'early_warning'
    or 'measured' (geometry facts about the offset path). These exist BEFORE any stone is generated."""
    out = []; sev = policy.row_severity
    for cid, cd in rowdiag.items():
        for row, rd in cd['rows'].items():
            ids = rd['sample_stone_ids']; base = dict(curve_ids=[cid], rows=[row]); sp = rd.get('spacing') or {}
            for k in sp.get('below_touch', []):
                a, b = ids[k], ids[k + 1]
                out.append(finding('ROW_SPACING_BELOW_TOUCH', sev['ROW_SPACING_BELOW_TOUCH_PRE_GENERATION'], 'early_warning', 'row_geometry', [a, b], f"row {row} of {cid}: the offset path puts stone {a} and {b} {sp['chord_m'][k] * 1e3:.2f} mm apart (touching distance {sp['touch_m'][k] * 1e3:.2f} mm) before any stone is generated",
                                   {'chord_mm': sp['chord_m'][k] * 1e3, 'touch_distance_mm': sp['touch_m'][k] * 1e3, 'ratio_to_touch': sp['ratio_to_touch'][k], 'ratio_to_centre': (sp['ratio_to_centre'][k] if sp.get('ratio_to_centre') else None), 'below_guaranteed_overlap': k in sp['below_guaranteed_overlap']}, None, 'row_geometry.spacing_report (offset path samples)', **base))
            dn = rd.get('dense')
            if dn:
                span = dn.get('affected_sample_pairs', [])
                aff = sorted({ids[i] for i in span} | {ids[min(i + 1, len(ids) - 1)] for i in span}) if span else ids[:1]
                if dn['self_intersection_count'] > 0: out.append(finding('ROW_SELF_INTERSECTION', sev['ROW_SELF_INTERSECTION'], 'measured', 'row_geometry', aff, f"row {row} of {cid}: the offset path crosses itself ({dn['self_intersection_count']} segment pairs on the dense path, step {rd['resolution']['step_m'] * 1e3:.2f} mm)", {'self_intersection_count': dn['self_intersection_count'], 'dense_step_mm': rd['resolution']['step_m'] * 1e3}, None, 'row_geometry.dense_path_metrics', **base))
                elif dn['fold_confirmed']: out.append(finding('ROW_FOLD', sev['ROW_FOLD'], 'measured', 'row_geometry', aff, f"row {row} of {cid}: the offset path runs against the centre curve over {dn['reversed_centre_arc_length_m'] * 1e3:.2f} mm of centre arc", {'reversed_centre_arc_length_mm': dn['reversed_centre_arc_length_m'] * 1e3, 'min_speed_ratio': dn['min_speed_ratio']}, None, 'row_geometry.dense_path_metrics', **base))
                if dn['near_cusp'] and not dn['fold_confirmed']: out.append(finding('ROW_NEAR_CUSP', sev['ROW_NEAR_CUSP'], 'early_warning', 'row_geometry', aff, f"row {row} of {cid}: the offset path nearly stalls (speed ratio {dn['min_speed_ratio']:.2f} of the centre curve) over {dn['near_cusp_centre_arc_length_m'] * 1e3:.2f} mm", {'min_speed_ratio': dn['min_speed_ratio'], 'near_cusp_arc_length_mm': dn['near_cusp_centre_arc_length_m'] * 1e3}, None, 'row_geometry.dense_path_metrics', **base))
                if dn.get('off_panel_points', 0) > 0: out.append(finding('ROW_OFF_PANEL', sev['ROW_OFF_PANEL'], 'measured', 'row_geometry', ids[:1], f"row {row} of {cid}: {dn['off_panel_points']} dense offset points leave the panel", {'off_panel_points': dn['off_panel_points']}, None, 'row_geometry.dense_path_metrics', **base))
            rs = rd.get('resolution')
            if rs and not rs['converged']: out.append(finding('ROW_RESOLUTION_INSUFFICIENT', sev['ROW_RESOLUTION_INSUFFICIENT'], 'unknown', 'row_geometry', ids[:1], f"row {row} of {cid}: the dense-path verdict changes when the step is halved ({rs['step_m'] * 1e3:.2f} mm); a narrower defect cannot be excluded", {'step_mm': rs['step_m'] * 1e3}, None, 'row_geometry.resolution_check', **base))
    return out


# ------------------------------------------------------------------------------------------------ approval
def decide_approval(outcome, engineering_severity, authorization_state, has_transform=True):
    """Pure decision from the three dimensions. Returns (decision, reasons)."""
    reasons = []; d = 'APPROVED'
    def worst(a, b): return a if APPROVAL_RANK[a] >= APPROVAL_RANK[b] else b
    if outcome == 'REJECTED': d = worst(d, 'NOT_APPROVED'); reasons.append('workflow outcome REJECTED')
    elif outcome in ('FLOATING_REVIEW', 'SIZE_SUBSTITUTION_CANDIDATE'): d = worst(d, 'HELD_FOR_REVIEW'); reasons.append(f'workflow outcome {outcome} requires review')
    elif outcome not in WORKFLOW_APPROVABLE: d = worst(d, 'NOT_APPROVED'); reasons.append(f'unknown workflow outcome {outcome}')
    if authorization_state != 'authorized': d = worst(d, 'NOT_APPROVED'); reasons.append(f"panel outward reference is '{authorization_state}', not authorized")
    if not has_transform: d = worst(d, 'NOT_APPROVED'); reasons.append('no transform was produced')
    if engineering_severity == 'BLOCKING': d = worst(d, 'NOT_APPROVED'); reasons.append('blocking engineering finding')
    elif engineering_severity == 'REVIEW': d = worst(d, 'HELD_FOR_REVIEW'); reasons.append('engineering finding requires review')
    elif engineering_severity == 'ADVISORY': d = worst(d, 'APPROVED_WITH_ADVISORY'); reasons.append('advisory engineering finding')
    return d, reasons


def validate_layout(records, collision_pairs, panel_auth, policy=None, verification=None, extra_findings=(), blocked_curves=(), design=None):
    """Aggregate every finding, assign per-stone engineering severity and approval, and summarise the layout. `records`: PlacementRun.records (optionally with `projection`). `collision_pairs`: run.collisions['pairs'] (the
    exact-hull engine output, unmodified). `panel_auth`: {panel_id: {'state', 'authorized_by', 'operator_kind', 'verdict', 'orientation_review'}}. Returns a JSON-able report. No record is modified or removed."""
    policy = policy or EngineeringPolicy(); st = {r['stone_id']: r for r in records}
    pairs = list(collision_pairs or []); cps = {frozenset((p['stone_a'], p['stone_b'])) for p in pairs if p['severity'].startswith('OVERLAP')}
    F = []; F += collision_findings(pairs, {k: v for k, v in st.items() if v['transform'] is not None}, policy, verification); F += projection_findings(records, policy); F += support_findings(records, policy); F += surface_relation_findings(records, policy)
    F += orientation_findings(records, panel_auth, policy); F += authorization_findings(records, panel_auth, policy); F += row_findings_post_generation(records, cps, policy); F += list(extra_findings)
    for b in blocked_curves: F.append(finding('CURVE_BLOCKED', 'BLOCKING', 'confirmed', 'design_intent', [], f"curve {b['curve_id']} is blocked: its panel authorization is {b.get('state')}; nothing was generated for it", {'authorization_state': b.get('state')}, None, 'design_intent.blocked_curves', [b['curve_id']], [], b.get('panel_id'), key=(b['curve_id'],)))
    # deduplicate by id (the same finding can only be produced once)
    seen = {}; [seen.setdefault(f['id'], f) for f in F]; F = list(seen.values())
    per = {}
    for f in F:
        for s in f['stones']: per.setdefault(s, []).append(f)
    stones_out = {}; xt = {}
    for r in records:
        fs = per.get(r['stone_id'], []); sev = max((f['severity'] for f in fs), key=lambda x: SEV_RANK[x], default='CLEAR'); au = (panel_auth.get(r['panel_id']) or {}).get('state', 'missing')
        dec, reasons = decide_approval(r['outcome'], sev, au, r['transform'] is not None); assessed = r['transform'] is not None
        stones_out[r['stone_id']] = {'workflow_outcome': r['outcome'], 'workflow_reason': r['reason'], 'engineering': {'assessed': assessed, 'severity': sev, 'finding_ids': [f['id'] for f in fs]},
                                     'approval': {'decision': dec, 'design_preview_eligible': dec in DESIGN_PREVIEW_ELIGIBLE, 'reasons': reasons, 'authorization_state': au}}
        xt.setdefault(r['outcome'], {}).setdefault(dec, 0); xt[r['outcome']][dec] += 1
    n = len(records); elig = [k for k, v in stones_out.items() if v['approval']['design_preview_eligible']]; exc = [k for k in stones_out if k not in set(elig)]
    by_sev = {s: sum(1 for v in stones_out.values() if v['engineering']['severity'] == s) for s in ('CLEAR',) + SEVERITIES}
    pair_rows = [{'stones': f['stones'], 'category': f['collision_category'], 'engine_severity': f['measurements']['engine_severity'], 'depth_mm': f['measurements']['engine_depth_mm'], 'severity': f['severity'], 'exact_intersects': (f.get('exact_verification') or {}).get('intersects_exact')} for f in F if f['code'] in ('COLLISION_OVERLAP', 'COLLISION_NEAR_CLEARANCE')]
    cat = {}
    for p in pair_rows: cat.setdefault(p['category'], {}).setdefault(p['engine_severity'], 0); cat[p['category']][p['engine_severity']] += 1
    worst_f = max((f['severity'] for f in F), key=lambda x: SEV_RANK[x], default='CLEAR')
    layout_status = {'BLOCKING': 'NOT_APPROVABLE', 'REVIEW': 'HELD_FOR_REVIEW', 'ADVISORY': 'APPROVABLE_WITH_ADVISORY', 'CLEAR': 'APPROVABLE'}[worst_f]
    code_sev = {}
    for f in F: code_sev.setdefault(f['code'], {}).setdefault(f['severity'], 0); code_sev[f['code']][f['severity']] += 1
    excl_reason = {}
    for k in exc:
        for rr in stones_out[k]['approval']['reasons']: excl_reason[rr] = excl_reason.get(rr, 0) + 1
    indep = {'accepted_workflow_stones_with_blocking_finding': sum(1 for k, v in stones_out.items() if v['workflow_outcome'] in WORKFLOW_APPROVABLE and v['engineering']['severity'] == 'BLOCKING'),
             'non_accepted_stones_with_clear_engineering': sum(1 for k, v in stones_out.items() if v['workflow_outcome'] not in WORKFLOW_APPROVABLE and v['engineering']['assessed'] and v['engineering']['severity'] == 'CLEAR')}
    return {'schema': 'betterstyle.engineering_validation/1', 'policy': policy.to_dict(), 'findings': F, 'stones': stones_out,
            'layout': {'status': layout_status, 'stones': n, 'engineering_severity_counts': by_sev, 'findings_by_code_and_severity': code_sev, 'collision_pairs_by_category': cat, 'unresolved_conflicts': [p for p in pair_rows if p['severity'] in ('BLOCKING', 'REVIEW')],
                       'workflow_by_approval': xt, 'independence': indep, 'design_preview': {'eligible': len(elig), 'excluded': len(exc), 'excluded_ids': exc, 'excluded_by_reason': excl_reason, 'rule': 'APPROVED and APPROVED_WITH_ADVISORY only; excluded placements stay visible in the Engineering preview'},
                       'orientation_by_panel': orientation_summary(records), 'blocked_curves': list(blocked_curves)}}
