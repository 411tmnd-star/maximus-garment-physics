"""authoring.py  Blender-side authoring library for the Stage-4 workflow (used by the add-on operators and directly by the tests).
It builds on the Stage-3 engine; it does not contain a second placement engine.

  inspect_panel_outward   examine a panel's outward reference against the mesh; verdict SUGGESTED_OK or AMBIGUOUS_REVIEW_REQUIRED, with the ambiguous locations for manual review. Never authorizes.
  author_curve_on_panel   snap operator-chosen points onto the panel along its declared outward direction and record an AUTHORED curve (design intent). Off-panel points are an error.
  tangent_preview         tangent arrows along a curve, computed from the curve only (independent of the outward direction).
  generate                design intent + surface contract -> Stage-3 bundle -> PlacementEngine.run -> GenerationResult (stones, outcomes, collisions). Curves on panels without a current authorization are
                          blocked and reported; nothing is generated for them and nothing is silently dropped.
  diff_runs               which stones changed between two generations (locality of an edit)."""
import copy, hashlib, math, sys, time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import bpy, mathutils
sys.path.insert(0, str(Path(__file__).resolve().parent))
import design_intent as DI, surfaces as SF, placement_v2 as PV, contracts as CT, arc_sampler as AS, placement_policy as PP


class AuthoringError(Exception): pass


def path_from_source(src):
    if src['kind'] == 'bezier_world': return AS.CubicBezierPath(np.array(src['segments'], float), closed=src.get('closed', False))
    if src['kind'] == 'polyline_world': return AS.PolylinePath(np.array(src['points'], float), closed=src.get('closed', False))
    raise AuthoringError(f"unsupported curve source {src['kind']}")


# ------------------------------------------------------------------------------------------------ panel outward inspection
@dataclass
class InspectThresholds:
    agree_min: float = 0.98            # fraction of the panel's selected faces whose winding normal agrees with the declared/suggested outward direction
    ambiguous_max: float = 0.05        # PROVISIONAL: fraction of faces nearly perpendicular to the reference (|dot| < ambiguity_dot) before review is required; shell rim faces at open edges alone contribute 2 to 4 percent
    ambiguity_dot: float = 0.17
    mixed_layers_min: float = 0.10     # both facing classes at least this share of the raw (unselected) faces = double-layer shell: the outer layer must be selected explicitly
    max_faces: int = 6000


def reference_suggestion(reference_object):
    """The reference-body method: outward = away from the nearest point of a reference object (e.g. the mannequin). A SUGGESTION only."""
    return {'kind': 'away_from_surface', 'reference_object': reference_object, 'frame': 'world'}


def inspect_panel_outward(contract, panel_id, objects, thr=None, registry=None, cache=None):
    thr = thr or InspectThresholds(); p = DI.panel(contract, panel_id); sr = p['surface_reference']; name = sr['object']
    cache = cache if cache is not None else {}
    if name not in cache: cache[name] = SF.ObjectData(objects[name])
    od = cache[name]; outward = SF.make_outward(p['outward_direction'], od, registry, objects)
    sel = sr.get('select', [])
    plain = [s for s in sel if s['kind'] not in ('faces_facing_outward', 'faces_facing_inward')]
    raw = SF.select_faces(od, plain, outward); chosen = SF.select_faces(od, sel, outward)
    rng = np.random.default_rng(0)
    def stats(ids):
        if len(ids) == 0: return None
        s = ids if len(ids) <= thr.max_faces else np.sort(rng.choice(ids, thr.max_faces, replace=False)); o = outward(od.centre[s]); d = (od.normal[s] * o).sum(1); return s, d
    notes = []; rep = {'panel_id': panel_id, 'object': name, 'outward_direction': copy.deepcopy(p['outward_direction']), 'raw_faces': int(len(raw)), 'selected_faces': int(len(chosen)), 'thresholds': thr.__dict__.copy()}
    r = stats(raw); c = stats(chosen)
    if r is None or c is None: raise AuthoringError(f'panel {panel_id}: empty face selection')
    _, dr = r; ids_c, dc = c
    out_share = float((dr > 0).mean()); in_share = float((dr < 0).mean()); rep.update(raw_facing_outward=out_share, raw_facing_inward=in_share, mixed_layers=bool(min(out_share, in_share) >= thr.mixed_layers_min))
    agree = float((dc > 0).mean()); amb = np.abs(dc) < thr.ambiguity_dot; amb_frac = float(amb.mean())
    ang = np.degrees(np.arccos(np.clip(dc, -1, 1))); rep['angle_deg'] = {'median': float(np.median(ang)), 'p95': float(np.percentile(ang, 95)), 'p99': float(np.percentile(ang, 99)), 'fraction_over_45': float((ang > 45).mean())}   # how loosely the reference follows the surface: informational, does not change the verdict
    rep.update(selected_agree=agree, selected_ambiguous_fraction=amb_frac, selected_ambiguous_faces=int(amb.sum()))
    loc = od.centre[ids_c[amb]]; keys = {}
    for q in loc: keys.setdefault(tuple(np.floor(q / 0.01).astype(int)), q)                          # one marker per 10 mm cell
    rep['ambiguous_locations_world'] = [q.tolist() for q in list(keys.values())[:60]]
    if rep['mixed_layers'] and not any(s['kind'] in ('faces_facing_outward', 'faces_facing_inward') for s in sel): notes.append('double-layer shell: both facing classes are present and the selection does not choose one'); agree_ok = False
    else: agree_ok = agree >= thr.agree_min
    if not agree_ok: notes.append(f'only {agree:.3f} of the selected faces agree with the outward reference')
    if amb_frac > thr.ambiguous_max: notes.append(f'{amb_frac:.3f} of the selected faces are nearly perpendicular to the outward reference ({int(amb.sum())} faces, {len(keys)} locations)')
    rep['verdict'] = 'SUGGESTED_OK' if (agree_ok and amb_frac <= thr.ambiguous_max) else 'AMBIGUOUS_REVIEW_REQUIRED'; rep['notes'] = notes; return rep


def suggest_panel_reference(contract, panel_id, reference_object, objects, thr=None, registry=None, cache=None):
    """Inspect with the reference-body method and RECORD the suggestion on the panel (status 'suggested'). The operator must still authorize."""
    p = DI.panel(contract, panel_id); old = copy.deepcopy(p['outward_direction']); p['outward_direction'] = reference_suggestion(reference_object)
    rep = inspect_panel_outward(contract, panel_id, objects, thr, registry, cache)
    DI.set_suggestion(contract, panel_id, p['outward_direction'], 'reference_body', {k: rep[k] for k in ('selected_agree', 'selected_ambiguous_fraction', 'selected_ambiguous_faces', 'raw_facing_outward', 'raw_facing_inward', 'mixed_layers', 'angle_deg', 'notes')}, rep['verdict'])
    return rep


# ------------------------------------------------------------------------------------------------ curves
def snap_point(gs, panel_id, co, cast_dist=0.03):
    """Project a world point onto the panel along the panel's declared outward direction. Raises if the panel has no face there."""
    pn = gs.panels[panel_id]; o = pn.outward(np.asarray(co, float)[None])[0]; h = gs.cast_panel(panel_id, np.asarray(co, float) + o * cast_dist, -o, cast_dist * 2)
    if h is None: raise AuthoringError(f'point {np.round(co, 4).tolist()} has no face of panel {panel_id} below it along the declared outward direction')
    return h[0], o


def author_curve_on_panel(design, gs, panel_id, curve_id, points_world, zone, closed=False, sampler=None, bezel_id=0, note='', by='unknown', handle_mode='auto', clock=None):
    """points_world: the operator's approximate points. They are snapped to the panel and become the anchors."""
    pn = gs.panels[panel_id]; snapped = [snap_point(gs, panel_id, q)[0] for q in points_world]
    return DI.author_curve(design, curve_id, panel_id, pn.surface_id, zone, [{'co': s.tolist(), 'handle_left': s.tolist(), 'handle_right': s.tolist()} for s in snapped] if handle_mode == 'manual' else snapped, closed, sampler, bezel_id, note, by, handle_mode, clock)


def resnap_curve(design, gs, curve_id, by='unknown', clock=None):
    """After a surface change or an anchor edit: re-snap every anchor to the panel (handles move with their anchor)."""
    c = DI.get_curve(design, curve_id)
    for i, a in enumerate(c['anchors']):
        s, _ = snap_point(gs, c['panel_id'], a['co'])
        if c['handle_mode'] == 'auto': a['co'] = s.tolist()
        else: DI.edit_anchor(design, curve_id, i, co=s.tolist(), by=by, clock=clock); continue
    if c['handle_mode'] == 'auto': c['anchors'] = DI.catmull_handles([a['co'] for a in c['anchors']], c['closed']); c['authored_revision'] = design['revision'] + 1; DI._bump(design, by, f'curve {curve_id} re-snapped', clock)


def tangent_preview(curve, spacing_m=0.01):
    """Tangent arrows along an authored curve: positions and unit tangents from the curve alone. Reversing the curve flips them; the panel's outward direction does not enter."""
    path = path_from_source({'kind': 'bezier_world', 'segments': DI.bezier_segments(curve['anchors'], curve['closed']), 'closed': curve['closed']}); n = max(2, int(path.length / spacing_m)); s = np.linspace(0, path.length, n, endpoint=not curve['closed'])
    return [{'arc_m': float(a), 'point': p.tolist(), 'tangent': t.tolist()} for a, p, t in zip(s, path.point(s), path.tangent(s))], path.length


def outward_preview(gs, panel_id, points):
    pn = gs.panels[panel_id]; return [{'point': np.asarray(p).tolist(), 'outward': o.tolist()} for p, o in zip(points, pn.outward(np.asarray(points, float)))]


def conform_path(gs, panel_id, path, step=0.0005, max_move=0.006):
    """Make an authored curve follow the cloth: sample it densely (<= 0.5 mm), move every point to the closest point of its panel, and return the resulting polyline, so that the sampler measures
    spacing ALONG THE SURFACE. A cubic Bezier through sparse anchors cuts across a curved surface; without this, the stone pitch measured on the 3D curve is not the pitch on the cloth.
    Points farther than max_move from the panel (the curve leaves the panel) are NOT moved, so the engine rejects the stones there with a stated reason instead of piling them on the panel edge."""
    n = max(2, int(math.ceil(path.length / step))); s = np.linspace(0, path.length, n, endpoint=not path.closed); P = path.point(s); pn = gs.panels[panel_id]; out = []; moved = 0; maxd = 0.0
    for p in P:
        h = pn.bvh.find_nearest(mathutils.Vector(p))
        if h[0] is not None and h[3] <= max_move: out.append(np.array(h[0], float)); moved += 1; maxd = max(maxd, float(h[3]))
        else: out.append(p)
    return AS.PolylinePath(np.array(out), closed=path.closed), {'points': int(n), 'moved_onto_panel': moved, 'left_in_place': int(n - moved), 'max_move_mm': maxd * 1e3, 'length_before_m': float(path.length)}


# ------------------------------------------------------------------------------------------------ generation (Stage-3 evaluator, unchanged)
@dataclass
class GenerationResult:
    run: object; bundle: object; blocked: list; design: dict; surface_contract: dict; digest: str; gs: object; paths: dict; thresholds_by_zone: dict; panel_states: dict; engine: object; conform: dict = field(default_factory=dict)


def generate(design, surface_contract, objects, engine_cfg=None, registry=None, extra_occluders=(), gs=None, strict_authorization=False, conform_to_surface=True):
    errs = DI.validate_design(design, surface_contract)
    if errs: raise AuthoringError('invalid design: ' + '; '.join(f"{e['code']} {e['path']}" for e in errs[:6]))
    errs = CT.validate_surface_contract(surface_contract)
    if errs: raise AuthoringError('invalid surface contract: ' + '; '.join(f"{e['code']} {e['path']}" for e in errs[:6]))
    blocked = DI.blocked_curves(design, surface_contract)
    if blocked and strict_authorization: raise AuthoringError('curves blocked until their panel reference is authorized: ' + ', '.join(f"{b['curve_id']}({b['state']})" for b in blocked))
    ok = copy.deepcopy(design); bid = {b['curve_id'] for b in blocked}; ok['curves'] = [c for c in ok['curves'] if c['curve_id'] not in bid]
    gs = gs or SF.GarmentSurfaces(surface_contract, objects, registry, extra_occluders)
    cc = DI.to_curve_contract(ok); bundle = CT.ContractBundle(surface_contract, cc); engine = PV.PlacementEngine(gs, bundle, engine_cfg or PV.EngineConfig(row_offset_mode='surface', projection_mode='normal', keep_footprint=True))
    paths = {}; conform = {}
    for c in cc['curves']:
        pth = path_from_source(c['source'])
        if conform_to_surface: pth, conform[c['curve_id']] = conform_path(gs, c['panel_id'], pth)
        paths[c['curve_id']] = pth
    run = engine.run(paths)
    th = {z: engine.cfg.thresholds.merged(v.get('thresholds', {})).__dict__.copy() for z, v in cc['zones'].items()}
    return GenerationResult(run, bundle, blocked, design, surface_contract, DI.design_digest(design, surface_contract), gs, paths, th, {p['panel_id']: DI.authorization_state(surface_contract, p['panel_id']) for p in surface_contract['panels']}, engine, conform)


def diff_runs(a, b, tol=1e-9):
    """Per curve: stones added, removed, moved (position / normal / outcome / size changed) between two PlacementRuns, matched by stone id."""
    ra = {r['stone_id']: r for r in a.records}; rb = {r['stone_id']: r for r in b.records}; out = {}
    for sid in sorted(set(ra) | set(rb)):
        cid = (ra.get(sid) or rb.get(sid))['curve_id']; d = out.setdefault(cid, {'added': 0, 'removed': 0, 'changed': 0, 'same': 0})
        if sid not in ra: d['added'] += 1; continue
        if sid not in rb: d['removed'] += 1; continue
        x, y = ra[sid], rb[sid]; same = (x['outcome'] == y['outcome'] and x['realized_size'] == y['realized_size'] and (x['transform'] is None) == (y['transform'] is None))
        if same and x['transform'] is not None: same = all(np.allclose(x['transform'][k], y['transform'][k], atol=tol) for k in ('base', 'normal', 'tangent'))
        d['same' if same else 'changed'] += 1
    return out
