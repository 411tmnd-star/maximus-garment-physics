"""contracts.py  explicit contracts between garment geometry and rhinestone placement (docs/PLACEMENT_CONTRACT.md, schemas/*.schema.json). Pure Python: no Blender.
Surface contract: which panels exist, on which surface and layer, what 'outward' means for each, which zones may be placed on it, and how to select its faces.
Curve contract:   which surface+panel+zone each placement curve belongs to, how to sample it, and per-zone policy (thresholds, size substitution).
validate_* return a list of {code, path, message}; an empty list means valid. Nothing is inferred or defaulted silently except values documented below."""
import json
from dataclasses import dataclass, field

SURFACE_SCHEMA = 'betterstyle.surface_contract/1'; CURVE_SCHEMA = 'betterstyle.curve_contract/1'
OUTWARD_KINDS = ('vector', 'radial_axis', 'from_point', 'toward_point', 'away_from_surface', 'registered_function')
SELECT_KINDS = ('all_faces', 'face_indices', 'vertex_group', 'material_index', 'faces_facing_outward', 'faces_facing_inward')
SOURCE_KINDS = ('synthetic_fixture', 'engine_archetype', 'production_asset')
CURVE_SOURCE_KINDS = ('blender_curve', 'polyline_world', 'bezier_world')                          # bezier_world: Stage 4 authored curves (anchors + handles in world space)
PROVENANCE_KINDS = ('authored', 'derived_slice', 'synthetic_fixture')                                 # authored = operator design intent; derived_slice = plane slice (Stage 3), NOT design intent
AUTH_STATUSES = ('suggested', 'authorized', 'rejected')
SIZE_CLASSES = ('SS10', 'SS16', 'SS20')
FRAMES = ('world', 'object_local')
THRESHOLD_KEYS = ('contact_tol_m', 'accept_centre_gap_m', 'bridge_max_gap_m', 'review_max_gap_m', 'min_rim_coverage', 'penetration_tol_m', 'max_normal_dev_deg', 'min_layer_support', 'max_footprint_misses', 'occlusion_tol_m', 'ambiguity_dot', 'mixed_normal_band')


def _e(errors, code, path, msg): errors.append({'code': code, 'path': path, 'message': msg})


def validate_surface_contract(c):
    errs = []
    if not isinstance(c, dict): _e(errs, 'C_TYPE', '$', 'contract must be an object'); return errs
    if c.get('schema') != SURFACE_SCHEMA: _e(errs, 'C_SCHEMA', '$.schema', f'expected {SURFACE_SCHEMA}')
    for k in ('garment_id', 'source_kind', 'panels'):
        if k not in c: _e(errs, 'C_MISSING', f'$.{k}', 'required')
    if c.get('source_kind') not in SOURCE_KINDS: _e(errs, 'C_ENUM', '$.source_kind', f'one of {SOURCE_KINDS}')
    seen = set(); layers = {}
    for i, p in enumerate(c.get('panels', [])):
        path = f'$.panels[{i}]'
        for k in ('panel_id', 'surface_id', 'layer_id', 'outward_direction', 'placement_zone', 'surface_reference'):
            if k not in p: _e(errs, 'C_MISSING', f'{path}.{k}', 'required')
        pid = p.get('panel_id')
        if pid in seen: _e(errs, 'C_DUPLICATE', f'{path}.panel_id', f'duplicate panel_id {pid}')
        seen.add(pid)
        if not isinstance(p.get('placement_zone', []), list) or not p.get('placement_zone'): _e(errs, 'C_ZONES', f'{path}.placement_zone', 'non-empty list of zone names')
        od = p.get('outward_direction', {})
        if od.get('kind') not in OUTWARD_KINDS: _e(errs, 'C_OUTWARD_KIND', f'{path}.outward_direction.kind', f'one of {OUTWARD_KINDS}')
        elif od['kind'] == 'vector' and len(od.get('v', [])) != 3: _e(errs, 'C_OUTWARD_PARAM', f'{path}.outward_direction.v', 'vector of 3 numbers')
        elif od['kind'] in ('radial_axis',) and not ('origin' in od and 'axis' in od): _e(errs, 'C_OUTWARD_PARAM', f'{path}.outward_direction', 'origin and axis required')
        elif od['kind'] in ('from_point', 'toward_point') and len(od.get('point', [])) != 3: _e(errs, 'C_OUTWARD_PARAM', f'{path}.outward_direction.point', 'point of 3 numbers')
        elif od['kind'] == 'away_from_surface' and not od.get('reference_object'): _e(errs, 'C_OUTWARD_PARAM', f'{path}.outward_direction.reference_object', 'required')
        elif od['kind'] == 'registered_function' and not od.get('name'): _e(errs, 'C_OUTWARD_PARAM', f'{path}.outward_direction.name', 'required (test fixtures only)')
        if od.get('frame', 'world') not in FRAMES: _e(errs, 'C_FRAME', f'{path}.outward_direction.frame', f'one of {FRAMES}')
        oa = p.get('outward_authorization')
        if oa is not None:
            if oa.get('status') not in AUTH_STATUSES: _e(errs, 'C_AUTH_STATUS', f'{path}.outward_authorization.status', f'one of {AUTH_STATUSES}')
            if oa.get('status') == 'authorized':
                for k in ('authorized_by', 'authorized_at', 'authorized_spec_hash'):
                    if not oa.get(k): _e(errs, 'C_AUTH_FIELD', f'{path}.outward_authorization.{k}', 'required when status is authorized')
        sr = p.get('surface_reference', {})
        if not sr.get('object'): _e(errs, 'C_SURFACE_REF', f'{path}.surface_reference.object', 'required')
        for j, s in enumerate(sr.get('select', [{'kind': 'all_faces'}])):
            if s.get('kind') not in SELECT_KINDS: _e(errs, 'C_SELECT_KIND', f'{path}.surface_reference.select[{j}].kind', f'one of {SELECT_KINDS}')
            if s.get('kind') == 'vertex_group' and not s.get('name'): _e(errs, 'C_SELECT_PARAM', f'{path}.surface_reference.select[{j}].name', 'required')
        if not isinstance(p.get('layer_rank', 0), int): _e(errs, 'C_LAYER_RANK', f'{path}.layer_rank', 'integer (0 = outermost)')
        layers.setdefault((p.get('surface_id'), p.get('layer_id')), []).append(pid)
    return errs


def validate_curve_contract(cc, surface_contract):
    errs = []
    if cc.get('schema') != CURVE_SCHEMA: _e(errs, 'C_SCHEMA', '$.schema', f'expected {CURVE_SCHEMA}')
    panels = {p['panel_id']: p for p in surface_contract.get('panels', [])}; zones = cc.get('zones', {}); seen = set()
    for zn, z in zones.items():
        for k in z.get('thresholds', {}):
            if k not in THRESHOLD_KEYS: _e(errs, 'C_THRESHOLD_KEY', f'$.zones.{zn}.thresholds.{k}', f'unknown threshold; allowed {THRESHOLD_KEYS}')
        sub = z.get('size_substitution', {'enabled': False})
        if sub.get('enabled') and not sub.get('allowed_to'): _e(errs, 'C_SUBSTITUTION', f'$.zones.{zn}.size_substitution', 'enabled substitution needs allowed_to (ordered list of smaller classes)')
        for s in sub.get('allowed_to', []):
            if s not in SIZE_CLASSES: _e(errs, 'C_SIZE_CLASS', f'$.zones.{zn}.size_substitution.allowed_to', f'{s} is not a size class')
        if not isinstance(z.get('zone_id', 0), int): _e(errs, 'C_ZONE_ID', f'$.zones.{zn}.zone_id', 'integer')
    for i, c in enumerate(cc.get('curves', [])):
        path = f'$.curves[{i}]'
        for k in ('curve_id', 'source', 'surface_id', 'panel_id', 'placement_zone', 'sampler'):
            if k not in c: _e(errs, 'C_MISSING', f'{path}.{k}', 'required: every curve must name its surface, panel and zone' if k in ('surface_id', 'panel_id', 'placement_zone') else 'required')
        if c.get('curve_id') in seen: _e(errs, 'C_DUPLICATE', f'{path}.curve_id', 'duplicate curve_id')
        seen.add(c.get('curve_id'))
        src = c.get('source', {})
        if src.get('kind') not in CURVE_SOURCE_KINDS: _e(errs, 'C_SOURCE_KIND', f'{path}.source.kind', f'one of {CURVE_SOURCE_KINDS}')
        elif src['kind'] == 'bezier_world':
            sg = src.get('segments')
            if not sg or any(len(x) != 4 or any(len(q) != 3 for q in x) for x in sg): _e(errs, 'C_BEZIER', f'{path}.source.segments', 'non-empty list of segments, each 4 points of 3 numbers')
        pv = c.get('provenance')
        if pv is not None:
            if pv.get('kind') not in PROVENANCE_KINDS: _e(errs, 'C_PROVENANCE', f'{path}.provenance.kind', f'one of {PROVENANCE_KINDS}')
            elif pv['kind'] == 'authored' and not (pv.get('design_id') and pv.get('revision') is not None): _e(errs, 'C_PROVENANCE', f'{path}.provenance', 'authored curves need design_id and revision')
        p = panels.get(c.get('panel_id'))
        if c.get('panel_id') is not None and p is None: _e(errs, 'C_UNKNOWN_PANEL', f'{path}.panel_id', f"panel {c.get('panel_id')} is not in the surface contract")
        elif p is not None:
            if p['surface_id'] != c.get('surface_id'): _e(errs, 'C_SURFACE_MISMATCH', f'{path}.surface_id', f"panel {p['panel_id']} belongs to surface {p['surface_id']}, curve says {c.get('surface_id')}")
            if c.get('placement_zone') not in p['placement_zone']: _e(errs, 'C_ZONE_NOT_ALLOWED', f'{path}.placement_zone', f"zone {c.get('placement_zone')} is not allowed on panel {p['panel_id']} (allowed {p['placement_zone']})")
        if c.get('placement_zone') not in zones: _e(errs, 'C_UNKNOWN_ZONE', f'{path}.placement_zone', f"zone {c.get('placement_zone')} has no entry in $.zones")
        for s in c.get('sampler', {}).get('size_pattern', []):
            if s not in SIZE_CLASSES: _e(errs, 'C_SIZE_CLASS', f'{path}.sampler.size_pattern', f'{s} is not a size class')
        if c.get('tangent', 'curve') != 'curve': _e(errs, 'C_TANGENT', f'{path}.tangent', "only 'curve' is defined; the tangent is never derived from the outward direction")
        if 'outward_direction' in c: _e(errs, 'C_OUTWARD_ON_CURVE', f'{path}.outward_direction', 'outward direction belongs to the panel, not the curve')
    return errs


@dataclass
class ContractBundle:
    surface: dict; curves: dict
    def validate(self): return validate_surface_contract(self.surface) + validate_curve_contract(self.curves, self.surface)
    def panel(self, pid): return next(p for p in self.surface['panels'] if p['panel_id'] == pid)


def load_bundle(surface_path, curves_path):
    return ContractBundle(json.load(open(surface_path)), json.load(open(curves_path)))
