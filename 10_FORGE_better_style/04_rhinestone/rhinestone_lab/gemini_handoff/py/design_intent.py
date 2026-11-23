"""design_intent.py  AUTHORED DESIGN INTENT, kept apart from the two things derived from it (pure Python, no Blender).

  1. design intent      curves an operator placed on a panel (anchors, handles, spacing, rows, zone)   -> design/*.design.json   (this module)
  2. generated stones   what the Stage-3 sampler + placement engine produce from the intent             -> *.bsrs (BSRS v1.1)
  3. evaluated outcomes ACCEPTED ... REJECTED, with measurements, collisions, reasons                  -> BSRS ext + evaluation report

The panel contract stays authoritative for 'outward'. A curve never carries an outward direction; it names a panel. The operator's decision about a panel's outward reference is an explicit
AUTHORIZATION recorded on the panel (`outward_authorization`): who, when, which method, which evidence, and a hash of the exact outward specification that was authorized. If the specification
changes afterwards the authorization goes stale and curves on that panel are blocked until it is authorized again. Nothing here authorizes automatically.
Curves produced by plane slicing (Stage 3) have provenance 'derived_slice' and are never authored design curves."""
import copy, hashlib, json, time
import numpy as np

DESIGN_SCHEMA = 'betterstyle.design_intent/1'
STATUSES = ('interim_engineering_test_layout', 'draft')                       # there is deliberately no 'approved_product_design' status here
AUTH_STATUSES = ('suggested', 'authorized', 'rejected')
VERDICTS = ('SUGGESTED_OK', 'AMBIGUOUS_REVIEW_REQUIRED')
DISCLAIMER = 'Interim engineering test layout on an engine-generated garment. Not the approved co-ord design and not a product placement drawing.'


class AuthorizationError(Exception): pass


def spec_hash(spec): return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]
def _now(clock=None): return clock() if clock else time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


# ------------------------------------------------------------------------------------------------ design file
def new_design(design_id, garment_id, garment_source_kind, status='interim_engineering_test_layout', author='unknown', clock=None):
    return {'schema': DESIGN_SCHEMA, 'design_id': design_id, 'revision': 1, 'status': status, 'garment_id': garment_id, 'garment_source_kind': garment_source_kind, 'disclaimer': DISCLAIMER,
            'zones': {}, 'curves': [], 'history': [{'revision': 1, 'at': _now(clock), 'by': author, 'change': 'created'}]}


def _bump(design, by, change, clock=None):
    design['revision'] += 1; design['history'].append({'revision': design['revision'], 'at': _now(clock), 'by': by, 'change': change})


def set_zone(design, name, zone_id, thresholds=None, size_substitution=None, by='unknown', clock=None):
    design['zones'][name] = {'zone_id': int(zone_id), **({'thresholds': dict(thresholds)} if thresholds else {}), 'size_substitution': dict(size_substitution or {'enabled': False})}; _bump(design, by, f'zone {name} set', clock)


def bezier_segments(anchors, closed=False):
    """Cubic Bezier segments (P0,P1,P2,P3) from anchors [{co, handle_left, handle_right}] exactly as Blender Bezier points define them."""
    n = len(anchors); m = n if closed else n - 1
    return [[anchors[i]['co'], anchors[i]['handle_right'], anchors[(i + 1) % n]['handle_left'], anchors[(i + 1) % n]['co']] for i in range(m)]


def catmull_handles(cos, closed=False, tension=1 / 6):
    """Smooth handles through points (Catmull-Rom as Bezier): used when the operator asks for automatic handles. Returns anchors with handles."""
    P = np.asarray(cos, float); n = len(P); out = []
    for i in range(n):
        prev = P[(i - 1) % n] if (closed or i > 0) else P[i]; nxt = P[(i + 1) % n] if (closed or i < n - 1) else P[i]; d = (nxt - prev) * tension * (1.0 if (closed or 0 < i < n - 1) else 2.0)
        out.append({'co': P[i].tolist(), 'handle_left': (P[i] - d).tolist(), 'handle_right': (P[i] + d).tolist()})
    return out


def author_curve(design, curve_id, panel_id, surface_id, zone, anchors, closed=False, sampler=None, bezel_id=0, note='', by='unknown', handle_mode='manual', clock=None):
    """Add an AUTHORED curve. `anchors` as in bezier_segments (or plain points with handle_mode 'auto'). The curve names its panel and surface; it carries no outward direction."""
    if any(c['curve_id'] == curve_id for c in design['curves']): raise ValueError(f'curve {curve_id} already exists')
    if handle_mode == 'auto': anchors = catmull_handles([a['co'] if isinstance(a, dict) else a for a in anchors], closed)
    c = {'curve_id': curve_id, 'panel_id': panel_id, 'surface_id': surface_id, 'placement_zone': zone, 'closed': bool(closed), 'anchors': copy.deepcopy(anchors), 'handle_mode': handle_mode,
         'sampler': copy.deepcopy(sampler or {'size_pattern': ['SS16']}), 'bezel_id': int(bezel_id), 'note': note, 'authored_revision': design['revision'] + 1}
    design['curves'].append(c); _bump(design, by, f'curve {curve_id} authored on {panel_id}', clock); return c


def get_curve(design, curve_id):
    for c in design['curves']:
        if c['curve_id'] == curve_id: return c
    raise KeyError(curve_id)


def edit_anchor(design, curve_id, index, co=None, handle_left=None, handle_right=None, by='unknown', clock=None):
    c = get_curve(design, curve_id); a = c['anchors'][index]
    if co is not None:
        if c['handle_mode'] == 'auto': a['co'] = list(map(float, co)); c['anchors'] = catmull_handles([x['co'] for x in c['anchors']], c['closed'])
        else:
            d = np.asarray(co, float) - np.asarray(a['co'], float); a['co'] = list(map(float, co)); a['handle_left'] = (np.asarray(a['handle_left']) + d).tolist(); a['handle_right'] = (np.asarray(a['handle_right']) + d).tolist()
    if handle_left is not None: a['handle_left'] = list(map(float, handle_left))
    if handle_right is not None: a['handle_right'] = list(map(float, handle_right))
    c['authored_revision'] = design['revision'] + 1; _bump(design, by, f'curve {curve_id} anchor {index} edited', clock)


def set_spacing(design, curve_id, by='unknown', clock=None, **sampler):
    c = get_curve(design, curve_id); c['sampler'].update(sampler); c['authored_revision'] = design['revision'] + 1; _bump(design, by, f'curve {curve_id} spacing {sorted(sampler)}', clock)


def reassign_panel(design, curve_id, panel_id, surface_id, zone, by='unknown', clock=None):
    c = get_curve(design, curve_id); c.update(panel_id=panel_id, surface_id=surface_id, placement_zone=zone); c['authored_revision'] = design['revision'] + 1; _bump(design, by, f'curve {curve_id} reassigned to {panel_id}', clock)


def remove_curve(design, curve_id, by='unknown', clock=None):
    design['curves'] = [c for c in design['curves'] if c['curve_id'] != curve_id]; _bump(design, by, f'curve {curve_id} removed', clock)


# ------------------------------------------------------------------------------------------------ panel outward authorization (stored on the surface contract's panels)
def panel(surface_contract, panel_id):
    for p in surface_contract['panels']:
        if p['panel_id'] == panel_id: return p
    raise KeyError(panel_id)


def set_suggestion(surface_contract, panel_id, outward_spec, method, evidence, verdict):
    """Record a SUGGESTED outward reference. This never authorizes."""
    assert verdict in VERDICTS
    p = panel(surface_contract, panel_id); p['outward_direction'] = copy.deepcopy(outward_spec)
    p['outward_authorization'] = {'status': 'suggested', 'method': method, 'verdict': verdict, 'evidence': evidence, 'spec_hash': spec_hash(outward_spec)}


def authorize_outward(surface_contract, panel_id, operator, note, acknowledge_ambiguity=False, clock=None):
    """The operator explicitly authorizes the panel's CURRENT outward specification. If the inspection verdict was AMBIGUOUS_REVIEW_REQUIRED the operator must acknowledge it (and give a note)."""
    p = panel(surface_contract, panel_id); a = p.get('outward_authorization')
    if not operator or not operator.strip() or operator == 'unknown': raise AuthorizationError('an operator name is required to authorize a panel reference')
    if not a or a.get('status') not in ('suggested', 'authorized', 'rejected'): raise AuthorizationError('the panel has no outward inspection record; inspect it first')
    if a.get('verdict') == 'AMBIGUOUS_REVIEW_REQUIRED' and not (acknowledge_ambiguity and note and note.strip()): raise AuthorizationError('the inspection found ambiguity: acknowledge it explicitly and give a note')
    a.update(status='authorized', authorized_by=operator, authorized_at=_now(clock), note=note or '', acknowledged_ambiguity=bool(acknowledge_ambiguity), authorized_spec_hash=spec_hash(p['outward_direction']))


def reject_outward(surface_contract, panel_id, operator, note, clock=None):
    p = panel(surface_contract, panel_id); a = p.setdefault('outward_authorization', {}); a.update(status='rejected', rejected_by=operator, rejected_at=_now(clock), note=note or '')


def authorization_state(surface_contract, panel_id):
    """'authorized' | 'stale' (spec changed after authorization) | 'suggested' | 'rejected' | 'missing'."""
    a = panel(surface_contract, panel_id).get('outward_authorization')
    if not a: return 'missing'
    if a.get('status') == 'authorized': return 'authorized' if a.get('authorized_spec_hash') == spec_hash(panel(surface_contract, panel_id)['outward_direction']) else 'stale'
    return a.get('status', 'missing')


def blocked_curves(design, surface_contract):
    """Curves that may NOT be generated: panel unknown or its outward reference is not currently authorized. Returns [{curve_id, panel_id, state}]."""
    ids = {p['panel_id'] for p in surface_contract['panels']}; out = []
    for c in design['curves']:
        st = authorization_state(surface_contract, c['panel_id']) if c['panel_id'] in ids else 'unknown_panel'
        if st != 'authorized': out.append({'curve_id': c['curve_id'], 'panel_id': c['panel_id'], 'state': st})
    return out


# ------------------------------------------------------------------------------------------------ conversion to the Stage-3 contracts (the engine is reused unchanged)
def validate_design(design, surface_contract):
    errs = []; e = lambda code, path, msg: errs.append({'code': code, 'path': path, 'message': msg})
    if design.get('schema') != DESIGN_SCHEMA: e('D_SCHEMA', '$.schema', f'expected {DESIGN_SCHEMA}')
    if design.get('status') not in STATUSES: e('D_STATUS', '$.status', f'one of {STATUSES}')
    panels = {p['panel_id']: p for p in surface_contract.get('panels', [])}; seen = set()
    for i, c in enumerate(design.get('curves', [])):
        p = f'$.curves[{i}]'
        for k in ('curve_id', 'panel_id', 'surface_id', 'placement_zone', 'anchors', 'sampler'):
            if k not in c: e('D_MISSING', f'{p}.{k}', 'required: an authored curve must name its panel, surface and zone' if k in ('panel_id', 'surface_id', 'placement_zone') else 'required')
        if 'outward_direction' in c or 'outward' in c: e('D_OUTWARD_ON_CURVE', p, 'a curve never defines outward; the panel contract does')
        if c.get('curve_id') in seen: e('D_DUPLICATE', f'{p}.curve_id', 'duplicate curve_id')
        seen.add(c.get('curve_id'))
        pn = panels.get(c.get('panel_id'))
        if pn is None: e('D_UNKNOWN_PANEL', f'{p}.panel_id', f"panel {c.get('panel_id')} is not in the surface contract")
        else:
            if pn['surface_id'] != c.get('surface_id'): e('D_SURFACE_MISMATCH', f'{p}.surface_id', f"panel belongs to {pn['surface_id']}")
            if c.get('placement_zone') not in pn['placement_zone']: e('D_ZONE_NOT_ALLOWED', f'{p}.placement_zone', f"zone not allowed on {pn['panel_id']}")
        if c.get('placement_zone') not in design.get('zones', {}): e('D_UNKNOWN_ZONE', f'{p}.placement_zone', 'zone has no entry in $.zones')
        an = c.get('anchors', [])
        if len(an) < 2: e('D_ANCHORS', f'{p}.anchors', 'at least two anchors')
        for j, a in enumerate(an):
            if not all(k in a and len(a[k]) == 3 for k in ('co', 'handle_left', 'handle_right')): e('D_ANCHOR_FIELDS', f'{p}.anchors[{j}]', 'co, handle_left, handle_right (3 numbers each)')
    return errs


def to_curve_contract(design):
    """Stage-3 curve contract (`betterstyle.curve_contract/1`) with 'bezier_world' sources and authored provenance."""
    from contracts import CURVE_SCHEMA
    curves = [{'curve_id': c['curve_id'], 'source': {'kind': 'bezier_world', 'segments': bezier_segments(c['anchors'], c['closed']), 'closed': c['closed']}, 'surface_id': c['surface_id'], 'panel_id': c['panel_id'], 'placement_zone': c['placement_zone'],
               'sampler': copy.deepcopy(c['sampler']), 'bezel_id': c.get('bezel_id', 0), 'tangent': 'curve', 'provenance': {'kind': 'authored', 'design_id': design['design_id'], 'revision': design['revision'], 'authored_revision': c['authored_revision']}} for c in design['curves']]
    return {'schema': CURVE_SCHEMA, 'zones': copy.deepcopy(design['zones']), 'curves': curves}


def design_digest(design, surface_contract):
    """Digest of everything that determines the generated placements (curves, zones, panels, outward specs). Used to tie a .bsrs to the design revision that produced it."""
    blob = json.dumps({'curves': design['curves'], 'zones': design['zones'], 'panels': [(p['panel_id'], p['outward_direction'], p['surface_reference']) for p in surface_contract['panels']]}, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
