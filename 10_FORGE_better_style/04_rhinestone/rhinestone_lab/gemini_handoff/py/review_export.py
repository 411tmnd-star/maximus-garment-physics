"""review_export.py  turn a GenerationResult into the engineering-review bundle (JSON + panel GLB) for the Three.js review viewer. Coordinates are converted ONCE to glTF Y-up: (x, y, z) -> (x, z, -y).
The BSRS v1.1 file stays the authoritative record of transforms, sizes, outcomes and collision metadata; this bundle adds what the file cannot hold (measurements, footprint support, curves, panel
references, reasons in words, thresholds) and carries the stone ids so the viewer can cross-check the two. Rejected placements are always included."""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
import bpy
sys.path.insert(0, str(Path(__file__).resolve().parent))
import design_intent as DI, authoring as AU, placement_policy as PP, rhs_format as RF

def g(v): v = np.asarray(v, float); return [float(v[..., 0]), float(v[..., 2]), float(-v[..., 1])] if v.ndim == 1 else np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1).tolist()
def gl(v): return [round(float(x), 6) for x in g(v)]


def stone_index_map(run):
    """Position of each stone in the BSRS file (records with a transform, in order)."""
    idx = {}; k = 0
    for r in run.records:
        if r['transform'] is not None: idx[r['stone_id']] = k; k += 1
    return idx


def build_review_bundle(res, extra_meta=None, polyline_step=0.002, outward_samples=40):
    run, gs, design, sc = res.run, res.gs, res.design, res.surface_contract; fidx = stone_index_map(run); rng = np.random.default_rng(1)
    meta = {'schema': 'betterstyle.review_bundle/1', 'design_id': design['design_id'], 'design_revision': design['revision'], 'design_status': design['status'], 'disclaimer': design['disclaimer'], 'garment_id': design['garment_id'], 'garment_source_kind': design['garment_source_kind'],
            'design_digest': res.digest, 'thresholds_by_zone': res.thresholds_by_zone, 'outcomes': list(PP.OUTCOMES), 'reasons': list(PP.REASONS), 'design_preview_outcomes': ['ACCEPTED', 'ACCEPTED_WITH_CLEARANCE'], 'coordinates': 'glTF Y-up metres (converted from Blender Z-up)',
            'provisional_thresholds': True}
    meta.update(extra_meta or {})
    panels = []
    for pid, pn in gs.panels.items():
        od = gs.data[pn.objname]; ids = pn.face_ids; s = ids if len(ids) <= outward_samples else np.sort(rng.choice(ids, outward_samples, replace=False)); P = od.centre[s]; O = pn.outward(P); auth = DI.panel(sc, pid).get('outward_authorization', {})
        panels.append({'panel_id': pid, 'surface_id': pn.surface_id, 'layer_id': pn.layer_id, 'layer_rank': pn.layer_rank, 'zones': pn.zones, 'object': pn.objname, 'faces': int(len(ids)), 'outward_spec': pn.spec['outward_direction'], 'authorization_state': res.panel_states.get(pid, 'missing'),
                       'authorization': {k: auth.get(k) for k in ('status', 'authorized_by', 'authorized_at', 'method', 'verdict', 'note', 'acknowledged_ambiguity', 'evidence')}, 'outward_samples': [{'p': gl(p), 'd': gl(o)} for p, o in zip(P, O)]})
    curves = []
    for c in res.bundle.curves['curves']:
        path = res.paths[c['curve_id']]; n = int(min(4000, max(2, path.length / polyline_step))); s = np.linspace(0, path.length, n, endpoint=not path.closed); pts = path.point(s)
        tprev, _ = AU.tangent_preview(DI.get_curve(design, c['curve_id']), 0.015)
        an = DI.get_curve(design, c['curve_id'])['anchors']
        curves.append({'curve_id': c['curve_id'], 'panel_id': c['panel_id'], 'surface_id': c['surface_id'], 'zone': c['placement_zone'], 'closed': bool(path.closed), 'provenance': c.get('provenance'), 'length_m': float(path.length), 'polyline': g(pts), 'anchors': [gl(a['co']) for a in an],
                       'tangents': [{'p': gl(t['point']), 't': gl(t['tangent'])} for t in tprev], 'sampler': run.curve_reports[c['curve_id']]['sampler'], 'attempted': run.curve_reports[c['curve_id']]['attempted']})
    stones = []
    for r in run.records:
        tf = r['transform']; fp = r.get('footprint')
        stones.append({'id': r['stone_id'], 'file_index': fidx.get(r['stone_id']), 'curve_id': r['curve_id'], 'panel_id': r['panel_id'], 'surface_id': r['surface_id'], 'layer_id': r['layer_id'], 'zone': r['zone'], 'zone_id': r['zone_id'], 'row': r['row'], 'sample': r['sample'], 'sample_index': r['sample_index'],
                       'outcome': r['outcome'], 'reason': r['reason'], 'notes': r['notes'], 'ambiguities': r['ambiguities'], 'intended_size': r['intended_size'], 'realized_size': r['realized_size'], 'substitution_would_pass': r.get('substitution_would_pass', []),
                       'measurement': r['measurement'], 'footprint': fp, 'flags': r.get('flags', 0), 'collisions': r.get('collisions', 0), 'collision_severity': r.get('collision_severity', 0), 'snap_mm': r.get('snap_mm'), 'lift_mm': r.get('lift_mm'), 'normal_flipped': r.get('normal_flipped'), 'bezel_id': r['bezel_id'],
                       'position_world': gl(r['position_world']), 'transform': None if tf is None else {'base': gl(tf['base']), 'normal': gl(tf['normal']), 'tangent': gl(tf['tangent']), 'diameter': tf['diameter']}})
    coll = [{**{k: p[k] for k in ('stone_a', 'stone_b', 'category', 'severity', 'depth_mm', 'separation_mm', 'zones', 'panels', 'layers')}, 'location': gl(p['location'])} for p in (run.collisions['pairs'] if run.collisions else [])]
    return {'meta': meta, 'panels': panels, 'curves': curves, 'stones': stones, 'collisions': coll, 'blocked_curves': res.blocked, 'counts': run.counts(), 'panel_states': res.panel_states}


def export_review_geometry(res, path, target_faces=6000, context_objects=()):
    """One decimated mesh per panel (named panel__<panel_id>) plus optional context objects (context__<name>); glTF Y-up. For orientation in the viewer only: accuracy numbers come from the evaluator."""
    gs = res.gs; made = []; dg = bpy.context.evaluated_depsgraph_get()
    def mk(name, od, ids):
        m = np.zeros(od.nf, bool); m[ids] = True; tri = od.tris[m[od.tri_poly]]
        if len(tri) == 0: return
        used, inv = np.unique(tri, return_inverse=True); me = bpy.data.meshes.new(name); me.from_pydata(od.W[used].tolist(), [], inv.reshape(-1, 3).tolist()); me.update(); ob = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(ob)
        if len(tri) > target_faces:
            md = ob.modifiers.new('dec', 'DECIMATE'); md.ratio = target_faces / len(tri); bpy.context.view_layer.update(); ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get()); me2 = bpy.data.meshes.new_from_object(ev); ob.modifiers.remove(md); ob.data = me2
        made.append(ob)
    for pid, pn in gs.panels.items(): mk(f'panel__{pid}', gs.data[pn.objname], pn.face_ids)
    for nm in context_objects:
        if nm not in gs.data: gs.data[nm] = SF_object(nm)
        mk(f'context__{nm}', gs.data[nm], np.arange(gs.data[nm].nf))
    bpy.ops.object.select_all(action='DESELECT')
    for o in made: o.select_set(True)
    bpy.context.view_layer.objects.active = made[0]; bpy.ops.export_scene.gltf(filepath=str(path), export_format='GLB', export_yup=True, use_selection=True)
    names = [o.name for o in made]
    for o in made: bpy.data.objects.remove(o, do_unlink=True)
    return names


def SF_object(name):
    import surfaces as SF
    return SF.ObjectData(bpy.data.objects[name])


def write_bundle(bundle, path):
    def j(o):
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, set): return sorted(o)
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(bundle, default=j, separators=(',', ':')))
