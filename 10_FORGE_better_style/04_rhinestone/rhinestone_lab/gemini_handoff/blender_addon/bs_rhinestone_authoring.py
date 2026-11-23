bl_info = {'name': 'Better Style rhinestone authoring (engineering)', 'author': 'rhinestone_lab', 'version': (0, 4, 0), 'blender': (4, 2, 0), 'location': 'View3D > Sidebar > Rhinestones', 'category': 'Object',
           'description': 'Author rhinestone placement curves on garment panels with explicit panel references, evaluate them with the Stage-3 engine, export an engineering review bundle. Interim engineering tool.'}
import json, sys
from pathlib import Path
import bpy
from bpy.props import StringProperty, BoolProperty, EnumProperty, FloatProperty, IntProperty
_LIB = Path(__file__).resolve().parent.parent / 'py'
if str(_LIB) not in sys.path: sys.path.insert(0, str(_LIB))
import numpy as np
import design_intent as DI, authoring as AU, surfaces as SF, review_export as RX, rhs_format as RF, contracts as CT

_CACHE = {'gs': None, 'key': None, 'objdata': {}}


def _objects(): return {o.name: o for o in bpy.data.objects}
def _p(scene, prop): return Path(bpy.path.abspath(getattr(scene, prop)))
def _load(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default
def _save(path, data): Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_text(json.dumps(data, indent=1))
def _surface(scene): return _load(_p(scene, 'bsr_surface_path'), {'schema': CT.SURFACE_SCHEMA, 'garment_id': scene.bsr_garment_id, 'source_kind': scene.bsr_source_kind, 'units': 'm', 'panels': []})
def _design(scene):
    d = _load(_p(scene, 'bsr_design_path'))
    return d or DI.new_design(scene.bsr_design_id, scene.bsr_garment_id, scene.bsr_source_kind, author=scene.bsr_operator)
def _gs(scene, contract, rebuild=False):
    key = json.dumps([(p['panel_id'], p['outward_direction'], p['surface_reference']) for p in contract['panels']], sort_keys=True)
    if rebuild or _CACHE['gs'] is None or _CACHE['key'] != key: _CACHE['gs'] = SF.GarmentSurfaces(contract, _objects(), None, tuple(x for x in scene.bsr_occluders.split(',') if x)); _CACHE['key'] = key
    return _CACHE['gs']


class BSR_OT_panel_assign(bpy.types.Operator):
    bl_idname = 'bsr.panel_assign'; bl_label = 'Assign panel'; bl_options = {'REGISTER', 'UNDO'}
    panel_id: StringProperty(); surface_id: StringProperty(); layer_id: StringProperty(default='shell_outer'); zones: StringProperty(default='test_zone'); object_name: StringProperty(); vertex_group: StringProperty()
    facing: EnumProperty(items=[('outward', 'outward', ''), ('inward', 'inward', ''), ('none', 'none', '')], default='outward'); layer_rank: IntProperty(default=0)
    outward_kind: EnumProperty(items=[('reference_body', 'reference body (suggestion)', ''), ('vector', 'vector', ''), ('radial_axis', 'radial axis', ''), ('from_point', 'from point', ''), ('toward_point', 'toward point', '')], default='reference_body')
    outward_params: StringProperty(default='{}', description='JSON for the non-reference kinds, e.g. {"v":[0,0,1],"frame":"object_local"}'); reference_object: StringProperty(default='')
    def execute(self, ctx):
        sc = ctx.scene; contract = _surface(sc); name = self.object_name or (ctx.active_object.name if ctx.active_object else '')
        if not name or name not in bpy.data.objects: self.report({'ERROR'}, 'no such object'); return {'CANCELLED'}
        select = ([{'kind': 'vertex_group', 'name': self.vertex_group, 'min_weight': 0.5}] if self.vertex_group else []) + ([{'kind': 'faces_facing_' + self.facing}] if self.facing != 'none' else [])
        if self.outward_kind == 'reference_body':
            if not self.reference_object: self.report({'ERROR'}, 'reference_body needs a reference object'); return {'CANCELLED'}
            spec = AU.reference_suggestion(self.reference_object)
        else: spec = {'kind': self.outward_kind, **json.loads(self.outward_params)}
        panel = {'panel_id': self.panel_id, 'surface_id': self.surface_id, 'layer_id': self.layer_id, 'layer_rank': self.layer_rank, 'outward_direction': spec, 'placement_zone': [z for z in self.zones.split(',') if z],
                 'surface_reference': {'object': name, **({'select': select} if select else {})}, 'outward_authorization': {'status': 'suggested', 'method': 'reference_body' if self.outward_kind == 'reference_body' else 'operator_specified', 'verdict': 'AMBIGUOUS_REVIEW_REQUIRED', 'evidence': {'note': 'not inspected yet'}, 'spec_hash': DI.spec_hash(spec)}}
        contract['panels'] = [q for q in contract['panels'] if q['panel_id'] != self.panel_id] + [panel]; errs = CT.validate_surface_contract(contract)
        if errs: self.report({'ERROR'}, f"{errs[0]['code']} {errs[0]['path']}"); return {'CANCELLED'}
        _save(_p(sc, 'bsr_surface_path'), contract); ob = bpy.data.objects[name]; ob['bs_panels'] = sorted(set(list(ob.get('bs_panels', [])) + [self.panel_id])); _CACHE['gs'] = None
        self.report({'INFO'}, f'panel {self.panel_id} assigned (outward reference NOT yet authorized)'); return {'FINISHED'}


class BSR_OT_panel_inspect(bpy.types.Operator):
    bl_idname = 'bsr.panel_inspect'; bl_label = 'Inspect outward reference'; bl_options = {'REGISTER'}
    panel_id: StringProperty()
    def execute(self, ctx):
        sc = ctx.scene; contract = _surface(sc); p = DI.panel(contract, self.panel_id); spec = p['outward_direction']
        try:
            if spec['kind'] == 'away_from_surface': rep = AU.suggest_panel_reference(contract, self.panel_id, spec['reference_object'], _objects(), cache=_CACHE['objdata'])
            else:
                rep = AU.inspect_panel_outward(contract, self.panel_id, _objects(), cache=_CACHE['objdata']); DI.set_suggestion(contract, self.panel_id, spec, 'operator_specified', {k: rep[k] for k in ('selected_agree', 'selected_ambiguous_fraction', 'selected_ambiguous_faces', 'raw_facing_outward', 'raw_facing_inward', 'mixed_layers', 'notes')}, rep['verdict'])
        except (AU.AuthoringError, SF.ContractGeometryError) as e: self.report({'ERROR'}, str(e)); return {'CANCELLED'}
        _save(_p(sc, 'bsr_surface_path'), contract); sc['bsr_last_inspection'] = json.dumps(rep)
        coll = bpy.data.collections.get('BSR_REVIEW') or bpy.data.collections.new('BSR_REVIEW')
        if coll.name not in sc.collection.children: sc.collection.children.link(coll)
        for q in rep['ambiguous_locations_world']:
            e = bpy.data.objects.new(f'BSR_AMBIG_{self.panel_id}', None); e.empty_display_type = 'SPHERE'; e.empty_display_size = 0.004; e.location = q; coll.objects.link(e)
        self.report({'INFO'}, f"{self.panel_id}: {rep['verdict']} (agree {rep['selected_agree']:.3f}, ambiguous {rep['selected_ambiguous_fraction']:.3f}); NOT authorized"); return {'FINISHED'}


class BSR_OT_panel_authorize(bpy.types.Operator):
    bl_idname = 'bsr.panel_authorize'; bl_label = 'Authorize panel outward reference'; bl_options = {'REGISTER', 'UNDO'}
    panel_id: StringProperty(); operator_name: StringProperty(); note: StringProperty(); acknowledge_ambiguity: BoolProperty(default=False)
    def execute(self, ctx):
        sc = ctx.scene; contract = _surface(sc)
        try: DI.authorize_outward(contract, self.panel_id, self.operator_name, self.note, self.acknowledge_ambiguity)
        except (DI.AuthorizationError, KeyError) as e: self.report({'ERROR'}, f'not authorized: {e}'); return {'CANCELLED'}
        _save(_p(sc, 'bsr_surface_path'), contract); self.report({'INFO'}, f'{self.panel_id} outward reference authorized by {self.operator_name}'); return {'FINISHED'}


class BSR_OT_panel_reject(bpy.types.Operator):
    bl_idname = 'bsr.panel_reject'; bl_label = 'Reject panel outward reference'; panel_id: StringProperty(); operator_name: StringProperty(); note: StringProperty()
    def execute(self, ctx):
        sc = ctx.scene; contract = _surface(sc); DI.reject_outward(contract, self.panel_id, self.operator_name, self.note); _save(_p(sc, 'bsr_surface_path'), contract); return {'FINISHED'}


class BSR_OT_zone_set(bpy.types.Operator):
    bl_idname = 'bsr.zone_set'; bl_label = 'Define placement zone'; bl_options = {'REGISTER', 'UNDO'}
    zone: StringProperty(); zone_id: IntProperty(default=0); thresholds_json: StringProperty(default='{}', description='per-zone threshold overrides (provisional)'); substitution_to: StringProperty(default='', description='csv of smaller sizes this zone PERMITS (empty = substitution not permitted)')
    def execute(self, ctx):
        sc = ctx.scene; design = _design(sc); sub = {'enabled': bool(self.substitution_to), **({'allowed_to': [s for s in self.substitution_to.split(',') if s]} if self.substitution_to else {})}
        DI.set_zone(design, self.zone, self.zone_id, json.loads(self.thresholds_json) or None, sub, sc.bsr_operator); _save(_p(sc, 'bsr_design_path'), design); return {'FINISHED'}


def _make_curve_object(c):
    cu = bpy.data.curves.new('BSR_' + c['curve_id'], 'CURVE'); cu.dimensions = '3D'; sp = cu.splines.new('BEZIER'); sp.bezier_points.add(len(c['anchors']) - 1)
    for bp, a in zip(sp.bezier_points, c['anchors']): bp.handle_left_type = bp.handle_right_type = 'FREE'; bp.co = a['co']; bp.handle_left = a['handle_left']; bp.handle_right = a['handle_right']
    sp.use_cyclic_u = c['closed']; name = 'BSR_' + c['curve_id']; old = bpy.data.objects.get(name)
    if old: bpy.data.objects.remove(old, do_unlink=True)
    ob = bpy.data.objects.new(name, cu); bpy.context.scene.collection.objects.link(ob)
    for k, v in (('bs_curve_id', c['curve_id']), ('bs_panel_id', c['panel_id']), ('bs_surface_id', c['surface_id']), ('bs_zone', c['placement_zone'])): ob[k] = v
    return ob


class BSR_OT_curve_create(bpy.types.Operator):
    bl_idname = 'bsr.curve_create'; bl_label = 'Create placement curve on panel'; bl_options = {'REGISTER', 'UNDO'}
    curve_id: StringProperty(); panel_id: StringProperty(); zone: StringProperty(); points_json: StringProperty(description='JSON [[x,y,z],...] approximate world points; they are snapped onto the panel'); closed: BoolProperty(default=False)
    size_pattern: StringProperty(default='SS16'); gap_m: FloatProperty(default=0.0004, precision=5); mode: EnumProperty(items=[('fit', 'fit', ''), ('fixed', 'fixed', '')], default='fit'); bezel_id: IntProperty(default=0); note: StringProperty()
    handle_mode: EnumProperty(items=[('auto', 'auto', ''), ('manual', 'manual', '')], default='auto'); rows_json: StringProperty(default='[0.0]')
    def execute(self, ctx):
        sc = ctx.scene; contract = _surface(sc); design = _design(sc)
        if self.zone not in design['zones']: self.report({'ERROR'}, f'zone {self.zone} is not defined in the design'); return {'CANCELLED'}
        try: allowed = DI.panel(contract, self.panel_id)['placement_zone']
        except KeyError: self.report({'ERROR'}, f'unknown panel {self.panel_id}'); return {'CANCELLED'}
        if self.zone not in allowed: self.report({'ERROR'}, f'zone {self.zone} is not allowed on panel {self.panel_id} (allowed: {allowed})'); return {'CANCELLED'}
        try:
            gs = _gs(sc, contract); pat = [s for s in self.size_pattern.split(',') if s]
            c = AU.author_curve_on_panel(design, gs, self.panel_id, self.curve_id, json.loads(self.points_json), self.zone, self.closed, {'size_pattern': pat, 'gap_m': {s: self.gap_m for s in ('SS10', 'SS16', 'SS20')}, 'mode': self.mode, 'lateral_offsets_m': json.loads(self.rows_json)}, self.bezel_id, self.note, sc.bsr_operator, self.handle_mode)
        except (AU.AuthoringError, ValueError, KeyError) as e: self.report({'ERROR'}, str(e)); return {'CANCELLED'}
        _make_curve_object(c); _save(_p(sc, 'bsr_design_path'), design); self.report({'INFO'}, f'curve {self.curve_id} authored on {self.panel_id} (revision {design["revision"]})'); return {'FINISHED'}


class BSR_OT_curve_sync(bpy.types.Operator):
    bl_idname = 'bsr.curve_sync'; bl_label = 'Sync edited curve into the design'; bl_options = {'REGISTER', 'UNDO'}
    curve_id: StringProperty(); resnap: BoolProperty(default=True)
    def execute(self, ctx):
        sc = ctx.scene; design = _design(sc); contract = _surface(sc); ob = bpy.data.objects.get('BSR_' + self.curve_id)
        if ob is None: self.report({'ERROR'}, 'no curve object'); return {'CANCELLED'}
        c = DI.get_curve(design, self.curve_id); mw = ob.matrix_world; bps = ob.data.splines[0].bezier_points
        if len(bps) != len(c['anchors']): self.report({'ERROR'}, 'adding or removing control points: re-create the curve (anchors must match)'); return {'CANCELLED'}
        c['handle_mode'] = 'manual'; changed = 0
        for i, (bp, a) in enumerate(zip(bps, c['anchors'])):
            co, hl, hr = [list(map(float, mw @ v)) for v in (bp.co, bp.handle_left, bp.handle_right)]
            if not (np.allclose(co, a['co'], atol=1e-9) and np.allclose(hl, a['handle_left'], atol=1e-9) and np.allclose(hr, a['handle_right'], atol=1e-9)): a.update(co=co, handle_left=hl, handle_right=hr); changed += 1
        if changed: c['authored_revision'] = design['revision'] + 1; DI._bump(design, sc.bsr_operator, f'curve {self.curve_id}: {changed} control points edited in Blender')
        if self.resnap and changed:
            try: AU.resnap_curve(design, _gs(sc, contract), self.curve_id, sc.bsr_operator)
            except AU.AuthoringError as e: self.report({'ERROR'}, str(e)); return {'CANCELLED'}
        _save(_p(sc, 'bsr_design_path'), design); self.report({'INFO'}, f'{changed} control points synced'); return {'FINISHED'}


class BSR_OT_curve_spacing(bpy.types.Operator):
    bl_idname = 'bsr.curve_spacing'; bl_label = 'Set curve spacing'; bl_options = {'REGISTER', 'UNDO'}
    curve_id: StringProperty(); size_pattern: StringProperty(default=''); gap_m: FloatProperty(default=-1.0, precision=5); mode: StringProperty(default=''); inset_start_m: FloatProperty(default=-1.0); inset_end_m: FloatProperty(default=-1.0); rows_json: StringProperty(default='')
    def execute(self, ctx):
        sc = ctx.scene; design = _design(sc); kw = {}
        if self.size_pattern: kw['size_pattern'] = [s for s in self.size_pattern.split(',') if s]
        if self.gap_m >= 0: kw['gap_m'] = {s: self.gap_m for s in ('SS10', 'SS16', 'SS20')}
        if self.mode: kw['mode'] = self.mode
        if self.inset_start_m >= 0: kw['inset_start_m'] = self.inset_start_m
        if self.inset_end_m >= 0: kw['inset_end_m'] = self.inset_end_m
        if self.rows_json: kw['lateral_offsets_m'] = json.loads(self.rows_json)
        DI.set_spacing(design, self.curve_id, sc.bsr_operator, **kw); _save(_p(sc, 'bsr_design_path'), design); return {'FINISHED'}


class BSR_OT_curve_assign(bpy.types.Operator):
    bl_idname = 'bsr.curve_assign'; bl_label = 'Reassign curve to panel'; bl_options = {'REGISTER', 'UNDO'}
    curve_id: StringProperty(); panel_id: StringProperty(); zone: StringProperty()
    def execute(self, ctx):
        sc = ctx.scene; design = _design(sc); contract = _surface(sc)
        try: p = DI.panel(contract, self.panel_id)
        except KeyError: self.report({'ERROR'}, 'unknown panel'); return {'CANCELLED'}
        DI.reassign_panel(design, self.curve_id, self.panel_id, p['surface_id'], self.zone, sc.bsr_operator); errs = DI.validate_design(design, contract)
        if errs: self.report({'ERROR'}, f"{errs[0]['code']}: {errs[0]['message']}"); return {'CANCELLED'}
        _save(_p(sc, 'bsr_design_path'), design); return {'FINISHED'}


class BSR_OT_tangent_preview(bpy.types.Operator):
    bl_idname = 'bsr.tangent_preview'; bl_label = 'Preview tangents and outward reference'; curve_id: StringProperty(); reverse: BoolProperty(default=False); length_m: FloatProperty(default=0.006)
    def execute(self, ctx):
        sc = ctx.scene; design = _design(sc); contract = _surface(sc); c = DI.get_curve(design, self.curve_id)
        prev, _ = AU.tangent_preview(c, 0.012); coll = bpy.data.collections.get('BSR_PREVIEW') or bpy.data.collections.new('BSR_PREVIEW')
        if coll.name not in sc.collection.children: sc.collection.children.link(coll)
        for nm in (f'BSR_TAN_{self.curve_id}', f'BSR_OUT_{self.curve_id}'):
            if nm in bpy.data.objects: bpy.data.objects.remove(bpy.data.objects[nm], do_unlink=True)
            if nm in bpy.data.meshes: bpy.data.meshes.remove(bpy.data.meshes[nm])
        sgn = -1.0 if self.reverse else 1.0; v = []; e = []
        for t in prev: p = np.array(t['point']); q = p + np.array(t['tangent']) * sgn * self.length_m; e.append((len(v), len(v) + 1)); v += [tuple(p), tuple(q)]
        me = bpy.data.meshes.new(f'BSR_TAN_{self.curve_id}'); me.from_pydata(v, e, []); coll.objects.link(bpy.data.objects.new(f'BSR_TAN_{self.curve_id}', me))
        gs = _gs(sc, contract); outs = AU.outward_preview(gs, c['panel_id'], [t['point'] for t in prev]); v = []; e = []
        for o in outs: p = np.array(o['point']); q = p + np.array(o['outward']) * self.length_m; e.append((len(v), len(v) + 1)); v += [tuple(p), tuple(q)]
        me = bpy.data.meshes.new(f'BSR_OUT_{self.curve_id}'); me.from_pydata(v, e, []); coll.objects.link(bpy.data.objects.new(f'BSR_OUT_{self.curve_id}', me)); return {'FINISHED'}


class BSR_OT_generate(bpy.types.Operator):
    bl_idname = 'bsr.generate'; bl_label = 'Generate and evaluate placements'; bl_options = {'REGISTER'}
    rebuild_surfaces: BoolProperty(default=False); strict_authorization: BoolProperty(default=False); name: StringProperty(default='placements')
    def execute(self, ctx):
        sc = ctx.scene; design = _design(sc); contract = _surface(sc); out = _p(sc, 'bsr_output_dir'); out.mkdir(parents=True, exist_ok=True)
        try:
            if self.rebuild_surfaces: _CACHE['objdata'].clear()
            gs = _gs(sc, contract, self.rebuild_surfaces); cfg = AU.PV.EngineConfig(keep_footprint=True, row_offset_mode='surface', projection_mode='normal'); res = AU.generate(design, contract, _objects(), cfg, None, tuple(x for x in sc.bsr_occluders.split(',') if x), gs, self.strict_authorization)
        except (AU.AuthoringError, SF.ContractGeometryError) as e: self.report({'ERROR'}, str(e)); return {'CANCELLED'}
        blob, M, P = res.run.to_bsrs(); (out / f'{self.name}.bsrs').write_bytes(blob); sc['bsr_last_counts'] = json.dumps(res.run.counts()); sc['bsr_last_blocked'] = json.dumps(res.blocked); _CACHE['last'] = res
        coll = bpy.data.collections.get('BSR_STONES') or bpy.data.collections.new('BSR_STONES')
        if coll.name not in sc.collection.children: sc.collection.children.link(coll)
        for o in list(coll.objects): bpy.data.objects.remove(o, do_unlink=True)
        for oc in PP_OUTCOMES:
            pts = [tuple(r['transform']['base']) for r in res.run.records if r['transform'] is not None and r['outcome'] == oc]
            if pts: me = bpy.data.meshes.new(f'BSR_{oc}'); me.from_pydata(pts, [], []); coll.objects.link(bpy.data.objects.new(me.name, me))
        self.report({'INFO'}, f"{res.run.counts()['attempted']} stones: {res.run.counts()['outcomes']}; blocked curves {len(res.blocked)}"); return {'FINISHED'}


PP_OUTCOMES = ('ACCEPTED', 'ACCEPTED_WITH_CLEARANCE', 'FLOATING_REVIEW', 'SIZE_SUBSTITUTION_CANDIDATE', 'REJECTED')


class BSR_OT_export_review(bpy.types.Operator):
    bl_idname = 'bsr.export_review'; bl_label = 'Export engineering review bundle'; name: StringProperty(default='review'); context_objects: StringProperty(default='')
    def execute(self, ctx):
        res = _CACHE.get('last')
        if res is None: self.report({'ERROR'}, 'generate first'); return {'CANCELLED'}
        out = _p(ctx.scene, 'bsr_output_dir'); out.mkdir(parents=True, exist_ok=True); blob, M, P = res.run.to_bsrs(); (out / f'{self.name}.bsrs').write_bytes(blob)
        RX.write_bundle(RX.build_review_bundle(res), out / f'{self.name}.review.json'); RX.export_review_geometry(res, out / f'{self.name}.geometry.glb', context_objects=tuple(x for x in self.context_objects.split(',') if x)); return {'FINISHED'}


class BSR_PT_main(bpy.types.Panel):
    bl_label = 'Rhinestone authoring'; bl_idname = 'BSR_PT_main'; bl_space_type = 'VIEW_3D'; bl_region_type = 'UI'; bl_category = 'Rhinestones'
    def draw(self, ctx):
        l = self.layout; sc = ctx.scene; l.label(text='INTERIM ENGINEERING TOOL: not the approved design', icon='INFO')
        for p in ('bsr_operator', 'bsr_garment_id', 'bsr_design_id', 'bsr_surface_path', 'bsr_design_path', 'bsr_output_dir', 'bsr_occluders'): l.prop(sc, p)
        b = l.box(); b.label(text='0 Zone'); b.operator('bsr.zone_set'); b = l.box(); b.label(text='1 Panel'); b.operator('bsr.panel_assign'); b.operator('bsr.panel_inspect'); b.operator('bsr.panel_authorize'); b.operator('bsr.panel_reject')
        b = l.box(); b.label(text='2 Curves (they name a panel; the panel owns outward)'); b.operator('bsr.curve_create'); b.operator('bsr.curve_sync'); b.operator('bsr.curve_spacing'); b.operator('bsr.curve_assign'); b.operator('bsr.tangent_preview')
        b = l.box(); b.label(text='3 Evaluate'); b.operator('bsr.generate'); b.operator('bsr.export_review')


_CLASSES = (BSR_OT_zone_set, BSR_OT_panel_assign, BSR_OT_panel_inspect, BSR_OT_panel_authorize, BSR_OT_panel_reject, BSR_OT_curve_create, BSR_OT_curve_sync, BSR_OT_curve_spacing, BSR_OT_curve_assign, BSR_OT_tangent_preview, BSR_OT_generate, BSR_OT_export_review, BSR_PT_main)
_PROPS = {'bsr_operator': StringProperty(name='Operator', default=''), 'bsr_garment_id': StringProperty(name='Garment id', default='garment'), 'bsr_design_id': StringProperty(name='Design id', default='design'), 'bsr_source_kind': StringProperty(default='engine_archetype'),
          'bsr_surface_path': StringProperty(name='Surface contract', subtype='FILE_PATH', default='//surface.json'), 'bsr_design_path': StringProperty(name='Design intent', subtype='FILE_PATH', default='//design.json'),
          'bsr_output_dir': StringProperty(name='Output dir', subtype='DIR_PATH', default='//out'), 'bsr_occluders': StringProperty(name='Extra occluder objects (csv)', default='')}


def register():
    for c in _CLASSES: bpy.utils.register_class(c)
    for k, v in _PROPS.items(): setattr(bpy.types.Scene, k, v)


def unregister():
    for c in reversed(_CLASSES): bpy.utils.unregister_class(c)
    for k in _PROPS: delattr(bpy.types.Scene, k)


if __name__ == '__main__': register()
