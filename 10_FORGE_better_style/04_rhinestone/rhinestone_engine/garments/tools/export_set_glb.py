"""export_set_glb.py - web 3D export of the BST two-piece (shirt + trousers + stones) as ONE Draco-compressed GLB per colourway.

usage: python -B export_set_glb.py --shirt-blend B --trousers-blend T --shirt-stones S.npz --trousers-stones S.npz --colour blue|red --out O.glb
                                   [--label front_label.png] [--decimate tee=0.2,collar=0.3,leg=0.3] [--tex tee=2048,collar=1024,leg=1024,small=512]

Same garment, materials and stone layout as tools/render_set.py (the published catalogue renders), so the 3D view and the stills agree:
  materials   the render_sku fabric graph is built as render_set.py builds it (albedo = the brand colour, not the Cycles-rig
              calibration: see the comment at `body`), then BAKED (Cycles) into a 0..1 atlas UV (UVMap_Bake) on the full-resolution mesh into
              glTF PBR maps per object: base colour (sRGB), roughness (G of the metallicRoughness texture) and a tangent-space normal map
              that carries the knit, seam and label relief. Objects without a fabric UV map (buttons, belt loops, fly) get flat PBR values.
  geometry    the dense draped panels are decimated (collapse) AFTER the bake; UVs survive the collapse, so the baked maps still fit.
              Stone seating is measured on the full-res and on the decimated surface and both are written to the report.
  stones      unchanged positions/frames/diameters from the solver's .npz (P, N, T, R, cls). Each stone is a linked duplicate of one
              16-sided chaton mesh per colour class (render_set.chaton), so the exporter writes EXT_mesh_gpu_instancing: three.js mounts
              them as InstancedMesh. Stones are metallic PBR facets, not refractive crystal (same limit as the renders).
  export      GLB, Y-up, KHR_draco_mesh_compression, EXT_texture_webp. A <out>.report.json records inputs (sha256), counts and seating.
Nothing outside --out (and its report) is written; the input .blend files are opened read-only and never saved."""
import sys, os, math, json, time, argparse, hashlib
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
import bpy, bmesh
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

HE = Path(os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/human_engine/src_catalog/better_style_catalog')))
sys.path.insert(0, str(HE / 'engine')); sys.path.insert(0, str(HE / 'engine' / 'components'))
import render_sku as RS

ap = argparse.ArgumentParser()
for k in ('--shirt-blend', '--trousers-blend', '--shirt-stones', '--trousers-stones', '--out'):
    ap.add_argument(k, required=True)
ap.add_argument('--colour', default='blue'); ap.add_argument('--label', default=None)
ap.add_argument('--decimate', default='tee=0.2,collar=0.3,leg=0.3'); ap.add_argument('--tex', default='tee=2048,collar=1024,leg=1024,small=512')
ap.add_argument('--bake-samples', type=int, default=4)
a = ap.parse_args()
DEC = {k: float(v) for k, v in (s.split('=') for s in a.decimate.split(','))}
TEX = {k: int(v) for k, v in (s.split('=') for s in a.tex.split(','))}
OUT = Path(a.out).resolve(); OUT.parent.mkdir(parents=True, exist_ok=True)
T0 = time.time()
def log(*x): print('[%5.0fs]' % (time.time() - T0), *x, flush=True)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

COL = {  # identical to render_set.py
    'blue': dict(body='#193D7C', main=(0.86, 0.90, 1.0), accent=(1.0, 0.74, 0.30), button='#0E2A5D', cal='blue'),
    'red': dict(body='#A3121A', main=(1.0, 0.72, 0.28), accent=(0.95, 0.93, 0.88), button='#6E0C12', cal=None),
}[a.colour]
sku = json.loads((HE / 'configs/skus/BS-CAMP-001.json').read_text()); look = dict(sku['material']); knit = look['knit']; rd = sku['render']
# Albedo: the brand colour itself (linear), NOT the render calibration. calibration/BS-CAMP-001.json darkens the albedo (blue: 0.0019,
# 0.0046, 0.040 against the brand's 0.0097, 0.047, 0.20) to cancel render_sku's strong studio lights and AgX +0.5 EV; that is a property of
# the Cycles rig, not of the cloth. A glTF material is lit by the viewer's neutral environment, so it carries the true colour.
body = np.asarray(RS.hex_to_lin(COL['body']), float)

# ---------------------------------------------------------------- scene: shirt blend + trousers garment appended (same body frame)
bpy.ops.wm.open_mainfile(filepath=a.shirt_blend); sc = bpy.context.scene
gcol = bpy.data.collections['Garment']
if 'Mannequin' in bpy.data.collections:
    for o in list(bpy.data.collections['Mannequin'].objects): bpy.data.objects.remove(o)
shirt_objs = [o for o in gcol.objects if o.type == 'MESH']
with bpy.data.libraries.load(a.trousers_blend, link=False) as (src, dst):
    dst.objects = [n for n in src.objects]
tcol = bpy.data.collections.new('Trousers'); sc.collection.children.link(tcol)
trouser_objs = []
for o in dst.objects:
    if o is not None and o.type == 'MESH':
        tcol.objects.link(o); trouser_objs.append(o)
log('scene: shirt', [o.name for o in shirt_objs], '| trousers', len(trouser_objs), 'objects')

# ---------------------------------------------------------------- materials, exactly as render_set.py
mats = {m.name: m for ob in shirt_objs for m in ob.data.materials if m}
art = None
if a.label:
    art = {'front': (Path(a.label), Path(a.label).with_name(Path(a.label).stem + '_height.png'))}
RS.build_fabric_material(bpy, mats['Main_Body'], body, look, knit, False, art)
if look.get('seam_groove', True): RS.overlay_line_mask(mats['Main_Body'], RS.seam_mask(mats['Main_Body']), darken=look.get('seam_darken', 0.22), relief=-0.8)
RS.build_fabric_material(bpy, mats['Collar_Rib'], body, look, knit, True)
if 'Placket' in mats:
    RS.build_fabric_material(bpy, mats['Placket'], body, look, knit, True, uv_name='UVMap_Placket')
    RS.overlay_line_mask(mats['Placket'], RS.placket_stitch_mask(mats['Placket'], 0.43, 0.032), darken=0.16, relief=0.5)
for nm, hole in (('Button', False), ('Button_Hole', True)):
    if nm in mats: RS.build_button_material(bpy, mats[nm], RS.hex_to_lin(COL['button']), hole)
tmats = {m.name: m for ob in trouser_objs for m in ob.data.materials if m}
for nm, m in tmats.items():
    if nm.startswith('Trouser'): RS.build_fabric_material(bpy, m, body, look, knit, nm.endswith('Trim'))
    elif nm == 'Button': RS.build_button_material(bpy, m, RS.hex_to_lin(COL['button']), False)

# ---------------------------------------------------------------- bake per object -> glTF PBR material
r = sc.render; r.engine = 'CYCLES'; c = sc.cycles; c.device = 'CPU'; c.samples = a.bake_samples; c.use_denoising = False
sc.render.bake.margin = 8; sc.render.bake.use_clear = True


def bake_uv(ob):
    """UV map that carries the fabric pattern for this object (None: flat PBR)."""
    for n in ('UVMap_Fabric', 'UVMap_Placket'):
        if n in ob.data.uv_layers: return n
    return None


def tex_size(ob):
    n = ob.name.lower()
    if n == 'tee': return TEX['tee']
    if n == 'collar': return TEX['collar']
    if n.startswith('leg_'): return TEX['leg']
    return TEX['small']


def principled_values(m):
    b = next((n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if b is None: return (0.5, 0.5, 0.5), 0.8
    bc = b.inputs['Base Color']; col = tuple(bc.default_value[:3])
    if bc.is_linked:                                   # fabric graph: base colour comes from the body RGB node
        rgb = next((n for n in m.node_tree.nodes if n.type == 'RGB'), None)
        if rgb: col = tuple(rgb.outputs[0].default_value[:3])
    return col, float(b.inputs['Roughness'].default_value)


BAKE_UV = 'UVMap_Bake'


def add_bake_uv(ob, uv, pad=0.004):
    """The pattern UV maps are in pattern units (Tee u 0..5.25, legs v 0.25..3.48, placket v to 1.43), not a 0..1 atlas, so a bake into
    them only writes the [0,1] tile and everything else samples wrapped, unbaked texels. Bake into a copy fitted to [pad, 1-pad] per axis
    instead; the procedural shader keeps reading the original map, the exported material reads this one."""
    src = ob.data.uv_layers[uv]; a = np.empty(len(src.data) * 2); src.data.foreach_get('uv', a); a = a.reshape(-1, 2)
    lo, hi = a.min(0), a.max(0); a = pad + (a - lo) / np.maximum(hi - lo, 1e-9) * (1 - 2 * pad)
    dst = ob.data.uv_layers.new(name=BAKE_UV); dst.data.foreach_set('uv', a.ravel())
    return dict(source_range=[lo.round(4).tolist(), hi.round(4).tolist()])


def bake(ob, uv, size):
    imgs = {k: bpy.data.images.new(f'{ob.name}_{k}', size, size, alpha=False, float_buffer=False) for k in ('base', 'rough', 'normal')}
    imgs['rough'].colorspace_settings.name = 'Non-Color'; imgs['normal'].colorspace_settings.name = 'Non-Color'
    ob.data.uv_layers.active = ob.data.uv_layers[BAKE_UV]; ob.data.uv_layers[BAKE_UV].active_render = True   # tangents for the normal bake
    for o in bpy.context.view_layer.objects: o.select_set(False)
    ob.select_set(True); bpy.context.view_layer.objects.active = ob
    for kind, btype, kw in (('base', 'DIFFUSE', dict(pass_filter={'COLOR'})), ('rough', 'ROUGHNESS', {}), ('normal', 'NORMAL', dict(normal_space='TANGENT'))):
        nodes = []
        for m in {m for m in ob.data.materials if m}:
            n = m.node_tree.nodes.new('ShaderNodeTexImage'); n.image = imgs[kind]; n.select = True; m.node_tree.nodes.active = n; nodes.append((m, n))
        t = time.time(); bpy.ops.object.bake(type=btype, margin=8, use_clear=True, uv_layer=BAKE_UV, **kw)
        for m, n in nodes: m.node_tree.nodes.remove(n)
        log(f'  bake {ob.name} {kind} {size}px {time.time() - t:.0f}s')
    return imgs


def gltf_material(name, imgs, uv):
    m = bpy.data.materials.new(name); m.use_nodes = True; N = m.node_tree.nodes; L = m.node_tree.links; N.clear()
    out = N.new('ShaderNodeOutputMaterial'); b = N.new('ShaderNodeBsdfPrincipled'); L.new(b.outputs['BSDF'], out.inputs['Surface'])
    uvn = N.new('ShaderNodeUVMap'); uvn.uv_map = uv
    tb = N.new('ShaderNodeTexImage'); tb.image = imgs['base']; L.new(uvn.outputs['UV'], tb.inputs['Vector']); L.new(tb.outputs['Color'], b.inputs['Base Color'])
    tr = N.new('ShaderNodeTexImage'); tr.image = imgs['rough']; L.new(uvn.outputs['UV'], tr.inputs['Vector'])
    sp = N.new('ShaderNodeSeparateColor'); L.new(tr.outputs['Color'], sp.inputs['Color']); L.new(sp.outputs['Green'], b.inputs['Roughness'])
    tn = N.new('ShaderNodeTexImage'); tn.image = imgs['normal']; L.new(uvn.outputs['UV'], tn.inputs['Vector'])
    nm = N.new('ShaderNodeNormalMap'); nm.uv_map = uv; L.new(tn.outputs['Color'], nm.inputs['Color']); L.new(nm.outputs['Normal'], b.inputs['Normal'])
    b.inputs['Metallic'].default_value = 0.0; b.inputs['Specular IOR Level'].default_value = look['specular_ior_level']
    m.use_backface_culling = False
    return m


def flat_material(name, rgb, rough, metallic=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True; b = next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    b.inputs['Base Color'].default_value = (*rgb, 1.0); b.inputs['Roughness'].default_value = rough; b.inputs['Metallic'].default_value = metallic
    m.use_backface_culling = False
    return m


garment = shirt_objs + trouser_objs
baked = {}
for ob in garment:
    uv = bake_uv(ob)
    if uv:
        fit = add_bake_uv(ob, uv)
        imgs = bake(ob, uv, tex_size(ob))
        for im in imgs.values(): im.pack()
        newm = gltf_material(f'{ob.name}_PBR', imgs, BAKE_UV)
        baked[ob.name] = dict(uv=uv, size=tex_size(ob), **fit)
        uv = BAKE_UV                                   # the only UV map kept for export
    else:                                              # no UVs: buttons, belt loops, fly pieces -> flat PBR from the built material
        src = next((m for m in ob.data.materials if m), None); rgb, rough = principled_values(src) if src else ((0.5, 0.5, 0.5), 0.8)
        newm = flat_material(f'{ob.name}_PBR', rgb, rough)
    if not uv and len({m.name for m in ob.data.materials if m}) > 1:   # placket_buttons: keep button + hole as two flat materials
        newms = []
        for m in ob.data.materials:
            rgb, rough = principled_values(m); newms.append(flat_material(f'{ob.name}_{m.name}_PBR', rgb, rough))
        for i, m in enumerate(newms): ob.data.materials[i] = m
    else:
        ob.data.materials.clear(); ob.data.materials.append(newm)
        for p in ob.data.polygons: p.material_index = 0
    for u in [u.name for u in ob.data.uv_layers if u.name != uv]: ob.data.uv_layers.remove(ob.data.uv_layers[u])
log('bake done:', baked)


# ---------------------------------------------------------------- stone seating, measured before and after decimation
def surface_bvh(objs):
    dg = bpy.context.evaluated_depsgraph_get(); bm = bmesh.new()
    for o in objs:
        me = o.evaluated_get(dg).to_mesh(); me.transform(o.matrix_world); bm.from_mesh(me); o.evaluated_get(dg).to_mesh_clear()
    bm.normal_update(); return BVHTree.FromBMesh(bm), bm


def seating(stones_npz, objs):
    S = np.load(stones_npz); P, Nn = S['P'], S['N']; bvh, bm = surface_bvh(objs)
    d = np.empty(len(P)); far = 0
    for i, (p, n) in enumerate(zip(P, Nn)):
        loc, nrm, _, dist = bvh.find_nearest(Vector(p))
        if loc is None: d[i] = np.nan; far += 1; continue
        d[i] = (np.asarray(p) - np.asarray(loc)) @ np.asarray(n)        # + = base centre above the cloth along the stone's own normal
    bm.free(); d_mm = d * 1000
    return dict(stones=int(len(P)), base_to_surface_mm=dict(min=round(float(np.nanmin(d_mm)), 3), p01=round(float(np.nanpercentile(d_mm, 1)), 3), median=round(float(np.nanmedian(d_mm)), 3),
                p99=round(float(np.nanpercentile(d_mm, 99)), 3), max=round(float(np.nanmax(d_mm)), 3)),
                sunk_over_0p1mm=int((d_mm < -0.1).sum()), sunk_over_0p3mm=int((d_mm < -0.3).sum()), floating_over_0p5mm=int((d_mm > 0.5).sum()), no_surface=far)


shirt_fabric = [o for o in shirt_objs if o.name in ('Tee', 'collar', 'placket_overlap')]
trouser_fabric = [o for o in trouser_objs if o.name.startswith(('Leg_', 'Waistband'))]
seat = {'full_res': dict(shirt=seating(a.shirt_stones, shirt_fabric), trousers=seating(a.trousers_stones, trouser_fabric))}
log('seating full-res', seat['full_res'])

# ---------------------------------------------------------------- decimate the dense panels (after the bake: UVs are preserved)
verts_before = {o.name: len(o.data.vertices) for o in garment}
for ob in garment:
    n = ob.name.lower(); ratio = DEC['tee'] if n == 'tee' else DEC['collar'] if n == 'collar' else DEC['leg'] if n.startswith('leg_') else None
    if not ratio: continue
    md = ob.modifiers.new('dec', 'DECIMATE'); md.decimate_type = 'COLLAPSE'; md.ratio = ratio; md.use_collapse_triangulate = True
    dg = bpy.context.evaluated_depsgraph_get(); me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg)); ob.modifiers.clear()
    old = ob.data; ob.data = me; bpy.data.meshes.remove(old)
for ob in garment:
    for p in ob.data.polygons: p.use_smooth = True
verts_after = {o.name: len(o.data.vertices) for o in garment}
log('decimated', {k: f'{verts_before[k]} -> {verts_after[k]}' for k in verts_before if verts_before[k] != verts_after[k]})
seat['web'] = dict(shirt=seating(a.shirt_stones, shirt_fabric), trousers=seating(a.trousers_stones, trouser_fabric))
log('seating web', seat['web'])


# ---------------------------------------------------------------- stones: linked duplicates -> EXT_mesh_gpu_instancing
def chaton(sides=16):           # render_set.chaton, verbatim
    ang = 2 * math.pi * np.arange(sides) / sides
    ring = lambda r, y, ph=0.0: np.stack([r * np.cos(ang + ph), np.full(sides, y), r * np.sin(ang + ph)], 1)
    V = np.vstack([ring(0.5, 0.0), ring(0.5, 0.06), ring(0.29, 0.30, math.pi / sides), [[0, 0.30, 0]], [[0, 0, 0]]]); F = []
    for i in range(sides):
        j = (i + 1) % sides
        F += [(i, j, sides + j), (i, sides + j, sides + i), (sides + i, sides + j, 2 * sides + i), (sides + j, 2 * sides + j, 2 * sides + i),
              (2 * sides + i, 2 * sides + j, 3 * sides), (j, i, 3 * sides + 1)]
    return V, np.array(F)


Vu, Fu = chaton()
stone_counts = {}
for part, npz in (('shirt', a.shirt_stones), ('trousers', a.trousers_stones)):
    S = np.load(npz); P, Nn, T, R, cls = S['P'], S['N'], S['T'], S['R'], S['cls']
    Nn = Nn / np.linalg.norm(Nn, axis=1, keepdims=True)
    T = T - (T * Nn).sum(1, keepdims=True) * Nn; T /= np.linalg.norm(T, axis=1, keepdims=True)      # exact frame: tangent made orthogonal to the normal
    B = np.cross(T, Nn)
    for ci, tint, tag in ((0, COL['main'], 'Main'), (1, COL['accent'], 'Accent')):
        idx = np.nonzero(cls == ci)[0]
        if not len(idx): continue
        me = bpy.data.meshes.new(f'Chaton_{part}_{tag}'); me.from_pydata(Vu.tolist(), [], Fu.tolist()); me.update()
        if np.array(me.polygons[4].normal) @ np.array([0.0, 1.0, 0.0]) < 0: me.flip_normals()   # local +Y is outward (render_set's winding check)
        sm = flat_material(f'Stone_{tag}_{a.colour}', tint, 0.16, metallic=1.0); me.materials.append(sm)
        parent = bpy.data.objects.new(f'Stones_{part}_{tag}', None); sc.collection.objects.link(parent)
        for k, i in enumerate(idx):
            d = 2 * R[i]; o = bpy.data.objects.new(f'st_{part}_{tag}_{k}', me); sc.collection.objects.link(o); o.parent = parent
            M = np.eye(4); M[:3, 0] = T[i] * d; M[:3, 1] = Nn[i] * d; M[:3, 2] = B[i] * d; M[:3, 3] = P[i]
            o.matrix_world = Matrix(M.tolist())
        stone_counts[f'{part}_{tag}'] = int(len(idx))
    log('stones', part, stone_counts)

# ---------------------------------------------------------------- export
bpy.ops.export_scene.gltf(filepath=str(OUT), export_format='GLB', use_selection=False, export_apply=True, export_yup=True,
                          export_texcoords=True, export_normals=True, export_tangents=False, export_materials='EXPORT',
                          export_image_format='WEBP', export_image_quality=88,
                          export_draco_mesh_compression_enable=True, export_draco_mesh_compression_level=6,
                          export_draco_position_quantization=14, export_draco_normal_quantization=10, export_draco_texcoord_quantization=12,
                          export_gpu_instances=True, export_cameras=False, export_lights=False, export_extras=False, export_animations=False)
log('exported', OUT, round(OUT.stat().st_size / 1e6, 2), 'MB')

report = dict(file=OUT.name, sha256=sha(OUT), bytes=OUT.stat().st_size, colour=a.colour, albedo=dict(source='brand hex, linear', hex=COL['body'], linear=[round(float(x), 5) for x in body]),
              inputs={k: dict(path=str(p), sha256=sha(p)) for k, p in (('shirt_blend', a.shirt_blend), ('trousers_blend', a.trousers_blend),
                                                                     ('shirt_stones', a.shirt_stones), ('trousers_stones', a.trousers_stones))},
              stones=stone_counts, stones_total=int(sum(stone_counts.values())), baked_maps=baked, decimate=DEC,
              vertices=dict(before=verts_before, after=verts_after), seating=seat,
              limits=['stones are metallic PBR facets, not refractive crystal',
                      'fabric albedo is the brand colour; the catalogue stills use the render calibration, so on-screen colour can differ slightly between the 3D view and the stills',
                      'normal/roughness maps are baked from the procedural knit at the listed texture sizes; finer knit detail than one texel is averaged'])
OUT.with_suffix('.report.json').write_text(json.dumps(report, indent=1))
log('DONE')
