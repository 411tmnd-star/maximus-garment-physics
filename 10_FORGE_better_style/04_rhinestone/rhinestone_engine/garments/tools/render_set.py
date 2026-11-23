"""render_set.py - catalogue-pattern floating renders (same rig, lights, view transform, framing and ground as engine/render_sku.py)
of a garment .blend with a stone layout.
usage: python -B render_set.py --blend B --stones S.npz --colour blue|red --kind shirt|trousers --out-dir D --name N
                               [--views front,back,three_quarter] [--res 1200x1500] [--samples 64] [--no-stones]
The live engine is imported read-only (no bytecode is written); nothing is saved outside --out-dir."""
import sys, os, math, json, time, argparse, hashlib
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
import bpy

HE = Path(os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/human_engine/src_catalog/better_style_catalog')))
sys.path.insert(0, str(HE / 'engine')); sys.path.insert(0, str(HE / 'engine' / 'components'))
import render_sku as RS, camera_framing as CF
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'prototypes' / 'P1_bstset_exact_hull'))
import convex_gap as CG

ap = argparse.ArgumentParser()
for k, d in (('--blend', None), ('--stones', None), ('--colour', 'blue'), ('--kind', 'shirt'), ('--out-dir', None), ('--name', 'BS-BSTSET-001'), ('--views', 'front,back'),
             ('--res', '1200x1500'), ('--samples', '64'), ('--label', None), ('--bundle', None), ('--cw', None), ('--sku', None)):
    ap.add_argument(k, default=d)
ap.add_argument('--no-stones', action='store_true'); ap.add_argument('--keep-blend', action='store_true')
a = ap.parse_args(); OUT = Path(a.out_dir).resolve(); OUT.mkdir(parents=True, exist_ok=True)
RX, RY = [int(x) for x in a.res.split('x')]; VIEWS = {'front': 0, 'three_quarter': 30, 'back': 180, 'side': 90}

COL = {  # fabric hex, stone main / accent (linear RGB), button hex
    'blue': dict(body='#193D7C', main=(0.86, 0.90, 1.0), accent=(1.0, 0.74, 0.30), button='#0E2A5D', cal='blue'),
    'red': dict(body='#A3121A', main=(1.0, 0.72, 0.28), accent=(0.95, 0.93, 0.88), button='#6E0C12', cal=None),
}[a.colour]

bpy.ops.wm.open_mainfile(filepath=a.blend); sc = bpy.context.scene
gcol = bpy.data.collections['Garment']
if 'Mannequin' in bpy.data.collections: bpy.data.collections['Mannequin'].hide_render = True
mats = {m.name: m for ob in gcol.objects if ob.type == 'MESH' for m in ob.data.materials if m}
sku = json.loads((HE / 'configs/skus/BS-CAMP-001.json').read_text()); look = dict(sku['material']); knit = look['knit']; rd = sku['render']
CAL = json.loads((HE / rd['calibration']).read_text())
if COL['cal']: body = np.asarray(CAL['colorways'][COL['cal']]['main_body_linear'], float)
else:        # no calibration entry for this colour: scale the brand colour by the albedo/target ratio the calibration found for gold (approximation)
    g_t = RS.hex_to_lin('#AB874F'); g_c = np.asarray(CAL['colorways']['gold']['main_body_linear'], float)
    body = np.asarray(RS.hex_to_lin(COL['body'])) * (g_c[0] / g_t[0]) * 0.72

# ---------------------------------------------------------------- materials (render_sku's fabric, no all-over print: the monogram is stones)
if a.kind == 'shirt':
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
else:
    for nm, m in mats.items():
        if nm.startswith('Trouser'): RS.build_fabric_material(bpy, m, body, look, knit, nm.endswith('Trim'))
        elif nm == 'Button': RS.build_button_material(bpy, m, RS.hex_to_lin(COL['button']), False)


# ---------------------------------------------------------------- stones: one mesh per colour class (round brilliant-style chaton, 16 sides)
def stone_material(name, tint):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; N = nt.nodes; L = nt.links; N.clear()
    out = N.new('ShaderNodeOutputMaterial'); b = N.new('ShaderNodeBsdfPrincipled')
    b.inputs['Base Color'].default_value = (*tint, 1); b.inputs['Metallic'].default_value = 1.0; b.inputs['Roughness'].default_value = 0.16
    # facet sparkle that does not depend on the studio world: a high-contrast pattern looked up by the reflection direction
    tc = N.new('ShaderNodeTexCoord'); vo = N.new('ShaderNodeTexVoronoi'); vo.inputs['Scale'].default_value = 3.4; L.new(tc.outputs['Reflection'], vo.inputs['Vector'])
    ramp = N.new('ShaderNodeValToRGB'); ramp.color_ramp.elements[0].position = 0.30; ramp.color_ramp.elements[1].position = 0.75; ramp.color_ramp.elements[0].color = (0.10, 0.10, 0.12, 1); L.new(vo.outputs['Distance'], ramp.inputs['Fac'])
    mul = N.new('ShaderNodeMixRGB'); mul.blend_type = 'MULTIPLY'; mul.inputs['Fac'].default_value = 1.0; mul.inputs['Color2'].default_value = (*tint, 1); L.new(ramp.outputs['Color'], mul.inputs['Color1'])
    b.inputs['Emission Strength'].default_value = 0.55; L.new(mul.outputs['Color'], b.inputs['Emission Color'])
    L.new(b.outputs['BSDF'], out.inputs['Surface']); return m


def chaton(sides=16):
    """unit-diameter stone, local +Y = outward: base ring, girdle ring, table ring (profile of chaton_v1)."""
    ang = 2 * math.pi * np.arange(sides) / sides
    ring = lambda r, y, ph=0.0: np.stack([r * np.cos(ang + ph), np.full(sides, y), r * np.sin(ang + ph)], 1)
    V = np.vstack([ring(0.5, 0.0), ring(0.5, 0.06), ring(0.29, 0.30, math.pi / sides), [[0, 0.30, 0]], [[0, 0, 0]]]); F = []
    for i in range(sides):
        j = (i + 1) % sides
        F += [(i, j, sides + j), (i, sides + j, sides + i), (sides + i, sides + j, 2 * sides + i), (sides + j, 2 * sides + j, 2 * sides + i),
              (2 * sides + i, 2 * sides + j, 3 * sides), (j, i, 3 * sides + 1)]
    return V, np.array(F)


n_st = 0
if a.stones and not a.no_stones:
    S = np.load(a.stones); P, Nn, T, R, cls = S['P'], S['N'], S['T'], S['R'], S['cls']; n_st = len(P)
    Vu, Fu = chaton(); B = np.cross(T, Nn)
    for c, tint, nm in ((0, COL['main'], 'Stone_Main'), (1, COL['accent'], 'Stone_Accent')):
        idx = np.nonzero(cls == c)[0]
        if not len(idx): continue
        d = (2 * R[idx])[:, None, None]
        W = P[idx][:, None, :] + d * (Vu[None, :, 0:1] * T[idx][:, None, :] + Vu[None, :, 1:2] * Nn[idx][:, None, :] + Vu[None, :, 2:3] * B[idx][:, None, :])
        F = (Fu[None] + (np.arange(len(idx)) * len(Vu))[:, None, None]).reshape(-1, 3)
        me = bpy.data.meshes.new(nm); me.from_pydata(W.reshape(-1, 3).tolist(), [], F.tolist()); me.update()
        ob = bpy.data.objects.new(nm, me); gcol.objects.link(ob); me.materials.append(stone_material(nm, tint))
        # outward winding check on the first stone
        if np.array(me.polygons[4].normal) @ Nn[idx[0]] < 0: me.flip_normals()

# ---------------------------------------------------------------- studio: identical to render_sku
piv = bpy.data.objects.new('piv', None); sc.collection.objects.link(piv); tgt = bpy.data.objects.new('tgt', None); sc.collection.objects.link(tgt)
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.lens = rd['lens_mm']; cam.data.sensor_width = 36; cam.parent = piv
lights = []


def area(name, loc, en, size, col=(1, 1, 1)):
    d = bpy.data.lights.new(name, 'AREA'); d.energy = en; d.size = size; d.color = col; o = bpy.data.objects.new(name, d); sc.collection.objects.link(o); o.parent = piv; o.location = loc
    t = o.constraints.new('TRACK_TO'); t.target = tgt; t.track_axis = 'TRACK_NEGATIVE_Z'; t.up_axis = 'UP_Y'; lights.append((d, en))


area('key', (-1.3, -2.0, 0.9), 260, 2.4); area('fill', (1.8, -1.7, -0.1), 70, 3.0, (0.95, 0.97, 1.0)); area('rim', (0.5, 1.9, 0.9), 320, 1.6, (1.0, 0.96, 0.9)); area('top', (0, -0.4, 1.7), 150, 1.8)
w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True; bg = w.node_tree.nodes['Background']; bg.inputs[0].default_value = (0.84, 0.835, 0.82, 1); bg.inputs[1].default_value = 0.06
r = sc.render; r.engine = 'CYCLES'; r.film_transparent = True; r.image_settings.file_format = 'PNG'; r.image_settings.color_mode = 'RGBA'; r.image_settings.color_depth = '8'
c = sc.cycles; c.device = 'CPU'; c.samples = int(a.samples); c.use_denoising = True; c.max_bounces = 6
try: c.denoiser = 'OPENIMAGEDENOISE'
except Exception: pass
sc.view_settings.view_transform = CAL['view_transform']; sc.view_settings.exposure = CAL.get('exposure', 0.0); sc.view_settings.look = CAL.get('look', 'None')
r.resolution_x, r.resolution_y = RX, RY
objs = [o for o in gcol.objects if o.type == 'MESH' and not o.name.startswith('Stone')]
gmin, gmax = RS.bbox_of(bpy, objs); H_REF = 0.62; FR = rd['framing']
SUBJ = CF.subject_points(objs, None, 'floating')
log = []
for vn in a.views.split(','):
    ctr = (gmin + gmax) / 2; Hh = gmax[2] - gmin[2]; k = Hh / H_REF; piv.location = ctr; tgt.location = ctr; piv.scale = (k, k, k)
    cam.location = (0, -(rd['lens_mm'] * 1.30 * H_REF / 36), 0); cam.rotation_euler = (math.radians(90), 0, 0)
    for d, en in lights: d.energy = en * k * k
    piv.rotation_euler = (0, 0, math.radians(-VIEWS[vn]))
    frep = CF.solve(sc, cam, piv, tgt, SUBJ, FR.get('target_occupancy', 0.78), FR.get('max_width', 0.92), light_datas=[(d, d.energy) for d, _ in lights])
    fn = OUT / f"{a.name}_{a.colour}_floating_{vn}.png"; r.filepath = str(fn); t = time.time(); bpy.ops.render.render(write_still=True); dt = time.time() - t
    frep['measured_alpha'] = CF.measure_alpha(fn); RS.composite_on_ground(fn, rd['background_hex'])
    log.append(dict(file=fn.name, view=vn, colour=a.colour, seconds=round(dt, 1), res=[RX, RY], samples=int(a.samples), framing=frep, background_hex=rd['background_hex'],
                    calibrated=bool(COL['cal']), stones=n_st, sha256=hashlib.sha256(fn.read_bytes()).hexdigest()))
    print('RENDER', fn.name, round(dt, 1), 's', flush=True)
lp = OUT / f"render_log_{a.name}_{a.colour}_{a.kind}.json"; prev = json.loads(lp.read_text())['renders'] if lp.exists() else []
prev = [p for p in prev if p['file'] not in {l['file'] for l in log}] + log
lp.write_text(json.dumps(dict(name=a.name, blend=str(a.blend), blend_sha256=hashlib.sha256(Path(a.blend).read_bytes()).hexdigest(), stones=str(a.stones),
                              stones_sha256=hashlib.sha256(Path(a.stones).read_bytes()).hexdigest() if a.stones else None,
                              view_transform=sc.view_settings.view_transform, look=sc.view_settings.look, exposure=sc.view_settings.exposure, renders=prev), indent=1))
if a.keep_blend: bpy.ops.wm.save_as_mainfile(filepath=str(OUT / f'_scene_{a.name}_{a.colour}.blend'))
