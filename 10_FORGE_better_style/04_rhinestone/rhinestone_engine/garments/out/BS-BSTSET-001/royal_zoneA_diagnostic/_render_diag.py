import sys, json, math, numpy as np, bpy
from mathutils import Vector
W, DG = sys.argv[1], sys.argv[2]
bpy.ops.wm.open_mainfile(filepath=W+'/work/camp_rebuilt.blend', load_ui=False)   # in-memory only, never saved
sc = bpy.context.scene; gcol = bpy.data.collections['Garment']
if 'Mannequin' in bpy.data.collections: bpy.data.collections['Mannequin'].hide_render = True
def mat(name, rgb, metal=0.0, rough=0.5, emit=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True; b = next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    b.inputs['Base Color'].default_value = (*rgb, 1); b.inputs['Metallic'].default_value = metal; b.inputs['Roughness'].default_value = rough
    if emit: b.inputs['Emission Color'].default_value = (*rgb, 1); b.inputs['Emission Strength'].default_value = emit
    return m
cloth = mat('diag_cloth', (0.012, 0.05, 0.22), 0, 0.8)
for o in gcol.objects:
    if o.type == 'MESH':
        o.data.materials.clear(); o.data.materials.append(cloth)
fin = np.load(W+'/out/BS-BSTSET-001/royal_zoneA_staging/zoneA_stones.npz'); pre = np.load(DG+'/pre_clearance.npz'); keep = np.load(DG+'/keep_mask.npy')
tr = json.load(open(DG+'/keystone_trace.json'))
def chaton(sides=16):
    ang = 2 * math.pi * np.arange(sides) / sides; ring = lambda r, y, ph=0.0: np.stack([r * np.cos(ang + ph), np.full(sides, y), r * np.sin(ang + ph)], 1)
    V = np.vstack([ring(0.5, 0.0), ring(0.5, 0.06), ring(0.29, 0.30, math.pi / sides), [[0, 0.30, 0]], [[0, 0, 0]]]); F = []
    for i in range(sides):
        j = (i + 1) % sides; F += [(i, j, sides + j), (i, sides + j, sides + i), (sides + i, sides + j, 2 * sides + i), (sides + j, 2 * sides + j, 2 * sides + i), (2 * sides + i, 2 * sides + j, 3 * sides), (j, i, 3 * sides + 1)]
    return V, np.array(F)
Vu, Fu = chaton()
def cloud(name, P, N, T, D, material, flat_h=None):
    if not len(P): return
    B = np.cross(T, N); Wd = P[:, None, :] + D[:, None, None] * (Vu[None, :, 0:1] * T[:, None, :] + Vu[None, :, 1:2] * N[:, None, :] + Vu[None, :, 2:3] * B[:, None, :])
    F = (Fu[None] + (np.arange(len(P)) * len(Vu))[:, None, None]).reshape(-1, 3); me = bpy.data.meshes.new(name); me.from_pydata(Wd.reshape(-1, 3).tolist(), [], F.tolist()); me.update()
    if np.array(me.polygons[4].normal) @ N[0] < 0: me.flip_normals()
    ob = bpy.data.objects.new(name, me); gcol.objects.link(ob); me.materials.append(material)
tag = fin['tag'].astype(str); P, N, T, R = fin['P'], fin['N'], fin['T'], fin['R']; shape = fin['shape'].astype(str)
cat = {'rail': np.isin(tag, ['base_rail', 'lapel_rail']), 'crown': np.isin(tag, ['band', 'peak', 'tip']), 'sol': tag == 'solitaire'}
for k, col in (('rail', (0.2, 0.9, 1.0)), ('crown', (1.0, 0.95, 0.85)), ('sol', (1.0, 0.75, 0.2))):
    m = cat[k]; cloud('S_' + k, P[m], N[m], T[m], 2 * R[m], mat('m_' + k, col, 1.0, 0.18, 0.6))
rm = ~keep; cloud('S_removed', pre['P'][rm], pre['N'][rm], pre['T'][rm], 2 * pre['R'][rm] * 1.35, mat('m_rm', (1, 0.05, 0.05), 0, 0.3, 4.0))   # removed stones, drawn 35 % larger so they read
kp = np.array(tr['design_point']); bpy.ops.mesh.primitive_uv_sphere_add(radius=0.0043, location=kp + np.array(tr['n']) * 0.002); mk = bpy.context.object; mk.data.materials.append(mat('m_key', (0.1, 1, 0.1), 0, 0.3, 4.0)); mk.name = 'keystone_expected'
w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True; bg = w.node_tree.nodes['Background']; bg.inputs[0].default_value = (0.8, 0.8, 0.8, 1); bg.inputs[1].default_value = 0.8
def light(loc, e):
    d = bpy.data.lights.new('l', 'AREA'); d.energy = e; d.size = 1.5; o = bpy.data.objects.new('l', d); sc.collection.objects.link(o); o.location = loc
    c = o.constraints.new('TRACK_TO'); c.target = tgt; c.track_axis = 'TRACK_NEGATIVE_Z'; c.up_axis = 'UP_Y'
tgt = bpy.data.objects.new('tgt', None); sc.collection.objects.link(tgt); tgt.location = (0, 0.0, 1.42)
light((0.6, 1.4, 2.2), 260); light((-0.9, -1.2, 2.0), 180); light((0, 0.3, 2.6), 150)
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam
r = sc.render; r.engine = 'CYCLES'; sc.cycles.samples = 24; sc.cycles.device = 'CPU'; sc.cycles.use_denoising = True; r.image_settings.file_format = 'PNG'
try: sc.view_settings.view_transform = 'Standard'
except Exception: pass
views = {  # name: (cam location, target, lens mm, res)
 'back_full': ((0, 0.9, 1.50), (0, 0.0, 1.40), 70, (1400, 900)),
 'back_closeup': ((0, 0.45, 1.58), (0, 0.04, 1.43), 85, (1400, 900)),
 'top_down': ((0, 0.0, 2.05), (0, -0.01, 1.43), 85, (1400, 1000)),
 'front_threequarter': ((0.55, -0.80, 1.72), (0, -0.04, 1.42), 85, (1400, 1000)),
 'front': ((0, -0.95, 1.55), (0, -0.04, 1.40), 80, (1400, 1000)),
 'left_lapel_closeup': ((0.26, -0.40, 1.60), (0.07, -0.08, 1.40), 85, (1400, 1000)),
}
for nm, (cl, tg, lens, res) in views.items():
    cam.location = cl; cam.data.lens = lens; tgt.location = tg; d = Vector(tg) - Vector(cl); cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    r.resolution_x, r.resolution_y = res; r.filepath = f'{DG}/render_{nm}.png'; bpy.ops.render.render(write_still=True); print('RENDER', nm, flush=True)
