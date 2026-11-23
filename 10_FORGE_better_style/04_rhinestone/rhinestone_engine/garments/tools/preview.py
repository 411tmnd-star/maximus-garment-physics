"""Quick Cycles previews of a .blend: python -I -B preview.py <blend> <out_prefix> view[,view..] [--res WxH] [--spp N] [--hide name,name] [--clay]
view = name:camx,camy,camz:tx,ty,tz:lens   (ortho if lens starts with 'o' -> ortho scale)"""
import sys, math, bpy
from mathutils import Vector
a = sys.argv[1:]; import os; blend, prefix, views = a[0], os.path.abspath(a[1]), a[2].split(';')
def opt(k, d): return a[a.index(k) + 1] if k in a else d
RX, RY = [int(x) for x in opt('--res', '700x700').split('x')]; spp = int(opt('--spp', '12'))
bpy.ops.wm.open_mainfile(filepath=blend, load_ui=False); sc = bpy.context.scene
for n in opt('--hide', '').split(','):
    if n in bpy.data.objects: bpy.data.objects[n].hide_render = True
    if n in bpy.data.collections: bpy.data.collections[n].hide_render = True
if '--clay' in a:
    m = bpy.data.materials.new('clay'); m.use_nodes = True
    for o in bpy.data.objects:
        if o.type == 'MESH' and o.name != 'Mannequin':
            o.data.materials.clear(); o.data.materials.append(m)
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = spp; sc.cycles.use_denoising = True
sc.render.resolution_x, sc.render.resolution_y = RX, RY; sc.render.film_transparent = False
w = sc.world or bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True; w.node_tree.nodes['Background'].inputs[0].default_value = (0.8, 0.8, 0.8, 1); w.node_tree.nodes['Background'].inputs[1].default_value = 0.7
cam = bpy.data.objects.new('pcam', bpy.data.cameras.new('pcam')); sc.collection.objects.link(cam); sc.camera = cam
sun = bpy.data.objects.new('psun', bpy.data.lights.new('psun', 'SUN')); sun.data.energy = 2.5; sun.data.angle = 0.3; sc.collection.objects.link(sun)
for v in views:
    name, c, t, lens = v.split(':'); c = Vector([float(x) for x in c.split(',')]); t = Vector([float(x) for x in t.split(',')])
    cam.location = c; d = (t - c); cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    sun.rotation_euler = (d.to_track_quat('-Z', 'Y').to_euler().x - 0.5, 0, d.to_track_quat('-Z', 'Y').to_euler().z + 0.5)
    if lens.startswith('o'): cam.data.type = 'ORTHO'; cam.data.ortho_scale = float(lens[1:])
    else: cam.data.type = 'PERSP'; cam.data.lens = float(lens)
    sc.render.filepath = f'{prefix}_{name}.png'; bpy.ops.render.render(write_still=True); print('WROTE', sc.render.filepath)
