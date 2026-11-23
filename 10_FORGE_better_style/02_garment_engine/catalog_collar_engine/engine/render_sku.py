"""render_sku.py  Stage 2 of the Two-Stage Garment Engine: render a SKU from a frozen archetype. No cloth simulation happens here.

  python engine/render_sku.py --manifest configs/skus/BS-RINGER-001.json [--out-dir DIR] [--res 1920x2400 | --hi-res] [--samples 96] [--only white-black] [--views front,back]

Loads the archetype master .blend (baked garment + posed mannequin, built once by engine/build_archetype.py), then for each colourway in the manifest:
  * rebuilds the four materials (Main_Body, Collar_Rib, Left_Cuff_Rib, Right_Cuff_Rib) from RGB values,
  * applies the compiled artwork plates (engine/preprocess_artwork.py): light-fabric or dark-fabric ink plate, front and back, plus the ink height map,
  * renders each requested orbit angle in `floating` mode (mannequin hidden) and/or `mannequin` mode (mannequin shown, finish M1/M2/M3).
Look: Specular IOR Level 0.18, sheen multiplied by (1 - ink alpha) so ink never goes grey, knit relief + a small albedo modulation so the weave reads at full-garment distance.
Schema: configs/schema/sku_manifest.schema.json
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def srgb_to_lin(c):
    c = np.asarray(c, float); return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(np.asarray(c, float), 0, 1); return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def hex_to_lin(h):
    h = h.lstrip('#'); return srgb_to_lin([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def luminance(lin): return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])


def recolour_wire_texture(png_path, hex_, out_path):
    """The engine's mannequin finish (appearance_variants/mannequin_finishes/code/mf_lib.py bake_body_albedo): out_lin = hex_lin * tex_lin / base_lin per
    channel, base = the most common texel (the flat skin), done in linear light so flat texels become exactly `hex_` and the wire lines keep their relative reflectance."""
    from PIL import Image
    im = np.asarray(Image.open(png_path).convert('RGBA')); rgb = im[..., :3].astype(np.float64)
    flat = rgb.reshape(-1, 3).astype(np.uint8); key = (flat[:, 0].astype(np.uint32) << 16) | (flat[:, 1].astype(np.uint32) << 8) | flat[:, 2]
    vals, counts = np.unique(key, return_counts=True); b = vals[counts.argmax()]; base = np.array([(b >> 16) & 255, (b >> 8) & 255, b & 255], float)
    t = np.array([int(hex_.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)], float)
    ratio = srgb_to_lin(rgb / 255) / srgb_to_lin(base / 255)
    out = np.round(lin_to_srgb(np.clip(srgb_to_lin(t / 255) * ratio, 0, 1)) * 255).astype(np.uint8)
    Image.fromarray(out, 'RGB').save(out_path); return out_path


def render_provenance(paths, bpy):
    """What produced a render set: sha256 of every input file, the engine commit and whether those files differ from it, Blender version.
    A render folder can hold images from earlier masters; render_log.json is the record of which inputs THIS run used (GI-02)."""
    import hashlib, subprocess
    files = {}
    for key, p in paths.items():
        if p is None: continue
        p = Path(p); files[key] = dict(path=p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.as_posix(),
                                       sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None)
    def git(*a):
        r = subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True); return r.stdout.strip() if r.returncode == 0 else None
    tracked = [str(Path(p)) for p in paths.values() if p is not None and Path(p).is_file() and Path(p).is_relative_to(ROOT)]
    dirty = git('status', '--porcelain', '--', *tracked) if tracked else None
    # every image file the scene actually loaded (artwork plates, height maps, print tile, recoloured mannequin skin): the bundle manifest
    # lists plate paths only, so a re-exported plate under the same name would otherwise go unrecorded
    images = {}
    for im in bpy.data.images:
        if im.source != 'FILE' or not im.filepath: continue
        p = Path(bpy.path.abspath(im.filepath))
        if p.is_file(): images[p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    ocio = os.environ.get('OCIO') or os.path.join(bpy.utils.resource_path('LOCAL'), 'datafiles', 'colormanagement', 'config.ocio')
    sc = bpy.context.scene
    return dict(files=files, image_files=images, git_commit=git('rev-parse', 'HEAD'), inputs_differ_from_commit=None if dirty is None else bool(dirty),
                blender=bpy.app.version_string, cycles_device=sc.cycles.device,
                ocio_config=dict(path=Path(ocio).as_posix(), sha256=hashlib.sha256(Path(ocio).read_bytes()).hexdigest() if os.path.isfile(ocio) else None,
                                 from_env=bool(os.environ.get('OCIO'))))


def file_sha256(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json_atomic(path, obj):
    """Write via a temp file in the same folder + os.replace, so a reader never sees a half-written log."""
    path = Path(path); tmp = path.with_name(path.name + '.tmp'); tmp.write_text(json.dumps(obj, indent=1)); os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------------------------- blender side
def build_fabric_material(bpy, mat, rgb_lin, look, knit, rib=False, art=None, uv_name='UVMap_Fabric', print_tile=None):
    """Principled fabric. art = dict(front=(albedo_path, height_path)|None, back=...|None)."""
    mat.use_nodes = True; nt = mat.node_tree; N = nt.nodes; Lk = nt.links; N.clear()
    out = N.new('ShaderNodeOutputMaterial'); b = N.new('ShaderNodeBsdfPrincipled'); Lk.new(b.outputs['BSDF'], out.inputs['Surface'])
    uv = N.new('ShaderNodeUVMap'); uv.uv_map = uv_name; mp = N.new('ShaderNodeMapping'); Lk.new(uv.outputs['UV'], mp.inputs['Vector'])
    sc = knit['rib_scale'] if rib else knit['scale']
    w1 = N.new('ShaderNodeTexWave'); w1.wave_type = 'BANDS'; w1.bands_direction = 'X'; w1.inputs['Scale'].default_value = sc; w1.inputs['Distortion'].default_value = knit.get('distortion', 1.0); w1.inputs['Detail'].default_value = 2
    w2 = N.new('ShaderNodeTexWave'); w2.wave_type = 'BANDS'; w2.bands_direction = 'Y'; w2.inputs['Scale'].default_value = sc * 0.8; w2.inputs['Distortion'].default_value = 0.7
    nz = N.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = sc * 5; nz.inputs['Detail'].default_value = 5
    mott = N.new('ShaderNodeTexNoise'); mott.inputs['Scale'].default_value = knit.get('mottle_scale', 9.0); mott.inputs['Detail'].default_value = 3   # low-frequency irregularity of the yarn
    for t in (w1, w2, nz, mott): Lk.new(mp.outputs['Vector'], t.inputs['Vector'])
    mul = N.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'; Lk.new(w1.outputs['Fac'], mul.inputs[0]); Lk.new(w2.outputs['Fac'], mul.inputs[1])
    ad = N.new('ShaderNodeMath'); ad.operation = 'ADD'; m2 = N.new('ShaderNodeMath'); m2.operation = 'MULTIPLY'; m2.inputs[1].default_value = 0.3
    Lk.new(mul.outputs['Value'], ad.inputs[0]); Lk.new(nz.outputs['Fac'], m2.inputs[0]); Lk.new(m2.outputs['Value'], ad.inputs[1])
    weave = ad.outputs['Value']
    # --- base colour, modulated by the weave so the knit is visible without grazing light
    c = float(knit['albedo_contrast']); mc = N.new('ShaderNodeMath'); mc.operation = 'MULTIPLY_ADD'; mc.inputs[1].default_value = c; mc.inputs[2].default_value = 1.0 - c; Lk.new(weave, mc.inputs[0])
    mott_m = N.new('ShaderNodeMath'); mott_m.operation = 'MULTIPLY_ADD'; mott_m.inputs[1].default_value = knit.get('mottle_contrast', 0.06); mott_m.inputs[2].default_value = 1.0 - knit.get('mottle_contrast', 0.06) / 2; Lk.new(mott.outputs['Fac'], mott_m.inputs[0])
    wm = N.new('ShaderNodeMath'); wm.operation = 'MULTIPLY'; Lk.new(mc.outputs['Value'], wm.inputs[0]); Lk.new(mott_m.outputs['Value'], wm.inputs[1])
    base = N.new('ShaderNodeRGB'); base.outputs[0].default_value = (*rgb_lin, 1.0)
    colour = base.outputs[0]; ink_alpha = None; ink_height = None
    if print_tile:                  # all-over repeat (camp monogram lattice): ground = rgb_lin, ink = print_tile['ink_lin'], mask = seamless tile on UVMap_Print
        pu = N.new('ShaderNodeUVMap'); pu.uv_map = print_tile.get('uv', 'UVMap_Print'); pm = N.new('ShaderNodeMapping'); pm.inputs['Scale'].default_value = (*print_tile['scale_uv'], 1.0); pm.inputs['Rotation'].default_value = (0.0, 0.0, math.radians(print_tile.get('rot_deg', 0.0)))
        Lk.new(pu.outputs['UV'], pm.inputs['Vector']); pt = N.new('ShaderNodeTexImage'); pt.image = bpy.data.images.load(str(print_tile['path']), check_existing=True)
        pt.image.colorspace_settings.name = 'Non-Color'; pt.extension = 'REPEAT'; pt.interpolation = 'Cubic'; Lk.new(pm.outputs['Vector'], pt.inputs['Vector'])
        pi = N.new('ShaderNodeRGB'); pi.outputs[0].default_value = (*print_tile['ink_lin'], 1.0)
        px = N.new('ShaderNodeMixRGB'); Lk.new(pt.outputs['Color'], px.inputs['Fac']); Lk.new(colour, px.inputs['Color1']); Lk.new(pi.outputs[0], px.inputs['Color2']); colour = px.outputs['Color']
    for side in ('front', 'back'):
        a = (art or {}).get(side)
        if not a: continue
        uva = N.new('ShaderNodeUVMap'); uva.uv_map = 'UVMap_Art_Front' if side == 'front' else 'UVMap_Art_Back'
        im = N.new('ShaderNodeTexImage'); im.image = bpy.data.images.load(str(a[0]), check_existing=True); im.extension = 'CLIP'; im.interpolation = 'Cubic'; Lk.new(uva.outputs['UV'], im.inputs['Vector'])
        art_c = im.outputs['Color']
        if look.get('art_albedo_scale', 1.0) != 1.0:      # placed art is an sRGB picture, not a calibrated albedo: scale it by the lift the calibration measured on a neutral body
            sca = N.new('ShaderNodeMixRGB'); sca.blend_type = 'MULTIPLY'; sca.inputs['Fac'].default_value = 1.0; sca.inputs['Color2'].default_value = (*([look['art_albedo_scale']] * 3), 1.0)
            Lk.new(art_c, sca.inputs['Color1']); art_c = sca.outputs['Color']
        mix = N.new('ShaderNodeMixRGB'); Lk.new(colour, mix.inputs['Color1']); Lk.new(art_c, mix.inputs['Color2']); Lk.new(im.outputs['Alpha'], mix.inputs['Fac']); colour = mix.outputs['Color']
        if ink_alpha is None: ink_alpha = im.outputs['Alpha']
        else:
            mx = N.new('ShaderNodeMath'); mx.operation = 'MAXIMUM'; Lk.new(ink_alpha, mx.inputs[0]); Lk.new(im.outputs['Alpha'], mx.inputs[1]); ink_alpha = mx.outputs['Value']
        hi = N.new('ShaderNodeTexImage'); hi.image = bpy.data.images.load(str(a[1]), check_existing=True); hi.image.colorspace_settings.name = 'Non-Color'; hi.extension = 'CLIP'; hi.interpolation = 'Cubic'; Lk.new(uva.outputs['UV'], hi.inputs['Vector'])
        if ink_height is None: ink_height = hi.outputs['Color']
        else:
            mh = N.new('ShaderNodeMath'); mh.operation = 'MAXIMUM'; Lk.new(ink_height, mh.inputs[0]); Lk.new(hi.outputs['Color'], mh.inputs[1]); ink_height = mh.outputs['Value']
    # knit shading only on bare fabric (the print sits on top of the knit)
    modmix = N.new('ShaderNodeMixRGB'); modmix.blend_type = 'MULTIPLY'; modmix.inputs['Fac'].default_value = 1.0
    Lk.new(colour, modmix.inputs['Color1'])
    wv = N.new('ShaderNodeCombineColor'); [Lk.new(wm.outputs['Value'], wv.inputs[i]) for i in range(3)]; Lk.new(wv.outputs['Color'], modmix.inputs['Color2'])
    if ink_alpha is not None:       # no knit modulation under ink: lerp the multiplier to 1 by ink alpha
        fm = N.new('ShaderNodeMixRGB'); fm.inputs['Color2'].default_value = (1, 1, 1, 1); Lk.new(wv.outputs['Color'], fm.inputs['Color1']); Lk.new(ink_alpha, fm.inputs['Fac']); Lk.new(fm.outputs['Color'], modmix.inputs['Color2'])
    Lk.new(modmix.outputs['Color'], b.inputs['Base Color'])
    b.inputs['Specular IOR Level'].default_value = look['specular_ior_level']; b.inputs['Roughness'].default_value = look['roughness']
    b.inputs['Sheen Roughness'].default_value = 0.6
    b.inputs['Sheen Tint'].default_value = (*[min(float(x) * look.get('sheen_tint_mix', 0.5) + look.get('sheen_tint_floor', 0.1), 1) for x in rgb_lin], 1)   # polo: mix 1.0, floor 0 so the sheen does not pale saturated colours
    if ink_alpha is not None:       # sheen = 0 under ink, roughness slightly lower on ink
        sh = N.new('ShaderNodeMath'); sh.operation = 'MULTIPLY_ADD'; sh.inputs[1].default_value = -look['sheen']; sh.inputs[2].default_value = look['sheen']; Lk.new(ink_alpha, sh.inputs[0]); Lk.new(sh.outputs['Value'], b.inputs['Sheen Weight'])
        rm = N.new('ShaderNodeMath'); rm.operation = 'MULTIPLY_ADD'; rm.inputs[1].default_value = -0.05; rm.inputs[2].default_value = look['roughness']; Lk.new(ink_alpha, rm.inputs[0]); Lk.new(rm.outputs['Value'], b.inputs['Roughness'])
    else: b.inputs['Sheen Weight'].default_value = look['sheen']
    # relief: knit (scaled down under ink) + ink thickness from the height map
    height = weave
    if ink_alpha is not None:
        keep = N.new('ShaderNodeMath'); keep.operation = 'MULTIPLY_ADD'; keep.inputs[1].default_value = -0.8; keep.inputs[2].default_value = 1.0; Lk.new(ink_alpha, keep.inputs[0])
        wk = N.new('ShaderNodeMath'); wk.operation = 'MULTIPLY'; Lk.new(weave, wk.inputs[0]); Lk.new(keep.outputs['Value'], wk.inputs[1])
        ih = N.new('ShaderNodeMath'); ih.operation = 'MULTIPLY'; ih.inputs[1].default_value = look.get('ink_relief_gain', 2.0); Lk.new(ink_height, ih.inputs[0])   # ink_height is an RGB: Math takes its first channel's luminance
        hs = N.new('ShaderNodeMath'); hs.operation = 'ADD'; Lk.new(wk.outputs['Value'], hs.inputs[0]); Lk.new(ih.outputs['Value'], hs.inputs[1]); height = hs.outputs['Value']
    bump = N.new('ShaderNodeBump'); bump.inputs['Strength'].default_value = knit['bump_strength']; bump.inputs['Distance'].default_value = knit['bump_distance_m']
    Lk.new(height, bump.inputs['Height']); Lk.new(bump.outputs['Normal'], b.inputs['Normal'])


def overlay_line_mask(mat, mask_socket, darken=0.2, relief=-0.6):
    """Darken the base colour and add relief where `mask_socket` (0..1) is high: seam grooves (relief < 0) or raised top-stitching (relief > 0).
    Inserted in front of the Principled BSDF's Base Color and the Bump's Height of a material made by build_fabric_material."""
    nt = mat.node_tree; N = nt.nodes; Lk = nt.links
    b = next(n for n in N if n.type == 'BSDF_PRINCIPLED'); bump = next(n for n in N if n.type == 'BUMP')
    src_c = b.inputs['Base Color'].links[0].from_socket; src_h = bump.inputs['Height'].links[0].from_socket
    dk = N.new('ShaderNodeMath'); dk.operation = 'MULTIPLY_ADD'; dk.inputs[1].default_value = -darken; dk.inputs[2].default_value = 1.0; Lk.new(mask_socket, dk.inputs[0])
    cc = N.new('ShaderNodeCombineColor'); [Lk.new(dk.outputs['Value'], cc.inputs[i]) for i in range(3)]
    mm = N.new('ShaderNodeMixRGB'); mm.blend_type = 'MULTIPLY'; mm.inputs['Fac'].default_value = 1.0; Lk.new(src_c, mm.inputs['Color1']); Lk.new(cc.outputs['Color'], mm.inputs['Color2'])
    Lk.new(mm.outputs['Color'], b.inputs['Base Color'])
    hh = N.new('ShaderNodeMath'); hh.operation = 'MULTIPLY_ADD'; hh.inputs[1].default_value = relief; Lk.new(mask_socket, hh.inputs[0]); Lk.new(src_h, hh.inputs[2])
    Lk.new(hh.outputs['Value'], bump.inputs['Height'])


def seam_mask(mat, lo=0.42, hi=0.85):
    """Mask of construction seams (raglan, side) from the per-vertex `seam` attribute written by build_archetype's bake (1 on sewn vertices)."""
    N = mat.node_tree.nodes; Lk = mat.node_tree.links
    at = N.new('ShaderNodeAttribute'); at.attribute_name = 'seam'; at.attribute_type = 'GEOMETRY'
    mr = N.new('ShaderNodeMapRange'); mr.inputs['From Min'].default_value = lo; mr.inputs['From Max'].default_value = hi; mr.clamp = True
    Lk.new(at.outputs['Fac'], mr.inputs['Value']); return mr.outputs['Result']


def placket_stitch_mask(mat, length_m, width_m, inset_m=0.0045, thread_m=0.0007, dash_m=0.004):
    """Top-stitching on the 3D placket from UVMap_Placket (u, v = metres across / down from the top, / 0.30): two dashed lines `inset_m` inside the
    long edges and a dashed box-stitch line across the bottom."""
    N = mat.node_tree.nodes; Lk = mat.node_tree.links
    uv = N.new('ShaderNodeUVMap'); uv.uv_map = 'UVMap_Placket'; sx = N.new('ShaderNodeSeparateXYZ'); Lk.new(uv.outputs['UV'], sx.inputs['Vector'])

    def m(op, a, b=None):
        n = N.new('ShaderNodeMath'); n.operation = op
        for i, x in enumerate((a, b)):
            if x is None: continue
            if isinstance(x, float): n.inputs[i].default_value = x
            else: Lk.new(x, n.inputs[i])
        return n.outputs['Value']
    um = m('MULTIPLY', sx.outputs['X'], 0.30); vm = m('MULTIPLY', sx.outputs['Y'], 0.30)
    dash_v = m('LESS_THAN', m('FRACT', m('DIVIDE', vm, dash_m)), 0.62); dash_u = m('LESS_THAN', m('FRACT', m('DIVIDE', um, dash_m)), 0.62)
    lineL = m('LESS_THAN', m('ABSOLUTE', m('SUBTRACT', um, inset_m)), thread_m); lineR = m('LESS_THAN', m('ABSOLUTE', m('SUBTRACT', um, width_m - inset_m)), thread_m)
    along = m('MULTIPLY', m('MULTIPLY', m('MAXIMUM', lineL, lineR), m('LESS_THAN', vm, length_m - inset_m)), dash_v)
    inside = m('MULTIPLY', m('GREATER_THAN', um, inset_m), m('LESS_THAN', um, width_m - inset_m))
    box = m('MULTIPLY', m('MULTIPLY', m('LESS_THAN', m('ABSOLUTE', m('SUBTRACT', vm, length_m - inset_m)), thread_m), inside), dash_u)
    return m('MAXIMUM', along, box)


def build_button_material(bpy, mat, rgb_lin, hole=False):
    mat.use_nodes = True; N = mat.node_tree.nodes; Lk = mat.node_tree.links; N.clear()
    out = N.new('ShaderNodeOutputMaterial'); b = N.new('ShaderNodeBsdfPrincipled'); Lk.new(b.outputs['BSDF'], out.inputs['Surface'])
    c = np.asarray(rgb_lin) * (0.08 if hole else 0.92); b.inputs['Base Color'].default_value = (*c, 1.0)
    b.inputs['Roughness'].default_value = 0.6 if hole else 0.32; b.inputs['Specular IOR Level'].default_value = 0.1 if hole else 0.5


def set_mannequin_material(bpy, mat, tex_path):
    mat.use_nodes = True; nt = mat.node_tree; N = nt.nodes; Lk = nt.links; N.clear()
    out = N.new('ShaderNodeOutputMaterial'); b = N.new('ShaderNodeBsdfPrincipled'); Lk.new(b.outputs['BSDF'], out.inputs['Surface'])
    uv = N.new('ShaderNodeUVMap'); uv.uv_map = 'UVMap'; im = N.new('ShaderNodeTexImage'); im.image = bpy.data.images.load(str(tex_path), check_existing=False); im.interpolation = 'Cubic'
    Lk.new(uv.outputs['UV'], im.inputs['Vector']); Lk.new(im.outputs['Color'], b.inputs['Base Color'])
    b.inputs['Roughness'].default_value = 0.8; b.inputs['Specular IOR Level'].default_value = 0.18


def composite_on_ground(fn, hex_):
    """Floating catalog output: the transparent render is kept as <name>_alpha.png and <name>.png becomes the garment alpha-over a flat ground of
    exactly `hex_` (Direction C: #F3EEE7 mineral bone).  Done on the written pixels, so the ground hex is exact whatever the view transform does."""
    from PIL import Image
    fn = Path(fn); im = Image.open(fn); im.load(); alpha_fn = fn.with_name(fn.stem + '_alpha.png'); im.save(alpha_fn)
    rgba = np.asarray(im.convert('RGBA')).astype(np.float64) / 255.0; a = rgba[..., 3:4]
    g = np.array([int(hex_.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0
    out = rgba[..., :3] * a + g * (1 - a)
    Image.fromarray(np.round(out * 255).astype(np.uint8), 'RGB').save(fn); return alpha_fn


def add_print_uv(bpy, ob, src='UVMap_Fabric', dst='UVMap_Print'):
    """Copy of the flat-pattern UVs in which every face reads the right way round from outside: a face whose UV frame is mirrored relative to its
    normal (back panel, one sleeve) gets u -> -u.  Only valid for a SEAMLESS repeat (no absolute placement), which is what it is used for."""
    me = ob.data
    if src not in me.uv_layers: return 0
    us = me.uv_layers[src]; ud = me.uv_layers.get(dst) or me.uv_layers.new(name=dst); flipped = 0
    for poly in me.polygons:
        li = list(poly.loop_indices); vi = [me.loops[l].vertex_index for l in li]
        if len(li) < 3: continue
        p0, p1, p2 = (np.array(me.vertices[v].co) for v in vi[:3]); t0, t1, t2 = (np.array(us.data[l].uv) for l in li[:3])
        e1, e2 = p1 - p0, p2 - p0; d1, d2 = t1 - t0, t2 - t0; det = d1[0] * d2[1] - d1[1] * d2[0]
        mirrored = False
        if abs(det) > 1e-14:
            T = (e1 * d2[1] - e2 * d1[1]) / det; B = (e2 * d1[0] - e1 * d2[0]) / det; mirrored = float(np.dot(np.cross(T, B), np.array(poly.normal))) < 0
        for l in li:
            u, v = us.data[l].uv; ud.data[l].uv = (-u, v) if mirrored else (u, v)
        flipped += mirrored
    return flipped


def bbox_of(bpy, objs):
    pts = np.array([o.matrix_world @ v.co for o in objs for v in o.data.vertices]); return pts.min(0), pts.max(0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--manifest', required=True); ap.add_argument('--out-dir', default=None); ap.add_argument('--res', default=None); ap.add_argument('--samples', type=int, default=None)
    ap.add_argument('--only', default=None, help='comma list of colourway ids'); ap.add_argument('--views', default=None, help='comma list of view names'); ap.add_argument('--modes', default=None, help='floating,mannequin')
    ap.add_argument('--hi-res', action='store_true', help='final master export: manifest render.hi_res_resolution (default 2400x3000); --res wins')
    ap.add_argument('--device', default='CPU', choices=['CPU', 'GPU'], help='GPU: Cycles on OptiX/CUDA if present (falls back to CPU)')
    ap.add_argument('--keep-blend', action='store_true', help='save the last prepared scene next to the renders for inspection')
    ap.add_argument('--depth16', action='store_true', help='16-bit PNG (colour calibration measurements)')
    ap.add_argument('--calibration', default=None, help='colour calibration JSON (components/color_calibration.py); default: manifest render.calibration')
    a = ap.parse_args(); man = json.loads(Path(a.manifest).read_text()); rd = man['render']
    import bpy
    master = ROOT / man['archetype']  # absolute paths override ROOT
    if not master.exists(): sys.exit(f'archetype master not found: {master}. Run engine/build_archetype.py first.')
    res = a.res or ('x'.join(str(x) for x in rd.get('hi_res_resolution', [2400, 3000])) if a.hi_res else None)
    RX, RY = [int(x) for x in res.lower().split('x')] if res else [int(x) for x in rd['resolution']]; samples = a.samples or rd['samples']
    out_dir = (Path(a.out_dir) if a.out_dir else ROOT / 'assets' / 'renders' / man['sku']).resolve()   # absolute: Blender and PIL must agree on it; out_dir.mkdir(parents=True, exist_ok=True)
    bundle = ROOT / man['artwork']['bundle'] if man.get('artwork', {}).get('bundle') else None
    bman = json.loads((bundle / 'uv_bundle_manifest.json').read_text()) if bundle and (bundle / 'uv_bundle_manifest.json').exists() else {'panels': {}}
    bpy.ops.wm.open_mainfile(filepath=str(master)); sc = bpy.context.scene
    try: bpy.ops.preferences.addon_enable(module='cycles')
    except Exception: pass
    tee = bpy.data.objects['Tee']; mann = bpy.data.objects['Mannequin']
    gcol = bpy.data.collections['Garment']; mcol = bpy.data.collections['Mannequin']
    mats = {m.name: m for ob in gcol.objects if ob.type == 'MESH' for m in ob.data.materials if m}; assert {'Main_Body', 'Collar_Rib'} <= set(mats), list(mats)
    spec_path = ROOT / man['archetype_spec'] if man.get('archetype_spec') else None
    cons = json.loads(spec_path.read_text()).get('construction', {}) if spec_path and spec_path.exists() else {}
    # ---- studio: pivot + camera + four area lights parented to the pivot (lighting is constant relative to the camera at every orbit angle)
    piv = bpy.data.objects.new('piv', None); sc.collection.objects.link(piv); tgt = bpy.data.objects.new('tgt', None); sc.collection.objects.link(tgt)
    cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.lens = rd['lens_mm']; cam.data.sensor_width = 36; cam.parent = piv
    lights = []

    def area(name, loc, en, size, col=(1, 1, 1)):
        d = bpy.data.lights.new(name, 'AREA'); d.energy = en; d.size = size; d.color = col; o = bpy.data.objects.new(name, d); sc.collection.objects.link(o); o.parent = piv; o.location = loc
        t = o.constraints.new('TRACK_TO'); t.target = tgt; t.track_axis = 'TRACK_NEGATIVE_Z'; t.up_axis = 'UP_Y'; lights.append((d, en))
    area('key', (-1.3, -2.0, 0.9), 260, 2.4); area('fill', (1.8, -1.7, -0.1), 70, 3.0, (0.95, 0.97, 1.0)); area('rim', (0.5, 1.9, 0.9), 320, 1.6, (1.0, 0.96, 0.9)); area('top', (0, -0.4, 1.7), 150, 1.8)
    w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True; bg = w.node_tree.nodes['Background']; bg.inputs[0].default_value = (0.84, 0.835, 0.82, 1); bg.inputs[1].default_value = 0.06
    r = sc.render; r.engine = 'CYCLES'; r.film_transparent = True; r.image_settings.file_format = 'PNG'; r.image_settings.color_mode = 'RGBA'; r.image_settings.color_depth = '16' if a.depth16 else '8'
    c = sc.cycles; c.device = 'CPU'; c.samples = samples; c.use_denoising = True; c.max_bounces = 6
    if a.device == 'GPU':                                                                            # opt-in (Colab L4); default stays CPU. render_log records cycles_device.
        cp = bpy.context.preferences.addons['cycles'].preferences
        for backend in ('OPTIX', 'CUDA'):
            try:
                cp.compute_device_type = backend; cp.get_devices()
                gpus = [d for d in cp.devices if d.type == backend]
                if gpus:
                    for d in cp.devices: d.use = d.type == backend
                    c.device = 'GPU'; print('cycles GPU:', backend, [d.name for d in gpus], flush=True); break
            except Exception as e: print('cycles backend', backend, 'unavailable:', e, flush=True)
        if c.device != 'GPU': print('WARNING: --device GPU requested but no OptiX/CUDA device found; rendering on CPU', flush=True)
    try: c.denoiser = 'OPENIMAGEDENOISE'
    except Exception: pass
    sc.view_settings.view_transform = rd.get('view_transform', 'AgX'); sc.view_settings.exposure = rd.get('exposure', 0.0)          # 'Standard' keeps saturated brand colours true; AgX lifts them toward coral
    cal_p = a.calibration or rd.get('calibration'); CAL = json.loads((ROOT / cal_p if not Path(cal_p).is_absolute() else Path(cal_p)).read_text()) if cal_p else None
    if CAL:                                                                                          # a calibration owns the view settings it was solved for
        sc.view_settings.view_transform = CAL['view_transform']; sc.view_settings.exposure = CAL.get('exposure', 0.0)
    if sc.view_settings.view_transform == 'AgX': sc.view_settings.look = (CAL or rd).get('look', 'AgX - High Contrast')
    PRINT = man['material'].get('print')
    if PRINT:
        for ob in gcol.objects:
            if ob.type == 'MESH': print('UVMap_Print on', ob.name, ': mirrored faces flipped', add_print_uv(bpy, ob))
    gmin, gmax = bbox_of(bpy, [tee]); H_REF = 0.62

    def frame(mode):
        if mode == 'mannequin':
            z0, z1 = rd['mannequin']['frame_z']; zc = (z0 + z1) / 2; Hh = z1 - z0; ctr = np.array([0.0, -0.03, zc])
        else:
            ctr = (gmin + gmax) / 2; Hh = gmax[2] - gmin[2]
        k = Hh / H_REF; piv.location = ctr; tgt.location = ctr; piv.scale = (k, k, k)
        cam.location = (0, -(rd['lens_mm'] * 1.30 * H_REF / 36), 0); cam.rotation_euler = (math.radians(90), 0, 0)
        for d, en in lights: d.energy = en * k * k          # same irradiance when the rig is scaled up for the full figure
        return ctr

    def orbit(deg): piv.rotation_euler = (0, 0, math.radians(-deg))                               # 0 = front, positive = camera moves round to the wearer's right; 180 = back

    colourways = [cw for cw in man['colorways'] if not a.only or cw['id'] in a.only.split(',')]
    modes_allowed = a.modes.split(',') if a.modes else None; log = []
    # Completion contract: render_log.json is (re)written only after every requested view rendered, and lists each image's sha256.
    # While a run is in progress, or after it crashed, render_log.inprogress.json is present: the folder may then mix this run's
    # images with an earlier run's, and the previous render_log.json (if any) does not describe them. Check the sha256 values.
    out_dir.mkdir(parents=True, exist_ok=True)              # (the mkdir on the out_dir line above is inside its comment and never ran)
    marker = out_dir / 'render_log.inprogress.json'
    write_json_atomic(marker, dict(sku=man['sku'], started=time.strftime('%Y-%m-%dT%H:%M:%S'), manifest=str(Path(a.manifest).resolve()), pid=os.getpid()))
    mfin = man.get('mannequin', {}); wire_src = ROOT / mfin.get('wire_texture', 'assets/archetypes/mannequin_wire_2048.png')
    finishes = json.loads((ROOT / 'configs' / 'mannequin_finishes.json').read_text())['finishes']
    CAL_BY_HEX = {cw_['main_body'].upper(): np.asarray(CAL['colorways'][cw_['id']]['main_body_linear'], float)       # a rib / trim in the same yarn colour as some
                  for cw_ in man['colorways'] if CAL and cw_['id'] in CAL.get('colorways', {})}                       # colourway's body gets that body's calibrated albedo

    def trim_lin(hex_): return CAL_BY_HEX.get(hex_.upper(), hex_to_lin(hex_))
    for cw in colourways:
        body = hex_to_lin(cw['main_body']); dark = luminance(body) < rd.get('dark_fabric_luminance', 0.15)       # ink plate choice: from the BRAND colour
        if CAL and cw['id'] in CAL.get('colorways', {}): body = np.asarray(CAL['colorways'][cw['id']]['main_body_linear'], float)   # calibrated render albedo
        ink = cw.get('ink', 'auto'); use_dark_plate = dark if ink == 'auto' else (ink == 'light')
        art = {}
        for side in ('front', 'back'):
            p = bman['panels'].get(side)
            if p: art[side] = (ROOT / p['albedo_for_dark_shirt' if use_dark_plate else 'albedo_for_light_shirt'], ROOT / p['height_map'])
        look = man['material']; knit = look['knit']
        ptile = None
        if PRINT and cw.get('print_ink'):
            tw, th = PRINT['tile_size_m']; ptile = dict(path=ROOT / PRINT['tile'], scale_uv=(0.30 / tw, 0.30 / th), ink_lin=trim_lin(cw['print_ink']))   # UVMap_Fabric = metres / 0.30
        build_fabric_material(bpy, mats['Main_Body'], body, look, knit, False, art, print_tile=ptile)
        if look.get('seam_groove', True): overlay_line_mask(mats['Main_Body'], seam_mask(mats['Main_Body']), darken=look.get('seam_darken', 0.22), relief=-0.8)
        build_fabric_material(bpy, mats['Collar_Rib'], trim_lin(cw['collar_rib']), look, knit, True, print_tile=ptile if cw.get('collar_print') else None)   # camp collar: same lattice as the body
        if 'Placket' in mats:                                                    # 3D placket (construction component): rib knit + top-stitching
            build_fabric_material(bpy, mats['Placket'], trim_lin(cw.get('placket', cw['collar_rib'])), look, knit, True, uv_name='UVMap_Placket',
                                  print_tile=dict(ptile, uv='UVMap_Placket', rot_deg=look.get('placket_print_rot_deg', 0.0), scale_uv=tuple(x * look.get('placket_print_scale', 1.0) for x in ptile['scale_uv'])) if ptile and cw.get('placket_print') else None)   # off by default: UVMap_Placket is not metric in both axes (open item)   # UVMap_Placket is also metres / 0.30
            overlay_line_mask(mats['Placket'], placket_stitch_mask(mats['Placket'], cons.get('placket', {}).get('length_m', 0.13), cons.get('placket', {}).get('width_m', 0.04)), darken=0.16, relief=0.5)
        for nm, hole in (('Button', False), ('Button_Hole', True)):
            if nm in mats: build_button_material(bpy, mats[nm], hex_to_lin(cw.get('button', cw.get('placket', cw['collar_rib']))), hole)
        for slot, key in (('Left_Cuff_Rib', 'left_cuff_rib'), ('Right_Cuff_Rib', 'right_cuff_rib')):          # cuffs exist only on archetypes that have them (the polo has plain hems)
            if slot in mats: build_fabric_material(bpy, mats[slot], trim_lin(cw[key]), look, knit, True)
        for rr in cw.get('renders', man['renders']):
            mode = rr['mode']
            if modes_allowed and mode not in modes_allowed: continue
            if rd.get('floating_only') and mode != 'floating': print('SKIP', cw['id'], mode, '(render.floating_only: the mannequin is a hidden collider only)'); continue
            fid = None                                                                      # mannequin finish id, recorded per render in render_log.json
            if mode == 'mannequin':
                fid = rr.get('finish', mfin.get('finish', 'M2')); tex = recolour_wire_texture(wire_src, finishes[fid]['hex'], out_dir / f'_skin_{fid}.png')
                set_mannequin_material(bpy, mann.data.materials[0], tex)
            mcol.hide_render = (mode != 'mannequin'); mcol.hide_viewport = mcol.hide_render
            ctr = frame(mode); ar = RX / RY
            FR = rd.get('framing')                                                          # measured framing (components/camera_framing.py) when configured
            r.resolution_x, r.resolution_y = (RX, RY) if FR else ((int(RX * mfin.get('res_scale_x', 0.85)), int(RY * mfin.get('res_scale_y', 0.95))) if mode == 'mannequin' else (RX, RY))
            if FR:
                sys.path.insert(0, str(ROOT / 'engine/components')); import camera_framing as CF
                SUBJ = CF.subject_points([o for o in gcol.objects if o.type == 'MESH'], mann, 'floating' if mode == 'floating' else FR.get('mannequin_profile', 'torso'), FR.get('below_hem_m', 0.12))
            for ang in rr['angles']:
                name = ang if isinstance(ang, str) else None
                view = next((v for v in rd['views'] if v['angle'] == ang), None) if not isinstance(ang, str) else None
                vname = (view or {}).get('name', f'a{int(ang)}');
                if a.views and vname not in a.views.split(','): continue
                if FR: frame(mode)
                orbit(ang); fn = out_dir / f"{man['sku']}_{cw['id']}_{mode}_{vname}.png"; r.filepath = str(fn)
                frep = CF.solve(sc, cam, piv, tgt, SUBJ, FR.get('target_occupancy', 0.78), FR.get('max_width', 0.92), light_datas=[(d, d.energy) for d, _ in lights]) if FR else None
                t = time.time(); bpy.ops.render.render(write_still=True); dt = time.time() - t
                if FR and mode == 'floating': frep['measured_alpha'] = CF.measure_alpha(fn)
                ground = rd.get('background_hex') if mode == 'floating' and not a.depth16 else None     # 16-bit runs are measurements: keep them raw
                if ground: composite_on_ground(fn, ground)
                log.append(dict(background_hex=ground, framing=frep, view_transform=sc.view_settings.view_transform, look=sc.view_settings.look, calibrated=bool(CAL and cw['id'] in CAL.get('colorways', {})), file=fn.name, colorway=cw['id'], mode=mode, finish=fid, finish_hex=finishes[fid]['hex'] if fid else None, angle=ang, seconds=round(dt, 1), res=[r.resolution_x, r.resolution_y], samples=samples, ink_plate='dark' if use_dark_plate else 'light',
                                sha256=file_sha256(fn)))                              # final image bytes, after composite_on_ground
                print('RENDER', fn.name, round(dt, 1), 's', flush=True)
    prov = render_provenance(dict(master=master, manifest=Path(a.manifest).resolve(), archetype_spec=spec_path, calibration=(ROOT / cal_p if not Path(cal_p).is_absolute() else Path(cal_p)) if cal_p else None,
                                  artwork_bundle=(bundle / 'uv_bundle_manifest.json') if bundle else None, render_code=Path(__file__).resolve()), bpy)
    write_json_atomic(out_dir / 'render_log.json', dict(sku=man['sku'], archetype=man['archetype'], complete=True, provenance=prov, renders=log))
    marker.unlink(missing_ok=True)
    if a.keep_blend: bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / '_scene_last.blend'))


if __name__ == '__main__':
    main()
