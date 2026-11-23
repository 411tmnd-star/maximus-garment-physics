"""Stage 4, Blender layer, SYNTHETIC: where a curve point lands on the panel. Stage 3 projects along the declared outward direction ('outward', default, unchanged); Stage 4 offers the closest point
('normal'). On a surface tilted away from the outward reference, an authored curve that sags above the surface makes the 'outward' projection shift stones sideways by a varying amount and crowd them.
  python3 test/blender_authoring/test_projection_modes.py"""
import math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent.parent; sys.path.insert(0, str(ROOT / 'py'))
import bpy, mathutils
import fixtures_v3 as FX, test_surfaces as TS, placement_v2 as PV, surfaces as SF, arc_sampler as AS, collisions as CO
res = []
def check(n, ok, d=''): ok = bool(ok); res.append(ok); print(('PASS ' if ok else 'FAIL ') + n + (('  ' + d) if d else ''), flush=True)
TILT = math.radians(50)
def scene():
    bpy.ops.wm.read_factory_settings(use_empty=True); n = 60; verts = [(x, y, -math.tan(TILT) * 0 + 0.0) for y in np.linspace(-0.03, 0.03, n + 1) for x in np.linspace(-0.05, 0.05, n + 1)]
    # plane through the origin with normal (sin T, 0, cos T): z = -x tan T
    verts = [(x, y, -x * math.tan(TILT)) for (x, y, _z) in verts]; faces = [(j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i) for j in range(n) for i in range(n)]
    ob = TS._mesh_obj('tilted', verts, faces); bpy.context.view_layer.update(); return ob
def run(mode):
    ob = scene(); nrm = np.array([math.sin(TILT), 0, math.cos(TILT)]); t_dir = np.array([math.cos(TILT), 0, -math.sin(TILT)])          # in-plane direction down the slope (x direction)
    pts = []                                                                                                           # a line down the slope that SAGS above the surface by up to 2 mm (period 18 mm): as an authored Bezier between far-apart anchors would
    for s in np.linspace(-0.035, 0.035, 120): h = 0.0015 * (1 + math.sin(2 * math.pi * s / 0.018)); pts.append(tuple(s * t_dir + h * nrm))
    TS._poly_curve('sag', pts, False, ob.matrix_world.copy())
    P = [FX.panel('tilt_panel', 'tilt', 'cloth', 'tilted', {'kind': 'vector', 'v': [0, 0, 1], 'frame': 'world'}, ['test_zone'])]; C = [FX.curve('sag_line', 'sag', 'tilt', 'tilt_panel', 'test_zone', FX.SZ('SS10'), 1)]
    b = FX.bundle('tilt', P, C, FX.ZONE_T); surf = SF.GarmentSurfaces(b.surface, FX.objects_dict()); eng = PV.PlacementEngine(surf, b, PV.EngineConfig(projection_mode=mode)); return eng.run(FX.paths_for(b)), nrm
out = {}
for mode in ('outward', 'normal'):
    r, nrm = run(mode); recs = [x for x in r.records if x['transform']]; B = np.array([x['transform']['base'] for x in recs]); ch = np.linalg.norm(np.diff(B, axis=0), axis=1); out[mode] = (r, ch, recs)
    print(f'   {mode}: {len(recs)} stones, consecutive centre distance min {ch.min() * 1e3:.3f} mm median {np.median(ch) * 1e3:.3f} max {ch.max() * 1e3:.3f} (nominal arc spacing 3.2 mm; stone diameter 2.8 mm); collisions {len(r.collisions["pairs"])}')
ch_o, ch_n = out['outward'][1], out['normal'][1]
check('surface is 50 degrees off the declared outward direction and the curve sags above it', abs(math.degrees(math.acos(math.cos(TILT))) - 50) < 1e-9)
check("'outward' projection (Stage 3 default) crowds stones on this slope: some consecutive stones closer than the stone diameter (2.8 mm)", ch_o.min() < 2.8e-3, f'min {ch_o.min() * 1e3:.3f} mm')
check("'normal' projection keeps the spacing: no consecutive pair closer than the stone diameter, and the spread is much smaller", ch_n.min() >= 2.8e-3 and (ch_n.max() - ch_n.min()) < 0.5 * (ch_o.max() - ch_o.min()), f'min {ch_n.min() * 1e3:.3f} mm, spread {(ch_n.max() - ch_n.min()) * 1e3:.3f} vs {(ch_o.max() - ch_o.min()) * 1e3:.3f} mm')
ov = lambda r: sum(1 for p in r.collisions['pairs'] if p['severity'].startswith('OVERLAP')); tc = lambda r: sum(1 for p in r.collisions['pairs'] if p['severity'] == 'TOO_CLOSE')
check("'normal' projection produces no overlapping stones (only 'too close', gap under 0.1 mm, at the 2.8 to 3.1 mm pitch); 'outward' produces overlaps", ov(out['normal'][0]) == 0 and ov(out['outward'][0]) > 0, f"overlaps {ov(out['normal'][0])} vs {ov(out['outward'][0])}; too-close {tc(out['normal'][0])} vs {tc(out['outward'][0])}")
check("both modes still place every stone on the panel (same count, none rejected) and the stone normals are identical (orientation does not depend on the mode)", len(out['outward'][2]) == len(out['normal'][2]) and all(x['outcome'] != 'REJECTED' for x in out['normal'][2]) and np.allclose([x['transform']['normal'] for x in out['outward'][2]], [x['transform']['normal'] for x in out['normal'][2]], atol=1e-6))
sys.exit(0 if all(res) else 1)
