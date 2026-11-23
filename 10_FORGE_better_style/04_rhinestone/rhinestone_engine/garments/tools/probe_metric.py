import sys; sys.dont_write_bytecode = True
import numpy as np, bpy
import stones_core as C
bpy.ops.wm.open_mainfile(filepath=sys.argv[1], load_ui=False)
tee = C.Mesh('Tee'); tee.outer = np.load(sys.argv[2])
for key, layer in (('front', 'UVMap_Art_Front'), ('back', 'UVMap_Art_Back')):
    uvt = tee.loop_uv(layer)[tee.TL]; inside = np.all((uvt >= 0) & (uvt <= 1), axis=(1, 2)); tee.build_uv(key, uvt, tee.outer & inside)
    print(key)
    for v in (0.9, 0.7, 0.5, 0.3, 0.1):
        row = []
        for u in (0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9):
            h = tee.uv_to_surface(key, (u, v))
            row.append('   --    ' if h is None else '%.2f/%.2f' % (np.linalg.norm(h[2]) / 0.638, np.linalg.norm(h[3]) / 0.700))
        print(' v=%.1f ' % v, ' '.join(row))
    h = tee.uv_to_surface(key, (0.5, 0.5)); print('  centre', h[0].round(3), 'dPu', C.unit(h[2]).round(2), 'dPv', C.unit(h[3]).round(2))
uvp, mir = C.print_uv_tri(tee); P = tee.V[tee.T]; o = tee.outer
a3 = tee.area2[o] / 2; d1, d2 = uvp[o][:, 1] - uvp[o][:, 0], uvp[o][:, 2] - uvp[o][:, 0]; a2 = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / 2 * 0.09
r = np.sqrt(a3 / np.maximum(a2, 1e-18)); print('print-UV linear scale 3D/UV: median %.3f p5 %.3f p95 %.3f' % (np.median(r), np.percentile(r, 5), np.percentile(r, 95)))
print('mirrored share', mir[o].mean())
