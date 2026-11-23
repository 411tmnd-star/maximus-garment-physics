import numpy as np, sys
key = sys.argv[1]
a = np.load('out_fbl_008/baseline/%s_collar_leaf_debug.npz' % key); b = np.load('out_fbl_008/dev/collar_leaf_debug.npz')
np.set_printoptions(linewidth=250, precision=1, suppress=True)
for nm, d in (('baseline', a), ('seated', b)):
    R, P = d['leaf_raw'], d['leaf']; h_raw = 1e3 * np.linalg.norm(R[4] - R[0], axis=1); h = 1e3 * np.linalg.norm(P[4] - P[0], axis=1); dz = 1e3 * (P[4, :, 2] - P[0, :, 2])
    print(nm, 'raw |T-B| min/max %.2f/%.2f' % (h_raw.min(), h_raw.max()), ' after body+lift |T-B| min/max %.2f/%.2f' % (h.min(), h.max()))
    print('  |T-B| every 4th col:', h[::4]); print('  dz    every 4th col:', dz[::4])
    print('  base z every 4th  :', 1e3 * P[0, ::4, 2])
