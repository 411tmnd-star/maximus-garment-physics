"""REQ-01 step 3: regenerate the collar through collar_builder.build_leaf_object, unseated (original logic) and seated (patched logic).
Never saves the input masters.  Each regenerated collar is written to its own small .blend (collar object only) in <outdir>.

  python -I req01_regen.py <outdir> <spec.json> <master.blend> [...]          (bpy module + scipy, Python 3.13)

The simulated COLLAR_ROWS of the original build are not stored in the masters, so the stand's control rows are RECONSTRUCTED from the baked collar:
per 1.5-degree column round the neck, base B = mean of the bottom-ring vertices, top T = mean of the highest pure-stand vertices; columns with no stand
(the front opening) are filled round the ring by periodic interpolation.  foldover_leaf only reads row 0 and the last row, so (B, T) is sufficient input.
Both variants are built from the SAME reconstructed rows, so the A/B difference is exactly the seating change."""
import sys, os, json, hashlib
import numpy as np
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import collar_builder as CB

outdir, spec_path, files = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), [os.path.abspath(a) for a in sys.argv[3:]]
os.makedirs(outdir, exist_ok=True)
SPECS = json.load(open(spec_path))
NCOL = 240


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''): h.update(ch)
    return h.hexdigest()


def arr(o):
    m = o.data; a = np.empty(len(m.vertices) * 3); m.vertices.foreach_get('co', a); return a.reshape(-1, 3)


def weights(o, name):
    gi = o.vertex_groups[name].index
    return np.array([next((g.weight for g in v.groups if g.group == gi), 0.0) for v in o.data.vertices])


def reconstruct_rows(C, n_rows=5):
    c = arr(C); ws = weights(C, 'collar_stand'); wl = weights(C, 'collar_leaf'); idx = np.flatnonzero((ws > 0) & (wl == 0)); P = c[idx]; cen = P[:, :2].mean(0)
    th = np.degrees(np.arctan2(P[:, 0] - cen[0], -(P[:, 1] - cen[1]))); col = np.floor((th + 180.0) / (360.0 / NCOL)).astype(int).clip(0, NCOL - 1)
    B = np.full((NCOL, 3), np.nan); T = np.full((NCOL, 3), np.nan)
    for k in np.unique(col):
        m = np.flatnonzero(col == k); z = P[m, 2]
        B[k] = P[m[z <= z.min() + 0.001]].mean(0); T[k] = P[m[z >= z.max() - 0.001]].mean(0)
    have = ~np.isnan(B[:, 0]); jj = np.arange(NCOL); jh = np.flatnonzero(have)
    for X in (B, T):
        for d in range(3): X[:, d] = np.interp(jj, jh, X[jh, d], period=NCOL)
    lam = np.linspace(0, 1, n_rows)[:, None, None]
    return B[None] + (T - B)[None] * lam, int((~have).sum())


summary = {}
for f in files:
    h0 = sha(f); name = os.path.basename(f).replace('.blend', '')
    key = 'camp_shirt' if 'camp' in name else 'rugby_polo'; cfg = dict(SPECS[key]); cfg['thickness_m'] = cfg.get('thickness_m', 0.002)
    bpy.ops.wm.open_mainfile(filepath=f, load_ui=False)
    Tee, Old, Man = bpy.data.objects['Tee'], bpy.data.objects['collar'], bpy.data.objects['Mannequin']
    Man.data.calc_loop_triangles(); bvh = BVHTree.FromPolygons([tuple(v.co) for v in Man.data.vertices], [tuple(t.vertices) for t in Man.data.loop_triangles])
    rows, n_fill = reconstruct_rows(Old)
    Old.name = 'collar_baked_original'
    entry = dict(input_sha256=h0, reconstructed_columns=NCOL, columns_without_stand_filled=n_fill, cfg=cfg)
    for variant, seat in (('unseated', False), ('seated', True)):
        ob, rep = CB.build_leaf_object(bpy, Tee, rows.copy(), cfg, name='collar_' + variant, body_bvh=bvh, seat_stand=seat)
        out = os.path.join(outdir, '%s_collar_%s.blend' % (key, variant)); bpy.data.libraries.write(out, {ob}, fake_user=True); assert os.path.getsize(out) > 1000, 'library write failed: ' + out
        entry[variant] = dict(file=out, report={k: (v if not isinstance(v, np.generic) else v.item()) for k, v in rep.items()})
        print(key, variant, json.dumps(entry[variant]['report'])[:600], flush=True)
    summary[name] = entry
    assert sha(f) == h0, 'input master changed!'
json.dump(summary, open(os.path.join(outdir, 'regen_summary.json'), 'w'), indent=1)
