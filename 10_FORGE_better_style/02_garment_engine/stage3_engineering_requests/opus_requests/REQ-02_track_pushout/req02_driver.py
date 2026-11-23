"""REQ-02 driver: imports the INSTRUMENTED scratch copy, builds jacket(bracket)+pants via assemble() only (no decals, no export,
no .blend save: the builder writes nothing), and measures inside counts / clearance at every push-out stage with two tests.
blender -b --factory-startup --python-exit-code 1 --python req02_driver.py -- <variant> [--full]
variants: BASE, V1_nolockz, V2_nosmooth, V3_nolockz_nosmooth, V4_6passes, V4b_6passes_literal_smooth, V5_floor9p5
"""
import os
import sys, json, math, time, importlib.util
from pathlib import Path
import numpy as np
import bpy, bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

R = Path(os.path.normpath(os.path.expanduser('~/Desktop/better_style_execution_workspace/08_KNOWLEDGE/stage3_engineering/opus_requests/REQ-02_track_pushout')))
SCR = R / 'scratch' / 'repo'
NPZ = R / 'scratch' / 'npz'; NPZ.mkdir(parents=True, exist_ok=True)
RUNS = R / 'runs'; RUNS.mkdir(parents=True, exist_ok=True)
argv = sys.argv[sys.argv.index('--') + 1:]; VAR = argv[0]; FULL = '--full' in argv
t_start = time.time()

src = SCR / 'BETTER_STYLE_VISUAL_CURATION/14_TRACKSUIT_DUAL_STATE_VIDEO/scripts/build_garment_g0_instr.py'
spec = importlib.util.spec_from_file_location('bg0i', str(src)); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
for p in (mod.REPO, mod.GARMENTS_DIR, mod.TEX_DIR, Path(mod.SPEC['mannequin_blend'])):
    assert str(Path(p).resolve()).lower().startswith(str(SCR.resolve()).lower()), p

# ---------------------------------------------------------------- variant knobs (one change each)
if VAR == 'V1_nolockz': mod.LOCKZ_DISABLED = True
elif VAR == 'V2_nosmooth': mod.PUSH_SMOOTH = False
elif VAR == 'V3_nolockz_nosmooth': mod.LOCKZ_DISABLED = True; mod.PUSH_SMOOTH = False
elif VAR == 'V4_6passes': mod.PUSH_PASSES = 6; mod.PUSH_SMOOTH_UNTIL = 5          # smooth after every pass but the last (generalises it<2 of 3)
elif VAR == 'V4b_6passes_literal_smooth': mod.PUSH_PASSES = 6; mod.PUSH_SMOOTH_UNTIL = 2   # literal `it < 2`
elif VAR == 'V5_floor9p5': mod.SPEC['pushout_floor_m'] = 0.0095
else: assert VAR == 'BASE', VAR

EV = {}
def collector(name, payload):
    if name == 'torso_ring':
        if payload['k'] == 0: EV['torso_ring'] = []
        EV['torso_ring'].append(payload); return
    if name == 'sleeve_ring':
        key = 'sleeve_ring_' + payload['side']
        if payload['j'] == 0: EV[key] = []
        EV[key].append(payload); return
    if name == 'sleeve_stack': EV['sleeve_stack_' + payload['side']] = payload['stack']; return
    EV[name] = payload
    if isinstance(payload, np.ndarray) and payload.ndim == 2 and name != 'torso_stack': EV.setdefault('_order', []).append(name)
mod.PROBE = collector

rig, baked = mod.load_theta_fit_mannequin()
orig = bpy.data.objects[mod.SPEC['body_name']]
ref = mod.BodyRef(baked, orig)
bm = bmesh.new(); bm.from_mesh(baked.data); bm.verts.ensure_lookup_table(); body_bvh = BVHTree.FromBMesh(bm)
nb_edges = sum(1 for e in bm.edges if len(e.link_faces) == 1); nm_edges = sum(1 for e in bm.edges if len(e.link_faces) > 2); bm.free()
me = baked.data; me.calc_loop_triangles()
BV = np.array([v.co[:] for v in me.vertices]); TRI = np.array([t.vertices[:] for t in me.loop_triangles])
POLY_V = [tuple(p.vertices) for p in me.polygons]

def cat_of(n):
    if mod.HEAD_RE.match(n): return 'head'
    if mod.HAND_RE.match(n): return 'hand'
    if mod.FOOT_RE.match(n): return 'foot'
    if mod.ARM_RE.match(n): return 'arm'
    if mod.LEG_RE.match(n): return 'leg'
    return 'torso'
VCAT = np.array([cat_of(n) for n in ref.name])
def face_region(fi):
    cs = [VCAT[i] for i in POLY_V[fi]]; return max(set(cs), key=cs.count)

# ---------------------------------------------------------------- inside tests
TA, TB, TC = BV[TRI[:, 0]].astype(np.float64), BV[TRI[:, 1]].astype(np.float64), BV[TRI[:, 2]].astype(np.float64)
def gwn(P, chunk=48):
    """Generalized winding number (Jacobson 2013; van Oosterom-Strackee solid angle) of the body triangle soup at P."""
    out = np.zeros(len(P))
    for s in range(0, len(P), chunk):
        q = P[s:s + chunk, None, :]
        a = TA[None] - q; b = TB[None] - q; c = TC[None] - q
        la = np.linalg.norm(a, axis=2); lb = np.linalg.norm(b, axis=2); lc = np.linalg.norm(c, axis=2)
        num = np.einsum('ijk,ijk->ij', a, np.cross(b, c))
        den = la * lb * lc + np.einsum('ijk,ijk->ij', a, b) * lc + np.einsum('ijk,ijk->ij', b, c) * la + np.einsum('ijk,ijk->ij', c, a) * lb
        out[s:s + chunk] = 2 * np.arctan2(num, den).sum(axis=1) / (4 * math.pi)
    return out
GW_SIGN = 1.0 if gwn(np.array([[0.0, ref.cy(1.2), 1.2]]))[0] > 0 else -1.0
RAY_DIRS = [Vector(d).normalized() for d in ((0.31, 0.17, 0.93), (-0.62, 0.71, -0.33), (0.83, -0.41, 0.38))]
def ray_parity(p):
    votes = 0
    for d in RAY_DIRS:
        o = Vector(p); n = 0
        for _ in range(64):
            loc, _, _, _ = body_bvh.ray_cast(o, d, 5.0)
            if loc is None: break
            n += 1; o = loc + d * 1e-6
        votes += n & 1
    return votes >= 2

NEAR = 0.03
def classify(P, full=False):
    n = len(P); d = np.zeros(n); fi = np.zeros(n, int); nz = np.zeros(n); L = np.zeros((n, 3))
    for k in range(n):
        loc, nrm, idx, dist = body_bvh.find_nearest(Vector(P[k]))
        inside = (Vector(P[k]) - loc).dot(nrm) < 0
        d[k] = -dist if inside else dist; fi[k] = idx; nz[k] = nrm.z; L[k] = loc[:]
    ins_a = d < 0
    use_g = np.ones(n, bool) if full else ((np.abs(d) < NEAR) | ins_a)
    ins_b = np.zeros(n, bool); g = np.full(n, np.nan)
    if use_g.any(): g[use_g] = gwn(P[use_g]) * GW_SIGN; ins_b[use_g] = g[use_g] > 0.5
    far = ~use_g
    for k in np.nonzero(far)[0]: ins_b[k] = ray_parity(P[k])
    return dict(d=d, fi=fi, nz=nz, loc=L, ins_a=ins_a, ins_b=ins_b, gwn=g, n_gwn=int(use_g.sum()), n_ray=int(far.sum()))

pct = lambda a, q: round(float(np.percentile(a, q)) * 1000, 3) if len(a) else None
def summ(C, P):
    d = C['d']; dis = C['ins_a'] != C['ins_b']; Pd = P[dis]
    r = dict(n=len(d), inside_a_signtest=int(C['ins_a'].sum()), inside_b_independent=int(C['ins_b'].sum()),
             min_mm=pct(d, 0), p1_mm=pct(d, 1), p5_mm=pct(d, 5), within_6mm_a=int((d < 0.006).sum()),
             disagree=int(dis.sum()), a_only=int((C['ins_a'] & ~C['ins_b']).sum()), b_only=int((~C['ins_a'] & C['ins_b']).sum()),
             test_b_method=dict(gwn=C['n_gwn'], ray_parity_3dir=C['n_ray']))
    if dis.any(): r['disagree_z'] = [round(float(Pd[:, 2].min()), 3), round(float(Pd[:, 2].max()), 3)]; r['disagree_absx'] = [round(float(np.abs(Pd[:, 0]).min()), 3), round(float(np.abs(Pd[:, 0]).max()), 3)]
    return r

# ---------------------------------------------------------------- build + measure
RES = dict(variant=VAR, knobs=dict(PUSH_PASSES=mod.PUSH_PASSES, PUSH_SMOOTH=mod.PUSH_SMOOTH, PUSH_SMOOTH_UNTIL=mod.PUSH_SMOOTH_UNTIL,
           LOCKZ_DISABLED=mod.LOCKZ_DISABLED, floor_m=mod.SPEC['pushout_floor_m']),
           body=dict(verts=len(BV), tris=len(TRI), boundary_edges=nb_edges, nonmanifold_edges=nm_edges, gwn_sign=GW_SIGN, near_threshold_m=NEAR,
                     ray_dirs=[list(map(lambda x: round(x, 4), d)) for d in RAY_DIRS]))
base_npz = NPZ / 'BASE.npz'; base = np.load(base_npz) if (VAR != 'BASE' and base_npz.exists()) else None
save = {}
for kind in ('jacket', 'pants'):
    EV.clear()
    obj = mod.assemble(kind, 'bracket', rig, ref, body_bvh, None)
    order = EV['_order']; pre = [s for s in order if s != 'solidify']; n = len(EV['loft'])
    S = EV['solidify']; assert len(S) == 2 * n, (len(S), n)
    last = EV[pre[-1]]; outer_same = float(np.abs(S[:n] - last).max())
    G = np.array([v.co[:] for v in obj.data.vertices]); nf = int(obj['n_nylon_faces'])
    shell_idx = sorted({i for p in obj.data.polygons[:nf] for i in p.vertices})
    R_k = dict(n_single_layer=n, stages=pre + ['solidify_outer', 'solidify_inner'], solidify_outer_equals_last_stage_max_abs_m=outer_same,
               evaluator_shell_verts=len(shell_idx), props=EV['props'], chest_calibration=EV.get('chest_calibration'))
    # offset direction of the solidify copy relative to the body (positive = away from the body)
    stage_full = FULL and VAR == 'BASE'
    if VAR == 'BASE' or True:
        stages_to_eval = pre if VAR == 'BASE' else [pre[-1]]
        CL = {}
        for s in stages_to_eval: CL[s] = classify(EV[s], full=stage_full and s in ('loft',))
        CL['solidify_outer'] = classify(S[:n]); CL['solidify_inner'] = classify(S[n:], full=stage_full)
        R_k['stage_table'] = {s: summ(CL[s], EV[s] if s in EV else (S[:n] if s == 'solidify_outer' else S[n:])) for s in CL}
        Sfin_ins_a = np.r_[CL['solidify_outer']['ins_a'], CL['solidify_inner']['ins_a']]
        Sfin_ins_b = np.r_[CL['solidify_outer']['ins_b'], CL['solidify_inner']['ins_b']]
        R_k['final'] = dict(inside_a_total=int(Sfin_ins_a.sum()), inside_b_total=int(Sfin_ins_b.sum()),
                            outer_a=int(CL['solidify_outer']['ins_a'].sum()), inner_a=int(CL['solidify_inner']['ins_a'].sum()),
                            outer_b=int(CL['solidify_outer']['ins_b'].sum()), inner_b=int(CL['solidify_inner']['ins_b'].sum()),
                            pre_solidify_a=int(CL[pre[-1]]['ins_a'].sum()), pre_solidify_b=int(CL[pre[-1]]['ins_b'].sum()))
        off = S[n:] - S[:n]; to_body = CL['solidify_outer']['loc'] - S[:n]
        R_k['solidify_offset'] = dict(len_mm_median=round(float(np.median(np.linalg.norm(off, axis=1))) * 1000, 3),
                                      share_towards_body=round(float((np.einsum('ij,ij->i', off, to_body) > 0).mean()), 4))
        # disagreement z histogram on final
        for sh in ('solidify_outer', 'solidify_inner'):
            C = CL[sh]; P = S[:n] if sh == 'solidify_outer' else S[n:]; dis = C['ins_a'] != C['ins_b']
            R_k['stage_table'][sh]['disagree_z_hist_0p1m'] = {f'{b:.1f}': int(((P[dis, 2] >= b) & (P[dis, 2] < b + 0.1)).sum()) for b in np.arange(0.0, 1.6, 0.1) if ((P[dis, 2] >= b) & (P[dis, 2] < b + 0.1)).any()}
            R_k['stage_table'][sh]['disagree_detail'] = [dict(z=round(float(P[k, 2]), 4), absx=round(float(abs(P[k, 0])), 4), d_mm=round(float(C['d'][k]) * 1000, 2),
                                                              gwn=(round(float(C['gwn'][k]), 3) if not np.isnan(C['gwn'][k]) else None), a=bool(C['ins_a'][k]), b=bool(C['ins_b'][k]),
                                                              region=face_region(int(C['fi'][k]))) for k in np.nonzero(dis)[0][:60]]
        if VAR == 'BASE':
            # ---------- per-vertex attribution for final inside vertices (#4)
            seq = pre
            for test in ('a', 'b'):
                rows = []
                for sh in ('solidify_outer', 'solidify_inner'):
                    C = CL[sh]; P = S[:n] if sh == 'solidify_outer' else S[n:]
                    for k in np.nonzero(C['ins_' + test])[0]:
                        hist = [bool(CL[s]['ins_' + test][k]) for s in seq]
                        def entry(h):
                            if all(h): return 'loft'
                            lastout = max(i for i, x in enumerate(h) if not x)
                            return seq[lastout + 1] if lastout + 1 < len(seq) else None
                        e = entry(hist)
                        if sh == 'solidify_inner' and e is None:
                            # counterpart outer vertex outside: solidify made it inside; sub-attribute by when the outer clearance last fell < 3.5 mm
                            h2 = [bool(CL[s]['d'][k] < 0.0035) for s in seq]
                            sub = entry(h2) if h2[-1] else 'outer>=3.5mm'
                            stage = 'solidify'; substage = sub
                        else:
                            stage = e; substage = None
                        fi = int(C['fi'][k])
                        rows.append(dict(sheet=sh.split('_')[1], idx=int(k), z=round(float(P[k, 2]), 4), absx=round(float(abs(P[k, 0])), 4),
                                         d_mm=round(float(C['d'][k]) * 1000, 3), body_nz=round(float(C['nz'][k]), 4), region=face_region(fi),
                                         region_absx_heuristic=('arm' if abs(C['loc'][k][0]) > 0.21 else 'torso_or_leg'),
                                         stage=stage, substage=substage, history=''.join('1' if x else '0' for x in hist),
                                         in_lockz_band=bool(kind == 'pants' or abs(EV[seq[-1]][k, 0]) < 0.15)))
                ct = {}
                for r_ in rows: ct.setdefault(r_['stage'], {}).setdefault(r_['region'], 0); ct[r_['stage']][r_['region']] += 1
                R_k[f'attribution_test_{test}'] = dict(contingency_stage_x_region=ct, n=len(rows),
                    by_sheet={sh: sum(1 for r_ in rows if r_['sheet'] == sh) for sh in ('outer', 'inner')},
                    solidify_substage={k2: sum(1 for r_ in rows if r_['stage'] == 'solidify' and r_['substage'] == k2) for k2 in set(r_['substage'] for r_ in rows if r_['stage'] == 'solidify')},
                    nz_stats=dict(median=round(float(np.median([r_['body_nz'] for r_ in rows])), 3) if rows else None,
                                  share_nz_lt_m0p5=round(float(np.mean([r_['body_nz'] < -0.5 for r_ in rows])), 3) if rows else None,
                                  share_absnz_gt_0p7=round(float(np.mean([abs(r_['body_nz']) > 0.7 for r_ in rows])), 3) if rows else None),
                    z_range=[min(r_['z'] for r_ in rows), max(r_['z'] for r_ in rows)] if rows else None,
                    rows=rows)
            # ---------- moved counts / transitions per stage
            tr = {}
            for i in range(1, len(seq)):
                A0, A1 = CL[seq[i - 1]]['ins_a'], CL[seq[i]]['ins_a']
                tr[f'{seq[i - 1]}->{seq[i]}'] = dict(out_to_in=int((~A0 & A1).sum()), in_to_out=int((A0 & ~A1).sum()),
                                                     max_disp_mm=round(float(np.linalg.norm(EV[seq[i]] - EV[seq[i - 1]], axis=1).max()) * 1000, 2))
            R_k['transitions_test_a'] = tr
    save[f'{kind}_pre'] = EV[pre[-1]]; save[f'{kind}_shell'] = S; save[f'{kind}_obj'] = G
    if base is not None:
        R_k['vs_baseline'] = dict(max_disp_pre_solidify_mm=round(float(np.linalg.norm(EV[pre[-1]] - base[f'{kind}_pre'], axis=1).max()) * 1000, 2),
                                  max_disp_shell_mm=round(float(np.linalg.norm(S - base[f'{kind}_shell'], axis=1).max()) * 1000, 2),
                                  n_moved_gt_0p1mm=int((np.linalg.norm(EV[pre[-1]] - base[f'{kind}_pre'], axis=1) > 1e-4).sum()))
    if VAR == 'BASE':
        # ---------- radial ease vs normal clearance (#5)
        if kind == 'jacket':
            save['torso_stack'] = EV['torso_stack']
            RINGS = dict(torso=EV['torso_ring'], sleeve_L=EV['sleeve_ring_L'], torso_stack=EV['torso_stack'], sleeve_stack_L=EV['sleeve_stack_L'])
    RES[kind] = R_k
    for o in list(bpy.data.objects):
        if o.name.startswith('G0_'): bpy.data.objects.remove(o, do_unlink=True)

np.savez_compressed(NPZ / f'{VAR}.npz', **save)

# ---------------------------------------------------------------- compare instrumented BASE with the copy of the original master
if VAR == 'BASE':
    with bpy.data.libraries.load(str(R / 'scratch/orig_ref/bs_track_001_g0_master.blend'), link=False) as (src_, dst_):
        dst_.objects = ['G0_jacket_bracket', 'G0_pants']
    cmp = {}
    for o, k in zip(dst_.objects, ('jacket', 'pants')):
        Gm = np.array([(o.matrix_world @ v.co)[:] for v in o.data.vertices])
        cmp[k] = dict(n_master=len(Gm), n_instr=len(save[f'{k}_obj']), max_abs_diff_m=float(np.abs(Gm - save[f'{k}_obj']).max()) if len(Gm) == len(save[f'{k}_obj']) else None)
    RES['instrumented_vs_original_master'] = cmp

    # ---------- #5 ease vs clearance
    def hull_hit_normal(H, c, d):
        """Exact: edge of convex polygon H hit by ray c + t d; returns t, outward unit normal of that edge."""
        best = None
        for i in range(len(H)):
            p, q = H[i] - c, H[(i + 1) % len(H)] - c; e = q - p; det = -d[0] * e[1] + e[0] * d[1]
            if abs(det) < 1e-12: continue
            t = (-p[0] * e[1] + e[0] * p[1]) / det; s = (d[0] * p[1] - d[1] * p[0]) / det
            if t > 0 and -1e-9 <= s <= 1 + 1e-9 and (best is None or t > best[0]):
                nrm = np.array([e[1], -e[0]]); nrm /= np.linalg.norm(nrm)
                if np.dot(nrm, (p + q) / 2) < 0: nrm = -nrm
                best = (t, nrm)
        return best
    def poly_dist(H, x):
        """signed distance of 2D point x to convex polygon H (positive outside)."""
        ds = []; inside = True
        for i in range(len(H)):
            p, q = H[i], H[(i + 1) % len(H)]; e = q - p; t = np.clip(np.dot(x - p, e) / np.dot(e, e), 0, 1); ds.append(np.linalg.norm(x - (p + t * e)))
            if (e[0] * (x[1] - p[1]) - e[1] * (x[0] - p[0])) < 0: inside = False
        # H from hull2d is CCW (monotone chain) -> left of every edge = inside
        return -min(ds) if inside else min(ds)
    def ring_stats(e, alpha, c2d, c3d, kind_, key):
        ecos = e * np.cos(alpha)
        return dict(ring=key, e_mm_median=round(float(np.median(e)) * 1000, 2), e_mm_min=round(float(e.min()) * 1000, 2), e_mm_max=round(float(e.max()) * 1000, 2),
                    alpha_deg_median=round(float(np.degrees(np.median(alpha))), 2), alpha_deg_p95=round(float(np.degrees(np.percentile(alpha, 95))), 2),
                    ecos_mm_median=round(float(np.median(ecos)) * 1000, 2),
                    clear2d_hull_mm_median=round(float(np.median(c2d)) * 1000, 2), clear2d_minus_ecos_mm_median=round(float(np.median(c2d - ecos)) * 1000, 3),
                    clear3d_body_loft_mm_median=round(float(np.median(c3d)) * 1000, 2), clear3d_body_loft_mm_min=round(float(c3d.min()) * 1000, 2),
                    clear3d_over_e_median=round(float(np.median(c3d / np.maximum(e, 1e-6))), 3), clear3d_over_ecos_median=round(float(np.median(c3d / np.maximum(ecos, 1e-6))), 3),
                    share_clear3d_lt_ecos=round(float((c3d < ecos).mean()), 3), share_clear3d_lt_6mm=round(float((c3d < 0.006).mean()), 3), share_clear3d_lt_0=round(float((c3d < 0).mean()), 3))
    def d3(p):
        loc, nrm, _, dist = body_bvh.find_nearest(Vector(p)); return -dist if (Vector(p) - loc).dot(nrm) < 0 else dist
    E5 = dict(method='e = r_total - r_hull per polar direction (includes lateral chest ease); alpha = angle between polar ray D_i and outward normal of the hull edge it hits '
                     '(torso: Ht if w<0.5 else Hta, w = shoulder blend weight); clear2d = signed distance of the raw ring point to that hull polygon in the slice plane; '
                     'clear3d = signed nearest-face distance to the body of the SMOOTHED loft vertex (torso: 8 axial + 2 circ passes; sleeve: 3 axial + 1 circ) before push-out',
              torso=[], sleeve_L=[])
    TR = RINGS['torso']; TS = RINGS['torso_stack']
    allE, allA, all3, all2 = [], [], [], []
    for k in [int(round(x)) for x in np.linspace(0, 63, 10)]:
        rg = TR[k]; H = rg['Ht'] if rg['w'] < 0.5 else rg['Hta']; c = rg['c']; D = rg['D']
        e = rg['r_tot'] - rg['r_hull']; al = np.zeros(len(D)); c2 = np.zeros(len(D)); c3 = np.zeros(len(D))
        for i, d in enumerate(D):
            hh = hull_hit_normal(H, c, d); al[i] = math.acos(np.clip(np.dot(d, hh[1]), -1, 1))
            c2[i] = poly_dist(H, c + rg['r_tot'][i] * d); c3[i] = d3(TS[k, i])
        st = ring_stats(e, al, c2, c3, 'torso', f'k={k} z={rg["z"]:.3f} w={rg["w"]:.2f}'); E5['torso'].append(st)
        allE += list(e); allA += list(al); all3 += list(c3); all2 += list(c2)
    allE, allA, all3, all2 = map(np.array, (allE, allA, all3, all2))
    E5['torso_pooled'] = dict(n=len(allE), corr_clear3d_vs_ecos=round(float(np.corrcoef(all3, allE * np.cos(allA))[0, 1]), 3), corr_clear3d_vs_e=round(float(np.corrcoef(all3, allE)[0, 1]), 3),
                              corr_clear2d_vs_ecos=round(float(np.corrcoef(all2, allE * np.cos(allA))[0, 1]), 3),
                              ols_clear3d_on_ecos=[round(float(x), 4) for x in np.polyfit(allE * np.cos(allA), all3, 1)],
                              alpha_deg_median=round(float(np.degrees(np.median(allA))), 2), alpha_deg_p95=round(float(np.degrees(np.percentile(allA, 95))), 2))
    SL = RINGS['sleeve_L']; SS = RINGS['sleeve_stack_L']
    allE, allA, all3, all2 = [], [], [], []
    for j in [int(round(x)) for x in np.linspace(0, len(SL) - 1, 10)]:
        rg = SL[j]; H = rg['H']; c = rg['c2']; N = len(rg['r_hull']); ths = 2 * math.pi * np.arange(N) / N; D = np.stack([np.cos(ths), np.sin(ths)], 1)
        e = rg['ease']; al = np.zeros(N); c2 = np.zeros(N); c3 = np.zeros(N)
        for i, d in enumerate(D):
            hh = hull_hit_normal(H, c, d); al[i] = math.acos(np.clip(np.dot(d, hh[1]), -1, 1))
            c2[i] = poly_dist(H, c + (rg['r_hull'][i] + e[i]) * d); c3[i] = d3(SS[j, i])
        st = ring_stats(e, al, c2, c3, 'sleeve', f'j={j} u={rg["u"]:.3f}'); E5['sleeve_L'].append(st)
        allE += list(e); allA += list(al); all3 += list(c3); all2 += list(c2)
    allE, allA, all3, all2 = map(np.array, (allE, allA, all3, all2))
    E5['sleeve_pooled'] = dict(n=len(allE), corr_clear3d_vs_ecos=round(float(np.corrcoef(all3, allE * np.cos(allA))[0, 1]), 3), corr_clear3d_vs_e=round(float(np.corrcoef(all3, allE)[0, 1]), 3),
                               corr_clear2d_vs_ecos=round(float(np.corrcoef(all2, allE * np.cos(allA))[0, 1]), 3),
                               ols_clear3d_on_ecos=[round(float(x), 4) for x in np.polyfit(allE * np.cos(allA), all3, 1)],
                               alpha_deg_median=round(float(np.degrees(np.median(allA))), 2), alpha_deg_p95=round(float(np.degrees(np.percentile(allA, 95))), 2))
    RES['ease_vs_clearance'] = E5

RES['runtime_s'] = round(time.time() - t_start, 1)
json.dump(RES, open(RUNS / f'{VAR}.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
print('[req02] done', VAR, RES['runtime_s'], 's')
for kind in ('jacket', 'pants'):
    print(kind, json.dumps(RES[kind].get('final')), json.dumps(RES[kind].get('vs_baseline')))
