"""build_archetype.py  Stage 1 of the Two-Stage Garment Engine: compile a garment archetype ONCE.

  python engine/build_archetype.py --spec configs/archetypes/ringer_tee.json --work <dir> [--body body.blend] [--stages collider,setup,sim,bake]

Output: assets/archetypes/<name>_master.blend  (baked garment in collection `Garment`, posed display mannequin in collection `Mannequin`,
        4 material slots, 3 UV layers) + <name>_master.quality.json.  Nothing is saved if the quality gate fails.

Stages (each writes files into --work so a stage can be re-run alone):
  collider  pose the mannequin twice (phase-1 arms, phase-2 relaxed arms), bake both as one mesh with a shape key; bake the display mannequin
  setup     sewing panels incl. PRE-CLOSED collar/cuff rings, wrap round the body, normal-push clearance, sewing edges, UVs, cloth settings
  sim       two-phase drape: phase 1 arms fixed; phase 2 arms relax inward over a few frames, then settle. Stepped frame by frame.
  bake      snap sewn pairs, weld, polish (CorrectiveSmooth + Solidify inward + Subsurf), quality gate, assemble + save the master blend

Hard-won rules kept from the pilot (docs/HANDOFF_SUMMARY.md section 4): step frames one at a time; never snap vertices onto the body (push along a
smoothed normal); sewing force 25; seam vertex counts must match; Solidify even-offset OFF; weld at 2 mm after snapping sewn pairs to their midpoint.
"""
import argparse, json, math, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import pattern as PT
sys.path.insert(0, os.path.join(HERE, 'components'))
import sleeve_builder as SB


# ------------------------------------------------------------------------------------------------------------------ helpers
def load_spec(p):
    with open(p) as f: return json.load(f)


def body_normals(V, F):
    bn = np.zeros_like(V)
    for q in F:
        n = np.cross(V[q[1]] - V[q[0]], V[q[2]] - V[q[0]]) + np.cross(V[q[2]] - V[q[0]], V[q[3]] - V[q[0]])
        for i in q: bn[i] += n
    bn /= np.maximum(np.linalg.norm(bn, axis=1, keepdims=True), 1e-12)
    if np.mean(np.einsum('ij,ij->i', V - V.mean(0), bn)) < 0: bn = -bn
    return bn


def tri_areas(V, T):
    return np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1) / 2


def fan_triangles(faces):
    T = []
    for f in faces:
        for k in range(1, len(f) - 1): T.append((f[0], f[k], f[k + 1]))
    return np.array(T)


# ------------------------------------------------------------------------------------------------------------------ stage: collider
def stage_collider(spec, W, body_blend):
    import bpy, mathutils
    bpy.ops.wm.open_mainfile(filepath=body_blend)
    rig = bpy.data.objects['Human.rig']; body = bpy.data.objects['Human']

    def angle_down(n):
        pb = rig.pose.bones[n]; h = rig.matrix_world @ pb.head; t = rig.matrix_world @ pb.tail; d = (t - h).normalized()
        return math.degrees(math.acos(max(-1, min(1, d.dot(mathutils.Vector((0, 0, -1)))))))

    def rot_about_head(pb, deg):
        head = pb.head.copy(); R = mathutils.Matrix.Rotation(math.radians(deg), 4, 'Y')
        T = mathutils.Matrix.Translation(head); pb.matrix = T @ R @ T.inverted() @ pb.matrix; bpy.context.view_layer.update()

    def set_angle(name, target):
        pb = rig.pose.bones[name]
        for _ in range(12):
            cur = angle_down(name)
            if abs(cur - target) < 0.2: break
            step = min(abs(cur - target), 8.0); before = cur; rot_about_head(pb, step)
            if abs(angle_down(name) - target) > abs(before - target): rot_about_head(pb, -2 * step)
        return angle_down(name)

    arm_keys = ('clavicle', 'shoulder', 'upperarm', 'lowerarm', 'wrist', 'hand', 'finger', 'thumb', 'index', 'middle', 'ring', 'pinky')
    gi = {g.index: g.name for g in body.vertex_groups}
    isarm = np.zeros(len(body.data.vertices), bool)
    for v in body.data.vertices:
        best = max(v.groups, key=lambda g: g.weight, default=None)
        if best and any(k in gi[best.group] for k in arm_keys): isarm[v.index] = True

    out = {}; mann_saved = False
    CPOSE = spec['body'].get('canonical_pose'); RAMP = []
    if CPOSE:                                                    # canonical pose: phase 2 is a ramp of N real poses from the A-pose into the target pose
        sys.path.insert(0, os.path.join(HERE, 'components')); import pose_retarget as PRT
        pose_spec = PRT.load_poses()[CPOSE['pose_id']]; RT = PRT.Retargeter(rig); N = int(CPOSE.get('ramp_steps', 6))
    for tag, key in (('A', 'pose_phase1'), ('B', 'pose_phase2')):
        P = spec['body'][key]; got = {}
        if tag == 'B' and CPOSE:
            # hand-contact clearance (REPORTED, not corrected).  The cloth needs collision room on both sides of a hand that rests on the body
            # (2 x (collision distance + collider thickness) + 3 mm); with less, the drape pinches the cloth and it slips past the hand (POSE_01 on
            # the long polo: hem wrapped around the wrist).  Moving the wrist target out does not fix it -- the palm's orientation follows the
            # forearm and swings back onto the thigh (tried: 7 cm of offset, clearance still 1-4 mm) -- the fix needs wrist orientation control.
            from scipy.spatial import cKDTree
            cl0 = spec['cloth']; need_gap = 2 * (cl0['collision_distance'] + cl0['collider_thickness_outer']) + 0.003
            hand_keys = ('wrist', 'hand', 'finger', 'thumb', 'index', 'middle', 'ring', 'pinky', 'metacarpal')
            side_of = {}; is_hand = np.zeros(len(body.data.vertices), bool)
            for v in body.data.vertices:
                best = max(v.groups, key=lambda g: g.weight, default=None)
                if best:                                                       # from the group names (the shared `isarm` mask lacks the metacarpals)
                    nm = gi[best.group]
                    if isarm[v.index] or any(k in nm for k in hand_keys): side_of[v.index] = 'L' if nm.endswith('.L') else ('R' if nm.endswith('.R') else '')
                    if any(k in nm for k in hand_keys): is_hand[v.index] = True
            side_arr = np.array([side_of.get(i, '') for i in range(len(body.data.vertices))])
            RAMP = []
            for i in range(1, N + 1):
                RT.apply(pose_spec, i / N); ev = body.evaluated_get(bpy.context.evaluated_depsgraph_get()); me = ev.to_mesh()
                RAMP.append(np.array([body.matrix_world @ v.co for v in me.vertices])); ev.to_mesh_clear()
            contact_rep = {}; Vf = RAMP[-1]; z_cloth = spec['pattern'].get('z_hem', 0.0) - 0.02       # only where cloth can be: above the hem
            for side in (pose_spec.get('hand_contacts') or {}):
                hs = is_hand & (side_arr == side) & (Vf[:, 2] > z_cloth); other = (side_arr != side) & (Vf[:, 2] > z_cloth)
                if not hs.any(): contact_rep[side] = dict(clearance_mm=None, note='hand below the hem'); continue
                dd, ii = cKDTree(Vf[other]).query(Vf[hs]); m_ = float(dd.min())
                contact_rep[side] = dict(clearance_mm=round(1000 * m_, 1), needed_mm=round(1000 * need_gap, 1), ok=bool(m_ >= need_gap))
            print('hand contacts (cloth region):', contact_rep)
            got = dict(canonical_pose=CPOSE['pose_id'], ramp_steps=N, hand_contact_clearance=contact_rep)
        else:
          for side in 'LR':
            got['upper' + side] = set_angle(f'upperarm01.{side}', P['upperarm_deg']); got['fore' + side] = set_angle(f'lowerarm01.{side}', P['forearm_deg'])
        print('pose', tag, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in got.items()})
        ev = body.evaluated_get(bpy.context.evaluated_depsgraph_get()); me = ev.to_mesh()
        V = np.array([body.matrix_world @ v.co for v in me.vertices]); F = np.array([list(p.vertices) for p in me.polygons if len(p.vertices) == 4])
        ev.to_mesh_clear()
        out[tag] = dict(V=V, F=F, joints={f'{bn}.{s}': np.array(rig.matrix_world @ rig.pose.bones[f'{bn}.{s}'].head) for s in 'LR' for bn in ('upperarm01', 'lowerarm01', 'wrist')}, angles=got)
        if tag == 'B':   # display mannequin = the FINAL (relaxed) pose, real mesh with UVs
            ev = body.evaluated_get(bpy.context.evaluated_depsgraph_get()); dm = bpy.data.meshes.new_from_object(ev); dm.name = 'Mannequin_Mesh'
            dm.transform(body.matrix_world)
            o = bpy.data.objects.new('Mannequin', dm); bpy.context.scene.collection.objects.link(o)
            for p in dm.polygons: p.use_smooth = True
            for ob in list(bpy.data.objects):
                if ob is not o: bpy.data.objects.remove(ob, do_unlink=True)
            bpy.ops.wm.save_as_mainfile(filepath=os.path.join(W, 'mannequin_posed.blend'))
            print('display mannequin: verts', len(dm.vertices), 'uv layers', [u.name for u in dm.uv_layers])
    A, B = out['A'], out['B']; assert A['F'].shape == B['F'].shape and A['V'].shape == B['V'].shape
    np.savez(os.path.join(W, 'collider_posed.npz'), V=A['V'], VB=B['V'], F=A['F'], isarm=isarm, VR=np.array(RAMP) if RAMP else np.zeros((0,) + A['V'].shape),
             **{k.replace('.', '_'): v for k, v in A['joints'].items()}, **{'B_' + k.replace('.', '_'): v for k, v in B['joints'].items()})
    # torso table (from phase-1 pose, torso only) for wrapping the panels
    V = A['V']; sel = (~isarm) & (np.abs(V[:, 0]) < 0.22); T = V[sel]; zs = np.arange(0.76, 1.52, 0.02); rows = []
    for z in zs:
        s = T[np.abs(T[:, 2] - z) < 0.012]
        if len(s) < 6: rows.append([z, np.nan, np.nan, np.nan, np.nan]); continue
        x0, x1 = np.percentile(s[:, 0], [1, 99]); y0, y1 = np.percentile(s[:, 1], [1, 99]); rows.append([z, (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2])
    R = np.array(rows)
    for k in range(1, 5):
        c = R[:, k]; ok = ~np.isnan(c); R[:, k] = np.interp(zs, zs[ok], c[ok])
    K = np.ones(3) / 3
    for k in range(1, 5): R[:, k] = np.convolve(np.pad(R[:, k], 1, mode='edge'), K, mode='valid')
    np.save(os.path.join(W, 'torso_table.npy'), R)
    # clearance between the arms and the torso in both poses (the relaxed pose must not push the arm into the body)
    from scipy.spatial import cKDTree
    for tag, d in out.items():
        kd = cKDTree(d['V'][~isarm & (d['V'][:, 2] > 1.0) & (d['V'][:, 2] < 1.45)]); dd, _ = kd.query(d['V'][isarm & (d['V'][:, 2] > 1.0) & (d['V'][:, 2] < 1.45)])
        print('pose %s: min arm-to-torso distance %.1f mm' % (tag, 1000 * dd.min()))
    print('collider verts', len(V), 'quads', len(A['F']), 'max vertex shift between phases %.1f mm' % (1000 * np.abs(A['V'] - B['V']).max()))


# ------------------------------------------------------------------------------------------------------------------ stage: setup
def stage_setup(spec, W):
    import bpy
    from scipy.spatial import cKDTree
    import scipy.sparse as sp
    pp = spec['pattern']; cl_ = spec['cloth']; Z_HEM = pp['z_hem']; CLEAR = pp['clearance']
    D = np.load(os.path.join(W, 'collider_posed.npz')); BV = D['V']; BVB = D['VB']; BF = D['F']; TT = np.load(os.path.join(W, 'torso_table.npy'))
    JU = {'p': D['upperarm01_L'], 'n': D['upperarm01_R']}; JE = {'p': D['lowerarm01_L'], 'n': D['lowerarm01_R']}
    bn = body_normals(BV, BF); kd = cKDTree(BV)

    pat = PT.build(**{k: pp[k] for k in ('P', 'L', 'xn', 'xs', 'drop', 'vA', 'depthF', 'depthB', 'ease', 'band_edge', 'collar_ratio', 'collar_width', 'cuff_width', 'cuff_ratio') + tuple(k for k in ('wc', 'hw', 'sl', 'cuffs', 'cut', 'raglan') if k in pp)})
    RAGLAN = pat['params'].get('cut') == 'raglan'
    for k, p in pat.items():
        if k != 'params': p.finish(pp['panel_triangle_area'])
    prm = pat['params']; P = prm['P']; A = P / 2

    def torso(z):
        z = np.clip(z, TT[0, 0], TT[-1, 0]); return [float(np.interp(z, TT[:, 0], TT[:, k])) for k in (1, 2, 3, 4)]

    def perim(a, b): return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
    _cache = {}

    def ring(z):
        key = round(z / 0.004)
        if key in _cache: return _cache[key]
        cx, cy, a, b = torso(key * 0.004); lo, hi = 0.0, 0.5
        for _ in range(40):
            d = (lo + hi) / 2
            if perim(a + d, b + d) < 2 * P: lo = d
            else: hi = d
        d = (lo + hi) / 2; aE, bE = a + d, b + d
        ph = np.linspace(-math.pi / 2, math.pi / 2, 400); x = aE * np.sin(ph); y = bE * np.cos(ph)
        s = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]; s -= s[len(s) // 2]
        _cache[key] = (cx, cy, aE, bE, ph, s); return _cache[key]

    def body_pos(u, v, front):
        z = Z_HEM + v; cx, cy, aE, bE, ph, s = ring(z); phi = float(np.interp(u, s, ph))
        x = cx + aE * math.sin(phi); y = cy - bE * math.cos(phi) if front else cy + bE * math.cos(phi)
        return np.array([x, y + (-0.0006 if front else 0.0006), z])

    def surf(p, k=8):
        p = np.atleast_2d(p); d, i = kd.query(p, k=k); w = 1.0 / (d + 2e-3); w /= w.sum(1, keepdims=True)
        n = np.einsum('nk,nkj->nj', w, bn[i]); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        q = np.einsum('nk,nkj->nj', w, BV[i]); sd = np.einsum('nj,nj->n', p - q, n); return sd, n

    def smooth(x): x = min(max(x, 0), 1); return x * x * (3 - 2 * x)

    def drape_blend(p, v):
        w = smooth((v - (prm['vA'] - 0.02)) / 0.12)
        if w <= 0: return p
        sd, n = surf(p); return p + w * n[0] * (CLEAR - sd[0])

    sl, wc, hw = prm['sl'], prm['wc'], prm['hw']

    def arm_frame(side):
        J0 = JU[side]; J1 = JE[side]; d = (J1 - J0) / np.linalg.norm(J1 - J0); Au = J0 + d * pp.get('arm_ring_offset', 0.07)
        up = np.array([0, 0, 1.0]); up = up - up.dot(d) * d; up /= np.linalg.norm(up)
        fr = np.array([0, -1.0, 0]); fr = fr - fr.dot(d) * d - fr.dot(up) * up; fr /= np.linalg.norm(fr)
        return Au, d, up, fr

    def sleeve_pos(u, v, side):
        Au, d, up, fr = arm_frame(side)
        if v <= sl: w = 2 * (hw + (wc - hw) * (v / sl)); R = w / (2 * math.pi); ax = Au + d * (sl - v)
        else: R = 2 * wc / (2 * math.pi); ax = Au - d * (v - sl)
        th = SLEEVE_WRAP * ((u / R) if side == 'p' else (-u / R))     # wrap slightly short of 360 deg so the two under-seam edges start ~5 mm apart (no coincident vertices)
        return ax + R * (math.cos(th) * up + math.sin(th) * fr)

    names = tuple(n for n in ('front', 'back', 'sleeve_p', 'sleeve_n', 'collar', 'cuff_p', 'cuff_n') if n in pat); ribs = [n for n in ('collar', 'cuff_p', 'cuff_n') if n in pat]
    verts = []; faces = []; mats = []; panel_of = []; flat = []; off = {}
    for name in names:
        pan = pat[name]; off[name] = len(verts)
        for (u, v) in pan.V:
            if name in ('front', 'back'): p = drape_blend(body_pos(u, v, name == 'front'), v)
            elif name.startswith('sleeve'): p = sleeve_pos(u, v, name[-1])
            else: p = np.zeros(3)
            verts.append(p); flat.append((u, v)); panel_of.append(name)
        faces += [tuple(int(a) + off[name] for a in f) for f in pan.faces]; mats += [pan.mat] * len(pan.faces)
    verts = np.array(verts); flat = np.array(flat)
    verts_flat = verts.copy()                                                  # positions BEFORE the drape push (body panels): wrapped ring only
    for k, pn in enumerate(panel_of):
        if pn in ('front', 'back'): verts_flat[k] = body_pos(flat[k][0], flat[k][1], pn == 'front')

    def idx(panel, piece, rev=False):
        ids = [i + off[panel] for i in pat[panel].idx[piece]]; return ids[::-1] if rev else ids

    # neckline loop: 29 front + 29 back vertices (consecutive pairs at the shoulders are sewn to each other)
    loop = []                                                                   # neckline loop in ring order (components/sleeve_builder.neckline_loop)
    for pn_, pc_, rev_, skip_ in SB.neckline_loop('raglan' if RAGLAN else 'set_in'): loop += idx(pn_, pc_, rev_)[1 if skip_ else 0:]
    RAG_PAIRS = SB.RAGLAN_PAIRS
    NC = prm['n_loop']; assert len(loop) == NC, (len(loop), NC)
    if RAGLAN:                                                               # sleeve tops first: the collar ring below is seated on the whole loop, sleeve tops included
        verts = place_raglan_saddles(verts, pat, off, idx, RAG_PAIRS, sl, JU, BV, BF, surf, CLEAR, sleeve_pos)
    NL = pp.get('neckline_landmarks')
    if NL:   # landmark-driven neckline: every loop vertex keeps its angle round the neck but is placed ON the body (ray from the neck axis) at a designed
             # height z(theta): back / side-neck point / front.  The clearance push alone shoves the side points up the neck slope.
        from mathutils.bvhtree import BVHTree
        from mathutils import Vector
        _t = []
        for q in BF: _t += [(int(q[0]), int(q[1]), int(q[2])), (int(q[0]), int(q[2]), int(q[3]))]
        _bv = BVHTree.FromPolygons([tuple(v) for v in BV], _t); ax0 = np.array(NL['axis_xy'], float); nl_rep = []
        for vid in loop:
            d_ = verts[vid][:2] - ax0; th = math.atan2(d_[0], -d_[1])                     # 0 = centre front, +-pi = centre back
            c_ = math.cos(th); zt = NL['side_z'] + (NL['front_z'] - NL['side_z']) * max(c_, 0) ** 1.5 + (NL['back_z'] - NL['side_z']) * max(-c_, 0) ** 1.5
            o_ = Vector((ax0[0], ax0[1], zt)); dr = Vector((math.sin(th), -math.cos(th), 0.0)); hit = _bv.ray_cast(o_, dr, 0.5)
            if hit[0] is None: continue
            verts[vid] = np.array(hit[0] + dr * NL.get('offset_m', 0.004)); nl_rep.append(round(zt, 3))
        print('neckline landmarks: %d loop vertices placed, z %.3f..%.3f' % (len(nl_rep), min(nl_rep), max(nl_rep)))
    # ---- collar: PRE-CLOSED ring (n_loop vertices round) standing above the neckline; ring vertex j is sewn to loop[j]
    col = pat['collar']; assert col.n == len(loop)
    clen = col.length; a0, b0 = 0.074, 0.086; cxn, cyn, zc = 0.0, -0.034, 1.485
    lo, hi = 0.3, 3.0
    for _ in range(50):
        m_ = (lo + hi) / 2
        if perim(a0 * m_, b0 * m_) < clen: lo = m_
        else: hi = m_
    aN, bN = a0 * (lo + hi) / 2, b0 * (lo + hi) / 2
    if RAGLAN:   # ring column j starts at the angle of loop vertex j (the raglan loop is not evenly spaced in angle: sleeve tops are short)
        _lv = verts[loop]; _al = np.unwrap(np.arctan2(-(_lv[:, 1] - cyn) / bN, _lv[:, 0] / aN))
    for k in range(off['collar'], off['collar'] + len(col.V)):
        u, v = flat[k]; al = 2 * math.pi * u / clen
        if RAGLAN:     # seat the ring ON the real neckline loop (pulled in by collar_ratio, 3 mm up): its rest shape is the neckline, not a flat ellipse
            j = int(round(u / clen * col.n)) % col.n; _c = _lv[:, :2].mean(0); bp = _lv[j].copy(); bp[2] += 0.003
            if not NL: bp[:2] = _c + (bp[:2] - _c) * pp['collar_ratio']                # (a landmark neckline already sits on the neck: no pull-in)
            out = np.array([*(bp[:2] - _c), 0.0]); out /= np.linalg.norm(out)
            verts[k] = bp + np.array([0, 0, pp.get('collar_up', 0.95) * v]) + out * pp.get('collar_out', 0.35) * v; continue
        x = cxn + aN * math.cos(al); y = cyn - bN * math.sin(al); out = np.array([math.cos(al), -math.sin(al) * (bN / aN), 0]); out /= np.linalg.norm(out)
        verts[k] = np.array([x, y, zc]) + np.array([0, 0, pp.get('collar_up', 0.95) * v]) + out * pp.get('collar_out', 0.35) * v
    # ---- cuffs: PRE-CLOSED rings (n_cuff = sleeve hem vertices - 1) round the sleeve openings; cuff vertex j is sewn to hem[j]
    for name, side in ((n_, sd_) for n_, sd_ in (('cuff_p', 'p'), ('cuff_n', 'n')) if n_ in pat):
        cp = pat[name]; Rc = cp.length / (2 * math.pi); Au, d, up, fr = arm_frame(side)
        for k in range(off[name], off[name] + len(cp.V)):
            u, v = flat[k]; th = -math.pi + 2 * math.pi * u / cp.length if side == 'p' else math.pi - 2 * math.pi * u / cp.length
            ax = Au + d * (sl + v); verts[k] = ax + Rc * (math.cos(th) * up + math.sin(th) * fr)
    # ---- normal-push clearance (never snap onto the body)
    T3 = fan_triangles(faces); r = []; c = []
    for f in T3:
        for a_, b_ in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])): r += [a_, b_]; c += [b_, a_]
    ADJ = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(len(verts), len(verts))).tocsr(); ADJ.data[:] = 1; DEG = np.asarray(ADJ.sum(1)).ravel()

    def clear(verts, iters=10):
        moved = 0
        for _ in range(iters):
            sd, n = surf(verts); need = np.maximum(CLEAR + 0.002 - sd, 0.0); need[sd >= CLEAR] = 0
            if not (need > 0).any(): break
            disp = n * need[:, None]
            for _s in range(4):
                disp = np.where((need > 0)[:, None], disp, 0) + 0 * disp; avg = (ADJ @ disp) / np.maximum(DEG, 1)[:, None]; disp = 0.6 * disp + 0.4 * avg
            verts += disp; moved += int((need > 0).sum())
        return moved
    # smooth the drape push over each panel BEFORE the clearance fix.  Blender cloth takes the INITIAL mesh as its rest shape, so a few vertices pushed
    # far from their neighbours near the throat (pilot: 12 cm edges) become permanent local stretch/pucker at the neckline.
    body_mask = np.array([pn in ('front', 'back') for pn in panel_of]); disp0 = verts - verts_flat
    # (not needed when the rest shape is the flat pattern: the initial placement is then only a starting position, so the panels sit right on the body)
    for _ in range(0 if cl_.get('rest_shape') == 'flat_pattern' else 25): disp0 = np.where(body_mask[:, None], 0.5 * disp0 + 0.5 * (ADJ @ disp0) / np.maximum(DEG, 1)[:, None], disp0)
    verts = np.where(body_mask[:, None], verts_flat + disp0, verts)
    v_before = verts.copy(); mv = clear(verts)
    sleeve_mask = np.array([pn.startswith('sleeve') for pn in panel_of])          # same idea for the sleeves: spread the clearance push so no edge is stretched to 2x its flat length
    push = verts - v_before
    for _ in range(12): push = np.where(sleeve_mask[:, None], 0.5 * push + 0.5 * (ADJ @ push) / np.maximum(DEG, 1)[:, None], push)
    verts = np.where(sleeve_mask[:, None], v_before + push, verts); mv += clear(verts, iters=6); sd, _ = surf(verts)
    _pushed = np.linalg.norm(verts - v_before, axis=1); print('clearance push: max %.1f mm, verts pushed > 20 mm: %d (panels %s)' % (1000 * _pushed.max(), int((_pushed > 0.02).sum()), sorted(set(panel_of[k] for k in np.where(_pushed > 0.02)[0]))))
    e = np.concatenate([np.linalg.norm(verts[T3[:, i]] - verts[T3[:, (i + 1) % 3]], axis=1) for i in range(3)]); ar = tri_areas(verts, T3)
    coincident = len(verts) - len(np.unique(np.round(verts, 5), axis=0))
    if coincident:
        _, inv, cnt = np.unique(np.round(verts, 5), axis=0, return_inverse=True, return_counts=True)
        print('coincident vertex panels:', sorted(set(panel_of[k] for k in np.where(cnt[inv] > 1)[0])), [(panel_of[k], tuple(np.round(flat[k], 4))) for k in np.where(cnt[inv] > 1)[0]][:10])
    _fl = {}
    for pn in ('front', 'back', 'sleeve_p', 'sleeve_n'):
        _f = [f for f in pat[pn].faces]; _o = off[pn]; _r = []
        for t in _f:
            for i in range(3):
                a_, b_ = t[i], t[(i + 1) % 3]; l3 = np.linalg.norm(verts[a_ + _o] - verts[b_ + _o]); l2 = np.linalg.norm(pat[pn].V[a_] - pat[pn].V[b_]); _r.append(l3 / max(l2, 1e-9))
        _fl[pn] = (float(np.percentile(_r, 99.9)), float(np.max(_r)))
    print('initial 3D edge / flat edge (p99.9, max) per panel:', {k: tuple(round(x, 2) for x in v) for k, v in _fl.items()})
    pre = dict(rest_edge_ratio_max={k: round(v[1], 3) for k, v in _fl.items()}, clearance_moved=mv, min_signed_dist_m=float(sd.min()), verts=len(verts), faces=len(faces), edge_min=float(e.min()), edge_median=float(np.median(e)), edge_max=float(e.max()),
               degenerate_triangles_presim=int((ar < 1e-9).sum()), coincident_verts_presim=int(coincident))
    print('pre-sim mesh:', pre)
    assert pre['degenerate_triangles_presim'] == 0 and coincident == 0, 'pre-sim mesh has degenerate geometry; Blender would silently skip the cloth'
    # ---- sewing edges
    S = []

    def sew(a, pa, b, pb, revb=False):
        A_ = idx(a, pa); B_ = idx(b, pb, revb); assert len(A_) == len(B_), (a, pa, len(A_), b, pb, len(B_)); S.extend(zip(A_, B_))
    for a_, pa_, b_, pb_, rv_ in SB.seam_map('raglan' if RAGLAN else 'set_in'): sew(a_, pa_, b_, pb_, rv_)
    for sl_, cf in ((a_, b_) for a_, b_ in (('sleeve_p', 'cuff_p'), ('sleeve_n', 'cuff_n')) if b_ in pat):                       # cuff ring <-> sleeve opening, 28 = 28 distinct hem vertices
        hem = idx(sl_, 'hem')[:pre_ncuff(prm)]; ring_ = idx(cf, 'long_a'); assert len(hem) == len(ring_) == prm['n_cuff'], (len(hem), len(ring_)); S.extend(zip(hem, ring_))
    S.extend(zip(idx('collar', 'long_a'), loop))
    print('sewing edges', len(S), '| mean initial seam gap %.3f m' % np.mean([np.linalg.norm(verts[a] - verts[b]) for a, b in S]))
    # ---- blender objects
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try: bpy.ops.preferences.addon_enable(module="cycles")
    except Exception: pass
    cm = bpy.data.meshes.new('Body_Collider'); cm.from_pydata([tuple(v) for v in BV], [], [tuple(int(x) for x in q) for q in BF]); cm.update()
    colo = bpy.data.objects.new('Body_Collider', cm); bpy.context.scene.collection.objects.link(colo)
    colo.shape_key_add(name='Basis', from_mix=False); sk = colo.shape_key_add(name='Relaxed', from_mix=False)
    for i, v in enumerate(BVB): sk.data[i].co = v
    VR = D['VR'] if 'VR' in D.files else np.zeros((0,)); ramp_keys = []
    for kk in range(len(VR)):                                                  # canonical pose ramp: one relative key per intermediate pose
        kb = colo.shape_key_add(name=f'Pose_{kk + 1}', from_mix=False)
        for i, v in enumerate(VR[kk]): kb.data[i].co = v
        ramp_keys.append(kb)
    colo.modifiers.new('Collision', 'COLLISION'); c = colo.collision
    c.thickness_outer = cl_['collider_thickness_outer']; c.thickness_inner = cl_['collider_thickness_inner']; c.cloth_friction = cl_['collider_friction']; c.damping = cl_['collider_damping']
    me = bpy.data.meshes.new('Tee'); me.from_pydata([tuple(v) for v in verts], [(int(a), int(b)) for a, b in S], faces); me.update()
    tee = bpy.data.objects.new('Tee', me); bpy.context.scene.collection.objects.link(tee)
    for nm in spec.get('slots', ('Main_Body', 'Collar_Rib', 'Left_Cuff_Rib', 'Right_Cuff_Rib')):
        m = bpy.data.materials.new(nm); m.use_nodes = True; me.materials.append(m)
    for p, mi in zip(me.polygons, mats): p.material_index = mi
    if cl_.get('rest_shape') == 'flat_pattern':
        # Spring rest lengths from the FLAT sewing pattern instead of the initial 3D placement (Blender's default).  Without this, any distortion of the
        # initial wrap (drape push at the throat, raglan saddle) becomes permanent stretch.  Rib rings keep their 3D placement as rest (a flat ring would
        # need a cut).  Panels are laid out apart in the shape key; sewing springs ignore rest length.
        tee.shape_key_add(name='Basis', from_mix=False); skf = tee.shape_key_add(name='FlatPattern', from_mix=False); skf.value = 0.0
        _lay = {}; _x = 0.0
        for pn in names:
            if pn in ribs: continue
            _lay[pn] = _x - pat[pn].V[:, 0].min(); _x += float(np.ptp(pat[pn].V[:, 0])) + 0.1
        for k, pn in enumerate(panel_of):
            skf.data[k].co = tuple(verts[k]) if pn in ribs else (flat[k][0] + _lay[pn], flat[k][1], 0.0)
    vg = tee.vertex_groups.new(name='rib'); vg.add([k for k, pn in enumerate(panel_of) if pn in ribs], 1.0, 'REPLACE')
    # ---- UVs: fabric atlas in metres/0.30; front art and back art layers (panel -> [0,1]^2, image top = neckline, back mirrored so it reads from behind)
    LAY = {'front': (A, 0), 'back': (P + 0.05 + A, 0), 'sleeve_p': (1.34, 0.0), 'sleeve_n': (1.34, 0.24), 'collar': (1.14, 0.55), 'cuff_p': (1.14, 0.64), 'cuff_n': (1.14, 0.71)}
    u0 = me.uv_layers.new(name='UVMap_Fabric'); uF = me.uv_layers.new(name='UVMap_Art_Front'); uB = me.uv_layers.new(name='UVMap_Art_Back')
    for poly in me.polygons:
        lis = list(poly.loop_indices); ks = [me.loops[li].vertex_index for li in lis]; pn = panel_of[ks[0]]
        us = np.array([flat[k][0] for k in ks])
        if pat[pn].ring is not None:                                          # the ring wraps: unwrap loops of the faces that cross the wrap
            Lr = pat[pn].ring['length']
            if us.max() - us.min() > Lr / 2: us = np.where(us < Lr / 2, us + Lr, us)
        for li, k, ux in zip(lis, ks, us):
            ox, oy = LAY[pn]; fy = flat[k][1]; u0.data[li].uv = ((ux + ox) / 0.30, (fy + oy) / 0.30)
            if pn == 'front': uF.data[li].uv = ((ux + A) / P, fy / prm['L']); uB.data[li].uv = (-5.0, -5.0)
            elif pn == 'back': uF.data[li].uv = (-5.0, -5.0); uB.data[li].uv = (1.0 - (ux + A) / P, fy / prm['L'])
            else: uF.data[li].uv = (-5.0, -5.0); uB.data[li].uv = (-5.0, -5.0)
    np.savez(os.path.join(W, 'tee_meta.npz'), off=np.array([(n, o) for n, o in off.items()], dtype=object), flat=flat, panel_of=np.array(panel_of), S=np.array(S), loop=np.array(loop), n_cuff=prm['n_cuff'], ribs=np.array(ribs), pre=json.dumps(pre))
    # ---- cloth
    cl = tee.modifiers.new('Cloth', 'CLOTH'); s = cl.settings; cs = cl.collision_settings
    s.quality = cl_['quality']; s.mass = cl_['mass']; s.tension_stiffness = cl_['tension']; s.compression_stiffness = cl_['compression']; s.shear_stiffness = cl_['shear']; s.bending_stiffness = cl_['bending']
    s.tension_damping = cl_['tension_damping']; s.compression_damping = cl_['compression_damping']; s.shear_damping = cl_['shear_damping']; s.bending_damping = cl_['bending_damping']; s.air_damping = cl_['air_damping']
    s.use_sewing_springs = True; s.sewing_force_max = cl_['sewing_force_max']
    if cl_.get('rest_shape') == 'flat_pattern': s.rest_shape_key = tee.data.shape_keys.key_blocks['FlatPattern']
    pin_keys = []
    if cl_.get('pin_collar'):                     # the collar sits on the neck and the shirt hangs from it: pin the ring where it was seated on the neckline
        pin_idx = [k for k, pn in enumerate(panel_of) if pn == 'collar']
        pg = tee.vertex_groups.new(name='pin'); pg.add(pin_idx, 1.0, 'REPLACE')
        s.vertex_group_mass = 'pin'; s.pin_stiffness = cl_.get('pin_stiffness', 1.0)
        if len(VR):                                                            # pinned vertices hold their INPUT position: under a canonical pose the neck
            from scipy.spatial import cKDTree                                  # moves, so the pins must move with it or the ring tears off the neckline
            if not (tee.data.shape_keys and 'Basis' in tee.data.shape_keys.key_blocks): tee.shape_key_add(name='Basis', from_mix=False)
            Pp = np.asarray(verts)[pin_idx]; _, nb = cKDTree(BV).query(Pp, k=40); pin_keys = []
            for kk in range(len(VR)):                                          # per pin vertex: best-fit rigid transform (Kabsch) of its 40 nearest body
                kb = tee.shape_key_add(name=f'PinPose_{kk + 1}', from_mix=False); kb.value = 0.0   # vertices, rest pose -> ramp pose kk
                for j, k in enumerate(pin_idx):
                    A_ = BV[nb[j]]; B_ = VR[kk][nb[j]]; ca, cb = A_.mean(0), B_.mean(0)
                    U, _s, Vt = np.linalg.svd((A_ - ca).T @ (B_ - cb)); dd = np.sign(np.linalg.det(Vt.T @ U.T))
                    R_ = Vt.T @ np.diag([1.0, 1.0, dd]) @ U.T; kb.data[k].co = tuple(R_ @ (Pp[j] - ca) + cb)
                pin_keys.append(kb)
    s.vertex_group_bending = 'rib'; s.bending_stiffness_max = cl_['rib_bending_max']
    cs.use_collision = True; cs.distance_min = cl_['collision_distance']; cs.collision_quality = 4; cs.use_self_collision = False
    sm = spec['sim']; total = sm['phase1_frames'] + sm['phase2_frames'] + sm['extra_settle_frames_max'] + 5
    bpy.context.scene.frame_start = 1; bpy.context.scene.frame_end = total
    # the phase-2 relaxation: shape key `Relaxed` ramps 0 -> 1 after phase 1
    f1 = sm['phase1_frames']
    if ramp_keys:                                                              # pairwise windows: at every frame consecutive keys sum to 1, so the collider
        st = max(2, int(sm.get('pose_ramp_frames', 24) // len(ramp_keys)))   # is always an interpolation between two REAL poses (no linear smear of a
        sk.value = 0.0; sk.keyframe_insert('value', frame=1)                   # 100-deg arm swing); the 'Relaxed' key is unused
        for kk, kb in enumerate(ramp_keys):
            up0, peak = f1 + kk * st, f1 + (kk + 1) * st
            kb.value = 0.0; kb.keyframe_insert('value', frame=1); kb.keyframe_insert('value', frame=up0)
            kb.value = 1.0; kb.keyframe_insert('value', frame=peak)
            if kk + 1 < len(ramp_keys): kb.value = 0.0; kb.keyframe_insert('value', frame=peak + st)
        for kk, kb in enumerate(pin_keys):                                     # the garment's pin keys on exactly the collider's timing
            up0, peak = f1 + kk * st, f1 + (kk + 1) * st
            kb.value = 0.0; kb.keyframe_insert('value', frame=1); kb.keyframe_insert('value', frame=up0)
            kb.value = 1.0; kb.keyframe_insert('value', frame=peak)
            if kk + 1 < len(pin_keys): kb.value = 0.0; kb.keyframe_insert('value', frame=peak + st)
    else:
        sk.value = 0.0; sk.keyframe_insert('value', frame=1); sk.keyframe_insert('value', frame=f1)
        sk.value = 1.0; sk.keyframe_insert('value', frame=f1 + sm['relax_ramp_frames'])
    try:                                                                  # linear ramp (Blender 5 slotted actions)
        for _ob in (colo, tee):                                            # the garment's pin keys too: same linear timing as the collider
            if not (_ob.data.shape_keys and _ob.data.shape_keys.animation_data and _ob.data.shape_keys.animation_data.action): continue
            act = _ob.data.shape_keys.animation_data.action
            fcs = [fc for layer in act.layers for strip in layer.strips for cb in strip.channelbags for fc in cb.fcurves] if hasattr(act, 'layers') else list(act.fcurves)
            for fc in fcs:
                for kp in fc.keyframe_points: kp.interpolation = 'LINEAR'
    except Exception as ex: print('note: keyframes left at default interpolation (%s)' % ex)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(W, 'tee_setup.blend')); print('saved tee_setup.blend')


SLEEVE_WRAP = 0.985


def place_raglan_saddles(verts, pat, off, idx, RAG_PAIRS, sl, JU, BV, BF, surf, CLEAR, sleeve_pos):
    """Initial 3D position of the raglan sleeve top (the 'saddle', flat v > sl), which reaches from the arm up to the neckline.

    The arm part (v <= sl) keeps its cylinder.  Each raglan-edge vertex sits 4 mm outside its sewing partner on the (already placed) body panel.
    Every other saddle vertex (u, v) is placed on a cross-section curve at its flat height v: a quadratic through the +u raglan edge point A(v),
    a shoulder-top centreline point C(v) and the -u raglan edge point B(v), at the parameter t = fraction of the flat width.  C(v) runs from the
    top of the arm ring (v = sl) to just above the middle of the sleeve's neck edge, bowed upward over the shoulder.  Then a few Jacobi smoothing
    passes (boundary held), then every saddle vertex is projected radially from the shoulder joint to at least CLEAR outside the body.
    (A harmonic fill was tried first: its interior collapses into the shoulder, and the radial rays then exit through the far side of the body.)
    Boundary lengths match the flat pattern (the raglan edges are cut to the body edge length), so the rest shape carries no large stretch.
    """
    from mathutils.bvhtree import BVHTree
    from mathutils import Vector
    tris = []
    for q in BF: tris += [(int(q[0]), int(q[1]), int(q[2])), (int(q[0]), int(q[2]), int(q[3]))]
    bvh = BVHTree.FromPolygons([tuple(v) for v in BV], tris); verts = verts.copy(); rep = {}
    up_w = np.array([0, 0, 1.0])
    for side, prs in RAG_PAIRS.items():
        sp_ = 'sleeve_' + side; pan = pat[sp_]; o = off[sp_]; nv = len(pan.V); UV = pan.V; fixed = {}
        for k in range(nv):
            if UV[k][1] <= sl + 1e-9: fixed[k] = verts[o + k].copy()
        edge_pos = {}
        for spc, bp, bpc, rev in prs:
            loc = pan.idx[spc]; part = idx(bp, bpc, rev); assert len(loc) == len(part), (spc, len(loc), len(part))
            _sd, n_ = surf(verts[part])
            for a_, b_, nn in zip(loc, part, n_):
                if UV[a_][1] > sl + 1e-9: fixed[a_] = verts[b_] + nn * 0.004      # the underarm end keeps its cylinder position (sewing closes the gap)
            ev = np.array([UV[a_][1] for a_ in loc]); order = np.argsort(ev); fixed.setdefault(loc[order[0]], verts[o + loc[order[0]]].copy())
            edge_pos[spc] = (ev[order], np.array([fixed[loc[j]] for j in order]), np.array([UV[loc[j]][0] for j in order]))
        nk = pan.idx['neck']; vt = float(np.mean([UV[k][1] for k in nk]))
        p0, p1 = fixed[nk[0]], fixed[nk[-1]]
        for t_, k in enumerate(nk): fixed[k] = p0 + (p1 - p0) * (t_ / (len(nk) - 1))                   # neck edge: straight between its two end points

        def at(spc, v):
            ev, P, U = edge_pos[spc]; return np.array([np.interp(v, ev, P[:, d]) for d in range(3)]), float(np.interp(v, ev, U))
        Ptop = np.asarray(sleeve_pos(0.0, sl, side), float)                       # top of the arm ring
        Pneck = (p0 + p1) / 2 + up_w * 0.008
        Pmid = (Ptop + Pneck) / 2 + up_w * 0.05 + (Ptop + Pneck - 2 * np.asarray(JU[side])) * 0.0     # bow upward over the shoulder

        def centre(v):
            s_ = np.clip((v - sl) / (vt - sl), 0, 1); return (1 - s_) ** 2 * Ptop + 2 * (1 - s_) * s_ * Pmid + s_ ** 2 * Pneck
        X = verts[o:o + nv].copy()
        for k in range(nv):
            if k in fixed: X[k] = fixed[k]; continue
            u, v = UV[k]; A_, ua = at('rag_a', v); B_, ub = at('rag_b', v); C_ = centre(v)
            t = np.clip((ua - u) / max(ua - ub, 1e-6), 0, 1)
            X[k] = A_ * (2 * (t - 0.5) * (t - 1)) + C_ * (-4 * t * (t - 1)) + B_ * (2 * t * (t - 0.5))   # Lagrange quadratic through A (t=0), C (t=.5), B (t=1)
        for k in nk: X[k] = fixed.setdefault(k, X[k])
        nb = [set() for _ in range(nv)]
        for f in pan.faces:
            for i in range(3): nb[f[i]].add(f[(i + 1) % 3]); nb[f[(i + 1) % 3]].add(f[i])
        free = [k for k in range(nv) if k not in fixed]
        for _ in range(15):
            X[free] = 0.5 * X[free] + 0.5 * np.array([X[list(nb[k])].mean(0) for k in free])
        saddle = [k for k in range(nv) if UV[k][1] > sl + 1e-9 and k not in fixed]
        lifted = 0
        for _ in range(6):                                                    # nearest-surface push (smoothed normal), as stage_setup.clear() does
            sd, n_ = surf(X[saddle]); need = np.maximum(CLEAR - sd, 0.0)
            if not (need > 0).any(): break
            X[saddle] += n_ * need[:, None]; lifted += int((need > 0).sum())
            for _s in range(3): X[free] = 0.6 * X[free] + 0.4 * np.array([X[list(nb[k])].mean(0) for k in free])
        verts[o:o + nv] = X; rep[sp_] = dict(saddle_verts=len(saddle), pushes=lifted, min_clear_mm=round(1000 * float(surf(X[saddle])[0].min()), 1))
    print('raglan saddles placed:', rep)
    return verts


def pre_ncuff(prm): return prm['n_cuff']


# ------------------------------------------------------------------------------------------------------------------ stage: sim
def stage_sim(spec, W):
    import bpy
    from scipy.spatial import cKDTree
    sm = spec['sim']; bpy.ops.wm.open_mainfile(filepath=os.path.join(W, 'tee_setup.blend'))
    sc = bpy.context.scene; tee = bpy.data.objects['Tee']; col = bpy.data.objects['Body_Collider']
    S = np.load(os.path.join(W, 'tee_meta.npz'), allow_pickle=True)['S']; F = np.load(os.path.join(W, 'collider_posed.npz'))['F']
    t0 = time.time(); prev = [None]; log = []

    def evaluated(ob):
        ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get()); me = ev.to_mesh(); V = np.array([v.co[:] for v in me.vertices]); ev.to_mesh_clear(); return V

    def stats(f, V):
        BV = evaluated(col); kd = cKDTree(BV); bn = body_normals(BV, F)
        d, i = kd.query(V); sd = np.einsum('ij,ij->i', V - BV[i], bn[i]); gap = np.linalg.norm(V[S[:, 0]] - V[S[:, 1]], axis=1)
        mv = 0 if prev[0] is None else float(np.percentile(np.linalg.norm(V - prev[0], axis=1), 99)); prev[0] = V   # p99 per-vertex move: a few jittering seam vertices must not block settling
        rec = dict(frame=f, gap_mean=float(gap.mean()), gap_max=float(gap.max()), inside=int((sd < 0).sum()), min_sd=float(sd.min()), max_move=mv, zmin=float(V[:, 2].min()), zmax=float(V[:, 2].max()))
        log.append(rec); print('f%03d t=%5.1fs  seam gap mean %.4f max %.4f | inside body %3d | min sd %+.4f | max move %.4f | z %.3f..%.3f | relax key %.2f' %
                               (f, time.time() - t0, rec['gap_mean'], rec['gap_max'], rec['inside'], rec['min_sd'], mv, rec['zmin'], rec['zmax'], col.data.shape_keys.key_blocks['Relaxed'].value), flush=True); return rec
    f1 = sm['phase1_frames']; f_end2 = f1 + sm['phase2_frames']; V = None
    for f in range(1, f_end2 + sm['extra_settle_frames_max'] + 1):
        sc.frame_set(f); V = evaluated(tee)
        rec = stats(f, V) if (f in (1, 2, 5, f1, f1 + 1, f_end2) or f % 5 == 0 or f > f_end2) else None
        if f > f_end2 and rec and rec['max_move'] < sm['settle_move_tol_m']: break          # phase 2 is 15 frames; extra settle frames only while the cloth still moves
    np.save(os.path.join(W, 'tee_final.npy'), V)
    json.dump(dict(frames_run=f, gap_pairs_worst=[[int(S[k][0]), int(S[k][1]), float(g)] for k, g in sorted(enumerate(np.linalg.norm(V[S[:, 0]] - V[S[:, 1]], axis=1)), key=lambda t: -t[1])[:12]], phase1_frames=f1, phase2_frames=sm['phase2_frames'], log=log), open(os.path.join(W, 'sim_log.json'), 'w'), indent=1)
    print('saved tee_final.npy after', f, 'frames')


# ------------------------------------------------------------------------------------------------------------------ stage: bake
def surface_defects(me, half_window=8, edges='seam'):
    """Final-mesh defect metrics for the gate.  seam_waviness: every chain of edges between two material slots (body <-> collar / cuff rib; the
    chain may be open where refinement branched it) is walked both ways; each vertex is compared with a quadratic fitted to its +-`half_window`
    chain neighbours, so the garment's own curvature is ignored and only scallops remain; the max deviation in mm is the lettuce-edge measure.
    spike_ratio: per vertex, |v - mean(neighbours)| / mean neighbour edge length; a stray spike scores far above the smooth-surface value (< 1)."""
    import bmesh
    from collections import defaultdict
    V = np.array([v.co[:] for v in me.vertices]); E = np.array([e.vertices[:] for e in me.edges]); n = len(V)
    deg = np.bincount(E.ravel(), minlength=n).astype(float)
    acc = np.zeros((n, 3)); np.add.at(acc, E[:, 0], V[E[:, 1]]); np.add.at(acc, E[:, 1], V[E[:, 0]])
    el = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1); lsum = np.bincount(E[:, 0], el, n) + np.bincount(E[:, 1], el, n)
    lap = np.linalg.norm(V - acc / np.maximum(deg, 1)[:, None], axis=1); ratio = lap / np.maximum(lsum / np.maximum(deg, 1), 1e-9)
    bm = bmesh.new(); bm.from_mesh(me); adj = defaultdict(list); vmat = {}
    for e in bm.edges:
        if (e.is_boundary if edges == 'boundary' else len(e.link_faces) == 2 and e.link_faces[0].material_index != e.link_faces[1].material_index):
            a, b = e.verts[0].index, e.verts[1].index; adj[a].append(b); adj[b].append(a); vmat[a] = vmat[b] = max(f.material_index for f in e.link_faces)
    bm.free(); seen = set(); loops = []
    def walk(start, first):
        out = []; prev, cur = start, first
        while cur not in seen and len(adj[cur]) == 2:
            out.append(cur); seen.add(cur); nx = [y for y in adj[cur] if y != prev]; prev, cur = cur, nx[0]
        return out, cur == start
    k = half_window
    for s in list(adj):
        if s in seen or len(adj[s]) != 2: continue
        seen.add(s); a_, closed = walk(s, adj[s][0]); b_, _ = walk(s, adj[s][1]) if not closed else ([], True)
        o = b_[::-1] + [s] + a_
        if len(o) < 2 * k + 1: continue
        P = V[o]; Pp = np.concatenate([P[-k:], P, P[:k]]) if closed else P
        sl = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Pp, axis=0), axis=1))]           # quadratic in ARC LENGTH: refinement midpoints make index spacing uneven
        d = []
        for i in range(k, len(Pp) - k):
            tt = sl[i - k:i + k + 1] - sl[i]; Bq = np.stack([np.ones_like(tt), tt, tt * tt], 1)
            d.append(np.linalg.norm(Pp[i] - (Bq @ np.linalg.lstsq(Bq, Pp[i - k:i + k + 1], rcond=None)[0])[k]))
        d = np.array(d)
        loops.append(dict(n=len(o), closed=bool(closed), material=int(vmat[s]), centroid=P.mean(0).round(3).tolist(), dev_rms_mm=round(1000 * float(np.sqrt((d ** 2).mean())), 2), dev_max_mm=round(1000 * float(d.max()), 2)))
    return dict(seam_loops=sorted(loops, key=lambda r: -r['dev_max_mm']), seam_waviness_max_mm=max((r['dev_max_mm'] for r in loops), default=0.0),
                spike_ratio_max=round(float(ratio.max()), 3), spike_at=V[int(np.argmax(ratio))].round(3).tolist())


def stage_bake(spec, W, out_blend):
    import bpy, bmesh
    from scipy.spatial import cKDTree
    pol = spec['polish']; gate = spec['quality_gate']; report = dict(archetype=spec['archetype'], gate=gate)
    bpy.ops.wm.open_mainfile(filepath=os.path.join(W, 'tee_setup.blend')); sc = bpy.context.scene
    tee = bpy.data.objects['Tee']; colo = bpy.data.objects['Body_Collider']
    M = np.load(os.path.join(W, 'tee_meta.npz'), allow_pickle=True); panel = M['panel_of']; S = M['S']; report['presim'] = json.loads(str(M['pre']))
    V = np.load(os.path.join(W, 'tee_final.npy')).copy(); me = tee.data; tee.modifiers.remove(tee.modifiers['Cloth'])
    if me.shape_keys: tee.shape_key_clear()                                          # the flat-pattern rest key is only for the simulation
    gap_pre = np.linalg.norm(V[S[:, 0]] - V[S[:, 1]], axis=1)
    report['seam_gap_presnap_mm'] = dict(mean=float(1000 * gap_pre.mean()), max=float(1000 * gap_pre.max()), over_6mm=int((gap_pre > 0.006).sum()), pairs=len(S))
    parent = list(range(len(V)))                                                 # sewn vertices form small connected groups (a shoulder vertex is sewn to the other panel AND the collar):
    def find(x):                                                                 # snap each whole group to its centroid so no pair is left apart because of a chain
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a_, b_ in S: parent[find(a_)] = find(b_)
    groups = {}
    for a_, b_ in S: groups.setdefault(find(a_), set()).update((a_, b_))
    for mem in groups.values():
        mem = sorted(mem); cen = V[mem].mean(0)
        for k in mem: V[k] = cen
    report['sewn_groups'] = len(groups); report['sewn_group_max_size'] = max(len(m) for m in groups.values())
    # ---- rib fit: the free rows of the collar and cuff rings come out of the sim scalloped (a free edge with little bending stiffness flutters, and the cuffs roll outward).
    # The sewn row stays exactly where the sim put it; every other row is re-laid from it with a circularly smoothed offset field, and the flare is damped.
    rf = pol['rib_fit']; fit_rep = {}; CONS = spec.get('construction', {}); FOLD = CONS.get('foldover_collar'); COLLAR_ROWS = None; R_COLLAR = None
    RIBS = [str(x) for x in M['ribs']]
    H_SPEC = rf.get('fourier_harmonics')                                          # optional: periodic low-pass of every ring row (lettuce-edge fix); the 1-2-1 passes
                                                                                  # only damp the top frequencies, a Fourier cut removes everything above H by construction.
    def H_of(rname):                                                              # int for every rib, or {"collar": 4, "cuff": 3}
        return H_SPEC.get('collar' if rname == 'collar' else 'cuff') if isinstance(H_SPEC, dict) else H_SPEC
    RROWS = {}

    def lowpass(P, H, ref=None):                                                  # closed-ring Fourier fit in ARC LENGTH of `ref` (default: P itself), so uneven
        if not H: return P                                                        # column spacing (front vs back neckline) cannot bend the result
        Q = P if ref is None else ref; seg = np.linalg.norm(np.roll(Q, -1, 0) - Q, axis=1)
        th = 2 * np.pi * np.r_[0.0, np.cumsum(seg)[:-1]] / seg.sum()
        Bf = np.column_stack([np.ones_like(th)] + [f(h * th) for h in range(1, H + 1) for f in (np.cos, np.sin)])
        return Bf @ np.linalg.lstsq(Bf, P, rcond=None)[0]
    for rname, nring in ((r_, int(M['loop'].shape[0]) if r_ == 'collar' else int(M['n_cuff'])) for r_ in RIBS):
        ids = np.where(panel == rname)[0]; rows = len(ids) // nring; assert rows * nring == len(ids), (rname, len(ids), nring)
        R_ = ids.reshape(rows, nring)                                             # R_[k, j] = vertex of row k, column j (rows are consecutive in the pattern)
        Vsim = np.load(os.path.join(W, 'tee_final.npy')); keep = rf['collar_flare_keep'] if rname == 'collar' else rf['cuff_flare_keep']
        if rf['edge_smooth_passes'] > 0:                                              # smooth the sewn row itself (saw-tooth from the sim) and move every vertex sewn to it with it
            e_ = V[R_[0]].copy()
            for _ in range(rf['edge_smooth_passes']): e_ = 0.25 * np.roll(e_, 1, 0) + 0.5 * e_ + 0.25 * np.roll(e_, -1, 0)
            e_ = lowpass(e_, H_of(rname))
            for j, vid in enumerate(R_[0]):
                for mem in groups.get(find(int(vid)), (int(vid),)): V[mem] = e_[j]
        base = V[R_[0]]; d_all = [Vsim[R_[k]] - Vsim[R_[0]] for k in range(1, rows)]
        ctr = base.mean(0); u_, s_, vt = np.linalg.svd(base - ctr); nrm = vt[2]
        if np.mean([np.einsum('ij,j->i', d, nrm).mean() for d in d_all]) < 0: nrm = -nrm   # ring axis, pointing away from the sewn edge
        wob = []; base_s = base.copy()
        for _ in range(rf['base_smooth_passes']): base_s = 0.25 * np.roll(base_s, 1, 0) + 0.5 * base_s + 0.25 * np.roll(base_s, -1, 0)
        base_s = lowpass(base_s, H_of(rname))
        for k, d in enumerate(d_all, start=1):
            ax = np.outer(d @ nrm, nrm); rad = d - ax; d2 = ax + keep * rad                  # damp the radial flare
            for _ in range(rf['smooth_passes']): d2 = 0.25 * np.roll(d2, 1, 0) + 0.5 * d2 + 0.25 * np.roll(d2, -1, 0)   # circular 1-2-1: no scallops
            d2 = lowpass(d2, H_of(rname), ref=base)
            L0 = np.linalg.norm(d, axis=1).mean(); L1 = np.linalg.norm(d2, axis=1); d2 = d2 * (L0 / np.maximum(L1, 1e-9))[:, None]   # same band width as the sim gave
            wob.append(float(np.linalg.norm(d2 - d, axis=1).max())); w_ = k / (rows - 1)
            V[R_[k]] = (1 - w_) * base + w_ * base_s + d2                           # free rim follows a smoothed copy of the sewn edge, so the neckline's lumps do not show on the rim
        if rname == 'collar' and FOLD:            # fold-over collar: the stand is CONSTRUCTED from the draped neckline (a free-simulated stand flutters into fins)
            lean = math.radians(FOLD.get('stand_lean_deg', 6.0)); hgt = spec['pattern']['collar_width']
            outv = base_s - base_s.mean(0); outv -= np.outer(outv @ nrm, nrm); outv /= np.linalg.norm(outv, axis=1, keepdims=True)
            upw = np.array([0.0, 0.0, 1.0]); outv = base_s - base_s.mean(0); outv[:, 2] = 0; outv /= np.linalg.norm(outv, axis=1, keepdims=True)
            dirv = upw * math.cos(lean) - outv * math.sin(lean)               # near-vertical with a small lean onto the neck (the tilted best-fit axis of a
                                                                              # saddle-shaped neckline leaned the side stand in by ~35 deg)
            for k in range(1, rows):
                w_ = k / (rows - 1); V[R_[k]] = (1 - w_) * base + w_ * base_s + (w_ * hgt) * dirv
        BAND = CONS.get('rib_band') if rname == 'collar' and not FOLD else None
        if BAND:                                  # ringer neckband CONSTRUCTED from the draped neckline: the simulated band crumples (its direction swings by
            lean = math.radians(BAND.get('lean_deg', -8.0)); hgt = BAND['height_m']         # +-170 deg round the ring), and smoothing a folded field leaves a pinch at the
            outv = base_s - base_s.mean(0); outv[:, 2] = 0; outv /= np.linalg.norm(outv, axis=1, keepdims=True)   # side neck.  Constant height, constant lean
            dirv = np.array([0.0, 0.0, 1.0]) * math.cos(lean) - outv * math.sin(lean)       # (+ = onto the neck, - = flared out), one smooth frame all the way round
            for k in range(1, rows):
                w_ = k / (rows - 1); V[R_[k]] = (1 - w_) * base + w_ * base_s + (w_ * hgt) * dirv
            fit_rep['collar_band'] = dict(constructed=True, height_mm=round(1000 * hgt, 1), lean_deg=BAND.get('lean_deg', -8.0))
        if rname == 'collar': R_COLLAR = R_.copy()
        RROWS[rname] = R_
        fit_rep[rname] = dict(rows=rows, max_row_change_mm=round(1000 * max(wob), 1))
    from mathutils.bvhtree import BVHTree
    from mathutils import Vector
    _Db = np.load(os.path.join(W, 'collider_posed.npz')); _t = []
    for q in _Db['F']: _t += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    _bvh = BVHTree.FromPolygons([tuple(v) for v in _Db['VB']], _t); pushed = 0
    rib_ids = np.concatenate([np.where(panel == r)[0] for r in RIBS])
    rounds = rf.get('lowpass_clearance_rounds', 1) if H_SPEC else 1
    for rnd in range(rounds):                                                  # per-vertex pushes re-inject bumps into a low-passed ring: alternate low-pass and
        if rnd:                                                                # clearance, always ENDING on a clearance pass so nothing is left inside the body
            for rname, R_ in RROWS.items():
                if rname == 'collar' and FOLD: continue
                for k in range(R_.shape[0]):
                    P2 = lowpass(V[R_[k]], H_of(rname))
                    for j, vid in enumerate(R_[k]):
                        for mem in (groups.get(find(int(vid)), (int(vid),)) if k == 0 else (int(vid),)): V[mem] = P2[j]
        pushed = 0
        for _ in range(12):                                                    # keep 1.5 mm clearance from the final body (straightening a cuff can sink it into a bent arm)
            for k in rib_ids:
                loc, n_, _i, _d = _bvh.find_nearest(Vector(V[k])); sdv = (Vector(V[k]) - loc).dot(n_)
                if sdv < 0.0015:
                    dv_ = np.array(n_) * (0.0015 - sdv); pushed += 1
                    for mem in groups.get(find(int(k)), (int(k),)): V[mem] = V[mem] + dv_      # move the whole sewn group: a rib-only push would reopen the seam
        print('rib low-pass/clearance round %d: %d pushes' % (rnd, pushed))
    fit_rep['clearance_pushes'] = pushed
    if 'collar' in RIBS: COLLAR_ROWS = V[R_COLLAR].copy()                             # AFTER the clearance pushes: the leaf must follow the stand as it really is
    report['rib_fit'] = fit_rep; print('rib fit:', fit_rep)
    gap = np.linalg.norm(V[S[:, 0]] - V[S[:, 1]], axis=1); seam_over = int((gap > 0.002).sum())
    report['seam_pairs_over_2mm'] = seam_over; print('seam pairs: pre-snap mean %.1f mm max %.1f mm; after midpoint snap %d of %d still over 2 mm' % (1000 * gap_pre.mean(), 1000 * gap_pre.max(), seam_over, len(S)))
    for v, p in zip(me.vertices, V): v.co = p
    # orient every panel outward using the FINAL (relaxed) body
    D = np.load(os.path.join(W, 'collider_posed.npz')); BV = D['VB']; bn = body_normals(BV, D['F']); kd = cKDTree(BV)
    cut = spec['pattern'].get('cut', 'set_in'); sleeve_kind = 'raglan_sleeve' if cut == 'raglan' else 'set_in_sleeve'
    SLOT_OF = {'front': 'body_front', 'back': 'body_back', 'sleeve_p': sleeve_kind + '_L', 'sleeve_n': sleeve_kind + '_R',
               'collar': 'collar_ring_sim' if FOLD else 'collar_rib', 'cuff_p': 'cuff_rib_L', 'cuff_n': 'cuff_rib_R'}
    slot_gi = {nm: tee.vertex_groups.new(name=nm).index for nm in sorted(set(SLOT_OF[str(p_)] for p_ in np.unique(panel)))}
    rib_gi = tee.vertex_groups['rib'].index
    bm = bmesh.new(); bm.from_mesh(me); bm.edges.ensure_lookup_table()
    dl = bm.verts.layers.deform.verify(); bm.verts.ensure_lookup_table()
    for v_ in bm.verts: v_[dl][slot_gi[SLOT_OF[str(panel[v_.index])]]] = 1.0          # structural slot of every vertex (read by the registry bootstrapper)
    seam_l = bm.verts.layers.float.new('seam')                                          # 1 on vertices of construction seams between panels (raglan, side): the
    _rib_set = set(str(x) for x in RIBS)                                                # render shader draws a stitched groove from it (interpolated by weld/subsurf)
    for a_, b_ in S:
        if str(panel[a_]) != str(panel[b_]) and str(panel[a_]) not in _rib_set and str(panel[b_]) not in _rib_set:
            bm.verts[int(a_)][seam_l] = 1.0; bm.verts[int(b_)][seam_l] = 1.0
    bmesh.ops.delete(bm, geom=[e for e in bm.edges if not e.link_faces], context='EDGES')
    bm.faces.ensure_lookup_table(); fc = np.array([f.calc_center_median()[:] for f in bm.faces]); fn = np.array([f.normal[:] for f in bm.faces])
    d, i = kd.query(fc, k=8); w = 1 / (d + 2e-3); w /= w.sum(1, keepdims=True); nb = np.einsum('nk,nkj->nj', w, bn[i]); dot = np.einsum('nj,nj->n', fn, nb)
    pf = np.array([panel[f.verts[0].index] for f in bm.faces]); flip = []
    for pn in np.unique(pf):
        m = pf == pn; frac = float((dot[m] > 0).mean()); print('panel %-9s faces facing outward: %.0f%%' % (pn, 100 * frac))
        if frac < 0.5: flip += [bm.faces[k] for k in np.where(m)[0]]
    if flip: bmesh.ops.reverse_faces(bm, faces=flip)
    n0 = len(bm.verts); bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=pol['weld_dist'])
    bmesh.ops.dissolve_degenerate(bm, dist=0.0025, edges=bm.edges[:])                  # collapse slivers (edges < 2.5 mm) left by the weld
    print('weld: %d -> %d verts' % (n0, len(bm.verts)))
    gap_over = pol['rib_fit'].get('over_body_gap', 0.0)
    if gap_over > 0:                                                                   # fold-over collar: it lies ON the shirt, not in it. Keep collar-only vertices `gap_over` above the shirt's own outer surface.
        bm.verts.ensure_lookup_table(); lt = bm.calc_loop_triangles(); body_tris = [tuple(l.vert.index for l in t) for t in lt if t[0].face.material_index == 0]
        sb = BVHTree.FromPolygons([tuple(v.co) for v in bm.verts], body_tris); only_rib = [v for v in bm.verts if v.link_faces and all(f.material_index >= 1 for f in v.link_faces)]; lifted = 0
        for _ in range(8):
            for v in only_rib:
                hit = sb.find_nearest(Vector(v.co))
                if hit[0] is None: continue
                loc, n_, _i, _d = hit; sdv = (Vector(v.co) - loc).dot(n_)
                if sdv < gap_over: v.co = v.co + n_ * (gap_over - sdv); lifted += 1
        report['collar_lifted_verts'] = lifted; print('collar lifted off the shirt surface:', lifted)
    # adaptive refinement: split any edge the drape stretched beyond refine_edge_m (front neckline, elbow cuffs), so the gate never has to be relaxed
    nsplit = 0
    for _pass in range(4):
        long_e = [e for e in bm.edges if e.calc_length() > pol['refine_edge_m']]
        if not long_e: break
        bmesh.ops.subdivide_edges(bm, edges=long_e, cuts=1, use_grid_fill=True); nsplit += len(long_e)
    flat_q = [f for f in bm.faces if len(f.verts) == 4 and any(l.calc_angle() > 2.9 for l in f.loops)]       # quads left with a straight corner (a split neighbour edge): triangulate across it
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4] + flat_q, quad_method='SHORT_EDGE', ngon_method='BEAUTY')
    bmesh.ops.dissolve_degenerate(bm, dist=0.0025, edges=bm.edges[:])                  # refinement can leave slivers: collapse them again
    nsl = 0
    for _pass in range(6):                                                              # sliver = triangle whose smallest altitude is < 0.4 mm: collapse its shortest edge
        bm.faces.ensure_lookup_table(); bad = []
        for f in bm.faces:
            if len(f.verts) != 3: continue
            ls = sorted(e.calc_length() for e in f.edges)
            if ls[2] > 0 and 2 * f.calc_area() / ls[2] < 0.0004: bad.append(min(f.edges, key=lambda e: e.calc_length()))
        bad = list({e.index: e for e in bad if e.is_valid}.values())
        if not bad: break
        nsl += len(bad); bmesh.ops.collapse(bm, edges=bad)
    if H_SPEC:                                                                          # weld / refinement / sliver collapse act per vertex: low-pass every rib rim (closed
        rim_e = [e for e in bm.edges if e.is_boundary and e.link_faces[0].material_index >= 1]   # boundary loop on a rib slot) once more, ring by ring
        rim_adj = {}
        for e in rim_e:
            for x, y in ((e.verts[0], e.verts[1]), (e.verts[1], e.verts[0])): rim_adj.setdefault(x, []).append(y)
        done = set(); rim_fixed = 0
        for s0 in list(rim_adj):
            if s0 in done or len(rim_adj[s0]) != 2: continue
            ring = [s0]; prev, cur = s0, rim_adj[s0][0]
            while cur is not s0 and len(rim_adj.get(cur, ())) == 2 and len(ring) <= len(rim_e):
                ring.append(cur); nx = [y for y in rim_adj[cur] if y is not prev]; prev, cur = cur, nx[0]
            done.update(ring)
            if cur is not s0: continue                                                  # open chain: leave it
            mat = max(f.material_index for v in ring[:1] for f in v.link_faces)
            P = np.array([v.co[:] for v in ring]); Hh = H_of('collar' if mat == 1 else 'cuff')
            th = 2 * np.pi * np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))] / (np.linalg.norm(np.diff(P, axis=0), axis=1).sum() + np.linalg.norm(P[0] - P[-1]))
            Bf = np.column_stack([np.ones_like(th)] + [f(h * th) for h in range(1, Hh + 1) for f in (np.cos, np.sin)])   # Fourier in arc length (uneven spacing)
            P2 = Bf @ np.linalg.lstsq(Bf, P, rcond=None)[0]
            for v, q in zip(ring, P2): v.co = q
            rim_fixed += 1
        report['rib_rims_lowpassed'] = rim_fixed
    rib_v = [v for v in bm.verts if any(f.material_index >= 1 for f in v.link_faces)]; pushed2 = 0; all_v = list(bm.verts)     # welding/refinement can leave rib vertices a hair inside the arm: final clearance pass
    rib_set = set(v.index for v in rib_v)
    for _ in range(12):                                                                # ribs keep 1.5 mm from the body, the rest of the garment 0.8 mm (cloth collision margin was 3 mm; armpits can still dip)
        for v in all_v:
            tgt = 0.0015 if v.index in rib_set else 0.0008
            loc, n_, _i, _d = _bvh.find_nearest(Vector(v.co)); sdv = (Vector(v.co) - loc).dot(n_)
            if sdv < tgt: v.co = v.co + n_ * (tgt - sdv); pushed2 += 1
    report['rib_final_pushes'] = pushed2
    if FOLD:                                                                            # ---- fold-over collar: open the stand at centre front, add the folded leaf
        sys.path.insert(0, os.path.join(HERE, 'components')); import collar_builder as CB
        ctr_, ax_, f_, g_, th_ = CB.ring_frame(COLLAR_ROWS); gh, rho = CB.gap_half_angle(COLLAR_ROWS, FOLD['front_gap_m'])
        bm.faces.ensure_lookup_table(); cut_f = []
        for f in bm.faces:
            if f.material_index != 1: continue
            r_ = np.array(f.calc_center_median()[:]) - ctr_
            if abs(math.atan2(r_ @ g_, r_ @ f_)) < gh: cut_f.append(f)
        all_collar = [f for f in bm.faces if f.material_index == 1]                    # the simulated ring only located the neckline; stand + leaf are
        bmesh.ops.delete(bm, geom=all_collar, context='FACES')                          # rebuilt as ONE construction object on the final shirt
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
        STAND_CUT = dict(stand_faces_cut_front=len(cut_f), gap_half_angle_deg=round(math.degrees(gh), 1), neck_radius_m=round(rho, 4))
        # the LEAF is built later, on the FINAL shirt (after smoothing / Solidify / Subsurf): built here, the shirt moved through it afterwards
    report['slivers_collapsed'] = nsl; print('collapsed %d sliver triangles (min altitude < 0.4 mm)' % nsl)
    report['refined_edges'] = nsplit; print('refined %d over-long edges (> %.0f mm)' % (nsplit, 1000 * pol['refine_edge_m']))
    bm.to_mesh(me); bm.free(); me.update()
    for poly in me.polygons: poly.use_smooth = True
    if pol.get('crease_rib_seams'):                                                     # Subsurf places a seam vertex from its irregular (triangulated body) neighbourhood,
        cr = me.attributes.get('crease_edge') or me.attributes.new('crease_edge', 'FLOAT', 'EDGE')   # which zig-zags the body/rib boundary: a full crease makes the
        e2f = {}                                                                        # seam a smooth B-spline of its own polyline (a sewn rib seam IS a crease)
        for pl in me.polygons:
            for ek in pl.edge_keys: e2f.setdefault(ek, set()).add(pl.material_index)
        nc = 0
        for e in me.edges:
            if len(e2f.get(e.key, ())) == 2: cr.data[e.index].value = float(pol['crease_rib_seams']); nc += 1
        report['creased_seam_edges'] = nc
    # sanity: closed-ring bands must be closed cylinders, i.e. no boundary edges other than the real garment openings
    bm2 = bmesh.new(); bm2.from_mesh(me); boundary = [e for e in bm2.edges if e.is_boundary]; nonman = [e for e in bm2.edges if len(e.link_faces) > 2]
    report['boundary_edges_welded'] = len(boundary); report['nonmanifold_edges_welded'] = len(nonman); bm2.free()
    pre_sd = {kind: surface_defects(me, half_window=3, edges=kind) for kind in ('seam', 'boundary')}     # before polish: where does a scallop enter?
    report['prepolish_defects'] = {kind: dict(waviness_max_mm=d_['seam_waviness_max_mm'], loops=d_['seam_loops']) for kind, d_ in pre_sd.items()}
    rim_w = max((l_['dev_max_mm'] for l_ in pre_sd['boundary']['seam_loops'] if l_['material'] >= 1), default=0.0)     # material >= 1: collar / cuff rib rims (0 is the body hem)
    report['prepolish_defects']['rib_rim_waviness_max_mm'] = rim_w
    print('pre-polish waviness: seams %.2f mm, rib rims %.2f mm, all free edges %.2f mm' % (pre_sd['seam']['seam_waviness_max_mm'], rim_w, pre_sd['boundary']['seam_waviness_max_mm']))
    print('welded mesh: boundary edges %d (hem, neck opening, sleeve openings expected), edges shared by >2 faces %d' % (len(boundary), len(nonman)))
    _a = np.array([v.co[:] for v in me.vertices]); _E = np.array([e.vertices[:] for e in me.edges]); _L = np.linalg.norm(_a[_E[:, 0]] - _a[_E[:, 1]], axis=1)
    print('welded mesh longest edges (m, position):', [(round(float(_L[k]), 4), _a[_E[k, 0]].round(3).tolist()) for k in np.argsort(-_L)[:5]])
    # polish
    bpy.context.view_layer.objects.active = tee; tee.select_set(True)
    cs = tee.modifiers.new('CorrectiveSmooth', 'CORRECTIVE_SMOOTH'); cs.factor = pol['smooth_factor']; cs.iterations = pol['smooth_iter']; cs.smooth_type = 'SIMPLE'; cs.rest_source = 'ORCO'; cs.use_only_smooth = True
    cs.vertex_group = 'rib'; cs.invert_vertex_group = True                          # smoothing shrinks closed rings: keep it off the collar/cuff bands, which are clean quad rings now
    cs2 = tee.modifiers.new('CorrectiveSmooth_Rib', 'CORRECTIVE_SMOOTH'); cs2.factor = pol['rib_smooth_factor']; cs2.iterations = pol['rib_smooth_iter']; cs2.smooth_type = 'SIMPLE'; cs2.rest_source = 'ORCO'; cs2.use_only_smooth = True; cs2.vertex_group = 'rib'; cs2.use_pin_boundary = True
    so = tee.modifiers.new('Solidify', 'SOLIDIFY'); so.thickness = pol['solidify_thickness']; so.offset = pol['solidify_offset']; so.use_rim = True; so.use_even_offset = bool(pol['solidify_even_offset']); so.nonmanifold_thickness_mode = 'FIXED'
    ss = tee.modifiers.new('Subsurf', 'SUBSURF'); ss.levels = pol['subsurf_levels']; ss.render_levels = pol['subsurf_levels']; ss.subdivision_type = 'CATMULL_CLARK'
    assert so.use_even_offset is False

    def edge_stats(label):
        a = np.array([v.co[:] for v in tee.data.vertices]); E = np.array([e.vertices[:] for e in tee.data.edges]); L_ = np.linalg.norm(a[E[:, 0]] - a[E[:, 1]], axis=1)
        k_ = int(np.argmax(L_)); print('  %-22s verts %7d  max edge %.4f  p99.9 %.4f  (longest at %s)' % (label, len(a), L_.max(), np.percentile(L_, 99.9), a[E[k_, 0]].round(3).tolist())); return float(L_.max())
    from mathutils.bvhtree import BVHTree
    from mathutils import Vector
    _bt = []
    for q in D['F']: _bt += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    bvh = BVHTree.FromPolygons([tuple(v) for v in BV], _bt)

    def penetration(label):                                                         # face-based signed distance to the FINAL (relaxed) body, for every vertex of the current mesh
        a = np.array([v.co[:] for v in tee.data.vertices]); sdv = np.empty(len(a))
        for k, p_ in enumerate(a):
            loc, n_, _i, _d = bvh.find_nearest(Vector(p_)); sdv[k] = (Vector(p_) - loc).dot(n_)
        bad = sdv < -0.0005; pan_after = np.zeros(len(a), int)
        for pl in tee.data.polygons: pan_after[list(pl.vertices)] = pl.material_index          # 0 body, 1 collar, 2/3 cuffs
        r_ = dict(verts_inside_body=int(bad.sum()), min_signed_dist_mm=round(float(1000 * sdv.min()), 2)); print('  penetration', label, r_)
        if bad.any(): print('    inside-vertex bbox', a[bad].min(0).round(3).tolist(), a[bad].max(0).round(3).tolist(), '| panels', dict(zip(*np.unique(pan_after[bad], return_counts=True))) if len(pan_after) == len(a) else '')
        return r_
    if pol['rib_smooth_iter'] <= 0: tee.modifiers.remove(tee.modifiers['CorrectiveSmooth_Rib'])
    edge_max = {}; pen = {}; tee_last_smooth = [m.name for m in tee.modifiers if m.name.startswith('CorrectiveSmooth')][-1]
    for m in [m.name for m in tee.modifiers]:
        bpy.ops.object.modifier_apply(modifier=m); edge_max[m] = edge_stats('after ' + m)
        if m == tee_last_smooth:
            _a = np.array([v.co[:] for v in tee.data.vertices]); _E = np.array([e.vertices[:] for e in tee.data.edges]); _L = np.linalg.norm(_a[_E[:, 0]] - _a[_E[:, 1]], axis=1)
            print('  longest edges after smooth:', [(round(float(_L[k]), 4), _a[_E[k, 0]].round(3).tolist()) for k in np.argsort(-_L)[:4]])
            moved_post = 0                                                      # smoothing can pull armpit vertices back into the arm: one more clearance pass on the smoothed surface
            # The pass moves a coherent PATCH, not single vertices: each needed push (capped at 2 mm per pass) is extended to the neighbours with a
            # 0.6-per-ring falloff.  Per-vertex pushes where cloth is pinched between two body parts (a POSE_01 hand on the hip) alternate between
            # the two surfaces and stretched single hem edges to 43 mm.
            import scipy.sparse as _sp
            _Vp = np.array([v.co[:] for v in tee.data.vertices]); _V0 = _Vp.copy(); _Ee = np.array([e.vertices[:] for e in tee.data.edges]); _n = len(_Vp)
            _Adj = _sp.coo_matrix((np.ones(2 * len(_Ee)), (np.r_[_Ee[:, 0], _Ee[:, 1]], np.r_[_Ee[:, 1], _Ee[:, 0]])), shape=(_n, _n)).tocsr()
            _Adj = _sp.diags(1.0 / np.maximum(np.asarray(_Adj.sum(1)).ravel(), 1)) @ _Adj
            for _ in range(8):
                need = np.zeros((_n, 3)); hasn = np.zeros(_n, bool)
                for i in range(_n):
                    loc, n_, _i, _d = bvh.find_nearest(Vector(_Vp[i])); sdv = (Vector(_Vp[i]) - loc).dot(n_)
                    if sdv < 0.0008: need[i] = np.array(n_) * min(0.0008 - sdv, 0.002); hasn[i] = True
                if not hasn.any(): break
                moved_post += int(hasn.sum()); Fd = need.copy()
                for _r in range(4): Fd = np.where(hasn[:, None], need, 0.6 * (_Adj @ Fd))
                _Vp += Fd
            for i, v in enumerate(tee.data.vertices): v.co = _Vp[i]
            _dm = np.linalg.norm(_Vp - _V0, axis=1); report['post_smooth_pushes'] = moved_post
            report['post_smooth_max_move_mm'] = round(1000 * float(_dm.max()), 2); edge_stats('after clearance pass')
            pen['outer_surface'] = penetration('outer surface (before Solidify)')
    report['penetration'] = pen
    report['max_edge_m_after'] = edge_max
    # final mesh checks
    me = tee.data; Vf = np.array([v.co[:] for v in me.vertices]); tri = []
    me.calc_loop_triangles(); T = np.array([t.vertices[:] for t in me.loop_triangles]); ar = tri_areas(Vf, T)
    report['final'] = dict(verts=len(Vf), polys=len(me.polygons), tris=len(T), degenerate_triangles=int((ar < 1e-10).sum()), min_tri_area_m2=float(ar.min()), median_tri_area_m2=float(np.median(ar)),
                           bbox_m=[round(float(x), 4) for x in (Vf.max(0) - Vf.min(0))], z_range_m=[round(float(Vf[:, 2].min()), 4), round(float(Vf[:, 2].max()), 4)])
    report['final']['verts_inside_body_outer_surface'] = pen['outer_surface']['verts_inside_body']
    print('final:', report['final'])
    # ---- the quality gate
    fails = []
    if report['final']['degenerate_triangles'] != gate['degenerate_triangles']: fails.append('degenerate_triangles=%d' % report['final']['degenerate_triangles'])
    if report['presim']['degenerate_triangles_presim'] != 0: fails.append('presim degenerate triangles')
    if seam_over > gate['seam_pairs_over_2mm_max']: fails.append('seam_pairs_over_2mm=%d > %d' % (seam_over, gate['seam_pairs_over_2mm_max']))
    if max(edge_max['Solidify'], edge_max['Subsurf']) >= gate['max_edge_after_solidify_m']: fails.append('max_edge_after_solidify=%.4f' % max(edge_max['Solidify'], edge_max['Subsurf']))
    if pen['outer_surface']['verts_inside_body'] > gate['inside_body_verts_max']: fails.append('outer-surface verts inside body=%d' % pen['outer_surface']['verts_inside_body'])
    if 'max_seam_waviness_mm' in gate or 'max_spike_ratio' in gate:
        sd = surface_defects(me); report['surface_defects'] = sd; print('surface defects: seam waviness max %.2f mm, spike ratio max %.2f at %s' % (sd['seam_waviness_max_mm'], sd['spike_ratio_max'], sd['spike_at']))
        if sd['seam_waviness_max_mm'] > gate.get('max_seam_waviness_mm', 1e9): fails.append('rib seam waviness %.2f mm' % sd['seam_waviness_max_mm'])
        if report['prepolish_defects']['rib_rim_waviness_max_mm'] > gate.get('max_rim_waviness_mm', 1e9): fails.append('rib rim waviness %.2f mm' % report['prepolish_defects']['rib_rim_waviness_max_mm'])
        if sd['spike_ratio_max'] > gate.get('max_spike_ratio', 1e9): fails.append('vertex spike ratio %.2f at %s' % (sd['spike_ratio_max'], sd['spike_at']))
    PLK = CONS.get('placket')
    if FOLD or PLK:                                                                   # construction components: real geometry, checked like the shell
        sys.path.insert(0, os.path.join(HERE, 'components')); import collar_builder as CB, placket_builder as PB
        crep = report.setdefault('construction', {})
        if FOLD:
            lob, c_ = CB.build_leaf_object(bpy, tee, COLLAR_ROWS, FOLD, W, body_bvh=bvh)
            c_ = dict(stand='constructed from the draped neckline, %.0f mm, lean %.0f deg' % (1000 * spec['pattern']['collar_width'], FOLD.get('stand_lean_deg', 6.0)), front_gap_m=FOLD['front_gap_m'], **STAND_CUT, **c_)
            crep['collar'] = c_; print('fold-over collar:', c_)
            if c_['final_leaf_min_gap_over_shirt_mm'] < 0.5: fails.append('collar leaf (final mesh) within %.2f mm of the shirt' % c_['final_leaf_min_gap_over_shirt_mm'])
            if c_['final_leaf_degenerate_triangles']: fails.append('collar leaf degenerate triangles=%d' % c_['final_leaf_degenerate_triangles'])
            if c_['final_collar_verts_inside_body']: fails.append('collar verts inside the body=%d' % c_['final_collar_verts_inside_body'])
            if c_.get('final_leaf_verts_under_shirt'): fails.append('collar leaf verts under the shirt=%d' % c_['final_leaf_verts_under_shirt'])
            if c_['rim_scallop_max_mm'] > CONS.get('max_rim_scallop_mm', 2.0): fails.append('collar rim scalloped (%.2f mm)' % c_['rim_scallop_max_mm'])
            if c_.get('stand_seated'):                                                 # FBL-008 gate: the stand base ring must lie ON the final shell neckline
                if c_['seat_ring_within_1mm_pct'] < 100.0: fails.append('collar stand ring seated %.2f %% within 1 mm of the shell (max %.2f mm)' % (c_['seat_ring_within_1mm_pct'], c_['seat_ring_surface_mm'][2]))
                if c_['seat_ring_inside_shell']: fails.append('collar stand ring vertices inside the shell=%d' % c_['seat_ring_inside_shell'])
            elif FOLD.get('seat_stand', True): fails.append('collar stand not seated on the shell')
        if PLK:
            th_cf = CB.ring_frame(COLLAR_ROWS)[4]; cf_base = CB._periodic_interp(th_cf, COLLAR_ROWS[0], np.array([0.0]))[0]
            pob, prep = PB.build_placket(bpy, tee, PLK, top_z=float(cf_base[2]) - PLK.get('top_below_neck_m', 0.004), cf_x=float(cf_base[0]), neck_z=float(cf_base[2]))
            bob, brep = PB.build_buttons(bpy, pob, PLK, top_z=float(cf_base[2]) - PLK.get('top_below_neck_m', 0.004))
            prep.pop('centre_surface', None); crep['placket'] = prep; crep['buttons'] = brep
            if prep['ray_misses'] > 0.1 * prep['rows'] * prep['cols']: fails.append('placket ray misses %d' % prep['ray_misses'])
            if len(brep) != len(PLK['buttons']): fails.append('buttons placed %d of %d' % (len(brep), len(PLK['buttons'])))
            print('placket:', prep, '| buttons:', brep)
        slots = sorted(g.name for g in tee.vertex_groups if g.name not in ('rib', 'pin')) + (sorted(g.name for g in bpy.data.objects['collar'].vertex_groups) if 'collar' in bpy.data.objects else []) + sorted(n for n in ('placket_overlap', 'placket_buttons') if n in bpy.data.objects)
        crep['slots_present'] = slots; print('structural slots present:', slots)
    report['gate_passed'] = not fails; report['gate_failures'] = fails
    qpath = os.path.splitext(out_blend)[0] + '.quality.json'
    if fails:
        json.dump(report, open(os.path.join(W, 'quality_FAILED.json'), 'w'), indent=1); bpy.ops.wm.save_as_mainfile(filepath=os.path.join(W, 'baked_FAILED_gate.blend'), compress=True)
        print('QUALITY GATE FAILED:', fails, '(inspection copy: baked_FAILED_gate.blend; nothing written to assets/)'); sys.exit(2)
    # ---- assemble the master: Garment + Mannequin collections, no collider, no lights (render_sku owns the studio)
    g = bpy.data.collections.new('Garment'); sc.collection.children.link(g)
    for c in list(tee.users_collection): c.objects.unlink(tee)
    g.objects.link(tee); tee.name = 'Tee'
    for nm in ('collar', 'placket_overlap', 'placket_buttons'):                        # construction components join the garment collection
        if nm in bpy.data.objects:
            ob = bpy.data.objects[nm]
            for c in list(ob.users_collection): c.objects.unlink(ob)
            g.objects.link(ob)
    bpy.data.objects.remove(colo, do_unlink=True)
    mc = bpy.data.collections.new('Mannequin'); sc.collection.children.link(mc)
    with bpy.data.libraries.load(os.path.join(W, 'mannequin_posed.blend')) as (src, dst): dst.objects = ['Mannequin']
    mo = dst.objects[0]; mc.objects.link(mo)
    mm = bpy.data.materials.new('Mannequin_Skin'); mm.use_nodes = True; mo.data.materials.clear(); mo.data.materials.append(mm)
    for poly in mo.data.polygons: poly.use_smooth = True
    sc['archetype'] = spec['archetype']; sc['quality_gate_passed'] = True; sc['garment_pose'] = (spec['body'].get('canonical_pose') or {}).get('pose_id', 'A_POSE')
    os.makedirs(os.path.dirname(out_blend), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend, compress=True)
    report['master_bytes'] = os.path.getsize(out_blend); json.dump(report, open(qpath, 'w'), indent=1)
    print('QUALITY GATE PASSED. Saved', out_blend, '(%.1f MB)' % (report['master_bytes'] / 1e6), 'and', qpath)


# ------------------------------------------------------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', required=True); ap.add_argument('--work', required=True)
    ap.add_argument('--body', default=None, help='03_full_rigged.blend; fetched from git (spec body.git_ref) when omitted')
    ap.add_argument('--stages', default='collider,setup,sim,bake'); ap.add_argument('--stage-run', default=None, help='internal: run one stage in this process')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(); spec = load_spec(a.spec); W = os.path.abspath(a.work); os.makedirs(W, exist_ok=True)
    out = os.path.abspath(a.out or os.path.join(ROOT, spec['output']))
    body = os.path.join(W, 'body_full.blend')
    if a.body and os.path.abspath(a.body) != body:
        import shutil; shutil.copy(a.body, body)
    elif not os.path.exists(body):
        with open(body, 'wb') as fh: subprocess.run(['git', 'show', f"{spec['body']['git_ref']}:{spec['body']['blend']}"], stdout=fh, check=True, cwd=ROOT)
    if a.stage_run:                                              # each stage in its own process: no Blender state leaks between stages
        {'collider': lambda: stage_collider(spec, W, body), 'setup': lambda: stage_setup(spec, W), 'sim': lambda: stage_sim(spec, W), 'bake': lambda: stage_bake(spec, W, out)}[a.stage_run]()
        return
    for st in a.stages.split(','):
        print('\n=== stage', st, flush=True); t = time.time()
        r = subprocess.run([sys.executable, os.path.abspath(__file__), '--spec', a.spec, '--work', W, '--body', body, '--stage-run', st, '--out', out])
        if r.returncode: sys.exit(r.returncode)
        print('=== stage %s done in %.0fs' % (st, time.time() - t))


if __name__ == '__main__':
    main()
