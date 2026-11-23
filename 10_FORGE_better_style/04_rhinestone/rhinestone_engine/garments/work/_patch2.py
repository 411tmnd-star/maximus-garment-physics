p='tools/art.py'; s=open(p).read()
a=s.index("def tile_shields"); b=s.index("# ------------------------------------------------------------------ back script + crown")
new = '''def tile_shield(tile_png, tile_w_mm, scale=1.0):
    """One crowned BST shield of the half-drop repeat -> (stones [(x_mm, y_mm, R_mm, cls)] about the shield centre, y down;
    shield centre in tile fractions (fx, fy from the top); tile size mm). `scale` shrinks the drawing before packing (stones keep
    their size). Outer frame: one row SS16 on its medial line. Inner outline, crown, letters: SS6."""
    im = np.asarray(Image.open(tile_png).convert('L')) > 127; H, W = im.shape; mmpp = tile_w_mm / W
    big = np.tile(im, (2, 2)).astype(np.uint8)
    n, lab, stats, cen = cv2.connectedComponentsWithStats(big, connectivity=8)
    x0, y0, x1, y1 = int(0.03 * W), int(0.90 * H), int(0.47 * W), int(1.60 * H)
    ids = [i for i in range(1, n) if stats[i, 0] >= x0 and stats[i, 1] >= y0 and stats[i, 0] + stats[i, 2] <= x1 and stats[i, 1] + stats[i, 3] <= y1 and stats[i, 4] > 30]
    frame = max(ids, key=lambda i: stats[i, 2] * stats[i, 3]); pad = 40
    bx0 = stats[frame, 0] - pad; by0 = stats[frame, 1] - pad; bx1 = stats[frame, 0] + stats[frame, 2] + pad; by1 = stats[frame, 1] + stats[frame, 3] + pad
    sub = lab[by0:by1, bx0:bx1]; fm = (sub == frame).astype(np.uint8); rest = np.isin(sub, [i for i in ids if i != frame]).astype(np.uint8)
    if scale != 1.0:
        sz = (int(sub.shape[1] * scale), int(sub.shape[0] * scale))
        fm = (cv2.resize(fm * 255, sz, interpolation=cv2.INTER_AREA) > 127).astype(np.uint8); rest = (cv2.resize(rest * 255, sz, interpolation=cv2.INTER_AREA) > 127).astype(np.uint8)
    grid = Grid2(4.0); stones = []
    sk = zhang_suen(fm); d_sk = cv2.distanceTransform((1 - sk).astype(np.uint8), cv2.DIST_L2, 5)
    st = contour_fill(d_sk <= (2.0 + 0.16) / mmpp, mmpp, 2.0, gap=0.3, ensure_min=False, max_rows=1, grid=grid, fill_pass=False)
    stones += [(x, y, 2.0, 0) for x, y, _, _ in st]
    st = contour_fill(rest, mmpp, 1.0, gap=0.25, ensure_min=True, grid=grid)
    stones += [(x, y, 1.0, 0) for x, y, _, _ in st]
    cx, cy = fm.shape[1] * mmpp / 2, fm.shape[0] * mmpp / 2
    fx = ((bx0 + bx1) / 2 % W) / W; fy = ((by0 + by1) / 2 % H) / H
    return [(x - cx, y - cy, r, c) for x, y, r, c in stones], (fx, fy), (W * mmpp, H * mmpp)


'''
s=s[:a]+new+s[b:]; open(p,'w').write(s)

p='tools/solve_shirt.py'; s=open(p).read()
s=s.replace("width = s_edge[:, -1]; s_edge[s_edge","width = s_edge[:, -1].copy(); s_edge[s_edge")
s=s.replace("pitch = 2 * R + 0.15 * MM","pitch = 2 * R + 0.20 * MM")
s=s.replace("0.25 * MM","0.35 * MM")
a=s.index("        e1 = C.unit(np.cross(a, [0, 1, 0])); e2 = np.cross(a, e1)"); b=s.index("        # hem edge per section = first hit")
new = '''        NTH = 1080
        for pas in (0, 1):
            e1 = C.unit(np.cross(a, [0, 1, 0])); e2 = np.cross(a, e1)
            tt = (tee.V[sm] - c0) @ a; t_hi = tt.max() + 2 * MM; t_lo = tt.max() - (90 if pas == 0 else 45) * MM
            ts = np.arange(t_hi, t_lo, -(1.0 if pas == 0 else 0.25) * MM); nth = 180 if pas == 0 else NTH; sec = np.full((nth, len(ts), 3), np.nan)
            for i in range(nth):
                th = 2 * math.pi * i / nth; dr = math.cos(th) * e1 + math.sin(th) * e2
                for j, t in enumerate(ts):
                    loc, nn, idx, dd = sbvh.ray_cast(Vector(c0 + t * a + 0.30 * dr), Vector(-dr), 0.30)
                    if loc is not None: sec[i, j] = loc
            if pas == 0:                                                           # hem ring -> its plane normal is the sleeve's own axis
                ring = np.array([sec[i, np.argmax(np.isfinite(sec[i, :, 0]))] for i in range(nth) if np.isfinite(sec[i, :, 0]).any()])
                rc = ring.mean(0); nrm = np.linalg.svd(ring - rc)[2][2]
                if nrm @ a < 0: nrm = -nrm
                tilt = math.degrees(math.acos(min(1.0, abs(nrm @ a)))); a = nrm; c0 = rc - 0.03 * a
'''
s=s[:a]+new+s[b:]
s=s.replace("rep['axis'] = a.round(4).tolist()","rep['axis'] = a.round(4).tolist(); rep['hem_plane_vs_arm_axis_deg'] = round(tilt, 1)")
a=s.index("def place_canvas("); b=s.index("if 'E' in ONLY:")
new = '''def canvas_point(key, x_mm, y_mm):
    h = tee.uv_to_surface(key, (x_mm / CANVAS[0], 1.0 - y_mm / CANVAS[1])); return None if h is None else h[0]


def fit_frame(centre, ex0, ey0, half_w, half_h):
    """least-squares plane through the outer surface under a design; x axis kept as close to ex0 as the plane allows."""
    n0 = C.unit(np.cross(ex0, ey0)); pts = []
    for ax in np.linspace(-1, 1, 9):
        for ay in np.linspace(-1, 1, 9):
            q = centre + ax * half_w * ex0 + ay * half_h * ey0
            loc, nn, idx, dd = tee.bvh.ray_cast(Vector(q + 0.06 * n0), Vector(-n0), 0.12)
            if loc is not None: pts.append(np.array(loc))
    pts = np.array(pts); c = pts.mean(0); n = np.linalg.svd(pts - c)[2][2]
    if n @ n0 < 0: n = -n
    ex = C.unit(ex0 - (ex0 @ n) * n); ey = np.cross(n, ex)
    if ey @ ey0 < 0: ey = -ey
    return ex, ey, n


def place_planar(zone, pts, centre, ex, ey, n, prio, depth=0.03, guard=None):
    """pts (x_mm, y_mm down, R_mm, cls) about `centre`; projected along -n onto the outer surface. A parallel projection onto a
    surface can only lengthen distances, so gaps designed in the plane are never reduced on the garment."""
    k = 0
    for x, y, r, c in pts:
        q = centre + x * MM * ex - y * MM * ey
        loc, nn, idx, dd = tee.bvh.ray_cast(Vector(q + depth * n), Vector(-n), 2 * depth)
        if loc is None: rej(zone, 'off_panel'); continue
        P0 = np.array(loc); nn = np.array(nn)
        if nn @ n < 0.2: rej(zone, 'steep'); continue
        if guard is not None and guard(P0): continue
        st, why = seater.seat(P0, nn, ex, r * MM)
        if st is None: rej(zone, why); continue
        add(st, zone, r * MM, c, prio); k += 1
    return k


def canvas_frame(key, cx, cy, hw, hh):
    c = canvas_point(key, cx, cy); a_ = canvas_point(key, cx + 60, cy); b_ = canvas_point(key, cx - 60, cy); u_ = canvas_point(key, cx, cy - 60); d_ = canvas_point(key, cx, cy + 60)
    ex0 = C.unit(a_ - b_); ey0 = C.unit(u_ - d_); ey0 = C.unit(ey0 - (ey0 @ ex0) * ex0)
    return (c,) + fit_frame(c, ex0, ey0, hw * MM, hh * MM)


'''
s=s[:a]+new+s[b:]
a=s.index("if 'E' in ONLY:"); b=s.index("# ================================================================== exact clearance")
new = '''if 'E' in ONLY:
    m, cimg, mp = art.elephant_masks(HE + 'engineering/PHASE3_2026-11-09/elephant_evidence/elephant_final_548x852.png', height_mm=235.0)
    S = art.contour_fill(m, mp, 1.0, gap=0.25, ensure_min=False, cls_img=cimg); ew, eh = m.shape[1] * mp, m.shape[0] * mp
    c, ex, ey, n = canvas_frame('front', 73.5 + 159 / 2, 392.0 + 252 / 2, ew / 2, eh / 2)
    k = place_planar('E_elephant', [(x - ew / 2, y - eh / 2, 1.0, cc) for x, y, _, cc in S], c, ex, ey, n, 1)
    report['zones']['E_elephant'] = dict(planned=len(S), placed=k, size_mm=[round(ew, 1), round(eh, 1)], plane_normal=n.round(3).tolist()); log('zone E', k, '/', len(S))
if 'C' in ONLY:
    m, mp = art.script_mask(width_mm=300.0); S = art.contour_fill(m, mp, 1.0, gap=0.25); sw, sh = m.shape[1] * mp, m.shape[0] * mp
    c, ex, ey, n = canvas_frame('back', 319.0, 300.0, sw / 2, sh / 2 + 40)
    n1 = place_planar('C_script', [(x - sw / 2, y - sh / 2, 1.0, 0) for x, y, _, _ in S], c, ex, ey, n, 1)
    mc, mp2 = art.crown_mask(74.0); Sc = art.contour_fill(mc, mp2, 1.0, gap=0.25); cw, ch = mc.shape[1] * mp2, mc.shape[0] * mp2
    n2 = place_planar('C_crown', [(x - cw / 2 + 14, y - sh / 2 - ch + 4, 1.0, 0) for x, y, _, _ in Sc], c, ex, ey, n, 1)
    report['zones']['C_script'] = dict(planned=len(S), placed=n1, size_mm=[round(sw, 1), round(sh, 1)], font='Brush Script MT (stand-in)')
    report['zones']['C_crown'] = dict(planned=len(Sc), placed=n2, size_mm=[round(cw, 1), round(ch, 1)]); log('zone C', n1, n2)

if 'D' in ONLY:
    SC = 0.90; shield, (fx, fy), (Wm, Hm) = art.tile_shield(HE + 'src_catalog/better_style_catalog/assets/artwork_camp/lattice_tile.png', 181.4, SC)
    uvp, mir = C.print_uv_tri(tee); tee.build_uv('print', uvp, tee.outer)
    art_pts = np.array([s['P'] for s in stones if s['zone'][0] in 'CE'])
    art_kd = cKDTree(art_pts) if len(art_pts) else None
    lab = [canvas_point('front', (1321 + ax * 307) / 3.21, (407 + ay * 85) / 3.21) for ax in np.linspace(0, 1, 12) for ay in np.linspace(0, 1, 4)]
    lab_kd = cKDTree(np.array([h for h in lab if h is not None]))
    def guard(P0):
        if art_kd is not None and art_kd.query(P0)[0] < 4.5 * MM: rej('D_monogram', 'under_art'); return True
        if lab_kd.query(P0)[0] < 7.0 * MM: rej('D_monogram', 'under_label'); return True
        return False
    um, vm = uvp[tee.outer][..., 0], uvp[tee.outer][..., 1]; tw_uv, th_uv = (181.4 * MM) / 0.30, (181.4 * Hm / Wm * MM) / 0.30
    occ = set(zip(np.floor(um.ravel() / tw_uv).astype(int).tolist(), np.floor(vm.ravel() / th_uv).astype(int).tolist()))
    def P_uv(u, v):
        h = tee.uv_to_surface('print', (u, v)); return None if h is None else h[0]
    centres = []; k = 0; inst = 0; dl = 0.02 / 0.30
    for (i, j) in sorted(occ):
        for (ax, ay) in ((fx, fy), ((fx + 0.5) % 1.0, (fy + 0.5) % 1.0)):
            u0, v0 = (i + ax) * tw_uv, (j + 1 - ay) * th_uv; c = P_uv(u0, v0)
            if c is None: continue
            nb = [P_uv(u0 + dl, v0), P_uv(u0 - dl, v0), P_uv(u0, v0 + dl), P_uv(u0, v0 - dl)]
            if any(q is None for q in nb): rej('D_monogram', 'instance_at_panel_edge'); continue
            ex0 = C.unit(nb[0] - nb[1]); ey0 = nb[2] - nb[3]; ey0 = C.unit(ey0 - (ey0 @ ex0) * ex0)
            hd = tee.uv_to_surface('print', (u0, v0)); nface = tee.FN[hd[4]]
            if np.cross(ex0, ey0) @ nface < 0: ex0 = -ex0                               # mirrored island: keep the shield readable from outside
            if centres and np.min(np.linalg.norm(np.array(centres) - c, axis=1)) < 70 * MM: rej('D_monogram', 'instance_overlaps_other_panel'); continue
            ex, ey, n = fit_frame(c, ex0, ey0, 0.03, 0.03); centres.append(c); inst += 1
            k += place_planar('D_monogram', shield, c, ex, ey, n, 2, depth=0.02, guard=guard)
    report['zones']['D_monogram'] = dict(stones_per_shield=len(shield), drawing_scale=SC, instances=inst, placed=k); log('zone D', k, 'stones in', inst, 'shields of', len(shield))

'''
s=s[:a]+new+s[b:]
open(p,'w').write(s); print('ok')
