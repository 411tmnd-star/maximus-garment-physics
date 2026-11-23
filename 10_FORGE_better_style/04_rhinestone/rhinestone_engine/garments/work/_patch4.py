p='tools/solve_shirt.py'; s=open(p).read()
# shields upright: up axis = world +Z projected on the surface, right = up x n (readable from outside)
old='''            ex0 = C.unit(nb[0] - nb[1]); ey0 = nb[2] - nb[3]; ey0 = C.unit(ey0 - (ey0 @ ex0) * ex0)
            hd = tee.uv_to_surface('print', (u0, v0)); nface = tee.FN[hd[4]]
            if np.cross(ex0, ey0) @ nface < 0: ex0 = -ex0                               # mirrored island: keep the shield readable from outside
'''
new='''            n0 = C.unit(np.cross(nb[0] - nb[1], nb[2] - nb[3])); hd = tee.uv_to_surface('print', (u0, v0))
            if n0 @ tee.FN[hd[4]] < 0: n0 = -n0
            up = np.array([0.0, 0.0, 1.0]); ey0 = up - (up @ n0) * n0
            if np.linalg.norm(ey0) < 0.35: rej('D_monogram', 'instance_on_horizontal_surface'); continue      # shoulder top: no upright direction
            ey0 = C.unit(ey0); ex0 = np.cross(ey0, n0)                                   # upright and readable from outside on every panel
            if abs(c[0]) < 0.060 and c[1] < 0: rej('D_monogram', 'instance_on_placket'); continue
'''
assert old in s; s=s.replace(old,new)
old='''            ex, ey, n = fit_frame(c, ex0, ey0, 0.03, 0.03); centres.append(c); inst += 1
            k += place_planar('D_monogram', shield, c, ex, ey, n, 2, depth=0.02, guard=guard)'''
new='''            ex, ey, n = fit_frame(c, ex0, ey0, 0.03, 0.03); start = len(stones)
            kk = place_planar('D_monogram', shield, c, ex, ey, n, 2, depth=0.02, guard=guard)
            if kk < 0.72 * len(shield):                                                 # a shield is whole or absent: no fragments at seams, collar or art
                del stones[start:]; rej('D_monogram', 'instance_incomplete'); continue
            centres.append(c); inst += 1; k += kk'''
assert old in s; s=s.replace(old,new)
s=s.replace("if art_kd is not None and art_kd.query(P0)[0] < 4.5 * MM","if art_kd is not None and art_kd.query(P0)[0] < 6.0 * MM")
# elephant: outline row in SS10 first, then SS6 fill
old='''    S = art.contour_fill(m, mp, 1.0, gap=0.25, ensure_min=False, cls_img=cimg); ew, eh = m.shape[1] * mp, m.shape[0] * mp'''
new='''    S = art.contour_fill(m, mp, 1.0, gap=0.25, ensure_min=False, cls_img=cimg); ew, eh = m.shape[1] * mp, m.shape[0] * mp'''
assert old in s
open(p,'w').write(s)

p='tools/art.py'; s=open(p).read()
old='''    edge = cv2.morphologyEx(reg.astype(np.uint8), cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.5 / mmpp) | 1,) * 2)).astype(bool)
    return alpha & ~edge, reg.astype(np.uint8), mmpp'''
new='''    edge = cv2.morphologyEx(reg.astype(np.uint8), cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(1.3 / mmpp) | 1,) * 2)).astype(bool)
    # drawn lines inside the body (ear, trunk, legs): strong luminance edges of the smoothed picture become stone-free channels
    g = cv2.GaussianBlur(cv2.cvtColor(a[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY), (0, 0), 1.2 / mmpp)
    lines = cv2.Canny(g, 14, 34) > 0
    lines = cv2.dilate(lines.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(1.1 / mmpp) | 1,) * 2)).astype(bool) & ~reg
    return alpha & ~edge & ~lines, reg.astype(np.uint8), mmpp'''
assert old in s; s=s.replace(old,new); open(p,'w').write(s)

p='tools/render_set.py'; s=open(p).read()
s=s.replace("'blue': dict(body='#193D7C', main=(0.92, 0.94, 1.0), accent=(0.05, 0.16, 0.75)","'blue': dict(body='#193D7C', main=(0.86, 0.90, 1.0), accent=(1.0, 0.74, 0.30)")
s=s.replace("b.inputs['Roughness'].default_value = 0.10","b.inputs['Roughness'].default_value = 0.16")
s=s.replace("b.inputs['Emission Strength'].default_value = 1.6","b.inputs['Emission Strength'].default_value = 0.55")
s=s.replace("vo.inputs['Scale'].default_value = 2.6","vo.inputs['Scale'].default_value = 3.4")
s=s.replace("ramp.color_ramp.elements[0].position = 0.25; ramp.color_ramp.elements[1].position = 0.60","ramp.color_ramp.elements[0].position = 0.30; ramp.color_ramp.elements[1].position = 0.75; ramp.color_ramp.elements[0].color = (0.10, 0.10, 0.12, 1)")
open(p,'w').write(s)

p='tools/stones_core.py'; s=open(p).read()
s=s.replace("'SS20': 2.35 * MM}","'SS20': 2.35 * MM, 'SS30': 3.20 * MM}"); open(p,'w').write(s)

p='tools/build_trousers.py'; s=open(p).read()
old=s[s.index("def prof(z):"):s.index("def leg(sx):")]
new='''def prof(z):
    """leg section at height z -> (x centre, semi-axis a across, semi-axis b front-back, y centre)"""
    if z >= Z_C:
        t = (z - Z_C) / (Z_W - Z_C); s = t * t * (3 - 2 * t)
        a = 0.108 + (0.110 - 0.108) * s; xc = 0.097 + (0.066 - 0.097) * s; b = 0.100 + (0.108 - 0.100) * s; yc = 0.005
        hip = math.exp(-((z - 0.92) / 0.08) ** 2); xc += 0.010 * hip; b += 0.004 * hip
    else:
        t = (Z_C - z) / (Z_C - Z_H); s = t ** 0.8
        a = 0.108 + (0.082 - 0.108) * s; b = 0.100 + (0.078 - 0.100) * s; xc = 0.097 + (0.112 - 0.097) * t; yc = 0.005 + 0.012 * t
    return xc, a, b, yc


'''
s=s.replace(old,new)
s=s.replace("R = C.SS['SS20']\n        for s in C.even_positions(L, 12.0 * MM, False, phase):","R = C.SS['SS30']\n        for s in C.even_positions(L, 13.0 * MM, False, phase):")
s=s.replace("pitch_mm=12.0, size='SS20'","pitch_mm=13.0, size='SS30 (6.4 mm)'")
s=s.replace("181.4, 0.80)","181.4, 1.0)").replace("drawing_scale=0.80","drawing_scale=1.0")
s=s.replace("c0 = np.array([0.112, -0.2, 0.845])","c0 = np.array([0.105, -0.2, 0.86])")
open(p,'w').write(s); print('ok')
