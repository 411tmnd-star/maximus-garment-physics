p='tools/solve_shirt.py'; s=open(p).read()
# rows = iso-lines of ONE distance field from the band's source edge
s=s.replace("s_edge: arc length of every section sample from the source edge (ns, nj). rows: [(d, R, phase)].",
            "s_edge: distance field phi of every section sample from the band's source edge (ns, nj). rows: [(d, R, phase)].")
s=s.replace("Every row is the station d_k of the SAME sections -> the row-to-row separation is d_{k+1} - d_k along each section by construction.",
            "Every row is the iso-line phi = d_k of the SAME field -> rows are parallel offsets of the edge and their separation is d_{k+1} - d_k by construction.")
s=s.replace('''            j = int(np.searchsorted(s[ok], d)); idx = np.nonzero(ok)[0]
            if j <= 0 or j >= len(idx): continue
            a, b = idx[j - 1], idx[j]; w = (d - s[a]) / max(s[b] - s[a], 1e-12); pts[i] = (1 - w) * sections[i, a] + w * sections[i, b]''',
'''            idx = np.nonzero(ok)[0]; cr = np.nonzero((s[idx][:-1] < d) & (s[idx][1:] >= d))[0]      # first crossing of the level along the section
            if not len(cr): continue
            a, b = idx[cr[0]], idx[cr[0] + 1]; w = (d - s[a]) / max(s[b] - s[a], 1e-12); pts[i] = (1 - w) * sections[i, a] + w * sections[i, b]''')
# collar: field from the outer edge + both front edges
a=s.index("    s_edge = np.concatenate([np.zeros((len(G), 1)), np.cumsum(np.linalg.norm(np.diff(G, axis=1), axis=2), axis=1)], 1)\n    width")
b=s.index("    rows = [(2.3 * MM, C.SS['SS10'], 0.0)]; d = 2.3 * MM")
new='''    arc = np.concatenate([np.zeros((len(G), 1)), np.cumsum(np.linalg.norm(np.diff(G, axis=1), axis=2), axis=1)], 1)
    width = arc[:, -1].copy(); keep_j = arc <= (width[:, None] - 9.0 * MM)                 # stay clear of the rolled fold
    src = np.vstack([C.resample(G[0, ::-1], 0.1 * MM)[0], C.resample(G[:, 0], 0.1 * MM)[0], C.resample(G[-1], 0.1 * MM)[0]])   # front edge, outer edge, front edge
    s_edge = cKDTree(src).query(G.reshape(-1, 3))[0].reshape(G.shape[:2]); s_edge[~keep_j] = np.nan
'''
s=s[:a]+new+s[b:]
s=s.replace("    while d < np.nanmax(width):","    while d < np.nanmax(s_edge):")
# cuff: field from the smoothed hem ring
a=s.index("        s_edge = np.full(sec.shape[:2], np.nan)\n        for i in range(NTH):"); b=s.index("        CORE = (lambda c0_, a_:")
new='''        s_edge = np.full(sec.shape[:2], np.nan); ring = np.full((NTH, 3), np.nan)
        for i in range(NTH):
            okj = np.isfinite(sec[i, :, 0]) & (ts <= tf_s[i] + 0.25 * MM)
            if okj.sum() < 4: continue
            idx = np.nonzero(okj)[0]; P = sec[i, idx]; stp = np.linalg.norm(np.diff(P, axis=0), axis=1)
            brk = np.nonzero(stp > 1.5 * MM)[0]                                   # a jump = the ray moved to another fold: stop the section there
            if len(brk): idx = idx[:brk[0] + 1]; P = P[:brk[0] + 1]
            s_edge[i, idx] = 0.0; ring[i] = P[0]
        okr = np.isfinite(ring[:, 0]); ring = np.stack([np.interp(np.arange(NTH), np.nonzero(okr)[0], ring[okr, k], period=NTH) for k in range(3)], 1)
        ring = gaussian_filter1d(ring, 5, axis=0, mode='wrap'); src = C.resample(ring, 0.1 * MM, closed=True)[0]
        m_ = np.isfinite(s_edge); s_edge[m_] = cKDTree(src).query(sec[m_])[0]
'''
s=s[:a]+new+s[b:]
open(p,'w').write(s)
p='tools/build_shirt.py'; s=open(p).read()
s=s.replace("[0.166, 0.040, 1.450], [0.126, 0.078, 1.445], [0.066, 0.096, 1.442], [0.0, 0.100, 1.441], [-0.066, 0.096, 1.442]])",
            "[0.160, 0.036, 1.458], [0.120, 0.066, 1.462], [0.062, 0.082, 1.462], [0.0, 0.086, 1.462], [-0.062, 0.082, 1.462]])")
open(p,'w').write(s); print('ok')
