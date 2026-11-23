"""geomfp.py  PROTOTYPE canonical geometry fingerprint (pure numpy). Not production code.

Two separate identities, never mixed:
  artifact_sha256(path)      sha256 of the exact file bytes. Identifies a FILE.
  fingerprint(...)           identities derived from the geometry and explicitly declared properties. Identifies a SHAPE.

A fingerprint is a record of several named hashes. Each answers one question, so a comparison can say WHY two assets differ:

  ordered_exact        positions (float64 bits, index order) + face sizes + loop indices.       "same arrays, bit for bit"
  ordered_topology     face sizes + loop indices only.                                           "same connectivity and vertex order"
  surface_exact        order-invariant: set of faces written as cyclic tuples of POSITIONS.      "same surface, any vertex/face order, any vertex duplication"
  surface_quantized    as surface_exact with positions rounded to `quantum_m`.                   "same surface on a declared grid"
  positions_quantized  sorted unique quantized positions.                                        "same point set"
  attr:<name>          optional per-vertex / per-corner / per-face attributes, hashed bound to the quantized surface.

Canonical frame: metres, right-handed, Z-up (Blender world). Inputs declare `units` and `frame`; conversion is explicit and recorded.
What is NOT treated as equivalent (documented, deliberate): different triangulations of the same polygons (unless `triangulate='fan'` is requested and
recorded), mirrored meshes, reversed winding, different transforms (apply the transform first and say so), positions further apart than the quantum.
What IS treated as equivalent by the surface_* hashes: vertex order, face order, the starting corner of a face, duplicated (seam-split) vertices at
identical positions, unreferenced vertices.
A quantized hash can differ for two meshes that agree within the quantum (values straddling a grid boundary). Hash equality is therefore sufficient
for equivalence on the grid but not necessary for equivalence within a tolerance; `correspond()` measures the latter directly."""
import hashlib, json
from pathlib import Path
import numpy as np

SCHEMA = "betterstyle.geometry_fingerprint/0-prototype"
UNIT_SCALE = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
FRAMES = {"blender_zup": np.eye(3),                                             # canonical
          "gltf_yup": np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])}       # (x, y, z)_gltf -> (x, -z, y)_blender ; inverse of stone_matrix.blender_to_gltf


def artifact_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def to_canonical(V, units="m", frame="blender_zup", transform=None):
    """Return float64 (n,3) in metres, Z-up. `transform` (4x4, in the INPUT frame/units) is applied first when given."""
    V = np.asarray(V, np.float64)
    if V.ndim != 2 or V.shape[1] != 3: raise ValueError(f"positions must be (n,3), got {V.shape}")
    if not np.isfinite(V).all(): raise ValueError("non-finite position")
    if units not in UNIT_SCALE: raise ValueError(f"unknown units {units!r}")
    if frame not in FRAMES: raise ValueError(f"unknown frame {frame!r}")
    if transform is not None:
        M = np.asarray(transform, np.float64); V = V @ M[:3, :3].T + M[:3, 3]
    V = (V * UNIT_SCALE[units]) @ FRAMES[frame].T
    return V + 0.0                                                              # -0.0 -> +0.0 so the bit pattern is unique


def fan_triangulate(face_sizes, loops):
    """Fan from the first corner of each polygon, in loop order. Deterministic given the INPUT corner order (it is a declared preprocessing, not canonical)."""
    fs = np.asarray(face_sizes, np.int64); lv = np.asarray(loops, np.int64); start = np.concatenate([[0], np.cumsum(fs)[:-1]]); out = []
    for k in np.unique(fs):
        if k < 3: continue
        s = start[fs == k]
        for j in range(1, k - 1): out.append(np.stack([lv[s], lv[s + j], lv[s + j + 1]], 1))
    T = np.vstack(out) if out else np.zeros((0, 3), np.int64)
    return np.full(len(T), 3, np.int64), T.ravel()


def _h(*parts):
    h = hashlib.sha256()
    for p in parts: h.update(p if isinstance(p, (bytes, bytearray)) else np.ascontiguousarray(p).tobytes())
    return h.hexdigest()


def _canon_faces(ids, fs):
    """ids: per-corner integer id (rank of the corner key). Returns list of (k, rows sorted) with each row rotated to its lexicographically smallest rotation (winding kept)."""
    start = np.concatenate([[0], np.cumsum(fs)[:-1]]); out = []
    for k in np.unique(fs):
        s = start[fs == k]; F = ids[s[:, None] + np.arange(k)[None]]
        mn = F.min(1, keepdims=True); first = (F == mn).argmax(1); ties = (F == mn).sum(1) > 1
        R = F[np.arange(len(F))[:, None], (first[:, None] + np.arange(k)[None]) % k]
        for i in np.nonzero(ties)[0]:                                           # the minimum id occurs twice in one face (degenerate / repeated position): take the smallest rotation
            R[i] = min(tuple(np.roll(F[i], -r)) for r in range(k))
        R = R[np.lexsort(R.T[::-1])]; out.append((int(k), R))
    return out


def _rank(keys):
    """keys (n, c) integer rows -> (unique sorted rows, inverse rank per row)."""
    u, inv = np.unique(keys, axis=0, return_inverse=True); return u, inv.reshape(-1)


def fingerprint(V, face_sizes, loops, *, units="m", frame="blender_zup", transform=None, quantum_m=1e-6, triangulate=None,
                vertex_attrs=None, corner_attrs=None, face_attrs=None, attr_quantum=1e-6, label=None):
    """vertex_attrs / corner_attrs / face_attrs: {name: array} with first dimension = n vertices / n corners / n faces (floats are quantized by attr_quantum,
    integers and strings are used as they are). Each attribute gets its own hash, bound to the quantized surface (so it is order-invariant too)."""
    Vc = to_canonical(V, units, frame, transform); fs = np.asarray(face_sizes, np.int64).ravel(); lv = np.asarray(loops, np.int64).ravel()
    if int(fs.sum()) != len(lv): raise ValueError("sum(face_sizes) != len(loops)")
    if len(lv) and (lv.min() < 0 or lv.max() >= len(Vc)): raise ValueError("loop index out of range")
    if (fs < 3).any(): raise ValueError("face with fewer than 3 corners")
    fs0, lv0 = fs, lv
    if triangulate == "fan": fs, lv = fan_triangulate(fs, lv)
    elif triangulate is not None: raise ValueError("triangulate must be None or 'fan'")
    if face_attrs and triangulate: raise ValueError("face attributes are not carried through triangulation in this prototype")
    used = np.zeros(len(Vc), bool); used[lv] = True
    q = np.rint(Vc / quantum_m).astype(np.int64)
    bits = Vc.view(np.int64).reshape(-1, 3)                                     # exact keys: the float64 bit patterns
    rec = dict(schema=SCHEMA, label=label, canonical_frame="metres, right-handed, Z-up", input_units=units, input_frame=frame, transform_applied=transform is not None,
               triangulate=triangulate, quantum_m=quantum_m, n_vertices=int(len(Vc)), n_vertices_referenced=int(used.sum()), n_faces=int(len(fs)),
               face_size_histogram={int(k): int(c) for k, c in zip(*np.unique(fs, return_counts=True))},
               bbox_min_m=Vc[used].min(0).tolist() if used.any() else None, bbox_max_m=Vc[used].max(0).tolist() if used.any() else None, hashes={})
    H = rec["hashes"]
    H["ordered_exact"] = _h(b"oe", Vc, fs0, lv0)
    H["ordered_topology"] = _h(b"ot", np.int64(len(Vc)), fs0, lv0)
    uq, rq = _rank(q[lv]); ue, re_ = _rank(bits[lv])
    rec["n_unique_positions_quantized"] = int(len(uq)); rec["n_unique_positions_exact"] = int(len(ue))
    H["positions_quantized"] = _h(b"pq", np.float64(quantum_m), uq)
    for name, table, rk in (("surface_quantized", uq, rq), ("surface_exact", ue, re_)):
        parts = [name.encode(), table] + ([np.float64(quantum_m)] if name.endswith("quantized") else [])
        for k, R in _canon_faces(rk, fs): parts += [np.int64(k), R]
        H[name] = _h(*parts)

    def _attr_key(a, n_expected, what):
        a = np.asarray(a)
        if len(a) != n_expected: raise ValueError(f"{what} attribute has {len(a)} rows, expected {n_expected}")
        a = a.reshape(len(a), -1)
        if a.dtype.kind == "f":
            if not np.isfinite(a).all(): raise ValueError("non-finite attribute")
            return np.rint(a / attr_quantum).astype(np.int64)
        if a.dtype.kind in "iub": return a.astype(np.int64)
        u, inv = np.unique(a.astype(str), return_inverse=True)                  # strings: hash the names themselves so ids do not depend on input order
        return np.array([int(hashlib.sha256(s.encode()).hexdigest()[:15], 16) for s in u], np.int64)[inv].reshape(len(a), -1)

    for kind, attrs in (("vertex", vertex_attrs), ("corner", corner_attrs)):
        for name, a in (attrs or {}).items():
            if kind == "corner" and triangulate: raise ValueError("corner attributes are not carried through triangulation in this prototype")
            ka = _attr_key(a, len(Vc) if kind == "vertex" else len(lv), kind); ka = ka[lv] if kind == "vertex" else ka
            u, rk = _rank(np.hstack([q[lv], ka])); parts = [f"attr:{kind}:{name}".encode(), np.float64(quantum_m), np.float64(attr_quantum), u]
            for k, R in _canon_faces(rk, fs): parts += [np.int64(k), R]
            H[f"attr:{name}"] = _h(*parts)
    for name, a in (face_attrs or {}).items():
        ka = _attr_key(a, len(fs), "face"); rows = []
        for (k, R), kk in zip(_canon_faces(rq, fs), np.unique(fs)):             # bind the face value to its canonical corner tuple
            start = np.concatenate([[0], np.cumsum(fs)[:-1]])[fs == kk]; F = rq[start[:, None] + np.arange(kk)[None]]
            mn = F.min(1, keepdims=True); first = (F == mn).argmax(1); Rr = F[np.arange(len(F))[:, None], (first[:, None] + np.arange(kk)[None]) % kk]
            ties = (F == mn).sum(1) > 1
            for i in np.nonzero(ties)[0]: Rr[i] = min(tuple(np.roll(F[i], -r)) for r in range(kk))
            X = np.hstack([Rr, ka[fs == kk]]); rows += [np.int64(kk), X[np.lexsort(X.T[::-1])]]
        H[f"attr:{name}"] = _h(f"attr:face:{name}".encode(), np.float64(quantum_m), uq, *rows)
    return rec


VERDICTS = ("IDENTICAL_ARRAYS", "SAME_SURFACE_EXACT", "SAME_SURFACE_ON_GRID", "SAME_TOPOLOGY_DIFFERENT_POSITIONS", "SAME_POINT_SET_DIFFERENT_FACES", "DIFFERENT")


def compare(a, b):
    """Verdict from two fingerprint records, strongest first. Refuses records made with different declared settings."""
    for k in ("quantum_m", "triangulate", "schema"):
        if a[k] != b[k]: return dict(verdict="NOT_COMPARABLE", reason=f"{k}: {a[k]!r} vs {b[k]!r}")
    ha, hb = a["hashes"], b["hashes"]; eq = {k: ha[k] == hb[k] for k in ha if k in hb}
    if eq["ordered_exact"]: v = "IDENTICAL_ARRAYS"
    elif eq["surface_exact"]: v = "SAME_SURFACE_EXACT"
    elif eq["surface_quantized"]: v = "SAME_SURFACE_ON_GRID"
    elif eq["ordered_topology"]: v = "SAME_TOPOLOGY_DIFFERENT_POSITIONS"
    elif eq["positions_quantized"]: v = "SAME_POINT_SET_DIFFERENT_FACES"
    else: v = "DIFFERENT"
    attrs = {k: e for k, e in eq.items() if k.startswith("attr:")}
    only = sorted(set(k for k in ha if k.startswith("attr:")) ^ set(k for k in hb if k.startswith("attr:")))
    return dict(verdict=v, hash_equal=eq, attributes_equal=attrs, attributes_only_in_one=only,
                note=None if v != "DIFFERENT" else "hashes differ; use correspond() to measure whether the shapes agree within a tolerance")


def correspond(Va, fa, la, Vb, fb, lb, tol_m=1e-6, **canon_b):
    """Tolerance comparison that does not depend on a grid. Welds each mesh's coincident positions, matches unique positions one-to-one within tol_m (KD-tree),
    then compares the face sets through the match. Inputs must already be canonical for A; `canon_b` (units, frame, transform) is applied to B."""
    from scipy.spatial import cKDTree
    A = to_canonical(Va); B = to_canonical(Vb, **canon_b); fa = np.asarray(fa, np.int64); la = np.asarray(la, np.int64); fb = np.asarray(fb, np.int64); lb = np.asarray(lb, np.int64)
    ua, ia = np.unique(A[np.unique(la)], axis=0, return_inverse=True); ub, ib = np.unique(B[np.unique(lb)], axis=0, return_inverse=True)
    ta = cKDTree(ua); d, j = ta.query(ub); res = dict(tol_m=tol_m, unique_positions=(int(len(ua)), int(len(ub))), max_nearest_distance_m=float(d.max()) if len(d) else 0.0)
    one_to_one = len(ua) == len(ub) and bool((d <= tol_m).all()) and len(np.unique(j)) == len(ub)
    res["positions_match_one_to_one"] = one_to_one
    if not one_to_one: res["verdict"] = "DIFFERENT_WITHIN_TOL"; return res
    mapa = np.full(len(A), -1, np.int64); mapa[np.unique(la)] = ia.reshape(-1); mapb = np.full(len(B), -1, np.int64); mapb[np.unique(lb)] = j[ib.reshape(-1)]
    key = lambda rk, fs: _h(*[x for k, R in _canon_faces(rk, fs) for x in (np.int64(k), R)])
    same = key(mapa[la], fa) == key(mapb[lb], fb); res["face_sets_equal"] = bool(same); res["verdict"] = "EQUIVALENT_WITHIN_TOL" if same else "SAME_POINTS_DIFFERENT_FACES_WITHIN_TOL"
    return res


def fingerprint_npz(path, **kw):
    z = np.load(path); return fingerprint(z["V"], z["face_sizes"], z["loops"], label=Path(path).name, **kw)


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]: print(json.dumps(fingerprint_npz(p), indent=1))
