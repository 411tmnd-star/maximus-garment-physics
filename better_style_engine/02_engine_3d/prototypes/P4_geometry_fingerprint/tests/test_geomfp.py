"""Deterministic tests for geomfp (pytest). Each test states the property it protects."""
import sys, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import geomfp as G

def cube():
    V = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float) * 0.1
    F = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    return V, np.full(6, 4), np.array(F).ravel()
def mixed(seed=0):                                   # irregular mesh with triangles and quads, no symmetry
    r = np.random.default_rng(seed); V = r.normal(size=(40, 3)) * 0.2; faces = [list(r.choice(40, k, replace=False)) for k in r.choice([3, 4, 5], 60)]
    return V, np.array([len(f) for f in faces]), np.concatenate(faces)
fp = G.fingerprint; H = lambda *a, **k: fp(*a, **k)["hashes"]

def test_deterministic():
    assert H(*mixed()) == H(*mixed())
def test_vertex_permutation_changes_ordered_not_surface():
    V, fs, lv = mixed(); p = np.random.default_rng(1).permutation(len(V)); inv = np.argsort(p)
    a, b = H(V, fs, lv), H(V[p], fs, inv[lv])
    assert a["ordered_exact"] != b["ordered_exact"] and a["surface_exact"] == b["surface_exact"] and a["surface_quantized"] == b["surface_quantized"]
    assert G.compare(fp(V, fs, lv), fp(V[p], fs, inv[lv]))["verdict"] == "SAME_SURFACE_EXACT"
def test_face_order_and_start_corner_invariant():
    V, fs, lv = mixed(); st = np.concatenate([[0], np.cumsum(fs)[:-1]]); faces = [list(lv[s:s + k]) for s, k in zip(st, fs)]
    r = np.random.default_rng(2); faces2 = [list(np.roll(f, r.integers(len(f)))) for f in faces]; order = r.permutation(len(faces2)); faces2 = [faces2[i] for i in order]
    b = H(V, np.array([len(f) for f in faces2]), np.concatenate(faces2)); a = H(V, fs, lv)
    assert a["surface_exact"] == b["surface_exact"] and a["ordered_exact"] != b["ordered_exact"]
def test_winding_reversal_is_different():
    V, fs, lv = cube(); st = np.concatenate([[0], np.cumsum(fs)[:-1]]); rev = np.concatenate([lv[s:s + k][::-1] for s, k in zip(st, fs)])
    assert H(V, fs, lv)["surface_exact"] != H(V, fs, rev)["surface_exact"]
def test_mirror_is_different():
    V, fs, lv = mixed(); assert H(V, fs, lv)["surface_quantized"] != H(V * [-1, 1, 1], fs, lv)["surface_quantized"]
def test_translation_is_different_unless_declared():
    V, fs, lv = cube(); M = np.eye(4); M[:3, 3] = [0.5, 0, 0]
    assert H(V, fs, lv)["surface_exact"] != H(V + [0.5, 0, 0], fs, lv)["surface_exact"]
    assert H(V + [0.5, 0, 0], fs, lv)["surface_exact"] == H(V, fs, lv, transform=M)["surface_exact"]
def test_one_micrometre_move_changes_exact_and_grid():
    V, fs, lv = cube(); W = V.copy(); W[3, 1] += 1e-6
    a, b = H(V, fs, lv), H(W, fs, lv); assert a["ordered_exact"] != b["ordered_exact"] and a["surface_quantized"] != b["surface_quantized"] and a["ordered_topology"] == b["ordered_topology"]
    assert G.compare(fp(V, fs, lv), fp(W, fs, lv))["verdict"] == "SAME_TOPOLOGY_DIFFERENT_POSITIONS"
def test_sub_quantum_noise_same_grid_hash_but_exact_differs():
    V, fs, lv = cube(); W = V + 1e-9                      # grid points are multiples of 1e-6; cube coords sit on the grid, 1e-9 does not cross a boundary
    a, b = H(V, fs, lv), H(W, fs, lv); assert a["surface_quantized"] == b["surface_quantized"] and a["surface_exact"] != b["surface_exact"]
def test_grid_boundary_straddle_is_reported_by_correspond_not_hidden():
    V, fs, lv = cube(); V = V + 0.5e-6; W = V.copy(); V[0, 0] -= 1e-9; W[0, 0] += 1e-9     # 2e-9 apart, opposite sides of a rounding boundary
    assert H(V, fs, lv)["surface_quantized"] != H(W, fs, lv)["surface_quantized"]           # documented limitation of any grid hash
    assert G.correspond(V, fs, lv, W, fs, lv, tol_m=1e-8)["verdict"] == "EQUIVALENT_WITHIN_TOL"
def test_units_and_frame_are_explicit():
    V, fs, lv = mixed(); a = H(V, fs, lv)
    assert H(V * 1000, fs, lv, units="mm")["surface_quantized"] == a["surface_quantized"]
    assert H(V * 1000, fs, lv)["surface_quantized"] != a["surface_quantized"]              # same numbers, undeclared units: NOT equal
    Vg = np.stack([V[:, 0], V[:, 2], -V[:, 1]], 1)                                          # Blender -> glTF
    assert H(Vg, fs, lv, frame="gltf_yup")["surface_exact"] == a["surface_exact"] and H(Vg, fs, lv)["surface_exact"] != a["surface_exact"]
def test_seam_split_vertices_same_surface_different_arrays():
    V, fs, lv = cube(); V2 = np.vstack([V, V[[1, 3]]]); lv2 = lv.copy(); lv2[4:8] = [4, 6, 7, 5]; lv2[np.array([20, 23])] = [8, 9]    # face 5 uses duplicated copies of verts 1 and 3
    a, b = fp(V, fs, lv), fp(V2, fs, lv2); assert a["hashes"]["surface_exact"] == b["hashes"]["surface_exact"] and a["hashes"]["ordered_exact"] != b["hashes"]["ordered_exact"]
    assert b["n_vertices"] == 10 and b["n_unique_positions_exact"] == 8
def test_unreferenced_vertices_do_not_change_surface():
    V, fs, lv = cube(); assert H(np.vstack([V, [[9, 9, 9]]]), fs, lv)["surface_exact"] == H(V, fs, lv)["surface_exact"]
def test_triangulation_not_silently_equal():
    V, fs, lv = cube(); t = G.fan_triangulate(fs, lv)
    assert H(V, fs, lv)["surface_exact"] != H(V, *t)["surface_exact"]                       # quads vs triangles: different unless requested
    assert H(V, fs, lv, triangulate="fan")["surface_exact"] == H(V, *t, triangulate="fan")["surface_exact"]
    assert G.compare(fp(V, fs, lv), fp(V, *t, triangulate="fan"))["verdict"] == "NOT_COMPARABLE"
def test_deleted_face_and_extra_face_differ():
    V, fs, lv = cube(); assert H(V, fs[:-1], lv[:-4])["surface_exact"] != H(V, fs, lv)["surface_exact"]
    assert H(V, np.append(fs, 4), np.append(lv, lv[:4]))["surface_exact"] != H(V, fs, lv)["surface_exact"]   # duplicated face counted (multiset)
def test_attributes_bound_and_order_invariant():
    V, fs, lv = mixed(); uv = np.random.default_rng(3).random((len(lv), 2)); mat = np.array(["cloth", "rib"])[np.arange(len(fs)) % 2]; nrm = np.random.default_rng(4).normal(size=V.shape)
    a = H(V, fs, lv, corner_attrs={"uv": uv}, face_attrs={"material": mat}, vertex_attrs={"normal": nrm})
    p = np.random.default_rng(5).permutation(len(V)); inv = np.argsort(p)
    b = H(V[p], fs, inv[lv], corner_attrs={"uv": uv}, face_attrs={"material": mat}, vertex_attrs={"normal": nrm[p]})
    assert all(a[k] == b[k] for k in ("attr:uv", "attr:material", "attr:normal", "surface_exact"))
    uv2 = uv.copy(); uv2[7, 0] += 0.01; mat2 = mat.copy(); mat2[3] = "rib" if mat[3] == "cloth" else "cloth"
    c = H(V, fs, lv, corner_attrs={"uv": uv2}, face_attrs={"material": mat2}, vertex_attrs={"normal": -nrm})
    assert c["attr:uv"] != a["attr:uv"] and c["attr:material"] != a["attr:material"] and c["attr:normal"] != a["attr:normal"] and c["surface_exact"] == a["surface_exact"]
def test_negative_zero_and_nan():
    V, fs, lv = cube(); W = V.copy(); W[0] = [-0.0, 0.0, -0.0]; assert H(V, fs, lv)["ordered_exact"] == H(W, fs, lv)["ordered_exact"]
    W[0, 0] = np.nan
    with pytest.raises(ValueError): fp(W, fs, lv)
def test_bad_input_rejected():
    V, fs, lv = cube()
    with pytest.raises(ValueError): fp(V, fs, lv[:-1])
    with pytest.raises(ValueError): fp(V, fs, lv + 100)
    with pytest.raises(ValueError): fp(V, fs, lv, units="inch")
def test_degenerate_face_repeated_position():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0]], float); fs = np.array([4]); a = H(V, fs, np.array([0, 1, 2, 3])); b = H(V, fs, np.array([2, 3, 0, 1]))
    assert a["surface_exact"] == b["surface_exact"]
