"""run_real.py  apply the fingerprint to real engine assets (arrays extracted read-only into ../_inputs/arrays). Writes out/real_assets.json.
  python -I run_real.py"""
import os
import sys, json, time, hashlib, shutil
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "common"))
import geomfp as G, glb_min
ARR = HERE.parent / "_inputs/arrays"; OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
ext = json.load(open(ARR / "extraction_record.json")); res = dict(inputs={k: dict(path=v["path"], artifact_sha256=v["sha256_before"], unchanged_after_read=v["unchanged"]) for k, v in ext.items() if not k.startswith("_")}, fingerprints={}, comparisons={}, timing_s={})
load = lambda n: np.load(ARR / n)
names = ["camp_master__Tee", "camp_master__collar", "camp_master__Mannequin", "polo_master__Tee", "polo_master__collar", "polo_master_superseded__Tee", "ringer_master__Tee", "body_native__Human", "g0_master__M2_Baked_ThetaFit", "g0_master__G0_jacket_bracket", "g0_master__G0_pants"]
FP = {}
for n in names:
    z = load(n + ".npz"); t = time.time(); FP[n] = G.fingerprint(z["V"], z["face_sizes"], z["loops"], label=n); res["timing_s"][n] = round(time.time() - t, 2)
    again = G.fingerprint(z["V"], z["face_sizes"], z["loops"], label=n); assert again["hashes"] == FP[n]["hashes"], "non-deterministic"
    res["fingerprints"][n] = FP[n]
cmp = lambda a, b: G.compare(FP[a], FP[b])
C = res["comparisons"]
C["polo_new_vs_superseded_Tee"] = cmp("polo_master__Tee", "polo_master_superseded__Tee")
C["camp_collar_vs_polo_collar"] = cmp("camp_master__collar", "polo_master__collar")
C["camp_Mannequin_vs_body_native"] = cmp("camp_master__Mannequin", "body_native__Human")
C["g0_body_vs_body_native"] = cmp("g0_master__M2_Baked_ThetaFit", "body_native__Human")

# ---- canonical GLB (14,517 glTF vertices, triangles, Y-up) against the native body (13,380 vertices, quads, Z-up)
glb = Path(os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/human_engine/assets/glb/mannequin_full_rigged_122276_morphed.glb')))
h0 = G.artifact_sha256(glb); js, binc = glb_min.load(glb); prims = [(mi, p) for mi, m in enumerate(js["meshes"]) for p in m["primitives"]]
mi, p = max(prims, key=lambda mp: js["accessors"][mp[1]["attributes"]["POSITION"]]["count"])
Pg = glb_min.accessor(js, binc, p["attributes"]["POSITION"]).astype(np.float64); Ig = glb_min.accessor(js, binc, p["indices"]).astype(np.int64).ravel()
src = glb_min.accessor(js, binc, p["attributes"]["_SOURCE_INDEX"]).ravel() if "_SOURCE_INDEX" in p["attributes"] else None
zb = load("body_native__Human.npz"); Vb, fb, lb = zb["V"], zb["face_sizes"], zb["loops"]
fg = np.full(len(Ig) // 3, 3, np.int64)
a = G.fingerprint(Vb, fb, lb, triangulate="fan", label="body_native fan-triangulated"); b = G.fingerprint(Pg, fg, Ig, frame="gltf_yup", triangulate="fan", label="canonical GLB")
a32 = G.fingerprint(Vb.astype(np.float32), fb, lb, triangulate="fan", label="body_native float32")
C["canonical_glb_vs_body_native"] = dict(glb_artifact_sha256=h0, glb_unchanged=G.artifact_sha256(glb) == h0, glb_mesh=js["meshes"][mi].get("name"), glb_vertices=int(len(Pg)), glb_triangles=int(len(fg)),
    body_vertices=int(len(Vb)), body_faces=int(len(fb)), glb_unique_positions_exact=b["n_unique_positions_exact"], body_unique_positions_exact=a["n_unique_positions_exact"],
    compare_fan=G.compare(a, b), compare_fan_body_as_float32=G.compare(a32, b),
    compare_without_triangulation=G.compare(G.fingerprint(Vb, fb, lb), G.fingerprint(Pg, fg, Ig, frame="gltf_yup")),
    compare_frame_not_declared=G.compare(a, G.fingerprint(Pg, fg, Ig, triangulate="fan")),
    correspond_tol_1um=G.correspond(Vb, *G.fan_triangulate(fb, lb), Pg, fg, Ig, tol_m=1e-6, frame="gltf_yup"),
    source_index=None if src is None else dict(dtype=str(src.dtype), integral=bool(np.all(src == np.rint(src))), max=int(src.max()), distinct=int(len(np.unique(src))),
        max_position_error_through_index_m=float(np.abs(G.to_canonical(Pg, frame="gltf_yup") - Vb[np.rint(src).astype(int)]).max())))

# ---- file bytes vs geometry: save the same scene twice and compare both identities
try:
    import bpy
    tmp = OUT / "_save_twice"; tmp.mkdir(exist_ok=True); srcb = Path(ext["g0_master"]["path"]); cp = tmp / "copy.blend"; shutil.copyfile(srcb, cp)
    def fp_of(path, obj):
        bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False); o = bpy.data.objects[obj]; me = o.data; M = np.array(o.matrix_world)
        co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co); fs = np.empty(len(me.polygons), np.int64); me.polygons.foreach_get("loop_total", fs); lv = np.empty(len(me.loops), np.int64); me.loops.foreach_get("vertex_index", lv)
        return G.fingerprint(co.reshape(-1, 3), fs, lv, transform=M)
    f0 = fp_of(cp, "G0_pants"); bpy.ops.wm.save_as_mainfile(filepath=str(tmp / "save1.blend"), copy=True); time.sleep(1.2)
    bpy.ops.wm.open_mainfile(filepath=str(cp), load_ui=False); bpy.ops.wm.save_as_mainfile(filepath=str(tmp / "save2.blend"), copy=True)
    f1, f2 = fp_of(tmp / "save1.blend", "G0_pants"), fp_of(tmp / "save2.blend", "G0_pants")
    C["save_twice"] = dict(source_copy_sha256=G.artifact_sha256(cp), save1_sha256=G.artifact_sha256(tmp / "save1.blend"), save2_sha256=G.artifact_sha256(tmp / "save2.blend"),
        artifact_hashes_equal=G.artifact_sha256(tmp / "save1.blend") == G.artifact_sha256(tmp / "save2.blend"), geometry_verdict_save1_vs_save2=G.compare(f1, f2)["verdict"],
        geometry_verdict_source_vs_save1=G.compare(f0, f1)["verdict"], engine_original_unchanged=G.artifact_sha256(srcb) == ext["g0_master"]["sha256_before"])
    for f in tmp.glob("*.blend*"): f.unlink()
    tmp.rmdir()
except Exception as e: C["save_twice"] = dict(error=repr(e))
json.dump(res, open(OUT / "real_assets.json", "w"), indent=1)
for k, v in C.items(): print(k, "->", json.dumps({kk: vv for kk, vv in v.items() if kk not in ("hash_equal",)})[:900])
print("timing", res["timing_s"])
