"""extract_blend_arrays.py  READ-ONLY extraction of mesh arrays from engine .blend files into .npz (prototype inputs).
Never saves a .blend. Records sha256 of every source file before and after.
  python -I extract_blend_arrays.py <out_dir>
Arrays per object: V (n,3) float64 world coords (metres, Blender Z-up), face_sizes (m,), loops (sum sizes,), tris (k,3) loop-triangle vertex indices,
tri_poly (k,) polygon index of each triangle, plus custom ints where present."""
import os
import sys, json, hashlib
from pathlib import Path
import numpy as np, bpy

HE = Path(os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/human_engine')))
BC = HE / "src_catalog/better_style_catalog"
JOBS = {
    "camp_master": (BC / "assets/archetypes/camp_shirt_master.blend", ["Tee", "collar", "placket_overlap", "placket_buttons", "Mannequin"]),
    "polo_master": (BC / "assets/archetypes/rugby_polo_master.blend", ["Tee", "collar"]),
    "polo_master_superseded": (BC / "assets/archetypes/_superseded/rugby_polo_master.collarfix-6d798191.blend", ["Tee"]),
    "ringer_master": (BC / "assets/archetypes/ringer_tee_master.blend", ["Tee"]),
    "g0_master": (HE / "src_catalog/BETTER_STYLE_VISUAL_CURATION/14_TRACKSUIT_DUAL_STATE_VIDEO/garments/bs_track_001_g0_master.blend", ["M2_Baked_ThetaFit", "G0_jacket_bracket", "G0_pants"]),
    "body_native": (HE / "src_garment_pipeline/human_build/human_engine/mannequin_modes/03_full_rigged/out/03_full_rigged.blend", None),
}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); rec = {}
for tag, (path, names) in JOBS.items():
    if not path.exists(): rec[tag] = dict(path=str(path), missing=True); continue
    h0 = sha(path); bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False)
    if names is None:                                       # body file: the mesh object with the most vertices
        names = [max((o for o in bpy.data.objects if o.type == 'MESH'), key=lambda o: len(o.data.vertices)).name]
    objs = {}
    for n in names:
        o = bpy.data.objects.get(n)
        if o is None: objs[n] = None; continue
        me = o.data; M = np.array(o.matrix_world); nv = len(me.vertices)
        co = np.empty(nv * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3); V = co @ M[:3, :3].T + M[:3, 3]
        fs = np.empty(len(me.polygons), np.int64); me.polygons.foreach_get("loop_total", fs)
        lv = np.empty(len(me.loops), np.int64); me.loops.foreach_get("vertex_index", lv)
        me.calc_loop_triangles(); tr = np.empty(len(me.loop_triangles) * 3, np.int64); me.loop_triangles.foreach_get("vertices", tr)
        tp = np.empty(len(me.loop_triangles), np.int64); me.loop_triangles.foreach_get("polygon_index", tp)
        extra = {k: int(o[k]) for k in o.keys() if isinstance(o[k], int)}
        np.savez_compressed(out / f"{tag}__{n}.npz", V=V, face_sizes=fs, loops=lv, tris=tr.reshape(-1, 3), tri_poly=tp, matrix_world=M,
                            **{f"prop_{k}": np.array(v) for k, v in extra.items()})
        objs[n] = dict(verts=nv, polys=int(len(fs)), tris=int(len(tp)), identity_transform=bool(np.allclose(M, np.eye(4))), modifiers=len(o.modifiers), props=extra,
                       vertex_groups=[g.name for g in o.vertex_groups], uv_layers=[u.name for u in me.uv_layers], materials=[s.material.name if s.material else None for s in o.material_slots])
    h1 = sha(path); rec[tag] = dict(path=str(path), sha256_before=h0, sha256_after=h1, unchanged=h0 == h1, objects=objs)
    print(tag, "unchanged" if h0 == h1 else "CHANGED!", {k: (v or {}).get("verts") for k, v in objs.items()})
rec["_tool"] = dict(blender=bpy.app.version_string, numpy=np.__version__)
json.dump(rec, open(out / "extraction_record.json", "w"), indent=1)
