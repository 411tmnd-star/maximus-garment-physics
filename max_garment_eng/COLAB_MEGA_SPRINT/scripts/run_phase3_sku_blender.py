"""run_phase3_sku_blender.py — Universal Blender 5.2 Extract, GLB Export, Hero Still & 120-Cam OptiX + Object-ID Runner.

Supports ALL 8 live Stage 1A SKU masters (single/multi-part .blend + optional canonical base trousers/footwear).
Pure bpy.data mounting/extraction (only bpy.ops.export_scene.gltf and bpy.ops.render.render used).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from gsplat_known_camera_runner import cam0_silhouette_iou_check  # type: ignore
from run_b1_blender_120cam import _rasterize_mesh_silhouette, _setup_optix_16bit  # type: ignore
from run_c0_optix_lookdev_and_reel import (  # type: ignore
    _apply_theta_fit_pose,
    _ensure_m1_obsidian_on_body,
    _ensure_studio_rig,
    _setup_cycles,
)


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser(description="Phase 3 Universal Blender 5.2 Extract & 120-Cam OptiX Runner")
    p.add_argument("--mode", choices=["extract", "render_and_glb"], required=True)
    p.add_argument("--sku", required=True)
    p.add_argument("--sku-blend", type=Path, required=True)
    p.add_argument("--base-trouser-blend", type=Path, default=None)
    p.add_argument("--base-shoe-blend", type=Path, default=None)
    p.add_argument("--dress-base-bottoms", action="store_true")
    p.add_argument("--include-base-footwear", action="store_true")
    p.add_argument("--hero-npz", type=Path, default=Path("inputs/hero_ringer_shell.npz"))
    p.add_argument("--theta-fit-json", type=Path, default=Path("inputs/m_theta_fit.json"))
    p.add_argument("--extract-npz", type=Path, default=None)
    p.add_argument("--solved-npz", type=Path, default=None)
    p.add_argument("--transforms-json", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--raw-glb", type=Path, default=None)
    p.add_argument("--cycles-device", choices=["OPTIX", "CUDA"], default="OPTIX")
    p.add_argument("--samples", type=int, default=48)
    p.add_argument("--iou-threshold", type=float, default=0.98)
    return p.parse_args(argv)


def _append_meshes(blend_path: Path, force_part_id: int | None = None) -> list[Any]:
    import bpy  # type: ignore

    abs_blend = Path(blend_path).resolve()
    scene_col = bpy.context.scene.collection
    with bpy.data.libraries.load(str(abs_blend), link=False) as (data_from, data_to):
        data_to.objects = list(data_from.objects)
    meshes = []
    for obj in data_to.objects:
        if obj is None:
            continue
        if obj.type != "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)
            continue
        if obj.name not in scene_col.objects:
            scene_col.objects.link(obj)
        mw = np.asarray(obj.matrix_world, dtype=np.float64)
        if not np.allclose(mw, np.eye(4), atol=1e-6):
            obj.data.transform(obj.matrix_world.copy())
            obj.matrix_world.identity()
        if force_part_id is not None:
            obj["part_id"] = int(force_part_id)
        meshes.append(obj)
    return meshes


def _infer_part_id(obj: Any, default_pid: int = 1) -> int:
    if "part_id" in obj:
        return int(obj["part_id"])
    nl = obj.name.lower()
    if "trouser" in nl or "pant" in nl:
        return 2
    if "shoe" in nl or "foot" in nl:
        return 3
    return default_pid


def _mount_head_to_toe(args: argparse.Namespace) -> tuple[Any, list[Any], list[Any], dict[str, Any]]:
    import bpy  # type: ignore

    body_obj = next((o for o in bpy.data.objects if o.type == "MESH" and len(o.data.vertices) == 13380), None)
    assert body_obj is not None, "Expected 13,380-vert Maximus body mesh in master_psd.blend"
    pose_info = _apply_theta_fit_pose(args.theta_fit_json, body_obj, args.hero_npz)
    _ensure_m1_obsidian_on_body(body_obj)
    body_obj.pass_index = 0

    sku_objs = _append_meshes(args.sku_blend)
    bottom_objs: list[Any] = []
    shoe_objs: list[Any] = []
    if args.dress_base_bottoms and args.base_trouser_blend and args.base_trouser_blend.is_file():
        bottom_objs = _append_meshes(args.base_trouser_blend, force_part_id=2)
    if args.include_base_footwear and args.base_shoe_blend and args.base_shoe_blend.is_file():
        shoe_objs = _append_meshes(args.base_shoe_blend, force_part_id=3)

    cloth_objs: list[Any] = []
    static_objs: list[Any] = []
    for o in sku_objs + bottom_objs + shoe_objs:
        pid = _infer_part_id(o, 1)
        o.pass_index = pid
        for poly in o.data.polygons:
            poly.use_smooth = True
        nl = o.name.lower()
        is_hardware_or_shoe = (pid == 3) or ("button" in nl) or ("snap" in nl) or ("shoe" in nl)
        if is_hardware_or_shoe:
            static_objs.append(o)
        else:
            cloth_objs.append(o)

    cloth_objs.sort(key=lambda o: o.name)
    static_objs.sort(key=lambda o: o.name)
    return body_obj, cloth_objs, static_objs, pose_info


def _collect_group_geometry(objs: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    V_parts, F_parts, fpid_parts, col_parts, ranges = [], [], [], [], []
    v_off = 0
    for o in objs:
        me = o.data
        nv = len(me.vertices)
        coords = np.empty(nv * 3, dtype=np.float32)
        me.vertices.foreach_get("co", coords)
        V_i = coords.reshape(-1, 3)
        tris = []
        for p in me.polygons:
            vs = p.vertices
            if len(vs) == 3:
                tris.append((vs[0] + v_off, vs[1] + v_off, vs[2] + v_off))
            elif len(vs) > 3:
                for k in range(1, len(vs) - 1):
                    tris.append((vs[0] + v_off, vs[k] + v_off, vs[k + 1] + v_off))
        F_i = np.array(tris, dtype=np.int32) if tris else np.zeros((0, 3), dtype=np.int32)
        pid = _infer_part_id(o, 1)
        if pid == 1:
            col_i = (V_i[:, 2] >= 1.43) & (np.hypot(V_i[:, 0], V_i[:, 1] + 0.02) <= 0.12)
        else:
            col_i = np.zeros(nv, dtype=bool)
        V_parts.append(V_i)
        F_parts.append(F_i)
        fpid_parts.append(np.full(len(F_i), pid, dtype=np.uint16))
        col_parts.append(col_i)
        ranges.append({"name": o.name, "start": v_off, "end": v_off + nv, "part_id": pid})
        v_off += nv

    V_all = np.vstack(V_parts).astype(np.float32) if V_parts else np.zeros((0, 3), dtype=np.float32)
    F_all = np.vstack(F_parts).astype(np.int32) if F_parts else np.zeros((0, 3), dtype=np.int32)
    fpid_all = np.concatenate(fpid_parts).astype(np.uint16) if fpid_parts else np.zeros((0,), dtype=np.uint16)
    col_all = np.concatenate(col_parts).astype(bool) if col_parts else np.zeros((0,), dtype=bool)
    return V_all, F_all, fpid_all, col_all, ranges


def _tri_edges(F: np.ndarray) -> np.ndarray:
    if len(F) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    e0 = F[:, [0, 1]]
    e1 = F[:, [1, 2]]
    e2 = F[:, [2, 0]]
    all_e = np.vstack([e0, e1, e2])
    all_e = np.sort(all_e, axis=1)
    return np.unique(all_e, axis=0).astype(np.int32)


def run_extract(args: argparse.Namespace) -> None:
    import bpy  # type: ignore

    body_obj, cloth_objs, static_objs, pose_info = _mount_head_to_toe(args)
    dg = bpy.context.evaluated_depsgraph_get()
    ev_me = body_obj.evaluated_get(dg).data
    ev_me.calc_loop_triangles()
    coords = np.empty(len(ev_me.vertices) * 3, dtype=np.float64)
    ev_me.vertices.foreach_get("co", coords)
    mw = np.asarray(body_obj.matrix_world, dtype=np.float64)
    V_body = (mw[:3, :3] @ coords.reshape(-1, 3).T).T + mw[:3, 3]
    F_body = np.array([lt.vertices[:] for lt in ev_me.loop_triangles], dtype=np.int32)

    V_fit, F, fpid, collar_mask, cloth_ranges = _collect_group_geometry(cloth_objs)
    edges = _tri_edges(F)
    sV, sF, sfpid, _, static_ranges = _collect_group_geometry(static_objs)

    out_npz = args.extract_npz or (args.out_dir / f"{args.sku}_extract.npz")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        sku=np.array(args.sku),
        V_fit=V_fit,
        F=F,
        edges=edges,
        collar_mask=collar_mask,
        face_part_ids=fpid,
        obj_vert_ranges_json=np.array(json.dumps(cloth_ranges)),
        body_ref_V=V_body.astype(np.float32),
        body_F=F_body.astype(np.int32),
        static_V=sV,
        static_F=sF,
        static_face_part_ids=sfpid,
        static_ranges_json=np.array(json.dumps(static_ranges)),
    )
    print(json.dumps({
        "mode": "extract",
        "sku": args.sku,
        "out_npz": str(out_npz),
        "cloth_verts": int(len(V_fit)),
        "cloth_faces": int(len(F)),
        "static_verts": int(len(sV)),
        "static_faces": int(len(sF)),
        "pose_info": pose_info,
    }, indent=2))


def run_render_and_glb(args: argparse.Namespace) -> None:
    import bpy  # type: ignore
    from mathutils import Matrix, Vector  # type: ignore
    from PIL import Image  # type: ignore

    assert args.solved_npz and args.solved_npz.is_file(), "Missing --solved-npz"
    assert args.extract_npz and args.extract_npz.is_file(), "Missing --extract-npz"
    assert args.transforms_json and args.transforms_json.is_file(), "Missing --transforms-json"

    body_obj, cloth_objs, static_objs, pose_info = _mount_head_to_toe(args)
    sol = np.load(args.solved_npz, allow_pickle=True)
    ext = np.load(args.extract_npz, allow_pickle=True)
    V_solved = sol["V_solved"].astype(np.float32)
    cloth_ranges = json.loads(str(ext["obj_vert_ranges_json"].item()))

    by_name = {o.name: o for o in cloth_objs}
    for rng in cloth_ranges:
        o = by_name[rng["name"]]
        sl = V_solved[rng["start"] : rng["end"]]
        assert len(sl) == len(o.data.vertices), f"Vertex mismatch on {o.name}: {len(sl)} vs {len(o.data.vertices)}"
        o.data.vertices.foreach_set("co", sl.ravel())
        o.data.update()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_glb = args.raw_glb or (args.out_dir / f"maximus_v2_{args.sku}_raw.glb")
    bpy.ops.export_scene.gltf(
        filepath=str(raw_glb),
        export_format="GLB",
        export_attributes=True,
        use_selection=False,
    )

    scene = bpy.context.scene
    cam_obj = _ensure_studio_rig(scene)

    # Render crisp hero still (720x1000 studio front) for unified QA contact sheet
    _setup_cycles(scene, args.cycles_device, min(48, args.samples), 720, 1000, view_transform="AgX")
    scene.render.film_transparent = True
    scene.view_settings.exposure = -0.34
    cam_obj.data.lens = 50.0
    cam_obj.location = (0.0, -2.55, 0.96)
    direction = Vector((0.0, 0.0, 0.92)) - Vector(cam_obj.location)
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    hero_png = args.out_dir / "still_01_front.png"
    scene.render.filepath = str(hero_png)
    bpy.ops.render.render(write_still=True)

    # Composite hero portrait cleanly over subtle studio slate backdrop for QA sheet legibility
    with Image.open(hero_png).convert("RGBA") as fg:
        bg = Image.new("RGBA", fg.size, (18, 20, 24, 255))
        Image.alpha_composite(bg, fg).convert("RGB").save(hero_png)

    # Render 120-cam known-pose views + Blender 5.2 Object-ID masks (Cam-0 IoU >= 0.98 gate)
    # Hide opaque bare body ONLY from direct camera rays so multi-view 3DGS has zero arm/head occlusion contradictions
    # while still casting accurate diffuse/glossy/shadow lighting onto the garments.
    body_obj.visible_camera = False
    body_obj.visible_diffuse = True
    body_obj.visible_glossy = True
    body_obj.visible_shadow = True
    body_obj.visible_transmission = True

    F_cloth = ext["F"].astype(np.int32)
    V_static = ext["static_V"].astype(np.float64)
    F_static = ext["static_F"].astype(np.int32)
    V_sil = [V_solved.astype(np.float64)]
    F_sil = [F_cloth]
    if len(V_static) > 0 and len(F_static) > 0:
        V_sil.append(V_static)
        F_sil.append(F_static)

    tf = json.loads(args.transforms_json.read_text(encoding="utf-8"))
    w, h = int(tf["w"]), int(tf["h"])
    cx, cy = float(tf["cx"]), float(tf["cy"])
    masks_dir = args.out_dir / "masks"
    dev_used, _ = _setup_optix_16bit(scene, args.cycles_device, args.samples, w, h, masks_dir=masks_dir)
    scene.view_settings.exposure = -0.34
    cam_data = cam_obj.data
    cam_data.sensor_width = 36.0
    img_dir = args.out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    frames = tf["frames"]
    f0 = frames[0]
    scene.frame_set(0)
    cam_data.lens = float(f0.get("focal_mm", 50.0))
    cam_obj.matrix_world = Matrix(f0["transform_matrix_bl"])
    cam0_path = args.out_dir / f0["file_path"]
    cam0_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(cam0_path)
    bpy.ops.render.render(write_still=True)

    alpha_0 = np.asarray(Image.open(cam0_path).convert("RGBA"), dtype=np.float32)[..., 3] / 255.0
    mesh_sil = _rasterize_mesh_silhouette(
        V_sil, F_sil, np.asarray(f0["transform_matrix"], dtype=np.float64),
        float(f0["fl_x"]), float(f0["fl_y"]), cx, cy, w, h,
    )
    iou_res = cam0_silhouette_iou_check(
        alpha_0, mesh_sil, out_diag_png=args.out_dir / "cam0_iou_diag.png", iou_threshold=args.iou_threshold
    )
    print(f"[{args.sku}] Cam-0 IoU gate passed:", json.dumps(iou_res), flush=True)

    for idx, fr in enumerate(frames[1:], start=1):
        scene.frame_set(idx)
        cam_data.lens = float(fr.get("focal_mm", 50.0))
        cam_obj.matrix_world = Matrix(fr["transform_matrix_bl"])
        fpath = args.out_dir / fr["file_path"]
        fpath.parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(fpath)
        bpy.ops.render.render(write_still=True)
        if idx % 20 == 0:
            print(f"[{args.sku} B1] rendered {idx + 1}/{len(frames)}", flush=True)

    tar_path = args.out_dir / "b1_120cam_dataset.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(args.transforms_json, arcname="transforms_cv.json")
        tar.add(img_dir, arcname="images")
        if masks_dir.exists():
            tar.add(masks_dir, arcname="masks")

    solved_sha = hashlib.sha256(args.solved_npz.read_bytes()).hexdigest()
    rep = {
        "sku": args.sku,
        "cycles_device_used": dev_used,
        "bare_body_verts": len(body_obj.data.vertices),
        "body_hidden_in_120cam": True,
        "solved_npz_sha256": solved_sha,
        "theta_fit_pose": pose_info,
        "n_rendered": len(frames),
        "cam0_iou_gate": iou_res,
        "hero_still": str(hero_png),
        "raw_glb": str(raw_glb),
        "dataset_tar": str(tar_path),
    }
    (args.out_dir / "B1_120CAM_REPORT.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=2))


def main() -> None:
    args = _parse_args()
    if args.mode == "extract":
        run_extract(args)
    else:
        run_render_and_glb(args)


if __name__ == "__main__":
    main()
