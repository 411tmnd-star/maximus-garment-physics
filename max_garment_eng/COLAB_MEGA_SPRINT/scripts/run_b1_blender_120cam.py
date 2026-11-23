"""COLAB_MEGA_SPRINT/scripts/run_b1_blender_120cam.py — Headless Blender 5.2 OptiX 120-Cam Renderer + Cam-0 IoU Gate.

1. Mounts c0_ringer_solved.npz onto maximus_v2_master_psd.blend (pure bpy.data, zero bpy.ops mesh ops).
2. Renders Camera 0 FIRST (1000x1000, 64 spp OIDN, 16-bit sRGB RGBA PNG, Khronos PBR Neutral).
3. Projects solved garment + bare body through transforms_cv.json[0] and enforces Cam-0 silhouette IoU >= 0.98
   via cam0_silhouette_iou_check() (aborts Stage 3+ cleanly with cam0_iou_diag.png if < 0.98).
4. Renders all 120 views and archives into b1_120cam_dataset.tar.
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


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser(description="Stage B1 Blender 5.2 OptiX 120-Cam Renderer")
    p.add_argument("--solved-npz", type=Path, required=True)
    p.add_argument("--hero-npz", type=Path, default=Path("inputs/hero_ringer_shell.npz"))
    p.add_argument("--theta-fit-json", type=Path, default=Path("inputs/m_theta_fit.json"))
    p.add_argument("--ringer-blend", type=Path, required=True)
    p.add_argument("--transforms-json", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cycles-device", choices=["OPTIX", "CUDA"], default="OPTIX")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--iou-threshold", type=float, default=0.98)
    return p.parse_args(argv)


def _rasterize_mesh_silhouette(
    V_list: list[np.ndarray],
    F_list: list[np.ndarray],
    c2w_cv: np.ndarray,
    fl_x: float,
    fl_y: float,
    cx: float,
    cy: float,
    w: int,
    h: int,
) -> np.ndarray:
    """Project 3D meshes through OpenCV c2w matrix and rasterize binary silhouette."""
    from PIL import Image, ImageDraw  # type: ignore

    w2c = np.linalg.inv(np.asarray(c2w_cv, dtype=np.float64))
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for V, F in zip(V_list, F_list):
        Xc = (V @ R.T) + t[None, :]
        z = Xc[:, 2]
        valid = z > 1e-4
        u = np.where(valid, fl_x * (Xc[:, 0] / np.maximum(z, 1e-4)) + cx, -1e6)
        v = np.where(valid, fl_y * (Xc[:, 1] / np.maximum(z, 1e-4)) + cy, -1e6)
        for tri in F:
            if valid[tri[0]] and valid[tri[1]] and valid[tri[2]]:
                pts = [(float(u[tri[0]]), float(v[tri[0]])), (float(u[tri[1]]), float(v[tri[1]])), (float(u[tri[2]]), float(v[tri[2]]))]
                draw.polygon(pts, fill=255)
    return (np.asarray(mask, dtype=np.float32) / 255.0)


def _setup_optix_16bit(scene: Any, device_pref: str, samples: int, w: int, h: int, masks_dir: Path | None = None) -> tuple[str, Any]:
    import bpy  # type: ignore

    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(samples)
    scene.cycles.use_denoising = True
    if hasattr(scene.cycles, "denoiser"):
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.render.resolution_x = int(w)
    scene.render.resolution_y = int(h)
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "16"
    scene.view_settings.view_transform = "Khronos PBR Neutral"

    # Enable Object-ID pass (body=0, top=1, bottom=2, footwear=3) for garment-only gsplat supervision (Blender 5.2 Compositor)
    mask_file_node = None
    for vl in scene.view_layers:
        vl.use_pass_object_index = True
    if masks_dir is not None:
        masks_dir.mkdir(parents=True, exist_ok=True)
        ng = bpy.data.node_groups.new("CompObjectID", "CompositorNodeTree")
        scene.compositing_node_group = ng
        ng.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        rl = ng.nodes.new("CompositorNodeRLayers")
        out_node = ng.nodes.new("NodeGroupOutput")
        ng.links.new(rl.outputs["Image"], out_node.inputs["Image"])
        math_div = ng.nodes.new("ShaderNodeMath")
        math_div.operation = "DIVIDE"
        math_div.inputs[1].default_value = 255.0  # encodes ID 0,1,2,3 cleanly into 8-bit Raw PNG pixel intensity ID/255
        mask_file_node = ng.nodes.new("CompositorNodeOutputFile")
        mask_file_node.directory = str(masks_dir)
        mask_file_node.file_name = ""
        mask_file_node.format.media_type = "IMAGE"
        mask_file_node.format.file_format = "PNG"
        mask_file_node.format.color_mode = "BW"
        mask_file_node.format.color_depth = "8"
        mask_file_node.format.color_management = "OVERRIDE"
        mask_file_node.format.view_settings.view_transform = "Raw"
        mask_file_node.file_output_items.clear()
        mask_file_node.file_output_items.new("FLOAT", "cam_part_id_####")
        ng.links.new(rl.outputs["Object Index"], math_div.inputs[0])
        ng.links.new(math_div.outputs[0], mask_file_node.inputs[0])

    prefs = bpy.context.preferences.addons["cycles"].preferences
    chosen = "CPU"
    for dev_type in ([device_pref, "CUDA"] if device_pref == "OPTIX" else ["CUDA"]):
        try:
            prefs.compute_device_type = dev_type
            prefs.get_devices()
            gpu_devs = [d for d in prefs.devices if d.type == dev_type]
            if gpu_devs:
                for d in prefs.devices:
                    d.use = d.type == dev_type
                scene.cycles.device = "GPU"
                chosen = dev_type
                break
        except Exception:
            continue
    return chosen, mask_file_node


def main() -> None:
    import bpy  # type: ignore
    from mathutils import Matrix  # type: ignore
    from PIL import Image  # type: ignore

    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from gsplat_known_camera_runner import cam0_silhouette_iou_check  # type: ignore
    from run_c0_optix_lookdev_and_reel import _ensure_studio_rig, _mount_ringer_onto_master  # type: ignore

    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = args.out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.solved_npz, allow_pickle=True)
    V_solved = d["V_solved"].astype(np.float64)
    F_garment = d["F"].astype(np.int32)

    body_obj, g_obj, pose_info = _mount_ringer_onto_master(
        args.solved_npz, args.ringer_blend, args.hero_npz, args.theta_fit_json
    )
    dg = bpy.context.evaluated_depsgraph_get()
    ev_me = body_obj.evaluated_get(dg).data
    ev_me.calc_loop_triangles()
    coords = np.empty(len(ev_me.vertices) * 3, dtype=np.float64)
    ev_me.vertices.foreach_get("co", coords)
    V_body = (
        np.asarray(body_obj.matrix_world, dtype=np.float64)[:3, :3] @ coords.reshape(-1, 3).T
    ).T + np.asarray(body_obj.matrix_world, dtype=np.float64)[:3, 3]
    F_body = np.array([lt.vertices[:] for lt in ev_me.loop_triangles], dtype=np.int32)
    assert len(F_body) > 0, "Expected non-empty triangulated body faces"

    body_obj.pass_index = 0  # body = 0
    g_obj.pass_index = int(g_obj.get("part_id", 1))  # top = 1, bottom = 2, footwear = 3
    for o in bpy.data.objects:
        if o.type == "MESH" and o not in (body_obj, g_obj):
            if "part_id" in o:
                o.pass_index = int(o["part_id"])
            elif "trouser" in o.name.lower() or "pant" in o.name.lower():
                o.pass_index = 2
            elif "shoe" in o.name.lower() or "foot" in o.name.lower():
                o.pass_index = 3

    tf = json.loads(args.transforms_json.read_text(encoding="utf-8"))
    w, h = int(tf["w"]), int(tf["h"])
    cx, cy = float(tf["cx"]), float(tf["cy"])
    scene = bpy.context.scene
    masks_dir = args.out_dir / "masks"
    dev_used, mask_file_node = _setup_optix_16bit(scene, args.cycles_device, args.samples, w, h, masks_dir=masks_dir)
    scene.view_settings.exposure = -0.34
    cam_obj = _ensure_studio_rig(scene)
    cam_data = cam_obj.data
    cam_data.sensor_width = 36.0

    frames = tf["frames"]
    # 1. Render Camera 0 FIRST & enforce IoU >= 0.98
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
        [V_solved, V_body], [F_garment, F_body],
        np.asarray(f0["transform_matrix"], dtype=np.float64),
        float(f0["fl_x"]), float(f0["fl_y"]), cx, cy, w, h,
    )
    iou_res = cam0_silhouette_iou_check(
        alpha_0, mesh_sil, out_diag_png=args.out_dir / "cam0_iou_diag.png", iou_threshold=args.iou_threshold
    )
    print("Cam-0 IoU gate passed:", json.dumps(iou_res), flush=True)

    # 2. Render remaining frames
    for idx, fr in enumerate(frames[1:], start=1):
        scene.frame_set(idx)
        cam_data.lens = float(fr.get("focal_mm", 50.0))
        cam_obj.matrix_world = Matrix(fr["transform_matrix_bl"])
        fpath = args.out_dir / fr["file_path"]
        fpath.parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(fpath)
        bpy.ops.render.render(write_still=True)
        if idx % 20 == 0:
            print(f"[B1_OPTIX_120CAM] rendered {idx + 1}/{len(frames)}", flush=True)

    tar_path = args.out_dir / "b1_120cam_dataset.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(args.transforms_json, arcname="transforms_cv.json")
        tar.add(img_dir, arcname="images")
        if masks_dir.exists():
            tar.add(masks_dir, arcname="masks")

    solved_sha = hashlib.sha256(args.solved_npz.read_bytes()).hexdigest() if args.solved_npz.exists() else ""
    rep = {
        "cycles_device_used": dev_used,
        "object_id_pass_map": {"body": 0, "top": 1, "bottom": 2, "footwear": 3},
        "solved_npz_sha256": solved_sha,
        "theta_fit_pose": pose_info,
        "n_rendered": len(frames),
        "cam0_iou_gate": iou_res,
        "dataset_tar": str(tar_path),
        "dataset_tar_mb": round(tar_path.stat().st_size / (1024.0 * 1024.0), 2),
    }
    (args.out_dir / "B1_120CAM_REPORT.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
