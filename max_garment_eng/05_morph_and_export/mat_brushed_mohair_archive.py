"""05_morph_and_export/mat_brushed_mohair_archive.py — Reusable capsule shader preset archive.

Preserves the Cycles brushed-mohair / heavyweight knit shader node tree used during the BS-RINGER-001
lookdev pass (`MAT_BRUSHED_MOHAIR_ARCHIVE`) as an asset-marked Blender material datablock (.blend)
plus machine-readable JSON metadata, leaving the active heavyweight knitted jersey GLB untouched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
OUT_BLEND = HERE / "MAT_BRUSHED_MOHAIR_ARCHIVE.blend"
OUT_JSON = HERE / "MAT_BRUSHED_MOHAIR_ARCHIVE.json"

PRESET_NAME = "MAT_BRUSHED_MOHAIR_ARCHIVE"
PRESET_PARAMS: dict[str, Any] = {
    "body_hex": "#E6E1D9",
    "rib_hex": "#181716",
    "body_weave_scale": 115.0,
    "rib_weave_scale": 175.0,
    "wave_x_factor": 0.85,
    "wave_y_factor": 0.65,
    "wave_distortion": 0.09,
    "noise_scale_factor": 1.85,
    "noise_weight": 0.34,
    "mottle_scale": 6.5,
    "bump_strength": 0.50,
    "bump_distance": 0.00045,
    "roughness": 0.90,
    "specular_ior_level": 0.10,
    "rib_sdf_range": [0.44, 0.56],
    "coverstitch_groove": [-0.35, 0.045],
    "webgl_normal_scale": 1.7,
    "webgl_micro_weave": [0.052, 0.038],
    "webgl_fresnel": [3.1, 0.18],
    "reference_render": "out/knitted_ringer_cycles_verify.png",
}


def srgb_to_lin(c: list[float]) -> tuple[float, float, float]:
    out: list[float] = []
    for ch in c:
        out.append(ch / 12.92 if ch <= 0.04045 else ((ch + 0.055) / 1.055) ** 2.4)
    return (float(out[0]), float(out[1]), float(out[2]))


def hex_to_lin(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return srgb_to_lin([int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)])


def build_mat_brushed_mohair_archive(
    bpy,
    mat=None,
    *,
    body_hex: str = PRESET_PARAMS["body_hex"],
    rib_hex: str = PRESET_PARAMS["rib_hex"],
    uv_fabric: str = "UVMap_Fabric",
    sdf_attr: str = "_RIB_SDF",
    front_albedo_path: str | Path | None = None,
    front_height_path: str | Path | None = None,
    uv_art_front: str = "UVMap_Art_Front",
):
    """Construct the MAT_BRUSHED_MOHAIR_ARCHIVE Cycles node tree onto a Blender material."""
    if mat is None:
        mat = bpy.data.materials.new(PRESET_NAME)
    mat.name = PRESET_NAME
    mat.use_nodes = True
    nt = mat.node_tree
    N = nt.nodes
    Lk = nt.links
    N.clear()

    out = N.new("ShaderNodeOutputMaterial")
    out.name = "ArchiveMaterialOutput"
    b = N.new("ShaderNodeBsdfPrincipled")
    b.name = "ArchivePrincipledBSDF"
    Lk.new(b.outputs["BSDF"], out.inputs["Surface"])

    attr_sdf = N.new("ShaderNodeAttribute")
    attr_sdf.name = "RibSDF"
    attr_sdf.attribute_name = sdf_attr
    rib_mr = N.new("ShaderNodeMapRange")
    rib_mr.name = "RibSmoothstep"
    rib_mr.interpolation_type = "SMOOTHSTEP"
    rib_mr.clamp = True
    rib_mr.inputs["From Min"].default_value = PRESET_PARAMS["rib_sdf_range"][0]
    rib_mr.inputs["From Max"].default_value = PRESET_PARAMS["rib_sdf_range"][1]
    rib_mr.inputs["To Min"].default_value = 0.0
    rib_mr.inputs["To Max"].default_value = 1.0
    Lk.new(attr_sdf.outputs["Fac"], rib_mr.inputs["Value"])
    rib_fac = rib_mr.outputs["Result"]

    uv = N.new("ShaderNodeUVMap")
    uv.name = "FabricUV"
    uv.uv_map = uv_fabric
    mp = N.new("ShaderNodeMapping")
    mp.name = "FabricMapping"
    Lk.new(uv.outputs["UV"], mp.inputs["Vector"])

    def make_weave(scale: float, prefix: str):
        w1 = N.new("ShaderNodeTexWave")
        w1.name = f"{prefix}WaveX"
        w1.wave_type = "BANDS"
        w1.bands_direction = "X"
        w1.inputs["Scale"].default_value = scale * PRESET_PARAMS["wave_x_factor"]
        w1.inputs["Distortion"].default_value = PRESET_PARAMS["wave_distortion"]
        w1.inputs["Detail"].default_value = 1.5
        w2 = N.new("ShaderNodeTexWave")
        w2.name = f"{prefix}WaveY"
        w2.wave_type = "BANDS"
        w2.bands_direction = "Y"
        w2.inputs["Scale"].default_value = scale * PRESET_PARAMS["wave_y_factor"]
        w2.inputs["Distortion"].default_value = PRESET_PARAMS["wave_distortion"]
        nz = N.new("ShaderNodeTexNoise")
        nz.name = f"{prefix}MicroFuzz"
        nz.inputs["Scale"].default_value = scale * PRESET_PARAMS["noise_scale_factor"]
        nz.inputs["Detail"].default_value = 1.6
        for t in (w1, w2, nz):
            Lk.new(mp.outputs["Vector"], t.inputs["Vector"])
        mul = N.new("ShaderNodeMath")
        mul.name = f"{prefix}CourseMul"
        mul.operation = "MULTIPLY"
        Lk.new(w1.outputs["Fac"], mul.inputs[0])
        Lk.new(w2.outputs["Fac"], mul.inputs[1])
        ad = N.new("ShaderNodeMath")
        ad.name = f"{prefix}WeaveAdd"
        ad.operation = "ADD"
        m2 = N.new("ShaderNodeMath")
        m2.name = f"{prefix}FuzzScale"
        m2.operation = "MULTIPLY"
        m2.inputs[1].default_value = PRESET_PARAMS["noise_weight"]
        Lk.new(mul.outputs["Value"], ad.inputs[0])
        Lk.new(nz.outputs["Fac"], m2.inputs[0])
        Lk.new(m2.outputs["Value"], ad.inputs[1])
        return ad.outputs["Value"]

    inv_rib = N.new("ShaderNodeMath")
    inv_rib.name = "InvRibFac"
    inv_rib.operation = "SUBTRACT"
    inv_rib.inputs[0].default_value = 1.0
    Lk.new(rib_fac, inv_rib.inputs[1])

    body_weave = make_weave(PRESET_PARAMS["body_weave_scale"], "Body")
    rib_weave = make_weave(PRESET_PARAMS["rib_weave_scale"], "Rib")
    bw_m = N.new("ShaderNodeMath")
    bw_m.name = "BodyWeaveMask"
    bw_m.operation = "MULTIPLY"
    Lk.new(body_weave, bw_m.inputs[0])
    Lk.new(inv_rib.outputs["Value"], bw_m.inputs[1])
    rw_m = N.new("ShaderNodeMath")
    rw_m.name = "RibWeaveMask"
    rw_m.operation = "MULTIPLY"
    Lk.new(rib_weave, rw_m.inputs[0])
    Lk.new(rib_fac, rw_m.inputs[1])
    weave_sum = N.new("ShaderNodeMath")
    weave_sum.name = "UnifiedWeave"
    weave_sum.operation = "ADD"
    Lk.new(bw_m.outputs["Value"], weave_sum.inputs[0])
    Lk.new(rw_m.outputs["Value"], weave_sum.inputs[1])
    weave = weave_sum.outputs["Value"]

    mott = N.new("ShaderNodeTexNoise")
    mott.name = "YarnMottle"
    mott.inputs["Scale"].default_value = PRESET_PARAMS["mottle_scale"]
    mott.inputs["Detail"].default_value = 2.0
    Lk.new(mp.outputs["Vector"], mott.inputs["Vector"])

    mc = N.new("ShaderNodeMath")
    mc.name = "WeaveTone"
    mc.operation = "MULTIPLY_ADD"
    mc.inputs[1].default_value = 0.13
    mc.inputs[2].default_value = 0.88
    Lk.new(weave, mc.inputs[0])
    mott_m = N.new("ShaderNodeMath")
    mott_m.name = "MottleTone"
    mott_m.operation = "MULTIPLY_ADD"
    mott_m.inputs[1].default_value = 0.045
    mott_m.inputs[2].default_value = 0.975
    Lk.new(mott.outputs["Fac"], mott_m.inputs[0])
    wm = N.new("ShaderNodeMath")
    wm.name = "WoolModulation"
    wm.operation = "MULTIPLY"
    Lk.new(mc.outputs["Value"], wm.inputs[0])
    Lk.new(mott_m.outputs["Value"], wm.inputs[1])

    body_rgb = N.new("ShaderNodeRGB")
    body_rgb.name = "BodyAlbedoRGB"
    body_rgb.outputs[0].default_value = (*hex_to_lin(body_hex), 1.0)
    rib_rgb = N.new("ShaderNodeRGB")
    rib_rgb.name = "RibAlbedoRGB"
    rib_rgb.outputs[0].default_value = (*hex_to_lin(rib_hex), 1.0)

    wv = N.new("ShaderNodeCombineColor")
    wv.name = "WeaveModRGB"
    for idx in range(3):
        Lk.new(wm.outputs["Value"], wv.inputs[idx])

    body_base_sock = body_rgb.outputs[0]
    ink_alpha = None
    ink_height = None
    if front_albedo_path:
        uva = N.new("ShaderNodeUVMap")
        uva.name = "FrontArtUV"
        uva.uv_map = uv_art_front
        im = N.new("ShaderNodeTexImage")
        im.name = "FrontArtAlbedo"
        im.image = bpy.data.images.load(str(front_albedo_path), check_existing=True)
        im.extension = "CLIP"
        im.interpolation = "Cubic"
        Lk.new(uva.outputs["UV"], im.inputs["Vector"])

        geom = N.new("ShaderNodeNewGeometry")
        sep_n = N.new("ShaderNodeSeparateXYZ")
        Lk.new(geom.outputs["Normal"], sep_n.inputs["Vector"])
        fmask = N.new("ShaderNodeMapRange")
        fmask.clamp = True
        fmask.inputs["From Min"].default_value = -0.15
        fmask.inputs["From Max"].default_value = -0.35
        fmask.inputs["To Min"].default_value = 0.0
        fmask.inputs["To Max"].default_value = 1.0
        Lk.new(sep_n.outputs["Y"], fmask.inputs["Value"])

        front_body_mask = N.new("ShaderNodeMath")
        front_body_mask.operation = "MULTIPLY"
        Lk.new(fmask.outputs["Result"], front_body_mask.inputs[0])
        Lk.new(inv_rib.outputs["Value"], front_body_mask.inputs[1])

        masked_a = N.new("ShaderNodeMath")
        masked_a.operation = "MULTIPLY"
        Lk.new(im.outputs["Alpha"], masked_a.inputs[0])
        Lk.new(front_body_mask.outputs["Value"], masked_a.inputs[1])
        ink_alpha = masked_a.outputs["Value"]

        mix_ink = N.new("ShaderNodeMixRGB")
        mix_ink.name = "FrontArtMix"
        Lk.new(body_base_sock, mix_ink.inputs["Color1"])
        Lk.new(im.outputs["Color"], mix_ink.inputs["Color2"])
        Lk.new(ink_alpha, mix_ink.inputs["Fac"])
        body_base_sock = mix_ink.outputs["Color"]

        if front_height_path:
            hi = N.new("ShaderNodeTexImage")
            hi.name = "FrontArtHeight"
            hi.image = bpy.data.images.load(str(front_height_path), check_existing=True)
            hi.image.colorspace_settings.name = "Non-Color"
            hi.extension = "CLIP"
            hi.interpolation = "Cubic"
            Lk.new(uva.outputs["UV"], hi.inputs["Vector"])
            masked_h = N.new("ShaderNodeMath")
            masked_h.operation = "MULTIPLY"
            Lk.new(hi.outputs["Color"], masked_h.inputs[0])
            Lk.new(front_body_mask.outputs["Value"], masked_h.inputs[1])
            ink_height = masked_h.outputs["Value"]

    modmix = N.new("ShaderNodeMixRGB")
    modmix.name = "BodyWoolMultiply"
    modmix.blend_type = "MULTIPLY"
    modmix.inputs["Fac"].default_value = 1.0
    Lk.new(body_base_sock, modmix.inputs["Color1"])
    if ink_alpha is not None:
        fm = N.new("ShaderNodeMixRGB")
        fm.name = "InkCleanMask"
        fm.inputs["Color2"].default_value = (1.0, 1.0, 1.0, 1.0)
        Lk.new(wv.outputs["Color"], fm.inputs["Color1"])
        Lk.new(ink_alpha, fm.inputs["Fac"])
        Lk.new(fm.outputs["Color"], modmix.inputs["Color2"])
    else:
        Lk.new(wv.outputs["Color"], modmix.inputs["Color2"])

    rib_tex_mod = N.new("ShaderNodeMixRGB")
    rib_tex_mod.name = "RibWoolMultiply"
    rib_tex_mod.blend_type = "MULTIPLY"
    rib_tex_mod.inputs["Fac"].default_value = 1.0
    Lk.new(rib_rgb.outputs[0], rib_tex_mod.inputs["Color1"])
    Lk.new(wv.outputs["Color"], rib_tex_mod.inputs["Color2"])

    final_col = N.new("ShaderNodeMixRGB")
    final_col.name = "BodyRibBlend"
    Lk.new(modmix.outputs["Color"], final_col.inputs["Color1"])
    Lk.new(rib_tex_mod.outputs["Color"], final_col.inputs["Color2"])
    Lk.new(rib_fac, final_col.inputs["Fac"])
    Lk.new(final_col.outputs["Color"], b.inputs["Base Color"])

    b.inputs["Specular IOR Level"].default_value = PRESET_PARAMS["specular_ior_level"]
    b.inputs["Roughness"].default_value = PRESET_PARAMS["roughness"]

    height_sock = weave
    if ink_alpha is not None:
        keep = N.new("ShaderNodeMath")
        keep.operation = "MULTIPLY_ADD"
        keep.inputs[1].default_value = -0.75
        keep.inputs[2].default_value = 1.0
        Lk.new(ink_alpha, keep.inputs[0])
        wk = N.new("ShaderNodeMath")
        wk.operation = "MULTIPLY"
        Lk.new(weave, wk.inputs[0])
        Lk.new(keep.outputs["Value"], wk.inputs[1])
        height_sock = wk.outputs["Value"]
    if ink_height is not None:
        ih = N.new("ShaderNodeMath")
        ih.operation = "MULTIPLY"
        ih.inputs[1].default_value = 2.0
        Lk.new(ink_height, ih.inputs[0])
        hs = N.new("ShaderNodeMath")
        hs.operation = "ADD"
        Lk.new(height_sock, hs.inputs[0])
        Lk.new(ih.outputs["Value"], hs.inputs[1])
        height_sock = hs.outputs["Value"]

    d_mid = N.new("ShaderNodeMath")
    d_mid.name = "SeamMidDelta"
    d_mid.operation = "SUBTRACT"
    d_mid.inputs[1].default_value = 0.50
    Lk.new(attr_sdf.outputs["Fac"], d_mid.inputs[0])
    d_abs = N.new("ShaderNodeMath")
    d_abs.name = "SeamAbsDelta"
    d_abs.operation = "ABSOLUTE"
    Lk.new(d_mid.outputs["Value"], d_abs.inputs[0])
    groove_mr = N.new("ShaderNodeMapRange")
    groove_mr.name = "CoverstitchGroove"
    groove_mr.interpolation_type = "SMOOTHSTEP"
    groove_mr.clamp = True
    groove_mr.inputs["From Min"].default_value = 0.0
    groove_mr.inputs["From Max"].default_value = PRESET_PARAMS["coverstitch_groove"][1]
    groove_mr.inputs["To Min"].default_value = PRESET_PARAMS["coverstitch_groove"][0]
    groove_mr.inputs["To Max"].default_value = 0.0
    Lk.new(d_abs.outputs["Value"], groove_mr.inputs["Value"])
    hs_seam = N.new("ShaderNodeMath")
    hs_seam.name = "HeightWithCoverstitch"
    hs_seam.operation = "ADD"
    Lk.new(height_sock, hs_seam.inputs[0])
    Lk.new(groove_mr.outputs["Result"], hs_seam.inputs[1])

    bump = N.new("ShaderNodeBump")
    bump.name = "MohairWeaveBump"
    bump.inputs["Strength"].default_value = PRESET_PARAMS["bump_strength"]
    bump.inputs["Distance"].default_value = PRESET_PARAMS["bump_distance"]
    Lk.new(hs_seam.outputs["Value"], bump.inputs["Height"])
    Lk.new(bump.outputs["Normal"], b.inputs["Normal"])
    return mat


def main() -> int:
    import bpy  # type: ignore

    bpy.ops.wm.read_factory_settings(use_empty=True)
    mat = bpy.data.materials.new(PRESET_NAME)
    build_mat_brushed_mohair_archive(bpy, mat)
    mat.use_fake_user = True
    mat.asset_mark()
    if getattr(mat, "asset_data", None) is not None:
        mat.asset_data.description = (
            "Reusable brushed-mohair / heavyweight knit Cycles shader preset archived from BS-RINGER-001 lookdev "
            "(smoothed _RIB_SDF seam mask + coverstitch groove + wool weave relief)."
        )
        for tag_name in ("wool", "mohair", "knitted", "ringer", "betterstyle", "archive"):
            mat.asset_data.tags.new(tag_name, skip_if_exists=True)

    bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND))

    nodes_meta = [
        {"name": n.name, "type": n.type, "bl_idname": n.bl_idname}
        for n in mat.node_tree.nodes
    ]
    payload = {
        "schema": "maximus.v2.shader_preset_archive/1",
        "preset_name": PRESET_NAME,
        "blend_file": OUT_BLEND.name,
        "blender_version": ".".join(str(x) for x in bpy.app.version),
        "asset_marked": bool(mat.asset_data is not None),
        "node_count": len(nodes_meta),
        "params": PRESET_PARAMS,
        "nodes": nodes_meta,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"preset_name": PRESET_NAME, "blend": str(OUT_BLEND), "json": str(OUT_JSON), "node_count": len(nodes_meta)}))
    return 0


if __name__ == "__main__":
    if "bpy" in sys.modules:
        raise SystemExit(main())
    print("Run inside Blender: blender --background --python-exit-code 1 --python mat_brushed_mohair_archive.py")
