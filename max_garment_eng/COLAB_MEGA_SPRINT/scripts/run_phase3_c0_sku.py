"""run_phase3_c0_sku.py — Universal Newton 1.8.0 SolverVBD + PBD Strain/GWN C0 Cloth Solver for All 8 SKUs.

Milestone 1 (docs/MAXIMUS_LAYERED_MATH_SPEC.md): N-layer UntangleAll. Cloth parts from `obj_vert_ranges_json` are sorted by
(layer_rank, part_id) and solved inner-to-outer against a LayerCollider (body + solved inner shells, per-collider signed
distance union); `_close_open_mesh_caps` is retired from the solve path. Gates add G_LAYER_ORDER (outer-edge x inner-triangle
crossings == 0 via maximus_core.clearance.segment_triangle_hits, and no outer vertex behind an inner shell).

`--local-cpu` runs without Newton (two-phase NumPy preconditioner in place of the SolverVBD drape) and says so in the report.
Without the flag a missing Newton is an error, so a Colab run can never silently skip the drape.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import canon_geometry_kernels as K  # type: ignore
from newton_cloth_template import (  # type: ignore
    ClothLayer,
    LayerCollider,
    NewtonClothConfig,
    check_runtime_availability,
    pbd_strain_limit_and_tangent_clearance,
    solve_cloth_drape,
    solve_layers_untangle_all,
)
from pyuipc_ipc_benchmark import inspect_pyuipc_runtime  # type: ignore

G_STRETCH_P99_LIMIT = 1.08
G_STRETCH_MAX_LIMIT = 1.15
BODY_MARGIN_M = 0.0025
TOP_OVER_TROUSER_MARGIN_M = 0.0030
LAYER_GAP_M = 0.0025
DEFAULT_THICKNESS_M = {"top_rank1": 0.0014, "trousers": 0.0020, "outer": 0.0025}


def _segment_triangle_hits():
    """maximus_core.clearance.segment_triangle_hits (single owner: 01_code_library corpus). On Colab the staged bundle must
    carry maximus_core/ on PYTHONPATH; locally it is found in the corpus."""
    try:
        from maximus_core.clearance import segment_triangle_hits  # type: ignore
        return segment_triangle_hits
    except ImportError:
        for up in THIS_DIR.parents:
            cand = up / "01_code_library" / "20_3d_rig_topology_and_physics" / "primitives" / "owned"
            if (cand / "maximus_core" / "clearance.py").is_file():
                sys.path.insert(0, str(cand))
                from maximus_core.clearance import segment_triangle_hits  # type: ignore
                return segment_triangle_hits
        raise ImportError("maximus_core not found: stage 01_code_library/.../primitives/owned/maximus_core with the scripts")


def layer_records(cloth_ranges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Spec §2 schema with backward-compatible defaults, sorted by (layer_rank, part_id). part_id 3 (static) is dropped."""
    has_trousers = any(int(r.get("part_id", 1)) == 2 for r in cloth_ranges)
    out = []
    for r in cloth_ranges:
        pid = int(r.get("part_id", 1))
        if pid == 3:
            continue
        rank = int(r.get("layer_rank", 1 if pid == 2 else (2 if has_trousers else 1)))
        if "thickness_m" in r:
            t = float(r["thickness_m"])
        elif pid == 2:
            t = DEFAULT_THICKNESS_M["trousers"]
        else:
            t = DEFAULT_THICKNESS_M["top_rank1"] if rank <= 1 else DEFAULT_THICKNESS_M["outer"]
        out.append({**r, "part_name": r.get("part_name", r.get("name", f"part_{pid}")), "part_id": pid, "layer_rank": rank,
                    "thickness_m": t, "closure_mode": r.get("closure_mode", "closed")})
    return sorted(out, key=lambda r: (r["layer_rank"], r["part_id"]))


def eval_layer_order(layers: list[Any], V_body: np.ndarray, F_body: np.ndarray) -> dict[str, Any]:
    """G_LAYER_ORDER: for every (outer, inner) pair with rank_outer > rank_inner: outer edges crossing inner triangles == 0, and
    outer vertices behind the inner shell (thickness-adjusted signed distance < 0 where the shell fully applies) == 0."""
    hits_fn = _segment_triangle_hits()
    pairs = []
    for j, L_out in enumerate(layers):
        for L_in in layers[:j]:
            if int(L_in.layer_rank) >= int(L_out.layer_rank):
                continue
            V_o = L_out.V_solved
            n_hits, _ = hits_fn(V_o[L_out.E[:, 0]], V_o[L_out.E[:, 1]], L_in.V_solved, L_in.F)
            probe = LayerCollider(V_body, F_body, eps_body=0.0, shells=[(L_in.V_solved, L_in.F, L_in.thickness_m, 0.0)])
            s_r, _, w = probe.shell_signed(0, V_o)
            active = w >= 1.0
            wrong = int(((s_r < 0.0) & active).sum())
            min_gap = float(s_r[active].min() * 1000.0) if active.any() else float("nan")
            pairs.append({"outer": L_out.name, "inner": L_in.name, "segment_triangle_hits": int(n_hits),
                          "outer_verts_behind_inner": wrong, "min_gap_mm": round(min_gap, 3),
                          "pass": bool(n_hits == 0 and wrong == 0)})
    return {"pairs": pairs, "evaluated": bool(pairs), "pass": all(p["pass"] for p in pairs)}


def _close_open_mesh_caps(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RETIRED from the solve path (spec §1): fan caps invent a lid across waist and hems. Kept one release for old callers.

    Return a temporary watertight copy of `(V, F)` by fan-capping open boundary loops."""
    from collections import defaultdict

    V_arr = np.asarray(V, dtype=np.float64)
    F_arr = np.asarray(F, dtype=np.int64)
    if len(V_arr) == 0 or len(F_arr) == 0:
        return V_arr, F_arr

    dir_edges: list[tuple[int, int]] = []
    undir_counts: dict[tuple[int, int], int] = defaultdict(int)
    for tri in F_arr:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            dir_edges.append((u, v))
            key = (u, v) if u < v else (v, u)
            undir_counts[key] += 1

    bndry_adj: dict[int, list[int]] = defaultdict(list)
    for u, v in dir_edges:
        key = (u, v) if u < v else (v, u)
        if undir_counts[key] == 1:
            bndry_adj[u].append(v)

    if not bndry_adj:
        return V_arr, F_arr

    visited_edges: set[tuple[int, int]] = set()
    extra_verts: list[np.ndarray] = []
    extra_faces: list[tuple[int, int, int]] = []
    v_offset = len(V_arr)

    for start_u in sorted(bndry_adj.keys()):
        for start_v in bndry_adj[start_u]:
            if (start_u, start_v) in visited_edges:
                continue
            loop = [start_u]
            visited_edges.add((start_u, start_v))
            prev_u, cur = start_u, start_v
            guard = 0
            while cur != start_u and guard < len(V_arr) + 16:
                loop.append(cur)
                nxt = None
                for cand in bndry_adj.get(cur, []):
                    if (cur, cand) not in visited_edges:
                        nxt = cand
                        break
                if nxt is None:
                    break
                visited_edges.add((cur, nxt))
                prev_u, cur = cur, nxt
                guard += 1
            if len(loop) >= 3:
                c_pos = V_arr[loop].mean(axis=0)
                c_idx = v_offset + len(extra_verts)
                extra_verts.append(c_pos)
                # Reverse directed boundary edge `(u -> v)` as `(v, u, c_idx)` for consistent outward CCW winding
                for k in range(len(loop)):
                    u = loop[k]
                    v = loop[(k + 1) % len(loop)]
                    extra_faces.append((v, u, c_idx))

    if not extra_faces:
        return V_arr, F_arr
    V_closed = np.vstack([V_arr, np.asarray(extra_verts, dtype=np.float64)])
    F_closed = np.vstack([F_arr, np.asarray(extra_faces, dtype=np.int64)])
    return V_closed, F_closed


def _extract_submesh(
    V_all: np.ndarray,
    F_all: np.ndarray,
    edges_all: np.ndarray,
    v_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition cloth vertices, triangles, and edges cleanly via boolean vertex mask."""
    sub_v_idx = np.flatnonzero(v_mask)
    old_to_new = np.full(len(V_all), -1, dtype=np.int64)
    old_to_new[sub_v_idx] = np.arange(len(sub_v_idx), dtype=np.int64)
    f_mask = np.all(v_mask[F_all], axis=1)
    e_mask = np.all(v_mask[edges_all], axis=1)
    return sub_v_idx, V_all[sub_v_idx].copy(), old_to_new[F_all[f_mask]], old_to_new[edges_all[e_mask]]


def eval_phase3_c0_gates(
    V_rest: np.ndarray,
    V_def: np.ndarray,
    edges: np.ndarray,
    V_body: np.ndarray,
    F_body: np.ndarray,
    collar_mask: np.ndarray,
    layer_order: dict[str, Any] | None = None,
    **_retired: Any,
) -> dict[str, Any]:
    """Independent C0 gates on saved geometry. `layer_order` is eval_layer_order(...) output (G_LAYER_ORDER).
    Retired kwargs (V_top_solved, V_bot_closed, F_bot_closed: the fan-capped TOP_OVER_TROUSER test) are accepted and ignored."""
    e_rest = np.linalg.norm(V_rest[edges[:, 0]] - V_rest[edges[:, 1]], axis=1)
    e_def = np.linalg.norm(V_def[edges[:, 0]] - V_def[edges[:, 1]], axis=1)
    ratios = e_def / np.maximum(e_rest, 1e-12)
    p99 = float(np.percentile(ratios, 99.0))
    rmax = float(ratios.max())

    signed_m, _ = K.signed_distance_to_mesh(V_def, V_body, F_body, k=16, device="auto")
    inside_all = int((signed_m < 0.0).sum())
    cs = signed_m[collar_mask] if collar_mask.any() else signed_m[:0]
    collar_inside = int((cs < 0.0).sum()) if len(cs) else 0
    collar_med_mm = float(np.median(cs) * 1000.0) if len(cs) else 0.0
    collar_min_mm = float(cs.min() * 1000.0) if len(cs) else 0.0

    lo = layer_order if layer_order is not None else {"pairs": [], "evaluated": False, "pass": True}
    stretch_pass = bool(p99 <= G_STRETCH_P99_LIMIT and rmax <= G_STRETCH_MAX_LIMIT)
    gwn_pass = bool(inside_all == 0)
    o7_pass = bool(collar_inside == 0)
    lo_pass = bool(lo["pass"])
    return {
        "G_STRETCH": {
            "p99_ratio": round(p99, 4),
            "max_ratio": round(rmax, 4),
            "edges_gt_1_15": int((ratios > G_STRETCH_MAX_LIMIT).sum()),
            "pass": stretch_pass,
        },
        "GG31_GWN": {
            "inside_count": inside_all,
            "min_signed_mm": round(float(signed_m.min() * 1000.0), 3),
            "median_signed_mm": round(float(np.median(signed_m) * 1000.0), 3),
            "pass": gwn_pass,
        },
        "O7_COLLAR": {
            "collar_verts": int(collar_mask.sum()),
            "collar_inside": collar_inside,
            "collar_min_mm": round(collar_min_mm, 3),
            "collar_median_mm": round(collar_med_mm, 3),
            "pass": o7_pass,
        },
        "G_LAYER_ORDER": lo,
        "c0_all_pass": bool(stretch_pass and gwn_pass and o7_pass and lo_pass),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 3 Universal Newton SolverVBD C0 Solve (N-layer UntangleAll)")
    p.add_argument("--extract-npz", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--sku", required=True)
    p.add_argument("--n-frames", type=int, default=120, help="SolverVBD frames at dt 1/240 s (120 = 0.5 s settle)")
    p.add_argument("--local-cpu", action="store_true",
                   help="Run without Newton: NumPy two-phase preconditioner replaces the SolverVBD drape (reported).")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(args.extract_npz, allow_pickle=True)
    V_rest = d["V_fit"].astype(np.float64)
    F_garment = d["F"].astype(np.int64)
    edges = d["edges"].astype(np.int64)
    collar_mask = d["collar_mask"].astype(bool)
    V_body = d["body_ref_V"].astype(np.float64)
    F_body = d["body_F"].astype(np.int64)
    cloth_ranges = json.loads(str(d["obj_vert_ranges_json"].item())) if "obj_vert_ranges_json" in d else []
    if not cloth_ranges:
        cloth_ranges = [{"name": args.sku, "start": 0, "end": len(V_rest), "part_id": 1}]

    status = check_runtime_availability()
    have_newton = bool(status["newton_installed"] and status["warp_installed"])
    if not have_newton and not args.local_cpu:
        raise RuntimeError("Newton/warp not installed. On Colab L4 install them; for a local CPU proof pass --local-cpu.")
    use_vbd = have_newton and not args.local_cpu

    cfg = NewtonClothConfig(
        solver_kind="vbd",
        substeps=10,
        iterations=10,
        dt=1.0 / 240.0,
        stretch_stiffness=1e4,
        bend_stiffness=1e-2,
        body_margin_m=BODY_MARGIN_M,
    )

    recs = layer_records(cloth_ranges)
    layers: list[ClothLayer] = []
    for r in recs:
        v_mask = np.zeros(len(V_rest), dtype=bool)
        v_mask[int(r["start"]) : int(r["end"])] = True
        v_idx, V_l, F_l, E_l = _extract_submesh(V_rest, F_garment, edges, v_mask)
        if int(r["part_id"]) == 2:
            pinned = V_l[:, 2] >= (float(V_l[:, 2].max()) - 0.035)   # trousers: top 35 mm (waistband) pinned in SolverVBD
        else:
            # Closed-neck tops settle UNPINNED in SolverVBD (re-canary 2026-11-23): pinning the collar where the extract leaves it,
            # ~15 mm above the shoulders, hung the yoke and sleeves (558 edges > 1.08 at z 1.3-1.6 m). O7_COLLAR still gates the
            # collar zone (collar_mask) after the solve; pins only affect the drape.
            pinned = np.zeros(len(V_l), dtype=bool)
        L = ClothLayer(name=str(r["part_name"]), V_rest=V_l, F=F_l, E=E_l, layer_rank=int(r["layer_rank"]),
                       part_id=int(r["part_id"]), thickness_m=float(r["thickness_m"]), pinned=pinned,
                       closure_mode=str(r["closure_mode"]))
        L.info["v_idx"] = v_idx
        layers.append(L)

    order = solve_layers_untangle_all(
        layers, V_body, F_body,
        eps_body_inner_m=BODY_MARGIN_M, eps_body_outer_m=TOP_OVER_TROUSER_MARGIN_M, layer_gap_m=LAYER_GAP_M,
        cfg=cfg, use_vbd=use_vbd, n_frames=args.n_frames, coupled_sweeps=10,
    )

    V_solved = V_rest.copy()
    for L in order:
        V_solved[L.info["v_idx"]] = L.V_solved
    l_bar = np.linalg.norm(V_rest[edges[:, 0]] - V_rest[edges[:, 1]], axis=1)

    layer_order = eval_layer_order(order, V_body, F_body)
    solved_gates = eval_phase3_c0_gates(V_rest, V_solved, edges, V_body, F_body, collar_mask, layer_order=layer_order)
    e_s = np.linalg.norm(V_solved[edges[:, 0]] - V_solved[edges[:, 1]], axis=1)
    er_ratio = e_s / np.maximum(l_bar, 1e-12)
    v_sum = np.zeros(len(V_solved), dtype=np.float64)
    v_cnt = np.zeros(len(V_solved), dtype=np.float64)
    np.add.at(v_sum, edges[:, 0], er_ratio)
    np.add.at(v_sum, edges[:, 1], er_ratio)
    np.add.at(v_cnt, edges[:, 0], 1.0)
    np.add.at(v_cnt, edges[:, 1], 1.0)
    tension = np.clip((v_sum / np.maximum(v_cnt, 1.0) - 0.92) / (1.12 - 0.92), 0.0, 1.0).astype(np.float32)

    solved_npz = args.out_dir / "c0_solved.npz"
    np.savez_compressed(
        solved_npz,
        sku=np.array(args.sku),
        V_rest=V_rest.astype(np.float32),
        V_solved=V_solved.astype(np.float32),
        F=F_garment.astype(np.int32),
        collar_mask=collar_mask,
        tension=tension,
        layer_records_json=np.array(json.dumps(recs)),
    )

    report = {
        "sku": args.sku,
        "solver": "newton.solvers.SolverVBD + LayerCollider tangential PBD" if use_vbd
        else "LOCAL CPU: NumPy two-phase preconditioner (no SolverVBD) + LayerCollider tangential PBD",
        "vbd_run": use_vbd,
        "newton_status": status,
        "pyuipc_status": inspect_pyuipc_runtime(),
        "layers": [{"name": L.name, "layer_rank": L.layer_rank, "part_id": L.part_id, "thickness_m": L.thickness_m,
                    "verts": int(len(L.V_rest)), "init": L.info.get("init", {}), "pbd": L.info.get("pbd", {}),
                    "pbd_coupled": L.info.get("pbd_coupled")} for L in order],
        "newton_vbd_gates": solved_gates,
        "solved_npz": str(solved_npz),
    }
    (args.out_dir / "C0_GATE_REPORT.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    assert np.isfinite(V_solved).all() and solved_gates["c0_all_pass"], f"[{args.sku}] C0 gate failed: {solved_gates}"


if __name__ == "__main__":
    main()
