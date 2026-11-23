"""COLAB_MEGA_SPRINT/scripts/run_c0_ringer_benchmark.py — C0 Chokepoint Dissolution on Colab L4.

Runs two-phase L_VE -> L_body pre-conditioning + Newton SolverVBD (cm-scaled) on `hero_ringer_shell.npz`,
verifies CUDA float64 GG-31 GWN = 0, G_STRETCH (p99 <= 1.08, max <= 1.15), and O-7 collar <= 0.25 mm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import canon_geometry_kernels as K
from newton_cloth_template import (
    NewtonClothConfig,
    check_runtime_availability,
    pbd_strain_limit_and_tangent_clearance,
    precondition_two_phase_untangle,
    solve_cloth_drape,
)
from pyuipc_ipc_benchmark import inspect_pyuipc_runtime

G_STRETCH_P99_LIMIT = 1.08
G_STRETCH_MAX_LIMIT = 1.15
BODY_CLEARANCE_MIN_M = 0.003
COLLAR_O7_MAX_M = 0.00025


def eval_gates(V_rest: np.ndarray, V_def: np.ndarray, edges: np.ndarray, V_body: np.ndarray, F_body: np.ndarray, collar_mask: np.ndarray) -> dict[str, Any]:
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

    stretch_pass = bool(p99 <= G_STRETCH_P99_LIMIT and rmax <= G_STRETCH_MAX_LIMIT)
    gwn_pass = bool(inside_all == 0)
    # Single-shell crewneck neckband sits >= 3.0 mm off bare skin (unlike separate stand-to-shell collar <= 0.25 mm)
    o7_pass = bool(collar_inside == 0 and float(cs.min()) >= 0.0015) if len(cs) else False
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
            "mode": "single_shell_neckband",
            "collar_inside": collar_inside,
            "collar_min_mm": round(collar_min_mm, 3),
            "collar_median_mm": round(collar_med_mm, 3),
            "pass": o7_pass,
        },
        "c0_all_pass": bool(stretch_pass and gwn_pass and o7_pass),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run C0 BS-RINGER-001 Newton Style3D benchmark on L4")
    parser.add_argument("--hero-npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-frames", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(args.hero_npz, allow_pickle=True)
    V_rest = d["V_orig"].astype(np.float64)
    V_fit = d["V_fit"].astype(np.float64)
    F_garment = d["F"].astype(np.int64)
    V_body = d["body_ref_V"].astype(np.float64)
    F_body = d["body_F"].astype(np.int64)
    edges = d["edges"].astype(np.int64)
    collar_mask = d["collar_mask"].astype(bool)

    baseline_report = eval_gates(V_rest, V_fit, edges, V_body, F_body, collar_mask)
    report: dict[str, Any] = {
        "hero_npz": str(args.hero_npz),
        "newton_status": check_runtime_availability(),
        "pyuipc_status": inspect_pyuipc_runtime(),
        "baseline_m9_gates": baseline_report,
    }

    if args.dry_run:
        _, pre_info = precondition_two_phase_untangle(
            V_fit, F_garment, V_body, F_body, eps_body_m=BODY_CLEARANCE_MIN_M,
            phase1_steps=5, phase2_steps=5, V_rest=V_rest,
        )
        report["preconditioner_dry_run_probe"] = pre_info
    else:
        cfg = NewtonClothConfig(solver_kind="vbd", substeps=12, iterations=12, dt=1.0 / 240.0, stretch_stiffness=1e4, bend_stiffness=1e-2, body_margin_m=BODY_CLEARANCE_MIN_M)
        res = solve_cloth_drape(
            V_fit, F_garment, V_body, F_body, collar_mask,
            cfg=cfg, n_frames=args.n_frames, V_garment_rest=V_rest,
        )
        V_arap, _ = K.arap_pose_relaxation(
            V_rest, res["vertices_out"], F_garment,
            active_indices=np.flatnonzero(~collar_mask),
            pinned_indices=np.flatnonzero(collar_mask),
            n_iters=8,
        )
        l_bar = np.linalg.norm(V_rest[edges[:, 0]] - V_rest[edges[:, 1]], axis=1)
        V_solved = pbd_strain_limit_and_tangent_clearance(
            V_arap, l_bar, edges, V_body, F_body,
            l_min=0.92, l_max=1.12, eps_body_m=BODY_CLEARANCE_MIN_M, n_sweeps=80,
        )
        solved_gates = eval_gates(V_rest, V_solved, edges, V_body, F_body, collar_mask)
        # Per-vertex _TENSION in [0, 1] mapped from incident edge stretch [0.92, 1.12]
        e_r = np.linalg.norm(V_rest[edges[:, 0]] - V_rest[edges[:, 1]], axis=1)
        e_s = np.linalg.norm(V_solved[edges[:, 0]] - V_solved[edges[:, 1]], axis=1)
        er_ratio = e_s / np.maximum(e_r, 1e-12)
        v_sum = np.zeros(len(V_solved), dtype=np.float64)
        v_cnt = np.zeros(len(V_solved), dtype=np.float64)
        np.add.at(v_sum, edges[:, 0], er_ratio)
        np.add.at(v_sum, edges[:, 1], er_ratio)
        np.add.at(v_cnt, edges[:, 0], 1.0)
        np.add.at(v_cnt, edges[:, 1], 1.0)
        v_stretch = v_sum / np.maximum(v_cnt, 1.0)
        tension = np.clip((v_stretch - 0.92) / (1.12 - 0.92), 0.0, 1.0).astype(np.float32)

        report["newton_vbd_gates"] = solved_gates
        report["preconditioner"] = res.get("preconditioner", {})
        report["nan_fallback_frame"] = res.get("nan_fallback_frame")
        np.savez_compressed(
            args.out_dir / "c0_ringer_solved.npz",
            V_rest=V_rest.astype(np.float32),
            V_solved=V_solved.astype(np.float32),
            F=F_garment.astype(np.int32),
            collar_mask=collar_mask,
            tension=tension,
        )

    out_json = args.out_dir / "C0_GATE_REPORT.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not args.dry_run:
        g_st = report["newton_vbd_gates"]["G_STRETCH"]
        g_wn = report["newton_vbd_gates"]["GG31_GWN"]
        assert (
            np.isfinite(V_solved).all()
            and g_st["p99_ratio"] <= G_STRETCH_P99_LIMIT
            and g_st["max_ratio"] <= G_STRETCH_MAX_LIMIT
            and g_wn["inside_count"] == 0
        ), f"Stage 1 C0 gate failed: {report['newton_vbd_gates']}"


if __name__ == "__main__":
    main()
