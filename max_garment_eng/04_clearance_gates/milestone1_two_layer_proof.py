"""milestone1_two_layer_proof.py — Local CPU proof: singlet (layer 1) + split-front jacket (layer 2), UntangleAll, zipper, lapel peel.

Spec: docs/MAXIMUS_LAYERED_MATH_SPEC.md (S3, S4, S6). No Blender, no bpy, no GPU.

  G1_Singlet = BS-TANK-001 cloth part from its theta_fit extract NPZ (written by run_phase3_sku_blender.py --mode extract).
  G2_Jacket  = clean split-front torso cage built here in NumPy: one ring per height around the outer envelope of the body torso
               loop (arms dropped) and the solved singlet, plus 10 mm ease, front meridian left OPEN with a gap equal to the closed zipper's two tape widths,
               so the two seam loops sit exactly on the closed tapes' sewn edges.
  1. UntangleAll: G1 against the body, G2 against LayerCollider(body, [G1]) (newton_cloth_template.solve_layers_untangle_all).
  2. Gates on the closed drape: GG31_GWN per layer, G_LAYER_ORDER, G_STRETCH per layer.
  3. Zipper on the draped seam (zipper_engine.extract_seam_centerline + build_zipper_numpy(isometric=True)), open O = 0.58.
  4. peel_jacket_lapels_over_layer1 (heat geodesic from each seam loop + tangential PBD against body + G1, seams pinned to tape).
  5. Gates on the peeled state + G_ZIP_CLEAR. Writes PROOF_REPORT.json and proof_state.npz.

python -B milestone1_two_layer_proof.py --extract-npz ../out_milestone1/extract/BS-TANK-001_extract.npz --out-dir ../out_milestone1/two_layer_proof
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent
for p in (ENGINE / "03_gpu_cloth_solver", ENGINE / "COLAB_MEGA_SPRINT" / "scripts", ENGINE / "02_alignment_and_collar",
          ENGINE.parent / "maximus_engine" / "colab_stage"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import canon_geometry_kernels as K  # noqa: E402
import zipper_engine as ZE  # noqa: E402
from newton_cloth_template import ClothLayer, LayerCollider, solve_layers_untangle_all  # noqa: E402
import run_phase3_c0_sku as C0  # noqa: E402

G_STRETCH_P99, G_STRETCH_MAX = 1.08, 1.15


def _edges(F: np.ndarray) -> np.ndarray:
    e = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    return np.unique(e, axis=0)


def _torso_loop(V_body: np.ndarray, F_body: np.ndarray, z: float) -> np.ndarray:
    """Closed XY polygon of the torso cross-section at height z (the slice loop that encloses the torso centre)."""
    S0, S1 = K.horizontal_slice_segments(V_body, F_body, z)
    if len(S0) == 0:
        raise ValueError(f"no body slice at z={z}")
    pts = np.vstack([S0, S1])[:, :2]
    key = np.round(pts / 1e-7).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n = len(S0)
    a, b = inv[:n], inv[n:]
    adj: dict[int, list[int]] = {}
    for u, v in zip(a, b):
        adj.setdefault(int(u), []).append(int(v))
        adj.setdefault(int(v), []).append(int(u))
    coord = np.zeros((int(inv.max()) + 1, 2))
    coord[inv] = pts
    seen: set[int] = set()
    loops = []
    for start in adj:
        if start in seen:
            continue
        loop, prev, cur = [start], -1, start
        seen.add(start)
        while True:
            nxt = [w for w in adj[cur] if w != prev and w not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            seen.add(cur)
            loop.append(cur)
        if len(loop) >= 8:
            loops.append(coord[loop])
    # torso centre: mean of slice points with |x| < 0.10
    c = pts[np.abs(pts[:, 0]) < 0.10].mean(axis=0)

    def contains(poly: np.ndarray, p: np.ndarray) -> bool:
        x, y = poly[:, 0], poly[:, 1]
        x2, y2 = np.roll(x, -1), np.roll(y, -1)
        cross = ((y > p[1]) != (y2 > p[1])) & (p[0] < (x2 - x) * (p[1] - y) / np.where(y2 != y, y2 - y, 1e-30) + x)
        return bool(cross.sum() % 2)

    cand = [lp for lp in loops if contains(lp, c)]
    if not cand:
        raise ValueError(f"no torso loop encloses the centre at z={z}")
    loop = max(cand, key=lambda lp: len(lp))
    area = 0.5 * float(np.sum(loop[:, 0] * np.roll(loop[:, 1], -1) - np.roll(loop[:, 0], -1) * loop[:, 1]))
    return loop if area > 0 else loop[::-1]          # counter-clockwise seen from +Z


def _polar_envelope(points_xy: np.ndarray, c: np.ndarray, n_bins: int = 720, dilate_bins: int = 4,
                    smooth_bins: int = 9, iters: int = 6) -> np.ndarray:
    """Upper radial envelope r(theta) of a point set around centre c: per-bin max, gaps filled, dilated, then smoothed
    with r = max(moving_average(r), r_dilated) so the result is smooth and never inside any point."""
    d = points_xy - c
    th = np.arctan2(d[:, 1], d[:, 0])
    rad = np.linalg.norm(d, axis=1)
    b = ((th + np.pi) / (2 * np.pi) * n_bins).astype(np.int64) % n_bins
    r = np.full(n_bins, np.nan)
    np.fmax.at(r, b, rad)
    ok = np.isfinite(r)
    idx = np.arange(n_bins)
    r = np.interp(idx, idx[ok], r[ok], period=n_bins)
    rd = np.max(np.stack([np.roll(r, k) for k in range(-dilate_bins, dilate_bins + 1)]), axis=0)
    out = rd.copy()
    ker = np.ones(smooth_bins) / smooth_bins
    for _ in range(iters):
        pad = np.concatenate([out[-smooth_bins:], out, out[:smooth_bins]])
        out = np.maximum(np.convolve(pad, ker, mode="same")[smooth_bins:-smooth_bins], rd)
    return out


def build_split_front_jacket(V_body: np.ndarray, F_body: np.ndarray, z_bot: float = 0.97, z_top: float = 1.16,
                             row_dz: float = 0.010, n_cols: int = 96, ease_m: float = 0.010,
                             half_gap_m: float = 0.0138, inner: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """Split-front torso cage. Each ring is the outer radial envelope of the body torso loop and (if given) the solved inner
    layer's cross-section at that height, plus ease_m; an outer layer must be larger than what it covers (BS-TANK-001 stands
    27 mm off the body at the median, 63 mm at most, in this band). The front meridian is open with half-gap half_gap_m of
    arc each side of front centre (the closed zipper tapes' outer edges). Columns run from one lapel edge round the back to the
    other; wearer's left is +X."""
    zs = np.linspace(z_bot, z_top, int(round((z_top - z_bot) / row_dz)) + 1)
    n_bins = 720
    th_bins = -np.pi + (np.arange(n_bins) + 0.5) * (2 * np.pi / n_bins)
    rows = []
    for z in zs:
        loop = _torso_loop(V_body, F_body, float(z))
        c = loop.mean(axis=0)
        pts = loop
        if inner is not None:
            S0, S1 = K.horizontal_slice_segments(inner[0], inner[1], float(z))
            if len(S0):
                pts = np.vstack([loop, S0[:, :2], S1[:, :2]])
        r = _polar_envelope(pts, c, n_bins=n_bins) + ease_m
        ring = c + r[:, None] * np.column_stack([np.cos(th_bins), np.sin(th_bins)])   # CCW from +Z
        closed = np.vstack([ring, ring[:1]])
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
        L = float(cum[-1])
        i_fc = int(np.argmin(np.abs(th_bins + np.pi / 2)))           # front centre: direction -Y
        s_fc = float(cum[i_fc])
        s_vals = np.linspace(s_fc + half_gap_m, s_fc + L - half_gap_m, n_cols)
        P = np.column_stack([np.interp(s_vals % L, cum, closed[:, k]) for k in range(2)])
        rows.append(np.column_stack([P, np.full(n_cols, z)]))
    V = np.vstack(rows)
    nr = len(zs)
    F = []
    for i in range(nr - 1):
        for j in range(n_cols - 1):
            a, b = i * n_cols + j, i * n_cols + j + 1
            c2, d = (i + 1) * n_cols + j + 1, (i + 1) * n_cols + j
            F += [[a, b, c2], [a, c2, d]]
    F = np.asarray(F, dtype=np.int64)
    col0 = np.arange(nr) * n_cols
    colN = col0 + (n_cols - 1)
    if V[colN, 0].mean() > V[col0, 0].mean():
        seam_L, seam_R = colN, col0
    else:
        seam_L, seam_R = col0, colN
    cen = V[F].mean(axis=1)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    radial = cen - np.column_stack([np.zeros(len(cen)), np.full(len(cen), V[:, 1].mean()), cen[:, 2]])
    if float(np.einsum("ij,ij->i", fn, radial).mean()) < 0:
        F = F[:, ::-1].copy()
    return {"V": V, "F": F, "seam_L": seam_L, "seam_R": seam_R, "rows": nr, "cols": n_cols}


def _stretch(V_rest: np.ndarray, V: np.ndarray, E: np.ndarray) -> dict:
    r = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1) / np.maximum(np.linalg.norm(V_rest[E[:, 0]] - V_rest[E[:, 1]], axis=1), 1e-12)
    p99, mx = float(np.percentile(r, 99)), float(r.max())
    return {"p99_ratio": round(p99, 4), "max_ratio": round(mx, 4), "min_ratio": round(float(r.min()), 4),
            "pass": bool(p99 <= G_STRETCH_P99 and mx <= G_STRETCH_MAX)}


def _gwn(V: np.ndarray, Vb: np.ndarray, Fb: np.ndarray) -> dict:
    s, _ = K.signed_distance_to_mesh(V, Vb, Fb, k=16, device="auto")
    return {"inside_count": int((s < 0).sum()), "min_signed_mm": round(float(s.min() * 1000), 3), "pass": bool((s < 0).sum() == 0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extract-npz", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--open-factor", type=float, default=0.58)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    d = np.load(args.extract_npz, allow_pickle=True)
    V_all = d["V_fit"].astype(np.float64)
    F_all = d["F"].astype(np.int64)
    E_all = d["edges"].astype(np.int64)
    Vb = d["body_ref_V"].astype(np.float64)
    Fb = d["body_F"].astype(np.int64)
    ranges = json.loads(str(d["obj_vert_ranges_json"].item()))
    tank = next(r for r in ranges if int(r.get("part_id", 1)) == 1)
    m = np.zeros(len(V_all), dtype=bool)
    m[int(tank["start"]):int(tank["end"])] = True
    _, V1, F1, E1 = C0._extract_submesh(V_all, F_all, E_all, m)

    G1 = ClothLayer(name="G1_Singlet", V_rest=V1, F=F1, E=E1, layer_rank=1, part_id=1, thickness_m=0.0014)
    (G1,) = solve_layers_untangle_all([G1], Vb, Fb, eps_body_inner_m=0.0025, use_vbd=False, coupled_sweeps=0)
    jk = build_split_front_jacket(Vb, Fb, inner=(G1.V_solved, F1))   # jacket sized around the solved singlet
    V2r, F2 = jk["V"], jk["F"]
    E2 = _edges(F2)

    G2 = ClothLayer(name="G2_Jacket", V_rest=V2r, F=F2, E=E2, layer_rank=2, part_id=1, thickness_m=0.0025, closure_mode="open_front")
    order = solve_layers_untangle_all([G2, G1], Vb, Fb, eps_body_inner_m=0.0025, eps_body_outer_m=0.0030, layer_gap_m=0.0025,
                                      use_vbd=False, coupled_sweeps=10)
    G1, G2 = order
    closed = {
        "GWN_G1": _gwn(G1.V_solved, Vb, Fb),
        "GWN_G2": _gwn(G2.V_solved, Vb, Fb),
        "G_LAYER_ORDER": C0.eval_layer_order(order, Vb, Fb),
        "G_STRETCH_G1": _stretch(V1, G1.V_solved, E1),
        "G_STRETCH_G2": _stretch(V2r, G2.V_solved, E2),
    }
    print("[closed drape]", json.dumps(closed), flush=True)

    C0c, N0c = ZE.extract_seam_centerline(G2.V_solved, jk["seam_L"], jk["seam_R"], Vb, Fb)
    zcfg = ZE.ZipperConfig(open_factor=float(args.open_factor))
    zip_open = ZE.build_zipper_numpy(C0c, zcfg, curve_normals=N0c, isometric=True)
    zip_closed = ZE.build_zipper_numpy(C0c, ZE.ZipperConfig(open_factor=0.0), curve_normals=N0c, isometric=True)
    closed_seam_err = float(max(
        np.abs(np.linalg.norm(G2.V_solved[jk["seam_L"]] - C0c, axis=1) - zip_closed["tape_edges"]["outer_offset_m"]).max(),
        np.abs(np.linalg.norm(G2.V_solved[jk["seam_R"]] - C0c, axis=1) - zip_closed["tape_edges"]["outer_offset_m"]).max()))

    coll = LayerCollider(Vb, Fb, eps_body=0.0030, shells=[(G1.V_solved, G1.F, G1.thickness_m, 0.0025)])
    try:
        import alignment_adapters as A  # libigl heat geodesic when installed, else the same clean-room kernel
        geo = A.heat_geodesic
        geo_name = "alignment_adapters.heat_geodesic (igl=%s)" % (A.libigl_version() or "absent -> MC-50 kernel")
    except Exception as exc:  # noqa: BLE001
        geo = K.heat_geodesic_distance
        geo_name = f"canon_geometry_kernels.heat_geodesic_distance ({type(exc).__name__}: alignment_adapters unavailable)"
    peel = ZE.peel_jacket_lapels_over_layer1(G2.V_solved, F2, jk["seam_L"], jk["seam_R"], zip_open, coll, sigma_peel_m=0.07,
                                             edges=E2, V_rest=V2r, geodesic_fn=geo)
    V2p = peel["V"]
    G2p = ClothLayer(name="G2_Jacket_peeled", V_rest=V2r, F=F2, E=E2, layer_rank=2, part_id=1, thickness_m=0.0025)
    G2p.V_solved = V2p
    opened = {
        "GWN_G1": closed["GWN_G1"],
        "GWN_G2": _gwn(V2p, Vb, Fb),
        "G_LAYER_ORDER": C0.eval_layer_order([G1, G2p], Vb, Fb),
        "G_STRETCH_G2": _stretch(V2r, V2p, E2),
        "G_ZIP_CLEAR": zip_open["G_ZIP_CLEAR"],
        "lapel_opening": {
            "open_factor": float(args.open_factor),
            "top_half_gap_mm": zip_open["slider_state"]["top_split_half_gap_mm"],
            "seam_max_move_mm": peel["seam_max_move_mm"],
            "seam_target_err_mm": peel["seam_target_err_mm"],
            "front_gap_top_mm": round(float(np.linalg.norm(V2p[jk["seam_L"][-1]] - V2p[jk["seam_R"][-1]]) * 1000), 2),
            "front_gap_top_closed_mm": round(float(np.linalg.norm(G2.V_solved[jk["seam_L"][-1]] - G2.V_solved[jk["seam_R"][-1]]) * 1000), 2),
        },
    }
    all_pass = bool(opened["GWN_G1"]["pass"] and opened["GWN_G2"]["pass"] and opened["G_LAYER_ORDER"]["pass"]
                    and opened["G_STRETCH_G2"]["pass"] and opened["G_ZIP_CLEAR"]["pass"]
                    and opened["G_ZIP_CLEAR"]["min_closed_intertooth_clearance_mm"] >= 0.05
                    and closed["GWN_G2"]["pass"] and closed["G_LAYER_ORDER"]["pass"] and closed["G_STRETCH_G2"]["pass"])
    report = {
        "proof": "Milestone 1 synthetic 2-layer (singlet + split-front jacket), local CPU, no SolverVBD",
        "inputs": {"extract_npz": str(args.extract_npz), "G1": tank["name"], "G1_verts": int(len(V1)),
                   "G2_cage": {"verts": int(len(V2r)), "faces": int(len(F2)), "rows": jk["rows"], "cols": jk["cols"],
                               "z_range_m": [0.97, 1.16], "ease_m": 0.010, "top_edge_note": "1.22 m ran into the right armpit (4-6 mm body room vs ~7 mm needed): infeasible for a sleeveless cut", "sized_around": "solved G1 envelope", "front_half_gap_m": 0.0138}},
        "geodesic": geo_name,
        "untangle": {L.name: {"init": L.info.get("init", {}), "pbd": L.info.get("pbd", {}), "pbd_coupled": L.info.get("pbd_coupled")} for L in order},
        "closed_state": closed,
        "closed_seam_vs_closed_tape_edge_err_mm": round(closed_seam_err * 1000, 3),
        "peeled_state": opened,
        "peel_pbd": peel["pbd"],
        "all_pass": all_pass,
        "seconds": round(time.time() - t0, 1),
    }
    (args.out_dir / "PROOF_REPORT.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    np.savez_compressed(args.out_dir / "proof_state.npz", V_body=Vb.astype(np.float32), F_body=Fb.astype(np.int32),
                        V_G1=G1.V_solved.astype(np.float32), F_G1=F1.astype(np.int32),
                        V_G2_rest=V2r.astype(np.float32), V_G2_closed=G2.V_solved.astype(np.float32), V_G2_peeled=V2p.astype(np.float32),
                        F_G2=F2.astype(np.int32), seam_L=jk["seam_L"], seam_R=jk["seam_R"],
                        zipper_V=zip_open["V"].astype(np.float32), zipper_F=zip_open["F"].astype(np.int32))
    print(json.dumps({"closed_state": closed, "peeled_state": opened, "all_pass": all_pass, "seconds": report["seconds"]}, indent=1, default=str))
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
