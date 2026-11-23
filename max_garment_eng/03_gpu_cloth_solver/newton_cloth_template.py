"""03_gpu_cloth_solver/newton_cloth_template.py — GPU Cloth Solver Template.

Licence: the upstream dependency `newton-physics/newton` is Apache-2.0; this wrapper script is
covered by this repository's LICENSE.

Uses `newton-physics/newton` v1.8.0 `SolverVBD` (deterministic unified compliant ALM VBD)
paired with a clean-room two-phase L_VE -> L_body pre-conditioner so t=0 enters physics
with GWN > 0.5 == 0.

NOTE (VERIFIED DEFECT newton#4930): `SolverStyle3D` hardcodes a 32-neighbor ELL matrix
capacity (`max_neighbor=32`) that silently drops stiffness entries at high-valence sewn
underarm junctions. `SolverVBD` is therefore the mandatory primary solver, with coordinates
scaled 100x into centimeters for Hessian conditioning and 0.01x back to meters.

Milestone 1 (2026-11-22, spec: docs/MAXIMUS_LAYERED_MATH_SPEC.md §3):
  * `LayerCollider`: per-collider signed-distance union (closed body + open inner cloth shells with
    shell thickness and an 8 mm boundary taper). Replaces fan-capped stacked collider meshes.
  * `pbd_strain_limit_and_tangent_clearance`: vectorised Jacobi strain limiting with outward
    half-space projection near any collider (an edge clamp can never pull a vertex inside) and
    per-vertex push-out capped at 0.25 x the shortest incident edge.
  * `solve_layers_untangle_all`: N-layer inner-to-outer solve + coupled relaxation sweeps.
  * Physical parameters are given in SI and converted once to the kg-cm-s unit system the
    cm-scaled VBD model runs in (`cm_unit_params`); `add_cloth_mesh` kwargs are filtered with
    `inspect.signature` so only names the installed Newton accepts are passed.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np


@dataclass
class NewtonClothConfig:
    solver_kind: Literal["vbd", "xpbd", "style3d"] = "vbd"
    substeps: int = 12
    iterations: int = 12
    dt: float = 1.0 / 240.0
    stretch_stiffness: float = 1e4   # membrane stiffness tri_ke, SI N/m (= J/m^2)
    bend_stiffness: float = 1e-2     # bending stiffness edge_ke, SI N (= J per metre of hinge; Newton: k = edge_ke * rest_len)
    density_kg_m2: float = 0.3       # areal density, SI kg/m^2 (280 gsm cotton ~ 0.28)
    contact_ke_n_per_m: float = 1.0e3  # particle-vs-shape penalty stiffness, SI N/m
    contact_kd: float = 1.0e-2
    body_friction: float = 0.25
    body_margin_m: float = 0.003
    gravity_z: float = -9.81
    cm_scale: float = 100.0  # Scale meters -> cm for SolverVBD Hessian conditioning
    max_drape_disp_m: float = 0.10       # displacement guard: a vertex further than this from its start stops the drape
                                         # (re-canary 2026-11-23: trousers settled at 40.3 mm; an unpinned top seating onto the
                                         # shoulders from ~15 mm above can legitimately reach 45-60 mm)
    max_frame_step_m: float = 0.010      # runaway guard: any vertex moving more than this in one frame (2.4 m/s at 240 fps)
                                         # stops the drape; the runaway ringer pass reached 11.2-12.3 mm/frame before stopping
    vbd_inner_shell_colliders: bool = False  # inner cloth shells are NOT SolverVBD colliders (open meshes have no inside);
                                             # layer order is enforced afterwards by the LayerCollider PBD pass
    rigid_body_contact_buffer_size: int = 512            # SolverVBD per-body body-body list (Newton default 64)
    rigid_body_particle_contact_buffer_size: int = 1024  # SolverVBD per-body soft-contact list (Newton default 256)


def cm_unit_params(cfg: NewtonClothConfig) -> dict[str, float]:
    """Convert SI cloth parameters into the kg-cm-s system the cm-scaled model runs in (length x s, mass and time unchanged).

    Factors audited against the Newton 1.8 VBD source vendored in primitives/extracted/newton-physics__newton (spec section 3a):
      membrane tri_ke (Neo-Hookean mu; kernels do force *= area)    J/m^2 = kg/s^2      x 1
      bending edge_ke (kernel: k = edge_ke * rest_len, E = k th^2/2) J/m   = N          x s       (100)
      contact soft_contact_ke (kernel: dE/dD = -k * depth)          N/m   = kg/s^2      x 1
      contact damping kd                                             N s/m = kg/s        x 1
      areal density (particle mass = density * area)                 kg/m^2              / s^2     (1e-4)
      gravity                                                        m/s^2               x s       (-981)
    Body-particle contact stiffness in Newton is 0.5 * (soft_contact_ke + shape_material_ke); solve_cloth_drape pins the
    shape side to the same value so the body contact equals contact_ke_n_per_m exactly.
    """
    s = float(cfg.cm_scale)
    return {
        "tri_ke": float(cfg.stretch_stiffness),
        "edge_ke": float(cfg.bend_stiffness) * s,
        "density": float(cfg.density_kg_m2) / (s * s),
        "soft_contact_ke": float(cfg.contact_ke_n_per_m),
        "soft_contact_kd": float(cfg.contact_kd),
        "shape_material_ke": float(cfg.contact_ke_n_per_m),
        "shape_material_kd": float(cfg.contact_kd),
        "gravity_z": float(cfg.gravity_z) * s,
    }


def unit_audit(cfg: NewtonClothConfig, mean_particle_area_m2: float, mean_edge_m: float) -> dict[str, float]:
    """Dimensionless per-substep stiffness ratios (k dt^2 / m) for the cm model vs the same physics in SI: they must match.

    A ratio is the stiffness of a term relative to the particle inertia in one VBD substep; it is unit-system invariant, so
    an SI ratio != cm ratio means a conversion error. Also returns what the pre-Milestone-1 template produced."""
    s = float(cfg.cm_scale)
    up = cm_unit_params(cfg)
    dt = float(cfg.dt) / max(1, int(cfg.substeps))
    m_si = float(cfg.density_kg_m2) * mean_particle_area_m2
    m_cm = up["density"] * mean_particle_area_m2 * s * s
    bend_si = float(cfg.bend_stiffness) * mean_edge_m          # Newton k = edge_ke * rest_len
    bend_cm = up["edge_ke"] * mean_edge_m * s
    area_si, area_cm = mean_particle_area_m2, mean_particle_area_m2 * s * s
    old_m = 0.3 * area_cm                                       # old template: density 0.3 passed straight into cm units
    return {
        "substep_dt_s": dt,
        "particle_mass_kg_si": m_si,
        "particle_mass_kg_cm_model": m_cm,
        "contact_ratio_si": float(cfg.contact_ke_n_per_m) * dt * dt / m_si,
        "contact_ratio_cm": up["soft_contact_ke"] * dt * dt / m_cm,
        "membrane_ratio_si": float(cfg.stretch_stiffness) * area_si * dt * dt / (m_si * mean_edge_m ** 2),
        "membrane_ratio_cm": up["tri_ke"] * area_cm * dt * dt / (m_cm * (mean_edge_m * s) ** 2),
        "bending_ratio_si": bend_si * dt * dt / (m_si * mean_edge_m ** 2),
        "bending_ratio_cm": bend_cm * dt * dt / (m_cm * (mean_edge_m * s) ** 2),
        "old_template_particle_mass_kg": old_m,
        "old_template_contact_ratio": 10.0 * dt * dt / old_m,
        "old_vs_new_contact_accel_factor": (up["soft_contact_ke"] / m_cm) / (10.0 / old_m),
    }


def _filter_kwargs(fn: Any, wanted: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Keep only kwargs `fn` accepts (inspect.signature); return (accepted, dropped_names)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}, sorted(wanted)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(wanted), []
    ok = {k: v for k, v in wanted.items() if k in params}
    return ok, sorted(set(wanted) - set(ok))


def check_runtime_availability() -> dict[str, str | bool]:
    info: dict[str, str | bool] = {"newton_installed": False, "warp_installed": False, "cuda_available": False}
    try:
        import warp as wp  # type: ignore

        info["warp_installed"] = True
        info["warp_version"] = getattr(wp, "__version__", "unknown")
        info["cuda_available"] = bool(wp.is_cuda_available()) if hasattr(wp, "is_cuda_available") else False
    except Exception:
        pass
    try:
        import newton  # type: ignore

        info["newton_installed"] = True
        info["newton_version"] = getattr(newton, "__version__", "unknown")
    except Exception:
        pass
    return info


def build_virtual_thickness_edges(V: np.ndarray, F: np.ndarray, max_dist_m: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """Pair opposite-facing triangles (n_a . n_b < -0.7) to create virtual thickness edges E_VE."""
    from scipy.spatial import cKDTree  # type: ignore

    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    centers = (A + B + C) / 3.0
    n_raw = np.cross(B - A, C - A)
    nrm = n_raw / np.maximum(np.linalg.norm(n_raw, axis=1, keepdims=True), 1e-12)
    tree = cKDTree(centers)
    dists, nn = tree.query(centers, k=min(8, len(centers)))
    pairs = []
    rest_lens = []
    for i in range(len(centers)):
        for j_idx in range(1, nn.shape[1]):
            j = int(nn[i, j_idx])
            if j <= i or float(dists[i, j_idx]) > max_dist_m:
                continue
            if float(np.dot(nrm[i], nrm[j])) < -0.7:
                va, vb = int(F[i, 0]), int(F[j, 0])
                if va != vb:
                    pairs.append((va, vb))
                    rest_lens.append(float(np.linalg.norm(V[va] - V[vb])))
                break
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float64)
    return np.asarray(pairs, dtype=np.int64), np.asarray(rest_lens, dtype=np.float64)


def precondition_two_phase_untangle(
    V_garment: np.ndarray,
    F_garment: np.ndarray,
    V_body: np.ndarray,
    F_body: np.ndarray,
    eps_body_m: float = 0.003,
    phase1_steps: int = 150,
    phase2_steps: int = 150,
    V_rest: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Two-phase L_VE -> L_body untangling so t=0 enters SolverVBD / pyuipc with GWN > 0.5 == 0."""
    from scipy.spatial import cKDTree  # type: ignore

    V = np.asarray(V_garment, dtype=np.float64).copy()
    F = np.asarray(F_garment, dtype=np.int64)
    V_b = np.asarray(V_body, dtype=np.float64)
    F_b = np.asarray(F_body, dtype=np.int64)
    V_ref = np.asarray(V_rest, dtype=np.float64) if V_rest is not None else V

    edges = np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    L_rest = np.linalg.norm(V_ref[edges[:, 0]] - V_ref[edges[:, 1]], axis=1)
    ve_edges, ve_rest = build_virtual_thickness_edges(V, F)

    # Phase 1: Virtual-edge wall preservation + edge-length maintenance (body collision disabled)
    step_lr = 0.15
    for _ in range(int(phase1_steps)):
        grad = np.zeros_like(V)
        diff = V[edges[:, 0]] - V[edges[:, 1]]
        dist = np.maximum(np.linalg.norm(diff, axis=1, keepdims=True), 1e-9)
        f_edge = ((dist[:, 0] - L_rest) / np.maximum(L_rest, 1e-6))[:, None] * (diff / dist)
        np.add.at(grad, edges[:, 0], f_edge)
        np.add.at(grad, edges[:, 1], -f_edge)
        if len(ve_edges) > 0:
            vdiff = V[ve_edges[:, 0]] - V[ve_edges[:, 1]]
            vdist = np.maximum(np.linalg.norm(vdiff, axis=1, keepdims=True), 1e-9)
            f_ve = 0.5 * ((vdist[:, 0] - ve_rest) / np.maximum(ve_rest, 1e-6))[:, None] * (vdiff / vdist)
            np.add.at(grad, ve_edges[:, 0], f_ve)
            np.add.at(grad, ve_edges[:, 1], -f_ve)
        V -= step_lr * np.clip(grad, -0.005, 0.005)

    # Phase 2: Swap L_VE for 3 mm cubic body-clearance barrier L_body
    b_normals = np.zeros_like(V_b)
    ab = V_b[F_b[:, 1]] - V_b[F_b[:, 0]]
    ac = V_b[F_b[:, 2]] - V_b[F_b[:, 0]]
    fn = np.cross(ab, ac)
    for k in range(3):
        np.add.at(b_normals, F_b[:, k], fn)
    b_normals /= np.maximum(np.linalg.norm(b_normals, axis=1, keepdims=True), 1e-12)
    tree = cKDTree(V_b)

    for _ in range(int(phase2_steps)):
        _, nn = tree.query(V)
        n_out = b_normals[nn]
        signed = np.einsum("ij,ij->i", V - V_b[nn], n_out)
        viol = signed < eps_body_m
        if not np.any(viol):
            break
        push = (eps_body_m - signed[viol])[:, None] * n_out[viol]
        V[viol] += 0.5 * push

    _, nn_final = tree.query(V)
    signed_final = np.einsum("ij,ij->i", V - V_b[nn_final], b_normals[nn_final])
    return V, {
        "ve_pairs": int(len(ve_edges)),
        "phase1_steps": int(phase1_steps),
        "phase2_steps": int(phase2_steps),
        "post_min_signed_mm": round(float(signed_final.min() * 1000.0), 3),
        "post_penetrating_approx": int((signed_final < 0.0).sum()),
    }


def solve_cloth_drape(
    V_garment_init: np.ndarray,
    F_garment: np.ndarray,
    V_body: np.ndarray,
    F_body: np.ndarray,
    pinned_mask: np.ndarray,
    cfg: NewtonClothConfig = NewtonClothConfig(),
    n_frames: int = 12,
    V_garment_rest: np.ndarray | None = None,
    extra_collider_meshes: Sequence[tuple[np.ndarray, np.ndarray]] = (),
) -> dict:
    """Run SolverVBD drape on Colab L4 (100x cm scaling -> solve -> 0.01x m); raises if Newton absent.

    `extra_collider_meshes`: already-solved inner cloth shells, each added as its own open shape mesh (never fan-capped
    or stacked into the body). The layer order itself is guaranteed afterwards by the LayerCollider PBD pass."""
    status = check_runtime_availability()
    if not (status["newton_installed"] and status["warp_installed"]):
        raise RuntimeError(
            "newton-physics/newton and warp-lang are Colab L4 runtime packages (not installed on local CPU host). "
            "Stage via COLAB_MEGA_SPRINT/ or install `newton==1.8.0` + `warp-lang==1.18.0` on Linux L4 CUDA."
        )
    if cfg.solver_kind == "style3d":
        raise ValueError("SolverStyle3D is blocked due to newton#4930 (max_neighbor=32 underarm drop); use solver_kind='vbd'.")

    import newton  # type: ignore
    import newton.solvers  # type: ignore
    from newton import Mesh, ParticleFlags  # type: ignore
    import warp as wp  # type: ignore

    wp.init()
    s = float(cfg.cm_scale)
    inv_s = 1.0 / s

    V_pre, pre_info = precondition_two_phase_untangle(
        V_garment_init, F_garment, V_body, F_body, eps_body_m=cfg.body_margin_m, V_rest=V_garment_rest
    )

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    V_garment_cm = (np.asarray(V_pre, dtype=np.float64) * s).astype(np.float32)
    F_garment_i32 = np.asarray(F_garment, dtype=np.int32).reshape(-1)
    V_body_cm = (np.asarray(V_body, dtype=np.float64) * s).astype(np.float32)
    F_body_i32 = np.asarray(F_body, dtype=np.int32).reshape(-1)

    # Use 0.015 cm (0.15 mm) radius so 0.21 mm collar/hem micro-edges do not self-overlap at t=0;
    # 3.0 mm body clearance is enforced by precondition_two_phase_untangle() and post-solve PBD sweeps.
    cloth_particle_radius_cm = 0.015
    up = cm_unit_params(cfg)
    cloth_kwargs, dropped = _filter_kwargs(builder.add_cloth_mesh, {
        "pos": wp.vec3(0.0, 0.0, 0.0),
        "rot": wp.quat_identity(),
        "vel": wp.vec3(0.0, 0.0, 0.0),
        "vertices": V_garment_cm.tolist(),
        "indices": F_garment_i32.tolist(),
        "density": up["density"],
        "scale": 1.0,
        "particle_radius": cloth_particle_radius_cm,
        "tri_ke": up["tri_ke"],
        "edge_ke": up["edge_ke"],
    })
    missing_core = {"vertices", "indices"} & set(dropped)
    if missing_core:
        raise RuntimeError(f"newton add_cloth_mesh signature lacks {sorted(missing_core)}; inspect and update the template")
    builder.add_cloth_mesh(**cloth_kwargs)
    # Colliders are STATIC world shapes (body=-1). Until the 2026-11-23 canary they were attached with body=builder.add_body(),
    # i.e. as free dynamic rigid bodies: the mannequin fell under gravity with the cloth, and in pass B the mannequin and the
    # solved inner shell were two overlapping rigid bodies (131 body-body contacts > Newton's per-body list of 64 -> the
    # "Per-body rigid contact buffer overflowed" warning). Static shapes have no rigid DOFs, so there are no body-body contacts.
    builder.add_shape_mesh(
        body=-1,
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        mesh=Mesh(V_body_cm, F_body_i32),
    )
    for V_sh, F_sh in extra_collider_meshes:
        builder.add_shape_mesh(
            body=-1,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            mesh=Mesh((np.asarray(V_sh, dtype=np.float64) * s).astype(np.float32), np.asarray(F_sh, dtype=np.int32).reshape(-1)),
        )
    builder.color()
    model = builder.finalize()
    model_body_count = int(getattr(model, "body_count", 0) or 0)

    pin_idx = np.flatnonzero(np.asarray(pinned_mask, dtype=bool))
    if len(pin_idx) > 0:
        flags = model.particle_flags.numpy()
        flags[pin_idx] = flags[pin_idx] & ~int(ParticleFlags.ACTIVE)
        model.particle_flags = wp.array(flags, dtype=model.particle_flags.dtype, device=model.device)

    model.soft_contact_ke = up["soft_contact_ke"]
    model.soft_contact_kd = up["soft_contact_kd"]
    # Body-particle contact uses 0.5 * (soft_contact_ke + shape_material_ke): pin the shape side so it is exactly ours.
    pinned_shape_material = False
    n_sh = int(model.shape_material_ke.shape[0]) if getattr(model, "shape_material_ke", None) is not None else 0
    if n_sh > 0:
        model.shape_material_ke = wp.array(np.full(n_sh, up["shape_material_ke"], dtype=np.float32), dtype=float, device=model.device)
        model.shape_material_kd = wp.array(np.full(n_sh, up["shape_material_kd"], dtype=np.float32), dtype=float, device=model.device)
        pinned_shape_material = True
    model.soft_contact_mu = float(cfg.body_friction)
    model.set_gravity((0.0, 0.0, up["gravity_z"]))

    vbd_kwargs, vbd_dropped = _filter_kwargs(newton.solvers.SolverVBD.__init__, {
        "model": model,
        "iterations": int(cfg.iterations),
        "rigid_body_contact_buffer_size": int(cfg.rigid_body_contact_buffer_size),
        "rigid_body_particle_contact_buffer_size": int(cfg.rigid_body_particle_contact_buffer_size),
    })
    solver = newton.solvers.SolverVBD(**vbd_kwargs)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()

    sim_dt = float(cfg.dt) / max(1, int(cfg.substeps))
    n_cloth_verts = len(V_garment_cm)
    V_last_finite = np.asarray(V_pre, dtype=np.float64).copy()
    frame_max_disp_m: list[float] = []      # per frame: max |x - x_pre| (telemetry; > 0.05 m or non-finite stops the drape)
    frame_step_max_m: list[float] = []      # per frame: max |x_frame - x_prev_frame| (jitter indicator)
    V_prev_frame = V_last_finite.copy()
    nan_fallback_frame: int | None = None
    ckpt_frames = {1, 20, 40, 60, 80, 100, 120}
    where: dict[int, dict[str, Any]] = {}    # frame -> max-displacement / max-step vertex (index, position)
    stop_info: dict[str, Any] | None = None
    overflow = {"body_body_frames": 0, "body_body_max": 0, "body_particle_frames": 0, "body_particle_max": 0,
                "soft_contact_max_count": 0, "soft_contact_capacity": None, "soft_contact_full_frames": 0}

    def _arr_max(name: str) -> int:
        a = getattr(solver, name, None)
        try:
            return int(a.numpy().max()) if a is not None and a.shape[0] > 0 else 0
        except Exception:  # noqa: BLE001
            return 0

    for frame_idx in range(int(n_frames)):
        collision_pipeline.collide(state_0, contacts)
        try:   # soft (particle-shape) contacts found vs the pipeline's capacity: equal means some were dropped
            cnt = int(contacts.soft_contact_count.numpy()[0])
            cap = int(getattr(contacts, "soft_contact_max", 0) or 0)
            overflow["soft_contact_max_count"] = max(overflow["soft_contact_max_count"], cnt)
            overflow["soft_contact_capacity"] = cap or None
            overflow["soft_contact_full_frames"] += int(cap > 0 and cnt >= cap)
        except Exception:  # noqa: BLE001
            pass
        for _sub in range(int(cfg.substeps)):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0
        bb = _arr_max("body_body_contact_overflow_max")
        bp = _arr_max("body_particle_contact_overflow_max")
        bb_cap = int(getattr(solver, "body_body_contact_buffer_pre_alloc", 0) or 0)
        bp_cap = int(getattr(solver, "body_particle_contact_buffer_pre_alloc", 0) or 0)
        overflow["body_body_max"] = max(overflow["body_body_max"], bb)
        overflow["body_particle_max"] = max(overflow["body_particle_max"], bp)
        overflow["body_body_frames"] += int(bb > bb_cap > 0 or (bb > 0 and bb_cap == 0))
        overflow["body_particle_frames"] += int(bp > bp_cap > 0 or (bp > 0 and bp_cap == 0))
        # Per-frame velocity damping (0.90) to prevent energy accumulation
        qd_np = state_0.particle_qd.numpy()
        if np.isfinite(qd_np).all():
            state_0.particle_qd = wp.array((qd_np * 0.90).astype(np.float32), dtype=wp.vec3, device=model.device)
        V_cur = state_0.particle_q.numpy()[:n_cloth_verts].astype(np.float64) * inv_s
        finite = bool(np.isfinite(V_cur).all())
        if finite:
            d_vec = np.linalg.norm(V_cur - V_pre, axis=1)
            s_vec = np.linalg.norm(V_cur - V_prev_frame, axis=1)
            i_d, i_s = int(np.argmax(d_vec)), int(np.argmax(s_vec))
            max_disp, max_step = float(d_vec[i_d]), float(s_vec[i_s])
        else:
            i_d = i_s = -1
            max_disp = max_step = float("inf")
        frame_max_disp_m.append(round(max_disp, 6))
        frame_step_max_m.append(round(max_step, 6))
        frame_idx1 = frame_idx + 1
        if frame_idx1 in ckpt_frames or not finite or max_disp > float(cfg.max_drape_disp_m) or max_step > float(cfg.max_frame_step_m):
            where[frame_idx1] = {
                "disp_vertex": {"i": i_d, "xyz_m": [round(float(x), 4) for x in V_cur[i_d]] if finite else None,
                                "start_xyz_m": [round(float(x), 4) for x in V_pre[i_d]] if finite else None},
                "step_vertex": {"i": i_s, "xyz_m": [round(float(x), 4) for x in V_cur[i_s]] if finite else None},
            }
            if frame_idx1 in ckpt_frames and finite:
                w = where[frame_idx1]
                print(f"[SolverVBD ckpt f{frame_idx1}] disp {max_disp * 1000.0:.2f} mm @v{i_d} {tuple(w['disp_vertex']['xyz_m'])} | "
                      f"step {max_step * 1000.0:.3f} mm @v{i_s} {tuple(w['step_vertex']['xyz_m'])}", flush=True)
        V_prev_frame = V_cur.copy() if finite else V_prev_frame
        reason = None
        if not finite:
            reason = "non-finite positions"
        elif max_step > float(cfg.max_frame_step_m):
            reason = f"runaway: max_step {max_step * 1000.0:.2f} mm/frame > {float(cfg.max_frame_step_m) * 1000.0:.1f} mm/frame"
        elif max_disp > float(cfg.max_drape_disp_m):
            reason = f"displacement: max_disp {max_disp * 1000.0:.2f} mm > {float(cfg.max_drape_disp_m) * 1000.0:.0f} mm"
        if reason is None:
            V_last_finite = V_cur
        else:
            nan_fallback_frame = frame_idx
            stop_info = {"frame": frame_idx1, "reason": reason, **where.get(frame_idx1, {})}
            w = where.get(frame_idx1, {})
            print(f"[SolverVBD] drape stopped at frame {frame_idx1}: {reason}; disp vertex {w.get('disp_vertex')}; "
                  f"step vertex {w.get('step_vertex')}; using last bounded state", flush=True)
            break

    V_out = V_last_finite
    qd_final = state_0.particle_qd.numpy()[:n_cloth_verts].astype(np.float64) * inv_s
    v_max_final = float(np.max(np.linalg.norm(qd_final, axis=1))) if np.isfinite(qd_final).all() else float("inf")
    checkpoints = [f for f in (1, 20, 40, 60, 80, 100, 120) if f <= len(frame_max_disp_m)]
    if len(frame_max_disp_m) and len(frame_max_disp_m) not in checkpoints:
        checkpoints.append(len(frame_max_disp_m))
    settle = {f"f{f}": {"disp_mm": round(frame_max_disp_m[f - 1] * 1000.0, 3), "step_mm": round(frame_step_max_m[f - 1] * 1000.0, 3),
                        **where.get(f, {})}
              for f in checkpoints}
    settle_line = " | ".join(f"f{f} disp {v['disp_mm']:.2f} step {v['step_mm']:.3f}" for f, v in
                             ((int(k[1:]), v) for k, v in settle.items()))
    print(f"[SolverVBD settle] {n_cloth_verts} verts, sim {len(frame_max_disp_m) * float(cfg.dt):.3f} s | {settle_line} | "
          f"final dx_max {frame_step_max_m[-1] * 1000.0 if frame_step_max_m else float('nan'):.3f} mm, v_max {v_max_final:.4f} m/s", flush=True)
    print(f"[SolverVBD contacts] bodies={model_body_count} body_body_overflow_frames={overflow['body_body_frames']} "
          f"body_particle_overflow_frames={overflow['body_particle_frames']} soft_contacts_max={overflow['soft_contact_max_count']}"
          f"/{overflow['soft_contact_capacity']} full_frames={overflow['soft_contact_full_frames']}", flush=True)
    return {
        "solver": "newton.solvers.SolverVBD",
        "sim_time_s": round(len(frame_max_disp_m) * float(cfg.dt), 6),
        "substep_dt_s": sim_dt,
        "settle_checkpoints": settle,
        "stop": stop_info,
        "guards": {"max_drape_disp_m": float(cfg.max_drape_disp_m), "max_frame_step_m": float(cfg.max_frame_step_m)},
        "final_dx_max_mm": round(frame_step_max_m[-1] * 1000.0, 4) if frame_step_max_m else None,
        "final_v_max_m_s": round(v_max_final, 6),
        "model_body_count": model_body_count,
        "contact_overflow": overflow,
        "solver_vbd_kwargs_passed": sorted(k for k in vbd_kwargs if k != "model"),
        "solver_vbd_kwargs_dropped": vbd_dropped,
        "cm_scale": s,
        "preconditioner": pre_info,
        "n_frames": int(n_frames),
        "substeps": int(cfg.substeps),
        "iterations": int(cfg.iterations),
        "pinned_count": int(len(pin_idx)),
        "nan_fallback_frame": nan_fallback_frame,
        "frame_max_disp_m": frame_max_disp_m,
        "frame_step_max_m": frame_step_max_m,
        "frames_completed": int(len(frame_max_disp_m)),
        "cm_unit_params": up,
        "shape_material_pinned": pinned_shape_material,
        "add_cloth_mesh_kwargs_passed": sorted(k for k in cloth_kwargs if k not in ("vertices", "indices")),
        "add_cloth_mesh_kwargs_dropped": dropped,
        "extra_collider_meshes": int(len(extra_collider_meshes)),
        "vertices_out": V_out,
    }


def _kernels():
    """canon_geometry_kernels: staged next to this file on Colab; locally it lives in maximus_engine/colab_stage."""
    try:
        import canon_geometry_kernels as K  # type: ignore
        return K
    except ImportError:
        import sys
        from pathlib import Path

        here = Path(__file__).resolve()
        for up in here.parents[1:4]:
            cand = up / "maximus_engine" / "colab_stage"
            if (cand / "canon_geometry_kernels.py").is_file():
                sys.path.insert(0, str(cand))
                import canon_geometry_kernels as K  # type: ignore
                return K
        raise


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class _TriSurface:
    """Closest-point queries on a triangle mesh (k nearest face centroids, exact point-triangle distance)."""

    def __init__(self, V: np.ndarray, F: np.ndarray, k: int = 16):
        from scipy.spatial import cKDTree  # type: ignore

        self.V = np.asarray(V, dtype=np.float64)
        self.F = np.asarray(F, dtype=np.int64)
        self.A, self.B, self.C = self.V[self.F[:, 0]], self.V[self.F[:, 1]], self.V[self.F[:, 2]]
        fn = np.cross(self.B - self.A, self.C - self.A)
        self.fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-15)
        self.cen = (self.A + self.B + self.C) / 3.0
        self.tree = cKDTree(self.cen)
        self.k = int(min(k, len(self.F)))

    def closest(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        K = _kernels()
        P = np.asarray(P, dtype=np.float64)
        _, idx = self.tree.query(P, k=self.k)
        idx = np.asarray(idx).reshape(len(P), self.k)
        best = np.full(len(P), np.inf)
        cp = np.zeros_like(P)
        fbest = np.zeros(len(P), dtype=np.int64)
        for c in range(self.k):
            f = idx[:, c]
            q = K._closest_point_on_triangles(P, self.A[f], self.B[f], self.C[f])
            d = np.linalg.norm(P - q, axis=1)
            better = d < best
            best = np.where(better, d, best)
            cp = np.where(better[:, None], q, cp)
            fbest = np.where(better, f, fbest)
        return best, cp, fbest


def _boundary_edge_flags(F: np.ndarray) -> np.ndarray:
    """(m, 3) bool: is edge (F[:,k], F[:,(k+1)%3]) a boundary edge (used by exactly one face)."""
    F = np.asarray(F, dtype=np.int64)
    e = np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1).reshape(-1, 2)
    key = np.sort(e, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return (cnt[inv.reshape(-1)] == 1).reshape(-1, 3)


def _face_adjacency(F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairs of faces sharing an edge: (face_a, face_b, same_direction) where same_direction means both faces traverse the
    shared edge u->v in the same direction (inconsistent winding)."""
    F = np.asarray(F, dtype=np.int64)
    m = len(F)
    e = np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1).reshape(-1, 2)
    fid = np.repeat(np.arange(m), 3)
    key = np.sort(e, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    ks, es, fs = key[order], e[order], fid[order]
    same_key = np.all(ks[1:] == ks[:-1], axis=1)
    i = np.flatnonzero(same_key)
    return fs[i], fs[i + 1], np.all(es[i] == es[i + 1], axis=1)


def _face_components(F: np.ndarray) -> np.ndarray:
    """Connected-component label per face (faces joined through shared edges)."""
    from scipy.sparse import coo_matrix  # type: ignore
    from scipy.sparse.csgraph import connected_components  # type: ignore

    a, b, _ = _face_adjacency(F)
    m = len(F)
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(m, m))
    return connected_components(g, directed=False)[1]


def _consistent_face_signs(F: np.ndarray) -> np.ndarray:
    """+1/-1 per face so that flipping the -1 faces gives consistent winding within each component (BFS). Non-orientable
    regions keep the first sign reached."""
    from collections import deque

    a, b, same = _face_adjacency(F)
    m = len(F)
    nbrs: list[list[tuple[int, bool]]] = [[] for _ in range(m)]
    for x, y, s in zip(a.tolist(), b.tolist(), same.tolist()):
        nbrs[x].append((y, s))
        nbrs[y].append((x, s))
    sign = np.zeros(m, dtype=np.int8)
    for start in range(m):
        if sign[start]:
            continue
        sign[start] = 1
        dq = deque([start])
        while dq:
            f = dq.popleft()
            for g, s in nbrs[f]:
                want = -sign[f] if s else sign[f]
                if not sign[g]:
                    sign[g] = want
                    dq.append(g)
    return sign.astype(np.float64)


def _point_segment_dist(P: np.ndarray, S0: np.ndarray, S1: np.ndarray) -> np.ndarray:
    d = S1 - S0
    t = np.clip(np.einsum("ij,ij->i", P - S0, d) / np.maximum(np.einsum("ij,ij->i", d, d), 1e-30), 0.0, 1.0)
    return np.linalg.norm(P - (S0 + t[:, None] * d), axis=1)


class LayerCollider:
    """Per-collider signed-distance union for layered cloth (spec §3).

    s_union(p) = min( s_body(p) - eps_body,  min_r( s_r(p) - margin_r ) ),  n_union = outward normal of the winning term.

    Body: closed mesh; sign from the closest face (fast) or the exact float64 generalised winding number (exact=True).
    Inner shell r: open cloth sheet with thickness t_r; s_r = sign((p - q_r) . n_r) * |p - q_r| - t_r / 2, with face normals
    oriented away from the body. When q_r lies on the shell's open boundary, repulsion fades out with the lateral distance of p
    beyond that boundary (full at 0, none past `taper_m` = 8 mm), so outer cloth over bare skin is not shoved sideways by a strap
    or hem. Points more than `wrong_side_guard_m` behind a shell are treated as not interacting with it (they belong to a
    different part of the body, e.g. across the torso), so the body term decides for them.
    """

    def __init__(
        self,
        V_body: np.ndarray,
        F_body: np.ndarray,
        eps_body: float = 0.0025,
        shells: Sequence[tuple[np.ndarray, np.ndarray, float, float]] = (),
        taper_m: float = 0.008,
        wrong_side_guard_m: float = 0.02,
        k: int = 16,
    ):
        self.body = _TriSurface(V_body, F_body, k=k)
        self.eps_body = float(eps_body)
        self.taper_m = float(taper_m)
        self.guard = float(wrong_side_guard_m)
        self.shells: list[dict[str, Any]] = []
        for V_r, F_r, t_r, m_r in shells:
            surf = _TriSurface(V_r, F_r, k=k)
            # Orient the shell AS A WHOLE away from the body: make face winding consistent across each connected component
            # (BFS over shared edges), then one sign per component by area-weighted majority vote against the body.
            # (A per-face "away from the closest body point" rule is wrong wherever a loose shell is nearer a limb than its
            # own torso: the BS-TANK-001 side panels lie between the torso and the hanging arm and got flipped inward.)
            sgn = _consistent_face_signs(np.asarray(F_r, dtype=np.int64))
            fn = surf.fn * sgn[:, None]
            d_b, q_b, f_b = self.body.closest(surf.cen)
            ref = np.where((d_b > 1e-6)[:, None], surf.cen - q_b, self.body.fn[f_b])
            area = 0.5 * np.linalg.norm(np.cross(surf.B - surf.A, surf.C - surf.A), axis=1)
            vote = np.sign(np.einsum("ij,ij->i", fn, ref)) * area
            comp = _face_components(np.asarray(F_r, dtype=np.int64))
            comp_vote = np.zeros(int(comp.max()) + 1)
            np.add.at(comp_vote, comp, vote)
            surf.fn = fn * np.where(comp_vote[comp] < 0.0, -1.0, 1.0)[:, None]
            self.shells.append({"surf": surf, "t": float(t_r), "m": float(m_r), "bflag": _boundary_edge_flags(F_r)})

    def body_signed(self, P: np.ndarray, exact: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(signed distance, outward unit normal, closest point) to the body; exact uses the float64 winding number."""
        P = np.asarray(P, dtype=np.float64)
        d, cp, f = self.body.closest(P)
        fn = self.body.fn[f]
        side = np.einsum("ij,ij->i", P - cp, fn)
        inside = side < 0.0
        if exact:
            K = _kernels()
            # the closest-face sign is only unreliable right at the surface (edges/vertices, folds) or where it already says
            # "inside"; evaluate the float64 winding number for those points only (it is O(points x body faces))
            cand = np.flatnonzero((d < 0.004) | inside)
            inside = inside.copy()
            if len(cand):
                inside[cand] = K.generalized_winding_number_float64(P[cand], self.body.V, self.body.F, device="auto") > 0.5
        s = np.where(inside, -d, d)
        dirv = (P - cp) * np.where(inside, -1.0, 1.0)[:, None]
        nrm = np.where((d > 1e-6)[:, None], dirv / np.maximum(d, 1e-15)[:, None], fn)
        return s, nrm, cp

    def shell_signed(self, r: int, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(thickness-adjusted signed distance with taper/guard applied, outward unit normal, interaction weight) for shell r."""
        sh = self.shells[r]
        surf = sh["surf"]
        P = np.asarray(P, dtype=np.float64)
        d, q, f = surf.closest(P)
        n = surf.fn[f]
        v = P - q
        along = np.einsum("ij,ij->i", v, n)
        sgn = np.where(along < 0.0, -1.0, 1.0)
        s = sgn * d - 0.5 * sh["t"]
        # on-boundary test: q within 1 micron of one of the winning face's boundary edges
        Fi = surf.F[f]
        on_b = np.zeros(len(P), dtype=bool)
        for e in range(3):
            fl = sh["bflag"][f, e]
            if np.any(fl):
                a = surf.V[Fi[fl, e]]
                b = surf.V[Fi[fl, (e + 1) % 3]]
                on_b[fl] |= _point_segment_dist(q[fl], a, b) < 1e-6
        lateral = np.linalg.norm(v - along[:, None] * n, axis=1)
        w = np.where(on_b, 1.0 - _smoothstep(lateral / max(self.taper_m, 1e-9)), 1.0)
        w = np.where(s < -(0.5 * sh["t"] + self.guard), 0.0, w)
        nrm = np.where((d > 1e-6)[:, None], v * sgn[:, None] / np.maximum(d, 1e-15)[:, None], n)
        return s, nrm, w

    def query(self, P: np.ndarray, exact: bool = False, margin_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(s_union, n_union, owner) with owner = -1 for the body, r for shell r. margin_scale scales eps_body and every margin_r."""
        s_b, n_b, _ = self.body_signed(P, exact=exact)
        s_u = s_b - margin_scale * self.eps_body
        n_u = n_b
        owner = np.full(len(s_u), -1, dtype=np.int64)
        for r, sh in enumerate(self.shells):
            s_r, n_r, w = self.shell_signed(r, P)
            s_eff = np.where(w > 0.0, s_r - margin_scale * sh["m"] + (1.0 - w) * 0.05, np.inf)
            better = s_eff < s_u
            s_u = np.where(better, s_eff, s_u)
            n_u = np.where(better[:, None], n_r, n_u)
            owner = np.where(better, r, owner)
        return s_u, n_u, owner


def pbd_strain_limit_and_tangent_clearance(
    V: np.ndarray,
    l_bar: np.ndarray,
    edges: np.ndarray,
    V_body: np.ndarray,
    F_body: np.ndarray,
    l_min: float = 0.92,
    l_max: float = 1.12,
    eps_body_m: float = 0.0032,
    n_sweeps: int = 80,
    collider: LayerCollider | None = None,
    pinned_mask: np.ndarray | None = None,
    stretch_target: float = 1.05,
    near_m: float = 0.0025,
    step_cap_frac: float = 0.25,
    final_margin_scale: float = 0.2,
    edge_probes: bool = True,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Tangential PBD strain limiting + clearance against a LayerCollider (spec §3).

    Edge lengths are pulled into [l_min, stretch_target] x rest (vectorised weighted Jacobi). For a vertex u near any collider
    (s_union[u] < near_m) its correction is projected onto the outward half-space, dx_u -= min(0, dx_u . n_u) n_u, so an edge
    clamp may slide or lift it but never pull it inside. Push-out steps are capped per vertex at step_cap_frac x the shortest
    incident rest edge. The last phase uses the exact float64 winding sign and relaxes the margins to final_margin_scale x their
    targets (the hard requirement is "outside", the margin is the aim), then hard-clamps edges above l_max.

    Backward compatible: with collider=None a body-only collider with eps_body_m is built (old call sites unchanged).
    """
    X = np.asarray(V, dtype=np.float64).copy()
    E = np.asarray(edges, dtype=np.int64)
    Lo = np.maximum(np.asarray(l_bar, dtype=np.float64), 1e-6)
    coll = collider if collider is not None else LayerCollider(V_body, F_body, eps_body=eps_body_m)
    free = np.ones(len(X), dtype=bool) if pinned_mask is None else ~np.asarray(pinned_mask, dtype=bool)
    deg = np.bincount(E.ravel(), minlength=len(X)).astype(np.float64)
    inv_deg = (0.95 / np.maximum(deg, 1.0))[:, None]
    min_edge = np.full(len(X), np.inf)
    np.minimum.at(min_edge, E[:, 0], Lo)
    np.minimum.at(min_edge, E[:, 1], Lo)
    min_edge = np.where(np.isfinite(min_edge), min_edge, 0.01)
    cap = step_cap_frac * min_edge
    info: dict[str, Any] = {"pushes": 0, "max_push_step_mm": 0.0}

    def strain_step(hi: float, s: np.ndarray, n: np.ndarray, iters: int) -> None:
        near = s < near_m
        for _ in range(iters):
            diff = X[E[:, 0]] - X[E[:, 1]]
            dist = np.maximum(np.linalg.norm(diff, axis=1), 1e-12)
            target = np.clip(dist, l_min * Lo, hi * Lo)
            if np.allclose(target, dist, rtol=0.0, atol=1e-9):
                break
            corr = 0.5 * ((dist - target) / dist)[:, None] * diff
            dX = np.zeros_like(X)
            np.add.at(dX, E[:, 0], -corr)
            np.add.at(dX, E[:, 1], corr)
            dX *= inv_deg
            if np.any(near):
                inward = np.minimum(np.einsum("ij,ij->i", dX[near], n[near]), 0.0)
                dX[near] -= inward[:, None] * n[near]
            dX[~free] = 0.0
            X[:] += dX

    # Edge-midpoint probes: a long edge can span a ridge of an inner layer with both endpoints outside it (seen on the ringer over
    # the trouser waist: 19.5 mm edge, endpoints 2.5 mm out, middle through the trouser). Midpoints of edges longer than 4 mm are
    # tested too; a violating midpoint lifts both endpoints along its normal (same per-vertex step cap).
    E_probe = E[Lo > 0.004] if edge_probes else E[:0]
    cap_edge = np.minimum(cap[E_probe[:, 0]], cap[E_probe[:, 1]]) if len(E_probe) else np.zeros(0)
    info["edge_probes"] = int(len(E_probe))

    def push_out(exact: bool, margin_scale: float, rounds: int) -> np.ndarray:
        s = np.zeros(len(X))
        for _ in range(rounds):
            s, n, _ = coll.query(X, exact=exact, margin_scale=margin_scale)
            viol = (s < 0.0) & free
            mid_viol = np.zeros(0, dtype=bool)
            if len(E_probe):
                sm, nm, _ = coll.query(0.5 * (X[E_probe[:, 0]] + X[E_probe[:, 1]]), exact=False, margin_scale=margin_scale)
                mid_viol = sm < 0.0
            if not np.any(viol) and not np.any(mid_viol):
                break
            if np.any(viol):
                step = np.minimum(-s[viol], cap[viol])
                X[viol] += step[:, None] * n[viol]
                info["pushes"] += int(viol.sum())
                info["max_push_step_mm"] = max(info["max_push_step_mm"], float(step.max() * 1000.0))
            if np.any(mid_viol):
                ev = E_probe[mid_viol]
                st = np.minimum(-sm[mid_viol], cap_edge[mid_viol])[:, None] * nm[mid_viol]
                dX = np.zeros_like(X)
                cnt = np.zeros(len(X))
                for k in (0, 1):
                    np.add.at(dX, ev[:, k], st)
                    np.add.at(cnt, ev[:, k], 1.0)
                moved = (cnt > 0) & free
                X[moved] += dX[moved] / cnt[moved, None]
                info["edge_pushes"] = info.get("edge_pushes", 0) + int(mid_viol.sum())
        return s

    n_outer = max(6, int(n_sweeps) // 8)
    for _ in range(n_outer):
        s, n, _ = coll.query(X)
        strain_step(stretch_target, s, n, iters=40)
        push_out(exact=False, margin_scale=1.0, rounds=6)

    # exact-sign polish: relaxed margins, alternate strain + capped push until both hold
    for it in range(30):
        push_out(exact=False, margin_scale=final_margin_scale, rounds=8)
        push_out(exact=True, margin_scale=final_margin_scale, rounds=1)
        s, n, _ = coll.query(X, exact=True, margin_scale=final_margin_scale)
        diff = X[E[:, 0]] - X[E[:, 1]]
        r = np.linalg.norm(diff, axis=1) / Lo
        ok_clear = bool(np.all(s[free] >= 0.0))
        if ok_clear and len(E_probe):
            sm, _, _ = coll.query(0.5 * (X[E_probe[:, 0]] + X[E_probe[:, 1]]), exact=False, margin_scale=final_margin_scale)
            ok_clear = bool(np.all(sm >= 0.0))
        if ok_clear and float(np.percentile(r, 99)) <= 1.075 and float(r.max()) <= min(l_max, 1.145):
            info["polish_iters"] = it
            break
        strain_step(min(stretch_target, 1.06), s, n, iters=25)
    else:
        info["polish_iters"] = 30

    s, _, _ = coll.query(X, exact=True, margin_scale=0.0)
    r = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1) / Lo
    info.update({
        "inside_count": int((s < 0.0).sum()),
        "min_clear_mm": round(float(s.min() * 1000.0), 3),
        "stretch_p99": round(float(np.percentile(r, 99)), 4),
        "stretch_max": round(float(r.max()), 4),
        "max_push_step_mm": round(info["max_push_step_mm"], 4),
    })
    return (X, info) if return_info else X


@dataclass
class ClothLayer:
    """One cloth part for the N-layer solve (spec §2 schema)."""

    name: str
    V_rest: np.ndarray
    F: np.ndarray
    E: np.ndarray
    layer_rank: int = 1
    part_id: int = 1
    thickness_m: float = 0.0014
    pinned: np.ndarray | None = None
    closure_mode: str = "closed"
    V_init: np.ndarray | None = None
    V_solved: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)


def solve_layers_untangle_all(
    layers: Sequence[ClothLayer],
    V_body: np.ndarray,
    F_body: np.ndarray,
    eps_body_inner_m: float = 0.0025,
    eps_body_outer_m: float = 0.0030,
    layer_gap_m: float = 0.0025,
    cfg: NewtonClothConfig | None = None,
    use_vbd: bool | None = None,
    n_frames: int = 6,
    coupled_sweeps: int = 10,
    log: Any = print,
) -> list[ClothLayer]:
    """UntangleAll: solve layers inner-to-outer, each against LayerCollider(body, already-solved inner shells), then coupled sweeps.

    Initial state per layer: SolverVBD drape when Newton is installed (use_vbd None -> auto), else the NumPy two-phase
    preconditioner. A layer passed in with V_solved already set is kept as solved (it still takes part in the coupled sweeps). Coupled sweeps re-project every layer, in rank order, against the body (>= eps_body_inner_m, i.e. >= 2 mm)
    and the current positions of all lower-rank layers, with push steps capped at 0.25 x the shortest incident edge.
    """
    cfg = cfg or NewtonClothConfig()
    if use_vbd is None:
        st = check_runtime_availability()
        use_vbd = bool(st["newton_installed"] and st["warp_installed"])
    order = sorted(layers, key=lambda L: (int(L.layer_rank), int(L.part_id)))

    def collider_for(i: int) -> LayerCollider:
        L = order[i]
        inner = [S for S in order[:i] if int(S.layer_rank) < int(L.layer_rank)]
        eps = eps_body_inner_m if int(L.layer_rank) <= 1 else eps_body_outer_m
        return LayerCollider(V_body, F_body, eps_body=eps,
                             shells=[(S.V_solved, S.F, S.thickness_m, layer_gap_m) for S in inner])

    for i, L in enumerate(order):
        if L.V_solved is not None:          # already solved (e.g. an inner layer the outer one was sized around): keep it
            L.info.setdefault("init", {"solver": "presolved (kept)"})
            log(f"[UntangleAll] {L.name} rank={L.layer_rank} presolved, kept")
            continue
        coll = collider_for(i)
        inner_meshes = ([(S.V_solved, S.F) for S in order[:i] if int(S.layer_rank) < int(L.layer_rank)]
                        if cfg.vbd_inner_shell_colliders else [])   # default: SolverVBD sees the watertight body only
        if use_vbd:
            res = solve_cloth_drape(L.V_rest, L.F, V_body, F_body,
                                    L.pinned if L.pinned is not None else np.zeros(len(L.V_rest), bool),
                                    cfg=cfg, n_frames=n_frames, V_garment_rest=L.V_rest, extra_collider_meshes=inner_meshes)
            L.V_init = res["vertices_out"]
            L.info["init"] = {k: v for k, v in res.items() if k != "vertices_out"}
        else:
            L.V_init, pre = precondition_two_phase_untangle(L.V_rest, L.F, V_body, F_body, eps_body_m=coll.eps_body, V_rest=L.V_rest)
            L.info["init"] = {"solver": "numpy_two_phase_preconditioner (SolverVBD not installed)", **pre}
        l_bar = np.linalg.norm(L.V_rest[L.E[:, 0]] - L.V_rest[L.E[:, 1]], axis=1)
        L.V_solved, L.info["pbd"] = pbd_strain_limit_and_tangent_clearance(
            L.V_init, l_bar, L.E, V_body, F_body, collider=coll, return_info=True)
        log(f"[UntangleAll] {L.name} rank={L.layer_rank} shells={len(coll.shells)} pbd={L.info['pbd']}")

    for sweep in range(int(coupled_sweeps)):
        worst = 0.0
        for i, L in enumerate(order):
            coll = collider_for(i)
            s, _, _ = coll.query(L.V_solved, exact=False, margin_scale=0.2)
            worst = min(worst, float(s.min()))
            if s.min() >= 0.0:
                continue
            l_bar = np.linalg.norm(L.V_rest[L.E[:, 0]] - L.V_rest[L.E[:, 1]], axis=1)
            L.V_solved, L.info["pbd_coupled"] = pbd_strain_limit_and_tangent_clearance(
                L.V_solved, l_bar, L.E, V_body, F_body, collider=coll, n_sweeps=8, return_info=True)
        log(f"[UntangleAll] coupled sweep {sweep}: worst relaxed clearance {worst * 1000.0:.3f} mm")
        if worst >= 0.0:
            break
    return order


def build_vbd_plan_summary(cfg: NewtonClothConfig = NewtonClothConfig()) -> dict[str, str | int | float]:
    return {
        "solver": "newton.solvers.SolverVBD",
        "rejected_solver": "SolverStyle3D (newton#4930 max_neighbor=32 underarm seam drop)",
        "cm_scale_conditioning": cfg.cm_scale,
        "preconditioner": "two_phase_L_VE_to_L_body (150+150 steps, 3mm barrier)",
        "pin_mechanism": "ParticleFlags.ACTIVE bitmask clear on model.particle_flags",
        "substeps": cfg.substeps,
        "iterations": cfg.iterations,
        "body_margin_m": cfg.body_margin_m,
    }


if __name__ == "__main__":
    print(json.dumps({"runtime": check_runtime_availability(), "verified_api": build_vbd_plan_summary()}, indent=2))
