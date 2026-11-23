# Maximus garment physics and geometry checks

Python tools for preparing garments, draping them on a mannequin and measuring whether the result is acceptable.

## Main parts (`max_garment_eng/`)
- `01_catalog_ingest`: garment repair and preflight checks
- `02_alignment_and_collar`: collar alignment and rest-length adapters
- `03_gpu_cloth_solver`: Newton/Warp cloth template, libuipc IPC benchmark, Blender fallback
- `04_clearance_gates`: edge-stretch, float64 winding-number penetration and collar gates
- `05_morph_and_export`: size morphs and compact GLB export
- `06_splat_binding_and_relight`: experimental Gaussian-splat binding and packing

Other folders hold earlier garment-engine, collar, rhinestone placement, avatar and clearance-diagnostic work.

## Status
In progress. The gates are written as pass/fail checks (for example stretch p99 <= 1.08). One of nine catalogue garments, the ringer tee, has passed all of them (p99 1.076, 0 penetrations). Multi-layer GPU draping is still being tuned.

## About this repository
Published from a larger private workspace. Third-party code, large binaries and generated data are left out, so some scripts and tests need inputs that are not included here. Paths under `~/Desktop/...` are the author's defaults; change them for your machine.
