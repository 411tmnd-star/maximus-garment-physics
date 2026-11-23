"""Better Style heavy run: every heavy job in one go on a Colab L4. Called by RUN_ALL.ipynb (one cell, Run all).

Jobs (each independent; one failing does not stop the others):
  renders     Cycles on the GPU: camp shirts (FBL-008 seated-collar master), polo from the FBL-008 polo master (floating + mannequin),
              ringer (review only). Same render_sku.py the engine uses; only --device GPU is added.
  fbl003      Human-engine clearance fix: rest-pose push-out -> derived scene -> exact rest check -> walk re-capture -> 9-frame GPU
              clearance gate (base.clearance, limit 0) with a numpy cross-check. Never touches the original scene or the baseline capture.
  rhinestone  rhinestone_lab full suite (npm run test:all: geometry, Blender/bpy, formats, contracts, policy, collisions, browser).

  python run_all.py --repo /content/work/repo --results /content/drive/MyDrive/COLAB_RESULTS/heavy --bpy-python /content/py313/bin/python
Finished jobs (JOB_STATUS.json with status PASS) are skipped on a re-run unless --force.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, threading, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repo", type=Path, required=True); ap.add_argument("--results", type=Path, required=True)
ap.add_argument("--bpy-python", required=True, help="Python 3.13 with bpy==5.2.3")
ap.add_argument("--jobs", default="renders,fbl003,rhinestone"); ap.add_argument("--force", action="store_true")
ap.add_argument("--scratch", type=Path, default=Path("/content/heavy_out"))
A = ap.parse_args()
REPO = A.repo.resolve(); RES = A.results; PY = sys.executable; BPY = A.bpy_python
ENG = REPO / "02_engine_3d/isolated_clones/better_style_catalog_collar"; AE = REPO / "02_engine_3d/avatar_engine"
LAB = ENG / "rhinestone_lab"; CH = REPO / "02_engine_3d/colab_heavy"
EXT = REPO.parent / "external"                                     # files that live outside the repo (the human-engine scene)
CLEAR_FRAMES = "61,69,76,84,91,99,106,114,300"
LOCK = threading.Lock()


def say(job, msg):
    with LOCK: print(f"[{time.strftime('%H:%M:%S')}] [{job}] {msg}", flush=True)


def run(job, name, cmd, cwd, logdir, env=None):
    """run a command, stream to a log file, return exit code"""
    logdir.mkdir(parents=True, exist_ok=True); log = logdir / f"{name}.log"; t0 = time.time()
    say(job, f"START {name}: {' '.join(str(c) for c in cmd)}")
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.run([str(c) for c in cmd], cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})})
    say(job, f"{'OK ' if p.returncode == 0 else 'FAIL'} {name} (exit {p.returncode}, {time.time() - t0:.0f}s) log: {log.name}")
    if p.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        with LOCK: print("\n".join("    | " + l for l in tail), flush=True)
    return p.returncode


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ job: renders (GPU)
def job_renders(out):
    logs = out / "logs"; steps = []
    plan = [  # (name, manifest, extra args)  -- camp: the exact FBL-008 storefront set (tools/fbl008_rerender_camp.sh)
        ("camp001_gold", "configs/skus/BS-CAMP-001.json", ["--only", "gold"]),
        ("camp001_rest", "configs/skus/BS-CAMP-001.json", ["--only", "blue,green,ivory,taupe", "--views", "front,three_quarter"]),
        ("camp002_all", "configs/skus/BS-CAMP-002.json", []),
        ("polo_floating", "configs/skus/BS-POLO-001.json", []),
        ("polo_mannequin", str(CH / "manifests/BS-POLO-001_mannequin.json"), ["--hi-res"]),
        ("ringer_review", str(CH / "manifests/BS-RINGER-001_all.json"), []),
    ]
    for name, man, extra in plan:
        rc = run("renders", name, [BPY, "engine/render_sku.py", "--manifest", man, "--out-dir", out / name, "--device", "GPU", *extra], ENG, logs)
        lp = out / name / "render_log.json"; log = json.loads(lp.read_text()) if lp.exists() else {}
        devs = sorted({log.get("provenance", {}).get("cycles_device") or "?"})
        steps.append(dict(step=name, exit=rc, complete=bool(log.get("complete")), renders=len(log.get("renders", [])), cycles_device=devs))
    # contact sheet of everything rendered
    try:
        from PIL import Image
        ims = sorted(p for p in out.glob("*/*.png") if not p.stem.endswith("_alpha") and not p.name.startswith("_"))
        w, h, cols = 240, 300, 6; rows = max(1, -(-len(ims) // cols)); sheet = Image.new("RGB", (w * cols, h * rows), (243, 238, 231))
        for i, p in enumerate(ims): sheet.paste(Image.open(p).convert("RGB").resize((w, h)), ((i % cols) * w, (i // cols) * h))
        sheet.save(out / "contact_sheet.jpg", quality=88)
    except Exception as e: steps.append(dict(step="contact_sheet", error=str(e)))
    ok = all(s.get("exit") == 0 and s.get("complete") for s in steps if "exit" in s)
    return dict(status="PASS" if ok else "FAIL", steps=steps)


# ------------------------------------------------------------------ job: FBL-003 push-out + clearance gate
def job_fbl003(out):
    logs = out / "logs"; D = AE / "diagnostics/fbl_003"; steps = []
    scene = EXT / "dual_state_scene_copy.blend"; derived = out / "dual_state_scene_pushout.blend"; cap = out / "capture_pushout_walk.npz"
    chain = [
        ("1_pushout", [PY, D / "rest_pushout.py", "--rest", D / "rest_meshes.npz", "--out-dir", out, "--max-iter", "20"], D, {}),
        ("2_apply_derived_blend", [BPY, D / "apply_pushout_blend.py", "--", scene, out / "rest_fixed.npz", derived, D / "rest_meshes.npz"], D, {}),
        ("3_rest_check", [PY, D / "verify_rest.py", derived.with_suffix(".rest.npz"), out / "rest_check.json"], D, {}),
        ("4_capture_walk", [BPY, AE / "harness/capture_walk_blender.py", "--", derived, cap, "--clearance-frames", CLEAR_FRAMES], AE / "harness", {}),
        ("5_gpu_clearance", [PY, AE / "diagnostics/fbl_004/gpu_clearance.py", "--capture", cap, "--ref-dir", "none", "--device", "cuda", "--out", out / "clearance"],
         AE / "diagnostics/fbl_004", {}),
    ]
    for name, cmd, cwd, env in chain:
        rc = run("fbl003", name, cmd, cwd, logs, env); steps.append(dict(step=name, exit=rc))
        if rc != 0: break                                  # each step needs the one before it
    gate = None; sp = out / "clearance/summary.json"
    if sp.exists():
        s = json.loads(sp.read_text()); gate = dict(s.get("gate", {}), cpu_checks_all_match=s.get("cpu_checks_all_match"))
    rest = json.loads((out / "rest_check.json").read_text()) if (out / "rest_check.json").exists() else None
    status = "PASS" if gate and gate.get("status") == "PASS" and gate.get("cpu_checks_all_match") else "FAIL"
    return dict(status=status, steps=steps, rest_check=rest and rest.get("status"), gate=gate,
                note="Original scene and baseline capture are never modified; promoting the derived scene is the owner's decision.")


# ------------------------------------------------------------------ job: rhinestone full suite
def job_rhinestone(out):
    logs = out / "logs"; env = {"PATH": f"{Path(BPY).parent}:{os.environ['PATH']}", "PYTHONUTF8": "1"}
    before = {str(p.relative_to(LAB)): sha(p) for p in (LAB / "contracts").rglob("*") if p.is_file()}
    rc = run("rhinestone", "test_all", ["npm", "run", "test:all"], LAB, logs, env)
    after = {str(p.relative_to(LAB)): sha(p) for p in (LAB / "contracts").rglob("*") if p.is_file()}
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    for sub in ("out", "contracts"):
        if (LAB / sub).exists(): shutil.copytree(LAB / sub, out / sub, dirs_exist_ok=True)
    return dict(status="PASS" if rc == 0 else "FAIL", steps=[dict(step="npm run test:all", exit=rc)], contracts_changed_by_tests=changed,
                note="The suite rewrites contracts/ (known behaviour); changed files are listed, never applied automatically.")


JOBS = {"renders": job_renders, "fbl003": job_fbl003, "rhinestone": job_rhinestone}


def do(job):
    dst = RES / job; st = dst / "JOB_STATUS.json"
    if st.exists() and not A.force and json.loads(st.read_text()).get("status") == "PASS":
        say(job, "already PASS in results, skipped (use --force to redo)"); return json.loads(st.read_text())
    out = A.scratch / job
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True); t0 = time.time()
    try: res = JOBS[job](out)
    except Exception as e:
        import traceback; res = dict(status="ERROR", error=repr(e), traceback=traceback.format_exc())
    res.update(job=job, seconds=round(time.time() - t0), finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (out / "JOB_STATUS.json").write_text(json.dumps(res, indent=1))
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(out, dst)                              # results land on Drive only when the job is finished
    say(job, f"=== {res['status']} in {res['seconds']}s -> {dst}"); return res


if __name__ == "__main__":
    RES.mkdir(parents=True, exist_ok=True); todo = [j for j in A.jobs.split(",") if j]
    results = {}
    # GPU chain (renders then fbl003) runs in parallel with the CPU-only rhinestone suite
    def gpu_chain():
        for j in ("renders", "fbl003"):
            if j in todo: results[j] = do(j)
    th = threading.Thread(target=gpu_chain); th.start()
    if "rhinestone" in todo: results["rhinestone"] = do("rhinestone")
    th.join()
    staging = json.loads((REPO.parent / "STAGING_INFO.json").read_text()) if (REPO.parent / "STAGING_INFO.json").exists() else {}
    summary = dict(finished=time.strftime("%Y-%m-%dT%H:%M:%S"), staging=staging,
                   jobs={j: {k: v for k, v in r.items() if k in ("status", "seconds", "gate", "rest_check", "steps", "contracts_changed_by_tests", "error")} for j, r in results.items()})
    (RES / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=1))
    lines = [f"{sha(p)}  {p.relative_to(RES).as_posix()}" for p in sorted(RES.rglob("*")) if p.is_file() and p.name != "RESULTS_MANIFEST.sha256"]
    (RES / "RESULTS_MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    print("\n================ RUN SUMMARY ================")
    for j, r in results.items(): print(f"{j:11s} {r['status']:6s} {r.get('seconds', 0):6}s")
    print("Download My Drive/COLAB_RESULTS/heavy, then on the PC run: python 02_engine_3d/colab_heavy/apply_results.py <downloaded heavy folder>")
