"""Better Style heavy run: every heavy job in one go on a Colab L4. Called by RUN_ALL.ipynb (one cell, Run all).

Jobs (each independent; one failing does not stop the others):
  mannequin   Storefront mannequin catalogue: polo, camp x2, ringer, every colourway on the visible mannequin in M1 / M2 / M3, front /
              three-quarter (45 deg) / back, torso framing, 2400x3000 RGBA, each render's finish in render_log.json. Garments whose master
              quality gate fails are skipped (stay floating-only). Results go to COLAB_RESULTS/catalogue_mannequin_renders/<SKU>/<finish>/.
  renders     Cycles on the GPU: camp shirts (FBL-008 seated-collar master), polo floating from the FBL-008 polo master,
              ringer (review only). Same render_sku.py the engine uses; only --device GPU is added.
  fbl003      Human-engine clearance fix: rest-pose push-out -> derived scene -> exact rest check -> walk re-capture -> 9-frame GPU
              clearance gate (base.clearance, limit 0) with a numpy cross-check. Never touches the original scene or the baseline capture.
  rhinestone  rhinestone_lab full suite (npm run test:all: geometry, Blender/bpy, formats, contracts, policy, collisions, browser).

  python run_all.py --repo /content/work/repo --results /content/drive/MyDrive/COLAB_RESULTS/heavy --bpy-python /content/py313/bin/python
Finished jobs (JOB_STATUS.json with status PASS) are skipped on a re-run unless --force.
Re-render some SKUs only (e.g. after a master change):  ... --jobs mannequin,renders --skus BS-CAMP-001,BS-CAMP-002
  renders just those SKUs and merges them into the existing results (other SKUs untouched; JOB_STATUS/manifest rebuilt, a `merges` entry
  records what was replaced), so apply_results.py sees one complete folder.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, threading, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repo", type=Path, required=True); ap.add_argument("--results", type=Path, required=True)
ap.add_argument("--bpy-python", required=True, help="Python 3.13 with bpy==5.2.3")
ap.add_argument("--jobs", default="mannequin,renders,fbl003,rhinestone"); ap.add_argument("--force", action="store_true")
ap.add_argument("--scratch", type=Path, default=Path("/content/heavy_out"))
ap.add_argument("--skus", default="", help="comma list, e.g. BS-CAMP-001,BS-CAMP-002: the mannequin and renders jobs render only these SKUs and merge "
                                           "them into the existing results folder (other SKUs' folders and steps are kept as they are)")
A = ap.parse_args()
SKUS = [s for s in A.skus.split(",") if s]
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
RENDER_PLAN = [  # (name, sku, manifest, extra args)  -- camp: the exact FBL-008 storefront set (tools/fbl008_rerender_camp.sh)
    ("camp001_gold", "BS-CAMP-001", "configs/skus/BS-CAMP-001.json", ["--only", "gold"]),
    ("camp001_rest", "BS-CAMP-001", "configs/skus/BS-CAMP-001.json", ["--only", "blue,green,ivory,taupe", "--views", "front,three_quarter"]),
    ("camp002_all", "BS-CAMP-002", "configs/skus/BS-CAMP-002.json", []),
    ("polo_floating", "BS-POLO-001", "configs/skus/BS-POLO-001.json", []),
    # polo mannequin views moved to the `mannequin` job (all colourways x M1/M2/M3 x front / three-quarter / back)
    ("ringer_review", "BS-RINGER-001", str(CH / "manifests/BS-RINGER-001_all.json"), []),
]


def renders_contact_sheet(folder):
    """contact sheet of every render in the folder"""
    from PIL import Image
    ims = sorted(p for p in folder.glob("*/*.png") if not p.stem.endswith("_alpha") and not p.name.startswith("_"))
    w, h, cols = 240, 300, 6; rows = max(1, -(-len(ims) // cols)); sheet = Image.new("RGB", (w * cols, h * rows), (243, 238, 231))
    for i, p in enumerate(ims): sheet.paste(Image.open(p).convert("RGB").resize((w, h)), ((i % cols) * w, (i // cols) * h))
    sheet.save(folder / "contact_sheet.jpg", quality=88)


def renders_status(steps): return "PASS" if all(s.get("exit") == 0 and s.get("complete") for s in steps if "exit" in s) else "FAIL"


def job_renders(out):
    logs = out / "logs"; steps = []
    for name, sku, man, extra in RENDER_PLAN:
        if SKUS and sku not in SKUS: continue
        rc = run("renders", name, [BPY, "engine/render_sku.py", "--manifest", man, "--out-dir", out / name, "--device", "GPU", *extra], ENG, logs)
        lp = out / name / "render_log.json"; log = json.loads(lp.read_text()) if lp.exists() else {}
        devs = sorted({log.get("provenance", {}).get("cycles_device") or "?"})
        steps.append(dict(step=name, sku=sku, exit=rc, complete=bool(log.get("complete")), renders=len(log.get("renders", [])), cycles_device=devs))
    try: renders_contact_sheet(out)
    except Exception as e: steps.append(dict(step="contact_sheet", error=str(e)))
    return dict(status=renders_status(steps), steps=steps)


# ------------------------------------------------------------------ job: storefront mannequin catalogue (GPU)
# Every engine-rendered garment on the visible mannequin, in each canonical finish (configs/mannequin_finishes.json), torso framing,
# 2400x3000 RGBA (transparent: the storefront's CSS studio shows behind it). One render_sku.py run per SKU x finish, into <SKU>/<finish>/
# (render_sku names files without the finish, so each finish needs its own folder). Results: My Drive/COLAB_RESULTS/catalogue_mannequin_renders.
MQ_SKUS = ["BS-POLO-001", "BS-CAMP-001", "BS-CAMP-002", "BS-RINGER-001"]
MQ_FINISHES = ["M1", "M2", "M3"]
MQ_VIEWS = [("front", 0), ("three_quarter", 45), ("back", 180)]
MQ_RES = (2400, 3000)
TORSO = {"target_occupancy": 0.78, "max_width": 0.92, "mannequin_profile": "torso", "below_hem_m": 0.12}   # = the polo / camp framing


def mannequin_manifest(sku, fid):
    """Derived from configs/skus/<sku>.json (the real config is not changed): mannequin mode only, one finish, the same torso
    framing and the same three views for every garment."""
    m = json.loads((ENG / "configs/skus" / f"{sku}.json").read_text(encoding="utf-8")); rd = m["render"]
    rd["floating_only"] = False
    rd["views"] = [dict(name=n, angle=a) for n, a in MQ_VIEWS]
    rd["framing"] = dict(TORSO)
    rd["hi_res_resolution"] = list(MQ_RES)
    m["renders"] = [dict(mode="mannequin", angles=[a for _, a in MQ_VIEWS], finish=fid)]
    for cw in m["colorways"]: cw.pop("renders", None)
    m["_colab_note"] = f"Derived by colab_heavy/run_all.py from configs/skus/{sku}.json: storefront mannequin catalogue, finish {fid}, torso framing."
    return m


def master_gate(man):
    """The master's own build gate (build_archetype.py): a garment is shown on the visible mannequin only if it passed,
    including inside_body_verts_max 0 (no mannequin poking through). Otherwise it stays floating-only."""
    q = ENG / Path(man["archetype"]).with_suffix(".quality.json")
    if not q.exists(): return dict(gate_passed=False, reason=f"{q.name} missing")
    d = json.loads(q.read_text(encoding="utf-8"))
    return dict(gate_passed=d.get("gate_passed") is True and not d.get("gate_failures"), quality=q.name,
                inside_body_verts_max=(d.get("gate") or {}).get("inside_body_verts_max"), gate_failures=d.get("gate_failures"))


def check_set(fid, od, man):
    """Cross-check one finish folder against its render_log.json: every colourway x view, sha256, RGBA with real transparency, 2400x3000."""
    from PIL import Image
    lp = od / "render_log.json"
    if not lp.exists(): return dict(ok=False, problems=["no render_log.json (run did not complete)"])
    lg = json.loads(lp.read_text()); probs = []; got = set()
    for r in lg["renders"]:
        f = od / r["file"]; view = next((n for n, a in MQ_VIEWS if a == r["angle"]), None); got.add((r["colorway"], view))
        if not f.exists(): probs.append(f"missing {r['file']}"); continue
        if sha(f) != r.get("sha256"): probs.append(f"{r['file']}: sha256 differs from render_log")
        if r.get("mode") != "mannequin" or r.get("finish") != fid: probs.append(f"{r['file']}: mode {r.get('mode')} finish {r.get('finish')}")
        with Image.open(f) as im:
            if im.mode != "RGBA" or im.size != MQ_RES: probs.append(f"{r['file']}: {im.mode} {im.size}")
            elif im.getchannel("A").getextrema()[0] > 0: probs.append(f"{r['file']}: no transparent pixels")
    want = {(cw["id"], n) for cw in man["colorways"] for n, _ in MQ_VIEWS}
    if want - got: probs.append(f"missing views {sorted(want - got)}")
    return dict(ok=lg.get("complete") is True and not probs, renders=len(lg["renders"]),
                cycles_device=(lg.get("provenance") or {}).get("cycles_device"), problems=probs)


def mannequin_sheet(sku_dir, man):
    """Review sheet per SKU: one row per colourway x finish, columns front / three-quarter / back, on the storefront studio tone."""
    from PIL import Image
    w, h = 240, 300; rows = [(cw["id"], f) for cw in man["colorways"] for f in MQ_FINISHES]
    sheet = Image.new("RGB", (w * len(MQ_VIEWS), h * len(rows)), (236, 232, 225))
    for i, (cw, f) in enumerate(rows):
        for j, (vn, _) in enumerate(MQ_VIEWS):
            p = sku_dir / f / f"{man['sku']}_{cw}_mannequin_{vn}.png"
            if p.exists():
                im = Image.open(p).convert("RGBA"); im.thumbnail((w, h)); cell = Image.new("RGBA", (w, h), (236, 232, 225, 255))
                cell.alpha_composite(im, ((w - im.width) // 2, (h - im.height) // 2)); sheet.paste(cell.convert("RGB"), (j * w, i * h))
    sheet.save(sku_dir / "_contact_sheet.jpg", quality=88)


def mannequin_status(steps): return "PASS" if all(s.get("exit") == 0 and s.get("ok") for s in steps if "exit" in s) and any("exit" in s for s in steps) else "FAIL"


def job_mannequin(out):
    logs = out / "logs"; steps = []; mdir = out / "_manifests"; mdir.mkdir(parents=True, exist_ok=True)
    for sku in MQ_SKUS:
        if SKUS and sku not in SKUS: continue
        base = json.loads((ENG / "configs/skus" / f"{sku}.json").read_text(encoding="utf-8")); gate = master_gate(base)
        if not gate["gate_passed"]:
            steps.append(dict(step=sku, skipped=True, reason="master quality gate did not pass: stays floating-only", gate=gate)); continue
        for fid in MQ_FINISHES:
            mp = mdir / f"{sku}_{fid}.json"; mp.write_text(json.dumps(mannequin_manifest(sku, fid), indent=1)); od = out / sku / fid
            rc = run("mannequin", f"{sku}_{fid}", [BPY, "engine/render_sku.py", "--manifest", mp, "--out-dir", od, "--device", "GPU", "--hi-res"], ENG, logs)
            steps.append(dict(step=f"{sku}_{fid}", sku=sku, finish=fid, exit=rc, master_gate=gate, **check_set(fid, od, base)))
        try: mannequin_sheet(out / sku, base)
        except Exception as e: steps.append(dict(step=f"{sku}_contact_sheet", sku=sku, error=repr(e)))
    return dict(status=mannequin_status(steps), steps=steps, finishes=MQ_FINISHES, views=dict(MQ_VIEWS), resolution=list(MQ_RES), framing=TORSO,
                note="Each SKU/finish folder is complete on its own (render_log.json + images); apply_results.py publishes only folders that pass check_set.")


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


JOBS = {"renders": job_renders, "mannequin": job_mannequin, "fbl003": job_fbl003, "rhinestone": job_rhinestone}
DST = {"mannequin": lambda: RES.parent / "catalogue_mannequin_renders"}   # My Drive/COLAB_RESULTS/catalogue_mannequin_renders


SUBSET = {  # jobs that --skus can limit: (SKU order, step -> SKU, folders a SKU owns in the results, status rule)
    "mannequin": (MQ_SKUS, lambda s: s.get("sku") or next((k for k in MQ_SKUS if s["step"] == k or s["step"].startswith(k + "_")), None),
                  lambda sku: [sku], mannequin_status),
    "renders": ([k for _, k, _, _ in RENDER_PLAN], lambda s: s.get("sku") or {n: k for n, k, _, _ in RENDER_PLAN}.get(s["step"]),
                lambda sku: [n for n, k, _, _ in RENDER_PLAN if k == sku], renders_status),
}


def merge_subset(job, out, dst, res):
    """--skus: put this run's SKUs into the existing results folder. Every other SKU's folders, logs, manifests and JOB_STATUS steps stay
    as they were; the SKUs re-run here lose all their old folders/steps first (a step that failed now does not leave an old image behind).
    Status is recomputed over the merged steps with the job's own rule. Returns the merged status record."""
    order, step_sku, owned, rule = SUBSET[job]
    if not (dst / "JOB_STATUS.json").exists():
        say(job, f"no earlier results at {dst}: the subset becomes the results"); return None
    old = json.loads((dst / "JOB_STATUS.json").read_text())
    replaced = [s["step"] for s in old.get("steps", []) if step_sku(s) in SKUS]
    for sku in SKUS:
        for d in owned(sku):
            if (dst / d).exists(): shutil.rmtree(dst / d)
    for name in replaced:
        for p in (dst / "logs" / f"{name}.log", dst / "_manifests" / f"{name}.json"):
            if p.exists(): p.unlink()
    for p in out.iterdir():
        if p.name in ("JOB_STATUS.json", "RESULTS_MANIFEST.sha256", "contact_sheet.jpg"): continue
        if p.is_dir(): shutil.copytree(p, dst / p.name, dirs_exist_ok=True)
        else: shutil.copy2(p, dst / p.name)
    rank = {k: i for i, k in enumerate(dict.fromkeys(order))}
    steps = [s for s in old.get("steps", []) if step_sku(s) not in SKUS] + res.get("steps", [])
    steps = sorted(steps, key=lambda s: rank.get(step_sku(s), len(rank)))          # stable: a SKU's own steps keep their order
    merged = dict(old, steps=steps, status=rule(steps), finished=res.get("finished"), seconds=res.get("seconds"))
    merged["merges"] = old.get("merges", []) + [dict(skus=SKUS, finished=res.get("finished"), seconds=res.get("seconds"), run_status=res.get("status"),
                                                     replaced_steps=replaced, new_steps=[s["step"] for s in res.get("steps", [])],
                                                     staging_created=json.loads((REPO.parent / "STAGING_INFO.json").read_text()).get("created")
                                                     if (REPO.parent / "STAGING_INFO.json").exists() else None)]
    if job == "renders":
        try: renders_contact_sheet(dst)
        except Exception as e: merged["merges"][-1]["contact_sheet_error"] = str(e)
    (dst / "JOB_STATUS.json").write_text(json.dumps(merged, indent=1))
    if job in DST:
        (dst / "RESULTS_MANIFEST.sha256").write_text("\n".join(f"{sha(p)}  {p.relative_to(dst).as_posix()}" for p in sorted(dst.rglob("*"))
                                                               if p.is_file() and p.name != "RESULTS_MANIFEST.sha256") + "\n")
    return merged


def do(job):
    dst = DST.get(job, lambda: RES / job)(); st = dst / "JOB_STATUS.json"
    subset = bool(SKUS) and job in SUBSET
    if st.exists() and not A.force and not subset and json.loads(st.read_text()).get("status") == "PASS":
        say(job, "already PASS in results, skipped (use --force to redo)"); return json.loads(st.read_text())
    out = A.scratch / job
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True); t0 = time.time()
    try: res = JOBS[job](out)
    except Exception as e:
        import traceback; res = dict(status="ERROR", error=repr(e), traceback=traceback.format_exc())
    res.update(job=job, seconds=round(time.time() - t0), finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (out / "JOB_STATUS.json").write_text(json.dumps(res, indent=1))
    if job in DST:                                         # a results folder outside heavy/ carries its own integrity manifest
        (out / "RESULTS_MANIFEST.sha256").write_text("\n".join(f"{sha(p)}  {p.relative_to(out).as_posix()}" for p in sorted(out.rglob("*"))
                                                               if p.is_file() and p.name != "RESULTS_MANIFEST.sha256") + "\n")
    if subset and (merged := merge_subset(job, out, dst, res)) is not None:
        say(job, f"=== {SKUS}: {res['status']} in {res['seconds']}s, merged into {dst} (merged status {merged['status']})"); return merged
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(out, dst)                              # results land on Drive only when the job is finished
    say(job, f"=== {res['status']} in {res['seconds']}s -> {dst}"); return res


if __name__ == "__main__":
    RES.mkdir(parents=True, exist_ok=True); todo = [j for j in A.jobs.split(",") if j]
    unknown = [s for s in SKUS if s not in MQ_SKUS and s not in {k for _, k, _, _ in RENDER_PLAN}]
    if unknown: sys.exit(f"--skus: unknown SKU(s) {unknown}; known: {sorted(set(MQ_SKUS) | {k for _, k, _, _ in RENDER_PLAN})}")
    results = {}
    # GPU chain (renders then fbl003) runs in parallel with the CPU-only rhinestone suite
    def gpu_chain():                                       # mannequin catalogue first: it is what the storefront is waiting for
        for j in ("mannequin", "renders", "fbl003"):
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
