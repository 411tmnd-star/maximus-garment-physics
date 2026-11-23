"""Build the Colab staging folder for the heavy run (local PC).
  python 02_engine_3d/colab_heavy/build_staging.py [--dest ~/Desktop/artifacts/03_harness_and_pipeline_outputs/COLAB_HEAVY]
Upload the resulting folder to My Drive/COLAB_HEAVY, open RUN_ALL.ipynb in Colab (L4 GPU), Runtime > Run all.
The folder holds a repo-shaped subset (repo/...), the one scene that lives outside the repo (external/), MANIFEST.sha256, STAGING_INFO.json.
"""
import argparse, hashlib, json, re, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENG = "02_engine_3d/isolated_clones/better_style_catalog_collar"
FILES = [  # repo-relative files
    "02_engine_3d/avatar_engine/baseline/capture_current_walk.npz",
    "02_engine_3d/avatar_engine/harness/gate_limits.json",
    "02_engine_3d/avatar_engine/harness/capture_walk_blender.py",
    "02_engine_3d/avatar_engine/diagnostics/fbl_003/rest_pushout.py",
    "02_engine_3d/avatar_engine/diagnostics/fbl_003/apply_pushout_blend.py",
    "02_engine_3d/avatar_engine/diagnostics/fbl_003/verify_rest.py",
    "02_engine_3d/avatar_engine/diagnostics/fbl_003/rest_meshes.npz",
    "02_engine_3d/avatar_engine/diagnostics/fbl_004/gpu_clearance.py",
    "02_engine_3d/prototypes/P2_clearance_field/clearance_field.py",
]
TREES = [  # repo-relative folders copied whole (minus EXCLUDE)
    "02_engine_3d/colab_heavy",
    f"{ENG}/engine", f"{ENG}/configs",
    f"{ENG}/assets/calibration", f"{ENG}/assets/compiled_uvs", f"{ENG}/assets/raw_art", f"{ENG}/assets/artwork_camp",
    f"{ENG}/rhinestone_lab",
]
ARCHETYPES = f"{ENG}/assets/archetypes"            # top-level files only (no _superseded/)
EXCLUDE_DIRS = {"node_modules", "__pycache__", ".pytest_cache"}
PINNED = {  # masters the renders must use (FBL-008 promotions)
    f"{ENG}/assets/archetypes/camp_shirt_master.blend": "2f39a4dfcaef8c804490daf0f9991e54bbeb07e260c3392856eeb1322ebdfc7c",
    f"{ENG}/assets/archetypes/rugby_polo_master.blend": "6b24ccfc94467048d3e20f61848ebd9de334225e1df51030087bf11374d7c59b",
}
SCENE_CANDIDATES = [Path.home() / "Desktop/better_style_execution_workspace/08_KNOWLEDGE/stage3_engineering/mannequin_solution/evidence/dual_state_scene_copy.blend",
                    Path.home() / "Desktop/better_style_execution_workspace/stage3_engineering/mannequin_solution/evidence/dual_state_scene_copy.blend"]
SCENE_SHA = "ae24fe6273a1955e4f9a2e438a3c58771f932bc061f185687709d434cb0a8fab"   # recorded in capture_current_walk.npz meta


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def git(*a):
    r = subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True); return r.stdout.strip() if r.returncode == 0 else None


def notebook():
    md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
    code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
    cells = [
        md("# Better Style: every heavy job, one run\n\n**Before you start:** Runtime > Change runtime type > **L4 GPU** > Save. Then Runtime > **Run all**.\n\n"
           "This folder must be at `My Drive/COLAB_HEAVY`. Results go to `My Drive/COLAB_RESULTS/heavy/` and "
           "`My Drive/COLAB_RESULTS/catalogue_mannequin_renders/`. If Colab disconnects, Run all again: finished jobs are skipped.\n\n"
           "Jobs: **mannequin** (polo, camp x2, ringer on the mannequin in M1 / M2 / M3, front / three-quarter / back, 2400x3000; Cycles GPU) | "
           "**renders** (camp, polo floating, ringer review; Cycles GPU) | **fbl003** (clearance fix: push-out, "
           "derived scene, rest check, walk re-capture, 9-frame GPU clearance gate) | **rhinestone** (full test suite). Expect about 2-3 h; "
           "to run only the mannequin catalogue, set `JOBS = 'mannequin'` in step 4."),
        code("# 1. Connect Google Drive (click Allow)\nfrom google.colab import drive\ndrive.mount('/content/drive')"),
        code("# 2. GPU + upload integrity\nimport subprocess\nROOT = '/content/drive/MyDrive/COLAB_HEAVY'\nRESULTS = '/content/drive/MyDrive/COLAB_RESULTS/heavy'\n"
             "print(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout or 'NO GPU: Runtime > Change runtime type > L4 GPU')\n"
             "r = subprocess.run('sha256sum -c --quiet MANIFEST.sha256', shell=True, cwd=ROOT, capture_output=True, text=True)\n"
             "assert r.returncode == 0, 'UPLOAD PROBLEM, re-upload these files:\\n' + r.stdout + r.stderr\nprint('All files intact')"),
        code("# 3. Install bpy 5.2.3 (Python 3.13), Node, Chromium; copy the repo to local disk (~5-8 min)\n!bash \"$ROOT/repo/02_engine_3d/colab_heavy/setup.sh\" \"$ROOT\""),
        code("# 4. Run the jobs (mannequin, renders, fbl003 one after another on the GPU; rhinestone suite on the CPU at the same time)\n"
             "JOBS = 'mannequin,renders,fbl003,rhinestone'   # e.g. 'mannequin' for the storefront mannequin catalogue only\n"
             "!cd /content/work/repo && python -u 02_engine_3d/colab_heavy/run_all.py --repo /content/work/repo --results \"$RESULTS\" --bpy-python /content/py313/bin/python --jobs \"$JOBS\""),
        code("# 5. Summary + contact sheets\nimport glob, json, os\nfrom IPython.display import Image, display\ns = json.load(open(f'{RESULTS}/RUN_SUMMARY.json'))\n"
             "for j, r in s['jobs'].items(): print(f\"{j:11s} {r['status']:6s}\", json.dumps(r.get('gate') or r.get('contracts_changed_by_tests') or '')[:300])\n"
             "for cs in [f'{RESULTS}/renders/contact_sheet.jpg'] + sorted(glob.glob(f'{os.path.dirname(RESULTS)}/catalogue_mannequin_renders/*/_contact_sheet.jpg')):\n"
             "    if os.path.exists(cs): print(cs); display(Image(cs, width=720))"),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dest", type=Path, default=Path.home() / "Desktop/COLAB_HEAVY"); a = ap.parse_args()
    dest = a.dest; tmp = dest.with_name(dest.name + "_building")
    if tmp.exists(): shutil.rmtree(tmp)
    (tmp / "repo").mkdir(parents=True); (tmp / "external").mkdir()
    copied = []

    def put(rel):
        src = REPO / rel; dst = tmp / "repo" / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst); copied.append(rel)

    for f in FILES:
        if not (REPO / f).exists(): sys.exit(f"missing {f}")
        put(f)
    for t in TREES:
        for p in sorted((REPO / t).rglob("*")):
            if p.is_file() and not (set(p.relative_to(REPO).parts) & EXCLUDE_DIRS): put(p.relative_to(REPO).as_posix())
    for p in sorted((REPO / ARCHETYPES).iterdir()):
        if p.is_file(): put(p.relative_to(REPO).as_posix())
    for rel, want in PINNED.items():
        got = sha(REPO / rel)
        if got != want: sys.exit(f"{rel} sha {got[:12]} is not the pinned FBL-008 master {want[:12]}")
    # every engine-relative asset path named in a manifest / bundle manifest must be staged
    eng_root = tmp / "repo" / ENG; missing = set()
    jsons = list((eng_root / "configs").rglob("*.json")) + list((tmp / "repo/02_engine_3d/colab_heavy/manifests").glob("*.json")) + list((eng_root / "assets/compiled_uvs").rglob("*.json"))
    for j in jsons:
        for s in re.findall(r'"(assets/[^"]+)"', j.read_text(encoding="utf-8", errors="replace")):
            if not (eng_root / s).exists() and (REPO / ENG / s).exists():
                missing.add(s)
    for s in sorted(missing):          # pull in anything referenced that the folder list missed
        put(f"{ENG}/{s}") if (REPO / ENG / s).is_file() else [put(p.relative_to(REPO).as_posix()) for p in (REPO / ENG / s).rglob("*") if p.is_file()]
    scene = next((p for p in SCENE_CANDIDATES if p.exists()), None)
    if not scene: sys.exit("dual_state_scene_copy.blend not found")
    if sha(scene) != SCENE_SHA: sys.exit(f"{scene} is not the scene the capture was made from")
    shutil.copy2(scene, tmp / "external" / scene.name)
    (tmp / "RUN_ALL.ipynb").write_text(json.dumps(notebook(), indent=1), encoding="utf-8")
    info = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), git_commit=git("rev-parse", "HEAD"), git_branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                git_dirty=[l for l in (git("status", "--porcelain") or "").splitlines()], files=len(copied) + 1,
                pinned_masters=PINNED, scene=dict(source=str(scene), sha256=SCENE_SHA), referenced_assets_added=sorted(missing))
    (tmp / "STAGING_INFO.json").write_text(json.dumps(info, indent=1))
    lines = [f"{sha(p)}  {p.relative_to(tmp).as_posix()}" for p in sorted(tmp.rglob("*")) if p.is_file() and p.suffix != ".ipynb"]
    (tmp / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    if dest.exists(): shutil.rmtree(dest)
    tmp.rename(dest)
    size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(f"staged {len(lines)} files, {size / 1e6:.0f} MB -> {dest}" + (f" (+{len(missing)} referenced assets)" if missing else ""))


if __name__ == "__main__":
    main()
