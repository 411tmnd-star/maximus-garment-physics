"""make_manifest.py  write REPRODUCIBILITY_MANIFEST.json for the four prototypes: re-run every test suite, re-hash the live engine files against the hashes
recorded before this work, hash every prototype source, input copy and output.  Read-only towards the engine.   python -I make_manifest.py"""
import os
import sys, json, hashlib, subprocess, platform, re, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; WS = HERE.parents[1]; REQ = WS / "stage3_engineering/opus_requests"
HE = Path(os.path.normpath(os.path.expanduser('~/Documents/Codex/dev array/411_01_BETTER_STYLE_ACTIVE/HUMAN_ENGINE')))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
man = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), workspace=str(WS), python=sys.version.split()[0], platform=platform.platform())
vers = {}
for m in ("numpy", "scipy", "pytest", "bpy"):
    try:
        mod = __import__(m); vers[m] = getattr(mod, "__version__", None) or mod.app.version_string
    except Exception as e: vers[m] = f"unavailable: {e}"
man["packages"] = vers

# ---- 1. live engine files: recorded hash (before this work) against the hash now
eng = []
ext = json.load(open(HERE / "_inputs/arrays/extraction_record.json"))
for tag, r in ext.items():
    if tag.startswith("_") or r.get("missing"): continue
    eng.append(dict(role=f"array extraction source ({tag})", path=r["path"], recorded=r["sha256_before"].lower()))
for line in open(REQ / "REQ-03_bst_factor_isolation/logs/he_hashes_before.txt", encoding="utf-8-sig"):
    m = re.match(r"\s*([0-9A-Fa-f]{64})\s+(.+?)\s*$", line)
    if m: eng.append(dict(role="REQ-03 source of the rhinestone copies used by P1", path=m.group(2), recorded=m.group(1).lower()))
for line in open(REQ / "REQ-04_pose_swing_twist_dqs/logs/he_rehash_after.txt", encoding="utf-8"):
    m = re.match(r"\s*([0-9a-f]{64})\s+\*(/c/.+?)\s*$", line)
    if m: eng.append(dict(role="REQ-04 source of the rig, retargeter and pose registry used by P3", path="C:" + m.group(2)[2:].replace("/", "\\"), recorded=m.group(1)))
he2 = json.load(open(REQ / "REQ-02_track_pushout/REQ-02_results.json")).get("he_originals")
if isinstance(he2, dict):
    for k, v in he2.items():
        h = v if isinstance(v, str) else (v.get("sha256_after") or v.get("sha256") or v.get("after")) if isinstance(v, dict) else None; pth = v.get("path", k) if isinstance(v, dict) else k
        if isinstance(h, str) and len(h) == 64: eng.append(dict(role="REQ-02 tracksuit source", path=pth, recorded=h.lower()))
seen = set(); out = []
for e in eng:
    key = e["path"].lower()
    if key in seen: continue
    seen.add(key); p = Path(e["path"]); e["exists"] = p.exists(); e["now"] = sha(p) if p.exists() else None; e["unchanged"] = bool(e["now"] == e["recorded"]); out.append(e)
man["live_engine_files"] = dict(checked=len(out), unchanged=sum(e["unchanged"] for e in out), changed_or_missing=[e for e in out if not e["unchanged"]], files=out)

# ---- 2. tests, executed now
tests = {}
for d in ("P1_bstset_exact_hull", "P2_clearance_field", "P3_pose_swing_twist", "P4_geometry_fingerprint"):
    r = subprocess.run([sys.executable, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=HERE / d, capture_output=True, text=True); tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
    m = re.search(r"(\d+) passed", tail); f = re.search(r"(\d+) failed", tail); tests[d] = dict(command="python -I -m pytest -q -p no:cacheprovider tests", exit_code=r.returncode, passed=int(m.group(1)) if m else 0, failed=int(f.group(1)) if f else 0, summary=tail)
man["tests"] = tests

# ---- 3. prototype files
def tree(d, pats):
    return {str(p.relative_to(HERE)).replace("\\", "/"): dict(sha256=sha(p), bytes=p.stat().st_size) for pat in pats for p in sorted(d.glob(pat)) if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts}
man["prototype_sources"] = tree(HERE, ["*.py", "common/*.py", "P*/*.py", "P*/tests/*.py"])
man["prototype_outputs"] = tree(HERE, ["P*/out/*"])
man["prototype_documents"] = tree(HERE, ["*.md", "*.csv", "P*/*.md"])
man["input_arrays"] = tree(HERE, ["_inputs/arrays/*"])
man["commands"] = {
    "P1": ["cd P1_bstset_exact_hull", "python -I -m pytest -q -p no:cacheprovider tests", "python -I verify_layout.py            # about 70 s -> out/verify_layout.json"],
    "P2": ["cd P2_clearance_field", "python -I -m pytest -q -p no:cacheprovider tests", "python -I compare_existing.py         # about 8 min (exact winding number) -> out/compare_existing_g0.json",
           "python -I legacy_tie_analysis.py      # needs the bpy module -> out/legacy_tie_analysis.json", "python -I body_self_intersection.py   # needs the bpy module -> out/body_self_intersection.json"],
    "P3": ["cd P3_pose_swing_twist", "python -I -m pytest -q -p no:cacheprovider tests", "python -I run_pose_construction.py            # needs the bpy module, about 25 s -> out/pose_construction.json",
           "python -I run_pose_construction.py --quick --scan   # -> out/swivel_scan.json (and out/pose_construction_quick.json)"],
    "P4": ["cd P4_geometry_fingerprint", "python -I -m pytest -q -p no:cacheprovider tests", "python -I run_real.py                 # -> out/real_assets.json"],
    "inputs": ["python -I common/extract_blend_arrays.py _inputs/arrays   # needs the bpy module; opens engine .blend files read-only, never saves"],
    "manifest": ["python -I make_manifest.py"]}
json.dump(man, open(HERE / "REPRODUCIBILITY_MANIFEST.json", "w"), indent=1)
print(json.dumps(dict(engine=dict(checked=len(out), unchanged=man["live_engine_files"]["unchanged"], bad=[e["path"] for e in out if not e["unchanged"]]), tests={k: v["summary"] for k, v in tests.items()}, packages=vers,
                      sources=len(man["prototype_sources"]), outputs=len(man["prototype_outputs"])), indent=1))
