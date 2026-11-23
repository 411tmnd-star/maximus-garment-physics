"""Apply a finished Colab heavy run to the repo (local PC), then test.
  python 02_engine_3d/colab_heavy/apply_results.py <downloaded COLAB_RESULTS/heavy folder> [--dry-run]

What it does (and does not do):
  * verifies every downloaded file against RESULTS_MANIFEST.sha256 (stops on any mismatch)
  * renders: copies the whole set to the engine (assets/renders/colab_<stamp>/). On the storefront it replaces ONLY images that
    already exist with the same file name (camp shirts, polo floating, polo mannequin burgundy). The old folder is backed up to
    sources/_pre_colab_<stamp>/ first and render_log.json is rebuilt from the run's logs. New views (e.g. black-white mannequin, ringer)
    are left for review, not published. Then: build.py, build.py --check, and the storefront python tests.
  * fbl003: copies the rest check, clearance summary, derived scene and new capture to avatar_engine/diagnostics/fbl_003/colab_<stamp>/.
    The baseline capture and the original scene are never replaced (that promotion is the owner's call).
  * rhinestone: copies test evidence to rhinestone_lab/out_colab/<stamp>/ and lists contracts the suite rewrote; contracts/ is not touched.
Writes APPLY_REPORT.json next to the results.
"""
import argparse, hashlib, json, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENG = REPO / "02_engine_3d/isolated_clones/better_style_catalog_collar"; STORE = REPO / "01_web_storefront/catalogue"
FBL = REPO / "02_engine_3d/avatar_engine/diagnostics/fbl_003"; LAB = ENG / "rhinestone_lab"
LIVE_DIRS = ["BS-CAMP-001", "BS-CAMP-002", "BS-POLO-001", "BS-POLO-001-MANNEQUIN"]   # storefront folders the run is allowed to refresh


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def run(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return dict(cmd=" ".join(map(str, cmd)), exit=p.returncode, tail=(p.stdout + p.stderr).strip().splitlines()[-12:])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("results", type=Path); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    R = a.results.resolve(); stamp = time.strftime("%Y%m%d_%H%M%S"); rep = dict(results=str(R), stamp=stamp, dry_run=a.dry_run)
    # 0. integrity
    man = R / "RESULTS_MANIFEST.sha256"
    if not man.exists(): sys.exit("RESULTS_MANIFEST.sha256 missing: download the whole COLAB_RESULTS/heavy folder")
    entries = [(h, f.lstrip("*").strip()) for h, f in (l.split(None, 1) for l in man.read_text().splitlines() if l.strip())]   # "hash  file" or "hash *file"
    bad = [f for h, f in entries if not (R / f).exists() or sha(R / f) != h]
    if bad: sys.exit(f"{len(bad)} files missing or corrupt, e.g. {bad[:5]}")
    summ = json.loads((R / "RUN_SUMMARY.json").read_text()); rep["run_summary"] = {j: r["status"] for j, r in summ["jobs"].items()}
    print("run:", rep["run_summary"])

    # 1. renders
    rr = R / "renders"; st = json.loads((rr / "JOB_STATUS.json").read_text()) if (rr / "JOB_STATUS.json").exists() else None
    if st:
        r = dict(status=st["status"], storefront={})
        pngs = {p.name: p for p in rr.glob("*/*.png") if not p.name.startswith("_")}
        logs = [json.loads(p.read_text()) for p in rr.glob("*/render_log.json")]
        by_file = {x["file"]: (lg, x) for lg in logs if lg.get("complete") for x in lg["renders"]}
        if not a.dry_run: shutil.copytree(rr, ENG / "assets/renders" / f"colab_{stamp}", dirs_exist_ok=True)
        for d in LIVE_DIRS:
            src = STORE / "sources" / d; live = sorted(p.name for p in src.glob("*.png") if not p.stem.endswith("_alpha") and not p.name.startswith("_"))
            have = [n for n in live if n in pngs and n in by_file]
            if not live or len(have) != len(live):
                r["storefront"][d] = dict(applied=False, reason=f"run has {len(have)}/{len(live)} of the live images", missing=sorted(set(live) - set(have))); continue
            if not a.dry_run:
                bak = STORE / "sources" / f"_pre_colab_{stamp}" / d; bak.parent.mkdir(parents=True, exist_ok=True); shutil.copytree(src, bak)
                for n in live:
                    shutil.copy2(pngs[n], src / n); al = pngs[n].with_name(Path(n).stem + "_alpha.png")
                    if al.exists(): shutil.copy2(al, src / al.name)
                runs = {id(by_file[n][0]): by_file[n][0] for n in live}.values()
                merged = dict(sku=next(iter(runs))["sku"], archetype=next(iter(runs)).get("archetype"), complete=True, source="colab heavy run " + summ.get("finished", ""),
                              renders=[by_file[n][1] for n in live], provenance_runs=[lg.get("provenance") for lg in runs])
                (src / "render_log.json").write_text(json.dumps(merged, indent=1))
            r["storefront"][d] = dict(applied=not a.dry_run, images=len(live), cycles_device=sorted({str((by_file[n][0].get("provenance") or {}).get("cycles_device")) for n in live}))
        r["review_only"] = sorted(set(pngs) - {n for d in LIVE_DIRS for n in (p.name for p in (STORE / "sources" / d).glob("*.png"))})
        if not a.dry_run and any(v.get("applied") for v in r["storefront"].values()):
            r["build"] = run([sys.executable, "build.py"], STORE); r["check"] = run([sys.executable, "build.py", "--check"], STORE)
            r["tests"] = run([sys.executable, "-m", "pytest", "-q", str(REPO / "01_web_storefront/tools/tests")], REPO)
        rep["renders"] = r; print("renders:", {k: v.get("applied") for k, v in r["storefront"].items()}, "| review only:", len(r["review_only"]))

    # 2. fbl003
    fb = R / "fbl003"
    if (fb / "JOB_STATUS.json").exists():
        st = json.loads((fb / "JOB_STATUS.json").read_text()); dst = FBL / f"colab_{stamp}"
        if not a.dry_run: shutil.copytree(fb, dst, dirs_exist_ok=True)
        rep["fbl003"] = dict(status=st["status"], rest_check=st.get("rest_check"), gate=st.get("gate"), steps=st.get("steps"), copied_to=str(dst),
                             next="If the gate PASSES: review the derived scene, then decide whether to promote it and the new capture (baseline is untouched).")
        print("fbl003:", st["status"], st.get("gate"))

    # 3. rhinestone
    rh = R / "rhinestone"
    if (rh / "JOB_STATUS.json").exists():
        st = json.loads((rh / "JOB_STATUS.json").read_text()); dst = LAB / "out_colab" / stamp
        if not a.dry_run: shutil.copytree(rh, dst, dirs_exist_ok=True)
        rep["rhinestone"] = dict(status=st["status"], contracts_changed_by_tests=st.get("contracts_changed_by_tests"), copied_to=str(dst))
        print("rhinestone:", st["status"], "| contracts rewritten by tests:", len(st.get("contracts_changed_by_tests") or []))

    (R / "APPLY_REPORT.json").write_text(json.dumps(rep, indent=1)); print("report:", R / "APPLY_REPORT.json")
    print("Nothing was committed. Review git status / the contact sheet, then commit.")


if __name__ == "__main__":
    main()
