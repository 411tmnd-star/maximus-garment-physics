"""Apply a finished Colab heavy run to the repo (local PC), then test.
  python 02_engine_3d/colab_heavy/apply_results.py <downloaded COLAB_RESULTS/heavy folder> [--mannequin <COLAB_RESULTS/catalogue_mannequin_renders>] [--dry-run]
  python 02_engine_3d/colab_heavy/apply_results.py <downloaded COLAB_RESULTS/catalogue_mannequin_renders folder> [--dry-run]

What it does (and does not do):
  * verifies every downloaded file against RESULTS_MANIFEST.sha256 (stops on any mismatch)
  * renders: copies the whole set to the engine (assets/renders/colab_<stamp>/). On the storefront it replaces ONLY images that
    already exist with the same file name (camp shirts, polo floating). The old folder is backed up to sources/_pre_colab_<stamp>/
    first and render_log.json is rebuilt from the run's logs. New views (e.g. ringer) are left for review, not published.
  * mannequin (storefront mannequin catalogue, M1/M2/M3): each <SKU>/<finish>/ folder that passed the run's cross-check AND re-checks
    here (sha256, every catalogue colourway x front / three-quarter / back, mode mannequin, that finish) is copied to
    catalogue/sources/<SKU>-MANNEQUIN/<finish>/ (an existing folder is backed up first) and wired into that product's extra_render_dirs
    in data/catalogue.json (targeted edit, the file's hand layout is kept). The polo's old M2 set at the folder root (pre-FBL-008 master)
    is unwired, not deleted, once the new M2 folder is in. Garments the run skipped (master gate failed) stay floating-only.
  * then, once: build.py, build.py --check, tools/validate_catalogue.py and the storefront python tests.
  * fbl003: copies the rest check, clearance summary, derived scene and new capture to avatar_engine/diagnostics/fbl_003/colab_<stamp>/.
    The baseline capture and the original scene are never replaced (that promotion is the owner's call).
  * rhinestone: copies test evidence to rhinestone_lab/out_colab/<stamp>/ and lists contracts the suite rewrote; contracts/ is not touched.
Writes APPLY_REPORT.json next to the results. Nothing is committed.
"""
import argparse, hashlib, json, re, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENG = REPO / "02_engine_3d/isolated_clones/better_style_catalog_collar"; SITE = REPO / "01_web_storefront"; STORE = SITE / "catalogue"
FBL = REPO / "02_engine_3d/avatar_engine/diagnostics/fbl_003"; LAB = ENG / "rhinestone_lab"
LIVE_DIRS = ["BS-POLO-001"]   # safe default allowlist (camp shirts excluded unless explicitly passed via --skus)
CATALOGUE = STORE / "data/catalogue.json"
MQ_ANGLES = {0: "front", 45: "three_quarter", 180: "back"}
MQ_LEGACY = {"BS-POLO-001": "sources/BS-POLO-001-MANNEQUIN"}   # old M2 burgundy front/back (master 83351813), superseded by <SKU>-MANNEQUIN/M2


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def run(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return dict(cmd=" ".join(map(str, cmd)), exit=p.returncode, tail=(p.stdout + p.stderr).strip().splitlines()[-12:])


def verify(folder):
    man = folder / "RESULTS_MANIFEST.sha256"
    if not man.exists(): sys.exit(f"{man} missing: download the whole folder")
    entries = [(h, f.lstrip("*").strip()) for h, f in (l.split(None, 1) for l in man.read_text().splitlines() if l.strip())]   # "hash  file" or "hash *file"
    bad = [f for h, f in entries if not (folder / f).exists() or sha(folder / f) != h]
    if bad: sys.exit(f"{folder.name}: {len(bad)} files missing or corrupt, e.g. {bad[:5]}")


def set_extra_render_dirs(text, sku, dirs):
    """Return catalogue.json text with product <sku>'s extra_render_dirs set to dirs, editing only that line (or adding it after "status")."""
    start = text.index(f'"sku": "{sku}"'); nxt = text.find('"sku": "', start + 1); end = nxt if nxt != -1 else len(text)
    block = text[start:end]; val = json.dumps(dirs)
    new, n = re.subn(r'"extra_render_dirs":\s*\[[^\]]*\]', f'"extra_render_dirs": {val}', block, count=1)
    if not n:
        new, n = re.subn(r'(\n([ \t]*)"status": "[^"]*",)', lambda m: f'{m.group(1)}\n{m.group(2)}"extra_render_dirs": {val},', block, count=1)
    if not n: raise ValueError(f"{sku}: no place for extra_render_dirs in catalogue.json")
    out = text[:start] + new + text[end:]
    got = next(p for p in json.loads(out)["products"] if p["sku"] == sku).get("extra_render_dirs")
    if got != dirs: raise ValueError(f"{sku}: catalogue.json edit did not round-trip ({got} != {dirs})")
    return out


def apply_mannequin(M, stamp, dry, allowed_skus=None):
    """Storefront mannequin catalogue (run_all.py job "mannequin"), gated by explicit SKU allowlist & SHA-256 verification."""
    verify(M); st = json.loads((M / "JOB_STATUS.json").read_text())
    cat_text = CATALOGUE.read_text(encoding="utf-8"); cat = json.loads(cat_text)
    products = {p["sku"]: p for p in cat["products"]}
    rep = dict(status=st["status"], allowed_skus=sorted(allowed_skus) if allowed_skus is not None else None, skipped=[s for s in st["steps"] if s.get("skipped")], products={})
    by_sku = {}
    for s in st["steps"]:
        if "finish" in s: by_sku.setdefault(s["sku"], []).append(s)
    for sku, steps in sorted(by_sku.items()):
        entry = products.get(sku); r = rep["products"][sku] = dict(applied=[], refused={})
        if allowed_skus is not None and sku not in allowed_skus:
            r["refused"]["*"] = f"blocked by --skus allowlist ({sorted(allowed_skus)})"; continue
        if not entry: r["refused"]["*"] = "not in data/catalogue.json"; continue
        for s in steps:
            fid = s["finish"]; src = M / sku / fid
            if s.get("exit") != 0 or not s.get("ok"): r["refused"][fid] = f"run check failed: {s.get('problems') or s.get('exit')}"; continue
            lg = json.loads((src / "render_log.json").read_text())
            probs = [x["file"] for x in lg["renders"] if not (src / x["file"]).exists() or sha(src / x["file"]) != x.get("sha256")]
            have = {(x["colorway"], MQ_ANGLES.get(x["angle"])) for x in lg["renders"] if x.get("mode") == "mannequin" and x.get("finish") == fid}
            need = {(cw, v) for cw in entry["colourways"] for v in MQ_ANGLES.values()}
            if probs or need - have or not lg.get("complete"):
                r["refused"][fid] = dict(sha_mismatch=probs, missing_views=sorted(need - have), complete=lg.get("complete")); continue
            dst = STORE / "sources" / f"{sku}-MANNEQUIN" / fid
            if not dry:
                if dst.exists():
                    bak = STORE / "sources" / f"_pre_colab_{stamp}" / f"{sku}-MANNEQUIN" / fid; bak.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(dst), str(bak))
                shutil.copytree(src, dst)
            r["applied"].append(f"sources/{sku}-MANNEQUIN/{fid}")
        if r["applied"]:
            old = list(entry.get("extra_render_dirs") or [])
            drop = {MQ_LEGACY[sku]} if sku in MQ_LEGACY and f"sources/{sku}-MANNEQUIN/M2" in r["applied"] else set()
            keep = [d for d in old if d not in drop]
            fins = sorted({d for d in keep + r["applied"] if re.search(r"-MANNEQUIN/M\d+$", d)}, key=lambda d: int(d.rsplit("/M", 1)[1]))
            new = [d for d in keep if d not in fins] + fins           # other folders keep their place; finish folders follow, M1, M2, M3
            r["extra_render_dirs"] = dict(before=old, after=new)
            if new != old: cat_text = set_extra_render_dirs(cat_text, sku, new)
    if not dry and cat_text != CATALOGUE.read_text(encoding="utf-8"):
        shutil.copy2(CATALOGUE, CATALOGUE.with_name(f"catalogue.pre_colab_{stamp}.json")); CATALOGUE.write_text(cat_text, encoding="utf-8")
    rep["any_applied"] = any(v["applied"] for v in rep["products"].values())
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("results", type=Path); ap.add_argument("--mannequin", type=Path, default=None)
    ap.add_argument("--skus", default=None, help="comma-separated SKU allowlist (e.g. BS-POLO-001,BS-RINGER-001); required for non-default SKUs to prevent accidental publication")
    ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    allowed_skus = {s.strip() for s in a.skus.split(",") if s.strip()} if a.skus else set(LIVE_DIRS)
    R = a.results.resolve(); stamp = time.strftime("%Y%m%d_%H%M%S"); rep = dict(results=str(R), stamp=stamp, dry_run=a.dry_run, allowed_skus=sorted(allowed_skus)); rebuild = False
    only_mq = (R / "JOB_STATUS.json").exists() and json.loads((R / "JOB_STATUS.json").read_text()).get("job") == "mannequin"
    M = R if only_mq else (a.mannequin.resolve() if a.mannequin else R.parent / "catalogue_mannequin_renders")

    if not only_mq:
        verify(R)
        summ = json.loads((R / "RUN_SUMMARY.json").read_text()); rep["run_summary"] = {j: r["status"] for j, r in summ["jobs"].items()}
        print("run:", rep["run_summary"])

        # 1. renders
        rr = R / "renders"; st = json.loads((rr / "JOB_STATUS.json").read_text()) if (rr / "JOB_STATUS.json").exists() else None
        if st:
            r = dict(status=st["status"], storefront={})
            pngs = {p.name: p for p in rr.glob("*/*.png") if not p.name.startswith("_")}
            logs = [json.loads(p.read_text()) for p in rr.glob("*/render_log.json")]
            by_file = {}
            for lg in logs:
                if not lg.get("complete"): continue
                for x in lg["renders"]:
                    p_img = pngs.get(x["file"])
                    if p_img is not None and x.get("sha256") and sha(p_img) != x["sha256"]:
                        sys.exit(f"SHA-256 mismatch for render {x['file']} against render_log.json")
                    by_file[x["file"]] = (lg, x)
            if not a.dry_run: shutil.copytree(rr, ENG / "assets/renders" / f"colab_{stamp}", dirs_exist_ok=True)
            for d in sorted(allowed_skus):
                src = STORE / "sources" / d
                if not src.exists():
                    r["storefront"][d] = dict(applied=False, reason=f"storefront source folder sources/{d} does not exist"); continue
                live = sorted(p.name for p in src.glob("*.png") if not p.stem.endswith("_alpha") and not p.name.startswith("_"))
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
            r["review_only"] = sorted(set(pngs) - {n for d in allowed_skus if (STORE / "sources" / d).exists() for n in (p.name for p in (STORE / "sources" / d).glob("*.png"))})
            rebuild |= not a.dry_run and any(v.get("applied") for v in r["storefront"].values())
            rep["renders"] = r; print("renders:", {k: v.get("applied") for k, v in r["storefront"].items()}, "| review only:", len(r["review_only"]))

    # 2. storefront mannequin catalogue
    if (M / "JOB_STATUS.json").exists():
        mq = rep["mannequin"] = apply_mannequin(M, stamp, a.dry_run, allowed_skus=allowed_skus); rebuild |= not a.dry_run and mq["any_applied"]
        print("mannequin:", {k: (v["applied"], list(v["refused"])) for k, v in mq["products"].items()}, "| skipped (floating-only):", [s["step"] for s in mq["skipped"]])
    elif only_mq or a.mannequin:
        sys.exit(f"{M}: no JOB_STATUS.json")

    if rebuild:   # once, after everything is in place
        rep["build"] = run([sys.executable, "-I", "build.py"], STORE); rep["check"] = run([sys.executable, "-I", "build.py", "--check"], STORE)
        rep["validate"] = run([sys.executable, "-I", "tools/validate_catalogue.py"], SITE)
        rep["tests"] = run([sys.executable, "-I", "-m", "unittest", "discover", "-s", "tools/tests"], SITE)
        print("build/check/validate/tests exit:", [rep[k]["exit"] for k in ("build", "check", "validate", "tests")])

    if not only_mq:
        # 3. fbl003
        fb = R / "fbl003"
        if (fb / "JOB_STATUS.json").exists():
            st = json.loads((fb / "JOB_STATUS.json").read_text()); dst = FBL / f"colab_{stamp}"
            if not a.dry_run: shutil.copytree(fb, dst, dirs_exist_ok=True)
            rep["fbl003"] = dict(status=st["status"], rest_check=st.get("rest_check"), gate=st.get("gate"), steps=st.get("steps"), copied_to=str(dst),
                                 next="If the gate PASSES: review the derived scene, then decide whether to promote it and the new capture (baseline is untouched).")
            print("fbl003:", st["status"], st.get("gate"))

        # 4. rhinestone
        rh = R / "rhinestone"
        if (rh / "JOB_STATUS.json").exists():
            st = json.loads((rh / "JOB_STATUS.json").read_text()); dst = LAB / "out_colab" / stamp
            if not a.dry_run: shutil.copytree(rh, dst, dirs_exist_ok=True)
            rep["rhinestone"] = dict(status=st["status"], contracts_changed_by_tests=st.get("contracts_changed_by_tests"), copied_to=str(dst))
            print("rhinestone:", st["status"], "| contracts rewritten by tests:", len(st.get("contracts_changed_by_tests") or []))

    (R / "APPLY_REPORT.json").write_text(json.dumps(rep, indent=1)); print("report:", R / "APPLY_REPORT.json")
    print("Nothing was committed. Review git status / the contact sheets, then commit.")


if __name__ == "__main__":
    main()
