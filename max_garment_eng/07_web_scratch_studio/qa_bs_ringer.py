"""Headless Playwright visual & telemetry QA for 07_web_scratch_studio BS-RINGER-001."""
from __future__ import annotations
import os

import functools
import http.server
import json
import sys
import threading
import time
from pathlib import Path

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

STUDIO_DIR = Path(__file__).resolve().parent
CHROMIUM_EXE = os.path.normpath(os.path.expanduser('~/AppData/Local/ms-playwright/chromium-1248/chrome-win64/chrome.exe'))


class _StudioHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "application/javascript; charset=utf-8",
        ".mjs": "application/javascript; charset=utf-8",
        ".wasm": "application/wasm",
        ".webp": "image/webp",
        ".glb": "model/gltf-binary",
        ".sog": "application/octet-stream",
        ".spz": "application/octet-stream",
    }

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def _serve_studio() -> tuple[http.server.ThreadingHTTPServer, int]:
    handler = functools.partial(_StudioHandler, directory=str(STUDIO_DIR))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, int(srv.server_address[1])


def main() -> int:
    srv, port = _serve_studio()
    url = f"http://127.0.0.1:{port}/index.html"
    out_sheet = STUDIO_DIR / "out_views" / "scratch_studio_bs_ringer_qa.png"
    out_sheet.parent.mkdir(parents=True, exist_ok=True)
    out_report = STUDIO_DIR / "out_views" / "scratch_studio_bs_ringer_qa.json"

    console_errors: list[str] = []
    page_errors: list[str] = []
    failed_requests: list[str] = []

    try:
        with sync_playwright() as p:
            launch_kwargs = {
                "args": [
                    "--use-angle=swiftshader",
                    "--enable-unsafe-swiftshader",
                    "--ignore-gpu-blocklist",
                ]
            }
            if Path(CHROMIUM_EXE).exists():
                launch_kwargs["executable_path"] = CHROMIUM_EXE
            browser = p.chromium.launch(**launch_kwargs)
            ctx = browser.new_context(viewport={"width": 1280, "height": 960})
            page = ctx.new_page()

            page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: page_errors.append(str(e)))
            page.on("requestfailed", lambda r: failed_requests.append(f"FAILED {r.url}: {r.failure}"))
            page.on("response", lambda r: failed_requests.append(f"HTTP {r.status} {r.url}") if r.status >= 400 else None)

            t0 = time.perf_counter()
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_function(
                "() => window.__SPARK_SCRATCH_CTRL__ && window.__SPARK_SCRATCH_CTRL__.state.ready",
                timeout=90000,
            )
            wall_ready_ms = round((time.perf_counter() - t0) * 1000, 1)

            # Let a couple frames render cleanly to warm Spark sorter & FPS counter
            page.wait_for_timeout(1800)

            # Measure steady-state SwiftShader render FPS over 5 consecutive rendered frames
            fps_stats = page.evaluate("async () => await window.__SPARK_SCRATCH_CTRL__.measureFps(5)")
            page.wait_for_timeout(400)

            stage = page.locator("#spark-stage")
            box = stage.bounding_box()
            assert box is not None, "Missing #spark-stage bounding box"
            clip = {"x": box["x"], "y": box["y"], "width": box["width"], "height": box["height"]}
            panel_imgs: list[tuple[str, Image.Image]] = []
            modes = [
                ("glb", "1. Meshopt GLB (0.606 MB)"),
                ("sog", "2. ringer_v3.sog (3.731 MiB, v2 SH0)"),
                ("spz", "3. ringer_v4.spz (4.883 MiB)"),
                ("overlay", "4. GLB Wireframe + SOG Alignment Overlay"),
            ]
            for mode, title in modes:
                page.locator(f'button[data-vmode="{mode}"]').click(timeout=60000)
                page.evaluate("""async () => {
                    window.__SPARK_SCRATCH_CTRL__.renderNow();
                    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
                    window.__SPARK_SCRATCH_CTRL__.renderNow();
                }""")
                page.wait_for_timeout(400)
                png_bytes = page.screenshot(clip=clip, animations="disabled", timeout=60000)
                im = Image.open(
                    __import__("io").BytesIO(png_bytes)
                ).convert("RGB")
                panel_imgs.append((title, im))

            # Compare SOG vs SPZ rendered pixel similarity on identical camera
            import numpy as np
            a_sog = np.asarray(panel_imgs[1][1], dtype=np.float32)
            a_spz = np.asarray(panel_imgs[2][1], dtype=np.float32)
            sog_spz_mad = float(np.mean(np.abs(a_sog - a_spz)))

            # Probe garment highlight toggle
            page.locator('button[data-gid="1"]').click(timeout=60000)
            page.wait_for_timeout(200)
            page.locator('button[data-gid="-1"]').click(timeout=60000)
            page.wait_for_timeout(200)

            diag = page.evaluate("() => window.__SPARK_SCRATCH_CTRL__.getDiagnostics()")
            diag["wall_ready_ms"] = wall_ready_ms
            diag["fps_measured_swiftshader"] = fps_stats["fps"]
            diag["frame_ms_swiftshader"] = fps_stats["frameMs"]
            diag["sog_vs_spz_pixel_mad_0_255"] = round(sog_spz_mad, 3)
            diag["console_errors"] = console_errors
            diag["page_errors"] = page_errors
            diag["failed_requests"] = failed_requests

            # Stitch 4-panel QA image (2x2 or 1x4 horizontal grid; 1x4 gives great visibility!)
            pw, ph = panel_imgs[0][1].size
            pad = 12
            hdr = 34
            sheet = Image.new("RGB", (4 * pw + 5 * pad, ph + hdr + 2 * pad), (14, 14, 16))
            draw = ImageDraw.Draw(sheet)
            for i, (label, im) in enumerate(panel_imgs):
                x0 = pad + i * (pw + pad)
                draw.text((x0 + 4, pad + 6), label, fill=(243, 241, 236))
                sheet.paste(im, (x0, pad + hdr))
            sheet.save(out_sheet)
            out_report.write_text(json.dumps(diag, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(diag, indent=2))
            browser.close()
    finally:
        srv.shutdown()

    if console_errors or page_errors or failed_requests:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
