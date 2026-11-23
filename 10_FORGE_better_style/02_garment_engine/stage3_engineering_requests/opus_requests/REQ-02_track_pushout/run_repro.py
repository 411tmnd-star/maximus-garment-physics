"""REQ-02 step 1: run the UNMODIFIED scratch copy of build_garment_g0.py (build_all) with every output path verified to be in scratch.
blender -b --factory-startup --python-exit-code 1 --python run_repro.py
"""
import os
import sys, importlib.util
from pathlib import Path
R = Path(os.path.normpath(os.path.expanduser('~/Desktop/better_style_execution_workspace/08_KNOWLEDGE/stage3_engineering/opus_requests/REQ-02_track_pushout')))
SCR = R / 'scratch' / 'repo'
HE = 'documents\\codex\\dev array'
src = SCR / 'BETTER_STYLE_VISUAL_CURATION/14_TRACKSUIT_DUAL_STATE_VIDEO/scripts/build_garment_g0.py'
spec = importlib.util.spec_from_file_location('bg0', str(src)); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
for p in (mod.REPO, mod.VIDEO_DIR, mod.GARMENTS_DIR, mod.TEX_DIR, Path(mod.SPEC['mannequin_blend'])):
    rp = str(Path(p).resolve()).lower()
    assert rp.startswith(str(SCR.resolve()).lower()), f'path escapes scratch: {p}'
    assert HE not in rp
print('[repro] all builder paths inside scratch:', mod.REPO)
import bpy
_save = bpy.ops.wm.save_as_mainfile
mod.build_all('both', crop_dir=str(mod.VIDEO_DIR))
print('[repro] done')
