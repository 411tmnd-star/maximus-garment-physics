"""Assemble REQ-01_results.json from the run outputs (python -I req01_assemble.py <REQ-01 folder>)."""
import json, os, sys
D = sys.argv[1]
L = lambda n: json.load(open(os.path.join(D, n)))
hashes = lambda n: [l.split(' *') for l in open(os.path.join(D, n)).read().splitlines()]
res = dict(
    request='REQ-01 collar seating / placket seating', date='2026-11-10',
    tools=dict(blender='5.2.3 LTS (e13b728c6880), run -b --factory-startup for inventory/surface/placket scripts',
               probe_runtime='bpy 5.2.3 module under Python 3.13.16 -I (probe needs scipy 1.18.1; Blender bundled Python has no scipy)',
               numpy='2.5.4 (Py3.13) / 2.3.4 (Blender)'),
    he_hashes_before={p: h for h, p in hashes('he_hashes_before.txt')},
    he_hashes_after={p: h for h, p in hashes('he_hashes_after.txt')},
    he_unchanged=open(os.path.join(D, 'he_hashes_before.txt')).read() == open(os.path.join(D, 'he_hashes_after.txt')).read(),
    inventory=L('inventory.json'), probe_unchanged_run=L('probe_output.json'),
    independent_surface=L('surface_measure.json'), placket_detail=L('placket_detail.json'),
    code_intent=dict(
        source='collar_builder.py + build_archetype.py',
        stand_base='Stand base = row 0 of COLLAR_ROWS, the sewn row of the simulated collar rib, captured at build_archetype.py:694 BEFORE weld (722), refinement, rim low-pass, final clearance pushes, CorrectiveSmooth/Solidify/Subsurf apply (848-851). Simulated collar faces are deleted from the Tee (792-794). Stand rebuilt in build_leaf_object: pushed >=3 mm off body (collar_builder.py:291-299), faired with fair_keep_radius (73-78), stand rows interpolated B->T (104-105), Solidify centred 2 mm + Subsurf 2 (327-328). No shirt-clearance or snap applied to stand (332-333: base is sewn onto the neckline by design).',
        intent='sit ON the neckline loop (share the seam row position); not overlap, not replace shell rows',
        controls=['COLLAR_ROWS[0] (draped sewn row, stale vs final shell)', 'body clearance 0.003 m (line 299)', 'fair_keep_radius 40+6 passes (73-78)',
                  'stand_rows (default 5)', 'pattern.collar_width + stand_lean_deg (build_archetype 652-658)', 'front_gap_m', 'thickness_m (0.002, offset 0)',
                  'Subsurf levels 2', 'shell solidify_thickness 0.002 offset -1 (configs)']),
)
json.dump(res, open(os.path.join(D, 'REQ-01_results.json'), 'w'), indent=1); print('ok', res['he_unchanged'])
