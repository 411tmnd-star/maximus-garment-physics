"""FBL-008 final report: 8-section gate comparison, fixed build vs the true dynamic baseline.  Reads only the JSON the gate scripts wrote.

  python -I -B fbl008_report.py <gate.json> <frontgap_camp.json> <frontgap_polo.json>

gate.json labels: camp_baseline, camp_fbl008, polo_baseline, polo_fbl008 (fbl008_gate.py); frontgap labels: baseline, fbl008 (fbl008_frontgap.py).
Exit code 0 = every gate passes for both garments, 1 = at least one gate fails."""
import sys, json

G = json.load(open(sys.argv[1])); FG = dict(camp=json.load(open(sys.argv[2])), polo=json.load(open(sys.argv[3])))
NAME = dict(camp='camp_shirt', polo='rugby_polo'); fails = []; W = 34


def row(label, b, f, gate='', ok=None):
    mark = '' if ok is None else ('PASS' if ok else 'FAIL')
    print('  %-*s %-26s %-26s %-22s %s' % (W, label, b, f, gate, mark))
    return ok


def pc(d): return '%.2f %% (%d/%d)' % (d['pct'], d['pass_count'], d['of'])


def hts(m): return {h['sector_from_deg']: h for h in m['stand_height_per_30deg_sector'] if h}


for g in ('camp', 'polo'):
    b, f = G[g + '_baseline'], G[g + '_fbl008']; fb, ff = FG[g]['baseline'], FG[g]['fbl008']; cb, cf = b['build_collar_report'], f['build_collar_report']; bad = []

    def chk(name, ok):
        if not ok: bad.append(name)
        return ok
    print('\n' + '=' * 124); print('FBL-008  %s      baseline: %s   |   fixed: %s' % (NAME[g], b['file'], f['file'])); print('=' * 124)
    print('  %-*s %-26s %-26s %-22s %s' % (W, 'measure', 'TRUE DYNAMIC BASELINE', 'FBL-008 FIXED BUILD', 'gate', 'result'))
    print('\n 1. RING (membership fixed before the pass, reused by the gate)')
    row('ring vertices', b['ring_vertices'], f['ring_vertices']); row('ring definition', 'REQ-01 dynamic', 'static build indices')
    row('surface distance min/med/max mm', '%s / %s / %s' % (b['surface_mm']['min'], b['surface_mm']['median'], b['surface_mm']['max']), '%s / %s / %s' % (f['surface_mm']['min'], f['surface_mm']['median'], f['surface_mm']['max']))
    print('\n 2. G1  ring -> shell within 1.0 mm')
    row('G1_surface (O-7 cloud, 0.25 mm)', pc(b['G1_surface_O7_lt_1mm']), pc(f['G1_surface_O7_lt_1mm']), '>= 99 %', chk('G1_surface', f['G1_surface_O7_lt_1mm']['pct'] >= 99.0))
    row('G1_raw (nearest vertex, logged)', pc(b['G1_raw_vertex_lt_1mm']), pc(f['G1_raw_vertex_lt_1mm']), 'not gated (O-7)')
    print('\n 3. G2  ring -> shell surface <= 1.0 mm')
    row('G2', pc(b['G2_surface_le_1mm']), pc(f['G2_surface_le_1mm']), '100 %', chk('G2', f['G2_surface_le_1mm']['pct'] == 100.0))
    row('REQ-01 window [0.1, 1.0] mm', pc(b['G2_req01_window_0p1_to_1p0mm']), pc(f['G2_req01_window_0p1_to_1p0mm']), 'logged')
    print('\n 4. G3  ring not inside the shell solid')
    row('G3', pc(b['G3_ring_outside_shell']), pc(f['G3_ring_outside_shell']), '100 %', chk('G3', f['G3_ring_outside_shell']['pct'] == 100.0))
    row('ring vertices inside shell', b['G3_ring_outside_shell']['inside'], f['G3_ring_outside_shell']['inside'])
    row('all collar verts inside Tee solid', b['audit_collar_vertices_inside_tee_solid'], f['audit_collar_vertices_inside_tee_solid'], 'audit only')
    print('\n 5. BODY  collar vertices inside the mannequin (all %d)' % f['contract']['vertices'])
    row('inside_body', b['collar_vertices_inside_mannequin'], f['collar_vertices_inside_mannequin'], '0', chk('inside_body', f['collar_vertices_inside_mannequin'] == 0))
    row('min signed distance to body mm', b['collar_min_signed_dist_to_mannequin_mm'], f['collar_min_signed_dist_to_mannequin_mm'])
    print('\n 6. MESH INTEGRITY')
    row('final_leaf_degenerate_triangles', cb.get('final_leaf_degenerate_triangles'), cf.get('final_leaf_degenerate_triangles'), '0', chk('degenerate_triangles', cf.get('final_leaf_degenerate_triangles') == 0))
    row('min triangle area m2', '-', '%.3e' % cf['final_collar_min_triangle_area_m2'], '>= 1e-10')
    row('sliver apex moves / max mm', '-', '%s / %s' % (cf.get('seat_sliver_vertices_moved'), cf.get('seat_sliver_max_move_mm')))
    row('leaf min quad-area ratio', cb.get('leaf_min_quad_area_ratio'), cf.get('leaf_min_quad_area_ratio'), '>= unseated', chk('leaf_quad_area_ratio', cf['leaf_min_quad_area_ratio'] >= cf['leaf_min_quad_area_ratio_unseated']))
    row('leaf min gap over shirt mm', cb.get('final_leaf_min_gap_over_shirt_mm'), cf.get('final_leaf_min_gap_over_shirt_mm'), '>= 0.5', chk('leaf_gap', cf['final_leaf_min_gap_over_shirt_mm'] >= 0.5))
    row('leaf verts under shirt', cb.get('final_leaf_verts_under_shirt'), cf.get('final_leaf_verts_under_shirt'), '0', chk('under_shirt', cf.get('final_leaf_verts_under_shirt') == 0))
    row('rim scallop max mm', cb.get('rim_scallop_max_mm'), cf.get('rim_scallop_max_mm'), '<= 2.0', chk('rim_scallop', cf['rim_scallop_max_mm'] <= 2.0))
    print('\n 7. FRONT OPENING + STAND HEIGHT')
    row('opening chord m', fb['opening_chord_m'], ff['opening_chord_m'], '>= baseline', chk('front_gap_chord', ff['opening_chord_m'] >= fb['opening_chord_m'] - 1e-4))
    row('edges / faces across the opening', '%d / %d' % (fb['bridging_edges'], fb['bridging_faces']), '%d / %d' % (ff['bridging_edges'], ff['bridging_faces']), '0 / 0', chk('front_gap_bridged', ff['bridging_edges'] == 0 and ff['bridging_faces'] == 0))
    row('verts in opening corridor', fb['vertices_in_opening'], ff['vertices_in_opening'], '0', chk('front_gap_vertices', ff['vertices_in_opening'] == 0))
    row('empty 10-deg ring sectors', len(b['sectors_empty']), len(f['sectors_empty']), '== baseline', chk('front_gap_sectors', f['sectors_empty'] == b['sectors_empty']))
    row('  from..to deg', '%s..%s' % (b['sectors_empty'][0], b['sectors_empty'][-1] + 10) if b['sectors_empty'] else '-', '%s..%s' % (f['sectors_empty'][0], f['sectors_empty'][-1] + 10) if f['sectors_empty'] else '-')
    row('seat: opening excluded deg', '-', cf.get('seat_front_opening_excluded_deg'))
    row('longest collar edge m', fb['max_edge_m'], ff['max_edge_m'])
    hb, hf = hts(b), hts(f); dmax = 0.0
    print('     stand column height, median mm per 30-deg sector (baseline -> fixed):')
    for s in sorted(set(hb) | set(hf)):
        x, y = hb.get(s), hf.get(s)
        if x and y: dmax = max(dmax, abs(y['column_height_mm_median'] - x['column_height_mm_median']))
        print('       %+4d..%+4d   %6s -> %6s' % (s, s + 30, x['column_height_mm_median'] if x else '-', y['column_height_mm_median'] if y else '-'))
    row('largest median height change mm', '-', '%.2f' % dmax, 'reported, not gated')
    print('\n 8. CONTRACT + BUILD QUALITY GATE')
    kb, kf = b['contract'], f['contract']
    row('collar vertices / faces', '%d / %d' % (kb['vertices'], kb['faces']), '%d / %d' % (kf['vertices'], kf['faces']), '67554, == baseline', chk('contract_counts', kf['vertices'] == 67554 and kf['faces'] == kb['faces']))
    for k, lab in (('vertex_groups', 'vertex groups'), ('material_slots', 'material slots'), ('uv_layers', 'UV layers'), ('garment_objects', 'Garment objects')):
        row(lab, 'n=%d' % len(kb[k]), 'n=%d %s' % (len(kf[k]), 'same' if kf[k] == kb[k] else 'DIFFERENT'), '== baseline', chk('contract_' + k, kf[k] == kb[k]))
    row('Tee vertices / faces', '%d / %d' % (kb['tee_vertices'], kb['tee_faces']), '%d / %d' % (kf['tee_vertices'], kf['tee_faces']), '== baseline', chk('contract_tee', (kf['tee_vertices'], kf['tee_faces']) == (kb['tee_vertices'], kb['tee_faces'])))
    row('collar modifiers left', kb['modifiers'], kf['modifiers'], '0', chk('contract_modifiers', kf['modifiers'] == 0))
    row('build_archetype quality gate', b['build_quality_gate_passed'], f['build_quality_gate_passed'], 'True', chk('build_quality_gate', f['build_quality_gate_passed'] is True))
    print('\n  %s: %s' % (NAME[g], 'ALL GATES PASS' if not bad else 'FAILED: ' + ', '.join(bad))); fails += [NAME[g] + ':' + x for x in bad]
print('\n' + '=' * 124); print('FBL-008 VERDICT: %s' % ('PASS (both garments, every gate)' if not fails else 'FAIL  ' + ' '.join(fails))); print('=' * 124)
sys.exit(1 if fails else 0)
