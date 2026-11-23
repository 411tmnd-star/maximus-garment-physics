"""Print the REQ-01 tables from probe_output.json + surface_measure.json (run from the REQ-01 folder with python -I)."""
import json, os, sys
here = sys.argv[1] if len(sys.argv) > 1 else '.'
d = json.load(open(os.path.join(here, 'surface_measure.json'))); p = json.load(open(os.path.join(here, 'probe_output.json')))
base = lambda s: s.replace('\\', '/').split('/')[-1]
for f, r in d.items():
    if f == '_meta': print(r); continue
    k = base(f)
    print('=====', k, 'tee', r['tee_verts'], r['tee_sheets'], 'base', r['bottom_ring_verts'], r['bottom_ring_columns_2deg'], r['bottom_ring_z_range_m'], 'unch', r['unchanged'])
    for g, t in r['bottom_ring_by_group'].items():
        print(' ', g, t['n_verts'])
        for m in ('to_all_mm', 'to_outer_mm', 'to_inner_mm', 'to_edge_mm', 'signed_vs_outer_normal_mm', 'dz_to_nearest_mm', 'dh_to_nearest_mm'):
            print('    ', m, t[m] and [t[m][x] for x in ('min', 'median', 'p95', 'max')])
        print('    ', t['nearest_sheet_counts'])
    print('  whole-stand surface min/sector', r['whole_stand_surface_min_sector_10deg_mm'])
    pk = [x for x in p if base(x) == k][0]
    print('  probe vertex min/sector       ', p[pk]['collar_stand']['sector_min_mm_from_back_-180_in_10deg_steps'])
    print('  base ring min_all/sector      ', [s and s['min_all'] for s in r['bottom_ring_sector_10deg_from_back_-180']])
    print('  base ring min_outer/sector    ', [s and s['min_outer'] for s in r['bottom_ring_sector_10deg_from_back_-180']])
    print('  base ring min_inner/sector    ', [s and s['min_inner'] for s in r['bottom_ring_sector_10deg_from_back_-180']])
    print('  base ring median dz/sector    ', [s and s['median_dz'] for s in r['bottom_ring_sector_10deg_from_back_-180']])
    for nm in ('placket_overlap', 'placket_buttons'): print(' ', nm, r[nm])
