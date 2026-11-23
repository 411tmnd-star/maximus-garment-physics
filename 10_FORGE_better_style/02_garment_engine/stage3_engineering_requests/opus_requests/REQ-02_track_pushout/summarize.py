"""Print compact tables from runs/*.json (python -I summarize.py)."""
import json, glob, os
R = os.path.normpath(os.path.expanduser('~/Desktop/better_style_execution_workspace/08_KNOWLEDGE/stage3_engineering/opus_requests/REQ-02_track_pushout'))
r = json.load(open(os.path.join(R, 'runs', 'BASE.json')))
print(json.dumps(r['body']), json.dumps(r['instrumented_vs_original_master']))
for k in ('jacket', 'pants'):
    J = r[k]; print('==', k, J['n_single_layer'], J['solidify_outer_equals_last_stage_max_abs_m'], J['solidify_offset'], J['props'])
    for s, t in J['stage_table'].items():
        print(f"{s:16s} a={t['inside_a_signtest']:5d} b={t['inside_b_independent']:5d} min={t['min_mm']} p1={t['p1_mm']} p5={t['p5_mm']} dis={t['disagree']} aonly={t['a_only']} bonly={t['b_only']} z={t.get('disagree_z')} x={t.get('disagree_absx')} m={t['test_b_method']}")
    for s, t in J['transitions_test_a'].items(): print('  ', s, t)
    for te in 'ab':
        A = J['attribution_test_' + te]; print(' attr', te, A['n'], A['by_sheet'], json.dumps(A['contingency_stage_x_region']), A['solidify_substage'], A['nz_stats'], A['z_range'])
    print(' hist outer', J['stage_table']['solidify_outer'].get('disagree_z_hist_0p1m'), 'inner', J['stage_table']['solidify_inner'].get('disagree_z_hist_0p1m'))
    dd = J['stage_table']['solidify_outer'].get('disagree_detail', [])[:8]
    for x in dd: print('   dis', x)
E = r['ease_vs_clearance']
for g in ('torso', 'sleeve_L'):
    for s in E[g]: print(g, json.dumps(s))
print(E['torso_pooled']); print(E['sleeve_pooled'])
for f in sorted(glob.glob(os.path.join(R, 'runs', 'V*.json'))):
    v = json.load(open(f)); print('##', v['variant'], v['knobs'], v['runtime_s'])
    for k in ('jacket', 'pants'):
        print('  ', k, json.dumps(v[k]['final']), json.dumps(v[k].get('vs_baseline')), json.dumps(v[k]['props']), v[k].get('chest_calibration') and v[k]['chest_calibration'][-1])
