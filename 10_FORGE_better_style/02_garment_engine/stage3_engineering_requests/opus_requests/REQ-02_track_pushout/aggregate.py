"""Aggregate runs/*.json + repro compare + hashes into REQ-02_results.json (python -I aggregate.py)."""
import json, glob, os, hashlib
R = os.path.normpath(os.path.expanduser('~/Desktop/better_style_execution_workspace/08_KNOWLEDGE/stage3_engineering/opus_requests/REQ-02_track_pushout'))
HE = os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/human_engine'))
V = HE + r'\src_catalog\BETTER_STYLE_VISUAL_CURATION'
orig = {'build_garment_g0.py': V + r'\14_TRACKSUIT_DUAL_STATE_VIDEO\scripts\build_garment_g0.py',
        'bs_track_001_g0_master.blend': V + r'\14_TRACKSUIT_DUAL_STATE_VIDEO\garments\bs_track_001_g0_master.blend',
        'eval_g0.py': HE + r'\engineering\PHASE3_2026-11-09\tracksuit_relax\eval_g0.py',
        'M2_ALABASTER_CHALK.blend': V + r'\11_CANONICAL_MANNEQUINS\BLENDER_ASSETS\M2_ALABASTER_CHALK.blend'}
before = {'build_garment_g0.py': '203cfb0e9017cd45c966e8a55ba6125466ab3bb071a74bebbe327a51ed581fc1',
          'bs_track_001_g0_master.blend': '3da4a44d76192dc0f472cbd711d93ac3b5d9e0ad524f6fe4d80ee86471b65920',
          'eval_g0.py': '430f8b03ac7620c9704c8f750aeb42f116ce857194300685a8574e39de22c48a',
          'M2_ALABASTER_CHALK.blend': 'e37f65c0fe0e8adacb9add31f68b5182ac8efe6ec9928e31d939495e986fdcb3'}
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
out = dict(request='REQ-02 / UNR-07', he_originals={k: dict(path=p, sha256_before=before[k], sha256_after=sha(p), unchanged=sha(p) == before[k]) for k, p in orig.items()},
           instrumented_script=dict(path=R + r'\scratch\repo\BETTER_STYLE_VISUAL_CURATION\14_TRACKSUIT_DUAL_STATE_VIDEO\scripts\build_garment_g0_instr.py',
                                    sha256=sha(R + r'\scratch\repo\BETTER_STYLE_VISUAL_CURATION\14_TRACKSUIT_DUAL_STATE_VIDEO\scripts\build_garment_g0_instr.py'),
                                    diff=R + r'\build_garment_g0_instr.diff'),
           reproduction=json.load(open(R + r'\logs\repro_compare.json')))
runs = {}
for f in sorted(glob.glob(R + r'\runs\*.json')):
    d = json.load(open(f)); runs[d['variant']] = d
out['baseline'] = runs.pop('BASE')
out['variants'] = {k: dict(knobs=v['knobs'], runtime_s=v['runtime_s'],
                           **{g: dict(final=v[g]['final'], vs_baseline=v[g].get('vs_baseline'), props=v[g]['props'],
                                      chest_calibration_final=(v[g]['chest_calibration'][-1] if v[g].get('chest_calibration') else None),
                                      stage_table=v[g]['stage_table'], solidify_offset=v[g]['solidify_offset']) for g in ('jacket', 'pants')})
                   for k, v in runs.items()}
expected = ['V1_nolockz', 'V2_nosmooth', 'V3_nolockz_nosmooth', 'V4_6passes', 'V4b_6passes_literal_smooth', 'V5_floor9p5']
out['variants_missing'] = [v for v in expected if v not in out['variants']]
json.dump(out, open(R + r'\REQ-02_results.json', 'w'), indent=1)
print('written; missing:', out['variants_missing'], 'HE unchanged:', all(x['unchanged'] for x in out['he_originals'].values()))
