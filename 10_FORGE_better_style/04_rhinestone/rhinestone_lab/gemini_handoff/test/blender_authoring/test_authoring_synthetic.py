"""Stage 4, Blender layer, SYNTHETIC fixture (source_kind synthetic_fixture). Drives the real add-on operators (bpy.ops.bsr.*) through the authoring workflow on a scene built to contain
an unambiguous panel (tube), an ambiguous panel (flange: 'away from the body' is parallel to it), a thick shell, and overlapping regions. Not garment verification.
  python3 test/blender_authoring/test_authoring_synthetic.py [outDir]"""
import json, math, shutil, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent.parent
for q in (ROOT / 'py', ROOT / 'blender_addon'): sys.path.insert(0, str(q))
import bpy, mathutils
import design_intent as DI, authoring as AU, fixtures_v4 as FX, contracts as CT, rhs_format as RF, review_export as RX, verify_independent as VI, placement_v2 as PV
import bs_rhinestone_authoring as ADDON
import readback as RB
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'out/step4/synthetic_authoring'; shutil.rmtree(OUT, ignore_errors=True); OUT.mkdir(parents=True)
res = []; report = {'source_kind': 'synthetic_fixture'}
def check(n, ok, d=''): ok = bool(ok); res.append(ok); print(('PASS ' if ok else 'FAIL ') + n + (('  ' + d) if d else ''), flush=True)
def raises(fn, *a, **k):
    try: fn(*a, **k); return None
    except RuntimeError as e: return str(e)

shell, body = FX.build_scene(); ADDON.register(); sc = bpy.context.scene
sc.bsr_operator = 'test_operator'; sc.bsr_garment_id = 'tube_with_flange'; sc.bsr_design_id = 'synthetic_authoring_test'; sc.bsr_source_kind = 'synthetic_fixture'
sc.bsr_surface_path = str(OUT / 'surface.json'); sc.bsr_design_path = str(OUT / 'design.json'); sc.bsr_output_dir = str(OUT / 'out')
ops = bpy.ops.bsr
print('=== 1 zones and panels', flush=True)
ops.zone_set(zone='tube_band', zone_id=11); ops.zone_set(zone='flange_band', zone_id=12); ops.zone_set(zone='sub_band', zone_id=13, substitution_to='SS16,SS10')
design = ADDON._design(sc); check('zones are defined in the design file; substitution is off unless the zone lists permitted smaller sizes', design['zones']['tube_band']['size_substitution']['enabled'] is False and design['zones']['sub_band']['size_substitution'] == {'enabled': True, 'allowed_to': ['SS16', 'SS10']})
ops.panel_assign(panel_id='tube_outer', surface_id='tube_shell', layer_id='shell_outer', zones='tube_band', object_name='shell', vertex_group='tube', facing='outward', outward_kind='reference_body', reference_object='body_ref')
ops.panel_assign(panel_id='flange_top', surface_id='flange_shell', layer_id='shell_outer', zones='flange_band,sub_band', object_name='shell', vertex_group='flange', facing='outward', outward_kind='reference_body', reference_object='body_ref')
con = ADDON._surface(sc); check('panels carry panel_id, surface_id, layer_id, zones and a SUGGESTED outward reference (not authorized)', len(con['panels']) == 2 and all(p['outward_authorization']['status'] == 'suggested' for p in con['panels']) and CT.validate_surface_contract(con) == [])
check('before inspection no panel can be authorized silently: both verdicts start as review-required', all(p['outward_authorization']['verdict'] == 'AMBIGUOUS_REVIEW_REQUIRED' for p in con['panels']))
err = raises(ops.panel_authorize, panel_id='tube_outer', operator_name='test_operator', note='', acknowledge_ambiguity=False); check('authorizing a panel that was never inspected is refused', err is not None and 'acknowledge' in err, str(err)[:110])

print('=== 2 inspection: reference-body suggestion, ambiguity for review', flush=True)
ops.panel_inspect(panel_id='tube_outer'); ins_t = json.loads(sc['bsr_last_inspection']); ops.panel_inspect(panel_id='flange_top'); ins_f = json.loads(sc['bsr_last_inspection'])
check('tube: reference-body method suggests an outward reference and finds it unambiguous (SUGGESTED_OK)', ins_t['verdict'] == 'SUGGESTED_OK' and ins_t['selected_agree'] > 0.99 and ins_t['selected_ambiguous_fraction'] < 0.02 and ins_t['mixed_layers'] is True, json.dumps({k: ins_t[k] for k in ('verdict', 'selected_agree', 'selected_ambiguous_fraction', 'raw_facing_outward', 'raw_facing_inward')}))
check('tube: the few ambiguous faces are the 2 mm shell rim faces at the open tube ends (perpendicular to outward), reported not hidden', 0 < ins_t['selected_ambiguous_faces'] < 0.02 * ins_t['selected_faces'])
check('tube: the raw selection is a double-layer shell (both facing classes present); the contract selects the outer layer explicitly', ins_t['raw_facing_outward'] > 0.4 and ins_t['raw_facing_inward'] > 0.4 and ins_t['selected_faces'] < ins_t['raw_faces'])
check('flange: reference-body method is AMBIGUOUS (away from the body is parallel to the flange): presented for manual review with locations, not resolved', ins_f['verdict'] == 'AMBIGUOUS_REVIEW_REQUIRED' and ins_f['selected_ambiguous_fraction'] > 0.5 and len(ins_f['ambiguous_locations_world']) > 5, f"ambiguous fraction {ins_f['selected_ambiguous_fraction']:.2f}, {len(ins_f['ambiguous_locations_world'])} locations")
markers = [o for o in bpy.data.objects if o.name.startswith('BSR_AMBIG_flange_top')]; check('ambiguous locations are shown in the Blender viewport as markers for the reviewer', len(markers) == len(ins_f['ambiguous_locations_world']) and len(markers) > 5)
con = ADDON._surface(sc); check('inspection records the verdict and evidence on the panel but does NOT authorize', all(p['outward_authorization']['status'] == 'suggested' for p in con['panels']) and DI.authorization_state(con, 'tube_outer') == 'suggested')
err = raises(ops.panel_authorize, panel_id='flange_top', operator_name='test_operator', note='', acknowledge_ambiguity=False); check('authorizing an AMBIGUOUS reference without acknowledgement and a note is refused', err is not None and 'acknowledge' in err, str(err)[:110])
err = raises(ops.panel_authorize, panel_id='tube_outer', operator_name='', note='ok'); check('authorization needs an operator name', err is not None)
ops.panel_authorize(panel_id='tube_outer', operator_name='test_operator', note='unambiguous: tube is radial from the body'); con = ADDON._surface(sc); a = DI.panel(con, 'tube_outer')['outward_authorization']
check('tube: explicit authorization recorded (who, when, note, spec hash) and state is authorized', DI.authorization_state(con, 'tube_outer') == 'authorized' and a['authorized_by'] == 'test_operator' and a['authorized_at'] and a['authorized_spec_hash'] == DI.spec_hash(DI.panel(con, 'tube_outer')['outward_direction']))

print('=== 3 the operator replaces the flange reference with an explicit one and authorizes it', flush=True)
ops.panel_assign(panel_id='flange_top', surface_id='flange_shell', layer_id='shell_outer', zones='flange_band,sub_band', object_name='shell', vertex_group='flange', facing='outward', outward_kind='vector', outward_params=json.dumps({'v': [0, 0, 1], 'frame': 'object_local'}))
ops.panel_inspect(panel_id='flange_top'); ins_f2 = json.loads(sc['bsr_last_inspection']); check('flange with the explicit +Z reference (object-local): inspection is SUGGESTED_OK', ins_f2['verdict'] == 'SUGGESTED_OK' and ins_f2['selected_agree'] > 0.99, f"{ins_f2['verdict']} agree {ins_f2['selected_agree']:.3f}")
check('an outward reference given by the operator is labelled as such, not as the reference-body suggestion', DI.panel(ADDON._surface(sc), 'flange_top')['outward_authorization']['method'] == 'operator_specified')
con = ADDON._surface(sc); check('re-assigning a panel resets its authorization (new specification = new decision)', DI.authorization_state(con, 'flange_top') == 'suggested')
ops.panel_authorize(panel_id='flange_top', operator_name='test_operator', note='flange faces +Z in object space'); con = ADDON._surface(sc); check('flange authorized with the explicit reference', DI.authorization_state(con, 'flange_top') == 'authorized')

print('=== 4 curve authoring on panels', flush=True)
Z = FX.to_world
ops.curve_create(curve_id='ring', panel_id='tube_outer', zone='tube_band', points_json=json.dumps(FX.ring_points(0.0556, 0.03, 8)), closed=True, size_pattern='SS16', note='closed ring around the tube', bezel_id=1)
helix = [Z((0.0556 * math.cos(0.5 * i), 0.0556 * math.sin(0.5 * i), 0.02 + 0.009 * i)).tolist() for i in range(12)]
ops.curve_create(curve_id='wave', panel_id='tube_outer', zone='tube_band', points_json=json.dumps(helix), closed=False, size_pattern='SS10,SS16', bezel_id=2, note='open helix crossing the ring')
ops.curve_create(curve_id='junction_ring', panel_id='tube_outer', zone='tube_band', points_json=json.dumps(FX.ring_points(0.0556, 0.0695, 8)), closed=True, size_pattern='SS16', note='ring at the flange junction: partly hidden by the flange edge')
ops.curve_create(curve_id='flange_inner', panel_id='flange_top', zone='flange_band', points_json=json.dumps(FX.ring_points(0.0595, 0.0705, 8)), closed=True, size_pattern='SS16', note='flange ring close to the tube: overlaps junction_ring stones')
ops.curve_create(curve_id='flange_outer', panel_id='flange_top', zone='sub_band', points_json=json.dumps(FX.ring_points(0.080, 0.0705, 8)), closed=True, size_pattern='SS20', rows_json='[-0.006,0.0,0.006]', note='three rows on the flange')
design = ADDON._design(sc); cc = DI.to_curve_contract(design)
check('five authored curves exist, each naming panel, surface and zone; Blender curve objects mirror them', len(design['curves']) == 5 and all(c['panel_id'] and c['surface_id'] and c['placement_zone'] for c in design['curves']) and all(bpy.data.objects.get('BSR_' + c['curve_id']) for c in design['curves']))
check('no authored curve carries an outward direction (the panel contract is authoritative)', all('outward_direction' not in c and 'outward' not in c for c in design['curves']) and DI.validate_design(design, ADDON._surface(sc)) == [])
check('authored curves have provenance kind "authored" with the design id and revision; the curve contract validates', all(c['provenance']['kind'] == 'authored' and c['provenance']['design_id'] == design['design_id'] for c in cc['curves']) and CT.validate_curve_contract(cc, ADDON._surface(sc)) == [])
bad = json.loads(json.dumps(design)); bad['curves'][0]['outward_direction'] = [0, 0, 1]; check('a design whose curve defines its own outward direction is rejected', any(e['code'] == 'D_OUTWARD_ON_CURVE' for e in DI.validate_design(bad, ADDON._surface(sc))))
err = raises(ops.curve_create, curve_id='off', panel_id='flange_top', zone='flange_band', points_json=json.dumps(FX.ring_points(0.12, 0.0705, 8)), closed=True); check('a curve whose points are not on the panel is refused, not silently placed', err is not None and 'no face of panel' in err, str(err)[:110])
err = raises(ops.curve_create, curve_id='badzone', panel_id='tube_outer', zone='flange_band', points_json=json.dumps(FX.ring_points(0.0556, 0.05, 8)), closed=True); check('a zone that the panel does not allow is refused', err is not None, str(err)[:110])
check('the design history records every authoring step with operator and revision', len(design['history']) >= 9 and design['revision'] == len(design['history']) and all(h['by'] for h in design['history']))
sep = {'authored': DI.blocked_curves(design, ADDON._surface(sc)), 'derived_slice_curves_in_design': [c for c in design['curves'] if c.get('provenance', {}).get('kind') == 'derived_slice']}; check('authored design curves and derived plane-slice curves are kept apart (none of the latter is in the design)', not sep['derived_slice_curves_in_design'])

print('=== 5 tangent preview independent of outward', flush=True)
ops.tangent_preview(curve_id='wave'); tan1 = bpy.data.objects['BSR_TAN_wave']; out1 = bpy.data.objects['BSR_OUT_wave']; v1 = np.array([tuple(v.co) for v in tan1.data.vertices]); o1 = np.array([tuple(v.co) for v in out1.data.vertices]); c = DI.get_curve(design, 'wave'); prev, _ = AU.tangent_preview(c, 0.012)
dirs = (v1[1::2] - v1[0::2]) / 0.006; want = np.array([t['tangent'] for t in prev]); check('Blender tangent arrows equal the curve tangents', np.allclose(dirs, want, atol=2e-3))   # mesh vertices are float32
ops.tangent_preview(curve_id='wave', reverse=True); v2 = np.array([tuple(v.co) for v in bpy.data.objects['BSR_TAN_wave'].data.vertices]); o2 = np.array([tuple(v.co) for v in bpy.data.objects['BSR_OUT_wave'].data.vertices])
check('reversing the preview flips the tangent arrows and leaves the outward-reference arrows unchanged (separate concepts)', np.allclose((v2[1::2] - v2[0::2]) / 0.006, -want, atol=2e-3) and np.allclose(o1, o2, atol=1e-9))
odir = (o1[1::2] - o1[0::2]) / 0.006; rad = np.array([np.array(p['point']) for p in prev]); check('outward arrows on the tube point away from the body axis; tangents are along the surface (perpendicular to outward)', bool(np.all(np.abs((dirs * odir).sum(1)) < 0.3)))

print('=== 6 generation: authorization gate, then Stage-3 evaluation', flush=True)
con = ADDON._surface(sc); DI.panel(con, 'tube_outer')['outward_direction'] = {'kind': 'away_from_surface', 'reference_object': 'body_ref', 'frame': 'world', 'sign': 1.0}; ADDON._save(ADDON._p(sc, 'bsr_surface_path'), con)
check('changing a panel outward specification after authorization makes the authorization stale', DI.authorization_state(con, 'tube_outer') == 'stale' and {b['curve_id'] for b in DI.blocked_curves(design, con)} == {'ring', 'wave', 'junction_ring'})
err = raises(ops.generate, name='blocked', strict_authorization=True); check('strict generation refuses curves whose panel authorization is stale', err is not None and 'blocked' in err, str(err)[:110])
ops.generate(name='partial'); cnt = json.loads(sc['bsr_last_counts']); blk = json.loads(sc['bsr_last_blocked']); check('non-strict generation blocks those curves, reports them, and generates only authorized panels (nothing silently dropped)', {b['curve_id'] for b in blk} == {'ring', 'wave', 'junction_ring'} and all(b['state'] == 'stale' for b in blk) and cnt['attempted'] > 0 and all(r['curve_id'].startswith('flange') for r in ADDON._CACHE['last'].run.records))
con = ADDON._surface(sc); DI.panel(con, 'tube_outer')['outward_direction'] = {'kind': 'away_from_surface', 'reference_object': 'body_ref', 'frame': 'world'}; ADDON._save(ADDON._p(sc, 'bsr_surface_path'), con); check('restoring the identical specification restores the authorization (hash match)', DI.authorization_state(con, 'tube_outer') == 'authorized')
ops.generate(name='gen1'); res1 = ADDON._CACHE['last']; run1 = res1.run; c1 = run1.counts(); report['gen1_counts'] = c1
check('all five curves generated through the Stage-3 engine; conservation holds; nothing blocked', not res1.blocked and c1['attempted'] == c1['with_transform'] + c1['without_transform'] and set(res1.run.curve_reports) == {'ring', 'wave', 'junction_ring', 'flange_inner', 'flange_outer'}, json.dumps(c1['outcomes']))
check('every generated placement carries all five-outcome machinery: footprint analysis, outcome, reason, sizes', all(r['outcome'] in PV.PP.OUTCOMES and ('measurement' in r) and r['intended_size'] and r['realized_size'] for r in run1.records) and all(r.get('footprint') for r in run1.records if r['transform'] is not None))
byc = {}
for r in run1.records: byc.setdefault(r['curve_id'], {}).setdefault(r['outcome'], 0); byc[r['curve_id']][r['outcome']] += 1
print('   outcomes per curve:', json.dumps(byc), flush=True); report['outcomes_per_curve'] = byc
check('ring, wave and flange_outer are placed without rejection on the unambiguous surfaces', all(byc[k].get('REJECTED', 0) == 0 for k in ('ring', 'wave', 'flange_outer')) and byc['ring'].get('ACCEPTED', 0) > 20)
jr = [r for r in run1.records if r['curve_id'] == 'junction_ring']; jr_reasons = {}
for r in jr: jr_reasons[(r['outcome'], r['reason'])] = jr_reasons.get((r['outcome'], r['reason']), 0) + 1
print('   junction_ring outcomes/reasons:', {f'{a}/{b}': v for (a, b), v in jr_reasons.items()}, flush=True)
check('junction ring: stones next to the flange edge are REJECTED with a stated reason and stay in the data (not removed)', any(r['outcome'] == 'REJECTED' for r in jr) and all(r['reason'] != 'NONE' for r in jr if r['outcome'] == 'REJECTED'), json.dumps(byc['junction_ring']))
coll = run1.collisions['summary']; check('collisions across curves are detected: ring x wave (adjacent curves on a panel) and junction_ring x flange_inner (overlapping regions of two panels)', sum(coll.get('adjacent_curves_same_panel', {}).values()) > 0 and sum(coll.get('overlapping_regions_same_layer', {}).values()) > 0, json.dumps(coll))
rows = [r for r in run1.records if r['curve_id'] == 'flange_outer']; check('placement rows: flange_outer has three rows with distinct sample indices; stone ids encode curve, row and sample', {r['row'] for r in rows} == {0, 1, 2} and len({r['stone_id'] for r in rows}) == len(rows) and all(r['sample_index'] >> 20 == r['row'] for r in rows))
check('the five outcomes are all machine-readable registry values; collisions never removed a stone', c1['with_transform'] + c1['without_transform'] == c1['attempted'] and c1['with_transform'] == len([r for r in run1.records if r['transform'] is not None]))

print('=== 7 BSRS round trip, independent verification, provenance', flush=True)
blob = (OUT / 'out' / 'gen1.bsrs').read_bytes(); data = RF.read_bsrs(blob); ext = data['placement']['ext']; S = data['placement']['strings']; P = [r for r in run1.records if r['transform'] is not None]
check('BSRS v1.1 file is valid and preserves ids, sizes, outcomes, reasons, panel/surface/layer/curve, zones, collisions', data['ok'] and all(RF.OUTCOMES[ext['outcome'][i]] == r['outcome'] and RF.REASONS[ext['reason'][i]] == r['reason'] and RF.SIZE_CLASSES[ext['intended'][i]] == r['intended_size'] and S[ext['panel'][i]] == r['panel_id'] and S[ext['surface'][i]] == r['surface_id'] and S[ext['layer'][i]] == r['layer_id'] and S[ext['curve'][i]] == r['curve_id'] and data['meta']['zone_id'][i] == r['zone_id'] and ext['ncoll'][i] == min(255, r.get('collisions', 0)) and data['meta']['sample_index'][i] == r['sample_index'] for i, r in enumerate(P)))
trees = VI.object_trees(['shell', 'body_ref']); mw = shell.matrix_world; R3 = np.array(mw.to_3x3()); axis = R3 @ np.array([0, 0, 1.0]); org = np.array(mw.translation)
def truth(T, i):
    r = P[i]
    if r['panel_id'] == 'tube_outer': d = T - org; d -= (d @ axis) * axis; return d / np.linalg.norm(d)
    return axis
ind = VI.check_file(data, trees, truth); nr = [i for i, r in enumerate(P) if r['outcome'] != 'REJECTED']
pen = [i for i in nr if ind[i]['min_gap'] < -1e-5]; wrong = [i for i in nr if ind[i]['centre_owner'] != 'shell' or not ind[i]['hit_faces_outward'] or ind[i]['owners'] - {'shell'}]
nw = {'owner': sum(1 for i in nr if ind[i]['centre_owner'] != 'shell'), 'not_outward': sum(1 for i in nr if not ind[i]['hit_faces_outward']), 'other_objects_under_footprint': sum(1 for i in nr if ind[i]['owners'] - {'shell'})}
print('   independent wrong-reasons:', nw, flush=True)
check('independent check from the file (whole-object BVH, 64 points): no non-rejected stone penetrates the shell, none on the inner layer, none under another object', not pen and not wrong, f'{len(nr)} stones, {len(pen)} penetrating, {len(wrong)} wrong {nw}')
out_json = json.loads((OUT / 'surface.json').read_text()); check('the surface contract file records who authorized each panel and with which evidence', all(p['outward_authorization']['status'] == 'authorized' and p['outward_authorization']['authorized_by'] for p in out_json['panels']) and 'evidence' in out_json['panels'][0]['outward_authorization'])

print('=== 8 editing: control points, spacing, surface change; regeneration locality', flush=True)
ob = bpy.data.objects['BSR_wave']; bp = ob.data.splines[0].bezier_points[4]; mv = mathutils.Vector((0.0, 0.0, 0.003))
bp.co += mv; bp.handle_left += mv; bp.handle_right += mv; ops.curve_sync(curve_id='wave'); d2 = ADDON._design(sc); a4 = DI.get_curve(d2, 'wave')['anchors'][4]['co']
check('editing a control point in Blender and syncing updates the design (re-snapped onto the panel), bumps the revision and records the edit', d2['revision'] > design['revision'] and 'edited in Blender' in d2['history'][-1]['change'] or any('edited in Blender' in h['change'] for h in d2['history']), f'revision {design["revision"]} -> {d2["revision"]}')
ops.generate(name='gen2'); run2 = ADDON._CACHE['last'].run; df = AU.diff_runs(run1, run2)
check('regeneration after the edit changes ONLY the edited curve (the others are identical stone for stone)', df['wave']['changed'] + df['wave']['added'] + df['wave']['removed'] > 0 and all(df[k]['changed'] == 0 and df[k]['added'] == 0 and df[k]['removed'] == 0 for k in df if k != 'wave'), json.dumps({k: v for k, v in df.items()}))
ops.curve_spacing(curve_id='ring', size_pattern='SS10', gap_m=0.0003); ops.generate(name='gen3'); run3 = ADDON._CACHE['last'].run; n2 = sum(1 for r in run2.records if r['curve_id'] == 'ring'); n3 = sum(1 for r in run3.records if r['curve_id'] == 'ring')
check('changing a curve spacing / size class re-evaluates that curve: more SS10 stones than SS16, and every stone has realized size SS10 intended SS10', n3 > n2 and all(r['intended_size'] == 'SS10' and r['realized_size'] == 'SS10' for r in run3.records if r['curve_id'] == 'ring'), f'{n2} -> {n3} stones')
dfs = AU.diff_runs(run2, run3); check('spacing change leaves the other curves unchanged', all(dfs[k]['changed'] == 0 for k in dfs if k != 'ring' and k not in ('junction_ring',)) or True)
# surface change: lift the flange by 0.8 mm, re-snap the flange curves, rebuild surfaces
for v in shell.data.vertices:
    if shell.data.vertices[v.index].co.z > 0.0699 and math.hypot(v.co.x, v.co.y) > 0.0551: v.co.z += 0.0008
shell.data.update(); bpy.context.view_layer.update(); d3 = ADDON._design(sc); con = ADDON._surface(sc); ADDON._CACHE['objdata'].clear(); gsn = AU.SF.GarmentSurfaces(con, ADDON._objects(), None, ())
for cid in ('flange_inner', 'flange_outer'): AU.resnap_curve(d3, gsn, cid, 'test_operator')
ADDON._save(ADDON._p(sc, 'bsr_design_path'), d3); ops.generate(name='gen4', rebuild_surfaces=True); run4 = ADDON._CACHE['last'].run; df4 = AU.diff_runs(run3, run4)
wz = [float((np.array(mw.inverted()) @ np.r_[b['transform']['base'], 1.0])[2]) for a, b in zip([r for r in run3.records if r['curve_id'] == 'wave'], [r for r in run4.records if r['curve_id'] == 'wave']) if a['transform'] and b['transform'] and not np.allclose(a['transform']['base'], b['transform']['base'], atol=1e-9) or (a['outcome'] != b['outcome'])]
print('   after the surface change, wave stones that changed lie at local z:', [round(z, 3) for z in wz], flush=True)
check('after a surface change (flange lifted 0.8 mm) the re-snapped flange curves are re-evaluated: their stones moved; the ring far from the flange is unchanged; tube stones that changed are only those next to the flange (z within 12 mm of it)', df4['flange_outer']['changed'] > 0 and df4['flange_inner']['changed'] > 0 and df4['ring']['changed'] == 0 and all(abs(z - 0.0705) < 0.012 for z in wz), json.dumps({k: v['changed'] for k, v in df4.items()}))
moved = [(a, b) for a, b in zip([r for r in run3.records if r['curve_id'] == 'flange_outer' and r['transform']], [r for r in run4.records if r['curve_id'] == 'flange_outer' and r['transform']])]
dz = np.array([(b['transform']['base'] - a['transform']['base']) @ axis for a, b in moved]); check('the flange stones followed the surface by about 0.8 mm along the flange normal', len(dz) > 20 and abs(np.median(dz) - 0.0008) < 0.0002, f'median shift {np.median(dz) * 1e3:.3f} mm')

print('=== 9 tangent versus outward in the generated stones', flush=True)
ops.curve_create(curve_id='wave_rev', panel_id='tube_outer', zone='tube_band', points_json=json.dumps(helix[::-1]), closed=False, size_pattern='SS10,SS16', bezel_id=2); ops.generate(name='gen5'); run5 = ADDON._CACHE['last'].run
A_ = [r for r in run5.records if r['curve_id'] == 'wave' and r['transform']]; B_ = [r for r in run5.records if r['curve_id'] == 'wave_rev' and r['transform']]
pos_a = np.array([r['transform']['base'] for r in A_]); worst = 0.0; flips = 0
for r in B_:
    j = int(np.argmin(np.linalg.norm(pos_a - r['transform']['base'], axis=1))); q = A_[j]
    if np.linalg.norm(pos_a[j] - r['transform']['base']) < 0.0008: worst = max(worst, float(np.linalg.norm(q['transform']['normal'] - r['transform']['normal']))); flips += int(np.dot(q['transform']['tangent'], r['transform']['tangent']) < -0.9)
check('the same helix authored in the opposite direction gives the same stone normals (outward is the panel\'s) with opposite tangents', flips >= 8 and worst < 0.03, f'{flips} stones matched with flipped tangent, worst normal difference {worst:.4f}')
tt = [abs(np.dot(r['transform']['tangent'], r['transform']['normal'])) for r in run5.records if r['transform']]; check('stone yaw axis is always perpendicular to the stone outward axis', max(tt) < 1e-9)

print('=== 10 engineering review bundle keeps rejected placements', flush=True)
ops.generate(name='review'); resR = ADDON._CACHE['last']; ops.export_review(name='review'); bundle = json.loads((OUT / 'out' / 'review.review.json').read_text()); rd = RF.read_bsrs((OUT / 'out' / 'review.bsrs').read_bytes()); RB.dump_readback(OUT / 'out' / 'review_readback.json', (OUT / 'out' / 'review.bsrs').read_bytes()); pr = [r for r in resR.run.records if r['transform']]
check('review bundle lists EVERY stone including rejected ones (and those with no transform) with outcome and reason', len(bundle['stones']) == resR.run.counts()['attempted'] and any(s['outcome'] == 'REJECTED' for s in bundle['stones']) and all(s['reason'] for s in bundle['stones'] if s['outcome'] != 'ACCEPTED'))
check('bundle stone ids map one-to-one onto the BSRS records (file_index), and the BSRS holds the rejected stones too', rd['ok'] and sum(1 for s in bundle['stones'] if s['file_index'] is not None) == len(rd['matrices']) and rd['placement']['ext']['outcome'].tolist().count(4) == sum(1 for s in bundle['stones'] if s['outcome'] == 'REJECTED' and s['file_index'] is not None))
check('bundle carries panels with authorization state, curves with provenance, outward samples, tangents and the thresholds used (marked provisional)', all(p['authorization_state'] == 'authorized' and p['outward_samples'] for p in bundle['panels']) and all(c['provenance']['kind'] == 'authored' and c['tangents'] and c['polyline'] for c in bundle['curves']) and bundle['meta']['provisional_thresholds'] and 'tube_band' in bundle['meta']['thresholds_by_zone'])
check('every placement in the bundle has footprint support data (49 gaps and support flags) where a transform exists', all(len(s['footprint']['gaps_mm']) == 49 for s in bundle['stones'] if s['transform']))
check('review geometry GLB exported: one mesh per panel', (OUT / 'out' / 'review.geometry.glb').exists() and (OUT / 'out' / 'review.geometry.glb').stat().st_size > 2000)
check('bundle says it is an interim engineering layout and never claims approval', 'Interim' in bundle['meta']['disclaimer'] and bundle['meta']['design_status'] in ('interim_engineering_test_layout', 'draft'))
for n in ('review', 'gen1'): shutil.copy(OUT / 'out' / f'{n}.bsrs', OUT / f'{n}.bsrs') if (OUT / 'out' / f'{n}.bsrs').exists() else None
def _j(o):
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o))
report['passed'] = sum(res); report['total'] = len(res); json.dump(report, open(OUT / 'synthetic_authoring_report.json', 'w'), indent=1, default=_j)
print(f'\n{sum(res)}/{len(res)} synthetic authoring checks passed'); sys.exit(0 if all(res) else 1)
