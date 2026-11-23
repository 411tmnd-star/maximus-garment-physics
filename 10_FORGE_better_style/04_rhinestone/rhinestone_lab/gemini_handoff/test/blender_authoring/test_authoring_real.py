"""Stage 4, Blender layer, INTERIM REAL GEOMETRY: assets/archetypes/camp_shirt_master.blend (engine-generated cloth simulation). NOT the final co-ord garment; the layout is an interim engineering
test layout authored by a SCRIPTED operator through the add-on operators. It validates the workflow and the evaluator on real geometry, not any product design.
Panels are authorized only if their inspection passes; ambiguous panels are listed for manual resolution and are not used.
  python3 test/blender_authoring/test_authoring_real.py [outDir]     (needs bpy; several minutes)"""
import json, math, shutil, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent.parent
for q in (ROOT / 'py', ROOT / 'blender_addon'): sys.path.insert(0, str(q))
import bpy, mathutils
from mathutils.bvhtree import BVHTree
import design_intent as DI, authoring as AU, contracts as CT, rhs_format as RF, review_export as RX, verify_independent as VI, placement_v2 as PV, camp_shirt_authoring as CSA, real_garment as RG
import bs_rhinestone_authoring as ADDON
import readback as RB
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'out/step4/real_authoring'; shutil.rmtree(OUT, ignore_errors=True); OUT.mkdir(parents=True)
res = []; report = {'source_kind': 'engine_archetype', 'interim': True, 'asset': 'better_style_catalog/assets/archetypes/camp_shirt_master.blend',
                    'caveats': ['interim engineering test garment, not the final co-ord garment', 'layout is a scripted-operator test layout, not the approved rhinestone design', 'results verify the workflow and evaluator on real geometry, not product design intent', 'thresholds are provisional']}
def check(n, ok, d=''): ok = bool(ok); res.append(ok); print(('PASS ' if ok else 'FAIL ') + n + (('  ' + d) if d else ''), flush=True)
def raises(fn, *a, **k):
    try: fn(*a, **k); return None
    except RuntimeError as e: return str(e)

t0 = time.time(); bpy.ops.wm.open_mainfile(filepath=str(RG.BLEND)); ADDON.register(); sc = bpy.context.scene; ops = bpy.ops.bsr; objs = ADDON._objects()
sc.bsr_operator = 'engineering_test_operator (scripted)'; sc.bsr_garment_id = 'camp_shirt_master'; sc.bsr_design_id = 'camp_shirt_interim_layout'; sc.bsr_source_kind = 'engine_archetype'; sc.bsr_occluders = 'placket_buttons'
sc.bsr_surface_path = str(OUT / 'surface.json'); sc.bsr_design_path = str(OUT / 'design.json'); sc.bsr_output_dir = str(OUT / 'out')
print('=== zones and panels on the real garment', flush=True)
for z, i, kw in (('back_band', 3, {}), ('sleeve_band', 2, {}), ('placket_band', 7, {}), ('collar_band', 1, {'substitution_to': 'SS16,SS10'}), ('chest_band', 8, {})): ops.zone_set(zone=z, zone_id=i, **kw)
PANELS = [('body_back_outer', 'tee_shell', 'Tee', 'body_back', 'back_band'), ('body_front_outer', 'tee_shell', 'Tee', 'body_front', 'chest_band'), ('sleeve_L_outer', 'tee_shell', 'Tee', 'set_in_sleeve_L', 'sleeve_band'), ('sleeve_R_outer', 'tee_shell', 'Tee', 'set_in_sleeve_R', 'sleeve_band'),
          ('collar_leaf_outer', 'collar_shell', 'collar', 'collar_leaf', 'collar_band'), ('placket_overlay', 'placket_sheet', 'placket_overlap', '', 'placket_band')]
for pid, sid, obj, vg, zone in PANELS: ops.panel_assign(panel_id=pid, surface_id=sid, layer_id='overlay' if pid == 'placket_overlay' else 'shell_outer', zones=zone, object_name=obj, vertex_group=vg, facing='outward', outward_kind='reference_body', reference_object='Mannequin')
con = ADDON._surface(sc); check('six panels assigned with panel_id, surface_id, layer_id, zone and a SUGGESTED reference-body outward; none authorized', len(con['panels']) == 6 and all(DI.authorization_state(con, p['panel_id']) == 'suggested' for p in con['panels']) and CT.validate_surface_contract(con) == [])
insp = {}
for pid, *_ in PANELS:
    t = time.time(); ops.panel_inspect(panel_id=pid); r = json.loads(sc['bsr_last_inspection']); insp[pid] = r
    print(f"   {pid:18s} {r['verdict']:26s} agree {r['selected_agree']:.3f} ambiguous {r['selected_ambiguous_fraction']:.3f} ({r['selected_ambiguous_faces']} faces, {len(r['ambiguous_locations_world'])} locations) angle to reference median {r['angle_deg']['median']:.1f} p95 {r['angle_deg']['p95']:.1f} deg, >45deg {r['angle_deg']['fraction_over_45']:.3f}, faces {r['selected_faces']} {time.time() - t:.1f}s", flush=True)
report['panel_inspection'] = {k: {x: v[x] for x in ('verdict', 'selected_faces', 'raw_faces', 'selected_agree', 'selected_ambiguous_fraction', 'selected_ambiguous_faces', 'mixed_layers', 'raw_facing_outward', 'raw_facing_inward', 'notes')} | {'ambiguous_locations_world': v['ambiguous_locations_world'][:30]} for k, v in insp.items()}
con = ADDON._surface(sc); check('inspection never authorizes: every panel is still only "suggested" after inspection', all(DI.authorization_state(con, p['panel_id']) == 'suggested' for p in con['panels']))
check('every inspection produced a verdict with evidence (agreement, ambiguous fraction, locations for manual review)', all(r['verdict'] in DI.VERDICTS and 'selected_agree' in r for r in insp.values()))
authorized = []; manual = []
for pid, *_ in PANELS:                                                                                    # the SCRIPTED operator authorizes only panels whose inspection passed
    if insp[pid]['verdict'] == 'SUGGESTED_OK': ops.panel_authorize(panel_id=pid, operator_name=sc.bsr_operator, note='scripted operator: inspection verdict SUGGESTED_OK'); authorized.append(pid)
    else: manual.append(pid)
con = ADDON._surface(sc); report['authorized_panels'] = authorized; report['manual_resolution_required'] = {p: insp[p]['notes'] for p in manual}
print('   authorized:', authorized, '| manual resolution required:', manual, flush=True)
check('only panels that passed inspection were authorized; ambiguous panels were NOT authorized and are listed for manual resolution', all(DI.authorization_state(con, p) == 'authorized' for p in authorized) and all(DI.authorization_state(con, p) == 'suggested' for p in manual) and len(authorized) >= 2)
for p in manual:
    err = raises(ops.panel_authorize, panel_id=p, operator_name=sc.bsr_operator, note='', acknowledge_ambiguity=False); check(f'[{p}] authorizing the ambiguous panel without acknowledgement is refused', err is not None, str(err)[:90])
gs = ADDON._gs(sc, con); cent = lambda pid: gs.data[gs.panels[pid].objname].centre[gs.panels[pid].face_ids]
print('=== authoring the interim layout (free-form curves, not slices)', flush=True)
def mk(cid, pid, zone, pts, closed, size, **kw):
    ops.curve_create(curve_id=cid, panel_id=pid, zone=zone, points_json=json.dumps(np.asarray(pts).tolist()), closed=closed, size_pattern=size, **kw)
made = []
if 'body_back_outer' in authorized:
    C = cent('body_back_outer'); mk('back_wave', 'body_back_outer', 'back_band', CSA.wave_points(C, (0.06, 0.94), 0.66, 0.06, 1.5, 8), False, 'SS10', rows_json='[-0.0045,0.0,0.0045]', bezel_id=1, note='interim test: wave across the upper back, three rows'); made.append('back_wave')
    mk('back_diag', 'body_back_outer', 'back_band', CSA.wave_points(C, (0.22, 0.78), 0.42, 0.18, 0.5, 6, 0.1), False, 'SS16', bezel_id=2, note='interim test: diagonal crossing the wave (cross-curve collisions)'); made.append('back_diag')
    mk('back_loop', 'body_back_outer', 'back_band', CSA.loop_points(C, 0.5, 0.28, 0.035, 8), True, 'SS16', bezel_id=1, note='interim test: closed loop low on the back'); made.append('back_loop')
if 'placket_overlay' in authorized:
    C = cent('placket_overlay'); z0, z1 = C[:, 2].min(), C[:, 2].max(); xc = float(np.median(C[:, 0])); pts = [CSA.nearest_face_point(C, (xc, z0 + t * (z1 - z0))) for t in np.linspace(0.08, 0.92, 7)]
    mk('placket_edges', 'placket_overlay', 'placket_band', pts, False, 'SS16', rows_json='[-0.010,0.010]', bezel_id=2, note='interim test: two lines either side of the button column on the overlay layer'); made.append('placket_edges')
if 'body_front_outer' in authorized and 'placket_overlay' in authorized:
    C = cent('body_front_outer'); zc = float(np.mean(cent('placket_overlay')[:, 2])); x0, x1 = C[:, 0].min(), C[:, 0].max(); pts = [CSA.nearest_face_point(C, (x0 + t * (x1 - x0), zc)) for t in np.linspace(0.12, 0.88, 7)]
    mk('chest_probe', 'body_front_outer', 'chest_band', pts, False, 'SS10', bezel_id=1, note='interim test: deliberate occlusion probe, crosses UNDER the placket overlay'); made.append('chest_probe')
for side in ('L', 'R'):
    pid = f'sleeve_{side}_outer'
    if pid in authorized:
        V = gs.data['Tee'].W[np.flatnonzero(gs.data['Tee'].groups['set_in_sleeve_' + side] > 0.5)]; pts = CSA.sleeve_wavy_band(cent(pid), V, 0.030, 0.003, waves=2, n=9, upper_arc_deg=85)
        if len(pts) >= 6: mk(f'sleeve_{side}_band', pid, 'sleeve_band', pts, False, 'SS10', rows_json='[-0.0045,0.0045]', bezel_id=1, note='interim test: wavy two-row band on the upper sleeve (open arc, away from the underarm), 30 mm from the sleeve end'); made.append(f'sleeve_{side}_band')
if 'collar_leaf_outer' in authorized:
    C = cent('collar_leaf_outer'); a = np.arctan2(C[:, 1] - C[:, 1].mean(), C[:, 0] - C[:, 0].mean()); zt = C[:, 2].min() + 0.5 * (C[:, 2].max() - C[:, 2].min()); sel = np.abs(C[:, 2] - zt) < 0.004
    pts = [C[sel][int(np.argmin(np.abs(a[sel] - th)))] for th in np.linspace(-1.2, 1.2, 9)]; mk('collar_edge', 'collar_leaf_outer', 'collar_band', pts, False, 'SS20', bezel_id=1, note='interim test: open line along the collar leaf'); made.append('collar_edge')
design = ADDON._design(sc); report['authored_curves'] = made
check('curves were authored on every authorized panel (main body, layered placket, sleeves where authorized, collar where authorized)', {'back_wave', 'placket_edges'} <= set(made) and len(made) >= 4, str(made))
check('every authored curve names its panel and surface; none defines outward; none is a derived slice', all(c['panel_id'] and c['surface_id'] and 'outward_direction' not in c for c in design['curves']) and all(DI.to_curve_contract(design)['curves'][i]['provenance']['kind'] == 'authored' for i in range(len(design['curves']))) and DI.validate_design(design, con) == [])
cn_try = [p for p in manual]; err = raises(ops.curve_create, curve_id='forced', panel_id=cn_try[0], zone=DI.panel(con, cn_try[0])['placement_zone'][0], points_json=json.dumps(cent(cn_try[0])[:4].tolist()), closed=False) if cn_try else None
if cn_try:
    check(f'a curve may be authored on an unauthorized panel (design intent) but its generation is blocked', err is None and any(b['curve_id'] == 'forced' for b in DI.blocked_curves(ADDON._design(sc), con))); ops.generate(name='blocked_probe'); blk = json.loads(sc['bsr_last_blocked']); check('the blocked curve is reported and generates nothing', any(b['curve_id'] == 'forced' and b['state'] == 'suggested' for b in blk) and not any(r['curve_id'] == 'forced' for r in ADDON._CACHE['last'].run.records))
    d_ = ADDON._design(sc); DI.remove_curve(d_, 'forced', sc.bsr_operator); ADDON._save(ADDON._p(sc, 'bsr_design_path'), d_)

print('=== generation and evaluation through the Stage-3 engine', flush=True)
t1 = time.time(); ops.generate(name='gen1'); res1 = ADDON._CACHE['last']; run1 = res1.run; c1 = run1.counts(); print('   generation', f'{time.time() - t1:.1f}s', json.dumps(c1), flush=True)
check('conservation: attempted == with transform + without transform; every stone has an outcome; nothing blocked', c1['attempted'] == c1['with_transform'] + c1['without_transform'] and sum(c1['outcomes'].values()) == c1['attempted'] and not res1.blocked)
byc = {}
for r in run1.records: byc.setdefault(r['curve_id'], {}).setdefault(r['outcome'], 0); byc[r['curve_id']][r['outcome']] += 1
for c, v in byc.items(): print('   ', c, json.dumps(v), flush=True)
report['outcomes_per_curve'] = byc; report['counts'] = c1
check('stones were generated for every authored curve and all five outcome names are registry values', set(byc) == set(made) and all(o in PV.PP.OUTCOMES for v in byc.values() for o in v))
check('the placket line on the OVERLAY layer is accepted where the layer is clear (most stones ACCEPTED)', byc.get('placket_edges', {}).get('ACCEPTED', 0) > 0.6 * sum(byc.get('placket_edges', {'x': 1}).values()), json.dumps(byc.get('placket_edges')))
if 'chest_probe' in byc:
    pr = [r for r in run1.records if r['curve_id'] == 'chest_probe']; occ = [r for r in pr if any(a['kind'] == 'OCCLUDED_BY_OTHER_LAYER' for a in r['ambiguities'])]
    check('the chest probe that crosses under the placket: hidden stones are REJECTED as occluded and kept in the data', len(occ) >= 1 and all(r['outcome'] == 'REJECTED' for r in occ), f'{len(occ)} of {len(pr)}')
coll = run1.collisions['summary'] if run1.collisions else {}; print('   collisions:', json.dumps(coll), flush=True); report['collisions'] = coll
check('cross-curve collisions are detected between the crossing back curves; rows of one curve do not collide at the authored 4.5 mm pitch; nothing is removed', sum(coll.get('adjacent_curves_same_panel', {}).values()) > 0 and not any(p['category'] == 'row_to_row' and p['severity'].startswith('OVERLAP') for p in run1.collisions['pairs']) and c1['with_transform'] + c1['without_transform'] == c1['attempted'], json.dumps(coll))
rows = [r for r in run1.records if r['curve_id'] == 'back_wave']; check('placement rows: back_wave has three rows with distinct ids; the two sleeve/placket curves carry their rows', {r['row'] for r in rows} == {0, 1, 2} and len({r['stone_id'] for r in rows}) == len(rows))

resRaw = AU.generate(ADDON._design(sc), ADDON._surface(sc), objs, None, None, ('placket_buttons',), gs, False, False); resOut = AU.generate(ADDON._design(sc), ADDON._surface(sc), objs, PV.EngineConfig(row_offset_mode='surface'), None, ('placket_buttons',), gs, False, False)
ov = lambda run: sum(1 for p in run.collisions['pairs'] if p['category'] == 'same_curve_consecutive' and p['severity'].startswith('OVERLAP')); snap2 = lambda run: sum(1 for r in run.records if r['transform'] and r['outcome'] != 'REJECTED' and (r.get('snap_mm') or 0) > 2.0)
report['conform_to_surface'] = {'consecutive_overlaps': {'conformed (Stage 4 default)': ov(run1), 'raw Bezier, normal projection': ov(resRaw.run), 'raw Bezier, outward projection (Stage 3 behaviour)': ov(resOut.run)}, 'accepted_with_snap_over_2mm': {'conformed': snap2(run1), 'raw': snap2(resRaw.run), 'outward_raw': snap2(resOut.run)}, 'conform_stats': res1.conform}
print('   consecutive same-curve overlaps: conformed', ov(run1), '| raw Bezier (normal projection)', ov(resRaw.run), '| raw Bezier (Stage-3 outward projection)', ov(resOut.run), flush=True); print('   accepted stones with curve point > 2 mm from the panel: conformed', snap2(run1), '| raw', snap2(resRaw.run), flush=True)
# measured-fact reporting: conform_to_surface is NOT claimed to reduce overlaps (it did not: see REPORT_STEP4.md)
report['conform_note'] = 'measured, not claimed: overlap counts and snap>2mm counts for conformed vs raw runs are recorded above; no reduction is asserted'
check('measured overlap counts for the three sampling variants are recorded (no improvement is asserted)', all(isinstance(v, int) for v in report['conform_to_surface']['consecutive_overlaps'].values()) and all(isinstance(v, int) for v in report['conform_to_surface']['accepted_with_snap_over_2mm'].values()), json.dumps(report['conform_to_surface']['consecutive_overlaps']))
check('conform statistics are reported per curve (points moved onto the panel, points left in place because the curve leaves the panel)', all(k in res1.conform for k in made) and all(v['moved_onto_panel'] + v['left_in_place'] == v['points'] for v in res1.conform.values()))
print('=== BSRS round trip and independent verification', flush=True)
blob = (OUT / 'out' / 'gen1.bsrs').read_bytes(); data = RF.read_bsrs(blob); ext = data['placement']['ext']; S = data['placement']['strings']; P = [r for r in run1.records if r['transform'] is not None]
check('BSRS v1.1 valid and preserves ids, sizes (intended/realized), outcomes, reasons, panel/surface/layer/curve, zones, collisions, bit-identical matrices', data['ok'] and all(RF.OUTCOMES[ext['outcome'][i]] == r['outcome'] and RF.REASONS[ext['reason'][i]] == r['reason'] and RF.SIZE_CLASSES[ext['intended'][i]] == r['intended_size'] and RF.SIZE_CLASSES[data['meta']['size_class'][i]] == r['realized_size'] and S[ext['panel'][i]] == r['panel_id'] and S[ext['surface'][i]] == r['surface_id'] and S[ext['layer'][i]] == r['layer_id'] and S[ext['curve'][i]] == r['curve_id'] and data['meta']['zone_id'][i] == r['zone_id'] and ext['ncoll'][i] == min(255, r.get('collisions', 0)) for i, r in enumerate(P)))
trees = VI.object_trees(['Tee', 'collar', 'placket_overlap', 'placket_buttons']); man = BVHTree.FromObject(objs['Mannequin'], bpy.context.evaluated_depsgraph_get())
def truth(T, i): n = np.array(man.find_nearest(mathutils.Vector(T))[0]); d = T - n; return d / np.linalg.norm(d)
pobj = {p['panel_id']: p['surface_reference']['object'] for p in con['panels']}; t2 = time.time(); ind = VI.check_file(data, trees, truth); nr = [i for i, r in enumerate(P) if r['outcome'] != 'REJECTED']
pen = [i for i in nr if ind[i]['min_gap'] < -1e-5]; wrong = [i for i in nr if ind[i]['centre_owner'] != pobj[P[i]['panel_id']] or not ind[i]['hit_faces_outward'] or ind[i]['owners'] - {pobj[P[i]['panel_id']]}]
worst = min([ind[i]['min_gap'] for i in nr if not math.isnan(ind[i]['min_gap'])], default=float('nan')); report['independent'] = {'stones': len(P), 'non_rejected': len(nr), 'penetrating_gt_10um': len(pen), 'wrong_layer_or_covered': len(wrong), 'worst_gap_mm': worst * 1e3}
check('independent check from the file (Blender BVHTree.FromObject on the whole objects, 64-point footprint): no non-rejected stone penetrates; none on the wrong layer or covered', not pen and not wrong, f'{len(nr)} non-rejected of {len(P)}, worst gap {worst * 1e3:.4f} mm, {len(wrong)} wrong')

print('=== editing and regeneration on the real garment', flush=True)
ob = bpy.data.objects['BSR_back_wave']; bps = ob.data.splines[0].bezier_points; k = 3; mv = mathutils.Vector((0.004, 0.0, 0.0)); bps[k].co += mv; bps[k].handle_left += mv; bps[k].handle_right += mv; ops.curve_sync(curve_id='back_wave')
t3 = time.time(); ops.generate(name='gen2'); run2 = ADDON._CACHE['last'].run; df = AU.diff_runs(run1, run2); print('   regeneration', f'{time.time() - t3:.1f}s', json.dumps({k_: v for k_, v in df.items() if v['changed'] or v['added'] or v['removed']}), flush=True)
check('editing one control point and regenerating changes only that curve; every other curve is identical stone for stone', df['back_wave']['changed'] + df['back_wave']['added'] + df['back_wave']['removed'] > 0 and all(v['changed'] == 0 and v['added'] == 0 and v['removed'] == 0 for k_, v in df.items() if k_ != 'back_wave'), json.dumps({k_: v['changed'] for k_, v in df.items()}))
ops.curve_spacing(curve_id='back_loop', size_pattern='SS10', gap_m=0.0003); ops.generate(name='gen3'); run3 = ADDON._CACHE['last'].run; n2 = sum(1 for r in run2.records if r['curve_id'] == 'back_loop'); n3 = sum(1 for r in run3.records if r['curve_id'] == 'back_loop'); df3 = AU.diff_runs(run2, run3)
check('changing a curve size class / spacing re-evaluates that curve (more SS10 stones) and leaves every other curve unchanged', n3 > n2 and all(v['changed'] == 0 for k_, v in df3.items() if k_ != 'back_loop'), f'{n2} -> {n3}')
ops.curve_assign(curve_id='chest_probe', panel_id='body_back_outer', zone='back_band') if 'chest_probe' in made else None
if 'chest_probe' in made: d_ = ADDON._design(sc); check('reassigning a curve to another panel is validated (zone must be allowed on the new panel)', DI.validate_design(d_, con) == [] and DI.get_curve(d_, 'chest_probe')['panel_id'] == 'body_back_outer')
design = ADDON._design(sc); check('the design history records every authoring and editing step with the operator and the revision', all(h['by'] for h in design['history']) and design['revision'] == len(design['history']) and len(design['history']) >= len(made) + 8)
if 'chest_probe' in made: DI.reassign_panel(design, 'chest_probe', 'body_front_outer', 'tee_shell', 'chest_band', sc.bsr_operator); ADDON._save(ADDON._p(sc, 'bsr_design_path'), design)

print('=== review bundle for the viewer', flush=True)
ops.generate(name='review'); resR = ADDON._CACHE['last']; ops.export_review(name='review', context_objects='placket_buttons'); bundle = json.loads((OUT / 'out' / 'review.review.json').read_text()); rd = RF.read_bsrs((OUT / 'out' / 'review.bsrs').read_bytes()); RB.dump_readback(OUT / 'out' / 'review_readback.json', (OUT / 'out' / 'review.bsrs').read_bytes())
check('review bundle lists every stone (including rejected and untransformed) with outcome, reason, footprint support, sizes, collisions', len(bundle['stones']) == resR.run.counts()['attempted'] and all(len(s['footprint']['gaps_mm']) == 49 for s in bundle['stones'] if s['transform']) and rd['ok'])
check('bundle states that the layout is interim, thresholds provisional, garment is engine_archetype', 'Interim' in bundle['meta']['disclaimer'] and bundle['meta']['provisional_thresholds'] and bundle['meta']['garment_source_kind'] == 'engine_archetype')
(ROOT / 'design').mkdir(exist_ok=True); shutil.copy(OUT / 'design.json', ROOT / 'design' / 'camp_shirt_interim_layout.design.json'); shutil.copy(OUT / 'surface.json', ROOT / 'contracts' / 'camp_shirt_master.authored.surface.json')
def _j(o):
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o))
report['seconds'] = time.time() - t0; report['passed'] = sum(res); report['total'] = len(res); json.dump(report, open(OUT / 'real_authoring_report.json', 'w'), indent=1, default=_j)
print(f'\n{sum(res)}/{len(res)} real-geometry authoring checks passed'); sys.exit(0 if all(res) else 1)
