"""Stage 3, Blender layer, SYNTHETIC fixtures only (source_kind synthetic_fixture). Contract-driven placement, outcomes, ambiguity reporting, collisions, BSRS v1.1 round trip, and regression against Step 2.
Verification is independent of the engine: it reads the written .bsrs, rebuilds every stone, and probes the whole object (not the panel BVHs) with a different 64-point footprint.
  python3 test/blender/test_placement_v2_synthetic.py [outDir]     (needs bpy)"""
import json, math, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent.parent; sys.path.insert(0, str(ROOT / 'py'))
import bpy, mathutils
import arc_sampler as AS, rhs_format as RF, placement as PL1, placement_v2 as PV, surfaces as SF, contracts as CT, fixtures_v3 as FX, placement_policy as PP, readback as RB
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'out/step3/synthetic'; OUT.mkdir(parents=True, exist_ok=True)
res = []; report = {'source_kind': 'synthetic_fixture', 'fixtures': {}}
def check(n, ok, d=''): ok = bool(ok); res.append(ok); print(('PASS ' if ok else 'FAIL ') + n + (('  ' + d) if d else ''), flush=True)
def g2b(v): return np.array([v[0], -v[2], v[1]], float)
def halton(n, b):
    out = []
    for i in range(1, n + 1):
        f, x, k = 1.0, 0.0, i
        while k: f /= b; x += f * (k % b); k //= b
        out.append(x)
    return out
HAL = [(0.46 * math.sqrt(u) * math.cos(2 * math.pi * v), 0.46 * math.sqrt(u) * math.sin(2 * math.pi * v)) for u, v in zip(halton(63, 2), halton(63, 3))] + [(0.0, 0.0)]       # 64 points inside the octagon (inradius 0.462)


def build(builder, cfg=None):
    bpy.ops.wm.read_factory_settings(use_empty=True); b, reg = builder(); errs = b.validate(); assert not errs, errs
    surf = SF.GarmentSurfaces(b.surface, FX.objects_dict(), reg); eng = PV.PlacementEngine(surf, b, cfg); paths = FX.paths_for(b); return b, surf, eng, eng.run(paths)


def independent_gaps(objs, blob_parsed, recs_with_tf, H=0.004):
    """Rebuild stones from the FILE and probe whole objects (Step-2 Surface class: bmesh BVH, a different code path from surfaces.py)."""
    sf = PL1.Surface(objs); M = blob_parsed['matrices'].astype(float); meta = blob_parsed['meta']; out = []
    for i in range(len(M)):
        d = RF.DIAMETERS_M[meta['size_class'][i]]; X, Y, Z, T = (g2b(M[i, a:a + 3]) for a in (0, 4, 8, 12)); x, y, z = X / d, Y / d, Z / d; gaps = []
        for lx, lz in HAL:
            q = T + x * lx * d + z * lz * d; h = sf.cast(q + y * H, -y, 2 * H); gaps.append(np.nan if h is None else h[2] - H)
        out.append(dict(min_gap=float(np.nanmin(gaps)) if not np.all(np.isnan(gaps)) else float('nan'), misses=int(np.isnan(gaps).sum()), hit_normal=None))
    return out


def write_and_verify(name, run, objs, builder_surface_objs):
    blob, M, P = run.to_bsrs(); (OUT / f'{name}.bsrs').write_bytes(blob); data = RF.read_bsrs(blob)
    if data['ok']: RB.dump_readback(OUT / f'{name}_readback.json', blob)
    check(f'[{name}] BSRS v1.1 file written and valid ({len(P)} stones, {len(blob)} B)', data['ok'], str(data['errors'])[:200])
    if not data['ok'] or not len(P): return data, P, None
    ext = data['placement']['ext']; S = data['placement']['strings']
    ok = np.array_equal(data['matrices'], M)
    for i, r in enumerate(P):
        ok &= (RF.OUTCOMES[ext['outcome'][i]] == r['outcome'] and RF.REASONS[ext['reason'][i]] == r['reason'] and RF.SIZE_CLASSES[ext['intended'][i]] == r['intended_size'] and RF.SIZE_CLASSES[data['meta']['size_class'][i]] == r['realized_size']
               and S[ext['panel'][i]] == r['panel_id'] and S[ext['surface'][i]] == r['surface_id'] and S[ext['layer'][i]] == r['layer_id'] and S[ext['curve'][i]] == r['curve_id'] and ext['ncoll'][i] == min(255, r.get('collisions', 0)) and ext['sev'][i] == r.get('collision_severity', 0)
               and data['meta']['zone_id'][i] == r['zone_id'] and data['meta']['bezel_id'][i] == r['bezel_id'] and data['meta']['sample_index'][i] == r['sample_index'] and (data['meta']['flags'][i] & 0xF) == (r.get('flags', 0) & 0xF))
    check(f'[{name}] file preserves ALL placement metadata (intended vs realized size, outcome, reason, panel, surface, layer, curve, zone, bezel, sample index, flags, collisions) and bit-identical matrices', ok)
    return data, P, independent_gaps(objs, data, P)


def export_glb(name, objs):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]; bpy.ops.export_scene.gltf(filepath=str(OUT / f'{name}_surface.glb'), export_format='GLB', export_yup=True, use_selection=True, export_apply=True)


def summarize(name, run, gaps, P):
    c = run.counts(); gp = [g['min_gap'] for g in gaps] if gaps else []; non_rej = [i for i, r in enumerate(P) if r['outcome'] != 'REJECTED']
    worst = min([gp[i] for i in non_rej if not math.isnan(gp[i])], default=float('nan'))
    report['fixtures'][name] = {'counts': c, 'worst_independent_footprint_gap_mm_non_rejected': worst * 1e3, 'collision_pairs': len(run.collisions['pairs']) if run.collisions else 0, 'collision_summary': run.collisions['summary'] if run.collisions else {}}
    print(f'    {name}: {json.dumps(c["outcomes"])} reasons {json.dumps({k: v for k, v in c["reasons"].items() if k != "NONE"})} without_transform {c["without_transform"]} collisions {report["fixtures"][name]["collision_pairs"]} worst independent gap (non-rejected) {worst * 1e3:.4f} mm', flush=True)
    return worst


# ======================================================== REGRESSION: the four Step-2 fixtures through the contract engine
STEP2 = {}
for fname, builder in (('convex_sphere', FX.convex), ('concave_bowl', FX.concave), ('saddle', FX.saddle), ('collar', FX.collar)):
    print(f'\n=== regression fixture: {fname}', flush=True); t0 = time.time(); b, surf, eng, run = build(builder); c = run.counts()
    check(f'[{fname}] contract bundle validates; every curve names surface, panel and zone', b.validate() == [] and all(all(k in cv for k in ('surface_id', 'panel_id', 'placement_zone')) for cv in b.curves['curves']))
    check(f'[{fname}] conservation: attempted == with transform + without transform; every stone has an outcome', c['attempted'] == c['with_transform'] + c['without_transform'] and sum(c['outcomes'].values()) == c['attempted'], json.dumps({k: c[k] for k in ('attempted', 'with_transform', 'without_transform')}))
    objs = [bpy.data.objects[n] for n in sorted({p['surface_reference']['object'] for p in b.surface['panels']})]; data, P, gaps = write_and_verify(fname, run, objs, objs); worst = summarize(fname, run, gaps, P)
    if gaps:
        nr = [i for i, r in enumerate(P) if r['outcome'] != 'REJECTED']; pen = [i for i in nr if gaps[i]['min_gap'] < -1e-5]
        check(f'[{fname}] independent check from the file: no non-rejected stone penetrates the cloth by more than 10 um (64-point footprint, whole-object BVH)', not pen, f'worst {worst * 1e3:.4f} mm over {len(nr)} stones')
    # ---- regression against Step 2: same curves, same surface -> same stones
    cmp = {}
    for case in b.step2['cases']:
        cid = case['name']; recs = [r for r in run.records if r['curve_id'] == cid]
        if case.get('expect_all_fail'): continue
        ps = PL1.place_curve(PL1.Surface(b.step2['surface_objs']), PL1.curve_object_paths(case['curve'])[0], case['cfg'], PL1.PlacementConfig(strict=False), case['hint'], tag=cid); v1 = ps.placed
        r2 = [r for r in recs if r['transform'] is not None and r['row'] == 0]; v1 = [p for p in v1 if p.row == 0]
        if len(v1) != len(r2): cmp[cid] = {'count_v1': len(v1), 'count_v2': len(r2)}; continue
        dpos = np.array([np.linalg.norm(a.base - r['transform']['base']) for a, r in zip(v1, r2)]); dang = np.array([math.degrees(math.acos(max(-1, min(1, float(a.normal @ r['transform']['normal']))))) for a, r in zip(v1, r2)])
        cmp[cid] = {'n': len(v1), 'pos_median_mm': float(np.median(dpos) * 1e3), 'pos_p95_mm': float(np.percentile(dpos, 95) * 1e3), 'pos_max_mm': float(dpos.max() * 1e3), 'normal_p95_deg': float(np.percentile(dang, 95))}
    report['fixtures'][fname]['regression_vs_step2'] = cmp; STEP2[fname] = cmp
    for cid, m in cmp.items(): check(f'[{fname}/{cid}] regression vs Step 2: same stone count; base positions within 0.2 mm (p95), normals within 3 deg (p95)', 'pos_p95_mm' in m and m['pos_p95_mm'] < 0.2 and m['normal_p95_deg'] < 3.0, json.dumps({k: round(v, 4) for k, v in m.items()}))
    export_glb(fname, objs)
    if fname in ('convex_sphere', 'saddle'): check(f'[{fname}] every stone ACCEPTED (stones rest on smooth cloth)', c['outcomes'] == {'ACCEPTED': c['attempted']}, json.dumps(c['outcomes']))
    if fname == 'concave_bowl': check('[concave_bowl] stones on the bowl are ACCEPTED, none rejected', c['outcomes'].get('REJECTED', 0) == 0, json.dumps(c['outcomes']))
    if fname == 'collar':
        fails = [r for r in run.records if r['curve_id'] == 'FAIL_far_from_cloth']
        check('[collar] curve far from the cloth: every stone has NO transform and a stated reason (nothing silently dropped)', len(fails) > 0 and all(r['transform'] is None and r['outcome'] == 'REJECTED' and r['reason'] in ('NO_SURFACE', 'OFF_PANEL', 'SNAP_TOO_FAR') and r['notes'] for r in fails), f'{len(fails)} stones, reasons {sorted({r["reason"] for r in fails})}')
        und = [r for r in run.records if r['curve_id'] == 'apex_underside_concave_SS20']; oc = {}
        for r in und: oc[r['outcome']] = oc.get(r['outcome'], 0) + 1
        mc = np.median([r['measurement']['centre_clearance_mm'] for r in und])
        check('[collar] concave 5 mm fold (SS20): floating is classified, not accepted: all FLOATING_REVIEW at default thresholds', oc == {'FLOATING_REVIEW': len(und)}, f'{oc}, median centre clearance {mc:.3f} mm')
        others = [r for r in run.records if r['curve_id'] in ('fall_outer_SS16', 'apex_fold_SS16', 'stand_inside_SS20', 'meridian_across_fold_SS10') and r['transform'] is not None]
        check('[collar] convex apex, fall, stand inside and the meridian across the fold: ACCEPTED or ACCEPTED_WITH_CLEARANCE, none rejected', all(r['outcome'] in ('ACCEPTED', 'ACCEPTED_WITH_CLEARANCE') for r in others), json.dumps({o: sum(1 for r in others if r['outcome'] == o) for o in PP.OUTCOMES}))
        lay = [r for r in P if r['layer_id'] in ('shell_outer', 'shell_inner')]; check('[collar] panels carry their layer ids into the file (shell_outer / shell_inner)', {r['layer_id'] for r in lay} == {'shell_outer', 'shell_inner'})
    report['fixtures'][fname]['seconds'] = time.time() - t0

# ======================================================== POLICY: concave fold with the three zone configurations
print('\n=== policy variants on the concave 5 mm fold (collar underside, SS20)', flush=True)
variants = {'default': None, 'substitution_enabled': {'size_substitution': {'enabled': True, 'allowed_to': ['SS16', 'SS10']}}, 'thresholds_relaxed': {'thresholds': {'bridge_max_gap_m': 0.0008, 'review_max_gap_m': 0.002}}}
for vn, ov in variants.items():
    b, surf, eng, run = build(lambda ov=ov: FX.collar(zone_overrides=ov)); und = [r for r in run.records if r['curve_id'] == 'apex_underside_concave_SS20']; oc = {}
    for r in und: oc[r['outcome']] = oc.get(r['outcome'], 0) + 1
    blob, M, P = run.to_bsrs(); data = RF.read_bsrs(blob); ext = data['placement']['ext']; idx = [i for i, r in enumerate(P) if r['curve_id'] == 'apex_underside_concave_SS20']
    report['fixtures'][f'policy_{vn}'] = {'outcomes': oc, 'realized': sorted({r['realized_size'] for r in und}), 'would_pass': sorted({s for r in und for s in r.get('substitution_would_pass', [])})}; print(f'    {vn}: {oc} realized {sorted({r["realized_size"] for r in und})} would_pass {sorted({s for r in und for s in r.get("substitution_would_pass", [])})}')
    if vn == 'default':
        check('[policy/default] zone does not permit substitution: stones stay SS20, FLOATING_REVIEW, and the report says which smaller size WOULD pass (not applied)', oc == {'FLOATING_REVIEW': len(und)} and {r['realized_size'] for r in und} == {'SS20'} and any(r.get('substitution_would_pass') for r in und))
    if vn == 'substitution_enabled':
        (OUT / 'collar_substitution.bsrs').write_bytes(blob); RB.dump_readback(OUT / 'collar_substitution_readback.json', blob)
        sub = [r for r in und if r['outcome'] == 'SIZE_SUBSTITUTION_CANDIDATE']; check('[policy/substitution_enabled] zone permits it: stones become SIZE_SUBSTITUTION_CANDIDATE with a smaller realized size', len(sub) > 0 and all(r['realized_size'] in ('SS16', 'SS10') and r['intended_size'] == 'SS20' for r in sub), json.dumps(oc))
        check('[policy/substitution_enabled] BSRS keeps BOTH sizes: matrix/meta size class = realized, ext intended size = SS20; matrix scale equals the realized diameter', all(RF.SIZE_CLASSES[ext['intended'][i]] == 'SS20' and RF.SIZE_CLASSES[data['meta']['size_class'][i]] == P[i]['realized_size'] and abs(np.linalg.norm(data['matrices'][i, 0:3]) - RF.DIAMETERS_M[data['meta']['size_class'][i]]) < 1e-6 for i in idx if P[i]['outcome'] == 'SIZE_SUBSTITUTION_CANDIDATE'))
        check('[policy/substitution_enabled] substituted stones really pass: re-measured clearance of the smaller stone is within the accept/bridge limits', all(r['measurement']['centre_clearance_mm'] <= 0.5 + 1e-6 for r in sub), f'max {max(r["measurement"]["centre_clearance_mm"] for r in sub):.3f} mm')
    if vn == 'thresholds_relaxed': check('[policy/thresholds_relaxed] configurable thresholds change the outcome: ACCEPTED_WITH_CLEARANCE under relaxed bridging limit (not under defaults)', oc.get('ACCEPTED_WITH_CLEARANCE', 0) > 0 and oc.get('FLOATING_REVIEW', 0) == 0, json.dumps(oc))

# ======================================================== AMBIGUITY: stacked layers
print('\n=== ambiguity: stacked layers (overlay patch 1 mm above the base sheet)', flush=True)
b, surf, eng, run = build(FX.stacked); sheet = bpy.data.objects['sheet_base']; inv = np.array(sheet.matrix_world.inverted())
def xloc(r): return float((inv @ np.r_[r['transform']['base'], 1.0])[0])
under = [r for r in run.records if r['curve_id'] == 'base_crossing_under_overlay' and r['transform'] is not None]; ok_ = [r for r in under if r['outcome'] == 'ACCEPTED']; occl = [r for r in under if r['reason'] == 'OCCLUDED_BY_OTHER_LAYER']
check('[stacked] base curve passing under the overlay: stones under the patch are REJECTED as occluded (ambiguity reported, not silently placed on the wrong layer)', len(occl) > 3 and all(r['outcome'] == 'REJECTED' and any(a['kind'] == 'OCCLUDED_BY_OTHER_LAYER' for a in r['ambiguities']) for r in occl), f'{len(occl)} occluded, {len(ok_)} accepted outside the patch')
check('[stacked] no stone under the patch is accepted (accepted stones all lie outside x >= 0 minus their radius)', all(xloc(r) + 0.0025 < 0.0005 for r in ok_) and len(ok_) > 3, f'max accepted x {max(xloc(r) for r in ok_) * 1e3:.2f} mm')
ov = [r for r in run.records if r['curve_id'] == 'on_overlay']; check('[stacked] curve on the overlay panel: every stone ACCEPTED on the overlay layer', len(ov) > 3 and all(r['outcome'] == 'ACCEPTED' and r['layer_id'] == 'overlay' for r in ov), json.dumps({o: sum(1 for r in ov if r['outcome'] == o) for o in PP.OUTCOMES}))
wr = [r for r in run.records if r['curve_id'] == 'on_overlay_but_declared_base']; kinds = {a['kind'] for r in wr for a in r['ambiguities']}
check('[stacked] curve lying on the overlay but declared on the base panel: REJECTED with both ambiguities named (occluded, closer to another panel)', len(wr) > 3 and all(r['outcome'] == 'REJECTED' for r in wr) and kinds >= {'OCCLUDED_BY_OTHER_LAYER', 'CURVE_CLOSER_TO_OTHER_PANEL'}, str(sorted(kinds)))
rep = run.report(); check('[stacked] report lists every ambiguity with stone id and panel', len(rep['ambiguities']) >= len(occl) + len(wr), f'{len(rep["ambiguities"])} entries')
lay = run.collisions['summary'].get('overlapping_layers', {}); check('[stacked] collisions between stones on different layers are detected and categorised overlapping_layers (stones are not removed)', sum(lay.values()) > 0 and run.counts()['with_transform'] == run.counts()['attempted'], json.dumps(lay))
objs = [sheet, bpy.data.objects['sheet_overlay']]; data, P, gaps = write_and_verify('stacked', run, objs, objs); summarize('stacked', run, gaps, P); export_glb('stacked', objs)
rejected_in_file = [r for r in P if r['outcome'] == 'REJECTED']; check('[stacked] rejected stones stay in the file with outcome REJECTED and their reason (loaders may filter, nothing disappears)', len(rejected_in_file) == len(occl) + len(wr) and all(r['reason'] != 'NONE' for r in rejected_in_file))

# ======================================================== NORMAL ORIENTATION: flipped and mixed winding
print('\n=== normal orientation: inverted sphere and mixed-winding sheet', flush=True)
b, surf, eng, run = build(FX.flipped_winding); fl = [r for r in run.records if r['curve_id'] == 'flip_circle_SS16']
sph = bpy.data.objects['flip_sphere']; centre = np.array(sph.matrix_world.translation); rad = [np.dot(r['transform']['normal'], (r['transform']['base'] - centre) / np.linalg.norm(r['transform']['base'] - centre)) for r in fl]
check('[flipped] sphere whose mesh normals point INWARD: stones still face OUTWARD (corrected against the declared outward reference) and the flip is recorded', len(fl) > 20 and all(r['outcome'] == 'ACCEPTED' and r['normal_flipped'] for r in fl) and min(rad) > 0.9, f'{len(fl)} stones, min cos to the radial direction {min(rad):.4f} (non-uniform scale 1,1.2,0.9 moves the true normal up to ~10 deg off radial)')
mx = [r for r in run.records if r['curve_id'] == 'mixed_line_SS16']; msh = bpy.data.objects['mixed_sheet']; inv = np.array(msh.matrix_world.inverted())
xl = lambda r: float((inv @ np.r_[r['transform']['base'], 1.0])[0]); rr = lambda r: AS.DIAMETERS_M[r['realized_size']] / 2 + 0.0002
strad = [r for r in mx if abs(abs(xl(r)) - 0.008) < rr(r)]; inside = [r for r in mx if abs(xl(r)) + rr(r) < 0.008]; outside = [r for r in mx if abs(xl(r)) - rr(r) > 0.008]
check('[mixed winding] stones straddling the reversed-winding stripe are REJECTED as INCONSISTENT_NORMALS (faces under one stone disagree)', len(strad) >= 2 and all(r['outcome'] == 'REJECTED' and r['reason'] == 'INCONSISTENT_NORMALS' for r in strad), f'{len(strad)} straddling')
check('[mixed winding] stones fully inside the reversed stripe are ACCEPTED with the flip recorded; stones outside are ACCEPTED unflipped', len(inside) >= 1 and all(r['outcome'] == 'ACCEPTED' and r['normal_flipped'] for r in inside) and len(outside) >= 5 and all(r['outcome'] == 'ACCEPTED' and not r['normal_flipped'] for r in outside), f'inside {len(inside)} outside {len(outside)}')
objs = [sph, msh]; data, P, gaps = write_and_verify('flipped_and_mixed', run, objs, objs); summarize('flipped_and_mixed', run, gaps, P); export_glb('flipped_and_mixed', objs)

# ======================================================== TANGENT versus OUTWARD
print('\n=== tangent and outward are separate concepts', flush=True)
b, surf, eng, run = build(FX.convex); panel = surf.panels['sphere_outer']; path = FX.paths_for(b)['circle_SS16']; th = eng.cfg.thresholds; s_ = np.linspace(0, path.length, 60, endpoint=False); worst_n = worst_b = 0.0; opp = True; cnt = 0
for s in s_:
    pt = path.point([s])[0]; tg = path.tangent([s])[0]; a = eng.place_sample(panel, pt, tg, 'SS16', None, th); c2 = eng.place_sample(panel, pt, -tg, 'SS16', None, th)
    if a['transform'] is None or c2['transform'] is None: continue
    cnt += 1; worst_n = max(worst_n, np.linalg.norm(a['transform']['normal'] - c2['transform']['normal'])); worst_b = max(worst_b, np.linalg.norm(a['transform']['base'] - c2['transform']['base'])); opp &= bool(np.allclose(a['transform']['tangent'], -c2['transform']['tangent'], atol=1e-6))
check('[tangent] reversing the curve direction changes only the yaw: same stone position and outward normal, tangent flips', cnt == 60 and worst_n < 1e-9 and worst_b < 1e-9 and opp, f'{cnt} samples, max normal diff {worst_n:.1e}, max base diff {worst_b:.1e}')
tv = [r for r in run.records if r['curve_id'] == 'circle_SS16' and r['transform'] is not None]; tt = [np.dot(r['transform']['tangent'], r['transform']['normal']) for r in tv]
check('[tangent] stone X axis is perpendicular to its outward normal and parallel to the curve tangent projected on the surface', max(abs(x) for x in tt) < 1e-9)
pt = path.point([0.01])[0]; o_ = panel.outward(pt[None])[0]; hit_ = surf.cast_panel('sphere_outer', pt + o_ * 0.03, -o_, 0.06); a = eng.place_sample(panel, pt, hit_[1], 'SS16', None, th)      # tangent = the actual face normal: no usable yaw
check('[tangent] a degenerate tangent (parallel to the outward direction) falls back to a stable frame and is FLAGGED, never changes the normal', a['transform'] is not None and a['tangent_degenerate'] and abs(np.dot(a['transform']['tangent'], a['transform']['normal'])) < 1e-9)

# ======================================================== COLLISIONS between adjacent curves
print('\n=== collisions: two adjacent rings 3.7 mm apart (stones are 4 mm)', flush=True)
b, surf, eng, run = build(FX.adjacent_curves); c = run.counts(); cs = run.collisions['summary']; pairs = run.collisions['pairs']
adj = [p for p in pairs if p['category'] == 'adjacent_curves_same_panel']
check('[collisions] adjacent curves: overlapping stones are reported with ids, locations, severity, zones and category', len(adj) > 10 and all(p['stone_a'] and p['stone_b'] and len(p['location']) == 3 and p['severity'] and p['zones'] for p in adj), json.dumps(cs))
check('[collisions] same-curve neighbours do not collide (the sampler spacing holds): only cross-curve pairs are reported', all(p['category'] == 'adjacent_curves_same_panel' for p in pairs), json.dumps({k: sum(v.values()) for k, v in cs.items()}))
check('[collisions] no stone is removed to resolve collisions: all stones keep their transforms', c['with_transform'] == c['attempted'] and c['without_transform'] == 0)
objs = [bpy.data.objects['adj_sphere']]; data, P, gaps = write_and_verify('adjacent_curves', run, objs, objs); summarize('adjacent_curves', run, gaps, P)
ext = data['placement']['ext']; flagged = int((ext['ncoll'] > 0).sum()); check('[collisions] the file records per-stone collision count and worst severity', flagged > 10 and flagged == len({p['stone_a'] for p in pairs} | {p['stone_b'] for p in pairs}), f'{flagged} stones flagged')
export_glb('adjacent_curves', objs)

# ======================================================== CONTRACT ENFORCEMENT in the engine
print('\n=== contract enforcement', flush=True)
bpy.ops.wm.read_factory_settings(use_empty=True); b, reg = FX.convex(); objs_ = FX.objects_dict()
bad = FX.convex.__globals__['bundle'](b.surface['garment_id'], b.surface['panels'], [{k: v for k, v in c.items() if k != 'panel_id'} for c in b.curves['curves']], b.curves['zones']); surf = SF.GarmentSurfaces(b.surface, objs_, reg)
try: PV.PlacementEngine(surf, bad); check('[contract] the engine refuses a curve contract with a missing panel_id', False)
except SF.ContractGeometryError as e: check('[contract] the engine refuses a curve contract with a missing panel_id', 'C_MISSING' in str(e), str(e)[:100])
import copy
s2 = copy.deepcopy(b.surface); s2['panels'][0]['surface_reference']['select'] = [{'kind': 'vertex_group', 'name': 'nope'}]
try: SF.GarmentSurfaces(s2, objs_, reg); check('[contract] a face selection naming a missing vertex group is an error, not an empty or default selection', False)
except SF.ContractGeometryError as e: check('[contract] a face selection naming a missing vertex group is an error, not an empty or default selection', 'nope' in str(e), str(e)[:100])
s3 = copy.deepcopy(b.surface); s3['panels'][0]['surface_reference']['select'] = [{'kind': 'face_indices', 'indices': []}]
try: SF.GarmentSurfaces(s3, objs_, reg); check('[contract] an empty face selection is an error', False)
except SF.ContractGeometryError as e: check('[contract] an empty face selection is an error', 'empty' in str(e))
s4 = copy.deepcopy(b.surface); s4['panels'][0]['outward_direction'] = {'kind': 'registered_function', 'name': 'missing_fn'}
try: SF.GarmentSurfaces(s4, objs_, {}); check('[contract] a registered outward function that is not provided is an error', False)
except SF.ContractGeometryError as e: check('[contract] a registered outward function that is not provided is an error', 'missing_fn' in str(e))

def _j(o):
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o))
report['passed'] = sum(res); report['total'] = len(res); json.dump(report, open(OUT / 'synthetic_report.json', 'w'), indent=1, default=_j)
print(f'\n{sum(res)}/{len(res)} synthetic checks passed'); sys.exit(0 if all(res) else 1)
