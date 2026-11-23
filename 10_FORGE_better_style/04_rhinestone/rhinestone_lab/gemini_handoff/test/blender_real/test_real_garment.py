"""Stage 3, REAL garment geometry: assets/archetypes/camp_shirt_master.blend (engine archetype, cloth-simulated; NOT synthetic, NOT a production-approved pattern).
Contract-driven placement on the actual Tee (thick shell, vertex-group panels), collar and placket layers; independent verification from the written file against the objects themselves.
  python3 test/blender_real/test_real_garment.py [outDir]     (needs bpy; a few minutes)"""
import json, math, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent.parent; sys.path.insert(0, str(ROOT / 'py'))
import bpy, mathutils
from mathutils.bvhtree import BVHTree
import arc_sampler as AS, rhs_format as RF, placement_v2 as PV, surfaces as SF, contracts as CT, real_garment as RG, placement_policy as PP, stone_matrix as SM, readback as RB
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'out/step3/real'; OUT.mkdir(parents=True, exist_ok=True)
res = []; report = {'source_kind': 'engine_archetype', 'asset': 'better_style_catalog/assets/archetypes/camp_shirt_master.blend', 'caveats': ['cloth-simulated engine output, not a production-approved pattern', 'not the rhinestone co-ord shirt itself (same camp-shirt family)', 'curves are plane slices derived from the geometry, not authored by a pattern maker', 'contract written by the test author, not by a garment engineer']}
def check(n, ok, d=''): ok = bool(ok); res.append(ok); print(('PASS ' if ok else 'FAIL ') + n + (('  ' + d) if d else ''), flush=True)
def g2b(v): return np.array([v[0], -v[2], v[1]], float)
t0 = time.time(); bpy.ops.wm.open_mainfile(filepath=str(RG.BLEND)); objs = {o.name: o for o in bpy.data.objects}
print('opened', RG.BLEND.name, f'{time.time() - t0:.1f}s', flush=True)
sc = RG.surface_contract(); errs = CT.validate_surface_contract(sc); check('[real] surface contract validates', errs == [], str(errs[:3]))
t0 = time.time(); gs = SF.GarmentSurfaces(sc, objs, extra_occluders=('placket_buttons',)); report['panel_faces'] = gs.stats; report['unassigned_rest'] = {k: 'bvh' for k in gs.rest}
print('panels built', f'{time.time() - t0:.1f}s', json.dumps(gs.stats), 'rest', list(gs.rest), flush=True)
tee = gs.data['Tee']; total = tee.nf
check('[real] every panel selected a non-empty set of faces from the real mesh', all(v['faces'] > 1000 for v in gs.stats.values() if v['object'] == 'Tee') and gs.stats['placket_overlay']['faces'] > 100 and gs.stats['collar_leaf_outer']['faces'] > 1000, json.dumps({k: v['faces'] for k, v in gs.stats.items()}))
share = sum(v['faces'] for v in gs.stats.values() if v['object'] == 'Tee') / total; report['tee_outer_panel_share_of_faces'] = share
check('[real] Tee outer-layer panels cover about half of the shell faces (the other half is the inner layer + seams, kept as occluders)', 0.4 < share < 0.55, f'{share:.3f} of {total} faces')
# the declared reference must agree with winding on most faces (reported, not assumed)
o_ref = gs.panels['body_back_outer'].outward(tee.centre[gs.panels['body_back_outer'].face_ids[::400]]); agree = float(((tee.normal[gs.panels['body_back_outer'].face_ids[::400]] * o_ref).sum(1) > 0).mean())
check('[real] by construction the selected outer faces face away from the mannequin (sampled check)', agree == 1.0, f'{agree:.3f}')
derived = RG.derive_curves(gs); cc = RG.curve_contract(derived); bundle = CT.ContractBundle(sc, cc)
errs = bundle.validate(); check('[real] curve contract validates: every curve names surface, panel and zone', errs == [], str(errs[:3]))
(ROOT / 'contracts').mkdir(exist_ok=True); json.dump(sc, open(ROOT / 'contracts/camp_shirt_master.surface.json', 'w'), indent=1); json.dump(cc, open(ROOT / 'contracts/camp_shirt_master.curves.json', 'w'), separators=(',', ':'))
print('derived curves:', {k: (len(v['points']), 'closed' if v['closed'] else 'open', round(float(np.linalg.norm(np.diff(v['points'], axis=0), axis=1).sum()) * 1e3)) for k, v in derived.items()}, flush=True)
check('[real] curves were derived for every zone (sleeves, back, collar, placket, chest band)', {d['zone'] for d in derived.values()} >= {'sleeve_border', 'back_script', 'collar_perimeter', 'placket_line', 'chest_band'}, str(sorted({d['zone'] for d in derived.values()})))
t0 = time.time(); eng = PV.PlacementEngine(gs, bundle); paths = {c['curve_id']: AS.PolylinePath(np.array(c['source']['points'], float), closed=c['source']['closed']) for c in cc['curves']}; run = eng.run(paths); dt = time.time() - t0
cn = run.counts(); print('placement', f'{dt:.1f}s', json.dumps(cn), flush=True)
check('[real] conservation: attempted == with transform + without transform; every stone has an outcome', cn['attempted'] == cn['with_transform'] + cn['without_transform'] and sum(cn['outcomes'].values()) == cn['attempted'], json.dumps({k: cn[k] for k in ('attempted', 'with_transform', 'without_transform')}))
check('[real] stones were attempted on every curve', all(cr['attempted'] > 5 for cr in run.curve_reports.values()), json.dumps({k: v['attempted'] for k, v in run.curve_reports.items()}))
blob, M, P = run.to_bsrs(); (OUT / 'camp_shirt_master.bsrs').write_bytes(blob); data = RF.read_bsrs(blob)
check(f'[real] BSRS v1.1 written and valid ({len(P)} stones, {len(blob)} B)', data['ok'], str(data['errors'])[:200])
ext = data['placement']['ext']; S = data['placement']['strings']
ok = np.array_equal(data['matrices'], M) and all(RF.OUTCOMES[ext['outcome'][i]] == r['outcome'] and RF.REASONS[ext['reason'][i]] == r['reason'] and RF.SIZE_CLASSES[ext['intended'][i]] == r['intended_size'] and RF.SIZE_CLASSES[data['meta']['size_class'][i]] == r['realized_size']
                                                   and S[ext['panel'][i]] == r['panel_id'] and S[ext['surface'][i]] == r['surface_id'] and S[ext['layer'][i]] == r['layer_id'] and S[ext['curve'][i]] == r['curve_id'] and ext['ncoll'][i] == min(255, r.get('collisions', 0)) for i, r in enumerate(P))
check('[real] file preserves all placement metadata and bit-identical matrices', ok)
RB.dump_readback(OUT / 'camp_shirt_master_readback.json', blob)
# ---------------- independent verification: whole objects via Blender's own BVHTree.FromObject (not the panel BVHs), a different 64-point footprint
dg = bpy.context.evaluated_depsgraph_get(); trees = {n: BVHTree.FromObject(objs[n], dg) for n in ('Tee', 'collar', 'placket_overlap', 'placket_buttons')}; man = BVHTree.FromObject(objs['Mannequin'], dg)
def halton(n, b):
    out = []
    for i in range(1, n + 1):
        f, x, k = 1.0, 0.0, i
        while k: f /= b; x += f * (k % b); k //= b
        out.append(x)
    return out
HAL = [(0.46 * math.sqrt(u) * math.cos(2 * math.pi * v), 0.46 * math.sqrt(u) * math.sin(2 * math.pi * v)) for u, v in zip(halton(63, 2), halton(63, 3))] + [(0.0, 0.0)]
H = 0.004; ind = []
for i in range(len(P)):
    d = RF.DIAMETERS_M[data['meta']['size_class'][i]]; Mi = data['matrices'][i].astype(float); X, Y, Z, T = (g2b(Mi[a:a + 3]) for a in (0, 4, 8, 12)); x, y, z = X / d, Y / d, Z / d; gaps = []; owners = []
    for lx, lz in HAL:
        q = T + x * lx * d + z * lz * d; best = None
        for nm, tr in trees.items():
            h = tr.ray_cast(mathutils.Vector(q + y * H), mathutils.Vector(-y), 2 * H)
            if h[0] is not None and (best is None or h[3] < best[0]): best = (h[3], nm, np.array(h[1]))
        if best is None: gaps.append(np.nan); owners.append(None)
        else: gaps.append(best[0] - H); owners.append(best[1])
    c = owners[-1]; hn = None
    q0 = T; hh = None
    for nm, tr in trees.items():
        h = tr.ray_cast(mathutils.Vector(q0 + y * H), mathutils.Vector(-y), 2 * H)
        if h[0] is not None and (hh is None or h[3] < hh[0]): hh = (h[3], nm, np.array(h[1]))
    nref = np.array(man.find_nearest(mathutils.Vector(T))[0]); out_ref = (T - nref) / np.linalg.norm(T - nref)
    ind.append({'min_gap': float(np.nanmin(gaps)) if not np.all(np.isnan(gaps)) else float('nan'), 'centre_owner': hh[1] if hh else None, 'hit_faces_outward': bool(hh is not None and float(hh[2] @ out_ref) > 0), 'owners': set(o for o in owners if o)})
panel_obj = {p['panel_id']: p['surface_reference']['object'] for p in sc['panels']}
non_rej = [i for i, r in enumerate(P) if r['outcome'] != 'REJECTED']
pen = [i for i in non_rej if ind[i]['min_gap'] < -1e-5]; wrong = [i for i in non_rej if ind[i]['centre_owner'] != panel_obj[P[i]['panel_id']] or not ind[i]['hit_faces_outward'] or ind[i]['owners'] - {panel_obj[P[i]['panel_id']]}]
worst = min([ind[i]['min_gap'] for i in non_rej if not math.isnan(ind[i]['min_gap'])], default=float('nan'))
check('[real] independent check from the file: no non-rejected stone penetrates the real cloth by more than 10 um', not pen, f'{len(non_rej)} stones, worst footprint gap {worst * 1e3:.4f} mm, {len(pen)} penetrating')
check('[real] independent check: every non-rejected stone lies on the OUTER face of its intended object, and no footprint point is covered by another object (no hidden or wrong-layer stone accepted)', not wrong, f'{len(wrong)} wrong of {len(non_rej)}')
rej = [r for r in P if r['outcome'] == 'REJECTED']; unt = [r for r in run.records if r['transform'] is None]
rep0 = run.report(); na = rep0['not_accepted']; n_expected = sum(1 for r in run.records if r['outcome'] not in ('ACCEPTED', 'ACCEPTED_WITH_CLEARANCE'))
check('[real] every rejected / untransformed / review stone carries a reason AND is listed in the report (none silently dropped)', all(r['reason'] != 'NONE' for r in rej + unt) and len(na) == n_expected and len(rep0['without_transform']) == len(unt), f'{len(rej)} rejected with transform, {len(unt)} without transform, {len(na)} listed (incl. review)')
oc = {}
for r in run.records: oc[(r['curve_id'], r['outcome'])] = oc.get((r['curve_id'], r['outcome']), 0) + 1
per_curve = {}
for (c, o), n in oc.items(): per_curve.setdefault(c, {})[o] = n
for c, v in per_curve.items(): print('   ', c, json.dumps(v), flush=True)
pk = [r for r in run.records if r['curve_id'] == 'chest_across_placket']; occ = [r for r in pk if any(a['kind'] == 'OCCLUDED_BY_OTHER_LAYER' for a in r['ambiguities'])]
check('[real] the chest curve that crosses UNDER the placket: occluded stones are REJECTED and reported as OCCLUDED_BY_OTHER_LAYER (not placed on the hidden body layer)', len(occ) >= 1 and all(r['outcome'] == 'REJECTED' for r in occ), f'{len(occ)} occluded of {len(pk)}')
amb_kinds = {}
for r in run.records:
    for a in r['ambiguities']: amb_kinds[a['kind']] = amb_kinds.get(a['kind'], 0) + 1
print('   ambiguity kinds:', json.dumps(amb_kinds), flush=True)
coll = run.collisions; csum = coll['summary'] if coll else {}; print('   collisions:', json.dumps(csum), 'pairs', len(coll['pairs']) if coll else 0, 'candidates', coll['candidates'] if coll else 0, flush=True)
adj = [p for p in coll['pairs'] if p['category'] == 'adjacent_curves_same_panel'] if coll else []
ring_pairs = [p for p in adj if ('ring_a' in p['stone_a'] and 'ring_c' in p['stone_b']) or ('ring_c' in p['stone_a'] and 'ring_a' in p['stone_b'])]
check('[real] collisions: the deliberately overlapping sleeve rings (ring_a and ring_c, 1.5 mm apart) are detected and reported; the spaced ring_b does not collide with ring_a', len(ring_pairs) > 5 and not any(('ring_b' in p['stone_a'] and 'ring_a' in p['stone_b']) or ('ring_a' in p['stone_a'] and 'ring_b' in p['stone_b']) for p in coll['pairs']), f'{len(ring_pairs)} ring_a/ring_c pairs')
check('[real] no stone was removed to resolve collisions', cn['attempted'] == cn['with_transform'] + cn['without_transform'] and cn['with_transform'] == len(P))
zone_in_file = {int(z) for z in data['meta']['zone_id']}; check('[real] zone ids from the contract reach the file', zone_in_file <= {1, 2, 3, 7, 8} and len(zone_in_file) >= 4, str(sorted(zone_in_file)))
sub = [r for r in P if r['outcome'] == 'SIZE_SUBSTITUTION_CANDIDATE']; check('[real] size substitution happens only in the zone that enables it (collar_perimeter)', all(r['zone'] == 'collar_perimeter' for r in sub), f'{len(sub)} substituted')
# ---------------- patch GLB of the real garment around the stones (all layers), for the Three.js raycast re-check
stones_b = np.array([r['transform']['base'] for r in run.records if r['transform'] is not None]); created = []
for nm in ('Tee', 'collar', 'placket_overlap', 'placket_buttons'):
    od = gs.data[nm]; keep = np.zeros(od.nf, bool)
    for c0 in range(0, od.nf, 20000): keep[c0:c0 + 20000] = (((od.centre[c0:c0 + 20000, None, :] - stones_b[None]) ** 2).sum(2).min(1) < 0.008 ** 2)
    tri = od.tris[keep[od.tri_poly]]
    if len(tri) == 0: continue
    used, inv = np.unique(tri, return_inverse=True); me = bpy.data.meshes.new(nm + '_patch'); me.from_pydata(od.W[used].tolist(), [], inv.reshape(-1, 3).tolist()); me.update(); ob = bpy.data.objects.new(nm + '_patch', me); bpy.context.scene.collection.objects.link(ob); created.append(ob)
    report.setdefault('patch_faces', {})[nm] = int(len(tri))
bpy.ops.object.select_all(action='DESELECT')
for o in created: o.select_set(True)
bpy.context.view_layer.objects.active = created[0]; bpy.ops.export_scene.gltf(filepath=str(OUT / 'camp_shirt_master_patch.glb'), export_format='GLB', export_yup=True, use_selection=True)
print('patch GLB', report.get('patch_faces'), (OUT / 'camp_shirt_master_patch.glb').stat().st_size // 1024, 'KB', flush=True)
rep = run.report(); rep['source_kind'] = 'engine_archetype'; rep['per_curve_outcomes'] = per_curve; rep['ambiguity_kinds'] = amb_kinds
rep['independent_verification'] = {'stones_checked': len(P), 'non_rejected': len(non_rej), 'penetrating_gt_10um': len(pen), 'worst_footprint_gap_mm_non_rejected': worst * 1e3, 'wrong_layer_or_covered': len(wrong)}
rep['timings_s'] = {'placement': dt}; rep['caveats'] = report['caveats']; rep['panel_faces'] = gs.stats
def _j(o):
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, (np.ndarray,)): return o.tolist()
    if isinstance(o, set): return sorted(o)
    raise TypeError(type(o))
json.dump(rep, open(OUT / 'real_garment_report.json', 'w'), indent=1, default=_j)
print(f'\n{sum(res)}/{len(res)} real-garment checks passed'); sys.exit(0 if all(res) else 1)
