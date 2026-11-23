"""FBL-008 development harness (NOT the acceptance run): rebuild only the collar on a baseline master, through the real
collar_builder.build_leaf_object, so a change can be tried in ~1 minute instead of a full bake.  The acceptance figures come from the real
build_archetype.py bake, never from this script.

  python -I -B dev_harness.py <camp_shirt|rugby_polo> [key=value ...]      (cfg overrides, e.g. seat_base_lift_m=0.0003)

Inputs: out_fbl_008/baseline/<key>_master.blend (final Tee), work_<key>/collider_posed.npz (the bake's body BVH), and the stand rows the baseline
bake handed to foldover_leaf (baseline/<key>_collar_leaf_debug.npz: COLLAR_ROWS after the 3 mm body push, which is idempotent).
Output: out_fbl_008/dev/<key>_dev.blend + <key>_dev.quality.json + <key>_dev_ring.npz."""
import sys, os, json, shutil
import numpy as np
import bpy
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.dirname(HERE); ROOT = os.path.dirname(OUT)
sys.path.insert(0, os.path.join(ROOT, 'engine', 'components'))
import collar_builder as CB

key = sys.argv[1]; spec = json.load(open(os.path.join(ROOT, 'configs', 'archetypes', key + '.json'))); cfg = dict(spec['construction']['foldover_collar'])
for kv in sys.argv[2:]:
    k, v = kv.split('='); cfg[k] = json.loads(v)
bpy.ops.wm.open_mainfile(filepath=os.path.join(OUT, 'baseline', key + '_master.blend'), load_ui=False)
old = bpy.data.objects['collar']; bpy.data.objects.remove(old, do_unlink=True)
D = np.load(os.path.join(OUT, 'work_' + key, 'collider_posed.npz')); t = []
for q in D['F']: t += [(int(q[0]), int(q[1]), int(q[2])), (int(q[0]), int(q[2]), int(q[3]))]
bvh = BVHTree.FromPolygons([tuple(v) for v in D['VB']], t)
rows = np.load(os.path.join(OUT, 'baseline', key + '_collar_leaf_debug.npz'))['stand_rows']
ob, rep = CB.build_leaf_object(bpy, bpy.data.objects['Tee'], rows, cfg, HERE, body_bvh=bvh)
g = bpy.data.collections['Garment']
for c in list(ob.users_collection): c.objects.unlink(ob)
g.objects.link(ob)
print(json.dumps(rep))
json.dump(dict(gate_passed=None, construction=dict(collar=rep)), open(os.path.join(HERE, key + '_dev.quality.json'), 'w'), indent=1)
if os.path.exists(os.path.join(HERE, 'collar_seat_ring.npz')): shutil.move(os.path.join(HERE, 'collar_seat_ring.npz'), os.path.join(HERE, key + '_dev_ring.npz'))
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, key + '_dev.blend'), compress=True)
