import os
import json,struct
from pathlib import Path
p=Path(os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/03_DOCS_AND_EVIDENCE/evidence/mannequin/STAGE_3_1_REPAIR/STAGE_3_1_EXPORT_VALIDATION/mannequin_M1_122276_walk_v1.glb')));d=p.read_bytes();n,t=struct.unpack_from('<II',d,12);g=json.loads(d[20:20+n]);nodes=g['nodes'];target={i:x.get('name') for i,x in enumerate(nodes)};a=[]
for anim in g['animations']:
 for c in anim['channels']:
  if target.get(c['target'].get('node')) in ('upperarm01.L','upperarm01.R') and c['target'].get('path')=='rotation':
   sm=anim['samplers'][c['sampler']]; oi=sm['output']; acc=g['accessors'][oi];a.append({'animation':anim.get('name'),'bone':target[c['target']['node']],'accessor':oi,'count':acc.get('count'),'min':acc.get('min'),'max':acc.get('max')})
Path(os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/03_DOCS_AND_EVIDENCE/evidence/mannequin/STAGE_3_2_FIT_GLB_REMEDIATION/glb_tests/arm_rotation_accessors.json'))).write_text(json.dumps(a,indent=2));print(json.dumps(a,indent=2))
