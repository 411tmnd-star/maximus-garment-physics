import os
import bpy,json
src=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/05_3D_ASSETS/glb/mannequin_full_rigged_122276_morphed.glb')); out=os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/03_DOCS_AND_EVIDENCE/evidence/mannequin/STAGE_1_VISUAL_BASELINE/mesh_edges.json'))
bpy.ops.import_scene.gltf(filepath=src); b=next(o for o in bpy.context.scene.objects if o.type=='MESH' and o.name=='mannequin'); vs=[list(b.matrix_world@v.co) for v in b.data.vertices]; es=[list(e.vertices) for e in b.data.edges];open(out,'w').write(json.dumps({'vertices':vs,'edges':es}))
