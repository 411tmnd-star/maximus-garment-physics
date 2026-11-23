import bpy, math, json, os
from mathutils import Vector
src = os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/05_3D_ASSETS/glb/mannequin_full_rigged_122276_morphed.glb'))
out = os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/03_governance_and_evidence/411_01_BETTER_STYLE_ACTIVE/03_DOCS_AND_EVIDENCE/evidence/mannequin/STAGE_1_VISUAL_BASELINE/VISUAL_EVIDENCE'))
bpy.ops.import_scene.gltf(filepath=src)
scene=bpy.context.scene
meshes=[o for o in scene.objects if o.type=='MESH']
armatures=[o for o in scene.objects if o.type=='ARMATURE']
# Canonical mannequin only; do not touch source data or save.
body=next((o for o in meshes if o.name.lower()=='mannequin'), meshes[0])
for o in scene.objects:
    o.hide_render = o not in [body]
# Evaluate stats on imported mesh datablock and scene bounds.
dg=bpy.context.evaluated_depsgraph_get(); ev=body.evaluated_get(dg); me=ev.to_mesh()
coords=[body.matrix_world @ v.co for v in me.vertices]
mins=[min(v[i] for v in coords) for i in range(3)]; maxs=[max(v[i] for v in coords) for i in range(3)]
meshinfo={'object':body.name,'vertices':len(me.vertices),'edges':len(me.edges),'polygons':len(me.polygons),'triangles':sum(max(0,len(p.vertices)-2) for p in me.polygons),'bounds_min':mins,'bounds_max':maxs,'dimensions':[maxs[i]-mins[i] for i in range(3)],'materials':[m.name if m else None for m in body.data.materials],'shape_keys': [k.name for k in body.data.shape_keys.key_blocks] if body.data.shape_keys else []}
ev.to_mesh_clear()
riginfo=[]
for a in armatures:
    riginfo.append({'object':a.name,'bones':len(a.data.bones),'bone_names':[b.name for b in a.data.bones]})
matinfo=[]
for m in bpy.data.materials:
    rec={'name':m.name,'nodes':[]}
    if m.use_nodes:
      for n in m.node_tree.nodes:
       nd={'type':n.bl_idname,'name':n.name}
       if n.type=='TEX_IMAGE' and n.image: nd['image']={'name':n.image.name,'packed':bool(n.image.packed_file),'filepath':n.image.filepath,'size':list(n.image.size)}
       if n.type=='BSDF_PRINCIPLED': nd['inputs']={s.name:(list(s.default_value) if hasattr(s.default_value,'__len__') else s.default_value) for s in n.inputs if s.name in ['Base Color','Roughness','Metallic']}
       rec['nodes'].append(nd)
    matinfo.append(rec)
report={'mesh':meshinfo,'armatures':riginfo,'materials':matinfo,'blender':bpy.app.version_string,'build_hash':bpy.app.build_hash.decode() if isinstance(bpy.app.build_hash,bytes) else str(bpy.app.build_hash)}
open(os.path.join(out,'scene_inventory.json'),'w').write(json.dumps(report,indent=2))
# temporary neutral studio world/floor/lights
scene.render.engine='BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in [i.identifier for i in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items] else 'BLENDER_EEVEE'
scene.eevee.taa_render_samples=16
scene.render.resolution_x=1200; scene.render.resolution_y=1600; scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG'; scene.render.film_transparent=False
scene.view_settings.view_transform='Standard'; scene.view_settings.look='None'; scene.view_settings.exposure=0; scene.view_settings.gamma=1
scene.world.color=(0.20,0.20,0.20)
# M0 flat neutral base color in linearized approximate color
m0=bpy.data.materials.new('TEMP_M0_clay'); m0.diffuse_color=(0.497,0.497,0.471,1); m0.use_nodes=True
bs=m0.node_tree.nodes.get('Principled BSDF'); bs.inputs['Base Color'].default_value=(0.497,0.497,0.471,1); bs.inputs['Roughness'].default_value=.78
# studio backdrop plane
bpy.ops.mesh.primitive_plane_add(size=200, location=(0,0,mins[2]-.03)); floor=bpy.context.object; floor.name='TEMP_floor'; floor.hide_render=False
fm=bpy.data.materials.new('TEMP_backdrop'); fm.diffuse_color=(.12,.12,.12,1); fm.use_nodes=True; fm.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(.12,.12,.12,1); fm.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=1
floor.data.materials.clear(); floor.data.materials.append(fm)
# Existing armature pose remains untouched. Define lights and reusable camera.
def area(name,loc,power,size,color):
 d=bpy.data.lights.new(name,'AREA'); d.energy=power; d.shape='DISK'; d.size=size; d.color=color
 o=bpy.data.objects.new(name,d); scene.collection.objects.link(o); o.location=loc; o.rotation_euler=(Vector((0,0,(mins[2]+maxs[2])/2))-o.location).to_track_quat('-Z','Y').to_euler(); return o
area('TEMP_key',(-3,-4,4),150,4,(1,.98,.95)); area('TEMP_fill',(3,-3,2.5),85,4,(.94,.97,1)); area('TEMP_rim',(0,2,4),110,3,(1,1,1))
camd=bpy.data.cameras.new('TEMP_camera'); cam=bpy.data.objects.new('TEMP_camera',camd); scene.collection.objects.link(cam); scene.camera=cam; camd.lens=55
# front=from -Y; profile=from -X; three-quarter from (-X,-Y)
def render_view(tag,target,offset,ortho,head=False):
 cam.location=Vector(target)+Vector(offset); cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler(); camd.type='ORTHO'; camd.ortho_scale=ortho
 scene.render.resolution_x=640 if head else 720; scene.render.resolution_y=640 if head else 960
 scene.render.filepath=os.path.join(out,tag+'.png'); bpy.ops.render.render(write_still=True)
mid=(mins[2]+maxs[2])/2; height=maxs[2]-mins[2]
def use_mat(mat): body.data.materials.clear(); body.data.materials.append(mat)
use_mat(m0)
for v,ofs in [('front',(0,-1,0)),('profile',(-1,0,0)),('three_quarter',(-.707,-.707,0))]: render_view('full_'+v,(0,0,mid),tuple(x*height*1.65 for x in ofs),height*1.12)
headz=mins[2]+height*.91
for v,ofs in [('front',(0,-1,0)),('profile',(-1,0,0)),('three_quarter',(-.707,-.707,0))]: render_view('head_'+v,(0,0,headz),tuple(x*height*.34 for x in ofs),height*.23,True)
# topology view: diagnostic viewport-style wire overlay generated as temporary material? use Freestyle-like wire via edit overlay not renderable.
# Temporarily render with wireframe node texture, explicitly labeled diagnostic and never saved.
wire=bpy.data.materials.new('TEMP_wire_diagnostic'); wire.use_nodes=True; nt=wire.node_tree; nt.nodes.clear(); outn=nt.nodes.new('ShaderNodeOutputMaterial'); em=nt.nodes.new('ShaderNodeEmission'); em.inputs['Color'].default_value=(.008,.008,.008,1); wf=nt.nodes.new('ShaderNodeWireframe'); wf.inputs['Size'].default_value=1.0; mix=nt.nodes.new('ShaderNodeMixShader'); clay=nt.nodes.new('ShaderNodeBsdfPrincipled'); clay.inputs['Base Color'].default_value=(.497,.497,.471,1); clay.inputs['Roughness'].default_value=.85; nt.links.new(wf.outputs['Fac'],mix.inputs[0]); nt.links.new(clay.outputs[0],mix.inputs[1]); nt.links.new(em.outputs[0],mix.inputs[2]); nt.links.new(mix.outputs[0],outn.inputs['Surface']); use_mat(wire); render_view('topology_front',(0,0,mid),(0,-height*1.65,0),height*1.12)
# solid M1 M2 M3 comparison, flat constant albedo finish, no diagnostic image pattern
for label,col in [('M1',(.010,.010,.012,1)),('M2',(.760,.776,.799,1)),('M3',(.680,.620,.535,1))]:
 mm=bpy.data.materials.new('TEMP_'+label); mm.use_nodes=True; mm.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=col; mm.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=.78; use_mat(mm); render_view('solid_'+label+'_front',(0,0,mid),(0,-height*1.65,0),height*1.12)
