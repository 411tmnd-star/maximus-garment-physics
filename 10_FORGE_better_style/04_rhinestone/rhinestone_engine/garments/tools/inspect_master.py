import sys, bpy, numpy as np
bpy.ops.wm.open_mainfile(filepath=sys.argv[1], load_ui=False)
for o in bpy.data.objects:
    if o.type!='MESH':
        print(o.name,o.type); continue
    me=o.data
    co=np.array([v.co[:] for v in me.vertices])
    print(o.name,len(me.vertices),len(me.polygons),'bbox',co.min(0).round(3),co.max(0).round(3),'uv',[u.name for u in me.uv_layers],'vg',[g.name for g in o.vertex_groups][:14],'mats',[m.name for m in me.materials],'mods',[m.type for m in o.modifiers],'hide',o.hide_render)
print([c.name for c in bpy.data.collections])
print('images',[(i.name,i.size[:]) for i in bpy.data.images])
