bl_info = {
    "name": "Even Shell",
    "author": "lapentab + ChatGPT",
    "version": (0, 6, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar (N) > Even Shell",
    "description": "Keeps shell/rim thickness uniform: Two-loops Flat+Smooth, or single-loop Shell offset.",
    "category": "Mesh",
}

import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy.props import FloatProperty, EnumProperty, IntProperty

# ----------------------------- cycles (closed edge loops) -----------------------------

def _selected_edge_cycles(bm):
    sel_edges = [e for e in bm.edges if e.select]
    if not sel_edges:
        return []
    v_deg, v_to_edges = {}, {}
    for e in sel_edges:
        v0, v1 = e.verts
        for v in (v0, v1):
            v_deg[v] = v_deg.get(v, 0) + 1
            v_to_edges.setdefault(v, set()).add(e)
    unused = set(sel_edges)
    cycles = []
    while unused:
        e0 = next(iter(unused))
        comp_edges = set()
        stack = [e0]
        while stack:
            e = stack.pop()
            if e not in unused:
                continue
            v0, v1 = e.verts
            if v_deg.get(v0, 0) != 2 or v_deg.get(v1, 0) != 2:
                unused.remove(e)
                continue
            comp_edges.add(e)
            unused.remove(e)
            for v in e.verts:
                for ne in v_to_edges.get(v, ()):
                    if ne in unused:
                        stack.append(ne)
        if not comp_edges:
            continue
        comp_verts = set()
        for e in comp_edges:
            comp_verts.update(e.verts)
        ok = True
        for v in comp_verts:
            cnt = sum(1 for e in v.link_edges if e in comp_edges)
            if cnt != 2:
                ok = False
                break
        if not ok:
            continue
        perim = sum((e.verts[1].co - e.verts[0].co).length for e in comp_edges)
        cycles.append({"verts": comp_verts, "edges": comp_edges, "length": len(comp_edges), "perimeter": perim})
    return cycles

def _pick_two_longest_cycles(cycles):
    if len(cycles) < 2:
        return None, None
    s = sorted(cycles, key=lambda c: (c["perimeter"], c["length"]), reverse=True)
    return s[0], s[1]

def _active_loop_priority(bm, loopA, loopB):
    act = bm.select_history.active
    if isinstance(act, bmesh.types.BMVert):
        if act in loopA["verts"]:
            return loopA, loopB
        if act in loopB["verts"]:
            return loopB, loopA
    return loopA, loopB

# ----------------------------- loop ordering / tangents -----------------------------

def _get_loop_neighbors_in_set(v, loop_edges_set):
    nbrs = []
    for e in v.link_edges:
        if e in loop_edges_set:
            nbrs.append(e.other_vert(v))
            if len(nbrs) == 2:
                break
    return nbrs

def _loop_tangent_at_vertex(v, loop_edges_set):
    nbrs = _get_loop_neighbors_in_set(v, loop_edges_set)
    if len(nbrs) == 2:
        t = (nbrs[0].co - nbrs[1].co)
        if t.length_squared > 1e-18:
            return t.normalized()
    acc = Vector((0.0, 0.0, 0.0))
    for e in v.link_edges:
        if e in loop_edges_set:
            acc += (e.other_vert(v).co - v.co)
    return acc.normalized() if acc.length_squared > 1e-18 else Vector((1,0,0))

def _order_loop_vertices(loop):
    edges_set = loop["edges"]
    verts_set = loop["verts"]
    v_start = next(iter(verts_set))
    nbrs = _get_loop_neighbors_in_set(v_start, edges_set)
    if not nbrs:
        return [v_start]
    v_prev, v_curr = v_start, nbrs[0]
    ordered = [v_start]
    visited = {v_start}
    max_steps = len(verts_set) + 2
    for _ in range(max_steps):
        ordered.append(v_curr); visited.add(v_curr)
        nbrs = _get_loop_neighbors_in_set(v_curr, edges_set)
        if len(nbrs) == 0:
            break
        next_v = nbrs[0] if len(nbrs) == 1 else (nbrs[0] if nbrs[1] == v_prev else nbrs[1])
        if next_v == v_start:
            break
        v_prev, v_curr = v_curr, next_v
    return ordered

# ----------------------------- BVH & normals (shell mode) -----------------------------

def _build_bvh_from_bmesh(bm):
    return BVHTree.FromBMesh(bm, epsilon=0.0)

def _avg_normal(v):
    n = Vector((0.0, 0.0, 0.0))
    for f in v.link_faces:
        n += f.normal
    if n.length_squared == 0.0:
        return v.normal.copy() if hasattr(v, "normal") else Vector((0,0,1))
    return n.normalized()

def _shell_hit_position(bvh, v, max_dist=1e6, method='RAYCAST'):
    p, n = v.co, _avg_normal(v)
    if method == 'RAYCAST':
        h = bvh.ray_cast(p + n * 1e-6, n, max_dist)
        if h is not None:
            return h[0]
        h = bvh.ray_cast(p - n * 1e-6, -n, max_dist)
        if h is not None:
            return h[0]
    loc = bvh.find_nearest(p)[0]
    return loc

# ----------------------------- ring smoothing -----------------------------

def _smooth_ring(points, iters, lam):
    n = len(points)
    if n < 3 or iters <= 0 or lam <= 0.0:
        return points
    pts = [p.copy() for p in points]
    for _ in range(iters):
        new_pts = [None]*n
        for i in range(n):
            prev = pts[(i-1) % n]; curr = pts[i]; nxt = pts[(i+1) % n]
            new_pts[i] = curr*(1.0 - lam) + (prev + nxt)*(0.5*lam)
        pts = new_pts
    return pts

# ----------------------------- operator -----------------------------

class EVEN_SHELL_OT_apply(bpy.types.Operator):
    bl_idname = "even_shell.apply"
    bl_label = "Apply (Even Shell)"
    bl_options = {'REGISTER', 'UNDO'}

    target_width: FloatProperty(name="Target Width", min=0.0, default=2.0)
    mode: EnumProperty(
        name="Mode",
        items=[
            ('FLAT_SMOOTH', "Two Loops: Flat + Smooth", ""),
            ('SHELL', "Shell (One Loop)", ""),
        ],
        default='FLAT_SMOOTH'
    )
    shell_method: EnumProperty(
        name="Shell Hit",
        items=[('RAYCAST', "Raycast (±normal)", ""), ('NEAREST', "Nearest Surface", "")],
        default='RAYCAST'
    )
    smooth_iters: IntProperty(name="Smooth Iters", min=0, max=100, default=12)
    smooth_lambda: FloatProperty(name="Smooth Lambda", min=0.0, max=1.0, default=0.5)

    def invoke(self, context, event):
        s = context.scene
        self.target_width = s.even_shell_width
        self.mode = s.even_shell_mode
        if self.mode == 'SHELL':
            self.shell_method = s.even_shell_shell_method
        self.smooth_iters = s.even_shell_smooth_iters
        self.smooth_lambda = s.even_shell_smooth_lambda
        return self.execute(context)

    def execute(self, context):
        obj = context.object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Active object must be a mesh"); return {'CANCELLED'}
        if context.mode != 'EDIT_MESH':
            self.report({'ERROR'}, "Switch to Edit Mode"); return {'CANCELLED'}

        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table(); bm.edges.ensure_lookup_table()

        cycles = _selected_edge_cycles(bm)

        # ---- SHELL ----
        if self.mode == 'SHELL':
            if len(cycles) < 1:
                self.report({'ERROR'}, "Shell: select ONE closed edge loop."); return {'CANCELLED'}
            loop = sorted(cycles, key=lambda c: (c["perimeter"], c["length"]), reverse=True)[0]
            sel_verts = loop["verts"]
            bvh = _build_bvh_from_bmesh(bm)
            moved = 0; tw = max(self.target_width, 0.0)
            for v in sel_verts:
                hit = _shell_hit_position(bvh, v, method=self.shell_method)
                if hit is None: continue
                direction = v.co - hit
                if direction.length_squared < 1e-12:
                    direction = _avg_normal(v)
                    if direction.length_squared < 1e-12: continue
                direction.normalize()
                v.co = hit + direction * tw
                moved += 1
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            self.report({'INFO'}, f"Shell: moved {moved} verts to {tw:.4f}.")
            return {'FINISHED'}

        # ---- FLAT_SMOOTH (two loops) ----
        if len(cycles) < 2:
            self.report({'ERROR'}, "Two Loops: select TWO closed loops."); return {'CANCELLED'}
        loopA, loopB = _pick_two_longest_cycles(cycles)
        if loopA is None or loopB is None:
            self.report({'ERROR'}, "Could not find two loops."); return {'CANCELLED'}
        loopA, loopB = _active_loop_priority(bm, loopA, loopB)
        A_verts, B_verts = loopA["verts"], loopB["verts"]

        # Build A->B mapping via connecting edges (quad strip)
        conn_edges = []; A_to_B = {}
        for e in bm.edges:
            v0, v1 = e.verts
            inA0, inA1 = (v0 in A_verts), (v1 in A_verts)
            inB0, inB1 = (v0 in B_verts), (v1 in B_verts)
            if inA0 and inB1:
                conn_edges.append(e); A_to_B[v0] = v1
            elif inA1 and inB0:
                conn_edges.append(e); A_to_B[v1] = v0
        if not conn_edges or len(A_to_B) < 3:
            self.report({'ERROR'}, "Need a quad strip with A↔B connectors."); return {'CANCELLED'}

        A_ordered = _order_loop_vertices(loopA)
        pairs = [(vA, A_to_B[vA]) for vA in A_ordered if vA in A_to_B]
        if len(pairs) < 3:
            self.report({'ERROR'}, "Not enough A↔B pairs."); return {'CANCELLED'}

        mids = []; across = []
        for vA, vB in pairs:
            pA, pB = vA.co.copy(), vB.co.copy()
            mids.append((pA + pB) * 0.5)
            across.append(pB - pA)

        mids = _smooth_ring(mids, self.smooth_iters, self.smooth_lambda)

        moved = set(); tw = max(self.target_width, 0.0)
        m_count = len(pairs)
        for i, (vA, vB) in enumerate(pairs):
            m_prev, m_curr, m_next = mids[(i-1) % m_count], mids[i], mids[(i+1) % m_count]
            t = (m_next - m_prev)
            if t.length_squared < 1e-18:
                t = _loop_tangent_at_vertex(vA, loopA["edges"])
            else:
                t.normalize()
            ab = across[i]
            ab_flat = ab - t * ab.dot(t)
            u = (ab if ab_flat.length_squared < 1e-18 else ab_flat).normalized()
            vA.co = m_curr - u * (tw * 0.5)
            vB.co = m_curr + u * (tw * 0.5)
            moved.update((vA, vB))

        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        self.report({'INFO'}, f"Flat+Smooth: moved {len(moved)} verts to width {tw:.4f}.")
        return {'FINISHED'}

# ----------------------------- UI -----------------------------

class VIEW3D_PT_even_shell(bpy.types.Panel):
    bl_label = "Even Shell"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Even Shell"

    def draw(self, context):
        s = context.scene
        layout = self.layout
        col = layout.column(align=True)
        col.prop(s, "even_shell_width", text="Target Width")
        col.prop(s, "even_shell_mode", text="Mode")
        if s.even_shell_mode == 'SHELL':
            col.prop(s, "even_shell_shell_method", text="Shell Hit")
        if s.even_shell_mode == 'FLAT_SMOOTH':
            col.prop(s, "even_shell_smooth_iters", text="Smooth Iters")
            col.prop(s, "even_shell_smooth_lambda", text="Smooth Lambda")
        op = col.operator("even_shell.apply", text="Apply (Even Shell)")
        op.target_width = s.even_shell_width
        op.mode = s.even_shell_mode
        if s.even_shell_mode == 'SHELL':
            op.shell_method = s.even_shell_shell_method
        if s.even_shell_mode == 'FLAT_SMOOTH':
            op.smooth_iters = s.even_shell_smooth_iters
            op.smooth_lambda = s.even_shell_smooth_lambda

# ----------------------------- registration -----------------------------

def register():
    bpy.utils.register_class(EVEN_SHELL_OT_apply)
    bpy.utils.register_class(VIEW3D_PT_even_shell)
    bpy.types.Scene.even_shell_width = FloatProperty(name="Target Width", min=0.0, default=2.0)
    bpy.types.Scene.even_shell_mode = EnumProperty(
        name="Mode",
        items=[('FLAT_SMOOTH', "Two Loops: Flat + Smooth", ""),
               ('SHELL', "Shell (One Loop)", "")],
        default='FLAT_SMOOTH'
    )
    bpy.types.Scene.even_shell_shell_method = EnumProperty(
        name="Shell Hit",
        items=[('RAYCAST', "Raycast (±normal)", ""), ('NEAREST', "Nearest Surface", "")],
        default='RAYCAST'
    )
    bpy.types.Scene.even_shell_smooth_iters = IntProperty(name="Smooth Iters", min=0, max=100, default=12)
    bpy.types.Scene.even_shell_smooth_lambda = FloatProperty(name="Smooth Lambda", min=0.0, max=1.0, default=0.5)

def unregister():
    del bpy.types.Scene.even_shell_smooth_lambda
    del bpy.types.Scene.even_shell_smooth_iters
    del bpy.types.Scene.even_shell_shell_method
    del bpy.types.Scene.even_shell_mode
    del bpy.types.Scene.even_shell_width
    bpy.utils.unregister_class(VIEW3D_PT_even_shell)
    bpy.utils.unregister_class(EVEN_SHELL_OT_apply)

if __name__ == "__main__":
    register()
