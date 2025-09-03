bl_info = {
    "name": "Even Shell",
    "author": "lapentab + ChatGPT",
    "version": (0, 6, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar (N) > Even Shell",
    "description": "Keep uniform thickness between two edge loops (SYMMETRIC/ANCHOR) or one loop vs shell; includes FLAT+SMOOTH mid-curve mode.",
    "category": "Mesh",
}

import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy.props import FloatProperty, EnumProperty, IntProperty

# ----------------------------------------------------------------------
# Cycle detection (closed edge loops from selection)
# ----------------------------------------------------------------------

def _selected_edge_cycles(bm):
    """Return list of cycles (dict: verts, edges, length, perimeter).
    A cycle is a selected-edges component where every vertex has degree 2 within the component.
    """
    sel_edges = [e for e in bm.edges if e.select]
    if not sel_edges:
        return []

    v_deg = {}
    v_to_edges = {}
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

        # verify 2-regular
        ok = True
        for v in comp_verts:
            cnt = 0
            for e in v.link_edges:
                if e in comp_edges:
                    cnt += 1
            if cnt != 2:
                ok = False
                break
        if not ok:
            continue

        perim = sum((e.verts[1].co - e.verts[0].co).length for e in comp_edges)
        cycles.append({
            "verts": comp_verts,
            "edges": comp_edges,
            "length": len(comp_edges),
            "perimeter": perim,
        })

    return cycles


def _pick_two_longest_cycles(cycles):
    """Pick two cycles by descending perimeter (fallback to edge count)."""
    if len(cycles) < 2:
        return None, None
    cycles_sorted = sorted(cycles, key=lambda c: (c["perimeter"], c["length"]), reverse=True)
    return cycles_sorted[0], cycles_sorted[1]


def _active_loop_priority(bm, loopA, loopB):
    """Prefer loop containing active vertex as A."""
    act = bm.select_history.active
    if isinstance(act, bmesh.types.BMVert):
        if act in loopA["verts"]:
            return loopA, loopB
        if act in loopB["verts"]:
            return loopB, loopA
    return loopA, loopB


# ----------------------------------------------------------------------
# Loop-A tangent helper (for FLAT_SMOOTH fallback)
# ----------------------------------------------------------------------

def _get_loop_neighbors_in_set(v, loop_edges_set):
    """Return the two neighbor vertices of v along the loop edges set (or fewer if non-manifold)."""
    nbrs = []
    for e in v.link_edges:
        if e in loop_edges_set:
            nbrs.append(e.other_vert(v))
            if len(nbrs) == 2:
                break
    return nbrs

def _loop_tangent_at_vertex(v, loop_edges_set):
    """Average tangent using the two loop neighbors: normalize(nbr1 - nbr2)."""
    nbrs = _get_loop_neighbors_in_set(v, loop_edges_set)
    if len(nbrs) == 2:
        t = (nbrs[0].co - nbrs[1].co)
        if t.length_squared > 1e-18:
            return t.normalized()
    # fallback: sum edges around v that are in the loop
    acc = Vector((0.0, 0.0, 0.0))
    for e in v.link_edges:
        if e in loop_edges_set:
            acc += (e.other_vert(v).co - v.co)
    if acc.length_squared > 1e-18:
        return acc.normalized()
    return Vector((1.0, 0.0, 0.0))  # degenerate fallback

def _order_loop_vertices(loop):
    """Return ordered list of vertices around a closed loop."""
    edges_set = loop["edges"]
    verts_set = loop["verts"]
    v_start = next(iter(verts_set))
    nbrs = _get_loop_neighbors_in_set(v_start, edges_set)
    if not nbrs:
        return [v_start]
    v_prev = v_start
    v_curr = nbrs[0]
    ordered = [v_start]
    visited = {v_start}
    max_steps = len(verts_set) + 2
    steps = 0
    while steps < max_steps:
        ordered.append(v_curr)
        visited.add(v_curr)
        nbrs = _get_loop_neighbors_in_set(v_curr, edges_set)
        if len(nbrs) == 0:
            break
        next_v = nbrs[0] if len(nbrs) == 1 else (nbrs[0] if nbrs[1] == v_prev else nbrs[1])
        if next_v == v_start:
            break
        v_prev, v_curr = v_curr, next_v
        steps += 1
    return ordered

# ----------------------------------------------------------------------
# BVH helpers for shell mode
# ----------------------------------------------------------------------

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
    p = v.co
    n = _avg_normal(v)
    hit = None

    if method == 'RAYCAST':
        h = bvh.ray_cast(p + n * 1e-6, n, max_dist)
        if h is not None:
            loc, normal, face_index, _ = h
            hit = loc
        if hit is None:
            h = bvh.ray_cast(p - n * 1e-6, -n, max_dist)
            if h is not None:
                loc, normal, face_index, _ = h
                hit = loc

    if hit is None:
        loc, normal, face_index, dist = bvh.find_nearest(p)
        if loc is not None:
            hit = loc

    return hit

# ----------------------------------------------------------------------
# Ring smoothing (for FLAT_SMOOTH)
# ----------------------------------------------------------------------

def _smooth_ring(points, iters, lam):
    """Cyclic Laplacian smoothing of a list of Vector points."""
    n = len(points)
    if n < 3 or iters <= 0 or lam <= 0.0:
        return points
    pts = [p.copy() for p in points]
    for _ in range(iters):
        new_pts = [None]*n
        for i in range(n):
            prev = pts[(i-1) % n]
            curr = pts[i]
            nxt  = pts[(i+1) % n]
            new_pts[i] = curr*(1.0 - lam) + (prev + nxt)*(0.5*lam)
        pts = new_pts
    return pts

# ----------------------------------------------------------------------
# Operator
# ----------------------------------------------------------------------

class MESH_OT_set_loop_gap(bpy.types.Operator):
    bl_idname = "mesh.set_loop_gap"
    bl_label = "Even Shell: Apply"
    bl_options = {'REGISTER', 'UNDO'}

    target_width: FloatProperty(
        name="Target Width",
        description="Desired distance between loops/surfaces",
        min=0.0, default=2.0
    )
    mode: EnumProperty(
        name="Mode",
        items=[
            ('SYMMETRIC', "Two Loops: Symmetric", "Move both loops equally"),
            ('ANCHOR_A', "Two Loops: Anchor A", "Keep Loop A, move Loop B"),
            ('ANCHOR_B', "Two Loops: Anchor B", "Keep Loop B, move Loop A"),
            ('FLAT_SMOOTH', "Two Loops: FLAT + Smooth", "Smooth the mid-curve around the rim, then set width"),
            ('SHELL', "Shell (One Loop)", "One loop vs opposite shell; move the selected loop"),
        ],
        default='SYMMETRIC'
    )
    shell_method: EnumProperty(
        name="Shell Hit",
        items=[
            ('RAYCAST', "Raycast (±normal)", "Trace along vertex normal; best for clean shells"),
            ('NEAREST', "Nearest Surface", "Use nearest point on surface"),
        ],
        default='RAYCAST'
    )
    smooth_iters: IntProperty(
        name="Smooth Iters",
        min=0, max=100, default=8,
        description="Iterations for FLAT_SMOOTH mid-curve smoothing"
    )
    smooth_lambda: FloatProperty(
        name="Smooth Lambda",
        min=0.0, max=1.0, default=0.5,
        description="Smoothing strength per iteration for FLAT_SMOOTH"
    )

    def invoke(self, context, event):
        # Sync from UI (Scene props)
        self.target_width = context.scene.loop_gap_width
        self.mode = context.scene.loop_gap_mode
        if self.mode == 'SHELL':
            self.shell_method = context.scene.loop_gap_shell_method
        self.smooth_iters = context.scene.loop_gap_smooth_iters
        self.smooth_lambda = context.scene.loop_gap_smooth_lambda
        return self.execute(context)

    def execute(self, context):
        obj = context.object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Active object must be a mesh")
            return {'CANCELLED'}
        if context.mode != 'EDIT_MESH':
            self.report({'ERROR'}, "Switch to Edit Mode")
            return {'CANCELLED'}

        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()

        cycles = _selected_edge_cycles(bm)

        # ----------------- SHELL MODE -----------------
        if self.mode == 'SHELL':
            if len(cycles) < 1:
                self.report({'ERROR'}, "Shell mode: select ONE closed edge loop on the shell.")
                return {'CANCELLED'}
            loop = sorted(cycles, key=lambda c: (c["perimeter"], c["length"]), reverse=True)[0]
            sel_verts = loop["verts"]

            bvh = _build_bvh_from_bmesh(bm)
            moved = 0
            tw = max(self.target_width, 0.0)

            for v in sel_verts:
                hit = _shell_hit_position(bvh, v, method=self.shell_method)
                if hit is None:
                    continue
                direction = (v.co - hit)
                L = direction.length
                if L < 1e-12:
                    direction = _avg_normal(v)
                    if direction.length_squared == 0.0:
                        continue
                else:
                    direction = direction.normalized()
                v.co = hit + direction * tw
                moved += 1

            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            self.report({'INFO'}, f"Shell mode: moved {moved} verts to {tw:.4f} thickness.")
            return {'FINISHED'}

        # --------------- TWO-LOOP / FLAT_SMOOTH ---------------
        if len(cycles) < 2:
            self.report({'ERROR'}, "Two-loop modes: select edges forming at least TWO closed loops.")
            return {'CANCELLED'}

        loopA, loopB = _pick_two_longest_cycles(cycles)
        if loopA is None or loopB is None:
            self.report({'ERROR'}, "Could not find two closed edge-loop cycles in selection.")
            return {'CANCELLED'}

        loopA, loopB = _active_loop_priority(bm, loopA, loopB)
        A_verts = loopA["verts"]
        B_verts = loopB["verts"]

        # Build connector edges (one vert in A, one in B) and map A->B
        conn_edges = []
        A_to_B = {}
        for e in bm.edges:
            v0, v1 = e.verts
            inA0, inA1 = (v0 in A_verts), (v1 in A_verts)
            inB0, inB1 = (v0 in B_verts), (v1 in B_verts)
            if (inA0 and inB1):
                conn_edges.append(e)
                A_to_B[v0] = v1
            elif (inA1 and inB0):
                conn_edges.append(e)
                A_to_B[v1] = v0

        if not conn_edges or len(A_to_B) < 3:
            self.report({'ERROR'}, "No connecting edges found (need a quad strip with A↔B pairs).")
            return {'CANCELLED'}

        moved = set()
        tw = max(self.target_width, 0.0)

        if self.mode == 'FLAT_SMOOTH':
            # order A loop to get consistent neighbor relations
            A_ordered = _order_loop_vertices(loopA)
            n = len(A_ordered)
            if n < 3:
                self.report({'ERROR'}, "FLAT_SMOOTH needs a proper closed loop A.")
                return {'CANCELLED'}

            # Build pairs in order (skip A verts without a connector)
            pairs = []
            for vA in A_ordered:
                vB = A_to_B.get(vA)
                if vB is not None:
                    pairs.append((vA, vB))
            if len(pairs) < 3:
                self.report({'ERROR'}, "FLAT_SMOOTH needs at least 3 A↔B pairs.")
                return {'CANCELLED'}

            # Midpoints and across vectors
            mids = []
            across = []
            for vA, vB in pairs:
                pA = vA.co.copy()
                pB = vB.co.copy()
                mids.append((pA + pB) * 0.5)
                across.append(pB - pA)

            # Smooth the mids (cyclic)
            mids = _smooth_ring(mids, self.smooth_iters, self.smooth_lambda)

            # Reposition each pair
            m_count = len(pairs)
            for i, (vA, vB) in enumerate(pairs):
                # tangent from neighboring mids
                m_prev = mids[(i - 1) % m_count]
                m_curr = mids[i]
                m_next = mids[(i + 1) % m_count]
                t = (m_next - m_prev)
                if t.length_squared < 1e-18:
                    t = _loop_tangent_at_vertex(vA, loopA["edges"])
                else:
                    t = t.normalized()

                ab = across[i]
                ab_flat = ab - t * ab.dot(t)
                u = ab.normalized() if ab_flat.length_squared < 1e-18 else ab_flat.normalized()

                # place points symmetrically about the (smoothed) mid
                vA.co = m_curr - u * (tw * 0.5)
                vB.co = m_curr + u * (tw * 0.5)
                moved.update((vA, vB))

        else:
            # Simple two-loop variants
            for e in conn_edges:
                v0, v1 = e.verts
                if (v0 in A_verts and v1 in B_verts):
                    vA, vB = v0, v1
                elif (v1 in A_verts and v0 in B_verts):
                    vA, vB = v1, v0
                else:
                    continue

                pA = vA.co.copy()
                pB = vB.co.copy()
                vec = pB - pA
                L = vec.length
                if L < 1e-12:
                    continue
                u = vec / L

                if self.mode == 'SYMMETRIC':
                    m = (pA + pB) * 0.5
                    vA.co = m - u * (tw * 0.5)
                    vB.co = m + u * (tw * 0.5)
                    moved.update((vA, vB))
                elif self.mode == 'ANCHOR_A':
                    vB.co = pA + u * tw
                    moved.add(vB)
                elif self.mode == 'ANCHOR_B':
                    vA.co = pB - u * tw
                    moved.add(vA)

        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        self.report({'INFO'}, f"{self.mode}: moved {len(moved)} verts.")
        return {'FINISHED'}

# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

class VIEW3D_PT_even_shell(bpy.types.Panel):
    bl_label = "Even Shell"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Even Shell"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.prop(context.scene, "loop_gap_width", text="Target Width")
        col.prop(context.scene, "loop_gap_mode", text="Mode")
        if context.scene.loop_gap_mode == 'SHELL':
            col.prop(context.scene, "loop_gap_shell_method", text="Shell Hit")
        if context.scene.loop_gap_mode == 'FLAT_SMOOTH':
            col.prop(context.scene, "loop_gap_smooth_iters", text="Smooth Iters")
            col.prop(context.scene, "loop_gap_smooth_lambda", text="Smooth Lambda")

        op = col.operator("mesh.set_loop_gap", text="Apply")
        op.target_width = context.scene.loop_gap_width
        op.mode = context.scene.loop_gap_mode
        if context.scene.loop_gap_mode == 'SHELL':
            op.shell_method = context.scene.loop_gap_shell_method
        if context.scene.loop_gap_mode == 'FLAT_SMOOTH':
            op.smooth_iters = context.scene.loop_gap_smooth_iters
            op.smooth_lambda = context.scene.loop_gap_smooth_lambda

# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------

def register():
    bpy.utils.register_class(MESH_OT_set_loop_gap)
    bpy.utils.register_class(VIEW3D_PT_even_shell)
    bpy.types.Scene.loop_gap_width = FloatProperty(
        name="Target Width", min=0.0, default=2.0)
    bpy.types.Scene.loop_gap_mode = EnumProperty(
        name="Mode",
        items=[
            ('SYMMETRIC', "Two Loops: Symmetric", ""),
            ('ANCHOR_A', "Two Loops: Anchor A", ""),
            ('ANCHOR_B', "Two Loops: Anchor B", ""),
            ('FLAT_SMOOTH', "Two Loops: FLAT + Smooth", ""),
            ('SHELL', "Shell (One Loop)", ""),
        ],
        default='SYMMETRIC'
    )
    bpy.types.Scene.loop_gap_shell_method = EnumProperty(
        name="Shell Hit",
        items=[
            ('RAYCAST', "Raycast (±normal)", ""),
            ('NEAREST', "Nearest Surface", ""),
        ],
        default='RAYCAST'
    )
    bpy.types.Scene.loop_gap_smooth_iters = IntProperty(
        name="Smooth Iters", min=0, max=100, default=8)
    bpy.types.Scene.loop_gap_smooth_lambda = FloatProperty(
        name="Smooth Lambda", min=0.0, max=1.0, default=0.5)

def unregister():
    del bpy.types.Scene.loop_gap_smooth_lambda
    del bpy.types.Scene.loop_gap_smooth_iters
    del bpy.types.Scene.loop_gap_shell_method
    del bpy.types.Scene.loop_gap_mode
    del bpy.types.Scene.loop_gap_width
    bpy.utils.unregister_class(VIEW3D_PT_even_shell)
    bpy.utils.unregister_class(MESH_OT_set_loop_gap)

if __name__ == "__main__":
    register()
