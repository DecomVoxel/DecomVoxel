"""Run inside Blender (4.5) to render predicted/GT GLB scenes for evaluation.

Configuration is provided via env var DECOMVOXEL_CONFIG_JSON (JSON-encoded dict):

{
  "jobs": [
    {
      "tag": "pred",                          # or "gt_render"
      "glb_path": "/abs/path/to/scene.glb",
      "output_dir": "/abs/path/render",
      "render_mode": "light" | "emission",
      "add_floor": true,
      "engine": "CYCLES" | "EEVEE",
      "samples": 64,
      "device": "GPU" | "CPU",
      "cameras": [
        {
          "image_name": "000000",
          "W": 1280, "H": 720,
          "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],
          "w2c": [[..],..]
        }, ...
      ]
    }, ...
  ]
}

Output: <output_dir>/<image_name>.png for every camera.
"""

from __future__ import annotations

import json
import math
import os
import sys

import bpy  # type: ignore
from mathutils import Matrix  # type: ignore

# Rotate imported GLBs by -90deg around X axis (clockwise looking from +X toward origin).
# Equivalent to converting y-up GLB to z-up world.
_GLB_IMPORT_ROT_X = Matrix.Rotation(-math.pi / 2.0, 4, "X")


CONFIG_JSON_ENV_VAR = "DECOMVOXEL_CONFIG_JSON"

# OpenCV camera (x right, y down, z forward) -> Blender camera (x right, y up, -z forward)
_CV_TO_BLENDER = Matrix(((1, 0, 0, 0),
                         (0, -1, 0, 0),
                         (0, 0, -1, 0),
                         (0, 0, 0, 1)))
DEBUG=True


# ---------------------------- debug helpers ---------------------------------
def _dbg(msg: str) -> None:
    if DEBUG:
        print(f"[DEBUG] {msg}")


def _dbg_dump_color_attributes(obj) -> None:
    if not DEBUG or obj.type != "MESH" or obj.data is None:
        return
    me = obj.data
    ca = getattr(me, "color_attributes", None)
    n = 0 if ca is None else len(ca)
    _dbg(f"  '{obj.name}' verts={len(me.vertices)} faces={len(me.polygons)} loops={len(me.loops)} "
         f"color_attributes={n}")
    if ca is not None:
        for i, a in enumerate(ca):
            try:
                _dbg(f"    [{i}] name='{a.name}' domain={a.domain} type={a.data_type}"
                     f" len={len(a.data)}")
                # sample first 3 entries
                samples = []
                for k in range(min(3, len(a.data))):
                    c = a.data[k].color  # length 4
                    samples.append(tuple(round(float(x), 3) for x in c))
                _dbg(f"        samples={samples}")
            except (AttributeError, RuntimeError) as e:
                _dbg(f"    [{i}] inspect failed: {e}")
    # legacy vertex_colors
    vc = getattr(me, "vertex_colors", None)
    if vc is not None and len(vc) > 0:
        _dbg(f"    legacy vertex_colors: {[v.name for v in vc]}")


def _dbg_dump_material(slot_idx: int, mat) -> None:
    if not DEBUG:
        return
    if mat is None:
        _dbg(f"      slot[{slot_idx}] = None")
        return
    _dbg(f"      slot[{slot_idx}] mat='{mat.name}' use_nodes={mat.use_nodes}")
    if not mat.use_nodes or mat.node_tree is None:
        return
    nt = mat.node_tree
    for n in nt.nodes:
        extra = ""
        if n.type == "BSDF_PRINCIPLED":
            bc = n.inputs.get("Base Color")
            if bc is not None:
                if bc.is_linked:
                    src = bc.links[0].from_node
                    extra = f" Base Color <- {src.type}('{src.name}')"
                else:
                    v = bc.default_value
                    extra = f" Base Color default=({v[0]:.3f},{v[1]:.3f},{v[2]:.3f},{v[3]:.3f})"
        elif n.type == "ATTRIBUTE":
            extra = f" attribute_type={n.attribute_type} name='{n.attribute_name}'"
        elif n.type == "TEX_IMAGE":
            img = n.image
            extra = f" image='{img.name if img else None}'"
        elif n.type == "EMISSION":
            ci = n.inputs.get("Color")
            if ci is not None:
                if ci.is_linked:
                    src = ci.links[0].from_node
                    extra = f" Color <- {src.type}('{src.name}')"
                else:
                    v = ci.default_value
                    extra = f" Color default=({v[0]:.3f},{v[1]:.3f},{v[2]:.3f},{v[3]:.3f})"
        _dbg(f"        node type={n.type} name='{n.name}'{extra}")
    for li, lk in enumerate(nt.links):
        _dbg(f"        link[{li}] {lk.from_node.type}.{lk.from_socket.name}"
             f" -> {lk.to_node.type}.{lk.to_socket.name}")


def _dbg_dump_object(obj, header: str) -> None:
    if not DEBUG:
        return
    _dbg(f"{header} obj='{obj.name}' type={obj.type} slots={len(obj.material_slots)}")
    _dbg_dump_color_attributes(obj)
    for i, slot in enumerate(obj.material_slots):
        _dbg_dump_material(i, slot.material)

# ----------------------------- scene reset ----------------------------------
def reset_scene() -> bpy.types.Scene:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs[1].default_value = 0.0  # zero ambient strength
    return scene


# ------------------------- render engine setup ------------------------------
def setup_render_engine(scene: bpy.types.Scene, engine: str, samples: int, device: str) -> None:
    if engine.upper() == "EEVEE":
        # Blender 4.5 has EEVEE NEXT
        scene.render.engine = "BLENDER_EEVEE_NEXT" if hasattr(scene, "eevee") and "NEXT" in str(getattr(scene.render, "engine", "")) else "BLENDER_EEVEE"
        # Robust selection
        for cand in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
            try:
                scene.render.engine = cand
                break
            except TypeError:
                continue
    else:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = int(samples)
        scene.cycles.use_denoising = True
        if device.upper() == "GPU":
            try:
                prefs = bpy.context.preferences.addons["cycles"].preferences
                for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
                    try:
                        prefs.compute_device_type = backend
                        prefs.get_devices()
                        gpus = [d for d in prefs.devices if d.type != "CPU"]
                        if gpus:
                            for d in prefs.devices:
                                d.use = (d.type != "CPU")
                            scene.cycles.device = "GPU"
                            print(f"[INFO] Cycles GPU backend: {backend}")
                            break
                    except (TypeError, AttributeError):
                        continue
            except (KeyError, AttributeError):
                pass

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True


# --------------------------- import GLB -------------------------------------
def _silence_blender(fn, *args, **kwargs):
    """Run a Blender op while suppressing C-level stdout/stderr (keeps our prints)."""
    sys.stdout.flush(); sys.stderr.flush()
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        return fn(*args, **kwargs)
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


def import_glb(glb_path: str) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    _silence_blender(bpy.ops.import_scene.gltf, filepath=glb_path)
    after = set(bpy.data.objects)
    new_objs = list(after - before)

    # Rotate top-level (parentless) imported objects -90deg around world X.
    # Children inherit the transform via parent, so we must NOT rotate them again.
    for obj in new_objs:
        if obj.parent is None or obj.parent not in new_objs:
            obj.matrix_world = _GLB_IMPORT_ROT_X @ obj.matrix_world
    bpy.context.view_layer.update()

    mesh_objs = [o for o in new_objs if o.type == "MESH"]
    print(f"[INFO] Imported {glb_path}: {len(mesh_objs)} mesh object(s) (rotated -90deg around X)")
    if DEBUG:
        _dbg(f"=== POST-IMPORT dump for {os.path.basename(glb_path)} ===")
        for o in mesh_objs:
            _dbg_dump_object(o, "[post-import]")
    return mesh_objs


# ---------------- color-attribute material fixup ----------------------------
def _first_color_attribute_name(mesh) -> str | None:
    """Return the name of a usable color attribute on this mesh, or None."""
    # Prefer mesh.color_attributes (Blender 3.2+); also accept legacy vertex_colors.
    try:
        ca = getattr(mesh, "color_attributes", None)
        if ca is not None and len(ca) > 0:
            # active first if set
            active = getattr(ca, "active_color", None)
            if active is not None and active.name:
                return active.name
            return ca[0].name
    except (AttributeError, RuntimeError):
        pass
    try:
        vc = getattr(mesh, "vertex_colors", None)
        if vc is not None and len(vc) > 0:
            return vc[0].name
    except (AttributeError, RuntimeError):
        pass
    return None


def _build_color_attr_material(name: str, attr_name: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    output = nt.nodes.new("ShaderNodeOutputMaterial")
    principled = nt.nodes.new("ShaderNodeBsdfPrincipled")
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = "GEOMETRY"
    attr.attribute_name = attr_name
    nt.links.new(attr.outputs["Color"], principled.inputs["Base Color"])
    nt.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    # Reduce specular so vertex color reads true under area light.
    spec = principled.inputs.get("Specular IOR Level") or principled.inputs.get("Specular")
    if spec is not None:
        spec.default_value = 0.0
    rough = principled.inputs.get("Roughness")
    if rough is not None:
        rough.default_value = 1.0
    return mat


def _material_has_principled(mat) -> bool:
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return False
    return any(n.type == "BSDF_PRINCIPLED" for n in mat.node_tree.nodes)


def attach_color_attribute_material(mesh_objs: list[bpy.types.Object]) -> None:
    """For meshes that have a color attribute but no usable material, attach a
    new Principled BSDF material whose Base Color is driven by the color attribute.
    Existing materials with a Principled BSDF are left untouched.
    """
    for obj in mesh_objs:
        if obj.type != "MESH" or obj.data is None:
            continue
        attr_name = _first_color_attribute_name(obj.data)
        if attr_name is None:
            _dbg(f"attach: '{obj.name}' has NO color attribute -> skip")
            continue
        # Decide if we need to (re)assign material.
        slots = list(obj.material_slots)
        has_principled = any(
            (slot.material is not None) and _material_has_principled(slot.material)
            for slot in slots
        )
        needs_new = (len(slots) == 0) or not has_principled
        _dbg(f"attach: '{obj.name}' attr='{attr_name}' slots={len(slots)} "
             f"has_principled={has_principled} -> needs_new={needs_new}")
        if not needs_new:
            continue
        new_mat = _build_color_attr_material(f"{obj.name}_vcol", attr_name)
        if len(slots) == 0:
            obj.data.materials.append(new_mat)
        else:
            for slot in slots:
                slot.material = new_mat
        print(f"[INFO] Attached color-attribute material '{attr_name}' on '{obj.name}'")


# --------------------------- floor + light ----------------------------------
def compute_world_bbox(objs: list[bpy.types.Object]):
    xs, ys, zs = [], [], []
    for o in objs:
        for corner in o.bound_box:
            wp = o.matrix_world @ Matrix.Translation((corner[0], corner[1], corner[2])).to_translation()
            xs.append(wp.x); ys.append(wp.y); zs.append(wp.z)
    if not xs:
        return None
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def add_floor(scene: bpy.types.Scene, mesh_objs: list[bpy.types.Object]) -> bpy.types.Object | None:
    bbox = compute_world_bbox(mesh_objs)
    if bbox is None:
        return None
    (xmin, ymin, zmin), (xmax, ymax, zmax) = bbox
    cx, cy = (xmin + xmax) * 0.5, (ymin + ymax) * 0.5
    diag = max(xmax - xmin, ymax - ymin)
    side = max(10.0 * diag, 20.0)
    bpy.ops.mesh.primitive_plane_add(size=side, location=(cx, cy, zmin))
    floor = bpy.context.active_object
    floor.name = "EvalFloor"
    # White diffuse material
    mat = bpy.data.materials.new("EvalFloorMat")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    diff = nt.nodes.new("ShaderNodeBsdfDiffuse")
    diff.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    nt.links.new(diff.outputs["BSDF"], out.inputs["Surface"])
    floor.data.materials.append(mat)
    return floor


def add_top_area_light(scene: bpy.types.Scene, mesh_objs: list[bpy.types.Object]) -> bpy.types.Object | None:
    bbox = compute_world_bbox(mesh_objs)
    if bbox is None:
        return None
    (xmin, ymin, zmin), (xmax, ymax, zmax) = bbox
    cx, cy = (xmin + xmax) * 0.5, (ymin + ymax) * 0.5
    diag = max(xmax - xmin, ymax - ymin, zmax - zmin)
    side = max(10.0 * diag, 20.0)
    bpy.ops.object.light_add(type="AREA", location=(cx, cy, zmax + 1.0))
    light = bpy.context.active_object
    light.name = "EvalTopLight"
    light.data.size = side
    light.data.energy = 2000.0 * (side * side / 100.0)  # heuristic
    light.rotation_euler = (0.0, 0.0, 0.0)  # downward by default (-Z)
    return light


# ------------------------- emission shading ---------------------------------
def _find_base_color_from_principled(node):
    inp = node.inputs.get("Base Color")
    if inp is None:
        return (0.8, 0.8, 0.8, 1.0), None
    if inp.is_linked:
        link = inp.links[0]
        return None, link.from_socket  # texture-driven; pass-through
    v = inp.default_value
    return (v[0], v[1], v[2], v[3]), None


def replace_with_emission(mesh_objs: list[bpy.types.Object]) -> None:
    """Replace material shaders with an Emission shader using the Principled BSDF base color.

    Operates on a copied material per object to avoid editing source data.
    Preserves the upstream base-color source node (TEX_IMAGE / ATTRIBUTE /
    VERTEX_COLOR / RGB / ...) and rewires it to Emission.Color.
    """
    color_src_keep_types = {"TEX_IMAGE", "TEX_COORD", "MAPPING", "UVMAP",
                             "ATTRIBUTE", "VERTEX_COLOR", "RGB", "MIX_RGB", "MIX",
                             "SEPARATE_COLOR", "COMBINE_COLOR"}
    for obj in mesh_objs:
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None or not mat.use_nodes:
                # Make a flat emission material if missing
                new_mat = bpy.data.materials.new(name=f"{obj.name}_emit")
                new_mat.use_nodes = True
                slot.material = new_mat
                mat = new_mat

            # Copy material so we don't touch the imported one's user data permanently
            mat = mat.copy()
            slot.material = mat
            nt = mat.node_tree

            principled = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)

            # Capture base-color source BEFORE deleting nodes.
            #   src_node, src_socket : upstream node feeding Principled.Base Color (if linked)
            #   const_rgba           : constant default value to use as fallback
            src_node = None
            src_socket_name = None
            const_rgba = (0.8, 0.8, 0.8, 1.0)
            if principled is not None:
                bc = principled.inputs.get("Base Color")
                if bc is not None:
                    if bc.is_linked:
                        lk = bc.links[0]
                        src_node = lk.from_node
                        src_socket_name = lk.from_socket.name
                    else:
                        v = bc.default_value
                        const_rgba = (v[0], v[1], v[2], v[3])

            # Find or create Material Output
            output = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
            if output is None:
                output = nt.nodes.new("ShaderNodeOutputMaterial")

            # Build set of nodes to keep: OUTPUT, plus the source node and its
            # ancestors (transitively), plus a generous list of color-related types.
            keep = {output}
            if src_node is not None:
                # BFS upstream from src_node
                stack = [src_node]
                while stack:
                    n = stack.pop()
                    if n in keep:
                        continue
                    keep.add(n)
                    for inp in n.inputs:
                        for lk in inp.links:
                            stack.append(lk.from_node)
            # Also keep any other harmless color-source nodes (textures etc.) so
            # subsequent re-runs / fallback heuristics still work.
            for n in list(nt.nodes):
                if n.type in color_src_keep_types:
                    keep.add(n)

            to_remove = [n for n in nt.nodes if n not in keep]
            for n in to_remove:
                nt.nodes.remove(n)

            emit = nt.nodes.new("ShaderNodeEmission")
            emit.inputs["Strength"].default_value = 1.0

            wired = False
            if src_node is not None and src_node.name in nt.nodes:
                # Re-resolve socket by name (object reference may still be valid
                # since src_node was kept alive).
                sock = src_node.outputs.get(src_socket_name) if src_socket_name else None
                if sock is None and len(src_node.outputs) > 0:
                    sock = src_node.outputs[0]
                if sock is not None:
                    nt.links.new(sock, emit.inputs["Color"])
                    wired = True

            if not wired:
                # Fallback to first available TEX_IMAGE / ATTRIBUTE / VERTEX_COLOR
                fallback = next((n for n in nt.nodes
                                 if n.type in ("TEX_IMAGE", "ATTRIBUTE", "VERTEX_COLOR")), None)
                if fallback is not None:
                    sock = fallback.outputs.get("Color") or fallback.outputs[0]
                    nt.links.new(sock, emit.inputs["Color"])
                else:
                    emit.inputs["Color"].default_value = const_rgba

            nt.links.new(emit.outputs["Emission"], output.inputs["Surface"])


# --------------------------- camera setup -----------------------------------
def make_camera(name: str, K, W: int, H: int, w2c) -> bpy.types.Object:
    cam_data = bpy.data.cameras.new(name)
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam_obj)

    fx = float(K[0][0]); fy = float(K[1][1])
    cx = float(K[0][2]); cy = float(K[1][2])

    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.lens_unit = "MILLIMETERS"
    cam_data.lens = (fx / float(W)) * cam_data.sensor_width
    # principal point shift (in units of max(W,H) sensor frame)
    max_dim = float(max(W, H))
    cam_data.shift_x = (W * 0.5 - cx) / max_dim
    cam_data.shift_y = (cy - H * 0.5) / max_dim
    cam_data.clip_start = 0.01
    cam_data.clip_end = 1000.0

    # extrinsics
    import numpy as _np
    w2c = _np.asarray(w2c, dtype=_np.float64)
    c2w = _np.linalg.inv(w2c)
    c2w_blender = c2w @ _np.array([[1, 0, 0, 0],
                                   [0, -1, 0, 0],
                                   [0, 0, -1, 0],
                                   [0, 0, 0, 1]], dtype=_np.float64)
    mat = Matrix([list(row) for row in c2w_blender])
    cam_obj.matrix_world = mat
    return cam_obj


def render_one(scene: bpy.types.Scene, cam_obj: bpy.types.Object, W: int, H: int, out_path: str) -> None:
    scene.camera = cam_obj
    scene.render.resolution_x = int(W)
    scene.render.resolution_y = int(H)
    scene.render.resolution_percentage = 100
    scene.render.filepath = out_path
    _silence_blender(bpy.ops.render.render, write_still=True)


# ---------------------------- main loop -------------------------------------
def run_job(job: dict) -> None:
    scene = reset_scene()
    setup_render_engine(scene, job.get("engine", "CYCLES"), job.get("samples", 64), job.get("device", "GPU"))

    glb_path = job["glb_path"]
    mesh_objs = import_glb(glb_path)
    if not mesh_objs:
        print(f"[WARN] No meshes imported from {glb_path}; skipping job '{job.get('tag')}'")
        return

    # Ensure meshes that only carry a vertex/color attribute (no material) get a
    # Principled BSDF whose Base Color is wired from the color attribute. This is
    # the common case for GT scene meshes built from `obj_*_colored.ply`.
    attach_color_attribute_material(mesh_objs)
    if DEBUG:
        _dbg(f"=== POST-ATTACH dump (job tag={job.get('tag')}) ===")
        for o in mesh_objs:
            _dbg_dump_object(o, "[post-attach]")

    if job.get("add_floor", True):
        add_floor(scene, mesh_objs)

    render_mode = job.get("render_mode", "light")
    if render_mode == "emission":
        replace_with_emission(mesh_objs)
    else:
        add_top_area_light(scene, mesh_objs)
    if DEBUG:
        _dbg(f"=== POST-SHADER dump (render_mode={render_mode}) ===")
        for o in mesh_objs:
            _dbg_dump_object(o, f"[post-{render_mode}]")

    out_dir = job["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    cams = job["cameras"]
    print(f"[INFO] Rendering {len(cams)} views for job '{job.get('tag')}' -> {out_dir}")
    for ci, cam_cfg in enumerate(cams):
        name = cam_cfg["image_name"]
        cam_obj = make_camera(f"cam_{ci:04d}", cam_cfg["K"], cam_cfg["W"], cam_cfg["H"], cam_cfg["w2c"])
        out_path = os.path.join(out_dir, f"{name}.png")
        render_one(scene, cam_obj, cam_cfg["W"], cam_cfg["H"], out_path)


def main() -> None:
    cfg_json = os.environ.get(CONFIG_JSON_ENV_VAR)
    if not cfg_json:
        print("[ERROR] DECOMVOXEL_CONFIG_JSON not set", file=sys.stderr)
        sys.exit(2)
    config = json.loads(cfg_json)

    for job in config["jobs"]:
        run_job(job)
    print("[INFO] Worker done.")


if __name__ == "__main__":
    main()
