"""Run inside Blender to merge multiple .ply meshes into a single .glb,
preserving per-vertex color attributes.

Why a Blender worker? trimesh's `util.concatenate` + GLB export silently drops
vertex colors when input meshes have heterogeneous `visual` types, which yields
gray renders downstream. Blender's PLY importer keeps vertex colors as a
`color_attributes` entry, and its glTF exporter (with `export_attributes=True`
or via a Principled BSDF material wired from a Color Attribute) writes a proper
COLOR_0 accessor.

Configuration via env var DECOMVOXEL_CONFIG_JSON:
{
  "ply_paths": ["/abs/a.ply", "/abs/b.ply", ...],
  "output_glb": "/abs/scene_mesh_color.glb"
}
"""

from __future__ import annotations

import json
import os
import sys

import bpy  # type: ignore


CONFIG_JSON_ENV_VAR = "DECOMVOXEL_CONFIG_JSON"


def _silence(fn, *args, **kwargs):
    sys.stdout.flush(); sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    s1, s2 = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1); os.dup2(devnull, 2)
        return fn(*args, **kwargs)
    finally:
        os.dup2(s1, 1); os.dup2(s2, 2)
        os.close(s1); os.close(s2); os.close(devnull)


def _import_ply(path: str) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    # Blender 4.x ships two PLY importers; prefer the new "fast" one which
    # preserves vertex colors as a color attribute named "Col".
    try:
        _silence(bpy.ops.wm.ply_import, filepath=path)
    except (AttributeError, RuntimeError):
        _silence(bpy.ops.import_mesh.ply, filepath=path)
    return [o for o in (set(bpy.data.objects) - before) if o.type == "MESH"]


def _color_attr_name(mesh) -> str | None:
    ca = getattr(mesh, "color_attributes", None)
    if ca is not None and len(ca) > 0:
        active = getattr(ca, "active_color", None)
        if active is not None and active.name:
            return active.name
        return ca[0].name
    vc = getattr(mesh, "vertex_colors", None)
    if vc is not None and len(vc) > 0:
        return vc[0].name
    return None


def _make_vertex_color_material(name: str, attr_name: str) -> bpy.types.Material:
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
    spec = principled.inputs.get("Specular IOR Level") or principled.inputs.get("Specular")
    if spec is not None:
        spec.default_value = 0.0
    rough = principled.inputs.get("Roughness")
    if rough is not None:
        rough.default_value = 1.0
    return mat


def main() -> None:
    cfg_json = os.environ.get(CONFIG_JSON_ENV_VAR)
    if not cfg_json:
        print("[ERROR] DECOMVOXEL_CONFIG_JSON not set", file=sys.stderr)
        sys.exit(2)
    cfg = json.loads(cfg_json)
    ply_paths = cfg["ply_paths"]
    output_glb = cfg["output_glb"]

    bpy.ops.wm.read_factory_settings(use_empty=True)

    all_mesh_objs: list[bpy.types.Object] = []
    for p in ply_paths:
        objs = _import_ply(p)
        if not objs:
            print(f"[WARN] No mesh imported from {p}")
            continue
        for obj in objs:
            attr = _color_attr_name(obj.data)
            if attr is None:
                print(f"[WARN] '{p}' -> '{obj.name}' has no vertex color attribute")
            else:
                # Wire a Principled BSDF material so the glTF exporter writes a
                # baseColorFactor + COLOR_0 referencing material (robust across
                # importer setups).
                mat = _make_vertex_color_material(f"{obj.name}_vcol", attr)
                if len(obj.data.materials) == 0:
                    obj.data.materials.append(mat)
                else:
                    obj.data.materials[0] = mat
            all_mesh_objs.append(obj)

    if not all_mesh_objs:
        print("[ERROR] No meshes imported; aborting", file=sys.stderr)
        sys.exit(3)

    print(f"[INFO] Imported {len(all_mesh_objs)} mesh objects from {len(ply_paths)} PLYs")

    os.makedirs(os.path.dirname(output_glb), exist_ok=True)

    # Select all imported meshes; export as a single GLB. We export each as a
    # separate node (no join) so per-mesh color attributes stay distinct.
    bpy.ops.object.select_all(action="DESELECT")
    for obj in all_mesh_objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = all_mesh_objs[0]

    export_kwargs = dict(
        filepath=output_glb,
        export_format="GLB",
        use_selection=True,
        export_apply=False,
        export_materials="EXPORT",
        # Keep Blender's z-up axes in the file (do NOT remap to glTF y-up).
        # The render worker applies a standard -90deg X rotation to every
        # imported GLB; that rotation is calibrated for non-axis-converted
        # input, so the GT GLB must skip the y-up swap to match the predicted
        # GLB's orientation in the rendered scene.
        export_yup=False,
    )
    # `export_attributes` exists in Blender 4.x and forces extra mesh attributes
    # (including non-standard color attribs) to be written as COLOR_0.
    try:
        _silence(bpy.ops.export_scene.gltf, export_attributes=True, **export_kwargs)
    except TypeError:
        _silence(bpy.ops.export_scene.gltf, **export_kwargs)

    print(f"[INFO] Wrote {output_glb}")


if __name__ == "__main__":
    main()
