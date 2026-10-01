"""Run inside Blender to render SVR object GLBs for evaluation.

Configuration is provided via env var DECOMVOXEL_CONFIG_JSON (JSON-encoded dict):

{
  "jobs": [
    {
      "glb_dir":     "/abs/path/to/glb",          # dir with object_*.glb files
      "exclude_ids": [255],                        # object IDs to skip (default [255])
      "output_dir":  "/abs/path/to/render",        # RGBA PNG output dir
      "engine":      "CYCLES",                     # or "EEVEE"
      "samples":     64,
      "device":      "GPU",                        # or "CPU"
      "cameras": [
        {
          "image_name": "000000",
          "W": 1280, "H": 720,
          "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],
          "w2c": [[..],..] // 4x4 OpenCV world->cam
        }, ...
      ]
    }, ...
  ]
}

Each camera renders a transparent-background RGBA PNG.
Transparent → white compositing is done by the outer Python script.

Material setup per object:
  ShaderNodeAttribute (color_attributes[0])
    -> Color -> ShaderNodeEmission
               -> ShaderNodeOutputMaterial.Surface

Scene: film_transparent = True, no floor, no area light.
"""

from __future__ import annotations

import glob as _glob
import json
import math
import os
import re
import sys

import bpy  # type: ignore
from mathutils import Matrix  # type: ignore


# ── constants ────────────────────────────────────────────────────────────────
# Rotate imported GLBs -90deg around X (y-up → z-up).
_GLB_ROT_X = Matrix.Rotation(-math.pi / 2.0, 4, "X")

CONFIG_JSON_ENV_VAR = "DECOMVOXEL_CONFIG_JSON"


# ── I/O silence helper ───────────────────────────────────────────────────────

def _silence(fn, *args, **kwargs):
    """Run a Blender op while suppressing C-level stdout/stderr."""
    sys.stdout.flush(); sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    so = os.dup(1); se = os.dup(2)
    try:
        os.dup2(devnull, 1); os.dup2(devnull, 2)
        return fn(*args, **kwargs)
    finally:
        os.dup2(so, 1); os.dup2(se, 2)
        os.close(so); os.close(se); os.close(devnull)


# ── scene reset ───────────────────────────────────────────────────────────────

def reset_scene() -> bpy.types.Scene:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    # Black world (doesn't matter with film_transparent, but be explicit)
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
        bg.inputs[1].default_value = 0.0
    return scene


# ── render engine setup ───────────────────────────────────────────────────────

def setup_render_engine(scene: bpy.types.Scene,
                         engine: str, samples: int, device: str) -> None:
    if engine.upper() == "EEVEE":
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

    # RGBA + transparent film
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True


# ── GLB import ────────────────────────────────────────────────────────────────

def import_glb(glb_path: str) -> list:
    before = set(bpy.data.objects)
    _silence(bpy.ops.import_scene.gltf, filepath=glb_path)
    after = set(bpy.data.objects)
    new_objs = list(after - before)

    # Rotate top-level objects -90deg around world X (GLB y-up → z-up).
    for obj in new_objs:
        if obj.parent is None or obj.parent not in new_objs:
            obj.matrix_world = _GLB_ROT_X @ obj.matrix_world
    bpy.context.view_layer.update()

    mesh_objs = [o for o in new_objs if o.type == "MESH"]
    print(f"[INFO] Imported {os.path.basename(glb_path)}: "
          f"{len(mesh_objs)} mesh object(s) (rotated -90° X)")
    return mesh_objs


# ── emission material with color attribute ────────────────────────────────────

def _first_color_attr_name(mesh) -> str | None:
    """Return name of the first usable color attribute, or None."""
    try:
        ca = getattr(mesh, "color_attributes", None)
        if ca is not None and len(ca) > 0:
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


def _build_emission_color_attr_mat(mat_name: str, attr_name: str) -> bpy.types.Material:
    """
    Build a material:
      Attribute(attr_name).Color -> Emission.Color -> Material Output.Surface
    """
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out_node  = nt.nodes.new("ShaderNodeOutputMaterial")
    emit_node = nt.nodes.new("ShaderNodeEmission")
    attr_node = nt.nodes.new("ShaderNodeAttribute")

    attr_node.attribute_type = "GEOMETRY"
    attr_node.attribute_name = attr_name
    emit_node.inputs["Strength"].default_value = 1.0

    nt.links.new(attr_node.outputs["Color"], emit_node.inputs["Color"])
    nt.links.new(emit_node.outputs["Emission"], out_node.inputs["Surface"])
    return mat


def attach_emission_materials(mesh_objs: list) -> None:
    """Replace all materials on each mesh with Emission(ColorAttribute)."""
    for obj in mesh_objs:
        if obj.type != "MESH" or obj.data is None:
            continue
        attr_name = _first_color_attr_name(obj.data)
        if attr_name is None:
            print(f"[WARN] '{obj.name}' has no color attribute; assigning grey emission.")
            attr_name = None

        if attr_name is not None:
            mat = _build_emission_color_attr_mat(f"{obj.name}_svr_emit", attr_name)
        else:
            mat = bpy.data.materials.new(name=f"{obj.name}_svr_emit_grey")
            mat.use_nodes = True
            nt = mat.node_tree
            for n in list(nt.nodes):
                nt.nodes.remove(n)
            out  = nt.nodes.new("ShaderNodeOutputMaterial")
            emit = nt.nodes.new("ShaderNodeEmission")
            emit.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
            emit.inputs["Strength"].default_value = 1.0
            nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])

        obj.data.materials.clear()
        obj.data.materials.append(mat)
        print(f"[INFO] '{obj.name}': emission mat "
              f"({'attr=' + attr_name if attr_name else 'grey fallback'})")


# ── GLB file listing ──────────────────────────────────────────────────────────

def _glb_obj_id(path: str) -> int | None:
    """Extract numeric ID from 'object_002.glb' -> 2, or None if not matching."""
    m = re.fullmatch(r"object_(\d+)\.glb", os.path.basename(path))
    return int(m.group(1)) if m else None


def list_object_glbs(glb_dir: str, exclude_ids: list[int]) -> list[str]:
    all_glbs = sorted(_glob.glob(os.path.join(glb_dir, "object_*.glb")))
    excluded = set(exclude_ids)
    result = [p for p in all_glbs
              if (oid := _glb_obj_id(p)) is not None and oid not in excluded]
    if not result:
        print(f"[WARN] No eligible GLB files in {glb_dir} "
              f"(found {len(all_glbs)}, excluded ids={excluded})")
    else:
        print(f"[INFO] {len(result)} GLB(s) to render "
              f"(excluded ids={excluded}): "
              + ", ".join(os.path.basename(p) for p in result))
    return result


# ── camera ────────────────────────────────────────────────────────────────────

def make_camera(name: str, K, W: int, H: int, w2c) -> bpy.types.Object:
    import numpy as _np

    cam_data = bpy.data.cameras.new(name)
    cam_obj  = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam_obj)

    fx = float(K[0][0]); cx = float(K[0][2]); cy = float(K[1][2])
    cam_data.sensor_fit   = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.lens_unit    = "MILLIMETERS"
    cam_data.lens         = (fx / float(W)) * cam_data.sensor_width
    max_dim               = float(max(W, H))
    cam_data.shift_x      = (W * 0.5 - cx) / max_dim
    cam_data.shift_y      = (cy - H * 0.5) / max_dim
    cam_data.clip_start   = 0.01
    cam_data.clip_end     = 1000.0

    w2c_arr = _np.asarray(w2c, dtype=_np.float64)
    c2w     = _np.linalg.inv(w2c_arr)
    # OpenCV -> Blender camera convention
    flip = _np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], dtype=_np.float64)
    c2w_bl = c2w @ flip
    cam_obj.matrix_world = Matrix([list(r) for r in c2w_bl])
    return cam_obj


def render_one(scene: bpy.types.Scene, cam_obj: bpy.types.Object,
               W: int, H: int, out_path: str) -> None:
    scene.camera = cam_obj
    scene.render.resolution_x = int(W)
    scene.render.resolution_y = int(H)
    scene.render.resolution_percentage = 100
    scene.render.filepath = out_path
    _silence(bpy.ops.render.render, write_still=True)


# ── job runner ────────────────────────────────────────────────────────────────

def run_job(job: dict) -> None:
    scene = reset_scene()
    setup_render_engine(
        scene,
        job.get("engine", "CYCLES"),
        job.get("samples", 64),
        job.get("device", "GPU"),
    )

    glb_dir     = job["glb_dir"]
    exclude_ids = [int(x) for x in job.get("exclude_ids", [255])]
    glb_files   = list_object_glbs(glb_dir, exclude_ids)

    if not glb_files:
        print("[WARN] No GLB files to render; skipping job.")
        return

    for glb_path in glb_files:
        mesh_objs = import_glb(glb_path)
        attach_emission_materials(mesh_objs)

    out_dir = job["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    cams = job["cameras"]
    print(f"[INFO] Rendering {len(cams)} view(s) -> {out_dir}")
    for ci, cam_cfg in enumerate(cams):
        name    = cam_cfg["image_name"]
        cam_obj = make_camera(
            f"cam_{ci:04d}",
            cam_cfg["K"], cam_cfg["W"], cam_cfg["H"], cam_cfg["w2c"],
        )
        out_path = os.path.join(out_dir, f"{name}.png")
        render_one(scene, cam_obj, cam_cfg["W"], cam_cfg["H"], out_path)
        # Remove camera to keep scene tidy between views
        bpy.data.objects.remove(cam_obj, do_unlink=True)
        print(f"[INFO]   rendered {name}.png")


# ── entry point ───────────────────────────────────────────────────────────────

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
