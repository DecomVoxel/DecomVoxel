"""Rotate a scene GLB by -90 degrees around the world X axis (matches the
GLB-import rotation applied in evaluation/utils/blender_render_worker.py) and
write it next to the input as `<stem>_90.glb`.

Usage:
    python evaluation/utils/rotate_scene_glb_x90.py <input.glb> [<output.glb>]

If <output.glb> is omitted, the result is written to:
    <input dir>/<input stem>_90.glb
"""

from __future__ import annotations

import os
import sys

import numpy as np
import trimesh


def _rot_x_minus_90() -> np.ndarray:
    c, s = 0.0, -1.0  # cos(-90), sin(-90)
    M = np.eye(4, dtype=np.float64)
    M[1, 1] = c;  M[1, 2] = -s
    M[2, 1] = s;  M[2, 2] = c
    return M


def rotate_scene_glb_x90(input_path: str, output_path: str | None = None) -> str:
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input GLB not found: {input_path}")
    if output_path is None:
        stem, _ = os.path.splitext(input_path)
        output_path = f"{stem}_90.glb"

    R = _rot_x_minus_90()
    loaded = trimesh.load(input_path, process=False)

    if isinstance(loaded, trimesh.Scene):
        # Apply the world-space rotation to every top-level node by composing it
        # onto each node's local transform relative to the scene root.
        try:
            roots = list(loaded.graph.nodes_geometry)
        except AttributeError:
            roots = list(loaded.graph.nodes)
        # Easiest robust approach: wrap the whole scene by applying the rotation
        # to scene.graph base transform via apply_transform on each geometry.
        new_scene = trimesh.Scene(base_frame=loaded.graph.base_frame)
        for node_name in loaded.graph.nodes_geometry:
            tf, geom_name = loaded.graph.get(node_name)
            if geom_name is None or geom_name not in loaded.geometry:
                continue
            new_tf = R @ np.asarray(tf, dtype=np.float64)
            new_scene.add_geometry(
                loaded.geometry[geom_name],
                node_name=node_name,
                geom_name=geom_name,
                transform=new_tf,
            )
        new_scene.export(output_path)
    elif isinstance(loaded, trimesh.Trimesh):
        m = loaded.copy()
        m.apply_transform(R)
        m.export(output_path)
    else:
        raise TypeError(f"Unsupported trimesh load type: {type(loaded)}")

    print(f"[INFO] Rotated GLB written: {output_path}")
    return output_path


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else None
    rotate_scene_glb_x90(input_path, output_path)


if __name__ == "__main__":
    main()
