"""
Two modes:

1. combine  (default)
   Combine all per-object GLB meshes in <input_root>/<scene>/semantic_result/object_mesh/
   into a single mesh_combine.glb saved under <output_root>/<scene>/.

2. copy-scene
   Copy <input_root>/<scene>/scene_sds/scene_combined.glb
   to   <output_root>/<scene>/scene_combined.glb for every scene.

3. copy-appearance-scene
   Copy <input_root>/<scene>/appearance_sds/scene_combined.glb
   to   <output_root>/<scene>/scene_combined.glb for every scene.

4. copy-bg
   Copy <input_root>/<scene>/see3d_guidance_views/bg_training/mesh/tsdf/tsdf_fusion_post.ply
   to   <output_root>/<scene>/bg.ply for every scene.

5. copy-ckpt
   Copy <input_root>/<scene>/checkpoints/
   to   <output_root>/<out_scene>/checkpoints/ for every scene.
   Use --strip_suffix to map e.g. scan1-7 -> scan1 for the output path.

6. copy-svr
   Copy <input_root>/<scene>/semantic_result/object_voxels/
   to   <output_root>/<out_scene>/object_voxels/ for every scene.

Usage:
    # Combine per-object meshes (original behaviour)
    python combine_scene_meshes.py --mode combine \
        --input_root outputs/exp_scene_replica_0323 \
        --output_root exps/Replica-svr

    # Copy scene_combined.glb files
    python combine_scene_meshes.py --mode copy-scene \
        --input_root outputs/exp_scene_replica_0323 \
        --output_root exps/Replica-ab-geo

    # Copy appearance scene combined glb
    python combine_scene_meshes.py --mode copy-appearance-scene \
        --input_root outputs/exp_appearance_0428 \
        --output_root exps/Replica-ab-app

    # Copy background mesh
    python combine_scene_meshes.py --mode copy-bg \
        --input_root outputs/bg_replica \
        --output_root exps/Replica-bg

    # Copy checkpoints (strip trailing -N suffix for output scene name)
    python combine_scene_meshes.py --mode copy-ckpt --strip_suffix \
        --input_root outputs/exp_scene_replica_0323 \
        --output_root exps/Replica-ab-svr

    # Copy SVR object voxels
    python combine_scene_meshes.py --mode copy-svr --strip_suffix \
        --input_root outputs/exp_scene_replica_0323 \
        --output_root exps/Replica-ab-svr

    # Single scene
    python combine_scene_meshes.py --mode copy-scene --scene scan1-7 \
        --input_root outputs/exp_scene_replica_0323 \
        --output_root exps/Replica-ab-geo
"""

import os
import glob
import re
import shutil
import argparse
import trimesh


def combine_scene(mesh_dir: str, output_path: str) -> bool:
    glb_files = sorted(glob.glob(os.path.join(mesh_dir, '*.glb')))
    if not glb_files:
        print(f"  [SKIP] No GLB files found in {mesh_dir}")
        return False

    print(f"  Found {len(glb_files)} GLB files, combining ...")
    scene = trimesh.Scene()
    loaded = 0
    for glb_path in glb_files:
        node_name = os.path.splitext(os.path.basename(glb_path))[0]
        try:
            scene_or_mesh = trimesh.load(glb_path, force='scene')
            if isinstance(scene_or_mesh, trimesh.Scene):
                geoms = [g for g in scene_or_mesh.geometry.values()
                         if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
                if not geoms:
                    continue
                mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
            elif isinstance(scene_or_mesh, trimesh.Trimesh):
                if len(scene_or_mesh.faces) == 0:
                    continue
                mesh = scene_or_mesh
            else:
                continue
            scene.add_geometry(mesh, node_name=node_name, geom_name=node_name)
            loaded += 1
        except Exception as e:
            print(f"  [WARN] Failed to load {os.path.basename(glb_path)}: {e}")

    if loaded == 0:
        print(f"  [SKIP] All GLB files empty or failed to load.")
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scene.export(output_path)
    print(f"  Saved -> {output_path}  ({loaded} objects)")
    return True


def _strip_suffix(scene_name: str) -> str:
    """Strip trailing -<digits> from a scene name, e.g. scan1-7 -> scan1."""
    return re.sub(r'-\d+$', '', scene_name)


def copy_ckpt(src_ckpt_dir: str, dst_ckpt_dir: str) -> bool:
    if not os.path.isdir(src_ckpt_dir):
        print(f"  [SKIP] Checkpoint dir not found: {src_ckpt_dir}")
        return False
    if os.path.exists(dst_ckpt_dir):
        shutil.rmtree(dst_ckpt_dir)
    shutil.copytree(src_ckpt_dir, dst_ckpt_dir)
    n_files = sum(len(files) for _, _, files in os.walk(dst_ckpt_dir))
    print(f"  Copied -> {dst_ckpt_dir}  ({n_files} file(s))")
    return True


def copy_scene_combined(src_path: str, dst_path: str) -> bool:
    if not os.path.exists(src_path):
        print(f"  [SKIP] Not found: {src_path}")
        return False
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)
    size_mb = os.path.getsize(dst_path) / 1024 / 1024
    print(f"  Copied -> {dst_path}  ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description='Combine or collect scene GLB meshes.')
    parser.add_argument('--mode', type=str, default='combine',
                        choices=['combine', 'copy-scene', 'copy-appearance-scene', 'copy-bg', 'copy-ckpt', 'copy-svr'],
                        help='"combine": merge per-object GLBs; "copy-scene": copy scene_sds/scene_combined.glb; '
                             '"copy-appearance-scene": copy appearance_sds/scene_combined.glb; '
                             '"copy-bg": copy bg tsdf_fusion_post.ply; '
                             '"copy-ckpt": copy checkpoints/ folder; '
                             '"copy-svr": copy semantic_result/object_voxels/ folder.')
    parser.add_argument('--input_root', type=str, default='outputs/exp_scene_replica_0323',
                        help='Root directory containing scene subdirectories.')
    parser.add_argument('--output_root', type=str, default='exps/Replica-ab-geo',
                        help='Root directory for output files.')
    parser.add_argument('--scene', type=str, default=None,
                        help='Process only this scene (e.g. scan4-2). Default: all scenes.')
    parser.add_argument('--mesh_subdir', type=str, default='semantic_result/object_mesh',
                        help='[combine mode] Subdirectory with per-object GLB files.')
    parser.add_argument('--output_name', type=str, default='mesh_combine.glb',
                        help='[combine mode] Output filename for the combined mesh.')
    parser.add_argument('--strip_suffix', action='store_true', default=False,
                        help='Strip trailing -<digits> from scene name for output path '
                             '(e.g. scan1-7 -> scan1). Useful with copy-ckpt.')
    args = parser.parse_args()

    input_root = args.input_root
    output_root = args.output_root

    if args.scene:
        scenes = [args.scene]
    else:
        scenes = sorted(
            d for d in os.listdir(input_root)
            if os.path.isdir(os.path.join(input_root, d))
        )

    print(f"Mode       : {args.mode}")
    print(f"Input root : {input_root}")
    print(f"Output root: {output_root}")
    print(f"Scenes     : {scenes}\n")

    success, skipped = 0, 0
    for scene in scenes:
        out_scene = _strip_suffix(scene) if args.strip_suffix else scene
        print(f"[{scene}]" + (f" -> [{out_scene}]" if out_scene != scene else ""))
        if args.mode == 'combine':
            mesh_dir = os.path.join(input_root, scene, args.mesh_subdir)
            output_path = os.path.join(output_root, out_scene, args.output_name)
            ok = combine_scene(mesh_dir, output_path)
        elif args.mode == 'copy-scene':
            src = os.path.join(input_root, scene, 'scene_sds', 'scene_combined.glb')
            dst = os.path.join(output_root, out_scene, 'scene_combined.glb')
            ok = copy_scene_combined(src, dst)
        elif args.mode == 'copy-appearance-scene':
            src = os.path.join(input_root, scene, 'appearance_sds', 'scene_combined.glb')
            dst = os.path.join(output_root, out_scene, 'scene_combined.glb')
            ok = copy_scene_combined(src, dst)
        elif args.mode == 'copy-ckpt':
            src = os.path.join(input_root, scene, 'checkpoints')
            dst = os.path.join(output_root, out_scene, 'checkpoints')
            ok = copy_ckpt(src, dst)
            # ok = True
            cfg_src = os.path.join(input_root, scene, 'config.yaml')
            cfg_dst = os.path.join(output_root, out_scene, 'config.yaml')
            copy_scene_combined(cfg_src, cfg_dst)
        elif args.mode == 'copy-svr':
            src = os.path.join(input_root, scene, 'semantic_result', 'object_voxels')
            dst = os.path.join(output_root, out_scene, 'object_voxels')
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
                n_files = sum(len(files) for _, _, files in os.walk(dst))
                print(f"  Copied -> {dst}  ({n_files} file(s))")
                ok = True
            else:
                print(f"  [SKIP] Not found: {src}")
                ok = False
        else:  # copy-bg
            src = os.path.join(input_root, scene,
                               'see3d_guidance_views', 'bg_training', 'mesh', 'tsdf', 'tsdf_fusion_post.ply')
            dst = os.path.join(output_root, out_scene, 'bg.ply')
            ok = copy_scene_combined(src, dst)
        if ok:
            success += 1
        else:
            skipped += 1

    print(f"\nDone. {success} scene(s) processed, {skipped} skipped.")


if __name__ == '__main__':
    main()
