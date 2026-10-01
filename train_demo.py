"""
DecomVoxel Pipeline
"""

import glob
import os
import sys
import time
import json
import argparse
import numpy as np
from omegaconf import DictConfig, OmegaConf
import dataclasses
import subprocess
import shutil
import tempfile

# Strict GPU isolation: select one physical GPU before importing torch so this
# process never touches other GPUs (including GPU0) via CUDA runtime contexts.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    from decomvoxel.utils.gpu_select import select_free_gpu
    _selected_gpu_idx = select_free_gpu(verbose=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_selected_gpu_idx)
    os.environ["DECOMVOXEL_PHYSICAL_GPU"] = str(_selected_gpu_idx)
    print(
        f"[gpu_select] Strict isolation enabled: physical GPU {_selected_gpu_idx} "
        f"-> CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
    )

import torch


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
 
# Add GeoSVR path for render_normalize
GEOSVR_PATH = os.path.join(PROJECT_ROOT, 'decomvoxel/representation/GeoSVR')
if GEOSVR_PATH not in sys.path:
    sys.path.insert(0, GEOSVR_PATH)

GEOSVR_SRC_PATH = os.path.join(GEOSVR_PATH, 'src')
if GEOSVR_SRC_PATH not in sys.path:
    sys.path.insert(0, GEOSVR_SRC_PATH)


from decomvoxel.pipeline.load_voxel import (
    load_object_voxel,
    voxel_to_sparse_structure,
)
from decomvoxel.pipeline.visualize_sparse_structure import (
    visualize_sparse_structure,
)
from decomvoxel.pipeline.score_distillation_sampling_ss import (
    SDSConfig,
    run_sds_completion_ss,
)
from decomvoxel.pipeline.sparse_structure_to_mesh import (
    SSToMeshConfig,
    sparse_structure_to_mesh,
)
from decomvoxel.utils.converting import (
    TransformInfo,
    sparse_structure_to_svr_voxels,
    load_and_inverse_transform_glb,
    validate_ss_geometry,
)
from decomvoxel.pipeline.score_distillation_sampling_appearance import encode_object_slat, decode_slat_to_mesh, slat_sds, AppearanceSDSConfig

from decomvoxel.pipeline.cond_image_generate import generate_cond_img, generate_cond_img_avo, generate_cond_img_avo_para
from decomvoxel.utils.parallel import run_parallel, run_parallel_per_gpu, PerDeviceCache, get_visible_devices
from decomvoxel.utils.pipeline_logger import PipelineLogger

from src.config import cfg, update_config
from src.utils.system_utils import seed_everything

DEBUG=False

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def _resolve_device(selected_device=None) -> torch.device:
    """Normalize optional device input and keep CUDA current device aligned."""
    if selected_device is None:
        selected_device = device
    resolved = torch.device(selected_device) if isinstance(selected_device, str) else selected_device
    if resolved.type == 'cuda':
        torch.cuda.set_device(resolved)
    return resolved


def _get_cuda_device_name(selected_device: torch.device) -> str:
    """Return CUDA device name for the selected device without touching GPU0 by default."""
    if not torch.cuda.is_available():
        return 'N/A'
    if selected_device.type != 'cuda':
        return 'CPU'
    dev_idx = selected_device.index
    if dev_idx is None:
        dev_idx = torch.cuda.current_device()
    return torch.cuda.get_device_name(dev_idx)


def _resolve_cond_image_dir(model_path: str) -> str:
    """Resolve conditioning-image directory with backward compatibility."""
    candidates = [
        os.path.join(model_path, "cond_images", "complete_cond_avo"),
        os.path.join(model_path, "cond_images", "complete_cond"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        "No conditioning image directory found. Tried: " + ", ".join(candidates)
    )


def _normalize_img_model(img_model: str) -> str:
    model_name = "Seedream" if img_model is None else str(img_model).strip()
    compact = model_name.lower().replace("_", "").replace("-", "")
    if compact == "seedream":
        return "Seedream"
    if compact == "nanobanana":
        return "NanoBanana"
    if compact in {"gptimage2", "gptimage", "gptimg2"}:
        return "GPTImage2"
    raise ValueError(f"Unsupported img_model '{img_model}'. Expected Seedream, NanoBanana, or GPTImage2.")


def _resolve_img_model(img_model: str = "Seedream", config_path: str = None) -> str:
    resolved_model = img_model
    if config_path and os.path.exists(config_path):
        config = OmegaConf.load(config_path)
        config_model = OmegaConf.select(config, "img_model", default=None)
        if config_model is not None:
            resolved_model = config_model
            print(f"[img_model] Using config img_model={resolved_model}")
    return _normalize_img_model(resolved_model)


def _resolve_img_model_params(img_model_params: dict = None, config_path: str = None) -> dict:
    resolved_params = dict(img_model_params or {})
    if config_path and os.path.exists(config_path):
        config = OmegaConf.load(config_path)
        config_params = OmegaConf.select(config, "img_model_params", default=None)
        if config_params is not None:
            if not isinstance(config_params, (dict, DictConfig)):
                raise ValueError("config.img_model_params must be a mapping/dict")
            resolved_params = OmegaConf.to_container(config_params, resolve=True)
            print(f"[img_model] Using config img_model_params={resolved_params}")
    return resolved_params


def run_train_geosvr(cfg_files: list, source_path: str, model_path: str,
                     extra_cfg_args: list = None,
                     test_iterations: list = None,
                     pg_view_every: int = 200,
                     load_iteration: int = None):
    
    cmd_lst = [
        '--source_path', source_path,
        '--model_path', model_path,
    ]
    if extra_cfg_args:
        cmd_lst.extend(extra_cfg_args)

    update_config(cfg_files, cmd_lst)

    seed_everything(cfg.procedure.seed)
    # device = set_free_gpu(verbose=True)

    os.makedirs(cfg.model.model_path, exist_ok=True)
    with open(os.path.join(cfg.model.model_path, "config.yaml"), "w") as f:
        f.write(cfg.dump())
    print(f"[train_geosvr] output dir: {cfg.model.model_path}")

    # Handle negative values in test_iterations.
    if test_iterations is None:
        test_iterations = [cfg.procedure.n_iter]
    for i in range(len(test_iterations)):
        if test_iterations[i] < 0:
            test_iterations[i] += cfg.procedure.n_iter + 1

    # Build an args namespace for training().
    from argparse import Namespace
    train_args = Namespace(
        load_iteration=load_iteration,
        load_optimizer=False,
        save_optimizer=False,
        save_quantized=False,
        detect_anomaly=False,
        test_iterations=test_iterations,
        pg_view_every=pg_view_every,
        checkpoint_iterations=[],
    )

    from decomvoxel.representation.GeoSVR.train import training
    training(train_args)
    print("[train_geosvr] Done.")

def run_segm_3d(source_path: str, model_path: str,
                mask_dir: str = None, iteration: int = -1,
                bg_id: int = 255, num_views: int = 30):
    """3D instance segmentation: fuse 2D masks into voxels and export results."""
    from decomvoxel.representation.GeoSVR.segm_3d import segm_3d
    segm_3d(
        source_path=source_path,
        model_path=model_path,
        mask_dir=mask_dir,
        iteration=iteration,
        bg_id=bg_id,
        num_views=num_views,
    )
    print("[segm_3d] Done")


def run_build_vis_grid(
    model_path: str,
    resolution: int = 256,
    certainty_threshold: int = 3,
    checkpoint_name: str = "iter020000_model.pt",
    device: str = "cuda",
    floor_min_z: float = None,
):
    from decomvoxel.pipeline.blank_space_uncertainty import get_scene_vis_grid

    print("=" * 60)
    print("DecomVoxel: Build Visibility Grid")
    print("=" * 60)
    print(f"Model path          : {model_path}")
    print(f"Resolution          : {resolution}")
    print(f"Certainty threshold : {certainty_threshold}")
    print(f"Checkpoint          : {checkpoint_name}")
    print("=" * 60)
    
    vis_grid_path = os.path.join(model_path, "vis_grid", "vis_grid.pt")
    if os.path.exists(vis_grid_path):
        print(f"[Build Vis Grid] Found existing vis grid at {vis_grid_path}, loading ...")
        from decomvoxel.utils.vis_grid import VisibilityGrid
        vis_grid = VisibilityGrid.load(model_path, device=device)
        if floor_min_z is not None:
            _apply_floor_certainty(vis_grid, floor_min_z)
        return vis_grid

    vis_grid = get_scene_vis_grid(
        model_path=model_path,
        resolution=resolution,
        certainty_threshold=certainty_threshold,
        checkpoint_name=checkpoint_name,
        device=device,
    )

    # Force certainty = 1 for all voxels below the ground-plane z threshold.
    # These voxels are underground and should definitely remain empty.
    if floor_min_z is not None:
        _apply_floor_certainty(vis_grid, floor_min_z)

    save_path = vis_grid.save(model_path)
    vis_dir = os.path.join(model_path, "vis_grid")
    vis_grid.visualize(vis_dir)
    vis_grid.vis_invisible_pnts(os.path.join(vis_dir, "invisible_points.ply"))

    print("=" * 60)
    print("Build Visibility Grid Complete")
    print(f"  Saved to : {save_path}")
    print(f"  Vis PNG  : {os.path.join(vis_dir, 'certainty_vis.png')}")
    print(f"  Invis PLY: {os.path.join(vis_dir, 'invisible_points.ply')}")
    print("=" * 60)

    return vis_grid


def _apply_floor_certainty(vis_grid, floor_min_z: float):
    """
    Set certainty = 1.0 for every voxel whose world-space z-centre is strictly
    below *floor_min_z*.  This marks the underground region as definitively
    empty, so the blank-region preservation loss will strongly penalise any
    occupancy there.
    """
    import torch
    dev = vis_grid.device
    nx, ny, nz = vis_grid.nx, vis_grid.ny, vis_grid.nz

    # z-coordinate of each voxel centre along the z-axis (axis index 2)
    z_arange = torch.arange(nz, dtype=torch.float32, device=dev)
    z_centers = vis_grid.bbox_min[2] + (z_arange + 0.5) * vis_grid.voxel_size  # (nz,)

    # Boolean mask: True where z_centre < floor_min_z
    below_floor = z_centers < floor_min_z  # (nz,)
    n_below = below_floor.sum().item()

    if n_below == 0:
        print(f"[apply_floor_certainty] No voxels below floor_min_z={floor_min_z:.4f}; nothing to update.")
        return

    # Broadcast across (nx, ny) and apply to the certainty grid
    # certainty_grid shape: (nx, ny, nz)
    vis_grid.certainty_grid[:, :, below_floor] = 1.0
    total = nx * ny * nz
    print(f"[apply_floor_certainty] floor_min_z={floor_min_z:.4f}: "
          f"{nx}×{ny}×{n_below} = {nx*ny*n_below:,}/{total:,} voxels set to certainty=1.0")


def run_scene_sds(
    voxel_dir: str,
    output_dir: str,
    cond_images: dict,
    dataset_dir: str = None,
    config_path: str = None,
    total_iters: int = 5000,
    preserve_known: bool = True,
    threshold: float = 0.2,
    value_colored: bool = False,
    slat_steps: int = 12,
    slat_cfg_strength: float = 3.0,
    simplify_ratio: float = 0.95,
    texture_size: int = 1024,
):
    """
    Run the full scene-level SDS + mesh generation pipeline.
    """
    import trimesh

    print("=" * 60)
    print("DecomVoxel: Scene-level SDS Pipeline")
    print("=" * 60)
    print(f"Voxel dir   : {voxel_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Objects     : {list(cond_images.keys())}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Compute global ground-plane z from all object voxels in the scene.
    from decomvoxel.utils.bbox_support import get_min_z
    floor_min_z = get_min_z(voxel_dir)
    if floor_min_z is not None:
        print(f"[Scene SDS] Global floor min_z = {floor_min_z:.4f}")
    else:
        print("[Scene SDS] Could not determine floor min_z; floor clamping skipped.")
    # floor_min_z = None
    
    # Compute chair bbox modifications (expand toward nearest table in XY)
    bbox_xyz_modify_dict = {}
    if dataset_dir is not None:
        categories_json = os.path.join(dataset_dir, "object_categories.json")
        # Also accept obj_prompt.json as an alternative name (compatibility)
        if not os.path.exists(categories_json):
            _alt = os.path.join(dataset_dir, "obj_prompt.json")
            if os.path.exists(_alt):
                print(f"[Scene SDS] object_categories.json not found; using obj_prompt.json instead.")
                categories_json = _alt
        if os.path.exists(categories_json):
            from decomvoxel.utils.bbox_support import chair_bbox_xy_modify
            _debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "debug")
            bbox_xyz_modify_dict = chair_bbox_xy_modify(
                voxel_dir=voxel_dir,
                categories_json=categories_json,
                vis_output_dir=_debug_dir,
            )
            print(f"[Scene SDS] Chair bbox modifications: {bbox_xyz_modify_dict}")
        else:
            print(f"[Scene SDS] Neither object_categories.json nor obj_prompt.json found at {dataset_dir}, skipping chair bbox modify.")

    model_path = os.path.abspath(os.path.join(voxel_dir, "..", ".."))
    print("[Scene SDS] Building visibility grid before per-object SDS ...")
    vis_grid = run_build_vis_grid(
        model_path=model_path,
        device=str(device),
        floor_min_z=floor_min_z,
        # floor_min_z=None,
    )

    # ---- Load SDS config overrides ----
    cfg = SDSConfig(total_iters=total_iters)
    if config_path and os.path.exists(config_path):
        config = OmegaConf.load(config_path)
        if hasattr(config, 'sds'):
            for key, value in config.sds.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
                    
                    
    # Load object ground information
    ground_info = {}
    if dataset_dir is not None:
        location_json_path = os.path.join(dataset_dir, "location_modify.json")
        if os.path.exists(location_json_path):
            with open(location_json_path) as f:
                ground_info = {str(k): int(v) for k, v in json.load(f).items()}
            print(f"[Scene SDS] Loaded ground info for {len(ground_info)} objects from {location_json_path}")
        else:
            print(f"[Scene SDS] location_modify.json not found at {location_json_path}")

    ordered_cond_images = cond_images
    if dataset_dir is not None:
        scene_graph_path = os.path.join(dataset_dir, "scene_graph.json")
        if os.path.exists(scene_graph_path):
            from decomvoxel.utils.physics import solve_scene_graph
            ordered_cond_images = solve_scene_graph(scene_graph_path, cond_images)
            print(f"[Scene SDS] Scene graph order: {list(ordered_cond_images.keys())}")
        else:
            print(f"[Scene SDS] scene_graph.json not found at {scene_graph_path}")

    # Accumulators
    all_transforms: dict = {}   # object_name → TransformInfo.to_dict()
    all_mesh_paths: dict = {}   # object_name → world-space mesh path
    

    for obj_name, cond_image_path in ordered_cond_images.items():
        print("\n" + "#" * 60)
        print(f"# Processing object: {obj_name}")
        print("#" * 60)

        # 1. Load voxel data
        voxel_path = os.path.join(voxel_dir, f"{obj_name}_voxels.pt")
        if not os.path.exists(voxel_path):
            print(f"[Scene SDS] WARNING: voxel file not found: {voxel_path}, skipping.")
            continue

        obj_output_dir = os.path.join(output_dir, obj_name)
        os.makedirs(obj_output_dir, exist_ok=True)

        voxel_data = load_object_voxel(voxel_path, device=device)

        # 2. Convert to sparse structure & capture TransformInfo
        obj_id = obj_name.split("_")[-1].lstrip("0") or "0"
        is_ground = ground_info.get(obj_id, 0) == 1
        obj_floor_min_z = floor_min_z if is_ground else None
        obj_bbox_modify = bbox_xyz_modify_dict.get(obj_id, None)
        # Treat all-zero modify as None (no-op)
        if obj_bbox_modify is not None and all(v == 0.0 for v in obj_bbox_modify):
            obj_bbox_modify = None
        sparse_structure_obj = voxel_to_sparse_structure(
            voxel_data,
            device=device,
            vis_grid=vis_grid,
            floor_min_z=obj_floor_min_z,
            bbox_modify=obj_bbox_modify,
        )
        transform_info: TransformInfo = sparse_structure_obj.transform_info
        
        if hasattr(cfg, 'rendering_obj_id'):
            setattr(cfg, 'rendering_obj_id', int(obj_id))

        if cfg.rotate_initial_structure:
            sparse_structure_obj.rotate_z(cfg.rotation_angle_z)
            transform_info.rotation_angle_z = cfg.rotation_angle_z

        sparse_structure = sparse_structure_obj.get_occupancy_grid()  # (1,1,64,64,64)

        # Save per-object TransformInfo (.pt + dict)
        ti_path = os.path.join(obj_output_dir, "transform_info.pt")
        transform_info.save(ti_path)
        all_transforms[obj_name] = transform_info.to_dict()

        print(f"[Scene SDS] TransformInfo for {obj_name}: {transform_info}")

        # 3. (Optional) Visualise initial sparse structure
        initial_viz_dir = os.path.join(obj_output_dir, 'initial_ss_visualization')
        os.makedirs(initial_viz_dir, exist_ok=True)
        visualize_sparse_structure(
            data=sparse_structure,
            output_dir=initial_viz_dir,
            mode='all',
            threshold=threshold,
            value_colored=value_colored,
            use_scatter=False,
            show=False,
            certainty=sparse_structure_obj.certainty_grid,
            blank_uncertainty=sparse_structure_obj.blank_space_uncertainty
        )

        # 4. Run SDS completion
        if not os.path.isabs(cond_image_path):
            cond_image_path = os.path.join(PROJECT_ROOT, cond_image_path)
        if not os.path.exists(cond_image_path):
            print(f"[Scene SDS] WARNING: cond image not found: {cond_image_path}, skipping {obj_name}.")
            continue

        print(f"[Scene SDS] Conditioning image: {cond_image_path}")

        sds_output_dir = os.path.join(obj_output_dir, 'sds_completion_ss')
        known_mask = None
        if preserve_known:
            known_mask = (sparse_structure > 0.5).float()
            print(f"[Scene SDS] Known mask: {known_mask.sum().item():.0f} voxels")

        sds_result = run_sds_completion_ss(
            sparse_structure=sparse_structure,
            image_path=cond_image_path,
            output_dir=sds_output_dir,
            cfg=cfg,
            known_mask=known_mask,
            certainty_grid=sparse_structure_obj.certainty_grid,
            refine=False,
            transform_info=sparse_structure_obj.transform_info,
            blank_space_uncertainty=sparse_structure_obj.blank_space_uncertainty,
        )

        completed_ss = sds_result.get('completed')
        final_target = sds_result.get('final_target')

        # Save completed sparse structure
        completed_ss_path = os.path.join(sds_output_dir, 'completed_ss.pt')
        if not os.path.exists(completed_ss_path):
            torch.save(completed_ss, completed_ss_path)

        # Use final_target (trellis prediction) for mesh generation if available,
        # otherwise fall back to completed_ss
        mesh_ss = final_target if final_target is not None else completed_ss

        # 5. Sparse Structure → Mesh (TRELLIS structured latents)
        print(f"[Scene SDS] Generating mesh for {obj_name} (using {'final_target' if final_target is not None else 'completed_ss'}) ...")
        mesh_output_dir = os.path.join(obj_output_dir, 'ss_to_mesh')
        ss_tensor = mesh_ss
        if ss_tensor.dim() == 3:
            ss_tensor = ss_tensor.unsqueeze(0).unsqueeze(0)
        elif ss_tensor.dim() == 4:
            ss_tensor = ss_tensor.unsqueeze(0)

        mesh_cfg = SSToMeshConfig(
            seed=42,
            slat_sampler_steps=slat_steps,
            slat_sampler_cfg_strength=slat_cfg_strength,
            simplify_ratio=simplify_ratio,
            texture_size=texture_size,
        )
        mesh_result = sparse_structure_to_mesh(
            sparse_structure=ss_tensor,
            image=cond_image_path,
            output_dir=mesh_output_dir,
            cfg=mesh_cfg,
            threshold=0.5,
            preprocess_image=True,
            export_glb=True,
            export_obj=False,
            export_ply=False,
        )

        exported = mesh_result.get('exported_files', {})
        glb_path = exported.get('glb', None)

        # 6. Inverse-transform mesh back to scene coordinates
        if glb_path and os.path.exists(glb_path):
            print(f"[Scene SDS] Inverse-transforming mesh to world coords ...")
            world_mesh = load_and_inverse_transform_glb(glb_path, transform_info)

            world_mesh_path = os.path.join(obj_output_dir, f"{obj_name}_world.glb")
            # Export as GLB in world coordinates
            world_mesh.export(world_mesh_path)
            all_mesh_paths[obj_name] = world_mesh_path
            print(f"[Scene SDS] World mesh saved: {world_mesh_path}")
        else:
            print(f"[Scene SDS] WARNING: no GLB exported for {obj_name}, skipping inverse transform.")

        ss_for_vis = mesh_ss
        if ss_for_vis.dim() == 4:
            ss_for_vis = ss_for_vis.squeeze(0)
        vis_update = sparse_structure_to_svr_voxels(
            sparse_structure=ss_for_vis,
            transform_info=transform_info,
            threshold=0.5,
            device=device,
        )
        n_updated = vis_grid.update_certainty_from_points(vis_update['centers'])
        vis_grid.save(model_path)
        print(f"[Scene SDS] Updated vis_grid after {obj_name}: {n_updated} cells set to certainty=1")

    # 7. Save aggregated transforms JSON
    json_path = os.path.join(output_dir, 'scene_transforms.json')
    json_data = {
        'description': 'Forward transform info (scene → 64^3 grid) for each object',
        'objects': all_transforms,
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\n[Scene SDS] Transform JSON saved: {json_path}")

    # 8. Merge all world-space meshes into one scene GLB by scanning the output
    # directory.  This is checkpoint-safe: any object that was already processed
    # in a previous run (or refined independently) is picked up automatically.
    print("\n[Scene SDS] Scanning output dir for world-space GLBs to merge ...")
    world_glb_paths = sorted(
        glob.glob(os.path.join(output_dir, "*", "*_world.glb"))
    )
    if world_glb_paths:
        combined_scene = trimesh.Scene()
        for glb_file in world_glb_paths:
            obj_name = os.path.basename(os.path.dirname(glb_file))  # e.g. "object_001"
            print(f"[Scene SDS]   adding {obj_name} from {glb_file}")
            loaded = trimesh.load(glb_file, force='scene')
            if isinstance(loaded, trimesh.Scene):
                for geom_name, geom in loaded.geometry.items():
                    try:
                        node_tf, _ = loaded.graph.get(geom_name)
                    except Exception:
                        node_tf = np.eye(4)
                    combined_scene.add_geometry(
                        geom,
                        node_name=f"{obj_name}_{geom_name}",
                        transform=node_tf,
                    )
            else:
                combined_scene.add_geometry(loaded, node_name=obj_name)
        combined_path = os.path.join(output_dir, 'scene_combined.glb')
        combined_scene.export(combined_path)
        print(f"[Scene SDS] Combined scene mesh ({len(world_glb_paths)} objects): {combined_path}")
    else:
        print("[Scene SDS] No world-space GLBs found; skipping scene merge.")

    print("\n" + "=" * 60)
    print("Scene-level SDS Pipeline Complete")
    print("=" * 60)
    for obj_name, mp in all_mesh_paths.items():
        print(f"  {obj_name}: {mp}")
    print(f"  transforms JSON: {json_path}")

    return {
        'transforms': all_transforms,
        'mesh_paths': all_mesh_paths,
        'json_path': json_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess-based parallel worker infrastructure
# Each object is processed in a completely independent Python subprocess so
# that every GPU worker has its own CUDA context and cuBLAS handle.
# Threads are used *only* to manage subprocess lifecycle (I/O-bound), not for
# CUDA work, so there is no shared CUDA state between concurrent workers.
# ─────────────────────────────────────────────────────────────────────────────

def _to_json_serializable(obj):
    """Recursively convert numpy arrays / torch tensors to JSON-safe types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    return obj


def _run_objects_subprocess(
    tasks: list,
    mode: str,
    devices: list,
    parallel_num_per_gpu: int,
    desc: str = "worker",
) -> list:
    """
    Launch each task as a subprocess worker (``python train_demo.py --mode <mode>``),
    keeping at most ``parallel_num_per_gpu`` jobs alive per GPU at any time.
    ``tasks`` is a list of ``(key, config_dict)`` pairs; each config_dict must
    contain ``'_result_path'`` where the worker writes its JSON result.
    Returns a list of ``(key, result_dict_or_None, exception_or_None)`` in
    completion order.  Threads here only block on subprocess I/O — no CUDA.
    """
    import queue
    from concurrent.futures import ThreadPoolExecutor, as_completed

    slot_q = queue.Queue()
    for d in devices:
        for _ in range(parallel_num_per_gpu):
            slot_q.put(d)

    n = len(tasks)
    _script = os.path.abspath(__file__)

    def _launch_one(idx, key, config_dict):
        device_str = slot_q.get()
        try:
            cfg = dict(config_dict)
            cfg['device'] = device_str
            # Write config next to result file so temp dirs are co-located
            cfg_path = cfg['_result_path'] + '.cfg.json'
            with open(cfg_path, 'w') as f:
                json.dump(cfg, f)
            print(f"[subprocess/{desc}] [{idx+1}/{n}] '{key}' START on {device_str}")
            cmd = [sys.executable, _script, '--mode', mode, '--worker_json', cfg_path]
            proc = subprocess.run(cmd)  # stdout/stderr pass through to parent
            if proc.returncode != 0:
                print(f"[subprocess/{desc}] [{idx+1}/{n}] '{key}' FAIL (exit {proc.returncode})")
                return key, None, RuntimeError(f"subprocess exit {proc.returncode}")
            result_path = cfg['_result_path']
            if os.path.exists(result_path):
                with open(result_path) as f:
                    result = json.load(f)
                print(f"[subprocess/{desc}] [{idx+1}/{n}] '{key}' OK")
                return key, result, None
            else:
                print(f"[subprocess/{desc}] [{idx+1}/{n}] '{key}' FAIL (no result file)")
                return key, None, RuntimeError("no result file written by worker")
        except Exception as e:
            print(f"[subprocess/{desc}] [{idx+1}/{n}] '{key}' EXCEPTION: {e}")
            return key, None, e
        finally:
            slot_q.put(device_str)

    results = []
    max_workers = len(devices) * parallel_num_per_gpu
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_launch_one, i, key, cfg) for i, (key, cfg) in enumerate(tasks)]
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


def _worker_sds_entry(json_path: str):
    """
    Subprocess entry point for scene-level geometry SDS (one object per call).
    Invoked as: python train_demo.py --mode _worker_sds --worker_json <path>
    """
    with open(json_path) as f:
        config = json.load(f)

    device_str   = config['device']
    local_device = torch.device(device_str)
    torch.cuda.set_device(local_device)

    from decomvoxel.utils.vis_grid import VisibilityGrid
    local_vis_grid = VisibilityGrid.load(config['model_path'], device=device_str)

    local_cfg = SDSConfig()
    for k, v in config.get('sds_cfg', {}).items():
        if hasattr(local_cfg, k):
            setattr(local_cfg, k, v)

    obj_name          = config['obj_name']
    voxel_dir         = config['voxel_dir']
    output_dir        = config['output_dir']
    cond_image_path   = config['cond_image_path']
    obj_floor_min_z   = config.get('obj_floor_min_z')
    obj_bbox_modify   = config.get('obj_bbox_modify')
    preserve_known    = config.get('preserve_known', True)
    threshold         = config.get('threshold', 0.2)
    value_colored     = config.get('value_colored', False)
    slat_steps        = config.get('slat_steps', 12)
    slat_cfg_strength = config.get('slat_cfg_strength', 3.0)
    simplify_ratio    = config.get('simplify_ratio', 0.95)
    texture_size      = config.get('texture_size', 1024)
    result_path       = config['_result_path']

    voxel_path = os.path.join(voxel_dir, f"{obj_name}_voxels.pt")
    if not os.path.exists(voxel_path):
        print(f"[worker_sds] WARNING: voxel file not found: {voxel_path}")
        return

    obj_output_dir = os.path.join(output_dir, obj_name)
    os.makedirs(obj_output_dir, exist_ok=True)

    voxel_data = load_object_voxel(voxel_path, device=local_device)
    obj_id = obj_name.split("_")[-1].lstrip("0") or "0"
    if obj_bbox_modify is not None and all(v == 0.0 for v in obj_bbox_modify):
        obj_bbox_modify = None

    sparse_structure_obj = voxel_to_sparse_structure(
        voxel_data,
        device=local_device,
        vis_grid=local_vis_grid,
        floor_min_z=obj_floor_min_z,
        bbox_modify=obj_bbox_modify,
    )
    transform_info = sparse_structure_obj.transform_info

    if hasattr(local_cfg, 'rendering_obj_id'):
        setattr(local_cfg, 'rendering_obj_id', int(obj_id))
    if local_cfg.rotate_initial_structure:
        sparse_structure_obj.rotate_z(local_cfg.rotation_angle_z)
        transform_info.rotation_angle_z = local_cfg.rotation_angle_z

    sparse_structure = sparse_structure_obj.get_occupancy_grid()

    ti_path = os.path.join(obj_output_dir, "transform_info.pt")
    transform_info.save(ti_path)

    initial_viz_dir = os.path.join(obj_output_dir, 'initial_ss_visualization')
    os.makedirs(initial_viz_dir, exist_ok=True)
    visualize_sparse_structure(
        data=sparse_structure,
        output_dir=initial_viz_dir,
        mode='all',
        threshold=threshold,
        value_colored=value_colored,
        use_scatter=False,
        show=False,
        certainty=sparse_structure_obj.certainty_grid,
        blank_uncertainty=sparse_structure_obj.blank_space_uncertainty,
    )

    if not os.path.isabs(cond_image_path):
        cond_image_path = os.path.join(PROJECT_ROOT, cond_image_path)
    if not os.path.exists(cond_image_path):
        print(f"[worker_sds] WARNING: cond image not found: {cond_image_path}")
        return

    sds_output_dir = os.path.join(obj_output_dir, 'sds_completion_ss')
    known_mask = None
    if preserve_known:
        known_mask = (sparse_structure > 0.5).float()

    sds_result = run_sds_completion_ss(
        sparse_structure=sparse_structure,
        image_path=cond_image_path,
        output_dir=sds_output_dir,
        cfg=local_cfg,
        known_mask=known_mask,
        certainty_grid=sparse_structure_obj.certainty_grid,
        refine=False,
        transform_info=sparse_structure_obj.transform_info,
        blank_space_uncertainty=sparse_structure_obj.blank_space_uncertainty,
    )

    completed_ss  = sds_result.get('completed')
    final_target  = sds_result.get('final_target')

    completed_ss_path = os.path.join(sds_output_dir, 'completed_ss.pt')
    if not os.path.exists(completed_ss_path):
        torch.save(completed_ss, completed_ss_path)

    mesh_ss = final_target if final_target is not None else completed_ss

    mesh_output_dir = os.path.join(obj_output_dir, 'ss_to_mesh')
    ss_tensor = mesh_ss
    if ss_tensor.dim() == 3:
        ss_tensor = ss_tensor.unsqueeze(0).unsqueeze(0)
    elif ss_tensor.dim() == 4:
        ss_tensor = ss_tensor.unsqueeze(0)

    mesh_cfg = SSToMeshConfig(
        seed=42,
        slat_sampler_steps=slat_steps,
        slat_sampler_cfg_strength=slat_cfg_strength,
        simplify_ratio=simplify_ratio,
        texture_size=texture_size,
    )
    mesh_result = sparse_structure_to_mesh(
        sparse_structure=ss_tensor,
        image=cond_image_path,
        output_dir=mesh_output_dir,
        cfg=mesh_cfg,
        threshold=0.5,
        preprocess_image=True,
        export_glb=True,
        export_obj=False,
        export_ply=False,
    )

    exported        = mesh_result.get('exported_files', {})
    glb_path        = exported.get('glb', None)
    world_mesh_path = None
    if glb_path and os.path.exists(glb_path):
        world_mesh = load_and_inverse_transform_glb(glb_path, transform_info)
        world_mesh_path = os.path.join(obj_output_dir, f"{obj_name}_world.glb")
        world_mesh.export(world_mesh_path)
    else:
        print(f"[worker_sds] WARNING: no GLB exported for {obj_name}")

    ss_for_vis = mesh_ss
    if ss_for_vis.dim() == 4:
        ss_for_vis = ss_for_vis.squeeze(0)
    vis_update  = sparse_structure_to_svr_voxels(
        sparse_structure=ss_for_vis,
        transform_info=transform_info,
        threshold=0.5,
        device=local_device,
    )
    vis_centers = vis_update['centers'].detach().cpu().numpy().tolist()

    result = {
        'obj_name':           obj_name,
        'transform_info_dict': _to_json_serializable(transform_info.to_dict()),
        'world_mesh_path':    world_mesh_path,
        'vis_update_centers': vis_centers,
    }
    with open(result_path, 'w') as f:
        json.dump(result, f)
    print(f"[worker_sds] {obj_name}: done → {result_path}")


def _worker_app_sds_entry(json_path: str):
    """
    Subprocess entry point for scene-level appearance SDS (one object per call).
    Invoked as: python train_demo.py --mode _worker_app_sds --worker_json <path>
    """
    from datetime import datetime
    from decomvoxel.utils.vis_slat import visualize_slat
    import json as _json

    with open(json_path) as f:
        config = json.load(f)

    device_str   = config['device']
    local_device = torch.device(device_str)
    torch.cuda.set_device(local_device)

    obj_id           = int(config['obj_id'])
    model_path       = config['model_path']
    dataset_dir_w    = config.get('dataset_dir')
    output_dir       = config['output_dir']
    geometry_sds_dir = config['geometry_sds_dir']
    voxel_dir        = config['voxel_dir']
    cond_image       = config['cond_image']
    obj_floor_min_z  = config.get('obj_floor_min_z')
    obj_bbox_modify  = config.get('obj_bbox_modify')
    config_path_w    = config.get('config_path')
    result_path      = config['_result_path']
    APP_SDS_MIN_VOXELS = config.get('APP_SDS_MIN_VOXELS', 32)
    APP_SDS_MAX_VOXELS = config.get('APP_SDS_MAX_VOXELS', 80000)

    # Backward compatibility for stale worker configs that still point to
    # cond_images/complete_cond/<id>.png while only complete_cond_avo exists.
    if not os.path.exists(cond_image):
        cond_image_alt = cond_image.replace('/complete_cond/', '/complete_cond_avo/')
        if cond_image_alt != cond_image and os.path.exists(cond_image_alt):
            print(f"[worker_app_sds] Cond image fallback: {cond_image} -> {cond_image_alt}")
            cond_image = cond_image_alt

    voxel_path = os.path.join(voxel_dir, f"object_{obj_id:03d}_voxels.pt")
    if not os.path.exists(voxel_path):
        print(f"[worker_app_sds] WARNING: voxel not found: {voxel_path}")
        return

    voxel_data = load_object_voxel(voxel_path, device=local_device)
    if obj_bbox_modify is not None and all(v == 0.0 for v in obj_bbox_modify):
        obj_bbox_modify = None

    sparse_structure_obj = voxel_to_sparse_structure(
        voxel_data,
        device=local_device,
        model_path=model_path,
        floor_min_z=obj_floor_min_z,
        bbox_modify=obj_bbox_modify,
    )
    sparse_structure = sparse_structure_obj.get_occupancy_grid()

    output_dir_obj = os.path.join(output_dir, f"obj_{obj_id:03d}")
    os.makedirs(output_dir_obj, exist_ok=True)

    ss_path = os.path.join(geometry_sds_dir, f"object_{obj_id:03d}", "sds_completion_ss", "final_target.pt")

    initial_viz_dir = os.path.join(output_dir_obj, 'initial_ss_visualization')
    os.makedirs(initial_viz_dir, exist_ok=True)
    visualize_sparse_structure(
        data=sparse_structure,
        output_dir=initial_viz_dir,
        mode='all',
        threshold=0.2,
        value_colored=False,
        use_scatter=False,
        show=False,
        certainty=sparse_structure_obj.certainty_grid,
    )

    out = sparse_structure_to_svr_voxels(
        sparse_structure, sparse_structure_obj.transform_info, threshold=0.2, device=local_device
    )
    centers_init     = out['centers']
    grid_coords_init = out['grid_coords']
    n_init           = int(grid_coords_init.shape[0])

    sds_grid_coords = sds_centers = None
    n_sds = -1
    if os.path.exists(ss_path):
        ss_loaded = torch.load(ss_path, map_location=local_device)
        if ss_loaded.dim() == 5:
            ss_loaded = ss_loaded.squeeze(0).squeeze(0)
        elif ss_loaded.dim() == 4:
            ss_loaded = ss_loaded.squeeze(0)
        out_ss = sparse_structure_to_svr_voxels(
            ss_loaded, sparse_structure_obj.transform_info, threshold=0.5, device=local_device
        )
        sds_grid_coords = out_ss['grid_coords']
        sds_centers     = out_ss['centers']
        n_sds           = int(sds_grid_coords.shape[0])
    else:
        print(f"[worker_app_sds] WARNING: final_target not found: {ss_path}")

    def _in_range(n):
        return APP_SDS_MIN_VOXELS <= n <= APP_SDS_MAX_VOXELS

    if _in_range(n_sds):
        grid_coords   = sds_grid_coords
        centers       = sds_centers
        ss_source_tag = 'sds'
    elif _in_range(n_init):
        print(f"[worker_app_sds] obj={obj_id:03d}: n_sds={n_sds} out of range, fallback to initial (n={n_init})")
        grid_coords   = grid_coords_init
        centers       = centers_init
        ss_source_tag = 'initial_fallback'
    else:
        print(f"[worker_app_sds] obj={obj_id:03d}: both SDS ({n_sds}) and initial ({n_init}) out of range, skip.")
        return

    app_cfg = AppearanceSDSConfig()
    for k, v in config.get('app_cfg', {}).items():
        if hasattr(app_cfg, k):
            setattr(app_cfg, k, v)

    latent = encode_object_slat(
        source_path=dataset_dir_w,
        obj_id=obj_id,
        voxel_coords=grid_coords,
        voxel_centers_world=centers,
        output_path=os.path.join(output_dir_obj, f'obj{obj_id}_slat.npz'),
        use_non_zero_init=app_cfg.use_non_zero_init,
        non_zero_init_k=app_cfg.non_zero_init_k,
        device=local_device,
    )
    visualize_slat(latent, output_dir=os.path.join(output_dir_obj, 'slat_visualization'), prefix='before_sds')

    log_info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': dataclasses.asdict(app_cfg),
        'input_info': {
            'cond_image': cond_image, 'output_dir': output_dir_obj,
            'obj_id': obj_id, 'config_path': config_path_w,
            'ss_path': ss_path, 'ss_source': ss_source_tag,
            'n_voxels': int(grid_coords.shape[0]),
            'n_voxels_sds': int(n_sds), 'n_voxels_initial': int(n_init),
        },
        'system_info': {
            'device': str(local_device),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_name': _get_cuda_device_name(local_device),
        },
    }
    log_path = os.path.join(output_dir_obj, 'appearance_sds_config_log.json')
    with open(log_path, 'w') as _f:
        _json.dump(log_info, _f, indent=2)

    latent = slat_sds(
        latent,
        cond_image=cond_image,
        output_dir=output_dir_obj,
        cfg=app_cfg,
        certainty_grid=sparse_structure_obj.certainty_grid,
        device=local_device,
    )
    visualize_slat(latent, output_dir=os.path.join(output_dir_obj, 'slat_visualization'), prefix='after_sds')

    glb_path = os.path.join(output_dir_obj, f'obj{obj_id}_mesh.glb')
    decode_slat_to_mesh(latent, output_path=glb_path, device=local_device)

    obj_name       = f'obj_{obj_id:03d}'
    transform_info = sparse_structure_obj.transform_info
    world_mesh_path = None
    if os.path.exists(glb_path):
        world_mesh = load_and_inverse_transform_glb(glb_path, transform_info)
        world_mesh_path = os.path.join(output_dir_obj, f'{obj_name}_world.glb')
        world_mesh.export(world_mesh_path)
        print(f"[worker_app_sds] World mesh saved: {world_mesh_path}")
    else:
        print(f"[worker_app_sds] WARNING: GLB not found for {obj_name}")

    result = {'obj_name': obj_name, 'world_mesh_path': world_mesh_path}
    with open(result_path, 'w') as f:
        json.dump(result, f)
    print(f"[worker_app_sds] obj_{obj_id:03d}: done → {result_path}")


def run_scene_sds_para(
    voxel_dir: str,
    output_dir: str,
    cond_images: dict,
    dataset_dir: str = None,
    config_path: str = None,
    total_iters: int = 5000,
    preserve_known: bool = True,
    threshold: float = 0.2,
    value_colored: bool = False,
    slat_steps: int = 12,
    slat_cfg_strength: float = 3.0,
    simplify_ratio: float = 0.95,
    texture_size: int = 1024,
    parallel_num: int = 4,
):
    """Parallel variant of ``run_scene_sds``.

    The per-object SDS loop is dispatched to a thread pool with
    ``max_workers=parallel_num``. The visibility grid used as input to each
    object is the snapshot built at the start (per-object updates are still
    accumulated and the merged grid is saved once at the end). All
    cross-object aggregation (transforms JSON, merged GLB) happens after the
    pool drains.
    """
    import trimesh

    print("=" * 60)
    print("DecomVoxel: Scene-level SDS Pipeline (parallel)")
    print("=" * 60)
    print(f"Voxel dir            : {voxel_dir}")
    print(f"Output dir           : {output_dir}")
    print(f"Objects              : {list(cond_images.keys())}")
    print(f"parallel_num_per_gpu : {parallel_num}")
    print(f"visible devices      : {get_visible_devices()}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    from decomvoxel.utils.bbox_support import get_min_z
    floor_min_z = get_min_z(voxel_dir)
    if floor_min_z is not None:
        print(f"[Scene SDS Para] Global floor min_z = {floor_min_z:.4f}")
    else:
        print("[Scene SDS Para] Could not determine floor min_z; floor clamping skipped.")

    bbox_xyz_modify_dict = {}
    if dataset_dir is not None:
        categories_json = os.path.join(dataset_dir, "object_categories.json")
        if not os.path.exists(categories_json):
            _alt = os.path.join(dataset_dir, "obj_prompt.json")
            if os.path.exists(_alt):
                print(f"[Scene SDS Para] object_categories.json not found; using obj_prompt.json instead.")
                categories_json = _alt
        if os.path.exists(categories_json):
            from decomvoxel.utils.bbox_support import chair_bbox_xy_modify
            _debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "debug")
            bbox_xyz_modify_dict = chair_bbox_xy_modify(
                voxel_dir=voxel_dir,
                categories_json=categories_json,
                vis_output_dir=_debug_dir,
            )
            print(f"[Scene SDS Para] Chair bbox modifications: {bbox_xyz_modify_dict}")
        else:
            print(f"[Scene SDS Para] Neither object_categories.json nor obj_prompt.json found at {dataset_dir}, skipping chair bbox modify.")

    model_path = os.path.abspath(os.path.join(voxel_dir, "..", ".."))
    print("[Scene SDS Para] Building visibility grid before per-object SDS ...")
    vis_grid = run_build_vis_grid(
        model_path=model_path,
        device=str(device),
        floor_min_z=floor_min_z,
    )

    cfg = SDSConfig(total_iters=total_iters)
    if config_path and os.path.exists(config_path):
        config = OmegaConf.load(config_path)
        if hasattr(config, 'sds'):
            for key, value in config.sds.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)

    ground_info = {}
    if dataset_dir is not None:
        location_json_path = os.path.join(dataset_dir, "location_modify.json")
        if os.path.exists(location_json_path):
            with open(location_json_path) as f:
                ground_info = {str(k): int(v) for k, v in json.load(f).items()}
            print(f"[Scene SDS Para] Loaded ground info for {len(ground_info)} objects from {location_json_path}")
        else:
            print(f"[Scene SDS Para] location_modify.json not found at {location_json_path}")

    ordered_cond_images = cond_images
    if dataset_dir is not None:
        scene_graph_path = os.path.join(dataset_dir, "scene_graph.json")
        if os.path.exists(scene_graph_path):
            from decomvoxel.utils.physics import solve_scene_graph
            ordered_cond_images = solve_scene_graph(scene_graph_path, cond_images)
            print(f"[Scene SDS Para] Scene graph order: {list(ordered_cond_images.keys())}")
        else:
            print(f"[Scene SDS Para] scene_graph.json not found at {scene_graph_path}")

    # ── Subprocess-based parallelism: one independent process per object ──────
    # Each subprocess has its own CUDA context, so cuBLAS handles never clash.
    devices = get_visible_devices()
    tasks_sp = []
    for i, (obj_name, cond_image_path) in enumerate(ordered_cond_images.items()):
        obj_id      = obj_name.split("_")[-1].lstrip("0") or "0"
        is_ground   = ground_info.get(obj_id, 0) == 1
        obj_floor_min_z = floor_min_z if is_ground else None
        obj_bbox_modify = bbox_xyz_modify_dict.get(obj_id, None)
        if obj_bbox_modify is not None and all(v == 0.0 for v in obj_bbox_modify):
            obj_bbox_modify = None
        result_path = os.path.join(output_dir, f'._sds_result_{i}_{obj_name}.json')
        config_dict = {
            'obj_name':         obj_name,
            'cond_image_path':  cond_image_path,
            'voxel_dir':        voxel_dir,
            'output_dir':       output_dir,
            'model_path':       model_path,
            'floor_min_z':      floor_min_z,
            'obj_floor_min_z':  obj_floor_min_z,
            'obj_bbox_modify':  obj_bbox_modify,
            'ground_info':      ground_info,
            'sds_cfg':          dataclasses.asdict(cfg),
            'preserve_known':   preserve_known,
            'threshold':        threshold,
            'value_colored':    value_colored,
            'slat_steps':       slat_steps,
            'slat_cfg_strength': slat_cfg_strength,
            'simplify_ratio':   simplify_ratio,
            'texture_size':     texture_size,
            '_result_path':     result_path,
        }
        tasks_sp.append((obj_name, config_dict))

    sp_results = _run_objects_subprocess(
        tasks=tasks_sp,
        mode='_worker_sds',
        devices=devices,
        parallel_num_per_gpu=parallel_num,
        desc='scene_sds',
    )

    all_transforms: dict = {}
    all_mesh_paths: dict = {}
    vis_updates: list = []
    failed_objects = []
    for obj_name, result, exc in sp_results:
        if exc is not None or result is None:
            failed_objects.append(obj_name)
            continue
        all_transforms[obj_name] = result.get('transform_info_dict', {})
        if result.get('world_mesh_path'):
            all_mesh_paths[obj_name] = result['world_mesh_path']
        centers_data = result.get('vis_update_centers')
        if centers_data:
            vis_updates.append((obj_name, torch.tensor(centers_data, dtype=torch.float32)))

    if failed_objects:
        raise RuntimeError(
            "Scene SDS subprocess workers failed for: " + ", ".join(sorted(failed_objects))
        )

    # Apply visibility-grid updates and persist
    for obj_name, centers in vis_updates:
        n_updated = vis_grid.update_certainty_from_points(centers)
        print(f"[Scene SDS Para] vis_grid update from {obj_name}: {n_updated} cells -> certainty=1")
    vis_grid.save(model_path)

    json_path = os.path.join(output_dir, 'scene_transforms.json')
    json_data = {
        'description': 'Forward transform info (scene → 64^3 grid) for each object',
        'objects': all_transforms,
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\n[Scene SDS Para] Transform JSON saved: {json_path}")

    print("\n[Scene SDS Para] Scanning output dir for world-space GLBs to merge ...")
    world_glb_paths = sorted(
        glob.glob(os.path.join(output_dir, "*", "*_world.glb"))
    )
    if world_glb_paths:
        combined_scene = trimesh.Scene()
        for glb_file in world_glb_paths:
            obj_name = os.path.basename(os.path.dirname(glb_file))
            print(f"[Scene SDS Para]   adding {obj_name} from {glb_file}")
            loaded = trimesh.load(glb_file, force='scene')
            if isinstance(loaded, trimesh.Scene):
                for geom_name, geom in loaded.geometry.items():
                    try:
                        node_tf, _ = loaded.graph.get(geom_name)
                    except Exception:
                        node_tf = np.eye(4)
                    combined_scene.add_geometry(
                        geom,
                        node_name=f"{obj_name}_{geom_name}",
                        transform=node_tf,
                    )
            else:
                combined_scene.add_geometry(loaded, node_name=obj_name)
        combined_path = os.path.join(output_dir, 'scene_combined.glb')
        combined_scene.export(combined_path)
        print(f"[Scene SDS Para] Combined scene mesh ({len(world_glb_paths)} objects): {combined_path}")
    else:
        print("[Scene SDS Para] No world-space GLBs found; skipping scene merge.")

    print("\n" + "=" * 60)
    print("Scene-level SDS Pipeline (parallel) Complete")
    print("=" * 60)
    for obj_name, mp in all_mesh_paths.items():
        print(f"  {obj_name}: {mp}")
    print(f"  transforms JSON: {json_path}")

    return {
        'transforms': all_transforms,
        'mesh_paths': all_mesh_paths,
        'json_path': json_path,
    }


def run_scene_appearance_sds(dataset_dir: str, model_path: str, config_path: str, device: str = "cuda",
                             cond_images: dict = None):
    """
    cond_images: optional dict mapping "object_XXX" -> abs/rel path to the conditioning PNG.
                 If None, all PNGs found in <model_path>/cond_images/complete_cond_avo/ are used.
                 Pass an explicit dict to restrict processing to specific objects.
    """
    device = _resolve_device(device)
    from decomvoxel.utils.vis_slat import visualize_slat

    cond_img_dir = _resolve_cond_image_dir(model_path)

    if cond_images is not None:
        # Explicit list: derive obj_ids from the dict keys ("object_XXX" → XXX)
        print(f"[Scene Appearance SDS] Using caller-supplied cond_images ({len(cond_images)} objects)")
        obj_ids_num = sorted([int(k.split("_")[-1]) for k in cond_images.keys()])
    else:
        # Default: scan the cond_images directory
        cond_img_names = sorted([f for f in os.listdir(cond_img_dir) if f.endswith(".png")])
        obj_ids = [f.split(".")[0] for f in cond_img_names]  # str: 002, 009, 011...
        obj_ids_num = [int(oid) for oid in obj_ids]
    
    output_dir = os.path.join(model_path, "appearance_sds")
    os.makedirs(output_dir, exist_ok=True)
    
    # scan1/scene_sds/object_002/sds_completion_ss/final_target.pt
    geometry_sds_dir = os.path.join(model_path, "scene_sds") 
    
    voxel_dir = os.path.join(model_path, "semantic_result", "object_voxels")

    # Keep transform computation identical to run_scene_sds.
    from decomvoxel.utils.bbox_support import get_min_z
    floor_min_z = get_min_z(voxel_dir)
    if floor_min_z is not None:
        print(f"[Scene Appearance SDS] Global floor min_z = {floor_min_z:.4f}")
    else:
        print("[Scene Appearance SDS] Could not determine floor min_z; floor clamping skipped.")

    bbox_xyz_modify_dict = {}
    if dataset_dir is not None:
        categories_json = os.path.join(dataset_dir, "object_categories.json")
        # Also accept obj_prompt.json as an alternative name (compatibility)
        if not os.path.exists(categories_json):
            _alt = os.path.join(dataset_dir, "obj_prompt.json")
            if os.path.exists(_alt):
                print(f"[Scene Appearance SDS] object_categories.json not found; using obj_prompt.json instead.")
                categories_json = _alt
        if os.path.exists(categories_json):
            from decomvoxel.utils.bbox_support import chair_bbox_xy_modify
            _debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "debug")
            bbox_xyz_modify_dict = chair_bbox_xy_modify(
                voxel_dir=voxel_dir,
                categories_json=categories_json,
                vis_output_dir=_debug_dir,
            )
            print(f"[Scene Appearance SDS] Chair bbox modifications: {bbox_xyz_modify_dict}")
        else:
            print(f"[Scene Appearance SDS] Neither object_categories.json nor obj_prompt.json found at {dataset_dir}, skipping chair bbox modify.")

    ground_info = {}
    if dataset_dir is not None:
        location_json_path = os.path.join(dataset_dir, "location_modify.json")
        if os.path.exists(location_json_path):
            import json
            with open(location_json_path) as f:
                ground_info = {str(k): int(v) for k, v in json.load(f).items()}
            print(f"[Scene Appearance SDS] Loaded ground info for {len(ground_info)} objects from {location_json_path}")
        else:
            print(f"[Scene Appearance SDS] location_modify.json not found at {location_json_path}")

    import trimesh
    import numpy as np
    all_mesh_paths: dict = {}   # obj_name → world-space mesh path

    app_cfg = AppearanceSDSConfig()
    if config_path and os.path.exists(config_path):
        raw = OmegaConf.load(config_path)
        if hasattr(raw, 'appearance'):
            for key, value in raw.appearance.items():
                if hasattr(app_cfg, key):
                    setattr(app_cfg, key, value)
                else:
                    print(f"[run_appearance_sds] WARNING: unknown appearance config key '{key}', ignored.")
        else:
            print("[run_appearance_sds] No 'appearance:' section found in config, using defaults.")
            
    for obj_id in obj_ids_num:
        
        voxel_path = os.path.join(voxel_dir, f"object_{obj_id:03d}_voxels.pt")
        if not os.path.exists(voxel_path):
            print(f"[run_scene_appearance_sds] WARNING: voxel file not found: {voxel_path}, skipping.")
            continue

        voxel_data = load_object_voxel(voxel_path, device=device)
        obj_id_str = str(obj_id)
        is_ground = ground_info.get(obj_id_str, 0) == 1
        obj_floor_min_z = floor_min_z if is_ground else None
        obj_bbox_modify = bbox_xyz_modify_dict.get(obj_id_str, None)
        if obj_bbox_modify is not None and all(v == 0.0 for v in obj_bbox_modify):
            obj_bbox_modify = None

        sparse_structure_obj = voxel_to_sparse_structure(
            voxel_data,
            device=device,
            model_path=model_path,
            floor_min_z=obj_floor_min_z,
            bbox_modify=obj_bbox_modify,
        )
        sparse_structure = sparse_structure_obj.get_occupancy_grid()  # (1, 1, 64, 64, 64)

        if cond_images is not None:
            _key = f"object_{obj_id:03d}"
            cond_image = cond_images.get(_key, os.path.join(cond_img_dir, f"{obj_id:03d}.png"))
        else:
            cond_image = os.path.join(cond_img_dir, f"{obj_id:03d}.png")
        
        output_dir_obj = os.path.join(output_dir, f"obj_{obj_id:03d}")
        os.makedirs(output_dir_obj, exist_ok=True)
        
        ss_path = os.path.join(geometry_sds_dir, f"object_{obj_id:03d}", "sds_completion_ss", "final_target.pt")
        
        print("Visualizing initial sparse structure...")
        initial_viz_dir = os.path.join(output_dir_obj, 'initial_ss_visualization')
        os.makedirs(initial_viz_dir, exist_ok=True)
        visualize_sparse_structure(
            data=sparse_structure,
            output_dir=initial_viz_dir,
            mode='all',
            threshold=0.2,
            value_colored=False,
            use_scatter=False,
            show=False,
            certainty=sparse_structure_obj.certainty_grid
        )
        print(f"Initial visualization saved to: {initial_viz_dir}")
    
        out = sparse_structure_to_svr_voxels(sparse_structure, sparse_structure_obj.transform_info, threshold=0.2, device=device)
        voxel_model = out['voxel_model']
        centers_init = out['centers']
        grid_coords_init = out['grid_coords']
        n_init = int(grid_coords_init.shape[0])

        # Voxel-count guard for appearance SDS. The sparse flow / SLatDecoder
        # via spconv internally allocates int32-indexed buffers, so very large
        # voxel counts trigger:
        #   "your data exceed int32 range" (cumm assert)
        # Empty SS also crashes downstream (e.g. .min() on a 0-length tensor).
        # We accept inputs whose voxel count lies in [MIN_VOXELS, MAX_VOXELS]
        # and otherwise fall back to the initial SS, finally skip.
        APP_SDS_MIN_VOXELS = 32
        APP_SDS_MAX_VOXELS = 80000

        if DEBUG:
            if n_init > 0:
                print(
                    f"[run_scene_appearance_sds] obj={obj_id:03d} STEP A: sparse_structure -> SVR voxels (threshold=0.2)\n"
                    f"  n_voxels={n_init}\n"
                    f"  grid_coords: x=[{grid_coords_init[:,0].min().item()},{grid_coords_init[:,0].max().item()}], "
                    f"y=[{grid_coords_init[:,1].min().item()},{grid_coords_init[:,1].max().item()}], "
                    f"z=[{grid_coords_init[:,2].min().item()},{grid_coords_init[:,2].max().item()}]\n"
                    f"  centers world: x=[{centers_init[:,0].min():.4f},{centers_init[:,0].max():.4f}], "
                    f"y=[{centers_init[:,1].min():.4f},{centers_init[:,1].max():.4f}], "
                    f"z=[{centers_init[:,2].min():.4f},{centers_init[:,2].max():.4f}]\n"
                    f"  transform_info: {sparse_structure_obj.transform_info}"
                )
            else:
                print(f"[run_scene_appearance_sds] obj={obj_id:03d} STEP A: initial SS produced 0 voxels at threshold=0.2")
            _debug_dir = os.path.join(output_dir_obj, 'debug_ss_validation')
            validate_ss_geometry(
                sparse_structure, sparse_structure_obj.transform_info,
                output_dir=_debug_dir, tag='initial', threshold=0.2, device=device,
            )

        # ---- Try to load SDS-completed SS ------------------------------
        sds_grid_coords = None
        sds_centers = None
        n_sds = -1
        if os.path.exists(ss_path):
            print(f"[run_appearance_sds] Loading completed sparse structure from: {ss_path}")
            ss_loaded = torch.load(ss_path, map_location=device)
            if DEBUG:
                _ss_raw = ss_loaded
                if _ss_raw.dim() == 5: _ss_raw = _ss_raw.squeeze(0).squeeze(0)
                elif _ss_raw.dim() == 4: _ss_raw = _ss_raw.squeeze(0)
                _occ = (_ss_raw > 0.5).sum().item()
                print(f"[run_scene_appearance_sds] obj={obj_id:03d} STEP B: loaded ss_path shape={ss_loaded.shape}, "
                      f"occupied (>0.5): {_occ}")
            if ss_loaded.dim() == 5:
                ss_loaded = ss_loaded.squeeze(0).squeeze(0)
            elif ss_loaded.dim() == 4:
                ss_loaded = ss_loaded.squeeze(0)
            out_ss = sparse_structure_to_svr_voxels(
                ss_loaded, sparse_structure_obj.transform_info, threshold=0.5, device=device
            )
            sds_grid_coords = out_ss['grid_coords']
            sds_centers = out_ss['centers']
            n_sds = int(sds_grid_coords.shape[0])
            print(f"[run_appearance_sds] {n_sds} voxels from ss_path (sds-completed)")
            if DEBUG and n_sds > 0:
                print(
                    f"[run_scene_appearance_sds] obj={obj_id:03d} STEP C: ss_path -> SVR voxels (threshold=0.5)\n"
                    f"  grid_coords: x=[{sds_grid_coords[:,0].min().item()},{sds_grid_coords[:,0].max().item()}], "
                    f"y=[{sds_grid_coords[:,1].min().item()},{sds_grid_coords[:,1].max().item()}], "
                    f"z=[{sds_grid_coords[:,2].min().item()},{sds_grid_coords[:,2].max().item()}]\n"
                    f"  centers world: x=[{sds_centers[:,0].min():.4f},{sds_centers[:,0].max():.4f}], "
                    f"y=[{sds_centers[:,1].min():.4f},{sds_centers[:,1].max():.4f}], "
                    f"z=[{sds_centers[:,2].min():.4f},{sds_centers[:,2].max():.4f}]"
                )
                validate_ss_geometry(
                    ss_loaded, sparse_structure_obj.transform_info,
                    output_dir=os.path.join(output_dir_obj, 'debug_ss_validation'),
                    tag='before_sds', threshold=0.5, device=device,
                )
        else:
            print(f"[run_scene_appearance_sds] WARNING: final_target not found: {ss_path} "
                  f"-- will try fallback to initial SS for {obj_id:03d}.")

        # ---- Decide which SS source to feed into appearance SDS --------
        def _in_range(n: int) -> bool:
            return APP_SDS_MIN_VOXELS <= n <= APP_SDS_MAX_VOXELS

        ss_source_tag = None
        if _in_range(n_sds):
            grid_coords = sds_grid_coords
            centers = sds_centers
            ss_source_tag = 'sds'
        elif _in_range(n_init):
            print(
                f"[run_scene_appearance_sds] obj={obj_id:03d}: SDS voxel count "
                f"n_sds={n_sds} out of range [{APP_SDS_MIN_VOXELS},{APP_SDS_MAX_VOXELS}]. "
                f"FALLING BACK to initial SS (n_init={n_init})."
            )
            grid_coords = grid_coords_init
            centers = centers_init
            ss_source_tag = 'initial_fallback'
        else:
            print(
                f"[run_scene_appearance_sds] obj={obj_id:03d}: BOTH SDS (n_sds={n_sds}) and initial "
                f"(n_init={n_init}) voxel counts are out of range "
                f"[{APP_SDS_MIN_VOXELS},{APP_SDS_MAX_VOXELS}]. Skipping object."
            )
            continue
        print(f"[run_scene_appearance_sds] obj={obj_id:03d}: using ss_source={ss_source_tag}, n_voxels={int(grid_coords.shape[0])}")
        
        latent = encode_object_slat(
            source_path=dataset_dir,
            obj_id=obj_id,
            voxel_coords=grid_coords,
            voxel_centers_world=centers,   # world-space positions required for GeoSVR camera projection
            output_path=os.path.join(output_dir_obj, f'obj{obj_id}_slat.npz'),
            use_non_zero_init=app_cfg.use_non_zero_init,
            non_zero_init_k=app_cfg.non_zero_init_k,
            device=device,
        )
        visualize_slat(
            latent,
            output_dir=os.path.join(output_dir_obj, 'slat_visualization'),
            prefix='before_sds',
        )

        # Pick a stable anchor voxel (smallest grid index in lex order) and
        # record its world center BEFORE slat_sds, so we can compare against
        # the same anchor AFTER slat_sds. slat_sds must NOT modify coords;
        # this print proves it.
        if DEBUG:
            _ti = sparse_structure_obj.transform_info
            _res = int(_ti.resolution)
            _cube_min = _ti.cube_min.cpu().numpy().astype(np.float64)
            _max_extent = float(_ti.max_extent)
            _coords_xyz = latent.coords[:, 1:].detach().cpu().long().numpy()
            _lex = _coords_xyz[:, 0] * _res * _res + _coords_xyz[:, 1] * _res + _coords_xyz[:, 2]
            _anchor_i = int(_lex.argmin())
            _anchor_grid_before = _coords_xyz[_anchor_i].copy()
            _anchor_world_before = (_anchor_grid_before + 0.5) / _res * _max_extent + _cube_min
            _feats_before_stats = (
                float(latent.feats.float().mean().item()),
                float(latent.feats.float().std().item()),
            )
            print(
                f"[DEBUG anchor] obj={obj_id:03d} BEFORE slat_sds:\n"
                f"  anchor_idx={_anchor_i}\n"
                f"  grid_coord = {_anchor_grid_before.tolist()}\n"
                f"  world_xyz  = ({_anchor_world_before[0]:.6f}, {_anchor_world_before[1]:.6f}, {_anchor_world_before[2]:.6f})\n"
                f"  feats stats: mean={_feats_before_stats[0]:.4f}, std={_feats_before_stats[1]:.4f}"
            )

        # ------------------------------------------------------------------
        # Log appearance SDS config before optimisation
        # ------------------------------------------------------------------
        import dataclasses
        from datetime import datetime
        log_info = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': dataclasses.asdict(app_cfg),
            'input_info': {
                'cond_image': cond_image,
                'output_dir': output_dir_obj,
                'obj_id': obj_id,
                'config_path': config_path,
                'ss_path': ss_path,
                'ss_source': ss_source_tag,
                'n_voxels': int(grid_coords.shape[0]),
                'n_voxels_sds': int(n_sds),
                'n_voxels_initial': int(n_init),
            },
            'system_info': {
                'device': str(device),
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_name': _get_cuda_device_name(device),
            },
        }
        log_path = os.path.join(output_dir_obj, 'appearance_sds_config_log.json')
        with open(log_path, 'w') as _f:
            import json as _json
            _json.dump(log_info, _f, indent=2)
        print(f"[run_appearance_sds] Config logged → {log_path}")
        print(f"[run_appearance_sds] appearance cfg: {dataclasses.asdict(app_cfg)}")

        latent = slat_sds(
            latent,
            cond_image=cond_image,
            output_dir=output_dir_obj,
            cfg=app_cfg,
            certainty_grid=sparse_structure_obj.certainty_grid,
            device=device,
        )
        visualize_slat(
            latent,
            output_dir=os.path.join(output_dir_obj, 'slat_visualization'),
            prefix='after_sds',
        )

        # Anchor xyz AFTER slat_sds + reconstruct an SS from latent.coords
        # and validate. This proves whether slat_sds touched coords or not
        # and re-checks that the post-SDS SS region in world coords matches
        # the pre-SDS one.
        if DEBUG:
            _coords_xyz_after = latent.coords[:, 1:].detach().cpu().long().numpy()
            _coords_match = (
                _coords_xyz_after.shape == _coords_xyz.shape
                and bool(np.array_equal(_coords_xyz_after, _coords_xyz))
            )
            _anchor_grid_after = _coords_xyz_after[_anchor_i] if _coords_match else _coords_xyz_after[0]
            _anchor_world_after = (_anchor_grid_after + 0.5) / _res * _max_extent + _cube_min
            _feats_after_stats = (
                float(latent.feats.float().mean().item()),
                float(latent.feats.float().std().item()),
            )
            print(
                f"[DEBUG anchor] obj={obj_id:03d} AFTER  slat_sds:\n"
                f"  coords_unchanged = {_coords_match}\n"
                f"  grid_coord = {_anchor_grid_after.tolist()}\n"
                f"  world_xyz  = ({_anchor_world_after[0]:.6f}, {_anchor_world_after[1]:.6f}, {_anchor_world_after[2]:.6f})\n"
                f"  feats stats: mean={_feats_after_stats[0]:.4f}, std={_feats_after_stats[1]:.4f}"
            )
            # Rebuild a (R,R,R) occupancy from latent.coords for validation.
            _ss_after = torch.zeros(_res, _res, _res, device=device)
            _cc = torch.from_numpy(_coords_xyz_after).long().to(device)
            _ss_after[_cc[:, 0], _cc[:, 1], _cc[:, 2]] = 1.0
            validate_ss_geometry(
                _ss_after, sparse_structure_obj.transform_info,
                output_dir=os.path.join(output_dir_obj, 'debug_ss_validation'),
                tag='after_sds', threshold=0.5, device=device,
            )

        glb_path = os.path.join(output_dir_obj, f'obj{obj_id}_mesh.glb')
        decode_slat_to_mesh(
            latent,
            output_path=glb_path,
            device=device,
        )

        # 6. Inverse-transform mesh back to scene (world) coordinates
        obj_name = f'obj_{obj_id:03d}'
        transform_info = sparse_structure_obj.transform_info
        if os.path.exists(glb_path):
            print(f"[run_scene_appearance_sds] Inverse-transforming {obj_name} mesh to world coords ...")
            world_mesh = load_and_inverse_transform_glb(glb_path, transform_info)
            world_mesh_path = os.path.join(output_dir_obj, f'{obj_name}_world.glb')
            world_mesh.export(world_mesh_path)
            all_mesh_paths[obj_name] = world_mesh_path
            print(f"[run_scene_appearance_sds] World mesh saved: {world_mesh_path}")
            if DEBUG:
                _wv = np.asarray(world_mesh.vertices, dtype=np.float64)
                _ssvr_min = centers.detach().cpu().numpy().min(axis=0)
                _ssvr_max = centers.detach().cpu().numpy().max(axis=0)
                print(
                    f"[DEBUG mesh-vs-ss] obj={obj_id:03d}\n"
                    f"  decoded mesh world bbox: "
                    f"x=[{_wv[:,0].min():.4f},{_wv[:,0].max():.4f}], "
                    f"y=[{_wv[:,1].min():.4f},{_wv[:,1].max():.4f}], "
                    f"z=[{_wv[:,2].min():.4f},{_wv[:,2].max():.4f}], "
                    f"centroid=({_wv[:,0].mean():.4f},{_wv[:,1].mean():.4f},{_wv[:,2].mean():.4f})\n"
                    f"  ss-after  centers bbox:  "
                    f"x=[{_ssvr_min[0]:.4f},{_ssvr_max[0]:.4f}], "
                    f"y=[{_ssvr_min[1]:.4f},{_ssvr_max[1]:.4f}], "
                    f"z=[{_ssvr_min[2]:.4f},{_ssvr_max[2]:.4f}]\n"
                    f"  delta_min=({_wv[:,0].min()-_ssvr_min[0]:+.4f},{_wv[:,1].min()-_ssvr_min[1]:+.4f},{_wv[:,2].min()-_ssvr_min[2]:+.4f})  "
                    f"delta_max=({_wv[:,0].max()-_ssvr_max[0]:+.4f},{_wv[:,1].max()-_ssvr_max[1]:+.4f},{_wv[:,2].max()-_ssvr_max[2]:+.4f})"
                )
        else:
            print(f"[run_scene_appearance_sds] WARNING: GLB not found for {obj_name}, skipping inverse transform.")

    # 8. Merge all world-space meshes into one scene GLB by scanning the output
    # directory — checkpoint-safe, picks up any previously completed objects.
    print("\n[Scene Appearance SDS] Scanning output dir for world-space GLBs to merge ...")
    world_glb_paths = sorted(
        glob.glob(os.path.join(output_dir, "*", "*_world.glb"))
    )
    if world_glb_paths:
        combined_scene = trimesh.Scene()
        for glb_file in world_glb_paths:
            obj_name = os.path.basename(os.path.dirname(glb_file))  # e.g. "obj_007"
            print(f"[Scene Appearance SDS]   adding {obj_name} from {glb_file}")
            loaded = trimesh.load(glb_file, force='scene')
            if isinstance(loaded, trimesh.Scene):
                for geom_name, geom in loaded.geometry.items():
                    try:
                        node_tf, _ = loaded.graph.get(geom_name)
                    except Exception:
                        node_tf = np.eye(4)
                    combined_scene.add_geometry(
                        geom,
                        node_name=f"{obj_name}_{geom_name}",
                        transform=node_tf,
                    )
            else:
                combined_scene.add_geometry(loaded, node_name=obj_name)
        combined_path = os.path.join(output_dir, 'scene_combined.glb')
        combined_scene.export(combined_path)
        print(f"[Scene Appearance SDS] Combined scene mesh ({len(world_glb_paths)} objects): {combined_path}")
    else:
        print("[Scene Appearance SDS] No world-space GLBs found; skipping scene merge.")

    print("\n" + "=" * 60)
    print("Scene Appearance SDS Pipeline Complete")
    print("=" * 60)
    for obj_name, mp in all_mesh_paths.items():
        print(f"  {obj_name}: {mp}")

    return {
        'mesh_paths': all_mesh_paths,
    }


def run_scene_appearance_sds_para(
    dataset_dir: str,
    model_path: str,
    config_path: str,
    device: str = "cuda",
    cond_images: dict = None,
    parallel_num: int = 4,
):
    """Parallel variant of ``run_scene_appearance_sds``.

    The per-object SLAT encode → slat_sds → decode loop is dispatched to a
    thread pool with ``max_workers=parallel_num``. Shared preprocessing
    (floor / bbox / ground-info loading, appearance config parsing) runs
    once before the pool. Aggregation (merged GLB) runs once after.
    """
    device = _resolve_device(device)
    from decomvoxel.utils.vis_slat import visualize_slat
    import dataclasses
    from datetime import datetime
    import threading
    import json as _json
    import trimesh

    cond_img_dir = _resolve_cond_image_dir(model_path)

    if cond_images is not None:
        print(f"[Scene Appearance SDS Para] Using caller-supplied cond_images ({len(cond_images)} objects)")
        obj_ids_num = sorted([int(k.split("_")[-1]) for k in cond_images.keys()])
    else:
        cond_img_names = sorted([f for f in os.listdir(cond_img_dir) if f.endswith(".png")])
        obj_ids = [f.split(".")[0] for f in cond_img_names]
        obj_ids_num = [int(oid) for oid in obj_ids]

    print(f"[Scene Appearance SDS Para] parallel_num_per_gpu={parallel_num}, visible devices={get_visible_devices()}")

    output_dir = os.path.join(model_path, "appearance_sds")
    os.makedirs(output_dir, exist_ok=True)

    geometry_sds_dir = os.path.join(model_path, "scene_sds")
    voxel_dir = os.path.join(model_path, "semantic_result", "object_voxels")

    from decomvoxel.utils.bbox_support import get_min_z
    floor_min_z = get_min_z(voxel_dir)
    if floor_min_z is not None:
        print(f"[Scene Appearance SDS Para] Global floor min_z = {floor_min_z:.4f}")
    else:
        print("[Scene Appearance SDS Para] Could not determine floor min_z; floor clamping skipped.")

    bbox_xyz_modify_dict = {}
    if dataset_dir is not None:
        categories_json = os.path.join(dataset_dir, "object_categories.json")
        if not os.path.exists(categories_json):
            _alt = os.path.join(dataset_dir, "obj_prompt.json")
            if os.path.exists(_alt):
                print(f"[Scene Appearance SDS Para] object_categories.json not found; using obj_prompt.json instead.")
                categories_json = _alt
        if os.path.exists(categories_json):
            from decomvoxel.utils.bbox_support import chair_bbox_xy_modify
            _debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "debug")
            bbox_xyz_modify_dict = chair_bbox_xy_modify(
                voxel_dir=voxel_dir,
                categories_json=categories_json,
                vis_output_dir=_debug_dir,
            )

    ground_info = {}
    if dataset_dir is not None:
        location_json_path = os.path.join(dataset_dir, "location_modify.json")
        if os.path.exists(location_json_path):
            with open(location_json_path) as f:
                ground_info = {str(k): int(v) for k, v in json.load(f).items()}

    app_cfg = AppearanceSDSConfig()
    if config_path and os.path.exists(config_path):
        raw = OmegaConf.load(config_path)
        if hasattr(raw, 'appearance'):
            for key, value in raw.appearance.items():
                if hasattr(app_cfg, key):
                    setattr(app_cfg, key, value)
                else:
                    print(f"[run_appearance_sds_para] WARNING: unknown appearance config key '{key}', ignored.")

    APP_SDS_MIN_VOXELS = 32
    APP_SDS_MAX_VOXELS = 80000

    # ── Subprocess-based parallelism: one independent process per object ──────
    devices = get_visible_devices()
    tasks_sp = []
    for i, obj_id in enumerate(obj_ids_num):
        obj_id_str      = str(obj_id)
        is_ground       = ground_info.get(obj_id_str, 0) == 1
        obj_floor_min_z_i = floor_min_z if is_ground else None
        obj_bbox_modify_i = bbox_xyz_modify_dict.get(obj_id_str, None)
        if obj_bbox_modify_i is not None and all(v == 0.0 for v in obj_bbox_modify_i):
            obj_bbox_modify_i = None

        if cond_images is not None:
            _key = f"object_{obj_id:03d}"
            cond_image_i = cond_images.get(_key, os.path.join(cond_img_dir, f"{obj_id:03d}.png"))
        else:
            cond_image_i = os.path.join(cond_img_dir, f"{obj_id:03d}.png")

        result_path = os.path.join(output_dir, f'._app_result_{i}_obj{obj_id:03d}.json')
        config_dict = {
            'obj_id':           obj_id,
            'device':           'placeholder',  # filled in by _run_objects_subprocess
            'model_path':       model_path,
            'dataset_dir':      dataset_dir,
            'output_dir':       output_dir,
            'geometry_sds_dir': geometry_sds_dir,
            'voxel_dir':        voxel_dir,
            'cond_image':       cond_image_i,
            'floor_min_z':      floor_min_z,
            'obj_floor_min_z':  obj_floor_min_z_i,
            'obj_bbox_modify':  obj_bbox_modify_i,
            'config_path':      config_path,
            'app_cfg':          dataclasses.asdict(app_cfg),
            'APP_SDS_MIN_VOXELS': APP_SDS_MIN_VOXELS,
            'APP_SDS_MAX_VOXELS': APP_SDS_MAX_VOXELS,
            '_result_path':     result_path,
        }
        tasks_sp.append((f"obj_{obj_id:03d}", config_dict))

    sp_results = _run_objects_subprocess(
        tasks=tasks_sp,
        mode='_worker_app_sds',
        devices=devices,
        parallel_num_per_gpu=parallel_num,
        desc='appearance_sds',
    )

    all_mesh_paths: dict = {}
    failed_app = []
    for obj_key, result, exc in sp_results:
        if exc is not None or result is None:
            failed_app.append(obj_key)
            continue
        if result.get('world_mesh_path'):
            all_mesh_paths[result['obj_name']] = result['world_mesh_path']
    if failed_app:
        print("[Scene Appearance SDS Para] WARNING: some objects failed: " + ", ".join(sorted(failed_app)))

    print("\n[Scene Appearance SDS Para] Scanning output dir for world-space GLBs to merge ...")
    world_glb_paths = sorted(
        glob.glob(os.path.join(output_dir, "*", "*_world.glb"))
    )
    if world_glb_paths:
        combined_scene = trimesh.Scene()
        for glb_file in world_glb_paths:
            obj_name = os.path.basename(os.path.dirname(glb_file))
            print(f"[Scene Appearance SDS Para]   adding {obj_name} from {glb_file}")
            loaded = trimesh.load(glb_file, force='scene')
            if isinstance(loaded, trimesh.Scene):
                for geom_name, geom in loaded.geometry.items():
                    try:
                        node_tf, _ = loaded.graph.get(geom_name)
                    except Exception:
                        node_tf = np.eye(4)
                    combined_scene.add_geometry(
                        geom,
                        node_name=f"{obj_name}_{geom_name}",
                        transform=node_tf,
                    )
            else:
                combined_scene.add_geometry(loaded, node_name=obj_name)
        combined_path = os.path.join(output_dir, 'scene_combined.glb')
        combined_scene.export(combined_path)
        print(f"[Scene Appearance SDS Para] Combined scene mesh ({len(world_glb_paths)} objects): {combined_path}")
    else:
        print("[Scene Appearance SDS Para] No world-space GLBs found; skipping scene merge.")

    print("\n" + "=" * 60)
    print("Scene Appearance SDS Pipeline (parallel) Complete")
    print("=" * 60)
    for obj_name, mp in all_mesh_paths.items():
        print(f"  {obj_name}: {mp}")

    return {
        'mesh_paths': all_mesh_paths,
    }


def run_full_pipeline(
    model_dir: str,
    dataset_dir: str,
    cfg_files: list = None,
    cond_text: str = None,
    img_model: str = "Seedream",
    img_model_params: dict = None,
    config_path: str = None,
    sds_iters: int = 5000,
    bg_id: int = 255,
    num_views: int = 30,
    slat_steps: int = 12,
    slat_cfg_strength: float = 3.0,
    simplify_ratio: float = 0.95,
    texture_size: int = 1024,
    parallel_num: int = 4,
    cli_args: dict = None,
):
    """Full pipeline: GeoSVR -> Segm3D -> GenCondImg -> SceneSDS -> SceneAppearanceSDS"""
    
    start_time = time.time()
    logger = PipelineLogger(model_dir, cli_args=cli_args)

    # Step 1: Train GeoSVR
    print("\n" + "=" * 60)
    print("[Full Pipeline] Step 1/5: Train GeoSVR")
    print("=" * 60)
    logger.start_step("Step 1/5: Train GeoSVR")
    run_train_geosvr(
        cfg_files=cfg_files or [],
        source_path=dataset_dir,
        model_path=model_dir,
    )
    end_train_geosvr = time.time()
    print(f"[Full Pipeline] Step 1 complete! Time taken: {end_train_geosvr - start_time:.2f} seconds")
    logger.end_step()


    # Step 2: 3D Segmentation
    print("\n" + "=" * 60)
    print("[Full Pipeline] Step 2/5: 3D Segmentation")
    print("=" * 60)
    logger.start_step("Step 2/5: 3D Segmentation")
    run_segm_3d(
        source_path=dataset_dir,
        model_path=model_dir,
        bg_id=bg_id,
        num_views=num_views,
    )
    end_segm_3d = time.time()
    print(f"[Full Pipeline] Step 2 complete! Time taken: {end_segm_3d - end_train_geosvr:.2f} seconds")
    logger.end_step()


    # Step 3: Generate conditioning images
    print("\n" + "=" * 60)
    print("[Full Pipeline] Step 3/5: Generate Conditioning Images")
    print("=" * 60)
    logger.start_step("Step 3/5: Generate Conditioning Images")
    cond_output_root = os.path.join(model_dir, "cond_images")
    resolved_img_model = _resolve_img_model(img_model=img_model, config_path=config_path)
    resolved_img_model_params = _resolve_img_model_params(img_model_params=img_model_params, config_path=config_path)
    print(f"[Full Pipeline] Conditioning image model: {resolved_img_model}")
    if resolved_img_model_params:
        print(f"[Full Pipeline] Conditioning image model params: {resolved_img_model_params}")
    
    object_category_path = os.path.join(dataset_dir, "object_categories.json")
    if os.path.exists(object_category_path):
        with open(object_category_path, "r") as f:
            object_categories = json.load(f)
    else:
        object_categories = None
        
    # selected objects
    # _selected_ids = {"35", "36"}
    # object_categories = {str(k): v for k, v in object_categories.items() if str(k) in _selected_ids} if object_categories else None
        
    # generated = generate_cond_img_avo(model_dir, cond_output_root, text=None, object_categories=object_categories)
    generated = generate_cond_img_avo_para(
        model_dir, cond_output_root,
        text=None,
        object_categories=object_categories,
        parallel_num=parallel_num,
        img_model=resolved_img_model,
        img_model_params=resolved_img_model_params,
    )
    cond_images = {f"object_{oid}": path for oid, path in generated.items()}
    
    # cond_img_dir = _resolve_cond_image_dir(model_dir)
    # cond_img_names = os.listdir(cond_img_dir)
    # cond_img_names = sorted(cond_img_names, key=lambda name: int(os.path.splitext(name)[0]))  # Sort by object ID
    # cond_images = {f"object_{name.split('.')[0]}": os.path.join(cond_img_dir, name) for name in cond_img_names}
    # print(f"Conditioning images for SDS: {cond_images}")
    
    # Ignore object_000 (background / obj_id=0) if present
    if "object_000" in cond_images:
        print("[Full Pipeline] Ignoring object_000 (obj_id=0).")
        cond_images = {k: v for k, v in cond_images.items() if k != "object_000"}
    
    end_cond_img_generate = time.time()
    print(f"[Full Pipeline] Step 3 complete! Time taken: {end_cond_img_generate - end_segm_3d:.2f} seconds")
    logger.end_step()


    # Step 4: Scene-level Geometry SDS + Mesh
    print("\n" + "=" * 60)
    print("[Full Pipeline] Step 4/5: Scene Geometry SDS")
    print("=" * 60)
    logger.start_step("Step 4/5: Scene Geometry SDS")
    voxel_dir = os.path.join(model_dir, "semantic_result", "object_voxels")
    sds_output_dir = os.path.join(model_dir, "scene_sds")
    run_scene_sds_para(
        voxel_dir=voxel_dir,
        output_dir=sds_output_dir,
        dataset_dir=dataset_dir,
        cond_images=cond_images,
        config_path=config_path,
        total_iters=sds_iters,
        slat_steps=slat_steps,
        slat_cfg_strength=slat_cfg_strength,
        simplify_ratio=simplify_ratio,
        texture_size=texture_size,
        parallel_num=parallel_num,
    )
    end_geo_sds = time.time()
    print(f"[Full Pipeline] Step 4 complete! Time taken: {end_geo_sds - end_cond_img_generate:.2f} seconds")
    logger.end_step()
    

    # Step 5: Scene-level Appearance SDS (per-object SLAT optimisation)
    print("\n" + "=" * 60)
    print("[Full Pipeline] Step 5/5: Scene Appearance SDS")
    print("=" * 60)
    logger.start_step("Step 5/5: Scene Appearance SDS")
    run_scene_appearance_sds_para(
        dataset_dir=dataset_dir,
        model_path=model_dir,
        config_path=config_path,
        device=str(device),
        cond_images=cond_images,
        parallel_num=parallel_num,
    )
    end_app_sds = time.time()
    print(f"[Full Pipeline] Step 5 complete! Time taken: {end_app_sds - end_geo_sds:.2f} seconds")
    logger.end_step()


    print("\n" + "=" * 60)
    print("[Full Pipeline] Done!")
    print("=" * 60)
    logger.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DecomVoxel Demo: Full Pipeline (GeoSVR → CondImg → SceneSDS → AppearanceSDS)"
    )

    parser.add_argument('--mode', type=str, default='full_pipeline',
                        choices=['full_pipeline', '_worker_sds', '_worker_app_sds'],
                        help='Pipeline mode (_worker_* modes are internal subprocess entry points)')
    parser.add_argument('--worker_json', type=str, default=None,
                        help='(internal) path to JSON config for subprocess worker modes')
    parser.add_argument('--source_path', type=str, default=None,
                        help='Path to dataset directory')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model/output directory')
    parser.add_argument('--cfg_files', nargs='*', default=[],
                        help='GeoSVR config files')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to SDS/appearance configuration YAML')
    parser.add_argument('--cond_text', type=str, default=None,
                        help='Text prompt for conditioning image generation')
    parser.add_argument('--img_model', type=str, default='Seedream',
                        help='Conditioning image model: Seedream, NanoBanana, or GPTImage2')
    parser.add_argument('--img_model_params_json', type=str, default=None,
                        help='Optional JSON object for cond image model specific parameters')
    parser.add_argument('--bg_id', type=int, default=255,
                        help='Background label ID for 3D segmentation')
    parser.add_argument('--num_views', type=int, default=30,
                        help='Number of camera views')
    parser.add_argument('--sds_iters', type=int, default=5000,
                        help='Number of SDS optimization iterations')
    parser.add_argument('--slat_steps', type=int, default=12,
                        help='Number of structured latent sampler steps')
    parser.add_argument('--slat_cfg_strength', type=float, default=3.0,
                        help='CFG strength for structured latent sampling')
    parser.add_argument('--simplify_ratio', type=float, default=0.95,
                        help='Mesh simplification ratio')
    parser.add_argument('--texture_size', type=int, default=1024,
                        help='Baked texture resolution')
    parser.add_argument('--parallel_num', type=int, default=4,
                        help='Max number of concurrent workers for parallel pipeline steps')

    args = parser.parse_args()

    if args.mode == '_worker_sds':
        _worker_sds_entry(args.worker_json)
    elif args.mode == '_worker_app_sds':
        _worker_app_sds_entry(args.worker_json)
    elif args.mode == 'full_pipeline':
        img_model_params = None
        if args.img_model_params_json:
            img_model_params = json.loads(args.img_model_params_json)
            if not isinstance(img_model_params, dict):
                raise ValueError("--img_model_params_json must decode to a JSON object")
        run_full_pipeline(
            model_dir=args.model_path,
            dataset_dir=args.source_path,
            cfg_files=args.cfg_files,
            cond_text=args.cond_text,
            img_model=args.img_model,
            img_model_params=img_model_params,
            config_path=args.config,
            sds_iters=args.sds_iters,
            bg_id=args.bg_id,
            num_views=args.num_views,
            slat_steps=args.slat_steps,
            slat_cfg_strength=args.slat_cfg_strength,
            simplify_ratio=args.simplify_ratio,
            texture_size=args.texture_size,
            parallel_num=args.parallel_num,
            cli_args=vars(args),
        )
