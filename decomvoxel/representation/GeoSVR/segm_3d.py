"""
Replica Instance Segmentation Fusion with 3D Voxel Export

This script extends replica_segm.py with additional capabilities:
1. Fuse 2D instance segmentation masks into 3D sparse voxels
2. Render each segmented object separately
3. Export individual voxel data for each object
4. Export colored mesh for each object using trimesh

Note: This script write voxel_to_mesh seprately from voxel_to_mesh.py

TODO: Some functionality here can be moved into GeoSVR classes (e.g., add mesh export in io).

"""

import os
import sys
import glob
import cv2
import numpy as np
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from yacs.config import CfgNode
from src.config import cfg
from src.dataloader.data_pack import DataPack
from src.sparse_voxel_model import SparseVoxelModel
from src.utils.fuser_utils import Fuser
from src.utils import activation_utils
from src.utils.octree_utils import level_2_vox_size
from src.sparse_voxel_gears.io import SVInOut

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not installed. Mesh export will be skipped.")


def create_unit_cube():
    vertices = np.array([
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
    ], dtype=np.float32)
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ], dtype=np.int32)
    return vertices, faces

def voxels_to_mesh(centers, sizes, colors):
    if torch.is_tensor(centers): centers = centers.cpu().numpy()
    if torch.is_tensor(sizes): sizes = sizes.cpu().numpy()
    if torch.is_tensor(colors): colors = colors.cpu().numpy()
    if sizes.ndim > 1: sizes = sizes.squeeze()
    colors = np.clip(colors, 0, 1)
    
    unit_verts, unit_faces = create_unit_cube()
    n_voxels = len(centers)
    
    all_vertices = (sizes[:, None, None] * unit_verts[None, :, :] + centers[:, None, :]).reshape(-1, 3).astype(np.float32)
    all_faces = (unit_faces[None, :, :] + (np.arange(n_voxels)[:, None, None] * 8)).reshape(-1, 3).astype(np.int32)
    
    colors_rgba = np.zeros((n_voxels, 8, 4), dtype=np.uint8)
    colors_rgba[:, :, :3] = (colors[:, None, :] * 255).astype(np.uint8)
    colors_rgba[:, :, 3] = 255
    all_colors = colors_rgba.reshape(-1, 4)
    
    return trimesh.Trimesh(vertices=all_vertices, faces=all_faces, vertex_colors=all_colors, process=False)


def render_select_object(voxel_model, cameras, object_mask, color, output_dir, object_id, num_views=5):
    os.makedirs(output_dir, exist_ok=True)
    
    # Store original parameters
    ori_sh0 = voxel_model.sh0.data.clone()
    ori_shs = voxel_model.shs.data.clone()
    
    # Keep original colors for object voxels, set others to white (background)
    # This way only the target object is visible with its original appearance
    bg_color = torch.tensor([1.0, 1.0, 1.0], device='cuda')  # White background
    # bg_color = torch.tensor([0.0, 0.0, 0.0], device='cuda')  # Black background
    bg_sh0 = activation_utils.rgb2shzero(bg_color)  # (3,)
    
    # Set non-object voxels to white background color
    new_sh0 = voxel_model.sh0.data.clone()
    new_shs = voxel_model.shs.data.clone()
    new_sh0[~object_mask] = bg_sh0.view(1, 3) # sh0 shape is (N, 1, 3)
    new_shs[~object_mask] = 0.0
    
    voxel_model.sh0.data = new_sh0
    voxel_model.shs.data = new_shs
    
    # Select views to render (evenly spaced)
    step = max(1, len(cameras) // num_views)
    selected_cams = cameras[::step][:num_views]
    
    with torch.no_grad():
        for i, cam in enumerate(selected_cams):
            res = voxel_model.render(cam, color_mode=None)
            img_tensor = res['color']
            
            # Convert to numpy image
            img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            save_path = os.path.join(output_dir, f"object_{object_id:03d}_view_{i:02d}.png")
            cv2.imwrite(save_path, img_bgr)
    
    # Restore original parameters
    voxel_model.sh0.data = ori_sh0
    voxel_model.shs.data = ori_shs


# Must align with voxel_model save function!
def export_object_voxels(voxel_model, object_mask, object_id, output_dir, original_id, recenter=True):
    os.makedirs(output_dir, exist_ok=True)
    
    # Get indices of voxels belonging to this object
    indices = torch.where(object_mask)[0]
    
    if len(indices) == 0:
        return None
    
    # Use the new utility to create a proper state_dict
    state_dict = SVInOut.create_state_dict_from_subset(
        voxel_model=voxel_model,
        voxel_mask=object_mask,
        recenter=recenter,
        outside_level=voxel_model.outside_level,
    )
    
    if state_dict is None:
        return None
    
    # Save in the standard format (compatible with SparseVoxelModel.load)
    save_path = os.path.join(output_dir, f"object_{original_id:03d}_voxels.pt")
    SVInOut.save_state_dict(state_dict, save_path, quantize=False)
    
    # Also save the original-coordinate version for mesh export and visualization
    # (without recentering, keeps original scene coordinates)
    obj_vox_center = voxel_model.vox_center[indices].cpu()
    obj_vox_size = voxel_model.vox_size[indices].cpu()
    
    bbox_min = obj_vox_center.min(dim=0).values
    bbox_max = obj_vox_center.max(dim=0).values
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_extent = bbox_max - bbox_min
    
    # Return metadata for summary and mesh export
    obj_data = {
        'original_id': original_id,
        'class_index': object_id,
        'num_voxels': len(indices),
        'voxel_indices': indices.cpu(),
        'vox_center': obj_vox_center,
        'vox_size': obj_vox_size,
        'sh0': voxel_model._sh0.data[indices].cpu(),
        'shs': voxel_model._shs.data[indices].cpu(),
        'bbox_min': bbox_min,
        'bbox_max': bbox_max,
        'bbox_center': bbox_center,
        'bbox_extent': bbox_extent,
        'scene_center': voxel_model.scene_center.cpu(),
        'scene_extent': voxel_model.scene_extent.cpu(),
        'inside_extent': voxel_model.inside_extent.cpu(),
        # Path to the loadable model file
        'model_path': save_path,
    }
    
    return obj_data


def create_cfg_data(source_path):
    """Create cfg_data config node for DataPack."""
    cfg_data = CfgNode()
    cfg_data.source_path = source_path
    cfg_data.images = "images"
    cfg_data.res_downscale = 0.
    cfg_data.res_width = 0
    cfg_data.extension = ".png"
    cfg_data.blend_mask = True
    cfg_data.depth_paths = ""
    cfg_data.depth_scale = 1.0
    cfg_data.data_device = "cpu"
    cfg_data.eval = False
    cfg_data.test_every = 8
    cfg_data.n_sparse = -1
    cfg_data.ncc_scale = 1.0
    return cfg_data


def create_cfg_model(model_path):
    """Create cfg_model config node for SparseVoxelModel."""
    cfg_model = CfgNode()
    cfg_model.model_path = model_path
    cfg_model.vox_geo_mode = "triinterp1"
    cfg_model.density_mode = "exp_linear_11"
    cfg_model.sh_degree = 3
    cfg_model.ss = 1.5
    cfg_model.outside_level = 5
    cfg_model.white_background = False
    cfg_model.black_background = False
    return cfg_model


def segm_3d(source_path, model_path, mask_dir=None, iteration=-1, bg_id=255, num_views=30):
    scene_name = os.path.basename(source_path)
    dataset_root = source_path
    output_root = model_path
    if mask_dir is None:
        # Try both common naming conventions
        for _candidate in ['instance_masks', 'instance_mask']:
            _candidate_path = os.path.join(dataset_root, _candidate)
            if os.path.isdir(_candidate_path):
                mask_dir = _candidate_path
                break
        if mask_dir is None:
            mask_dir = os.path.join(dataset_root, 'instance_masks')  # fallback (will warn later)
    target_iteration = iteration
    BG_ID = bg_id
    num_render_views = num_views

    print(f"Processing scene: {scene_name}")
    print(f"Dataset root: {dataset_root}")
    print(f"Output root: {output_root}")
    print(f"Mask dir: {mask_dir}")

    # --- Load DataPack ---
    cfg_data = create_cfg_data(dataset_root)
    data_pack = DataPack(cfg_data)
    cameras = data_pack.get_train_cameras()
    print(f"Loaded {len(cameras)} cameras.")

    # --- Load SVR Model ---
    cfg_model = create_cfg_model(output_root)
    voxel_model = SparseVoxelModel(cfg_model)
    
    voxel_model.load_iteration(target_iteration)
    voxel_model.freeze_vox_geo()
    print("Model loaded.")

    # --- Pre-scan Masks to find unique Object IDs ---
    print("Scanning masks to identify object IDs...")
    mask_files = sorted(glob.glob(os.path.join(mask_dir, '*')))
    unique_ids = set()
    
    for mf in tqdm(mask_files, desc="Scanning IDs"):
        mask = cv2.imread(mf, cv2.IMREAD_UNCHANGED)
        if mask is None: continue
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
        
        u = np.unique(mask)
        unique_ids.update(u.tolist())
    
    all_ids = sorted(list(unique_ids))
    print(f"Found {len(all_ids)} unique IDs: {all_ids}")
    
    # Define ID Mapping
    # Background (BG_ID) -> Class 0
    # Objects -> Class 1, 2, ...
    object_ids = [uid for uid in all_ids if uid != BG_ID]
    id_to_class = {BG_ID: 0}
    for i, oid in enumerate(object_ids):
        id_to_class[oid] = i + 1
    
    num_classes = len(object_ids) + 1
    print(f"Total classes: {num_classes} (1 Background + {len(object_ids)} Objects)")

    # Prepare Lookup Table for fast mapping
    max_id_val = max(unique_ids) if unique_ids else 0
    lut = torch.zeros(max_id_val + 1, dtype=torch.int64, device='cuda') # lookup table
    for uid, cidx in id_to_class.items():
        if uid <= max_id_val:
            lut[uid] = cidx

    # --- Initialize Fuser ---
    finest_vox_size = level_2_vox_size(voxel_model.scene_extent, voxel_model.octlevel.max()).item() # 'level to vox size', get the smallest voxel size
    
    fuser = Fuser(
        xyz=voxel_model.vox_center,
        bandwidth=50 * finest_vox_size,
        use_trunc=False,
        fuse_tsdf=False,
        feat_dim=num_classes,
        crop_border=0.0,
        normal_weight=False,
        depth_weight=False,
        border_weight=False,
        use_half=True
    )
    
    # --- Fusion Loop ---
    print("Fusing masks into 3D...")
    
    with torch.no_grad():
        for cam in tqdm(cameras, desc="Fusion"):
            img_name = cam.image_name
            basename = os.path.splitext(img_name)[0]
            
            # Handle naming convention: image "000000_rgb.png" -> mask "000000.png"
            for suffix in ['_rgb', '_color', '_image']:
                if basename.endswith(suffix):
                    basename = basename[:-len(suffix)]
                    break
            
            # Search for matching mask file
            mask_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.bmp']:
                p = os.path.join(mask_dir, basename + ext)
                if os.path.exists(p):
                    mask_path = p
                    break
            
            if mask_path is None:
                continue

            # Load and preprocess mask
            mask_cv = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask_cv is None: continue
            if len(mask_cv.shape) > 2: mask_cv = mask_cv[:,:,0]
            
            if mask_cv.shape[0] != cam.image_height or mask_cv.shape[1] != cam.image_width:
                mask_cv = cv2.resize(mask_cv, (cam.image_width, cam.image_height), interpolation=cv2.INTER_NEAREST)

            mask_tensor = torch.from_numpy(mask_cv.astype(np.int64)).cuda()
            class_mask = lut[mask_tensor]
            
            # One-Hot Encoding
            probs = torch.nn.functional.one_hot(class_mask, num_classes=num_classes)
            probs = probs.permute(2, 0, 1).float()
            
            # Render depth for geometric consistency
            render_pkg = voxel_model.render(cam, output_depth=True)
            depth = render_pkg['depth'][2]
            
            fuser.integrate(cam=cam, feat=probs, depth=depth)

    # --- Extract Results ---
    print("Fusion complete. Computing voxel labels...")
    
    feature_vol = fuser.feature.nan_to_num_(0)
    voxel_labels = feature_vol.argmax(dim=1)
    
    # --- Setup Output Directories ---
    result_dir = os.path.join(output_root, 'semantic_result')
    viz_dir = os.path.join(result_dir, 'viz_images')
    object_render_dir = os.path.join(result_dir, 'object_renders')
    object_env_render_dir = os.path.join(result_dir, 'env_renders')
    object_voxel_dir = os.path.join(result_dir, 'object_voxels')
    object_mesh_dir = os.path.join(result_dir, 'object_mesh')
    
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(object_render_dir, exist_ok=True)
    os.makedirs(object_env_render_dir, exist_ok=True)
    os.makedirs(object_voxel_dir, exist_ok=True)
    os.makedirs(object_mesh_dir, exist_ok=True)
    
    # --- Generate Color Palette ---
    np.random.seed(42)
    palette = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = [255, 255, 255]  # Background -> White
    palette_tensor = torch.from_numpy(palette).float().cuda() / 255.0
    
    # --- Render Full Semantic Visualization ---
    print(f"Rendering full semantic visualization to {viz_dir}...")
    
    ori_sh0 = voxel_model.sh0.data.clone()
    ori_shs = voxel_model.shs.data.clone()
    
    voxel_colors = palette_tensor[voxel_labels]
    new_sh0 = activation_utils.rgb2shzero(voxel_colors)
    
    voxel_model.sh0.data = new_sh0
    voxel_model.shs.data.fill_(0.0)
    
    step = max(1, len(cameras) // 20)
    viz_cams = cameras[::step]
    
    with torch.no_grad():
        for i, cam in enumerate(tqdm(viz_cams, desc="Rendering full scene")):
            res = voxel_model.render(cam, color_mode=None)
            img_tensor = res['color']
            
            img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            fname = cam.image_name
            if not os.path.splitext(fname)[1]:
                fname += ".png"
            
            save_path = os.path.join(viz_dir, f"sem_{fname}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img_bgr)
    
    # Restore original appearance
    voxel_model.sh0.data = ori_sh0
    voxel_model.shs.data = ori_shs
    
    # --- Process Each Object ---
    print("\nProcessing individual objects...")
    
    class_to_id = {v: k for k, v in id_to_class.items()}
    object_summary = []
    
    for cidx in tqdm(range(1, num_classes), desc="Exporting objects"):
        original_id = class_to_id[cidx]
        object_mask = (voxel_labels == cidx)
        num_voxels = object_mask.sum().item()
        
        if num_voxels == 0:
            print(f"  Object {original_id}: No voxels found, skipping.")
            continue
        
        print(f"  Object {original_id}: {num_voxels} voxels")
        
        # Get object color
        obj_color = palette_tensor[cidx]
        
        # Render individual object views
        obj_render_subdir = os.path.join(object_render_dir, f"object_{original_id:03d}")
        obj_env_render_subdir = os.path.join(object_env_render_dir, f"object_{original_id:03d}")
        render_select_object(
            voxel_model=voxel_model,
            cameras=cameras,
            object_mask=object_mask,
            color=obj_color,
            output_dir=obj_render_subdir,
            object_id=original_id,
            num_views=num_render_views
        )
        full_mask = torch.ones_like(object_mask)
        render_select_object(
            voxel_model=voxel_model,
            cameras=cameras,
            object_mask=full_mask,
            color=obj_color,
            output_dir=obj_env_render_subdir,
            object_id=original_id,
            num_views=num_render_views
        )
        
        # Export object voxel data
        obj_data = export_object_voxels(
            voxel_model=voxel_model,
            object_mask=object_mask,
            object_id=cidx,
            output_dir=object_voxel_dir,
            original_id=original_id
        )
        
        if obj_data:
            # Export object mesh
            if TRIMESH_AVAILABLE:
                centers = obj_data['vox_center']
                sizes = obj_data['vox_size']
                sh0 = obj_data['sh0']
                colors = activation_utils.shzero2rgb(sh0.squeeze(1)).clamp(0, 1)
                
                mesh = voxels_to_mesh(centers, sizes, colors)
                mesh_path = os.path.join(object_mesh_dir, f"object_{original_id:03d}.glb")
                mesh.export(mesh_path)

            object_summary.append({
                'original_id': original_id,
                'class_index': cidx,
                'num_voxels': num_voxels,
                'bbox_min': obj_data['bbox_min'].tolist(),
                'bbox_max': obj_data['bbox_max'].tolist(),
                'bbox_center': obj_data['bbox_center'].tolist(),
                'bbox_extent': obj_data['bbox_extent'].tolist(),
                'color_rgb': (palette[cidx] / 255.0).tolist(),
            })
    
    # --- Export Background Voxels ---
    print("\nExporting background voxels...")
    bg_mask = (voxel_labels == 0)
    num_bg_voxels = bg_mask.sum().item()
    print(f"  Background: {num_bg_voxels} voxels")

    bg_data = export_object_voxels(
        voxel_model=voxel_model,
        object_mask=bg_mask,
        object_id=0,
        output_dir=object_voxel_dir,
        original_id=BG_ID,
        recenter=False,
    )
    
    # if bg_data and TRIMESH_AVAILABLE:
    #     centers = bg_data['vox_center']
    #     sizes = bg_data['vox_size']
    #     sh0 = bg_data['sh0']
    #     colors = activation_utils.shzero2rgb(sh0.squeeze(1)).clamp(0, 1)
        
    #     mesh = voxels_to_mesh(centers, sizes, colors)
    #     mesh_path = os.path.join(object_mesh_dir, f"object_{BG_ID:03d}_background.glb")
    #     mesh.export(mesh_path)
    
    # --- Save Global Segmentation Data ---
    labels_cpu = voxel_labels.cpu().numpy()
    obj_voxel_map = {}
    
    for cidx in range(1, num_classes):
        oid = class_to_id[cidx]
        indices = np.where(labels_cpu == cidx)[0]
        if len(indices) > 0:
            obj_voxel_map[oid] = indices
    
    global_data_path = os.path.join(result_dir, 'voxel_semantic_data.pt')
    torch.save({
        'voxel_labels': voxel_labels.cpu(),
        'object_voxels': obj_voxel_map,
        'id_mapping': id_to_class,
        'class_to_id': class_to_id,
        'vox_centers': voxel_model.vox_center.cpu(),
        'palette': palette,
        'object_summary': object_summary,
    }, global_data_path)
    
    # --- Print Summary ---
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total voxels: {voxel_model.num_voxels}")
    print(f"Total objects: {len(object_summary)}")
    print(f"\nOutput directory: {result_dir}")
    print(f"  - Full scene semantic: {viz_dir}")
    print(f"  - Individual object renders: {object_render_dir}")
    print(f"  - Individual object voxels: {object_voxel_dir}")
    print(f"  - Individual object meshes: {object_mesh_dir}")
    print(f"  - Global data: {global_data_path}")
    
    print("\nObject details:")
    for obj in object_summary:
        print(f"  Object {obj['original_id']:3d}: {obj['num_voxels']:6d} voxels, "
              f"bbox center = [{obj['bbox_center'][0]:.2f}, {obj['bbox_center'][1]:.2f}, {obj['bbox_center'][2]:.2f}]")
    
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse 2D instance segmentation masks into 3D sparse voxels with object export.")
    parser.add_argument('--source_path', type=str, required=True,
                        help='Path to dataset folder (e.g., dataset/Replica/scan1)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model output folder (e.g., outputs/Replica/scan1)')
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='Path to instance masks folder. Default: <source_path>/instance_masks')
    parser.add_argument('--iteration', type=int, default=-1,
                        help='Model iteration to load (default: -1, load latest)')
    parser.add_argument('--bg_id', type=int, default=255,
                        help='Background ID in mask images (default: 255)')
    parser.add_argument('--num_views', type=int, default=8,
                        help='Number of views to render per object (default: 8)')
    args = parser.parse_args()

    segm_3d(
        source_path=args.source_path,
        model_path=args.model_path,
        mask_dir=args.mask_dir,
        iteration=args.iteration,
        bg_id=args.bg_id,
        num_views=args.num_views
    )
