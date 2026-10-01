import os
import time
import numpy as np
from tqdm import tqdm
from os import makedirs
import imageio
import open3d as o3d
import torch

from src.config import cfg, update_argparser, update_config

from src.dataloader.data_pack import DataPack
from src.sparse_voxel_model import SparseVoxelModel

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Sparse voxels raster rendering.")
    parser.add_argument('model_path')
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--suffix", default="", type=str)
    parser.add_argument("--overwrite_ss", default=None, type=float)
    parser.add_argument("--overwrite_vox_geo_mode", default=None)
        
    parser.add_argument("--mesh_path", default="", type=str)
    parser.add_argument("--output_dir", default="", type=str)
    
    args = parser.parse_args()
    print("Rendering " + args.model_path)

    # Load config
    update_config(os.path.join(args.model_path, 'config.yaml'))


    # Load data
    data_pack = DataPack(cfg.data, cfg.model.white_background)

    cfg.model.model_path = args.model_path
    
    # Load model
    voxel_model = SparseVoxelModel(cfg.model)
    loaded_iter = voxel_model.load_iteration(args.iteration)

    # Output path suffix
    suffix = args.suffix
    if not args.suffix:
        if cfg.data.res_downscale > 0:
            suffix += f"_r{cfg.data.res_downscale}"
        if cfg.data.res_width > 0:
            suffix += f"_w{cfg.data.res_width}"

    if args.overwrite_ss:
        voxel_model.ss = args.overwrite_ss
        if not args.suffix:
            suffix += f"_ss{args.overwrite_ss:.2f}"
    
    if args.overwrite_vox_geo_mode:
        voxel_model.vox_geo_mode = args.overwrite_vox_geo_mode
        if not args.suffix:
            suffix += f"_{args.overwrite_vox_geo_mode}"

    voxel_model.freeze_vox_geo()
    views = data_pack.get_train_cameras()

    
    tsdf_path = os.path.join(args.model_path, "mesh", "tsdf")

    mesh = o3d.io.read_triangle_mesh(os.path.join(tsdf_path, "tsdf_fusion_post.ply") if args.mesh_path=="" else args.mesh_path)
    mesh.vertex_colors = o3d.utility.Vector3dVector() 
    mesh.triangle_material_ids = o3d.utility.IntVector() 
    
    vertices = np.asarray(mesh.vertices)
    vertices[:, 1] *= -1  
    vertices[:, 2] *= -1 
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    
    mesh.compute_vertex_normals()


    ############################ simple ##################################

    material = o3d.visualization.rendering.MaterialRecord()

    material.shader = "defaultLit"
    material.base_color = [0.8, 0.8, 0.8, 1.0] 
    material.base_roughness = 0.5
    material.base_metallic = 0.0
    material.base_reflectance = 0.2 

    w, h = views[0].image_width, views[0].image_height
    renderer_simple = o3d.visualization.rendering.OffscreenRenderer(w, h)
    scene = renderer_simple.scene

    scene.add_geometry("mesh", mesh, material)
    
    scene.set_background(np.array([0.4, 0.4, 0.4, 1.0], dtype=np.float32)) 
    
    scene.set_lighting(scene.LightingProfile.SOFT_SHADOWS, (0.5, 0.5, 0.5)) 
    scene.scene.enable_sun_light(True)
    scene.scene.set_sun_light(
        direction=[-1, -1, 1],
        color=[1, 1, 1],
        intensity=110000
    )
    
    ############################ normal ##################################
    normal_renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)
    mesh.compute_vertex_normals()
    
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "normals"
    mat.base_roughness = 1
    mat.base_metallic = 0.0
    mat.base_reflectance = 0
    
    normal_renderer.scene.add_geometry("mesh", mesh, mat)
    normal_renderer.scene.set_background(np.array([1, 1, 1, 1.0], dtype=np.float32))
    normal_renderer.scene.scene.enable_sun_light(False)
    
    normal_renderer.scene.view.set_post_processing(False)

        
    render_mesh_path = os.path.join(voxel_model.model_path, "train", f"ours_{loaded_iter}{suffix}", "mesh") if args.output_dir=="" else args.output_dir
    os.makedirs(render_mesh_path, exist_ok=True)
    
    colmap_to_opengl = np.eye(4)
    colmap_to_opengl[1, 1] = -1
    colmap_to_opengl[2, 2] = -1

    for idx, view in enumerate(tqdm(views)):
        pose = np.eye(4)
        pose[:3, :3] = view.R.transpose(-1, -2)
        pose[:3, 3] = view.T 
        
        pose = pose @ colmap_to_opengl

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=w,
            height=h,
            fx=view.Fx,
            fy=view.Fy,
            cx=view.Cx,
            cy=view.Cy
        )
    
        renderer_simple.setup_camera(intrinsic, pose)
        image = renderer_simple.render_to_image()
        o3d.io.write_image(os.path.join(render_mesh_path, view.image_name + "_simple.png"), image)
        
        
        normal_renderer.setup_camera(intrinsic, pose)
        world_normal_image = normal_renderer.render_to_image()
        o3d.io.write_image(os.path.join(render_mesh_path, view.image_name + "_normal.png"), world_normal_image)