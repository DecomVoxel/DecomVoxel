import os
import sys
import argparse
import colorsys
import numpy as np
import trimesh
from scipy.spatial import KDTree, Delaunay
import cv2
import torch
from tqdm import tqdm
from PIL import Image
import open3d as o3d

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GEOSVR_ROOT = os.path.dirname(PROJECT_ROOT)
if GEOSVR_ROOT not in sys.path:
    sys.path.insert(0, GEOSVR_ROOT)

from cam_util import load_selected_camera_params
from projection_util import compute_world_points


def _intrinsics_from_camera_spec(spec):
    """Build 3x3 OpenCV-style K from MiniCam fields in a camera spec dict."""
    cam = spec["camera"]
    w = float(spec["width"])
    h = float(spec["height"])
    fovx = float(cam.fovx)
    fovy = float(cam.fovy)
    fx = w / (2.0 * np.tan(fovx * 0.5))
    fy = h / (2.0 * np.tan(fovy * 0.5))
    cx = w * float(cam.cx_p)
    cy = h * float(cam.cy_p)
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _view_dict_from_camera_spec(spec, device):
    """view_dict compatible with projection_util.compute_world_points."""
    K = _intrinsics_from_camera_spec(spec)
    intrinsics = torch.from_numpy(K).to(device)
    c2w = torch.from_numpy(np.asarray(spec["c2w"], dtype=np.float32)).to(device)
    return {
        "intrinsics": intrinsics,
        "cam2world": c2w,
        "camera_name": spec["camera_name"],
    }


def plane_id_to_debug_rgba(plane_id):
    """Distinct RGBA for each plane_id; gray for 0 (unassigned)."""
    if int(plane_id) == 0:
        return np.array([128, 128, 128, 255], dtype=np.uint8)
    hashed = (int(plane_id) * 2654435761) % (2 ** 32)
    hue = hashed / float(2 ** 32)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return np.array(
        [int(r * 255), int(g * 255), int(b * 255), 255],
        dtype=np.uint8,
    )


def vertex_colors_from_plane_ids(plane_ids):
    """plane_ids: [N] int — returns [N, 4] uint8 RGBA."""
    plane_ids = np.asarray(plane_ids, dtype=np.int64).reshape(-1)
    out = np.zeros((len(plane_ids), 4), dtype=np.uint8)
    for pid in np.unique(plane_ids):
        c = plane_id_to_debug_rgba(int(pid))
        out[plane_ids == pid] = c
    return out


def assign_vertex_planes_from_global_equations(
    mesh_v,
    vertex_to_plane,
    global_plane_params,
    dist_thresh,
):
    """
    Second pass: for vertices still unassigned (id 0), use fitted global planes
    from merge_3d_plane (global_plane_params.npy): plane n·x + d = 0 with unit n.

    Args:
        mesh_v: (V, 3) float32
        vertex_to_plane: (V,) int32, modified in place
        global_plane_params: (K, 4) row i = [nx, ny, nz, d]; row 0 unused
        dist_thresh: assign if min_p |n·x + d| < dist_thresh
    """
    global_plane_params = np.asarray(global_plane_params, dtype=np.float64)
    unassigned = np.where(vertex_to_plane == 0)[0]
    if len(unassigned) == 0:
        return vertex_to_plane

    plane_ids = []
    normals = []
    ds = []
    for pid in range(1, global_plane_params.shape[0]):
        n_raw = global_plane_params[pid, :3]
        d_raw = float(global_plane_params[pid, 3])
        norm = np.linalg.norm(n_raw)
        if norm < 1e-8:
            continue
        n_unit = n_raw / norm
        d_unit = d_raw / norm
        plane_ids.append(pid)
        normals.append(n_unit)
        ds.append(d_unit)

    if not plane_ids:
        return vertex_to_plane

    plane_ids = np.array(plane_ids, dtype=np.int32)
    Nm = np.stack(normals, axis=0)
    d_vec = np.array(ds, dtype=np.float64)

    pts = mesh_v[unassigned].astype(np.float64)
    dists = np.abs(pts @ Nm.T + d_vec)
    best_j = np.argmin(dists, axis=1)
    best_d = dists[np.arange(len(pts)), best_j]
    take = best_d < float(dist_thresh)
    if np.any(take):
        vertex_to_plane[unassigned[take]] = plane_ids[best_j[take]]
        n_filled = int(np.sum(take))
        print(
            f"[Global plane equations] Filled {n_filled} / {len(unassigned)} "
            f"previously unassigned vertices (dist_thresh={dist_thresh})"
        )
    else:
        print(
            f"[Global plane equations] No vertices within dist_thresh={dist_thresh} "
            f"({len(unassigned)} unassigned remain)"
        )

    return vertex_to_plane


def save_pcds_by_plane_id(pcd_vertices, pcd_plane_id, out_dir, prefix="plane_pcd"):
    """Save one point cloud per plane id (skips id 0)."""
    os.makedirs(out_dir, exist_ok=True)
    ids_flat = np.asarray(pcd_plane_id, dtype=np.int64).reshape(-1)
    for plane_id in np.unique(ids_flat):
        # if plane_id == 0:
        #     continue
        mask = ids_flat == plane_id
        pts = pcd_vertices[mask]
        if len(pts) == 0:
            print(f"************** No points found for plane {plane_id} **************")
            continue
        c = plane_id_to_debug_rgba(int(plane_id))[:3]
        colors = np.broadcast_to(c.astype(np.float64) / 255.0, (len(pts), 3)).copy()
        pcd = trimesh.PointCloud(np.asarray(pts, dtype=np.float64, order="C"), colors=colors)
        path = os.path.join(out_dir, f"{prefix}_{int(plane_id):03d}.ply")
        pcd.export(path)


def save_tensor_as_pcd(pcd, path, pcd_colors=None):

    if isinstance(pcd, torch.Tensor):
        pcd = pcd.detach().cpu().numpy()
    pcd = trimesh.PointCloud(pcd)
    if pcd_colors is not None:
        if isinstance(pcd_colors, torch.Tensor):
            pcd_colors = pcd_colors.detach().cpu().numpy()
        pcd.colors = pcd_colors
    pcd.export(path)


def load_pcd_from_views(camera_json_path, render_depth_path, global_plane_mask_path, downsample_pixel_grid, downsample_voxel_size, device='cuda'):
    """
    Load planar point cloud from depth maps and global plane masks, with voxel downsampling.

    Args:
        camera_json_path: Path to camera_params.json (See3D / save_selected_camera_params format)
        render_depth_path: Directory containing merge_refined_depth_frame*.tiff
        global_plane_mask_path: Directory containing global_plane_mask_frame*.npy
        downsample_pixel_grid: Pixel stride for 2D subsampling (e.g. 4 -> one sample per 4x4 block)
        downsample_voxel_size: Open3D voxel size for downsampling (often delta/2)
        device: torch device for projection
    """
    camera_specs = load_selected_camera_params(camera_json_path, keep_metadata=True)

    pcd_vertices_list = []
    pcd_planeID_list = []

    print(f"Projecting {len(camera_specs)} views to generate PCD...")
    for spec in tqdm(camera_specs):
        frame_idx = int(spec["camera_name"])
        file_prefix = f"{frame_idx:06d}"

        depth_file = os.path.join(render_depth_path, f"merge_refined_depth_frame{file_prefix}.tiff")
        mask_file = os.path.join(global_plane_mask_path, f"global_plane_mask_frame{file_prefix}.npy")

        if not os.path.exists(depth_file) or not os.path.exists(mask_file):
            continue

        depth = torch.from_numpy(np.array(Image.open(depth_file))).float().to(device)
        global_mask = torch.from_numpy(np.load(mask_file)).int().to(device)

        view_dict = _view_dict_from_camera_spec(spec, device)
        pts_world, valid_mask = compute_world_points(depth, view_dict)

        plane_mask = (global_mask > 0) & valid_mask

        if downsample_pixel_grid > 1:
            grid_mask = torch.zeros_like(plane_mask, dtype=torch.bool)
            grid_mask[::downsample_pixel_grid, ::downsample_pixel_grid] = True
            plane_mask = plane_mask & grid_mask

        if plane_mask.sum() == 0:
            continue

        flat_pts = pts_world[plane_mask].cpu().numpy()
        flat_ids = global_mask[plane_mask].cpu().numpy()

        pcd_vertices_list.append(flat_pts)
        pcd_planeID_list.append(flat_ids)

    if not pcd_vertices_list:
        raise ValueError("No planar points found in the provided views.")

    all_v = np.concatenate(pcd_vertices_list, axis=0)
    all_ids = np.concatenate(pcd_planeID_list, axis=0).reshape(-1, 1)

    print(f"Voxel downsampling (original: {len(all_v)} points)...")

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(all_v)

    num_points = len(all_v)
    colors = np.zeros((num_points, 3))
    colors[:, 0] = np.arange(num_points) / max(num_points, 1)
    o3d_pcd.colors = o3d.utility.Vector3dVector(colors)

    downsampled_pcd = o3d_pcd.voxel_down_sample(downsample_voxel_size)

    downsampled_colors = np.asarray(downsampled_pcd.colors)
    recovered_indices = np.round(downsampled_colors[:, 0] * num_points).astype(int)
    recovered_indices = np.clip(recovered_indices, 0, num_points - 1)

    pcd_vertices = np.array(all_v[recovered_indices], dtype=np.float32, copy=True)
    pcd_planeID = np.array(all_ids[recovered_indices], copy=True)

    print(f"Downsampled to {len(pcd_vertices)} points (factor: {num_points/len(pcd_vertices):.2f}x)")

    return pcd_vertices, pcd_planeID


def refine_mesh_layout(
    init_mesh,
    pcd_vertices,
    pcd_planeID,
    voxel_size=0.01,
    global_plane_params=None,
    plane_equation_dist_thresh=None,
):
    """
    Mesh layout refinement (vectorized PCA, edge-based boundaries, outlier filtering).

    Args:
        init_mesh: trimesh.Trimesh
        pcd_vertices: np.ndarray [N, 3]
        pcd_planeID: np.ndarray [N, 1], plane id per point (0 = non-planar)
        voxel_size: delta in the paper
        global_plane_params: optional (K, 4) from merge_3d_plane/global_plane_params.npy;
            second pass assigns unassigned vertices by min |n·x+d| < threshold
        plane_equation_dist_thresh: distance threshold for second pass; default 3.0 * voxel_size

    Returns:
        refined_mesh: trimesh.Trimesh with per-vertex RGBA colors by assigned plane id (process=False)
    """
    mesh_v = np.array(init_mesh.vertices, dtype=np.float32)
    mesh_f = np.array(init_mesh.faces, dtype=np.int32)

    print(f"Memory Check: Mesh has {len(mesh_v)} vertices, {len(mesh_f)} faces.")

    print("Building KDTree for PCD...")
    pcd_tree = KDTree(pcd_vertices)

    vertex_to_plane = np.zeros(len(mesh_v), dtype=np.int32)

    print(f"Assigning {len(mesh_v)} vertices (KDTree from sparse PCD)...")
    dists, idxs = pcd_tree.query(mesh_v, k=1)
    print("KDTree query done.")

    nearest_pcd_ids = pcd_planeID[idxs, 0].astype(np.int32)
    assign_mask = (dists < 1.5 * voxel_size) & (nearest_pcd_ids != 0)
    vertex_to_plane[assign_mask] = nearest_pcd_ids[assign_mask]

    del dists, idxs, nearest_pcd_ids, assign_mask

    if global_plane_params is not None:
        print("[INFO]: use assign vertex planes")
        if plane_equation_dist_thresh is None:
            plane_equation_dist_thresh = 3.0 * float(voxel_size)
        assign_vertex_planes_from_global_equations(
            mesh_v,
            vertex_to_plane,
            global_plane_params,
            plane_equation_dist_thresh,
        )
    else:
        print("[WARNING]: not use assign vertex planes")

    unique_planes = np.unique(vertex_to_plane)
    unique_planes = unique_planes[unique_planes != 0]
    unique_planes = np.sort(unique_planes)

    print(f"Planes with at least one assigned vertex: {len(unique_planes)}")

    print("Computing mesh edges (may take some time)...")
    edges = init_mesh.edges

    new_vertices_list = [mesh_v]
    final_plane_id_parts = [vertex_to_plane.copy()]
    current_total_v = len(mesh_v)
    all_new_faces = []

    print("Starting plane-by-plane refinement with Outlier Filtering...")
    for idx, plane_id in enumerate(unique_planes):
        print(f"\n[Plane {idx+1}/{len(unique_planes)}] Processing Plane ID: {plane_id}")

        plane_v_indices = np.where(vertex_to_plane == plane_id)[0]
        if len(plane_v_indices) < 3:
            continue

        pts_initial = mesh_v[plane_v_indices]
        center_initial = pts_initial.mean(axis=0)
        cov_initial = np.cov(pts_initial.T)
        _, eig_vecs = np.linalg.eigh(cov_initial)
        normal_initial = eig_vecs[:, 0]

        dists_to_plane = np.abs((pts_initial - center_initial) @ normal_initial)

        outlier_mask = dists_to_plane > (1.0 * voxel_size)
        if np.any(outlier_mask):
            num_outliers = np.sum(outlier_mask)
            outlier_indices = plane_v_indices[outlier_mask]
            vertex_to_plane[outlier_indices] = 0
            plane_v_indices = plane_v_indices[~outlier_mask]
            print(f" - Filtered {num_outliers} outlier vertices (far from plane).")

        if len(plane_v_indices) < 3:
            continue

        print(" - Step 1: Detecting boundary via Edges...")
        v0_in_plane = (vertex_to_plane[edges[:, 0]] == plane_id)
        v1_in_plane = (vertex_to_plane[edges[:, 1]] == plane_id)
        is_boundary_edge = v0_in_plane ^ v1_in_plane

        b_v_indices = np.unique(np.concatenate([
            edges[is_boundary_edge & v0_in_plane, 0],
            edges[is_boundary_edge & v1_in_plane, 1]
        ]))

        if len(b_v_indices) < 3:
            print(" - Not enough boundary vertices, skipping.")
            continue

        interior_v_indices = np.setdiff1d(plane_v_indices, b_v_indices)
        print(f" - Boundary: {len(b_v_indices)}, Interior: {len(interior_v_indices)}")

        print(" - Step 2: Precise PCA Projection...")
        pts = mesh_v[plane_v_indices]
        center = pts.mean(axis=0)
        cov = np.cov(pts.T)
        _, eigenvectors = np.linalg.eigh(cov)
        u, v = eigenvectors[:, 2], eigenvectors[:, 1]

        def project_2d(p3d):
            rel = p3d - center
            return np.stack([rel @ u, rel @ v], axis=1)

        v_b_2d = project_2d(mesh_v[b_v_indices])
        v_i_2d = project_2d(mesh_v[interior_v_indices])
        all_v_2d = np.concatenate([v_b_2d, v_i_2d], axis=0)

        print(" - Step 3: MER and Grid Generation...")
        rect = cv2.minAreaRect(all_v_2d.astype(np.float32))
        box = cv2.boxPoints(rect)
        grid_dist = voxel_size * 2
        x_min, y_min = np.min(box, axis=0)
        x_max, y_max = np.max(box, axis=0)

        nx, ny = int((x_max - x_min) / grid_dist), int((y_max - y_min) / grid_dist)
        if nx * ny > 500_000:
            print(f" !! Grid resolution too high ({nx*ny}). Downsampling.")
            grid_dist = max((x_max - x_min), (y_max - y_min)) / 400.0

        grid_x, grid_y = np.meshgrid(np.arange(x_min, x_max, grid_dist), np.arange(y_min, y_max, grid_dist))
        grid_pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

        hull = cv2.convexHull(all_v_2d.astype(np.float32))
        mask = [cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), False) >= 0 for p in grid_pts]
        grid_pts_filtered = grid_pts[mask]

        combined_2d = np.concatenate([v_b_2d, grid_pts_filtered], axis=0)
        if len(combined_2d) < 3:
            continue
        print(f" - Step 5: Delaunay on {len(combined_2d)} points...")
        tri = Delaunay(combined_2d)

        new_v_3d = center + combined_2d[:, 0:1] * u + combined_2d[:, 1:2] * v
        start_v_idx = current_total_v
        new_vertices_list.append(new_v_3d.astype(np.float32))
        final_plane_id_parts.append(np.full(len(new_v_3d), plane_id, dtype=np.int32))
        current_total_v += len(new_v_3d)

        index_map = {}
        for i, old_idx in enumerate(b_v_indices):
            index_map[i] = old_idx
        for i in range(len(grid_pts_filtered)):
            index_map[i + len(b_v_indices)] = start_v_idx + len(b_v_indices) + i

        for t in tri.simplices:
            all_new_faces.append([index_map[t[0]], index_map[t[1]], index_map[t[2]]])

        print(f" - Plane {plane_id} successfully simplified.")

    print("\nFinalizing: Merging all data and cleaning planar regions...")

    face_v_planes = vertex_to_plane[mesh_f]
    is_planar_face = (face_v_planes[:, 0] == face_v_planes[:, 1]) & \
                     (face_v_planes[:, 1] == face_v_planes[:, 2]) & \
                     (face_v_planes[:, 0] != 0)

    final_v = np.concatenate(new_vertices_list, axis=0)
    final_vertex_plane_ids = np.concatenate(final_plane_id_parts, axis=0)
    final_f = np.concatenate([
        mesh_f[~is_planar_face],
        np.array(all_new_faces, dtype=np.int32)
    ], axis=0)

    print(f"Refinement complete. Final vertices: {len(final_v)}, Faces: {len(final_f)}")

    vertex_colors = vertex_colors_from_plane_ids(final_vertex_plane_ids)
    refined_mesh = trimesh.Trimesh(
        vertices=final_v,
        faces=final_f,
        vertex_colors=vertex_colors,
        process=False,
    )

    return refined_mesh


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simplify planar regions in a reconstructed background mesh.")
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.path.join("outputs", "bg", "scan1", "see3d_guidance_views"),
        help="Background guidance root containing camera_params.json and reconstruction outputs",
    )
    args = parser.parse_args()

    data_root_path = args.data_root
    mesh_root_path = os.path.join(data_root_path, "tsdf_refine_bg")

    camera_json_path = os.path.join(data_root_path, "camera_params.json")
    render_depth_path = os.path.join(data_root_path, "merge_3d_plane")
    global_plane_mask_path = os.path.join(data_root_path, "merge_3d_plane")
    global_plane_params_path = os.path.join(data_root_path, "merge_3d_plane", "global_plane_params.npy")

    init_mesh_path = os.path.join(mesh_root_path, "tsdf_refine_bg_post.ply")
    save_mesh_path = os.path.join(mesh_root_path, "refined_mesh_simplified.ply")
    save_sparse_pcd_root_path = os.path.join(mesh_root_path, "vis_pcd")
    os.makedirs(save_sparse_pcd_root_path, exist_ok=True)
    save_sparse_pcd_path = os.path.join(save_sparse_pcd_root_path, "sparse_pcd.ply")

    delta = 0.01
    downsample_pixel_grid = 4
    downsample_voxel_size = 0.5 * delta
    plane_equation_dist_thresh = 3.0 * delta

    global_plane_params = None
    if os.path.isfile(global_plane_params_path):
        global_plane_params = np.load(global_plane_params_path)
        print(f"Loaded global_plane_params: {global_plane_params_path} shape={global_plane_params.shape}")
    else:
        print(f"Warning: global_plane_params not found at {global_plane_params_path}; skipping equation pass.")

    print("Step 1: Loading initial mesh...")
    if not os.path.exists(init_mesh_path):
        print(f"Error: Mesh file not found at {init_mesh_path}")
        exit()
    init_mesh = trimesh.load(init_mesh_path)

    print("Step 2: Generating sparse planar point cloud from depth maps...")
    try:
        pcd_v, pcd_ids = load_pcd_from_views(
            camera_json_path,
            render_depth_path,
            global_plane_mask_path,
            downsample_pixel_grid,
            downsample_voxel_size,
        )
        save_tensor_as_pcd(pcd_v, save_sparse_pcd_path)
        save_pcds_by_plane_id(pcd_v, pcd_ids, save_sparse_pcd_root_path, prefix="plane_pcd")
        print(f"Generated PCD with {len(pcd_v)} points (merged + per-plane ply).")
    except Exception as e:
        print(f"Error during PCD generation: {e}")
        exit()

    print("Step 3: Refining mesh layout...")
    refined_mesh = refine_mesh_layout(
        init_mesh,
        pcd_v,
        pcd_ids,
        voxel_size=delta,
        global_plane_params=global_plane_params,
        plane_equation_dist_thresh=plane_equation_dist_thresh,
    )

    print(f"Step 4: Saving refined mesh to {save_mesh_path}...")
    refined_mesh.export(save_mesh_path)

    print("Mesh layout refinement completed successfully.")
