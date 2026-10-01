import os
import sys
import json
import argparse
import colorsys
import collections
import itertools

import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import Delaunay
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GEOSVR_ROOT = os.path.dirname(PROJECT_ROOT)
if GEOSVR_ROOT not in sys.path:
    sys.path.insert(0, GEOSVR_ROOT)

from cam_util import load_selected_camera_params


def plane_id_to_debug_rgba(plane_id):
    if int(plane_id) == 0:
        return np.array([128, 128, 128, 255], dtype=np.uint8)
    hashed = (int(plane_id) * 2654435761) % (2 ** 32)
    hue = hashed / float(2 ** 32)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return np.array([int(r * 255), int(g * 255), int(b * 255), 255], dtype=np.uint8)


def vertex_colors_from_plane_ids(plane_ids):
    plane_ids = np.asarray(plane_ids, dtype=np.int64).reshape(-1)
    colors = np.zeros((len(plane_ids), 4), dtype=np.uint8)
    for pid in np.unique(plane_ids):
        colors[plane_ids == pid] = plane_id_to_debug_rgba(int(pid))
    return colors


def normalize_np(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def oriented_average_normals(normals):
    if len(normals) == 0:
        raise ValueError("Empty normals for oriented average.")
    normals = [normalize_np(np.asarray(n, dtype=np.float32).reshape(1, 3))[0] for n in normals]
    ref = normals[0]
    aligned = []
    for n in normals:
        aligned.append(n if float(np.dot(ref, n)) >= 0.0 else -n)
    mean_n = np.mean(np.stack(aligned, axis=0), axis=0)
    return normalize_np(mean_n.reshape(1, 3))[0].astype(np.float32)


def _intrinsics_from_camera_spec(spec):
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


def load_global_plane_info(global_plane_info_path):
    with open(global_plane_info_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    plane_info = {}
    for key, value in payload.items():
        pid = int(key)
        plane_params = value.get("plane_params", None)
        normal = value.get("normal", None)
        mean = value.get("mean", None)
        if plane_params is None or normal is None or mean is None:
            continue

        plane_params = np.asarray(plane_params, dtype=np.float32)
        normal = normalize_np(np.asarray(normal, dtype=np.float32).reshape(1, 3))[0]
        mean = np.asarray(mean, dtype=np.float32)

        d = float(plane_params[3])
        plane_info[pid] = {
            "normal": normal,
            "mean": mean,
            "plane_params": np.array([normal[0], normal[1], normal[2], d], dtype=np.float32),
            "num_views": int(value.get("num_views", 0)),
            "num_voxels": int(value.get("num_voxels", 0)),
        }
    return plane_info


class UnionFind:
    def __init__(self, items):
        self.parent = {int(x): int(x) for x in items}

    def find(self, x):
        x = int(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


class MeshDepthRenderer:
    def __init__(self, mesh):
        self.mesh = self._to_o3d_render_mesh(mesh)
        self.renderers = {}
        self.material = o3d.visualization.rendering.MaterialRecord()
        self.material.shader = "defaultUnlit"

        self.colmap_to_opengl = np.eye(4, dtype=np.float64)
        self.colmap_to_opengl[1, 1] = -1.0
        self.colmap_to_opengl[2, 2] = -1.0

    @staticmethod
    def _to_o3d_render_mesh(mesh):
        render_mesh = o3d.geometry.TriangleMesh()
        vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
        faces = np.asarray(mesh.faces, dtype=np.int32).copy()

        vertices[:, 1] *= -1.0
        vertices[:, 2] *= -1.0

        render_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        render_mesh.triangles = o3d.utility.Vector3iVector(faces)
        render_mesh.compute_vertex_normals()
        return render_mesh

    def _get_renderer(self, width, height):
        key = (int(width), int(height))
        if key in self.renderers:
            return self.renderers[key]

        renderer = o3d.visualization.rendering.OffscreenRenderer(int(width), int(height))
        renderer.scene.add_geometry("mesh", self.mesh, self.material)
        renderer.scene.set_background(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        self.renderers[key] = renderer
        return renderer

    def render_depth(self, spec):
        width = int(spec["width"])
        height = int(spec["height"])
        renderer = self._get_renderer(width, height)

        K = _intrinsics_from_camera_spec(spec)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            float(K[0, 0]),
            float(K[1, 1]),
            float(K[0, 2]),
            float(K[1, 2]),
        )

        pose = np.asarray(spec["c2w"], dtype=np.float64).copy()
        pose = pose @ self.colmap_to_opengl
        renderer.setup_camera(intrinsic, pose)

        try:
            depth_image = renderer.render_to_depth_image(z_in_view_space=True)
        except TypeError as exc:
            raise RuntimeError(
                "Current Open3D does not support render_to_depth_image(z_in_view_space=True)."
            ) from exc

        depth = np.asarray(depth_image).astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth <= 0.0] = 0.0
        return depth


def project_vertices_to_view(mesh_vertices, spec):
    c2w = np.asarray(spec["c2w"], dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    K = _intrinsics_from_camera_spec(spec).astype(np.float64)

    pts_cam = mesh_vertices @ w2c[:3, :3].T + w2c[:3, 3]
    z = pts_cam[:, 2].astype(np.float32)

    uv = np.zeros((mesh_vertices.shape[0], 2), dtype=np.float32)
    valid_z = z > 1e-8
    uv[valid_z, 0] = (K[0, 0] * pts_cam[valid_z, 0] / z[valid_z]) + K[0, 2]
    uv[valid_z, 1] = (K[1, 1] * pts_cam[valid_z, 1] / z[valid_z]) + K[1, 2]
    return uv, z


def collect_vertex_votes_from_views(
    mesh_vertices,
    camera_specs,
    global_plane_mask_root,
    depth_renderer,
    depth_rel_tol=1e-3,
):
    num_vertices = mesh_vertices.shape[0]
    vertex_votes = [collections.Counter() for _ in range(num_vertices)]
    vertex_visible_views = np.zeros(num_vertices, dtype=np.int32)
    vertex_plane_views = np.zeros(num_vertices, dtype=np.int32)

    for spec in tqdm(camera_specs, desc="Project mesh vertices to masks"):
        frame_idx = int(spec["camera_name"])
        mask_path = os.path.join(
            global_plane_mask_root,
            f"global_plane_mask_frame{frame_idx:06d}.npy",
        )
        if not os.path.isfile(mask_path):
            continue

        plane_mask = np.load(mask_path).astype(np.int32)
        render_depth = depth_renderer.render_depth(spec)

        if render_depth.shape != plane_mask.shape:
            raise ValueError(
                f"Depth/mask shape mismatch for frame {frame_idx}: "
                f"{render_depth.shape} vs {plane_mask.shape}"
            )

        uv, proj_depth = project_vertices_to_view(mesh_vertices, spec)

        height, width = plane_mask.shape
        u = np.rint(uv[:, 0]).astype(np.int32)
        v = np.rint(uv[:, 1]).astype(np.int32)

        in_image = (
            (proj_depth > 0.0)
            & (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )
        if not np.any(in_image):
            continue

        visible_idx = np.where(in_image)[0]
        sampled_render_depth = render_depth[v[visible_idx], u[visible_idx]]
        valid_render_depth = sampled_render_depth > 0.0
        if not np.any(valid_render_depth):
            continue

        visible_idx = visible_idx[valid_render_depth]
        sampled_render_depth = sampled_render_depth[valid_render_depth]
        sampled_u = u[visible_idx]
        sampled_v = v[visible_idx]
        sampled_proj_depth = proj_depth[visible_idx]

        relative_diff = np.abs(sampled_proj_depth - sampled_render_depth) / np.maximum(
            sampled_render_depth,
            1e-6,
        )
        depth_valid = relative_diff <= float(depth_rel_tol)
        if not np.any(depth_valid):
            continue

        valid_idx = visible_idx[depth_valid]
        valid_u = sampled_u[depth_valid]
        valid_v = sampled_v[depth_valid]

        vertex_visible_views[valid_idx] += 1

        plane_ids = plane_mask[valid_v, valid_u].astype(np.int32)
        plane_valid = plane_ids > 0
        if not np.any(plane_valid):
            continue

        assign_idx = valid_idx[plane_valid]
        assign_plane_ids = plane_ids[plane_valid]
        vertex_plane_views[assign_idx] += 1

        for vidx, pid in zip(assign_idx.tolist(), assign_plane_ids.tolist()):
            vertex_votes[vidx][int(pid)] += 1

    return vertex_votes, vertex_visible_views, vertex_plane_views


def build_plane_conflict_graph(vertex_votes):
    pair_counter = collections.Counter()
    plane_vertex_support = collections.Counter()

    for vote_counter in vertex_votes:
        if not vote_counter:
            continue

        plane_ids = sorted(int(pid) for pid in vote_counter.keys() if int(pid) > 0)
        for pid in plane_ids:
            plane_vertex_support[pid] += 1

        if len(plane_ids) < 2:
            continue

        for a, b in itertools.combinations(plane_ids, 2):
            pair_counter[(int(a), int(b))] += 1

    return pair_counter, plane_vertex_support


def build_plane_remap(
    vertex_votes,
    global_plane_info,
    merge_normal_angle_deg=5.0,
    merge_plane_dist_thresh=0.03,
    merge_min_shared_vertices=128,
    merge_min_shared_ratio=0.10,
):
    pair_counter, plane_vertex_support = build_plane_conflict_graph(vertex_votes)
    all_plane_ids = sorted(plane_vertex_support.keys())
    uf = UnionFind(all_plane_ids)

    cos_thresh = float(np.cos(np.deg2rad(float(merge_normal_angle_deg))))
    merged_pairs = []

    for (a, b), shared_vertices in pair_counter.items():
        if shared_vertices < int(merge_min_shared_vertices):
            continue
        if a not in global_plane_info or b not in global_plane_info:
            continue

        support_base = min(plane_vertex_support[a], plane_vertex_support[b])
        if support_base <= 0:
            continue
        shared_ratio = float(shared_vertices) / float(support_base)
        if shared_ratio < float(merge_min_shared_ratio):
            continue

        info_a = global_plane_info[a]
        info_b = global_plane_info[b]

        normal_a = info_a["normal"]
        normal_b = info_b["normal"]
        cos_sim = float(np.abs(np.dot(normal_a, normal_b)))
        if cos_sim < cos_thresh:
            continue

        mean_a = info_a["mean"]
        mean_b = info_b["mean"]
        d_a = float(info_a["plane_params"][3])
        d_b = float(info_b["plane_params"][3])

        dist_ab = abs(float(np.dot(normal_a, mean_b) + d_a))
        dist_ba = abs(float(np.dot(normal_b, mean_a) + d_b))
        sym_plane_dist = max(dist_ab, dist_ba)

        if sym_plane_dist > float(merge_plane_dist_thresh):
            continue

        uf.union(a, b)
        merged_pairs.append(
            {
                "plane_a": int(a),
                "plane_b": int(b),
                "shared_vertices": int(shared_vertices),
                "shared_ratio": float(shared_ratio),
                "cos_sim": float(cos_sim),
                "sym_plane_dist": float(sym_plane_dist),
            }
        )

    groups = collections.defaultdict(list)
    for pid in all_plane_ids:
        groups[uf.find(pid)].append(pid)

    remap = {}
    for _, group in groups.items():
        canonical = max(
            group,
            key=lambda pid: (plane_vertex_support.get(pid, 0), -int(pid)),
        )
        for pid in group:
            remap[int(pid)] = int(canonical)

    return remap, merged_pairs, plane_vertex_support


def build_merged_global_plane_info(global_plane_info, plane_remap, plane_vertex_support):
    grouped = collections.defaultdict(list)
    for pid, info in global_plane_info.items():
        canonical = int(plane_remap.get(pid, pid))
        grouped[canonical].append((pid, info))

    merged_info = {}
    for canonical, items in grouped.items():
        normals = []
        means = []
        weights = []

        for pid, info in items:
            normals.append(info["normal"])
            means.append(info["mean"])
            weights.append(max(1, int(plane_vertex_support.get(pid, info.get("num_views", 1)))))

        normal = oriented_average_normals(normals)
        mean = np.average(
            np.stack(means, axis=0),
            axis=0,
            weights=np.asarray(weights, dtype=np.float32),
        ).astype(np.float32)
        d = -float(np.dot(normal, mean))

        merged_info[canonical] = {
            "normal": normal.astype(np.float32),
            "mean": mean.astype(np.float32),
            "plane_params": np.array([normal[0], normal[1], normal[2], d], dtype=np.float32),
        }

    return merged_info


def resolve_vertex_plane_ids(
    vertex_votes,
    plane_remap=None,
    min_votes=1,
    min_vote_ratio=0.5,
):
    num_vertices = len(vertex_votes)
    vertex_to_plane = np.zeros(num_vertices, dtype=np.int32)
    vote_confidence = np.zeros(num_vertices, dtype=np.float32)

    for vidx, vote_counter in enumerate(tqdm(vertex_votes, desc="Resolve vertex plane IDs")):
        if not vote_counter:
            continue

        if plane_remap is None:
            merged_counter = vote_counter
        else:
            merged_counter = collections.Counter()
            for pid, count in vote_counter.items():
                canonical = int(plane_remap.get(int(pid), int(pid)))
                merged_counter[canonical] += int(count)

        best_pid, best_votes = merged_counter.most_common(1)[0]
        total_votes = int(sum(merged_counter.values()))
        best_ratio = float(best_votes) / max(total_votes, 1)

        if best_votes >= int(min_votes) and best_ratio >= float(min_vote_ratio):
            vertex_to_plane[vidx] = int(best_pid)
            vote_confidence[vidx] = float(best_ratio)

    return vertex_to_plane, vote_confidence


def assign_vertex_planes_from_global_equations(
    mesh_v,
    vertex_to_plane,
    global_plane_info,
    dist_thresh,
):
    unassigned = np.where(vertex_to_plane == 0)[0]
    if len(unassigned) == 0:
        return vertex_to_plane

    plane_ids = []
    normals = []
    ds = []
    for pid, info in sorted(global_plane_info.items()):
        plane_params = np.asarray(info["plane_params"], dtype=np.float64)
        n_raw = plane_params[:3]
        d_raw = float(plane_params[3])
        norm = np.linalg.norm(n_raw)
        if norm < 1e-8:
            continue
        plane_ids.append(int(pid))
        normals.append(n_raw / norm)
        ds.append(d_raw / norm)

    if len(plane_ids) == 0:
        return vertex_to_plane

    plane_ids = np.asarray(plane_ids, dtype=np.int32)
    normals = np.stack(normals, axis=0)
    ds = np.asarray(ds, dtype=np.float64)

    points = mesh_v[unassigned].astype(np.float64)
    dists = np.abs(points @ normals.T + ds[None, :])
    best_j = np.argmin(dists, axis=1)
    best_dist = dists[np.arange(len(points)), best_j]
    valid = best_dist < float(dist_thresh)

    if np.any(valid):
        vertex_to_plane[unassigned[valid]] = plane_ids[best_j[valid]]

    return vertex_to_plane


def save_colored_mesh(mesh, vertex_plane_ids, path):
    color_mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int32),
        vertex_colors=vertex_colors_from_plane_ids(vertex_plane_ids),
        process=False,
    )
    color_mesh.export(path)


def refine_mesh_layout_from_vertex_planes(
    init_mesh,
    vertex_to_plane,
    voxel_size=0.01,
):
    mesh_v = np.asarray(init_mesh.vertices, dtype=np.float32)
    mesh_f = np.asarray(init_mesh.faces, dtype=np.int32)

    unique_planes = np.unique(vertex_to_plane)
    unique_planes = unique_planes[unique_planes != 0]
    unique_planes = np.sort(unique_planes)

    print(f"Planes with assigned vertices: {len(unique_planes)}")
    print("Computing mesh edges...")
    edges = init_mesh.edges

    new_vertices_list = [mesh_v]
    final_plane_id_parts = [vertex_to_plane.copy()]
    current_total_v = len(mesh_v)
    all_new_faces = []

    for idx, plane_id in enumerate(unique_planes):
        print(f"[Plane {idx + 1}/{len(unique_planes)}] Processing Plane ID: {plane_id}")
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
            outlier_indices = plane_v_indices[outlier_mask]
            vertex_to_plane[outlier_indices] = 0
            plane_v_indices = plane_v_indices[~outlier_mask]
            print(f"  Filtered {int(np.sum(outlier_mask))} outlier vertices.")

        if len(plane_v_indices) < 3:
            continue

        v0_in_plane = vertex_to_plane[edges[:, 0]] == plane_id
        v1_in_plane = vertex_to_plane[edges[:, 1]] == plane_id
        is_boundary_edge = v0_in_plane ^ v1_in_plane

        b_v_indices = np.unique(
            np.concatenate(
                [
                    edges[is_boundary_edge & v0_in_plane, 0],
                    edges[is_boundary_edge & v1_in_plane, 1],
                ]
            )
        )
        if len(b_v_indices) < 3:
            print("  Not enough boundary vertices, skipping.")
            continue

        interior_v_indices = np.setdiff1d(plane_v_indices, b_v_indices)
        pts = mesh_v[plane_v_indices]
        center = pts.mean(axis=0)
        cov = np.cov(pts.T)
        _, eigenvectors = np.linalg.eigh(cov)
        u, v = eigenvectors[:, 2], eigenvectors[:, 1]

        def project_2d(points_3d):
            rel = points_3d - center
            return np.stack([rel @ u, rel @ v], axis=1)

        v_b_2d = project_2d(mesh_v[b_v_indices])
        v_i_2d = project_2d(mesh_v[interior_v_indices]) if len(interior_v_indices) > 0 else np.zeros((0, 2), dtype=np.float32)
        all_v_2d = np.concatenate([v_b_2d, v_i_2d], axis=0)

        rect = cv2.minAreaRect(all_v_2d.astype(np.float32))
        box = cv2.boxPoints(rect)
        grid_dist = voxel_size * 2.0

        x_min, y_min = np.min(box, axis=0)
        x_max, y_max = np.max(box, axis=0)

        nx = max(1, int((x_max - x_min) / max(grid_dist, 1e-8)))
        ny = max(1, int((y_max - y_min) / max(grid_dist, 1e-8)))
        if nx * ny > 500_000:
            grid_dist = max((x_max - x_min), (y_max - y_min)) / 400.0

        grid_x, grid_y = np.meshgrid(
            np.arange(x_min, x_max, grid_dist),
            np.arange(y_min, y_max, grid_dist),
        )
        grid_pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

        hull = cv2.convexHull(all_v_2d.astype(np.float32))
        inside_mask = np.array(
            [cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), False) >= 0 for p in grid_pts],
            dtype=bool,
        )
        grid_pts_filtered = grid_pts[inside_mask]

        combined_2d = np.concatenate([v_b_2d, grid_pts_filtered], axis=0)
        if len(combined_2d) < 3:
            continue

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

        for tri_idx in tri.simplices:
            all_new_faces.append(
                [
                    index_map[int(tri_idx[0])],
                    index_map[int(tri_idx[1])],
                    index_map[int(tri_idx[2])],
                ]
            )

    face_v_planes = vertex_to_plane[mesh_f]
    is_planar_face = (
        (face_v_planes[:, 0] == face_v_planes[:, 1])
        & (face_v_planes[:, 1] == face_v_planes[:, 2])
        & (face_v_planes[:, 0] != 0)
    )

    final_v = np.concatenate(new_vertices_list, axis=0)
    final_plane_ids = np.concatenate(final_plane_id_parts, axis=0)
    final_f = np.concatenate(
        [
            mesh_f[~is_planar_face],
            np.asarray(all_new_faces, dtype=np.int32),
        ],
        axis=0,
    )

    refined_mesh = trimesh.Trimesh(
        vertices=final_v,
        faces=final_f,
        vertex_colors=vertex_colors_from_plane_ids(final_plane_ids),
        process=False,
    )
    return refined_mesh, final_plane_ids


def dump_merge_debug(path, plane_remap, merged_pairs):
    payload = {
        "plane_remap": {str(k): int(v) for k, v in sorted(plane_remap.items())},
        "merged_pairs": merged_pairs,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_default_paths(data_root):
    mesh_root = os.path.join(data_root, "tsdf_refine_bg")
    merge_root = os.path.join(data_root, "merge_3d_plane")
    return {
        "camera_json_path": os.path.join(data_root, "camera_params.json"),
        "global_plane_mask_root": merge_root,
        "global_plane_info_path": os.path.join(merge_root, "global_plane_info.json"),
        "mesh_path": os.path.join(mesh_root, "tsdf_refine_bg_post.ply"),
        "save_vertex_plane_mesh_path": os.path.join(mesh_root, "mesh_vertex_planes_by_proj.ply"),
        "save_refined_mesh_path": os.path.join(mesh_root, "refined_mesh_simplified_by_proj.ply"),
        "save_merge_debug_path": os.path.join(mesh_root, "merged_plane_debug.json"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Mesh plane assignment by multi-view projection.")
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.path.join("outputs", "bg", "scan1", "see3d_guidance_views"),
    )
    parser.add_argument("--camera_json_path", type=str, default=None)
    parser.add_argument("--global_plane_mask_root", type=str, default=None)
    parser.add_argument("--global_plane_info_path", type=str, default=None)
    parser.add_argument("--mesh_path", type=str, default=None)
    parser.add_argument("--save_vertex_plane_mesh_path", type=str, default=None)
    parser.add_argument("--save_refined_mesh_path", type=str, default=None)
    parser.add_argument("--save_merge_debug_path", type=str, default=None)

    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--depth_rel_tol", type=float, default=1e-3)
    parser.add_argument("--min_votes", type=int, default=1)
    parser.add_argument("--min_vote_ratio", type=float, default=0.5)

    parser.add_argument("--disable_plane_merge", action="store_true")
    parser.add_argument("--merge_normal_angle_deg", type=float, default=5.0)
    parser.add_argument("--merge_plane_dist_thresh", type=float, default=0.03)
    parser.add_argument("--merge_min_shared_vertices", type=int, default=128)
    parser.add_argument("--merge_min_shared_ratio", type=float, default=0.10)

    parser.add_argument("--fill_unassigned_from_plane", action="store_true")
    parser.add_argument("--plane_equation_dist_thresh", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    defaults = build_default_paths(args.data_root)

    camera_json_path = args.camera_json_path or defaults["camera_json_path"]
    global_plane_mask_root = args.global_plane_mask_root or defaults["global_plane_mask_root"]
    global_plane_info_path = args.global_plane_info_path or defaults["global_plane_info_path"]
    mesh_path = args.mesh_path or defaults["mesh_path"]
    save_vertex_plane_mesh_path = args.save_vertex_plane_mesh_path or defaults["save_vertex_plane_mesh_path"]
    save_refined_mesh_path = args.save_refined_mesh_path or defaults["save_refined_mesh_path"]
    save_merge_debug_path = args.save_merge_debug_path or defaults["save_merge_debug_path"]

    os.makedirs(os.path.dirname(save_vertex_plane_mesh_path), exist_ok=True)

    print("Loading camera specs...")
    camera_specs = load_selected_camera_params(camera_json_path, keep_metadata=True)

    print("Loading mesh...")
    init_mesh = trimesh.load(mesh_path, process=False)
    mesh_vertices = np.asarray(init_mesh.vertices, dtype=np.float32)

    print("Loading global plane info...")
    global_plane_info = load_global_plane_info(global_plane_info_path)

    print("Creating mesh depth renderer...")
    depth_renderer = MeshDepthRenderer(init_mesh)

    print("Collecting per-vertex votes from projected global plane masks...")
    vertex_votes, vertex_visible_views, vertex_plane_views = collect_vertex_votes_from_views(
        mesh_vertices=mesh_vertices,
        camera_specs=camera_specs,
        global_plane_mask_root=global_plane_mask_root,
        depth_renderer=depth_renderer,
        depth_rel_tol=args.depth_rel_tol,
    )

    if args.disable_plane_merge:
        plane_remap = {}
        merged_pairs = []
        plane_vertex_support = collections.Counter()
        merged_plane_info = global_plane_info
    else:
        print("Building plane merge remap from vote conflicts...")
        plane_remap, merged_pairs, plane_vertex_support = build_plane_remap(
            vertex_votes=vertex_votes,
            global_plane_info=global_plane_info,
            merge_normal_angle_deg=args.merge_normal_angle_deg,
            merge_plane_dist_thresh=args.merge_plane_dist_thresh,
            merge_min_shared_vertices=args.merge_min_shared_vertices,
            merge_min_shared_ratio=args.merge_min_shared_ratio,
        )
        merged_plane_info = build_merged_global_plane_info(
            global_plane_info=global_plane_info,
            plane_remap=plane_remap,
            plane_vertex_support=plane_vertex_support,
        )
        dump_merge_debug(save_merge_debug_path, plane_remap, merged_pairs)

    print("Resolving final vertex plane IDs...")
    vertex_to_plane, vote_confidence = resolve_vertex_plane_ids(
        vertex_votes=vertex_votes,
        plane_remap=plane_remap if len(plane_remap) > 0 else None,
        min_votes=args.min_votes,
        min_vote_ratio=args.min_vote_ratio,
    )

    if args.fill_unassigned_from_plane:
        plane_dist_thresh = (
            3.0 * float(args.delta)
            if args.plane_equation_dist_thresh is None
            else float(args.plane_equation_dist_thresh)
        )
        vertex_to_plane = assign_vertex_planes_from_global_equations(
            mesh_v=mesh_vertices,
            vertex_to_plane=vertex_to_plane,
            global_plane_info=merged_plane_info,
            dist_thresh=plane_dist_thresh,
        )

    assigned = int((vertex_to_plane > 0).sum())
    total = int(len(vertex_to_plane))
    print(f"Assigned vertices: {assigned} / {total} ({100.0 * assigned / max(total, 1):.2f}%)")

    print(f"Saving vertex-plane debug mesh to {save_vertex_plane_mesh_path}")
    save_colored_mesh(init_mesh, vertex_to_plane, save_vertex_plane_mesh_path)

    print("Refining mesh layout from projected plane IDs...")
    refined_mesh, _ = refine_mesh_layout_from_vertex_planes(
        init_mesh=init_mesh,
        vertex_to_plane=vertex_to_plane.copy(),
        voxel_size=args.delta,
    )

    print(f"Saving refined mesh to {save_refined_mesh_path}")
    refined_mesh.export(save_refined_mesh_path)
    print("Done.")


if __name__ == "__main__":
    main()
