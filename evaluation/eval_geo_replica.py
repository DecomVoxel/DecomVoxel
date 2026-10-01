import argparse
import glob
import json
import os
import re

import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import trimesh


def nn_correspondance(verts1, verts2):
	if len(verts1) == 0 or len(verts2) == 0:
		return [], []
	kdtree = KDTree(verts1)
	distances, indices = kdtree.query(verts2)
	return distances.reshape(-1), indices


def evaluate(mesh_pred, mesh_trgt, threshold=0.05, down_sample=0.02):
	pcd_trgt = o3d.geometry.PointCloud()
	pcd_pred = o3d.geometry.PointCloud()
	pcd_trgt.points = o3d.utility.Vector3dVector(mesh_trgt.vertices[:, :3])
	pcd_pred.points = o3d.utility.Vector3dVector(mesh_pred.vertices[:, :3])

	if down_sample:
		pcd_pred = pcd_pred.voxel_down_sample(down_sample)
		pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)

	verts_pred = np.asarray(pcd_pred.points)
	verts_trgt = np.asarray(pcd_trgt.points)

	dist1, _ = nn_correspondance(verts_pred, verts_trgt)
	dist2, _ = nn_correspondance(verts_trgt, verts_pred)

	precision = np.mean((dist2 < threshold).astype(float))
	recal = np.mean((dist1 < threshold).astype(float))
	den = precision + recal
	fscore = 2 * precision * recal / den if den > 0 else 0.0

	n = 200000
	pointcloud_pred, idx = mesh_pred.sample(n, return_index=True)
	pointcloud_pred = pointcloud_pred.astype(np.float32)
	normal_pred = mesh_pred.face_normals[idx]

	pointcloud_trgt, idx = mesh_trgt.sample(n, return_index=True)
	pointcloud_trgt = pointcloud_trgt.astype(np.float32)
	normal_trgt = mesh_trgt.face_normals[idx]

	_, index1 = nn_correspondance(pointcloud_pred, pointcloud_trgt)
	_, index2 = nn_correspondance(pointcloud_trgt, pointcloud_pred)

	normal_acc = np.abs((normal_pred * normal_trgt[index2.reshape(-1)]).sum(axis=-1)).mean()
	normal_comp = np.abs((normal_trgt * normal_pred[index1.reshape(-1)]).sum(axis=-1)).mean()
	normal_consistency = 0.5 * (normal_acc + normal_comp)

	return {
		"Acc": np.mean(dist2),
		"Comp": np.mean(dist1),
		"Chamfer-L1": 0.5 * (np.mean(dist2) + np.mean(dist1)),
		"Prec": precision,
		"Recal": recal,
		"F-score": fscore,
		"Normal-Acc": normal_acc,
		"Normal-Comp": normal_comp,
		"Normal-Consistency": normal_consistency,
	}


def load_mesh(path):
	if not os.path.exists(path):
		raise FileNotFoundError(f"Mesh file not found: {path}")
	mesh = trimesh.load(path, process=False)
	if isinstance(mesh, trimesh.Scene):
		mesh = mesh.dump(concatenate=True)
	return mesh


def merge_mesh_paths(mesh_paths):
	if len(mesh_paths) == 0:
		raise FileNotFoundError("No mesh paths provided for merging")
	meshes = [load_mesh(p) for p in mesh_paths]
	if len(meshes) == 1:
		return meshes[0]
	return trimesh.util.concatenate(meshes)


def build_pred_mesh(exp_path, mode, objects_name, background_name):
	if mode == "objects_only":
		return load_mesh(os.path.join(exp_path, objects_name))
	if mode == "background_only":
		return load_mesh(os.path.join(exp_path, background_name))
	obj_mesh = load_mesh(os.path.join(exp_path, objects_name))
	bg_mesh = load_mesh(os.path.join(exp_path, background_name))
	return trimesh.util.concatenate([obj_mesh, bg_mesh])


def _extract_num(text, pattern):
	m = re.search(pattern, text)
	if m is None:
		return None
	return int(m.group(1))


def _extract_object_id(text):
	# Support both obj_### and object_### naming styles.
	return _extract_num(text, r"(?:obj|object)_(\d+)")


def _list_gt_object_ply_paths(scan_gt_dir):
	"""Return GT per-object PLYs (obj_<num>.ply), excluding obj_<num>_colored.ply etc."""
	candidates = glob.glob(os.path.join(scan_gt_dir, "obj_*.ply"))
	filtered = []
	for p in candidates:
		name = os.path.basename(p)
		if re.fullmatch(r"obj_\d+\.ply", name) is None:
			continue
		filtered.append(p)
	return filtered


def _build_graph_children(scene):
	children = {}
	parents = {}
	try:
		edges = scene.graph.to_edgelist()
	except Exception:
		edges = []

	for edge in edges:
		if len(edge) < 2:
			continue
		parent = edge[0]
		child = edge[1]
		children.setdefault(parent, []).append(child)
		parents.setdefault(child, set()).add(parent)

	for node in list(children.keys()):
		children[node] = sorted(set(children[node]))

	all_nodes = set(scene.graph.nodes)
	root_candidates = sorted([n for n in all_nodes if n not in parents])
	return children, root_candidates


def debug_print_scene_graph(scene, scene_path):
	print(f"[DEBUG] scene graph for: {scene_path}")

	# Flat listing first so users can grep quickly.
	print(f"[DEBUG] total graph nodes: {len(list(scene.graph.nodes))}")
	for node_name in sorted(scene.graph.nodes):
		geom_name = None
		try:
			_, geom_name = scene.graph.get(node_name)
		except Exception:
			pass
		print(f"[DEBUG] node={node_name} geom={geom_name}")

	children, roots = _build_graph_children(scene)
	if not roots:
		roots = sorted(scene.graph.nodes)

	print("[DEBUG] recursive tree:")
	visited = set()

	def _dfs(node, depth):
		indent = "  " * depth
		geom_name = None
		try:
			_, geom_name = scene.graph.get(node)
		except Exception:
			pass
		print(f"[DEBUG] {indent}- {node} (geom={geom_name})")
		if node in visited:
			print(f"[DEBUG] {indent}  (visited)")
			return
		visited.add(node)
		for c in children.get(node, []):
			_dfs(c, depth + 1)

	for r in roots:
		_dfs(r, 0)


def load_pred_object_meshes(scene_path, debug=False):
	if not os.path.exists(scene_path):
		raise FileNotFoundError(f"Pred object scene file not found: {scene_path}")

	loaded = trimesh.load(scene_path, process=False)
	if isinstance(loaded, trimesh.Trimesh):
		raise ValueError(f"Expected a scene with multiple objects in {scene_path}, but got a single mesh")
	if not isinstance(loaded, trimesh.Scene):
		raise ValueError(f"Unsupported pred scene type for {scene_path}: {type(loaded)}")

	if debug:
		debug_print_scene_graph(loaded, scene_path)

	entries = []
	for node_name in loaded.graph.nodes:
		geom_name = None
		node_tf = None
		try:
			node_tf, geom_name = loaded.graph.get(node_name)
		except Exception:
			pass

		obj_num = _extract_object_id(str(node_name))
		if obj_num is None and geom_name is not None:
			obj_num = _extract_object_id(str(geom_name))
		if obj_num is None:
			continue

		if geom_name is None or geom_name not in loaded.geometry:
			continue

		base_mesh = loaded.geometry[geom_name]
		if base_mesh is None or len(base_mesh.vertices) == 0:
			continue

		submesh = base_mesh.copy()
		if node_tf is not None:
			try:
				submesh.apply_transform(node_tf)
			except Exception:
				# Keep untransformed mesh if transform application fails.
				pass

		entries.append({
			"obj_num": obj_num,
			"pred_node": node_name,
			"pred_geom": geom_name,
			"mesh": submesh,
		})

	if not entries:
		debug_print_scene_graph(loaded, scene_path)
		raise FileNotFoundError(
			f"No object nodes found in {scene_path}. Checked node/geometry names for obj_### and object_### patterns"
		)

	entries.sort(key=lambda x: x["obj_num"])
	return entries


def load_gt_object_meshes(gtmesh_root, scan_name):
	scan_gt_dir = os.path.join(gtmesh_root, scan_name)
	if not os.path.isdir(scan_gt_dir):
		raise FileNotFoundError(f"GT scan dir not found: {scan_gt_dir}")

	obj_paths = sorted(
		_list_gt_object_ply_paths(scan_gt_dir),
		key=lambda p: _extract_num(os.path.basename(p), r"obj_(\d+)"),
	)
	if not obj_paths:
		raise FileNotFoundError(f"No object GT meshes found in {scan_gt_dir} (expected obj_*.ply)")

	entries = []
	for p in obj_paths:
		gt_num = _extract_num(os.path.basename(p), r"obj_(\d+)")
		entries.append({
			"gt_num": gt_num,
			"gt_file": os.path.basename(p),
			"gt_path": p,
			"mesh": load_mesh(p),
		})
	return entries


def load_scene_graph_object_ids(scan_path, scene_graph_name="scene_graph.json"):
	# Try the preferred file first, then fall back to other files that also use
	# object ids as their top-level keys.
	candidate_names = [scene_graph_name, "object_categories.json", "obj_prompt.json"]
	# Preserve order, drop duplicates (e.g. when caller passes one of the fallbacks).
	seen = set()
	candidates = []
	for n in candidate_names:
		if n and n not in seen:
			seen.add(n)
			candidates.append(n)

	scene_graph_path = None
	for n in candidates:
		p = os.path.join(scan_path, n)
		if os.path.exists(p):
			scene_graph_path = p
			break
	if scene_graph_path is None:
		raise FileNotFoundError(
			f"Scene graph file not found in {scan_path}; tried: {candidates}"
		)

	with open(scene_graph_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	if not isinstance(data, dict):
		raise ValueError(f"Scene graph must be a JSON object: {scene_graph_path}")

	object_ids = []
	for k in data.keys():
		try:
			object_ids.append(int(k))
		except (TypeError, ValueError):
			raise ValueError(f"Invalid object id key '{k}' in {scene_graph_path}")

	if not object_ids:
		raise ValueError(f"No object ids found in scene graph: {scene_graph_path}")

	return object_ids, scene_graph_path


def evaluate_objects_only(exp_path, objects_name, gtmesh_root, scan_name, scan_path, threshold, down_sample, debug_object_nodes=False):
	pred_scene_path = os.path.join(exp_path, objects_name)
	pred_entries = load_pred_object_meshes(pred_scene_path, debug=debug_object_nodes)
	gt_entries = load_gt_object_meshes(gtmesh_root, scan_name)
	scene_object_ids, scene_graph_path = load_scene_graph_object_ids(scan_path)

	pred_by_obj_num = {e["obj_num"]: e for e in pred_entries}
	if len(scene_object_ids) < len(gt_entries):
		raise ValueError(
			f"Scene graph object ids are fewer than GT objects for scan {scan_name}: "
			f"scene_graph={len(scene_object_ids)}, gt={len(gt_entries)} ({scene_graph_path})"
			f"gt object ids: {[e['gt_num'] for e in gt_entries]}"
		)

	per_object_results = []
	missing_pred = []
	for gt_e in gt_entries:
		gt_idx = int(gt_e["gt_num"])
		if gt_idx <= 0 or gt_idx > len(scene_object_ids):
			continue

		pred_obj_num = int(scene_object_ids[gt_idx - 1])
		pred_e = pred_by_obj_num.get(pred_obj_num)
		if pred_e is None:
			missing_pred.append((gt_idx, pred_obj_num, gt_e["gt_file"]))
			continue

		obj_metrics = evaluate(
			mesh_pred=pred_e["mesh"],
			mesh_trgt=gt_e["mesh"],
			threshold=threshold,
			down_sample=down_sample,
		)
		per_object_results.append({
			"object_index": gt_idx,
			"pred_node": pred_e["pred_node"],
			"pred_geom": pred_e["pred_geom"],
			"pred_obj_num": pred_e["obj_num"],
			"mapped_pred_obj_num": pred_obj_num,
			"gt_file": gt_e["gt_file"],
			"gt_obj_num": gt_e["gt_num"],
			"metrics": obj_metrics,
		})

	if missing_pred:
		for gt_idx, pred_obj_num, gt_file in missing_pred:
			print(
				"[WARN] Missing predicted object for GT "
				f"obj_{gt_idx} ({gt_file}), expected pred object id {pred_obj_num} from scene graph"
			)

	if not per_object_results:
		raise ValueError(
			f"No matched GT/pred objects found for scan {scan_name}. "
			f"Pred object ids: {sorted(pred_by_obj_num.keys())}, scene graph: {scene_graph_path}"
		)

	metric_keys = list(per_object_results[0]["metrics"].keys())
	scene_metrics = {}
	for k in metric_keys:
		scene_metrics[k] = float(np.mean([r["metrics"][k] for r in per_object_results]))

	return scene_metrics, per_object_results


def build_gt_mesh_from_replica(gtmesh_root, scan_name, mode):
	scan_gt_dir = os.path.join(gtmesh_root, scan_name)
	if not os.path.isdir(scan_gt_dir):
		raise FileNotFoundError(f"GT scan dir not found: {scan_gt_dir}")

	obj_paths = sorted(_list_gt_object_ply_paths(scan_gt_dir))
	bg_path = os.path.join(scan_gt_dir, "bg_repair.ply")

	if mode == "objects_only":
		if not obj_paths:
			raise FileNotFoundError(f"No object GT meshes found in {scan_gt_dir} (expected obj_*.ply)")
		return merge_mesh_paths(obj_paths)

	if mode == "background_only":
		return load_mesh(bg_path)

	# merged mode: objects + background
	parts = []
	if obj_paths:
		parts.extend(obj_paths)
	if os.path.exists(bg_path):
		parts.append(bg_path)
	if not parts:
		raise FileNotFoundError(f"No GT meshes found in {scan_gt_dir} (expected obj_*.ply and/or bg_repair.ply)")
	return merge_mesh_paths(parts)


def export_gt_object_mesh(gtmesh_root, scan_name, output_path):
	scan_gt_dir = os.path.join(gtmesh_root, scan_name)
	obj_paths = sorted(_list_gt_object_ply_paths(scan_gt_dir))
	if not obj_paths:
		raise FileNotFoundError(f"No object GT meshes found in {scan_gt_dir} (expected obj_*.ply)")

	obj_mesh = merge_mesh_paths(obj_paths)
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	obj_mesh.export(output_path)
	return output_path


def _parse_metrics_txt(metrics_path):
	metrics = {}
	with open(metrics_path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line or ":" not in line:
				continue
			k, v = line.split(":", 1)
			k = k.strip()
			v = v.strip()
			try:
				metrics[k] = float(v)
			except ValueError:
				metrics[k] = v
	return metrics


def _parse_objects_only_metrics_txt(metrics_path):
	scene_metrics = {}
	object_map = {}
	with open(metrics_path, "r", encoding="utf-8") as f:
		for raw in f:
			line = raw.strip()
			if not line or ":" not in line:
				continue
			k, v = line.split(":", 1)
			k = k.strip()
			v = v.strip()
			m = re.match(r"obj_(\d+)\.(.+)", k)
			if m is None:
				try:
					scene_metrics[k] = float(v)
				except ValueError:
					scene_metrics[k] = v
				continue

			obj_idx = int(m.group(1))
			subkey = m.group(2)
			if obj_idx not in object_map:
				object_map[obj_idx] = {}
			try:
				object_map[obj_idx][subkey] = float(v)
			except ValueError:
				object_map[obj_idx][subkey] = v

	objects = []
	for obj_idx in sorted(object_map.keys()):
		objects.append((obj_idx, object_map[obj_idx]))
	return scene_metrics, objects


def _fmt_metric(v):
	if isinstance(v, (float, np.floating)):
		return f"{v:.6f}"
	return str(v)


def _is_number(v):
	return isinstance(v, (int, float, np.integer, np.floating))


def generate_overview_md(exps_replica_root, filename="metrics_merged.txt", output_name="overview_metrics_merged.md"):
	pattern = os.path.join(exps_replica_root, "*", filename)
	metric_files = sorted(glob.glob(pattern))
	if not metric_files:
		return None

	rows = []
	all_keys = set()
	all_object_rows = []
	for fp in metric_files:
		scan_name = os.path.basename(os.path.dirname(fp))
		if filename == "metrics_objects_only.txt":
			scene_metrics, object_rows = _parse_objects_only_metrics_txt(fp)
			rows.append((scan_name, scene_metrics, fp))
			all_keys.update(scene_metrics.keys())
			for obj_idx, obj_data in object_rows:
				all_object_rows.append((scan_name, obj_idx, obj_data, fp))
		else:
			metrics = _parse_metrics_txt(fp)
			rows.append((scan_name, metrics, fp))
			all_keys.update(metrics.keys())

	priority = [
		"Acc", "Comp", "Chamfer-L1", "Prec", "Recal", "F-score",
		"Normal-Acc", "Normal-Comp", "Normal-Consistency",
	]
	if "ObjectCount" in all_keys:
		priority.append("ObjectCount")
	ordered_keys = [k for k in priority if k in all_keys] + sorted(k for k in all_keys if k not in priority)

	def _scan_sort_key(x):
		s = x[0]
		num = "".join(ch for ch in s if ch.isdigit())
		return (int(num) if num else 10**9, s)

	rows.sort(key=_scan_sort_key)

	avg_metrics = {}
	for k in ordered_keys:
		vals = []
		for _, metrics, _ in rows:
			v = metrics.get(k, None)
			if _is_number(v):
				vals.append(float(v))
		if vals:
			avg_metrics[k] = float(np.mean(vals))

	overview_path = os.path.join(exps_replica_root, output_name)
	overview_tag = os.path.splitext(filename)[0]
	with open(overview_path, "w", encoding="utf-8") as f:
		f.write(f"# Replica Evaluation Overview ({overview_tag})\n\n")
		f.write(f"Scanned files: `{pattern}`\n\n")

		f.write("## Scene-Level Metrics\n\n")
		headers = ["Scan"] + ordered_keys + ["File"]
		f.write("| " + " | ".join(headers) + " |\n")
		f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")

		for scan_name, metrics, fp in rows:
			vals = [scan_name]
			for k in ordered_keys:
				vals.append(_fmt_metric(metrics.get(k, "")))
			vals.append(os.path.relpath(fp, exps_replica_root))
			f.write("| " + " | ".join(vals) + " |\n")

		avg_vals = ["Dataset-Average"]
		for k in ordered_keys:
			avg_vals.append(_fmt_metric(avg_metrics.get(k, "")))
		avg_vals.append("-")
		f.write("| " + " | ".join(avg_vals) + " |\n")

		if filename == "metrics_objects_only.txt" and all_object_rows:
			obj_metric_keys = set()
			for _, _, obj_data, _ in all_object_rows:
				for k, v in obj_data.items():
					if _is_number(v):
						obj_metric_keys.add(k)

			ordered_obj_metric_keys = [k for k in priority if k in obj_metric_keys] + sorted(
				k for k in obj_metric_keys if k not in priority
			)

			f.write("\n## Object-Level Metrics\n\n")
			obj_headers = ["Scan", "Object", "PredNode", "GTFile"] + ordered_obj_metric_keys + ["File"]
			f.write("| " + " | ".join(obj_headers) + " |\n")
			f.write("| " + " | ".join(["---"] * len(obj_headers)) + " |\n")

			all_object_rows.sort(key=lambda x: (_scan_sort_key((x[0], {}, "")), x[1]))
			obj_avg = {}
			for k in ordered_obj_metric_keys:
				vals = [
					float(obj_data[k])
					for _, _, obj_data, _ in all_object_rows
					if _is_number(obj_data.get(k, None))
				]
				if vals:
					obj_avg[k] = float(np.mean(vals))

			for scan_name, obj_idx, obj_data, fp in all_object_rows:
				vals = [
					scan_name,
					f"obj_{obj_idx}",
					str(obj_data.get("PredNode", "")),
					str(obj_data.get("GTFile", "")),
				]
				for k in ordered_obj_metric_keys:
					vals.append(_fmt_metric(obj_data.get(k, "")))
				vals.append(os.path.relpath(fp, exps_replica_root))
				f.write("| " + " | ".join(vals) + " |\n")

			obj_avg_vals = ["Dataset-Average", "-", "-", "-"]
			for k in ordered_obj_metric_keys:
				obj_avg_vals.append(_fmt_metric(obj_avg.get(k, "")))
			obj_avg_vals.append("-")
			f.write("| " + " | ".join(obj_avg_vals) + " |\n")

	return overview_path


def main():
	parser = argparse.ArgumentParser(description="Replica geometry evaluation for this project")
	parser.add_argument("--scan_path", type=str, required=True)
	parser.add_argument("--exp_path", type=str, required=True)
	parser.add_argument("--mode", type=str, default="merged", choices=["merged", "objects_only", "background_only"])
	parser.add_argument("--objects_mesh", type=str, default="scene_combined.glb")
	parser.add_argument("--background_mesh", type=str, default="tsdf_refine_bg_post.ply")
	parser.add_argument("--gtmesh_root", type=str, default="datasets/Replica/GTmesh")
	parser.add_argument("--threshold", type=float, default=0.05)
	parser.add_argument("--down_sample", type=float, default=0.02)
	parser.add_argument("--output_txt", type=str, default=None)
	parser.add_argument("--export_gt_object_mesh", action="store_true", default=True)
	parser.add_argument("--no_export_gt_object_mesh", action="store_false", dest="export_gt_object_mesh")
	parser.add_argument("--gt_object_mesh_name", type=str, default="gt_objects_mesh.ply")
	parser.add_argument("--exps_replica_root", type=str, default="exps/Replica")
	parser.add_argument("--overview_name", type=str, default=None)
	parser.add_argument("--update_overview", action="store_true", default=True)
	parser.add_argument("--no_update_overview", action="store_false", dest="update_overview")
	parser.add_argument("--debug_object_nodes", action="store_true", help="Print scene graph hierarchy for object node matching")
	args = parser.parse_args()

	scan_name = os.path.basename(os.path.normpath(args.scan_path))
	gtmesh_root = os.path.abspath(args.gtmesh_root)
	output_txt = args.output_txt or os.path.join(args.exp_path, f"metrics_{args.mode}.txt")

	per_object_results = None
	if args.mode == "objects_only":
		metrics, per_object_results = evaluate_objects_only(
			exp_path=args.exp_path,
			objects_name=args.objects_mesh,
			gtmesh_root=gtmesh_root,
			scan_name=scan_name,
			scan_path=args.scan_path,
			threshold=args.threshold,
			down_sample=args.down_sample,
			debug_object_nodes=args.debug_object_nodes,
		)
		metrics["ObjectCount"] = float(len(per_object_results))
	else:
		pred_mesh = build_pred_mesh(
			exp_path=args.exp_path,
			mode=args.mode,
			objects_name=args.objects_mesh,
			background_name=args.background_mesh,
		)
		gt_mesh = build_gt_mesh_from_replica(
			gtmesh_root=gtmesh_root,
			scan_name=scan_name,
			mode=args.mode,
		)
		metrics = evaluate(pred_mesh, gt_mesh, threshold=args.threshold, down_sample=args.down_sample)

	print(metrics)
	with open(output_txt, "w", encoding="utf-8") as f:
		for k, v in metrics.items():
			f.write(f"{k}: {v}\n")
		if per_object_results is not None:
			for row in per_object_results:
				obj_prefix = f"obj_{row['object_index']}"
				f.write(f"{obj_prefix}.PredObjNumFromSceneGraph: {row['mapped_pred_obj_num']}\n")
				f.write(f"{obj_prefix}.PredNode: {row['pred_node']}\n")
				f.write(f"{obj_prefix}.PredGeom: {row['pred_geom']}\n")
				f.write(f"{obj_prefix}.GTFile: {row['gt_file']}\n")
				for mk, mv in row["metrics"].items():
					f.write(f"{obj_prefix}.{mk}: {mv}\n")

	if args.export_gt_object_mesh:
		gt_obj_out = os.path.join(args.exp_path, args.gt_object_mesh_name)
		exported = export_gt_object_mesh(
			gtmesh_root=gtmesh_root,
			scan_name=scan_name,
			output_path=gt_obj_out,
		)
		print(f"GT object mesh exported: {exported}")

	if args.update_overview:
		exps_replica_root = os.path.abspath(args.exps_replica_root)
		overview_targets = [
			("metrics_merged.txt", "overview_metrics_merged.md"),
			("metrics_objects_only.txt", "overview_metrics_objects_only.md"),
		]

		default_mode_overview = f"overview_metrics_{args.mode}.md"
		if args.overview_name is not None and args.overview_name != default_mode_overview:
			overview_targets.append((f"metrics_{args.mode}.txt", args.overview_name))

		seen = set()
		for filename, output_name in overview_targets:
			key = (filename, output_name)
			if key in seen:
				continue
			seen.add(key)
			overview_path = generate_overview_md(
				exps_replica_root=exps_replica_root,
				filename=filename,
				output_name=output_name,
			)
			if overview_path is not None:
				print(f"Overview updated: {overview_path}")
 


if __name__ == "__main__":
	main()
