import json


def _parse_scene_graph_rank(rank_str: str):
    level_str, order_str = rank_str.split("-")
    return int(level_str), int(order_str)


def _extract_object_id(obj_name: str) -> int:
    return int(obj_name.split("_")[-1])


def solve_scene_graph(scene_graph_path: str, cond_images: dict) -> dict:
    with open(scene_graph_path, "r") as f:
        scene_graph = json.load(f)

    sorted_ids = sorted(
        scene_graph.items(),
        key=lambda kv: (*_parse_scene_graph_rank(kv[1]), int(kv[0])),
    )

    cond_by_id = {
        _extract_object_id(obj_name): (obj_name, image_path)
        for obj_name, image_path in cond_images.items()
    }

    ordered_items = []
    for obj_id_str, _ in sorted_ids:
        obj_id = int(obj_id_str)
        if obj_id in cond_by_id:
            ordered_items.append(cond_by_id[obj_id])

    return dict(ordered_items)
