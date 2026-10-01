import os
import sys
import glob
import shutil

import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


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


def _get_image_model_caller(img_model: str):
    resolved_model = _normalize_img_model(img_model)
    if resolved_model == "Seedream":
        from decomvoxel.model.Seedream.call_seedream import call_seedream
        return resolved_model, call_seedream
    if resolved_model == "NanoBanana":
        from decomvoxel.model.NanoBanana.call_nanobanana import call_nanobanana
        return resolved_model, call_nanobanana

    from decomvoxel.model.GPTImage2.call_gpt_image_2 import call_gpt_image_2
    return resolved_model, call_gpt_image_2


def select_cond_img(model_path: str, output_root: str) -> dict:
    """Return {obj_id_str: {'partial': [center_path, area_path], 'env': [env_center_path, env_area_path]}}."""
    renders_dir = os.path.join(model_path, "semantic_result", "object_renders")
    if not os.path.isdir(renders_dir):
        raise FileNotFoundError(f"Object renders directory not found: {renders_dir}")
    env_renders_dir = os.path.join(model_path, "semantic_result", "env_renders")

    output_dir = os.path.join(output_root, "partial_cond")
    env_output_dir = os.path.join(output_root, "env_cond")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(env_output_dir, exist_ok=True)

    selected: dict = {}  # obj_id_str -> {'partial': [...], 'env': [...]}

    for obj_dir_name in sorted(os.listdir(renders_dir)):
        obj_dir = os.path.join(renders_dir, obj_dir_name)
        if not os.path.isdir(obj_dir):
            continue

        parts = obj_dir_name.split("_")
        if len(parts) < 2:
            continue
        obj_id_str = parts[-1]

        # Collect area and bbox-center distance-to-image-center for each view.
        candidates = []  # (img_path, area, center_dist)
        for img_name in sorted(os.listdir(obj_dir)):
            if not img_name.lower().endswith(".png"):
                continue
            img_path = os.path.join(obj_dir, img_name)
            img = np.array(Image.open(img_path))

            non_white = np.any(img < 250, axis=-1) if img.ndim == 3 else img < 250 # with a marginal
            area = int(non_white.sum())
            if area == 0:
                continue

            # Non-white pixels center of mass
            ys, xs = np.where(non_white)
            bbox_cy = ys.mean()
            bbox_cx = xs.mean()
            img_cy, img_cx = img.shape[0] / 2.0, img.shape[1] / 2.0
            center_dist = np.sqrt((bbox_cy - img_cy) ** 2 + (bbox_cx - img_cx) ** 2)
            candidates.append((img_path, area, center_dist))

        if not candidates:
            print(f"[select_cond_img] WARNING: {obj_dir_name} has no valid renders, skipping")
            continue

        # Pick 1: bbox center closest to image center.
        best_center = min(candidates, key=lambda x: x[2])
        # Pick 2: largest foreground area.
        best_area = max(candidates, key=lambda x: x[1])

        out1 = os.path.join(output_dir, f"cond_{obj_id_str}_1.png")
        out2 = os.path.join(output_dir, f"cond_{obj_id_str}_2.png")
        shutil.copy2(best_center[0], out1)
        shutil.copy2(best_area[0], out2)

        # Also copy same-named environment renders from env_renders to env_cond.
        env_paths = []
        for idx, (src_path, label) in enumerate([(best_center[0], '1'), (best_area[0], '2')], start=1):
            img_filename = os.path.basename(src_path)
            env_src = os.path.join(env_renders_dir, obj_dir_name, img_filename)
            if os.path.isfile(env_src):
                env_out = os.path.join(env_output_dir, f"env_{obj_id_str}_{label}.png")
                shutil.copy2(env_src, env_out)
                env_paths.append(os.path.abspath(env_out))
            else:
                print(f"[select_cond_img] WARNING: environment render not found {env_src}, skipping")

        selected[obj_id_str] = {
            'partial': [os.path.abspath(out1), os.path.abspath(out2)],
            'env': env_paths,
        }
        print(f"[select_cond_img] {obj_dir_name}: "
              f"center={os.path.basename(best_center[0])}(dist={best_center[2]:.1f}), "
              f"area={os.path.basename(best_area[0])}(area={best_area[1]}), "
              f"env_images={len(env_paths)}")

    print(f"[select_cond_img] Done: {len(selected)} objects, partial -> {output_dir}, env -> {env_output_dir}")
    return selected


def pad_and_recenter_cond_img(
    img: Image.Image,
    white_threshold: int = 20,
    target_ratio: float = 0.40, # 0.50
    output_size: int = None,
) -> Image.Image:
    """
    Recenter and rescale the non-white content of a conditioning image so that
    its max bounding-box extent equals *target_ratio* of the output image size,
    then replace all near-white pixels with pure white (255).
    """
    arr = np.array(img.convert("RGB"))  # (H, W, 3)
    if output_size is None:
        output_size = max(arr.shape[0], arr.shape[1])

    # --- 1. Identify non-white (foreground) pixels -------------------------
    white_lo = 255 - white_threshold  # e.g. 235
    fg_mask = np.any(arr < white_lo, axis=-1)  # True = foreground

    # --- 2. Sample background colour from the outermost border ring --------
    H, W = arr.shape[:2]
    border_mask = np.zeros((H, W), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    # Keep only pixels that are "white" (near-background) in the border
    bg_border_mask = border_mask & ~fg_mask
    if bg_border_mask.any():
        border_pixels = arr[bg_border_mask]  # (N, 3)
        # Pack each RGB triplet into a single uint32 for fast mode computation
        packed = (border_pixels[:, 0].astype(np.uint32) << 16
                  | border_pixels[:, 1].astype(np.uint32) << 8
                  | border_pixels[:, 2].astype(np.uint32))
        unique, counts = np.unique(packed, return_counts=True)
        mode_packed = unique[counts.argmax()]
        bg_color = np.array([
            (mode_packed >> 16) & 0xFF,
            (mode_packed >> 8) & 0xFF,
            mode_packed & 0xFF,
        ], dtype=np.uint8)
    else:
        bg_color = np.array([255, 255, 255], dtype=np.uint8)

    if not fg_mask.any():
        canvas = np.broadcast_to(bg_color, (output_size, output_size, 3)).copy()
        return Image.fromarray(canvas)

    # --- 3. Bounding box of foreground -------------------------------------
    ys, xs = np.where(fg_mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    bbox_h = y_max - y_min + 1
    bbox_w = x_max - x_min + 1
    max_extent = max(bbox_h, bbox_w)

    # --- 4. Crop to bbox ---------------------------------------------------
    crop = arr[y_min:y_max + 1, x_min:x_max + 1]  # (bbox_h, bbox_w, 3)

    # --- 5. Rescale so that max_extent → target_ratio * output_size --------
    desired_extent = int(round(target_ratio * output_size))
    scale = desired_extent / max_extent
    new_h = max(1, int(round(bbox_h * scale)))
    new_w = max(1, int(round(bbox_w * scale)))
    crop_pil = Image.fromarray(crop).resize((new_w, new_h), Image.LANCZOS)
    crop_arr = np.array(crop_pil)

    # --- 6. Paste centred onto a background-coloured canvas ----------------
    canvas = np.full((output_size, output_size, 3), bg_color, dtype=np.uint8)
    paste_y = (output_size - new_h) // 2
    paste_x = (output_size - new_w) // 2
    canvas[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = crop_arr

    # --- 7. Replace near-white pixels (all channels >= 252) with pure white --
    # near_white = np.all(canvas >= 252, axis=-1)
    # canvas[near_white] = 255

    # --- 8. Increase overall brightness by 5 --------------------------------
    canvas = np.clip(canvas.astype(np.int16) + 5, 0, 255).astype(np.uint8)

    return Image.fromarray(canvas)


def generate_cond_img(
    model_path: str,
    output_root: str,
    text: str = None,
    object_categories: dict = None,
    img_model: str = "Seedream",
    img_model_params: dict = None,
) -> dict:
    resolved_model, call_image_model = _get_image_model_caller(img_model)
    img_model_params = img_model_params or {}

    partial_images = select_cond_img(model_path, output_root)
    
    # XXX only keep selected objects with category info (for demo)
    if object_categories is not None:
        selected_ids = {str(int(k)) for k in object_categories.keys()}
        partial_images = {k: v for k, v in partial_images.items() if str(int(k)) in selected_ids}
        print(f"[generate_cond_img] Filtered to {len(partial_images)} objects: {sorted(partial_images.keys())}")
        
    complete_dir = os.path.join(output_root, "complete_cond")
    os.makedirs(complete_dir, exist_ok=True)

    generated: dict = {}  # obj_id_str -> output_path

    for obj_id_str, data in partial_images.items():
        out_path = os.path.join(complete_dir, f"{obj_id_str}.png")
        partial_paths = data['partial']
        env_paths = data.get('env', [])

        if text is not None:
            _obj_text = text
        else:
            _default_text = ("The first two reference images show the target object, and the last two are for the reference environment. "
                             "Please complete the target object, providing its full form, including any potentially obscured structures. "
                             "Restore any holes or noise on the object to its original appearance. "
                             "Place the object in the center of the image and display it in its entirety. "
                             "Note that the background must remain completely blank (white); do not add any background or shadows, and maintain a 1:1 aspect ratio.")
            if object_categories is not None:
                _id = str(int(obj_id_str))
                category = object_categories.get(_id)
                if category:
                    _obj_text = _default_text.replace("object", category)
                    if "table" in category.lower():
                        # "Should the table be selected, return exclusively the table, excluding all surrounding information! It must be observed from the side!"
                        _obj_text += ("Note that you should return exclusively the table, excluding all surrounding object! Show the table from a side view where the structure of the legs is clearly apparent!")
                    elif "chair" in category.lower():
                        _obj_text += ("For example, you should provide the complete chair instead of the backrest of it! ")
                    elif "desk" in category.lower():
                        _obj_text += ("Note that you should return exclusively the desk, excluding all surrounding objects on the desktop!")
                    elif "poster" in category.lower():
                        _obj_text += ("Note that you should return the poster in the shape of rectangle or square.")
                    elif "painting" in category.lower():
                        _obj_text += ("Note that you should return the painting in the shape of rectangle or square.")
                    print(f"[generate_cond_img] object {obj_id_str}: category='{category}', text replaced 'object' -> '{category}'")
                else:
                    _obj_text = _default_text
                    print(f"[generate_cond_img] object {obj_id_str}: category not found, using default text")
            else:
                _obj_text = _default_text
        
        images = [Image.open(p) for p in partial_paths + env_paths]
        print(f"[generate_cond_img] Calling {resolved_model}: object {obj_id_str}, "
              f"{len(partial_paths)} partial + {len(env_paths)} env = {len(images)} reference images ...")
        print(f"Text prompt: {_obj_text}")
        call_image_model(text=_obj_text, images=images, output_path=out_path, **img_model_params)

        # Post-process: recenter & pad so object fills ~50% of the image
        raw_img = Image.open(out_path)
        padded_img = pad_and_recenter_cond_img(raw_img)
        padded_img.save(out_path)
        print(f"[generate_cond_img] object {obj_id_str} pad+recenter done")

        generated[obj_id_str] = os.path.abspath(out_path)
        print(f"[generate_cond_img] object {obj_id_str} -> {out_path}")

    print(f"[generate_cond_img] Done: saved {len(generated)} images to {complete_dir}")
    return generated


def select_cond_img_avo(model_path: str, output_root: str) -> dict:
    """Return {obj_id_str: {'partial': [center_path, area_path, avo_path], 'env': [env_center_path, env_area_path, env_avo_path]}}.

    Same input/output format as select_cond_img, with one extra selected view:
        - partial[0]: view whose bbox center is closest to image center (same as select_cond_img)
        - partial[1]: view with largest foreground area (same as select_cond_img)
        - partial[2]: view with highest AVO score (best center-region foreground coverage)
    The env list corresponds one-to-one with the partial list when matching env images exist.
    """
    from decomvoxel.utils.avo_view_selection import select_best_avo_view
    
    renders_dir = os.path.join(model_path, "semantic_result", "object_renders")
    if not os.path.isdir(renders_dir):
        raise FileNotFoundError(f"Object renders directory not found: {renders_dir}")
    env_renders_dir = os.path.join(model_path, "semantic_result", "env_renders")

    output_dir = os.path.join(output_root, "partial_cond_avo")
    env_output_dir = os.path.join(output_root, "env_cond_avo")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(env_output_dir, exist_ok=True)

    selected: dict = {}  # obj_id_str -> {'partial': [...], 'env': [...]}

    for obj_dir_name in sorted(os.listdir(renders_dir)):
        obj_dir = os.path.join(renders_dir, obj_dir_name)
        if not os.path.isdir(obj_dir):
            continue

        parts = obj_dir_name.split("_")
        if len(parts) < 2:
            continue
        obj_id_str = parts[-1]

        # Collect area and bbox-center distance-to-image-center for each view.
        candidates = []  # (img_path, area, center_dist)
        for img_name in sorted(os.listdir(obj_dir)):
            if not img_name.lower().endswith(".png"):
                continue
            img_path = os.path.join(obj_dir, img_name)
            img = np.array(Image.open(img_path))

            non_white = np.any(img < 250, axis=-1) if img.ndim == 3 else img < 250
            area = int(non_white.sum())
            if area == 0:
                continue

            ys, xs = np.where(non_white)
            bbox_cy = ys.mean()
            bbox_cx = xs.mean()
            img_cy, img_cx = img.shape[0] / 2.0, img.shape[1] / 2.0
            center_dist = np.sqrt((bbox_cy - img_cy) ** 2 + (bbox_cx - img_cx) ** 2)
            candidates.append((img_path, area, center_dist))

        if not candidates:
            print(f"[select_cond_img_avo] WARNING: {obj_dir_name} has no valid renders, skipping")
            continue

        # Pick 1: bbox center closest to image center.
        best_center = min(candidates, key=lambda x: x[2])
        # Pick 2: largest foreground area.
        best_area = max(candidates, key=lambda x: x[1])
        # Pick 3: highest AVO score (best center-region foreground coverage).
        avo_result = select_best_avo_view(candidates)  # (img_path, area, center_dist, avo_score)
        best_avo = avo_result[:3]  # (img_path, area, center_dist)

        out1 = os.path.join(output_dir, f"cond_{obj_id_str}_1.png")
        out2 = os.path.join(output_dir, f"cond_{obj_id_str}_2.png")
        out3 = os.path.join(output_dir, f"cond_{obj_id_str}_3.png")
        shutil.copy2(best_center[0], out1)
        shutil.copy2(best_area[0], out2)
        shutil.copy2(best_avo[0], out3)

        # Also copy same-named environment renders from env_renders to env_cond_avo.
        env_paths = []
        for src_path, label in [(best_center[0], '1'), (best_area[0], '2'), (best_avo[0], '3')]:
            img_filename = os.path.basename(src_path)
            env_src = os.path.join(env_renders_dir, obj_dir_name, img_filename)
            if os.path.isfile(env_src):
                env_out = os.path.join(env_output_dir, f"env_{obj_id_str}_{label}.png")
                shutil.copy2(env_src, env_out)
                env_paths.append(os.path.abspath(env_out))
            else:
                print(f"[select_cond_img_avo] WARNING: environment render not found {env_src}, skipping")

        selected[obj_id_str] = {
            'partial': [os.path.abspath(out1), os.path.abspath(out2), os.path.abspath(out3)],
            'env': env_paths,
        }
        print(f"[select_cond_img_avo] {obj_dir_name}: "
              f"center={os.path.basename(best_center[0])}(dist={best_center[2]:.1f}), "
              f"area={os.path.basename(best_area[0])}(area={best_area[1]}), "
              f"AVO={os.path.basename(best_avo[0])}(avo_score={avo_result[3]:.0f}), "
              f"env_images={len(env_paths)}")

    print(f"[select_cond_img_avo] Done: {len(selected)} objects, partial -> {output_dir}, env -> {env_output_dir}")
    return selected


def generate_cond_img_avo(
    model_path: str,
    output_root: str,
    text: str = None,
    object_categories: dict = None,
    img_model: str = "Seedream",
    img_model_params: dict = None,
) -> dict:
    """Same input/output as generate_cond_img, but use select_cond_img_avo to choose conditioning views.

    Images sent to Seedream:
    3 partial views (center + max-area + AVO)
    + 1 environment image corresponding to the AVO view.
    """
    resolved_model, call_image_model = _get_image_model_caller(img_model)
    img_model_params = img_model_params or {}

    partial_images = select_cond_img_avo(model_path, output_root)

    # Keep only objects with category information (same logic as generate_cond_img)
    if object_categories is not None:
        selected_ids = {str(int(k)) for k in object_categories.keys()}
        partial_images = {k: v for k, v in partial_images.items() if str(int(k)) in selected_ids}
        print(f"[generate_cond_img_avo] Filtered to {len(partial_images)} objects: {sorted(partial_images.keys())}")

    complete_dir = os.path.join(output_root, "complete_cond_avo")
    os.makedirs(complete_dir, exist_ok=True)

    generated: dict = {}  # obj_id_str -> output_path

    for obj_id_str, data in partial_images.items():
        out_path = os.path.join(complete_dir, f"{obj_id_str}.png")
        partial_paths = data['partial']          # [center, area, avo] three partial images
        env_paths = data.get('env', [])          # one-to-one correspondence with partial

        # Use env that corresponds to AVO view (3rd partial -> 3rd env, index 2)
        avo_env_paths = [env_paths[2]] if len(env_paths) >= 3 else (
            [env_paths[-1]] if env_paths else []
        )

        if text is not None:
            _obj_text = text
        else:
            _default_text = (
                "The first three reference images show the target object from different viewpoints, "
                "and the last image is for the reference environment. "
                "Please complete the target object, providing its full form, including any potentially obscured structures. "
                "Restore any holes or noise on the object to its original appearance. "
                "Place the object in the center of the image and display it in its entirety. "
                "Note that the background must remain completely blank (white); do not add any background or shadows, and maintain a 1:1 aspect ratio."
            )
            if object_categories is not None:
                _id = str(int(obj_id_str))
                category = object_categories.get(_id)
                if category:
                    _obj_text = _default_text.replace("object", category)
                    if "table" in category.lower():
                        _obj_text += ("Note that you should return exclusively the table, excluding all surrounding object! Show the table from a side view where the structure of the legs is clearly apparent!")
                    elif "chair" in category.lower():
                        _obj_text += ("For example, you should provide the complete chair instead of the backrest of it! ")
                    elif "desk" in category.lower():
                        _obj_text += ("Note that you should return exclusively the desk, excluding all surrounding objects on the desktop!")
                    elif "poster" in category.lower():
                        _obj_text += ("Note that you should return the poster in the shape of rectangle or square.")
                    elif "painting" in category.lower():
                        _obj_text += ("Note that you should return the painting in the shape of rectangle or square.")
                    print(f"[generate_cond_img_avo] Object {obj_id_str}: category='{category}', replaced 'object' -> '{category}' in prompt")
                else:
                    _obj_text = _default_text
                    print(f"[generate_cond_img_avo] Object {obj_id_str}: category not found, using default prompt")
            else:
                _obj_text = _default_text

        # 3 partial images + 1 AVO environment image
        input_images = [Image.open(p) for p in partial_paths + avo_env_paths]
        print(f"[generate_cond_img_avo] Calling {resolved_model}: object {obj_id_str}, "
              f"3 partial + {len(avo_env_paths)} avo_env = {len(input_images)} reference images ...")
        print(f"Prompt: {_obj_text}")
        call_image_model(text=_obj_text, images=input_images, output_path=out_path, **img_model_params)

        # Post-process: recenter & pad
        raw_img = Image.open(out_path)
        padded_img = pad_and_recenter_cond_img(raw_img)
        padded_img.save(out_path)
        print(f"[generate_cond_img_avo] Object {obj_id_str} pad+recenter completed")

        generated[obj_id_str] = os.path.abspath(out_path)
        print(f"[generate_cond_img_avo] Object {obj_id_str} -> {out_path}")

    print(f"[generate_cond_img_avo] Done, saved {len(generated)} images to {complete_dir}")
    return generated


def generate_cond_img_avo_para(
    model_path: str,
    output_root: str,
    text: str = None,
    object_categories: dict = None,
    parallel_num: int = 4,
    img_model: str = "Seedream",
    img_model_params: dict = None,
) -> dict:
    """Same input/output as generate_cond_img_avo, but calls Seedream in parallel.

    - select_cond_img_avo stays serial (light local I/O).
    - The main loop (call_seedream + pad_and_recenter_cond_img) runs in a thread pool.
    """
    from decomvoxel.utils.parallel import run_parallel

    resolved_model, call_image_model = _get_image_model_caller(img_model)
    img_model_params = img_model_params or {}

    partial_images = select_cond_img_avo(model_path, output_root)

    if object_categories is not None:
        selected_ids = {str(int(k)) for k in object_categories.keys()}
        partial_images = {k: v for k, v in partial_images.items() if str(int(k)) in selected_ids}
        print(f"[generate_cond_img_avo_para] Filtered to {len(partial_images)} objects: {sorted(partial_images.keys())}")

    complete_dir = os.path.join(output_root, "complete_cond_avo")
    os.makedirs(complete_dir, exist_ok=True)

    def _build_text(obj_id_str: str) -> str:
        if text is not None:
            return text
        _default_text = (
            "The first three reference images show the target object from different viewpoints, "
            "and the last image is for the reference environment. "
            "Please complete the target object, providing its full form, including any potentially obscured structures. "
            "Restore any holes or noise on the object to its original appearance. "
            "Place the object in the center of the image and display it in its entirety. "
            "Note that the background must remain completely blank (white); do not add any background or shadows, and maintain a 1:1 aspect ratio."
        )
        if object_categories is None:
            return _default_text
        _id = str(int(obj_id_str))
        category = object_categories.get(_id)
        if not category:
            return _default_text
        _obj_text = _default_text.replace("object", category)
        cat_lower = category.lower()
        if "table" in cat_lower:
            _obj_text += ("Note that you should return exclusively the table, excluding all surrounding object! Show the table from a side view where the structure of the legs is clearly apparent!")
        elif "chair" in cat_lower:
            _obj_text += ("For example, you should provide the complete chair instead of the backrest of it! ")
        elif "desk" in cat_lower:
            _obj_text += ("Note that you should return exclusively the desk, excluding all surrounding objects on the desktop!")
        elif "poster" in cat_lower:
            _obj_text += ("Note that you should return the poster in the shape of rectangle or square.")
        elif "painting" in cat_lower:
            _obj_text += ("Note that you should return the painting in the shape of rectangle or square.")
        return _obj_text

    def _process_one(obj_id_str: str, data: dict) -> str:
        out_path = os.path.join(complete_dir, f"{obj_id_str}.png")
        partial_paths = data['partial']
        env_paths = data.get('env', [])
        avo_env_paths = [env_paths[2]] if len(env_paths) >= 3 else (
            [env_paths[-1]] if env_paths else []
        )
        _obj_text = _build_text(obj_id_str)
        input_images = [Image.open(p) for p in partial_paths + avo_env_paths]
        print(f"[generate_cond_img_avo_para] Calling {resolved_model}: object {obj_id_str}, "
              f"3 partial + {len(avo_env_paths)} avo_env = {len(input_images)} reference images")
        call_image_model(text=_obj_text, images=input_images, output_path=out_path, **img_model_params)

        raw_img = Image.open(out_path)
        padded_img = pad_and_recenter_cond_img(raw_img)
        padded_img.save(out_path)
        print(f"[generate_cond_img_avo_para] Object {obj_id_str} -> {out_path}")
        return os.path.abspath(out_path)

    tasks = [(obj_id_str, data) for obj_id_str, data in partial_images.items()]
    results = run_parallel(_process_one, tasks, max_workers=parallel_num, desc="cond_img_avo")

    generated: dict = {}
    for key, value, exc in results:
        if exc is None and value is not None:
            generated[key] = value
        else:
            print(f"[generate_cond_img_avo_para] Object {key} failed, skipped")

    print(f"[generate_cond_img_avo_para] Done, saved {len(generated)} images to {complete_dir}")
    return generated


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Select / generate conditioning images.")
    parser.add_argument("model_path", type=str,
                        help="Model directory (e.g. outputs/replica_recon_large/scan1)")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Output root dir (default: <model_path>/cond_images)")
    parser.add_argument("--select_only", action="store_true",
                        help="Only run select_cond_img, skip cond image generation")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--img_model", type=str, default="Seedream",
                        help="Conditioning image model: Seedream, NanoBanana, or GPTImage2")
    parser.add_argument("--img_model_params_json", type=str, default=None,
                        help="Optional JSON object for cond image model specific parameters")
    args = parser.parse_args()

    output_root = args.output_root or os.path.join(args.model_path, "cond_images")
    
    if args.text:
        text = args.text
    else:
        text = "Complete this object, including holes or noise on its surface, and restore those regions to the object's original appearance. Make the object clearer and cleaner, place it in the center of the image, and show the full object. The first two reference images are target-object views, and the last two are environment references. Keep the background empty (zero value), do not add any background or shadows, and maintain a 1:1 aspect ratio."

    img_model_params = None
    if args.img_model_params_json:
        img_model_params = json.loads(args.img_model_params_json)
        if not isinstance(img_model_params, dict):
            raise ValueError("--img_model_params_json must decode to a JSON object")

    if args.select_only:
        select_cond_img(args.model_path, output_root)
    else:
        generate_cond_img(
            args.model_path,
            output_root,
            text=text,
            img_model=args.img_model,
            img_model_params=img_model_params,
        )
