import base64
import io
import os
import time

import requests
from PIL import Image


GENERATE_URL = "https://api.atlascloud.ai/api/v1/model/generateImage"
POLL_URL_TEMPLATE = "https://api.atlascloud.ai/api/v1/model/prediction/{prediction_id}"
MODEL_NAME = "google/nano-banana-2/reference-to-image-developer"
POLL_INTERVAL_SECONDS = 2
MAX_POLL_ATTEMPTS = 300


def _image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _normalize_images(images):
    if images is None:
        return []
    if isinstance(images, Image.Image):
        return [images]
    return list(images)


def _describe_image_inputs(image_list) -> list[str]:
    descriptions = []
    for idx, image in enumerate(image_list, start=1):
        image_path = getattr(image, "filename", None)
        if image_path:
            descriptions.append(image_path)
        else:
            descriptions.append(f"<in-memory-image-{idx}>")
    return descriptions


def _extract_output_url(output_item):
    if isinstance(output_item, str):
        return output_item
    if isinstance(output_item, dict):
        return output_item.get("url") or output_item.get("output") or output_item.get("uri")
    return None


def _save_response_image(image_url: str, output_path: str) -> None:
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)


def call_nanobanana(
    text,
    images,
    output_path,
    aspect_ratio: str = "1:1",
    enable_base64_output: bool = False,
    enable_sync_mode: bool = False,
    enable_web_search: bool = False,
    resolution: str = "2k",
    thinking_level: str = "default",
    **_,
):
    api_key = os.environ.get("ATLASCLOUD_API_KEY")
    if not api_key:
        raise EnvironmentError("ATLASCLOUD_API_KEY is not set")

    image_list = _normalize_images(images)
    image_descriptions = _describe_image_inputs(image_list)
    print(f"[call_nanobanana] Prompt: {text}")
    print(f"[call_nanobanana] Input images: {image_descriptions}")

    payload = {
        "model": MODEL_NAME,
        "aspect_ratio": aspect_ratio,
        "enable_base64_output": enable_base64_output,
        "enable_sync_mode": enable_sync_mode,
        "enable_web_search": enable_web_search,
        "prompt": text,
        "resolution": resolution,
        "thinking_level": thinking_level,
    }
    if image_list:
        payload["images"] = [_image_to_base64(img) for img in image_list]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    generate_response = requests.post(GENERATE_URL, headers=headers, json=payload, timeout=60)
    generate_response.raise_for_status()
    generate_result = generate_response.json()

    prediction_id = generate_result.get("data", {}).get("id")
    if not prediction_id:
        raise RuntimeError(f"NanoBanana request failed: {generate_result}")

    poll_url = POLL_URL_TEMPLATE.format(prediction_id=prediction_id)
    for attempt in range(MAX_POLL_ATTEMPTS):
        response = requests.get(poll_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
        response.raise_for_status()
        result = response.json()
        data = result.get("data", {})
        status = data.get("status")

        if status == "completed":
            outputs = data.get("outputs") or []
            if not outputs:
                raise RuntimeError(f"NanoBanana returned no outputs: {result}")
            image_url = _extract_output_url(outputs[0])
            if not image_url:
                raise RuntimeError(f"NanoBanana output is not a downloadable URL: {outputs[0]}")
            print(image_url)
            _save_response_image(image_url, output_path)
            print(f"Image saved to {output_path}")
            return

        if status == "failed":
            raise RuntimeError(data.get("error") or "NanoBanana generation failed")

        if attempt == MAX_POLL_ATTEMPTS - 1:
            raise TimeoutError(f"NanoBanana polling timed out for prediction {prediction_id}")
        time.sleep(POLL_INTERVAL_SECONDS)
