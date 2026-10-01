import base64
import io
import os
import time

import requests
from PIL import Image


GENERATE_URL = "https://api.atlascloud.ai/api/v1/model/generateImage"
POLL_URL_TEMPLATE = "https://api.atlascloud.ai/api/v1/model/prediction/{prediction_id}"
MODEL_NAME = "openai/gpt-image-2/edit"
POLL_INTERVAL_SECONDS = 2
MAX_POLL_ATTEMPTS = 300


def _image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _normalize_images(images):
    if images is None:
        return []
    if isinstance(images, (str, Image.Image)):
        return [images]
    return list(images)


def _describe_image_inputs(image_list) -> list[str]:
    descriptions = []
    for idx, image in enumerate(image_list, start=1):
        if isinstance(image, str):
            descriptions.append(image)
            continue
        image_path = getattr(image, "filename", None)
        if image_path:
            descriptions.append(image_path)
        else:
            descriptions.append(f"<in-memory-image-{idx}>")
    return descriptions


def _to_image_input(image) -> str:
    if isinstance(image, str):
        return image
    if isinstance(image, Image.Image):
        return _image_to_base64(image)
    raise TypeError(f"Unsupported image input type: {type(image)!r}")


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


def call_gpt_image_2(
    text,
    images,
    output_path,
    output_format: str = "jpeg",
    quality: str = "medium",
    size: str = "1024x1024",
    moderation: str = "low",
    enable_base64_output: bool = False,
    enable_sync_mode: bool = False,
    **_,
):
    api_key = os.environ.get("ATLASCLOUD_API_KEY")
    if not api_key:
        raise EnvironmentError("ATLASCLOUD_API_KEY is not set")

    image_list = _normalize_images(images)
    if not image_list:
        raise ValueError("GPT-Image-2 requires at least one reference image")

    image_descriptions = _describe_image_inputs(image_list)
    print(f"[call_gpt_image_2] Prompt: {text}")
    print(f"[call_gpt_image_2] Input images: {image_descriptions}")

    payload = {
        "model": MODEL_NAME,
        "enable_base64_output": enable_base64_output,
        "enable_sync_mode": enable_sync_mode,
        "images": [_to_image_input(image) for image in image_list],
        "output_format": output_format,
        "prompt": text,
        "quality": quality,
        "size": size,
        "moderation": moderation,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    generate_response = requests.post(GENERATE_URL, headers=headers, json=payload, timeout=60)
    generate_response.raise_for_status()
    generate_result = generate_response.json()

    prediction_id = generate_result.get("data", {}).get("id")
    if not prediction_id:
        raise RuntimeError(f"GPT-Image-2 request failed: {generate_result}")

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
                raise RuntimeError(f"GPT-Image-2 returned no outputs: {result}")
            image_url = _extract_output_url(outputs[0])
            if not image_url:
                raise RuntimeError(f"GPT-Image-2 output is not a downloadable URL: {outputs[0]}")
            print(image_url)
            _save_response_image(image_url, output_path)
            print(f"Image saved to {output_path}")
            return

        if status == "failed":
            raise RuntimeError(data.get("error") or "GPT-Image-2 generation failed")

        if attempt == MAX_POLL_ATTEMPTS - 1:
            raise TimeoutError(f"GPT-Image-2 polling timed out for prediction {prediction_id}")
        time.sleep(POLL_INTERVAL_SECONDS)
