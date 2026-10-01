import os
import io
import base64
import requests
from PIL import Image
from openai import OpenAI


def _image_to_data_uri(image) -> str:
    """PIL Image -> data URI"""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def call_seedream(text, images, output_path, **_):
    """images: a single PIL Image or a list of PIL Images"""
    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=os.environ.get("ARK_API_KEY"),
    )

    # Normalize to a list
    if isinstance(images, Image.Image):
        images = [images]

    image_uris = [_image_to_data_uri(img) for img in images]
    # Pass a string for one image, and a list for multiple images
    image_param = image_uris[0] if len(image_uris) == 1 else image_uris

    imagesResponse = client.images.generate(
        model="doubao-seedream-5-0-260128",
        prompt=text,
        size="1920x1920",
        response_format="url",
        extra_body={
            "image": image_param,
            "watermark": False,
        }
    )
    print(imagesResponse.data[0].url)

    image_url = imagesResponse.data[0].url
    response = requests.get(image_url)
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Image saved to {output_path}")