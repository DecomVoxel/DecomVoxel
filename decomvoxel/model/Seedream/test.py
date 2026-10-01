import os
import requests
from openai import OpenAI

# Make sure your API key is stored in the ARK_API_KEY environment variable.
# Initialize the Ark client and read the API key from environment variables.
client = OpenAI( 
    # This is the default endpoint. Adjust it based on your deployment region.
    base_url="https://ark.cn-beijing.volces.com/api/v3", 
    # Read the API key from environment variables (default behavior).
    api_key=os.environ.get("ARK_API_KEY"), 
) 

imagesResponse = client.images.generate( 
    model="doubao-seedream-5-0-260128", 
    prompt="Generate a close-up scene of a dog lying on grass.",
    size="2K",
    response_format="url",
    extra_body = {
        "image": "https://ark-project.tos-cn-beijing.volces.com/doc_image/seedream4_imageToimage.png",
        "watermark": False
    }
) 
print(imagesResponse.data[0].url)

image_url = imagesResponse.data[0].url
response = requests.get(image_url)
with open("generated_image.png", "wb") as f:
    f.write(response.content)
print("Image saved as generated_image.png")