#!/usr/bin/env python3
"""
Image Generation Script with multiple providers
Supports: OpenRouter, Qwen (DashScope)
"""

import os
import sys
import base64
import time
import json
import urllib.request
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Get the project root directory (parent of scripts folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from project root
load_dotenv(PROJECT_ROOT / ".env")

# Provider configurations
PROVIDERS = {
    "1": {
        "name": "openrouter",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "bytedance-seed/seedream-4.5",
    },
    "2": {
        # Qwen now uses the native DashScope HTTP API directly,
        # not the OpenAI-compatible /compatible-mode endpoint.
        "name": "qwen",
        "api_key_env": "DASHSCOPE_API_KEY",
        "models": {
            "1": "qwen-image-2.0",
            "2": "qwen-image-2.0-pro",
        },
    },
}

OUTPUT_DIR = PROJECT_ROOT / "generated_images"
REFERENCE_IMAGES_DIR = PROJECT_ROOT / "reference_images"

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


def select_provider():
    """Let user select which provider to use."""
    print("\nSelect API Provider:")
    print("-" * 40)
    print("  [1] OpenRouter (ByteDance Seedream 4.5)")
    print("  [2] Qwen (DashScope)")
    print("-" * 40)

    while True:
        choice = input("\nEnter provider number: ").strip()
        if choice in PROVIDERS:
            provider = PROVIDERS[choice].copy()
            # If Qwen is selected, prompt for model variant
            if provider["name"] == "qwen":
                provider["model"] = select_qwen_model()
            return provider
        print("Please enter 1 or 2")


def select_qwen_model():
    """Let user select which Qwen model variant to use."""
    print("\nSelect Qwen Model:")
    print("-" * 40)
    print("  [1] qwen-image-2.0 (Standard)")
    print("  [2] qwen-image-2.0-pro (Pro - higher quality)")
    print("-" * 40)

    while True:
        choice = input("\nEnter model number: ").strip()
        if choice in PROVIDERS["2"]["models"]:
            return PROVIDERS["2"]["models"][choice]
        print("Please enter 1 or 2")


def get_client(provider):
    """
    Initialize and return the client/model for the selected provider.

    - For OpenRouter, returns an OpenAI-compatible client and model name.
    - For Qwen, we do NOT use the compatible-mode client anymore; the
      DashScope HTTP API is called directly inside generate_image, so we
      just validate the API key and return (None, model).
    """
    api_key = os.getenv(provider["api_key_env"])
    if not api_key:
        print(f"ERROR: Please set your {provider['api_key_env']} in the .env file")
        sys.exit(1)

    if provider["name"] == "qwen":
        # Raw HTTP via DashScope; no OpenAI client needed.
        return None, provider["model"]

    return OpenAI(
        base_url=provider["base_url"],
        api_key=api_key,
    ), provider["model"]


def ensure_directories():
    """Create necessary directories if they don't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    REFERENCE_IMAGES_DIR.mkdir(exist_ok=True)


def get_reference_images():
    """Get list of all reference images in the reference_images folder."""
    if not REFERENCE_IMAGES_DIR.exists():
        return []

    images = []
    for file in REFERENCE_IMAGES_DIR.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(file)
    return sorted(images, key=lambda x: x.name)


def display_reference_images(images):
    """Display the list of reference images with numbers."""
    if not images:
        print("No reference images found in the 'reference_images' folder.")
        return None

    print("\nAvailable reference images:")
    print("-" * 40)
    for i, img in enumerate(images, 1):
        print(f"  [{i}] {img.name}")
    print(f"  [0] No image (text-only generation)")
    print("-" * 40)

    return images


def select_reference_image(images):
    """Let user select a reference image or skip."""
    if not images:
        return None

    while True:
        try:
            choice = input("\nSelect an image number (or 0 for text-only): ").strip()
            if choice == "":
                print("No selection made. Using text-only generation.")
                return None

            choice_num = int(choice)
            if choice_num == 0:
                return None
            elif 1 <= choice_num <= len(images):
                return images[choice_num - 1]
            else:
                print(f"Please enter a number between 0 and {len(images)}")
        except ValueError:
            print("Please enter a valid number")


def image_to_base64_url(image_path):
    """Convert an image file to a base64 data URL."""
    with open(image_path, "rb") as f:
        image_data = f.read()

    ext = image_path.suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp"
    }
    mime_type = mime_types.get(ext, "image/png")

    base64_data = base64.b64encode(image_data).decode("utf-8")
    return f"data:{mime_type};base64,{base64_data}"


def generate_image(client, model, prompt, reference_image=None, provider_name="unknown"):
    """Generate an image using the selected provider."""
    print(f"\nGenerating image...")
    print(f"Provider: {provider_name}")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")

    if reference_image:
        print(f"Reference image: {reference_image.name}")

    try:
        # Special handling for Qwen: call DashScope multimodal-generation API directly
        if provider_name == "qwen":
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                print("ERROR: DASHSCOPE_API_KEY is not set in the environment.")
                return None, provider_name

            # Build DashScope-style content array
            content_items = []
            if reference_image:
                image_data_url = image_to_base64_url(reference_image)
                content_items.append({"image": image_data_url})
            content_items.append({"text": prompt})

            payload = {
                "model": model,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": content_items,
                        }
                    ]
                },
                "parameters": {
                    "n": 1,
                    "negative_prompt": " ",
                    "prompt_extend": True,
                    "watermark": False,
                    "size": "1536*1024",
                },
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            url = (
                "https://dashscope-intl.aliyuncs.com/api/v1/services/"
                "aigc/multimodal-generation/generation"
            )

            start_time = time.time()
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    status = resp.getcode()
                    body = resp.read().decode("utf-8", errors="replace")
            except Exception as e:
                print(f"Error calling DashScope API: {e}")
                return None, provider_name

            elapsed = time.time() - start_time
            print(f"API request took: {elapsed:.2f} seconds")

            if status != 200:
                print(f"DashScope returned HTTP {status}")
                return None, provider_name

            try:
                data = json.loads(body)
                choices = data.get("output", {}).get("choices", [])
                if not choices:
                    print("DashScope response had no choices")
                    return None, provider_name

                message = choices[0].get("message", {})
                content_list = message.get("content", []) or []

                image_url = None
                for item in content_list:
                    if isinstance(item, dict) and "image" in item:
                        image_url = item["image"]
                        break

                if not image_url:
                    print("DashScope response contained no image URL")
                    return None, provider_name

                try:
                    with urllib.request.urlopen(image_url, timeout=120) as img_resp:
                        img_bytes = img_resp.read()
                except Exception as e:
                    print(f"Error downloading image from DashScope URL: {e}")
                    return None, provider_name

                base64_data = base64.b64encode(img_bytes).decode("utf-8")
                return base64_data, provider_name

            except Exception as e:
                print(f"Error parsing DashScope response: {e}")
                return None, provider_name

        # Build message content for OpenRouter (OpenAI-compatible)
        if reference_image:
            image_url = image_to_base64_url(reference_image)
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        else:
            content = prompt

        start_time = time.time()

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body={"modalities": ["image"]}
        )
        elapsed = time.time() - start_time
        print(f"API request took: {elapsed:.2f} seconds")
        message = response.choices[0].message
        content = getattr(message, "content", None)

        # Prefer multi-part content list (OpenRouter/OpenAI multimodal)
        if isinstance(content, list):
            for part in content:
                image_url_obj = getattr(part, "image_url", None)
                if image_url_obj is not None:
                    url = getattr(image_url_obj, "url", None)
                    if not url:
                        continue
                    # Data URL case
                    if url.startswith("data:") and ";base64," in url:
                        base64_data = url.split(";base64,")[1]
                        return base64_data, provider_name
                    # Remote URL case – download then convert
                    try:
                        with urllib.request.urlopen(url, timeout=120) as img_resp:
                            img_bytes = img_resp.read()
                        base64_data = base64.b64encode(img_bytes).decode("utf-8")
                        return base64_data, provider_name
                    except Exception as e:
                        print(f"Error downloading image from URL: {e}")
                        return None, provider_name

                base64_attr = getattr(part, "image_base64", None)
                if isinstance(base64_attr, str) and base64_attr:
                    return base64_attr, provider_name

        # Legacy message.images-style payloads
        if hasattr(message, "images") and getattr(message, "images"):
            images = message.images
            try:
                image_url = images[0]["image_url"]["url"]
                if image_url.startswith("data:") and ";base64," in image_url:
                    base64_data = image_url.split(";base64,")[1]
                    return base64_data, provider_name
                try:
                    with urllib.request.urlopen(image_url, timeout=120) as img_resp:
                        img_bytes = img_resp.read()
                    base64_data = base64.b64encode(img_bytes).decode("utf-8")
                    return base64_data, provider_name
                except Exception as e:
                    print(f"Error downloading image from legacy images URL: {e}")
                    return None, provider_name
            except Exception as e:
                print(f"Failed to parse message.images: {e}")

        print("No image data found in response.")
        return None, provider_name

    except Exception as e:
        print(f"Error generating image: {e}")
        return None, provider_name


def save_generated_image(base64_data, provider_name):
    """Save the generated image to the output directory."""
    if not base64_data:
        print("No image data received.")
        return None

    # Generate filename with timestamp and provider name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{provider_name}_{timestamp}.png"
    output_path = OUTPUT_DIR / filename

    image_data = base64.b64decode(base64_data)

    with open(output_path, "wb") as f:
        f.write(image_data)

    print(f"Image saved to: {output_path}")
    return output_path


def main():
    """Main function to run the image generation script."""
    print("=" * 50)
    print("  Multi-Provider Image Generation")
    print("=" * 50)

    # Ensure directories exist
    ensure_directories()

    # Select provider
    provider = select_provider()

    # Initialize client
    client, model = get_client(provider)

    # Get reference images and let user select
    reference_images = get_reference_images()
    display_reference_images(reference_images)
    selected_image = select_reference_image(reference_images)

    # Get prompt from user
    print("\nEnter your prompt for image generation.")
    print("Press Enter when done.\n")
    prompt = input("Prompt: ").strip()

    if not prompt:
        print("No prompt provided. Exiting.")
        sys.exit(0)

    # Generate image
    base64_data, provider_name = generate_image(
        client, model, prompt, selected_image, provider["name"]
    )

    # Save the result
    if base64_data:
        save_generated_image(base64_data, provider_name)

    print("\nDone!")


if __name__ == "__main__":
    main()
