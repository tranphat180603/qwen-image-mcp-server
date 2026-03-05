#!/usr/bin/env python3
"""
Minimal Image Generation UI with Gradio
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
import gradio as gr

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

OUTPUT_DIR = PROJECT_ROOT / "generated_images"

# Provider and model configurations
PROVIDERS = {
    "OpenRouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "models": {
            "ByteDance Seedream 4.5": "bytedance-seed/seedream-4.5",
        },
    },
    "Qwen": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "models": {
            "Qwen Image 2.0": "qwen-image-2.0",
            "Qwen Image 2.0 Pro": "qwen-image-2.0-pro",
        },
    },
}


def get_models_for_provider(provider):
    """Return list of model names for the selected provider."""
    if provider in PROVIDERS:
        return list(PROVIDERS[provider]["models"].keys())
    return []


def generate_image(provider, model_name, prompt, reference_image):
    """Generate an image using the selected provider and model."""
    if not prompt:
        return None, "Please enter a prompt."

    provider_config = PROVIDERS.get(provider)
    if not provider_config:
        return None, f"Unknown provider: {provider}"

    model = provider_config["models"].get(model_name)
    if not model:
        return None, f"Unknown model: {model_name}"

    api_key = os.getenv(provider_config["api_key_env"])
    if not api_key:
        return None, f"Please set {provider_config['api_key_env']} in .env"

    OUTPUT_DIR.mkdir(exist_ok=True)

    try:
        if provider == "Qwen":
            return generate_with_qwen(api_key, model, prompt, reference_image)
        else:
            return generate_with_openrouter(api_key, provider_config["base_url"], model, prompt, reference_image)
    except Exception as e:
        return None, f"Error: {str(e)}"


def generate_with_qwen(api_key, model, prompt, reference_image):
    """Generate image using Qwen/DashScope API."""
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

    url = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode("utf-8")

    data = json.loads(body)
    choices = data.get("output", {}).get("choices", [])

    if not choices:
        return None, "No image generated"

    content_list = choices[0].get("message", {}).get("content", [])
    image_url = next((item["image"] for item in content_list if isinstance(item, dict) and "image" in item), None)

    if not image_url:
        return None, "No image URL in response"

    # Download and save
    with urllib.request.urlopen(image_url, timeout=120) as img_resp:
        img_bytes = img_resp.read()

    output_path = save_image(img_bytes, "qwen")
    return str(output_path), f"Generated with {model}"


def generate_with_openrouter(api_key, base_url, model, prompt, reference_image):
    """Generate image using OpenRouter API."""
    client = OpenAI(base_url=base_url, api_key=api_key)

    if reference_image:
        image_url = image_to_base64_url(reference_image)
        content = [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": prompt},
        ]
    else:
        content = prompt

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        extra_body={"modalities": ["image"]},
    )

    message = response.choices[0].message
    resp_content = getattr(message, "content", None)

    img_bytes = None

    # Handle multi-part content
    if isinstance(resp_content, list):
        for part in resp_content:
            image_url_obj = getattr(part, "image_url", None)
            if image_url_obj is not None:
                url = getattr(image_url_obj, "url", None)
                if url:
                    if url.startswith("data:") and ";base64," in url:
                        img_bytes = base64.b64decode(url.split(";base64,")[1])
                    else:
                        with urllib.request.urlopen(url, timeout=120) as img_resp:
                            img_bytes = img_resp.read()
                    break

            # Check for direct base64 attribute
            base64_attr = getattr(part, "image_base64", None)
            if isinstance(base64_attr, str) and base64_attr:
                img_bytes = base64.b64decode(base64_attr)
                break

    # Legacy message.images-style payloads
    if not img_bytes and hasattr(message, "images") and getattr(message, "images"):
        images = message.images
        try:
            image_url = images[0]["image_url"]["url"]
            if image_url.startswith("data:") and ";base64," in image_url:
                img_bytes = base64.b64decode(image_url.split(";base64,")[1])
            else:
                with urllib.request.urlopen(image_url, timeout=120) as img_resp:
                    img_bytes = img_resp.read()
        except Exception:
            pass

    if not img_bytes:
        return None, "No image data in response"

    output_path = save_image(img_bytes, "openrouter")
    return str(output_path), f"Generated with {model}"


def image_to_base64_url(image_path):
    """Convert an image file to a base64 data URL."""
    with open(image_path, "rb") as f:
        image_data = f.read()

    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif",
        ".webp": "image/webp", ".bmp": "image/bmp",
    }
    mime_type = mime_types.get(ext, "image/png")
    base64_data = base64.b64encode(image_data).decode("utf-8")
    return f"data:{mime_type};base64,{base64_data}"


def save_image(img_bytes, provider_name):
    """Save generated image and return path."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{provider_name}_{timestamp}.png"
    output_path = OUTPUT_DIR / filename

    with open(output_path, "wb") as f:
        f.write(img_bytes)

    return output_path


def update_models(provider):
    """Update model dropdown based on provider selection."""
    models = get_models_for_provider(provider)
    return gr.Dropdown(choices=models, value=models[0] if models else None)


# Build the UI
with gr.Blocks(title="Image Gen") as app:
    gr.Markdown("# Image Generation")

    with gr.Row():
        with gr.Column(scale=1):
            provider_dropdown = gr.Dropdown(
                choices=list(PROVIDERS.keys()),
                value="OpenRouter",
                label="Provider",
            )
            model_dropdown = gr.Dropdown(
                choices=get_models_for_provider("OpenRouter"),
                value="ByteDance Seedream 4.5",
                label="Model",
            )
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Describe the image you want to generate...",
                lines=3,
            )
            reference_image = gr.File(
                label="Reference Image (optional)",
                file_types=["image"],
            )
            generate_btn = gr.Button("Generate", variant="primary")
            status_text = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Image", type="filepath")

    # Events
    provider_dropdown.change(
        fn=update_models,
        inputs=provider_dropdown,
        outputs=model_dropdown,
    )

    generate_btn.click(
        fn=generate_image,
        inputs=[provider_dropdown, model_dropdown, prompt_input, reference_image],
        outputs=[output_image, status_text],
    )


if __name__ == "__main__":
    app.launch(
        allowed_paths=[str(OUTPUT_DIR)],
    )
